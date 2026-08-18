from __future__ import annotations

import io
import json
import math
import threading
import uuid
import zipfile
import zlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..schemas.temporal_rotation_quality import (
    TemporalRotationQualityResearchRequest,
    TemporalRotationQualityValidationRequest,
)

TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION = "temporal_rotation_quality_research"
TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION = "temporal_rotation_quality_validations"

_REQUIRED_FILES = {
    "temporal_intelligence_multi_horizon.csv",
    "temporal_intelligence_multi_horizon_equity_curve.csv",
    "temporal_intelligence_multi_horizon_daily_assets.csv",
    "temporal_intelligence_multi_horizon_folds.csv",
    "temporal_intelligence_summary.csv",
}


class TemporalRotationQualityNotFound(RuntimeError):
    pass


class TemporalRotationQualityConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayInputs:
    summary: pd.DataFrame
    multi: pd.DataFrame
    equity: pd.DataFrame
    daily_assets: pd.DataFrame
    folds: pd.DataFrame
    return_map: dict[tuple[int, pd.Timestamp, str], float]
    score_map: dict[tuple[int, pd.Timestamp, str], float]


@dataclass
class ReplayResult:
    metrics: dict[str, Any]
    fold_rows: list[dict[str, Any]]
    equity_rows: list[dict[str, Any]]
    blocked_rows: list[dict[str, Any]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _public_document(document: dict[str, Any] | None, *, include_candidates: bool = False) -> dict[str, Any] | None:
    if not document:
        return None
    payload = {key: value for key, value in document.items() if key != "_id"}
    if not include_candidates:
        payload.pop("candidates", None)
        payload.pop("folds", None)
    return payload


def _load_inputs(export_bytes: bytes) -> ReplayInputs:
    with zipfile.ZipFile(io.BytesIO(export_bytes)) as archive:
        names = set(archive.namelist())
        missing = sorted(_REQUIRED_FILES - names)
        if missing:
            raise TemporalRotationQualityConflict(
                "The selected Temporal Intelligence run does not contain the frozen replay artifacts required for Rotation Quality research. "
                f"Missing: {', '.join(missing)}"
            )
        summary = pd.read_csv(archive.open("temporal_intelligence_summary.csv"))
        multi = pd.read_csv(archive.open("temporal_intelligence_multi_horizon.csv"))
        equity = pd.read_csv(archive.open("temporal_intelligence_multi_horizon_equity_curve.csv"))
        daily_assets = pd.read_csv(archive.open("temporal_intelligence_multi_horizon_daily_assets.csv"))
        folds = pd.read_csv(archive.open("temporal_intelligence_multi_horizon_folds.csv"))

    equity["decision_timestamp"] = pd.to_datetime(equity["decision_timestamp"], utc=True)
    daily_assets["timestamp"] = pd.to_datetime(daily_assets["timestamp"], utc=True)

    return_map = (
        daily_assets.set_index(["fold_id", "timestamp", "symbol"])["open_to_open_return"]
        .astype(float)
        .to_dict()
    )
    score_map = (
        daily_assets.set_index(["fold_id", "timestamp", "symbol"])["entry_rank_score"]
        .astype(float)
        .to_dict()
    )
    return ReplayInputs(
        summary=summary,
        multi=multi,
        equity=equity,
        daily_assets=daily_assets,
        folds=folds,
        return_map=return_map,
        score_map=score_map,
    )


def _replay(
    data: ReplayInputs,
    *,
    candidate_id: str,
    drawdown_trigger: float | None,
    rotation_score_tolerance: float | None,
    challenger_quality_floor: float | None = None,
) -> ReplayResult:
    all_returns: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for fold_id, fold_frame in data.equity.groupby("fold_id", sort=True):
        fold_frame = fold_frame.sort_values("decision_timestamp")
        equity = 10_000.0
        peak = equity
        current_symbol = "CASH"
        switch_count = 0
        strong_challenger_overrides = 0
        fold_returns: list[float] = []
        fold_drawdowns: list[float] = []

        for row in fold_frame.itertuples(index=False):
            timestamp = row.decision_timestamp
            original_target = str(row.target_symbol)
            equity_before = equity
            drawdown_before = equity_before / peak - 1.0

            chosen_target = original_target
            blocked = False
            strong_challenger_override = False
            incumbent_score = float("nan")
            challenger_score = float("nan")
            score_advantage = float("nan")

            eligible_rotation = (
                drawdown_trigger is not None
                and rotation_score_tolerance is not None
                and current_symbol != "CASH"
                and original_target != current_symbol
                and drawdown_before <= drawdown_trigger
            )
            if eligible_rotation:
                incumbent_score = data.score_map.get((int(fold_id), timestamp, current_symbol), float("nan"))
                challenger_score = data.score_map.get((int(fold_id), timestamp, original_target), float("nan"))
                if np.isfinite(incumbent_score) and np.isfinite(challenger_score):
                    score_advantage = challenger_score - incumbent_score
                    if score_advantage < rotation_score_tolerance:
                        if challenger_quality_floor is not None and challenger_score >= challenger_quality_floor:
                            strong_challenger_override = True
                            strong_challenger_overrides += 1
                        else:
                            chosen_target = current_symbol
                            blocked = True

            chosen_key = (int(fold_id), timestamp, chosen_target)
            original_key = (int(fold_id), timestamp, original_target)
            if chosen_key not in data.return_map or original_key not in data.return_map:
                raise TemporalRotationQualityConflict(
                    "Frozen market replay is incomplete for Rotation Quality research: "
                    f"fold={fold_id}, timestamp={timestamp}, chosen={chosen_target}, original={original_target}."
                )

            chosen_return = float(data.return_map[chosen_key])
            original_return = float(data.return_map[original_key])
            if chosen_target != current_symbol:
                switch_count += 1

            equity = equity_before * (1.0 + chosen_return)
            peak = max(peak, equity)
            drawdown_after = equity / peak - 1.0
            fold_returns.append(chosen_return)
            fold_drawdowns.append(drawdown_after)
            all_returns.append(chosen_return)

            incremental_return = chosen_return - original_return
            incremental_dollars = equity_before * incremental_return
            if blocked:
                blocked_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "fold_id": int(fold_id),
                        "timestamp": timestamp.isoformat(),
                        "simulated_incumbent": current_symbol,
                        "original_target": original_target,
                        "chosen_target": chosen_target,
                        "drawdown_before": drawdown_before,
                        "incumbent_entry_rank_score": incumbent_score,
                        "challenger_entry_rank_score": challenger_score,
                        "challenger_minus_incumbent_score": score_advantage,
                        "rotation_score_tolerance": rotation_score_tolerance,
                        "original_interval_return": original_return,
                        "chosen_interval_return": chosen_return,
                        "incremental_interval_return": incremental_return,
                        "equity_before": equity_before,
                        "immediate_incremental_dollars": incremental_dollars,
                        "original_interval_was_negative": original_return < 0.0,
                        "block_improved_next_interval": incremental_return > 0.0,
                    }
                )

            equity_rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold_id": int(fold_id),
                    "decision_timestamp": timestamp.isoformat(),
                    "simulated_current_symbol": current_symbol,
                    "original_target_symbol": original_target,
                    "chosen_target_symbol": chosen_target,
                    "rotation_blocked": blocked,
                    "strong_challenger_override": strong_challenger_override,
                    "challenger_quality_floor": challenger_quality_floor,
                    "drawdown_before": drawdown_before,
                    "incumbent_entry_rank_score": incumbent_score,
                    "challenger_entry_rank_score": challenger_score,
                    "challenger_minus_incumbent_score": score_advantage,
                    "interval_return": chosen_return,
                    "original_interval_return": original_return,
                    "equity_before": equity_before,
                    "strategy_equity": equity,
                    "strategy_drawdown": drawdown_after,
                }
            )
            current_symbol = chosen_target

        fold_rets = pd.Series(fold_returns, dtype="float64")
        fold_std = float(fold_rets.std(ddof=1)) if len(fold_rets) > 1 else 0.0
        fold_sharpe = float(fold_rets.mean() / fold_std * math.sqrt(252)) if fold_std > 0 else float("nan")
        fold_rows.append(
            {
                "candidate_id": candidate_id,
                "fold_id": int(fold_id),
                "initial_capital": 10_000.0,
                "ending_capital": equity,
                "total_return": equity / 10_000.0 - 1.0,
                "sharpe": fold_sharpe,
                "max_drawdown": min(fold_drawdowns) if fold_drawdowns else 0.0,
                "switch_count": switch_count,
                "blocked_rotations": sum(
                    1 for item in blocked_rows if item["candidate_id"] == candidate_id and item["fold_id"] == int(fold_id)
                ),
                "strong_challenger_overrides": strong_challenger_overrides,
            }
        )

    stitched_returns = pd.Series(all_returns, dtype="float64")
    overall_ending = 10_000.0
    for fold in fold_rows:
        overall_ending *= fold["ending_capital"] / 10_000.0

    decision_days = len(stitched_returns)
    cagr = (overall_ending / 10_000.0) ** (252.0 / decision_days) - 1.0 if decision_days > 0 else float("nan")
    overall_std = float(stitched_returns.std(ddof=1)) if len(stitched_returns) > 1 else 0.0
    sharpe = float(stitched_returns.mean() / overall_std * math.sqrt(252)) if overall_std > 0 else float("nan")
    max_drawdown = min((float(fold["max_drawdown"]) for fold in fold_rows), default=0.0)

    positive_immediate = sum(max(0.0, float(row["immediate_incremental_dollars"])) for row in blocked_rows)
    negative_immediate = sum(max(0.0, -float(row["immediate_incremental_dollars"])) for row in blocked_rows)
    metrics = {
        "candidate_id": candidate_id,
        "drawdown_trigger": drawdown_trigger,
        "rotation_score_tolerance": rotation_score_tolerance,
        "challenger_quality_floor": challenger_quality_floor,
        "initial_capital": 10_000.0,
        "ending_capital": overall_ending,
        "total_return": overall_ending / 10_000.0 - 1.0,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "decision_days": decision_days,
        "switch_count": int(sum(row["switch_count"] for row in fold_rows)),
        "blocked_rotations": len(blocked_rows),
        "strong_challenger_overrides": int(sum(row.get("strong_challenger_overrides", 0) for row in fold_rows)),
        "blocks_improving_next_interval": int(sum(bool(row["block_improved_next_interval"]) for row in blocked_rows)),
        "blocked_original_negative_intervals": int(sum(bool(row["original_interval_was_negative"]) for row in blocked_rows)),
        "immediate_loss_avoided_dollars": positive_immediate,
        "immediate_profit_missed_dollars": negative_immediate,
        "immediate_net_rotation_benefit_dollars": positive_immediate - negative_immediate,
    }
    return ReplayResult(metrics=metrics, fold_rows=fold_rows, equity_rows=equity_rows, blocked_rows=blocked_rows)


def _monthly_returns(equity_rows: pd.DataFrame) -> pd.DataFrame:
    if equity_rows.empty:
        return pd.DataFrame(columns=["candidate_id", "fold_id", "month", "start_equity", "ending_equity", "monthly_return"])
    frame = equity_rows.copy()
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True)
    frame["month"] = frame["decision_timestamp"].dt.strftime("%Y-%m")
    rows: list[dict[str, Any]] = []
    for (candidate_id, fold_id, month), group in frame.groupby(["candidate_id", "fold_id", "month"], sort=True):
        group = group.sort_values("decision_timestamp")
        start_equity = float(group.iloc[0]["equity_before"])
        ending_equity = float(group.iloc[-1]["strategy_equity"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "fold_id": int(fold_id),
                "month": month,
                "start_equity": start_equity,
                "ending_equity": ending_equity,
                "monthly_return": ending_equity / start_equity - 1.0,
            }
        )
    return pd.DataFrame(rows)


def _research_gate_evaluation(
    result: ReplayResult,
    control: ReplayResult,
    request: TemporalRotationQualityResearchRequest,
) -> None:
    control_fold_map = {int(row["fold_id"]): float(row["ending_capital"]) for row in control.fold_rows}
    fold_map = {int(row["fold_id"]): float(row["ending_capital"]) for row in result.fold_rows}
    fold_wins = sum(
        1 for fold_id, control_capital in control_fold_map.items()
        if fold_id in fold_map and fold_map[fold_id] > control_capital
    )
    required_fold_wins = (
        int(request.research_gate.required_fold_wins)
        if request.research_gate.required_fold_wins is not None
        else len(control_fold_map)
    )
    if required_fold_wins > len(control_fold_map):
        raise TemporalRotationQualityConflict(
            f"Research gate requires {required_fold_wins} fold wins but the source run contains only {len(control_fold_map)} folds."
        )

    ending_capital = float(result.metrics["ending_capital"])
    control_ending = float(control.metrics["ending_capital"])
    capital_lift = ending_capital / control_ending - 1.0
    sharpe_delta = float(result.metrics["sharpe"]) - float(control.metrics["sharpe"])
    drawdown_delta = float(result.metrics["max_drawdown"]) - float(control.metrics["max_drawdown"])
    result.metrics.update(
        {
            "capital_lift_vs_control": capital_lift,
            "sharpe_delta_vs_control": sharpe_delta,
            "max_drawdown_delta_vs_control": drawdown_delta,
            "switch_delta_vs_control": int(result.metrics["switch_count"]) - int(control.metrics["switch_count"]),
            "folds_beating_control": int(fold_wins),
            "required_fold_wins": int(required_fold_wins),
            "all_folds_beat_control": bool(fold_wins == len(control_fold_map)),
        }
    )
    result.metrics["robust_vs_control"] = bool(
        capital_lift >= float(request.research_gate.minimum_capital_lift)
        and sharpe_delta >= float(request.research_gate.minimum_sharpe_delta)
        and drawdown_delta >= float(request.research_gate.minimum_max_drawdown_delta)
        and fold_wins >= required_fold_wins
    )
    for fold_id in sorted(control_fold_map):
        result.metrics[f"fold_{fold_id}_ending_capital"] = fold_map[fold_id]
        result.metrics[f"fold_{fold_id}_lift_vs_control"] = fold_map[fold_id] / control_fold_map[fold_id] - 1.0


def _surrogate_metrics(result: ReplayResult) -> dict[str, Any]:
    worst_fold_return = min((float(row.get("total_return") or 0.0) for row in result.fold_rows), default=0.0)
    return {
        "ending_capital": float(result.metrics["ending_capital"]),
        "sharpe": float(result.metrics["sharpe"]),
        "maximum_drawdown": float(result.metrics["max_drawdown"]),
        "worst_fold_return": float(worst_fold_return),
        "eligible": True,
    }


def _caro_candidate_results(
    data: ReplayInputs,
    control: ReplayResult,
    request: TemporalRotationQualityResearchRequest,
    *,
    progress_callback: Any | None = None,
) -> list[ReplayResult]:
    from .model_tuning_probability import (
        champion_gate_evaluation,
        evolve_probability_search,
        initial_probability_state,
        propose_champion_probability_candidate,
        propose_unified_space_filling_candidate,
        unified_caro_next_mode,
    )

    config = request.caro
    if request.strong_challenger_override:
        search_space = [
            {
                "name": "challenger_quality_floor",
                "type": "float",
                "min": float(config.challenger_quality_floor_min),
                "max": float(config.challenger_quality_floor_max),
                "precision": 6,
            },
        ]
        base_values = {
            "challenger_quality_floor": round((float(config.challenger_quality_floor_min) + float(config.challenger_quality_floor_max)) / 2.0, 6),
        }
    else:
        search_space = [
            {
                "name": "drawdown_trigger",
                "type": "float",
                "min": float(config.drawdown_trigger_min),
                "max": float(config.drawdown_trigger_max),
                "precision": 6,
            },
            {
                "name": "rotation_score_tolerance",
                "type": "float",
                "min": float(config.rotation_score_tolerance_min),
                "max": float(config.rotation_score_tolerance_max),
                "precision": 6,
            },
        ]
        base_values = {
            "drawdown_trigger": round((float(config.drawdown_trigger_min) + float(config.drawdown_trigger_max)) / 2.0, 6),
            "rotation_score_tolerance": round((float(config.rotation_score_tolerance_min) + float(config.rotation_score_tolerance_max)) / 2.0, 6),
        }
    probability_config = {
        "candidate_pool_size": int(config.candidate_pool_size),
        "space_filling_pool_size": int(config.space_filling_pool_size),
        "exploration_weight": float(config.exploration_weight),
        "minimum_exploration_trials": config.minimum_exploration_trials,
        "initial_exploration_fraction": float(config.initial_exploration_fraction),
        "minimum_exploration_fraction": float(config.minimum_exploration_fraction),
        "stagnation_recovery_trials": int(config.stagnation_recovery_trials),
        "min_capital_improvement": float(config.minimum_capital_improvement),
        "sharpe_tolerance": float(config.sharpe_tolerance),
        "drawdown_tolerance": float(config.drawdown_tolerance),
        "min_worst_fold_return": float(config.minimum_worst_fold_return),
    }
    control_surrogate = _surrogate_metrics(control)
    document: dict[str, Any] = {
        "search_space": search_space,
        "base_tuning_values": base_values,
        "candidate_count": int(config.trials),
        "total_candidates": int(config.trials),
        "seed": int(config.seed),
        "probability_config": probability_config,
        "probability_state": initial_probability_state(),
        "baseline_execution": {"metrics": control_surrogate},
        "probability_anchor": None,
        "prior_observations": [],
        "candidates": [],
    }

    results: list[ReplayResult] = []
    total = int(config.trials)
    for index in range(total):
        policy = unified_caro_next_mode(document)
        if policy.get("mode") == "space_filling":
            proposal = propose_unified_space_filling_candidate(document)
        else:
            proposal = propose_champion_probability_candidate(document)
        settings = dict(proposal.get("settings") or {})
        public_candidate_id = f"RQ-C{index + 1:03d}"
        replay = _replay(
            data,
            candidate_id=public_candidate_id,
            drawdown_trigger=(float(request.baseline_drawdown_trigger) if request.strong_challenger_override else float(settings["drawdown_trigger"])),
            rotation_score_tolerance=(float(request.baseline_rotation_score_tolerance) if request.strong_challenger_override else float(settings["rotation_score_tolerance"])),
            challenger_quality_floor=(float(settings["challenger_quality_floor"]) if request.strong_challenger_override else None),
        )
        replay.metrics["search_method"] = "caro"
        replay.metrics["caro_kind"] = str(proposal.get("kind") or "")
        replay.metrics["caro_proposal"] = deepcopy(proposal.get("proposal") or {})
        _research_gate_evaluation(replay, control, request)
        results.append(replay)

        surrogate = _surrogate_metrics(replay)
        observation = deepcopy(proposal)
        observation["status"] = "completed"
        observation["metrics"] = surrogate
        gate = champion_gate_evaluation(document, surrogate)
        observation["champion_gate_passed"] = bool(gate.get("passed"))
        evolution = evolve_probability_search(document, observation, surrogate, gate)
        document["candidates"].append(observation)
        document["probability_state"] = evolution["state"]
        if evolution.get("probability_anchor") is not None:
            document["probability_anchor"] = evolution["probability_anchor"]

        if progress_callback:
            progress_callback(
                10.0 + 80.0 * ((index + 1) / max(1, total)),
                f"Unified Adaptive CARO · candidate {index + 1}/{total} · {policy.get('mode', 'research')}",
            )
    return results


def _research_payload_from_export(
    export_bytes: bytes,
    request: TemporalRotationQualityResearchRequest,
    *,
    progress_callback: Any | None = None,
) -> tuple[dict[str, Any], ReplayInputs, ReplayResult, ReplayResult]:
    data = _load_inputs(export_bytes)
    source_run_id = str(data.summary.iloc[0]["run_id"])
    if source_run_id != request.source_run_id:
        raise TemporalRotationQualityConflict(
            f"Temporal export run_id mismatch: expected {request.source_run_id}, got {source_run_id}."
        )

    if progress_callback:
        progress_callback(5.0, "Reproducing Temporal Control")
    exported_control_capital = float(data.multi.iloc[0]["ending_capital"])
    control = _replay(data, candidate_id="CONTROL", drawdown_trigger=None, rotation_score_tolerance=None)
    replayed_control_capital = float(control.metrics["ending_capital"])
    control_difference = replayed_control_capital - exported_control_capital
    if abs(control_difference) > request.control_tolerance_usd:
        raise TemporalRotationQualityConflict(
            "Control replay does not reproduce the Temporal Intelligence source run within the requested tolerance. "
            f"Export={exported_control_capital:.6f}, replay={replayed_control_capital:.6f}, difference={control_difference:.6f}."
        )

    control_fold_map = {int(row["fold_id"]): float(row["ending_capital"]) for row in control.fold_rows}
    source_fold_count = len(control_fold_map)
    required_research_wins = (
        int(request.research_gate.required_fold_wins)
        if request.research_gate.required_fold_wins is not None
        else source_fold_count
    )
    if required_research_wins > source_fold_count:
        raise TemporalRotationQualityConflict(
            f"Research gate requires {required_research_wins} fold wins but source run contains {source_fold_count}."
        )

    candidate_results: list[ReplayResult] = []
    if request.search_method == "caro":
        candidate_results = _caro_candidate_results(
            data,
            control,
            request,
            progress_callback=progress_callback,
        )
    else:
        if request.search_method == "manual":
            specifications = [
                (
                    f"RQ-M{index:03d}",
                    float(item.drawdown_trigger),
                    float(item.rotation_score_tolerance),
                    (
                        float(item.challenger_quality_floor)
                        if request.strong_challenger_override and item.challenger_quality_floor is not None else None
                    ),
                )
                for index, item in enumerate(request.manual_candidates, start=1)
            ]
        else:
            specifications = []
            candidate_index = 1
            if request.strong_challenger_override:
                for quality_floor in request.challenger_quality_floors:
                    specifications.append((
                        f"RQ-S{candidate_index:03d}",
                        float(request.baseline_drawdown_trigger),
                        float(request.baseline_rotation_score_tolerance),
                        float(quality_floor),
                    ))
                    candidate_index += 1
            else:
                for trigger in request.drawdown_triggers:
                    for tolerance in request.rotation_score_tolerances:
                        specifications.append((f"RQ-{candidate_index:03d}", float(trigger), float(tolerance), None))
                        candidate_index += 1

        total = len(specifications)
        for index, (candidate_id, trigger, tolerance, quality_floor) in enumerate(specifications, start=1):
            result = _replay(
                data,
                candidate_id=candidate_id,
                drawdown_trigger=trigger,
                rotation_score_tolerance=tolerance,
                challenger_quality_floor=quality_floor,
            )
            result.metrics["search_method"] = request.search_method
            _research_gate_evaluation(result, control, request)
            candidate_results.append(result)
            if progress_callback:
                progress_callback(
                    10.0 + 80.0 * (index / max(1, total)),
                    f"{request.search_method.title()} research · candidate {index}/{total}",
                )

    control.metrics.update(
        {
            "search_method": "control",
            "capital_lift_vs_control": 0.0,
            "sharpe_delta_vs_control": 0.0,
            "max_drawdown_delta_vs_control": 0.0,
            "switch_delta_vs_control": 0,
            "folds_beating_control": source_fold_count,
            "required_fold_wins": required_research_wins,
            "all_folds_beat_control": True,
            "robust_vs_control": True,
        }
    )
    for fold_id in sorted(control_fold_map):
        control.metrics[f"fold_{fold_id}_ending_capital"] = control_fold_map[fold_id]
        control.metrics[f"fold_{fold_id}_lift_vs_control"] = 0.0

    candidates = pd.DataFrame([control.metrics] + [item.metrics for item in candidate_results])
    robust = candidates[(candidates["candidate_id"] != "CONTROL") & (candidates["robust_vs_control"] == True)]  # noqa: E712
    if not robust.empty:
        best_candidate_id = str(robust.sort_values(["ending_capital", "sharpe"], ascending=[False, False]).iloc[0]["candidate_id"])
        selection_reason = "highest ending capital among candidates satisfying the configured research gate"
    else:
        non_control = candidates[candidates["candidate_id"] != "CONTROL"]
        if non_control.empty:
            raise TemporalRotationQualityConflict("Rotation Quality research did not generate any candidate.")
        best_candidate_id = str(non_control.sort_values(["ending_capital", "sharpe"], ascending=[False, False]).iloc[0]["candidate_id"])
        selection_reason = "no candidate passed the configured research gate; highest ending capital retained for diagnostics only"
    best = next(item for item in candidate_results if item.metrics["candidate_id"] == best_candidate_id)

    folds = control.fold_rows + [row for result in candidate_results for row in result.fold_rows]
    created_at = _utc_now()
    search_metadata: dict[str, Any] = {
        "method": request.search_method,
        "strong_challenger_override": bool(request.strong_challenger_override),
    }
    if request.strong_challenger_override:
        search_metadata["baseline_drawdown_trigger"] = request.baseline_drawdown_trigger
        search_metadata["baseline_rotation_score_tolerance"] = request.baseline_rotation_score_tolerance
    if request.search_method == "grid":
        if request.strong_challenger_override:
            search_metadata["challenger_quality_floors"] = list(request.challenger_quality_floors)
        else:
            search_metadata["drawdown_triggers"] = list(request.drawdown_triggers)
            search_metadata["rotation_score_tolerances"] = list(request.rotation_score_tolerances)
    elif request.search_method == "manual":
        search_metadata["manual_candidates"] = [item.model_dump() for item in request.manual_candidates]
    else:
        search_metadata["caro"] = request.caro.model_dump()
        search_metadata["caro_proposal_counts"] = {
            "space_filling": sum(1 for item in candidate_results if item.metrics.get("caro_kind") == "unified_exploration"),
            "adaptive_probability": sum(1 for item in candidate_results if item.metrics.get("caro_kind") == "champion_probability"),
        }

    payload = {
        "experiment": (
            "drawdown_adaptive_rotation_quality_gate_strong_challenger_override"
            if request.strong_challenger_override
            else "drawdown_adaptive_rotation_quality_gate"
        ),
        "status": "completed",
        "stage": "Completed",
        "progress": 100.0,
        "source_run_id": source_run_id,
        "source_fold_count": source_fold_count,
        "focus_month": request.focus_month,
        "created_at": created_at,
        "updated_at": created_at,
        "search": search_metadata,
        # Keep the legacy grid shape for backward-compatible consumers.
        "grid": {
            "drawdown_triggers": list(request.drawdown_triggers) if request.search_method == "grid" and not request.strong_challenger_override else [],
            "rotation_score_tolerances": list(request.rotation_score_tolerances) if request.search_method == "grid" and not request.strong_challenger_override else [],
            "challenger_quality_floors": list(request.challenger_quality_floors) if request.search_method == "grid" and request.strong_challenger_override else [],
            "candidate_count": len(candidate_results),
        },
        "research_gate": {
            **request.research_gate.model_dump(),
            "resolved_required_fold_wins": required_research_wins,
        },
        "control": {
            "exported_ending_capital": exported_control_capital,
            "replayed_ending_capital": replayed_control_capital,
            "difference": control_difference,
            "sharpe": float(control.metrics["sharpe"]),
            "max_drawdown": float(control.metrics["max_drawdown"]),
            "switch_count": int(control.metrics["switch_count"]),
        },
        "best_candidate": {
            **best.metrics,
            "selection_reason": selection_reason,
        },
        "candidate_count": len(candidate_results) + 1,
        "robust_candidate_count": int(((candidates["candidate_id"] != "CONTROL") & (candidates["robust_vs_control"] == True)).sum()),  # noqa: E712
        "candidates": candidates.replace({np.nan: None}).to_dict(orient="records"),
        "folds": pd.DataFrame(folds).replace({np.nan: None}).to_dict(orient="records"),
        "best_blocked_rotations": pd.DataFrame(best.blocked_rows).replace({np.nan: None}).to_dict(orient="records"),
        "decision_policy": {
            "future_information_used_for_decision": False,
            "features": [
                "simulated strategy drawdown before decision",
                "entry_rank_score of simulated incumbent",
                "entry_rank_score of original Temporal target",
                *(
                    ["absolute entry_rank_score of challenger for Strong Challenger Override"]
                    if request.strong_challenger_override else []
                ),
            ],
            "strong_challenger_override": bool(request.strong_challenger_override),
        },
    }
    if progress_callback:
        progress_callback(95.0, "Persisting Rotation Quality results")
    return payload, data, control, best

def run_temporal_rotation_quality_research(
    db: Any,
    request: TemporalRotationQualityResearchRequest,
    *,
    actor_email: str | None,
) -> dict[str, Any]:
    from .temporal_intelligence import (
        TemporalIntelligenceConflict,
        TemporalIntelligenceNotFound,
        build_temporal_intelligence_export,
    )

    try:
        export_bytes = build_temporal_intelligence_export(db, request.source_run_id)
    except TemporalIntelligenceNotFound as exc:
        raise TemporalRotationQualityNotFound(str(exc)) from exc
    except TemporalIntelligenceConflict as exc:
        raise TemporalRotationQualityConflict(str(exc)) from exc

    payload, _data, control, best = _research_payload_from_export(export_bytes, request)
    research_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-rotation-quality-" + uuid.uuid4().hex[:8]
    payload.update({"id": research_id, "actor_email": actor_email})
    db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].insert_one(payload)
    try:
        from .temporal_rotation_quality_analytics import persist_rotation_quality_analytics
        persist_rotation_quality_analytics(
            db,
            processing_id=research_id,
            research_id=research_id,
            processing_kind="research",
            source_run_id=request.source_run_id,
            candidate_metrics=best.metrics,
            control_metrics=control.metrics,
            candidate_equity_rows=best.equity_rows,
            control_equity_rows=control.equity_rows,
            created_at=payload.get("created_at"),
            finished_at=payload.get("finished_at") or payload.get("updated_at"),
        )
    except Exception as analytics_exc:
        db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].update_one(
            {"id": research_id}, {"$set": {"analytics_failure_message": str(analytics_exc)}}
        )
    return _public_document(payload) or {}



def _research_progress(db: Any, research_id: str, percent: float, stage: str) -> None:
    db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].update_one(
        {"id": str(research_id)},
        {
            "$set": {
                "progress": max(0.0, min(100.0, float(percent))),
                "stage": str(stage),
                "updated_at": _utc_now(),
            }
        },
    )


def _run_temporal_rotation_quality_research_background(
    db: Any,
    research_id: str,
    request_payload: dict[str, Any],
) -> None:
    from .temporal_intelligence import (
        TemporalIntelligenceConflict,
        TemporalIntelligenceNotFound,
        build_temporal_intelligence_export,
    )

    try:
        request = TemporalRotationQualityResearchRequest.model_validate(request_payload)
        now = _utc_now()
        db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].update_one(
            {"id": str(research_id)},
            {
                "$set": {
                    "status": "running",
                    "stage": "Loading frozen Temporal Intelligence replay",
                    "progress": 1.0,
                    "started_at": now,
                    "updated_at": now,
                }
            },
        )
        try:
            export_bytes = build_temporal_intelligence_export(db, request.source_run_id)
        except TemporalIntelligenceNotFound as exc:
            raise TemporalRotationQualityNotFound(str(exc)) from exc
        except TemporalIntelligenceConflict as exc:
            raise TemporalRotationQualityConflict(str(exc)) from exc

        payload, _data, control, best = _research_payload_from_export(
            export_bytes,
            request,
            progress_callback=lambda percent, stage: _research_progress(db, research_id, percent, stage),
        )
        finished = _utc_now()
        payload.update(
            {
                "id": str(research_id),
                "actor_email": (db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one(
                    {"id": str(research_id)}, {"actor_email": 1}
                ) or {}).get("actor_email"),
                "request": request.model_dump(),
                "status": "completed",
                "stage": "Completed",
                "progress": 100.0,
                "finished_at": finished,
                "updated_at": finished,
                "failure_message": None,
            }
        )
        # Preserve the original creation/start timestamps from the queued document.
        existing = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one(
            {"id": str(research_id)}, {"created_at": 1, "started_at": 1}
        ) or {}
        payload["created_at"] = existing.get("created_at") or payload.get("created_at") or finished
        payload["started_at"] = existing.get("started_at") or finished
        db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].replace_one(
            {"id": str(research_id)},
            deepcopy(payload),
        )
        try:
            from .temporal_rotation_quality_analytics import persist_rotation_quality_analytics
            persist_rotation_quality_analytics(
                db,
                processing_id=str(research_id),
                research_id=str(research_id),
                processing_kind="research",
                source_run_id=request.source_run_id,
                candidate_metrics=best.metrics,
                control_metrics=control.metrics,
                candidate_equity_rows=best.equity_rows,
                control_equity_rows=control.equity_rows,
                created_at=payload.get("created_at"),
                finished_at=finished,
            )
        except Exception as analytics_exc:
            db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].update_one(
                {"id": str(research_id)},
                {"$set": {"analytics_failure_message": str(analytics_exc), "updated_at": _utc_now()}},
            )
    except Exception as exc:  # background research must persist diagnostics
        finished = _utc_now()
        db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].update_one(
            {"id": str(research_id)},
            {
                "$set": {
                    "status": "failed",
                    "stage": "Failed",
                    "updated_at": finished,
                    "finished_at": finished,
                    "failure_message": str(exc),
                }
            },
        )


def start_temporal_rotation_quality_research(
    db: Any,
    request: TemporalRotationQualityResearchRequest,
    *,
    actor_email: str | None,
    start_thread: bool = True,
) -> dict[str, Any]:
    active = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 0, "id": 1},
    )
    if active:
        raise TemporalRotationQualityConflict(
            f"Wait for Rotation Quality research {active.get('id', 'unknown')} to finish before starting another research execution."
        )
    active_validation = db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 0, "id": 1},
    )
    if active_validation:
        raise TemporalRotationQualityConflict(
            f"Wait for Rotation Quality evidence run {active_validation.get('id', 'unknown')} to finish before starting research."
        )

    research_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-rotation-quality-" + uuid.uuid4().hex[:8]
    now = _utc_now()
    document = {
        "id": research_id,
        "experiment": (
            "drawdown_adaptive_rotation_quality_gate_strong_challenger_override"
            if request.strong_challenger_override
            else "drawdown_adaptive_rotation_quality_gate"
        ),
        "source_run_id": str(request.source_run_id),
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "search": {"method": request.search_method},
        "request": request.model_dump(),
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "actor_email": actor_email,
        "failure_message": None,
    }
    db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].insert_one(deepcopy(document))
    if start_thread:
        threading.Thread(
            target=_run_temporal_rotation_quality_research_background,
            args=(db, research_id, request.model_dump()),
            daemon=True,
        ).start()
    return _public_document(document) or {}

def list_temporal_rotation_quality_research(
    db: Any,
    *,
    source_run_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if source_run_id:
        query["source_run_id"] = str(source_run_id)
    cursor = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find(query).sort("created_at", -1).limit(int(limit))
    return [_public_document(item) or {} for item in cursor]


def get_temporal_rotation_quality_research(db: Any, research_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one({"id": str(research_id)})
    if not document:
        raise TemporalRotationQualityNotFound(f"Temporal Rotation Quality research {research_id} was not found.")
    return _public_document(document) or {}


def get_temporal_rotation_quality_candidates(
    db: Any,
    research_id: str,
    *,
    limit: int = 100,
) -> dict[str, Any]:
    document = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one({"id": str(research_id)}, {"_id": 0, "candidates": 1})
    if not document:
        raise TemporalRotationQualityNotFound(f"Temporal Rotation Quality research {research_id} was not found.")
    candidates = list(document.get("candidates") or [])
    candidates.sort(key=lambda item: (float(item.get("ending_capital") or 0.0), float(item.get("sharpe") or -999.0)), reverse=True)
    return {"research_id": str(research_id), "items": candidates[: int(limit)], "count": min(len(candidates), int(limit)), "total": len(candidates)}


def _write_csv(archive: zipfile.ZipFile, name: str, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    archive.writestr(name, frame.to_csv(index=False).encode("utf-8"))


def build_temporal_rotation_quality_export(db: Any, research_id: str) -> bytes:
    document = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one({"id": str(research_id)}, {"_id": 0})
    if not document:
        raise TemporalRotationQualityNotFound(f"Temporal Rotation Quality research {research_id} was not found.")
    if str(document.get("status") or "") != "completed":
        raise TemporalRotationQualityConflict("Rotation Quality research export requires a completed execution.")

    from .temporal_intelligence import build_temporal_intelligence_export

    source_run_id = str(document["source_run_id"])
    export_bytes = build_temporal_intelligence_export(db, source_run_id)
    data = _load_inputs(export_bytes)

    best_cfg = document.get("best_candidate") or {}
    control = _replay(data, candidate_id="CONTROL", drawdown_trigger=None, rotation_score_tolerance=None)
    best = _replay(
        data,
        candidate_id=str(best_cfg.get("candidate_id") or "BEST"),
        drawdown_trigger=float(best_cfg["drawdown_trigger"]),
        rotation_score_tolerance=float(best_cfg["rotation_score_tolerance"]),
        challenger_quality_floor=(
            float(best_cfg["challenger_quality_floor"])
            if best_cfg.get("challenger_quality_floor") is not None else None
        ),
    )

    control_equity = pd.DataFrame(control.equity_rows)
    best_equity = pd.DataFrame(best.equity_rows)
    control_monthly = _monthly_returns(control_equity)
    best_monthly = _monthly_returns(best_equity)
    monthly_compare = control_monthly.merge(best_monthly, on=["fold_id", "month"], how="outer", suffixes=("_control", "_best"))
    monthly_compare["monthly_return_delta"] = monthly_compare["monthly_return_best"] - monthly_compare["monthly_return_control"]

    focus_month = str(document.get("focus_month") or "").strip()
    focus_compare = pd.DataFrame()
    if focus_month:
        focus_best = best_equity[
            pd.to_datetime(best_equity["decision_timestamp"], utc=True).dt.strftime("%Y-%m") == focus_month
        ].copy()
        focus_control = control_equity[
            pd.to_datetime(control_equity["decision_timestamp"], utc=True).dt.strftime("%Y-%m") == focus_month
        ][["fold_id", "decision_timestamp", "original_target_symbol", "interval_return", "equity_before", "strategy_equity", "strategy_drawdown"]].copy()
        focus_control = focus_control.rename(
            columns={
                "original_target_symbol": "control_target_symbol",
                "interval_return": "control_interval_return",
                "equity_before": "control_equity_before",
                "strategy_equity": "control_strategy_equity",
                "strategy_drawdown": "control_strategy_drawdown",
            }
        )
        focus_compare = focus_best.merge(focus_control, on=["fold_id", "decision_timestamp"], how="left")

    candidates = list(document.get("candidates") or [])
    candidate_frame = pd.DataFrame(candidates)
    non_control = candidate_frame[candidate_frame["candidate_id"] != "CONTROL"] if not candidate_frame.empty and "candidate_id" in candidate_frame else pd.DataFrame()
    if not non_control.empty and "challenger_quality_floor" in non_control.columns and non_control["challenger_quality_floor"].notna().any():
        surface_columns = [
            column for column in (
                "candidate_id", "challenger_quality_floor", "ending_capital", "capital_lift_vs_control",
                "sharpe", "max_drawdown", "strong_challenger_overrides", "robust_vs_control",
            ) if column in non_control.columns
        ]
        surface = non_control[surface_columns].sort_values("challenger_quality_floor")
    elif not non_control.empty and {"drawdown_trigger", "rotation_score_tolerance", "ending_capital"}.issubset(non_control.columns):
        surface = non_control.pivot_table(
            index="drawdown_trigger",
            columns="rotation_score_tolerance",
            values="ending_capital",
            aggfunc="max",
        ).sort_index(ascending=False)
    else:
        surface = pd.DataFrame()

    manifest = {key: value for key, value in document.items() if key not in {"candidates", "folds", "best_blocked_rotations"}}
    for key in ("created_at", "updated_at", "started_at", "finished_at"):
        value = manifest.get(key)
        if isinstance(value, datetime):
            manifest[key] = value.isoformat()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        archive.writestr("temporal_rotation_quality_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        _write_csv(archive, "candidates.csv", candidates)
        _write_csv(archive, "folds.csv", list(document.get("folds") or []))
        _write_csv(archive, "blocked_rotations.csv", list(best.blocked_rows))
        archive.writestr("temporal_rotation_quality_candidates.csv", pd.DataFrame(candidates).to_csv(index=False).encode("utf-8"))
        archive.writestr("temporal_rotation_quality_folds.csv", pd.DataFrame(list(document.get("folds") or [])).to_csv(index=False).encode("utf-8"))
        archive.writestr("temporal_rotation_quality_surface.csv", surface.to_csv().encode("utf-8"))
        archive.writestr("temporal_rotation_quality_best_equity.csv", best_equity.to_csv(index=False).encode("utf-8"))
        archive.writestr("temporal_rotation_quality_best_blocked_rotations.csv", pd.DataFrame(best.blocked_rows).to_csv(index=False).encode("utf-8"))
        archive.writestr("temporal_rotation_quality_best_monthly_comparison.csv", monthly_compare.to_csv(index=False).encode("utf-8"))
        if focus_month:
            archive.writestr(
                f"temporal_rotation_quality_best_{focus_month.replace('-', '_')}.csv",
                focus_compare.to_csv(index=False).encode("utf-8"),
            )
    return buffer.getvalue()


def build_temporal_rotation_quality_validation_export(
    db: Any,
    research_id: str,
    validation_id: str,
) -> bytes:
    document = db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].find_one(
        {"id": str(validation_id), "research_id": str(research_id)},
        {"_id": 0},
    )
    if not document:
        raise TemporalRotationQualityNotFound(
            f"Temporal Rotation Quality validation {validation_id} was not found for research {research_id}."
        )
    if str(document.get("status") or "") != "completed":
        raise TemporalRotationQualityConflict("Rotation Quality validation/certification export requires a completed execution.")

    control = deepcopy(document.get("control") or {})
    control_folds = list(control.pop("folds", []) or [])
    candidates = deepcopy(document.get("candidates") or [])
    flat_candidates: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for candidate in candidates:
        row = {key: value for key, value in candidate.items() if key not in {"folds", "blocked_rotation_details"}}
        flat_candidates.append(row)
        candidate_id = str(candidate.get("candidate_id") or "")
        for fold in candidate.get("folds") or []:
            folds.append({"candidate_id": candidate_id, **dict(fold)})
        for event in candidate.get("blocked_rotation_details") or []:
            blocked.append({"candidate_id": candidate_id, **dict(event)})
    for fold in control_folds:
        folds.append({"candidate_id": "CONTROL", **dict(fold)})

    summary = {
        key: value
        for key, value in document.items()
        if key not in {"control", "candidates", "frozen_candidates"}
    }
    summary["control"] = control
    summary["frozen_candidates"] = deepcopy(document.get("frozen_candidates") or [])
    for key in ("created_at", "updated_at", "started_at", "finished_at"):
        value = summary.get(key)
        if isinstance(value, datetime):
            summary[key] = value.isoformat()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        archive.writestr("control.json", json.dumps({**control, "folds": control_folds}, ensure_ascii=False, indent=2, default=str))
        _write_csv(archive, "candidates.csv", flat_candidates)
        _write_csv(archive, "folds.csv", folds)
        _write_csv(archive, "blocked_rotations.csv", blocked)
        archive.writestr(
            "validation_policy.json",
            json.dumps(document.get("validation_policy") or {}, ensure_ascii=False, indent=2, default=str),
        )
        archive.writestr(
            "candidate_details.json",
            json.dumps(candidates, ensure_ascii=False, indent=2, default=str),
        )
    return buffer.getvalue()


# Public pure function used by tests/research validation without MongoDB.
def evaluate_temporal_rotation_quality_export(
    export_bytes: bytes,
    request: TemporalRotationQualityResearchRequest,
) -> dict[str, Any]:
    payload, _data, _control, _best = _research_payload_from_export(export_bytes, request)
    return _public_document(payload) or {}


def _public_validation_document(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if not document:
        return None
    return {key: value for key, value in document.items() if key != "_id"}


def _validation_source_artifact_rows(db: Any, run_id: str, kind: str) -> list[dict[str, Any]]:
    from ..infrastructure.persistence.mongo_repository import TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION

    rows: list[dict[str, Any]] = []
    cursor = db[TEMPORAL_INTELLIGENCE_ARTIFACTS_COLLECTION].find(
        {"run_id": str(run_id), "kind": str(kind)},
        {"_id": 0, "sequence": 1, "encoding": 1, "payload": 1, "rows": 1},
    ).sort("sequence", 1)
    for item in cursor:
        artifact_rows = item.get("rows") or []
        if item.get("encoding") == "zlib-json-v1" and item.get("payload"):
            artifact_rows = json.loads(zlib.decompress(bytes(item["payload"])).decode("utf-8"))
        rows.extend(dict(row) for row in artifact_rows if isinstance(row, dict))
    return rows


def _validation_replay_inputs(
    db: Any,
    *,
    source_run_id: str,
    fold_count: int,
    progress_callback: Any | None = None,
    cancel_callback: Any | None = None,
) -> tuple[ReplayInputs, float]:
    """Retrain Temporal Intelligence on a new fold protocol and expose a frozen replay surface."""
    from ..engine.market_data import load_market_bars, validate_and_clean_bars
    from ..engine.temporal_intelligence import run_temporal_intelligence
    from ..infrastructure.persistence.mongo_repository import TEMPORAL_INTELLIGENCE_RUNS_COLLECTION
    from ..schemas.requests import BacktestExecutionRequest

    source_run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(source_run_id)})
    if source_run is None or str(source_run.get("status") or "") != "completed":
        raise TemporalRotationQualityNotFound(
            f"Source Temporal Intelligence run {source_run_id} is unavailable or not completed."
        )

    request_payload = deepcopy(source_run.get("request") or {})
    snapshot_id = str(
        source_run.get("market_data_snapshot_id")
        or request_payload.get("research_market_data_snapshot_id")
        or ""
    ).strip().lower()
    if not snapshot_id:
        raise TemporalRotationQualityConflict(
            "The source Temporal Intelligence run does not contain a frozen market-data snapshot id."
        )

    request_payload.update(
        {
            "research_market_data_mode": "database_only",
            "research_market_data_snapshot_id": snapshot_id,
            "expected_market_data_signature_sha256": snapshot_id,
            "deterministic_execution": True,
            "numeric_thread_limit": 1,
            "xgb_n_jobs": 1,
            "walk_forward_fold_count_override": int(fold_count),
        }
    )
    request = BacktestExecutionRequest.model_validate(request_payload)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    anchor_assets = set(request.calendar_anchor_assets)
    for position, symbol in enumerate(request.assets, start=1):
        if progress_callback:
            progress_callback(
                2.0 + 12.0 * ((position - 1) / max(1, len(request.assets))),
                f"Loading frozen market data {position}/{len(request.assets)}",
            )
        asset_request = request if symbol in anchor_assets else request.model_copy(
            update={"market_data_require_complete_history": False}
        )
        raw = load_market_bars(symbol, asset_request)
        bars_by_symbol[symbol] = validate_and_clean_bars(raw, asset_request)

    source_result = source_run.get("result") if isinstance(source_run.get("result"), dict) else {}
    winner_summary = deepcopy(source_result.get("winner_reference") or {})
    winner_daily = _validation_source_artifact_rows(db, source_run_id, "winner_reference_daily")
    winner_trades = _validation_source_artifact_rows(db, source_run_id, "winner_reference_trades")
    if not winner_summary or not winner_daily:
        raise TemporalRotationQualityConflict(
            "The source Temporal Intelligence run does not contain the immutable Winner replay required for validation."
        )

    result = run_temporal_intelligence(
        bars_by_symbol,
        request,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        winner_reference_override={
            "summary": winner_summary,
            "daily_rows": winner_daily,
            "trade_rows": winner_trades,
        },
        candidate_evaluation_only=True,
    )
    observations = list(result.pop("_multi_horizon_observations", []) or [])
    multi_horizon = result.get("multi_horizon_metrics") if isinstance(result.get("multi_horizon_metrics"), dict) else {}
    capital = multi_horizon.get("shadow_capital") if isinstance(multi_horizon.get("shadow_capital"), dict) else {}
    equity_rows = [dict(row) for row in (capital.get("economic_curve") or []) if isinstance(row, dict)]
    exported_control_capital = float(capital.get("ending_capital") or 0.0)
    if not observations or not equity_rows or exported_control_capital <= 0.0:
        raise TemporalRotationQualityConflict(
            "The requested Temporal validation run did not produce the replay artifacts required by Rotation Quality."
        )

    equity = pd.DataFrame(equity_rows)
    daily_assets = pd.DataFrame(observations)
    if "decision_timestamp" not in equity.columns or "timestamp" not in daily_assets.columns:
        raise TemporalRotationQualityConflict("Temporal validation replay timestamps are missing.")
    equity["decision_timestamp"] = pd.to_datetime(equity["decision_timestamp"], utc=True)
    daily_assets["timestamp"] = pd.to_datetime(daily_assets["timestamp"], utc=True)

    required_daily = {"fold_id", "symbol", "open_to_open_return", "entry_rank_score"}
    missing_daily = sorted(required_daily - set(daily_assets.columns))
    if missing_daily:
        raise TemporalRotationQualityConflict(
            "Temporal validation observations are missing required columns: " + ", ".join(missing_daily)
        )

    return_map = (
        daily_assets.set_index(["fold_id", "timestamp", "symbol"])["open_to_open_return"]
        .astype(float)
        .to_dict()
    )
    score_map = (
        daily_assets.set_index(["fold_id", "timestamp", "symbol"])["entry_rank_score"]
        .astype(float)
        .to_dict()
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in result.get("multi_horizon_fold_metrics") or []:
        if not isinstance(fold, dict):
            continue
        fold_rows.append(
            {
                "fold_id": int(fold.get("fold_id") or len(fold_rows) + 1),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
            }
        )

    inputs = ReplayInputs(
        summary=pd.DataFrame([{"run_id": source_run_id, "walk_forward_fold_count": int(fold_count)}]),
        multi=pd.DataFrame([{"ending_capital": exported_control_capital}]),
        equity=equity,
        daily_assets=daily_assets,
        folds=pd.DataFrame(fold_rows),
        return_map=return_map,
        score_map=score_map,
    )
    return inputs, exported_control_capital


def _validation_candidate_metrics(
    candidate: ReplayResult,
    control: ReplayResult,
    *,
    fold_count: int,
    required_fold_wins: int | None = None,
    minimum_capital_lift: float = 0.0,
    minimum_sharpe_delta: float = 0.0,
    minimum_max_drawdown_delta: float = 0.0,
) -> dict[str, Any]:
    control_folds = {int(row["fold_id"]): row for row in control.fold_rows}
    candidate_folds = {int(row["fold_id"]): row for row in candidate.fold_rows}
    fold_details: list[dict[str, Any]] = []
    fold_wins = 0
    for fold_id in sorted(control_folds):
        if fold_id not in candidate_folds:
            raise TemporalRotationQualityConflict(f"Candidate replay is missing validation fold {fold_id}.")
        control_row = control_folds[fold_id]
        candidate_row = candidate_folds[fold_id]
        control_capital = float(control_row["ending_capital"])
        candidate_capital = float(candidate_row["ending_capital"])
        lift = candidate_capital / control_capital - 1.0 if control_capital else float("nan")
        if candidate_capital > control_capital:
            fold_wins += 1
        fold_details.append(
            {
                **candidate_row,
                "control_ending_capital": control_capital,
                "capital_lift_vs_control": lift,
                "beats_control": bool(candidate_capital > control_capital),
            }
        )

    resolved_fold_wins = max(0, int(required_fold_wins)) if required_fold_wins is not None else max(1, int(fold_count) - 1)
    if resolved_fold_wins > int(fold_count):
        raise TemporalRotationQualityConflict("required_fold_wins cannot exceed fold_count.")
    ending_capital = float(candidate.metrics["ending_capital"])
    control_ending = float(control.metrics["ending_capital"])
    capital_lift = ending_capital / control_ending - 1.0
    sharpe = float(candidate.metrics["sharpe"])
    control_sharpe = float(control.metrics["sharpe"])
    sharpe_delta = sharpe - control_sharpe
    max_drawdown = float(candidate.metrics["max_drawdown"])
    control_max_drawdown = float(control.metrics["max_drawdown"])
    max_drawdown_delta = max_drawdown - control_max_drawdown
    capital_pass = bool(
        capital_lift > 0.0
        if abs(float(minimum_capital_lift)) < 1e-15
        else capital_lift >= float(minimum_capital_lift)
    )
    sharpe_pass = bool(sharpe_delta >= float(minimum_sharpe_delta))
    max_drawdown_pass = bool(max_drawdown_delta >= float(minimum_max_drawdown_delta))
    folds_pass = bool(fold_wins >= resolved_fold_wins)
    return {
        **candidate.metrics,
        "capital_lift_vs_control": capital_lift,
        "sharpe_delta_vs_control": sharpe_delta,
        "max_drawdown_delta_vs_control": max_drawdown_delta,
        "switch_delta_vs_control": int(candidate.metrics["switch_count"]) - int(control.metrics["switch_count"]),
        "fold_count": int(fold_count),
        "folds_beating_control": int(fold_wins),
        "required_fold_wins": int(resolved_fold_wins),
        "minimum_capital_lift": float(minimum_capital_lift),
        "minimum_sharpe_delta": float(minimum_sharpe_delta),
        "minimum_max_drawdown_delta": float(minimum_max_drawdown_delta),
        "capital_pass": capital_pass,
        "sharpe_pass": sharpe_pass,
        "max_drawdown_pass": max_drawdown_pass,
        "folds_pass": folds_pass,
        "validation_pass": bool(capital_pass and sharpe_pass and max_drawdown_pass and folds_pass),
        "folds": fold_details,
        "blocked_rotation_details": deepcopy(candidate.blocked_rows),
    }

def _validation_progress(db: Any, validation_id: str, percent: float, stage: str) -> None:
    db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].update_one(
        {"id": str(validation_id)},
        {
            "$set": {
                "progress": max(0.0, min(100.0, float(percent))),
                "stage": str(stage),
                "updated_at": _utc_now(),
            }
        },
    )


def _run_temporal_rotation_quality_validation(db: Any, validation_id: str) -> None:
    validation = db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].find_one({"id": str(validation_id)})
    if not validation:
        return
    db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].update_one(
        {"id": str(validation_id)},
        {
            "$set": {
                "status": "running",
                "stage": f"Preparing {int(validation.get('fold_count') or 0)}-fold Temporal {str(validation.get('kind') or 'validation')}",
                "progress": 1.0,
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
            }
        },
    )
    try:
        fold_count = int(validation["fold_count"])
        inputs, engine_control_capital = _validation_replay_inputs(
            db,
            source_run_id=str(validation["source_run_id"]),
            fold_count=fold_count,
            progress_callback=lambda percent, stage: _validation_progress(
                db, validation_id, min(90.0, max(2.0, float(percent) * 0.88)), stage
            ),
        )
        control = _replay(inputs, candidate_id="CONTROL", drawdown_trigger=None, rotation_score_tolerance=None)
        replay_difference = float(control.metrics["ending_capital"]) - float(engine_control_capital)
        if abs(replay_difference) > 1.0:
            raise TemporalRotationQualityConflict(
                "Validation Control replay failed to reproduce the newly trained Temporal Control within US$ 1.00. "
                f"Engine={engine_control_capital:.6f}, replay={float(control.metrics['ending_capital']):.6f}, "
                f"difference={replay_difference:.6f}."
            )

        _validation_progress(db, validation_id, 92.0, "Evaluating frozen Rotation Quality candidates")
        candidate_results: list[dict[str, Any]] = []
        candidate_replays: dict[str, ReplayResult] = {}
        for frozen in validation.get("frozen_candidates") or []:
            replay = _replay(
                inputs,
                candidate_id=str(frozen["candidate_id"]),
                drawdown_trigger=float(frozen["drawdown_trigger"]),
                rotation_score_tolerance=float(frozen["rotation_score_tolerance"]),
                challenger_quality_floor=(
                    float(frozen["challenger_quality_floor"])
                    if frozen.get("challenger_quality_floor") is not None else None
                ),
            )
            evaluated = _validation_candidate_metrics(
                replay,
                control,
                fold_count=fold_count,
                required_fold_wins=validation.get("required_fold_wins"),
                minimum_capital_lift=float(validation.get("minimum_capital_lift") or 0.0),
                minimum_sharpe_delta=float(validation.get("minimum_sharpe_delta") or 0.0),
                minimum_max_drawdown_delta=float(validation.get("minimum_max_drawdown_delta") or 0.0),
            )
            evaluated["research_ending_capital"] = frozen.get("research_ending_capital")
            evaluated["research_capital_lift_vs_control"] = frozen.get("research_capital_lift_vs_control")
            candidate_results.append(evaluated)
            candidate_replays[str(evaluated.get("candidate_id") or replay.metrics.get("candidate_id") or "")] = replay

        candidate_results.sort(
            key=lambda item: (
                bool(item.get("validation_pass")),
                float(item.get("ending_capital") or 0.0),
                float(item.get("sharpe") or -999.0),
            ),
            reverse=True,
        )
        passing = [item for item in candidate_results if bool(item.get("validation_pass"))]
        now = _utc_now()
        db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].update_one(
            {"id": str(validation_id)},
            {
                "$set": {
                    "status": "completed",
                    "stage": "Completed",
                    "progress": 100.0,
                    "updated_at": now,
                    "finished_at": now,
                    "control": {
                        **control.metrics,
                        "engine_ending_capital": float(engine_control_capital),
                        "replay_difference": replay_difference,
                        "folds": control.fold_rows,
                    },
                    "candidates": candidate_results,
                    "passing_candidate_count": len(passing),
                    "best_validated_candidate": passing[0] if passing else None,
                    "validation_policy": {
                        "kind": str(validation.get("kind") or "validation"),
                        "parameters_frozen_from_research": True,
                        "new_walk_forward_training": True,
                        "fold_count": fold_count,
                        "required_fold_wins": int(validation.get("required_fold_wins") if validation.get("required_fold_wins") is not None else max(1, fold_count - 1)),
                        "minimum_capital_lift": float(validation.get("minimum_capital_lift") or 0.0),
                        "minimum_sharpe_delta": float(validation.get("minimum_sharpe_delta") or 0.0),
                        "minimum_max_drawdown_delta": float(validation.get("minimum_max_drawdown_delta") or 0.0),
                        "capital_rule": "candidate capital lift satisfies the configured minimum",
                        "sharpe_rule": "candidate Sharpe delta satisfies the configured minimum",
                        "max_drawdown_rule": "candidate MaxDD delta satisfies the configured minimum",
                        "fold_rule": f"candidate beats validation Control in at least {int(validation.get('required_fold_wins') if validation.get('required_fold_wins') is not None else max(1, fold_count - 1))} of {fold_count} folds",
                        "future_information_used_for_decision": False,
                    },
                    "failure_message": None,
                }
            },
        )
        try:
            from .temporal_rotation_quality_analytics import persist_rotation_quality_analytics

            for candidate in candidate_results:
                candidate_id = str(candidate.get("candidate_id") or "")
                replay = candidate_replays.get(candidate_id)
                if replay is None:
                    continue
                persist_rotation_quality_analytics(
                    db,
                    processing_id=str(validation_id),
                    research_id=str(validation.get("research_id") or ""),
                    processing_kind=str(validation.get("kind") or "validation"),
                    source_run_id=str(validation.get("source_run_id") or ""),
                    candidate_metrics=candidate,
                    control_metrics=control.metrics,
                    candidate_equity_rows=replay.equity_rows,
                    control_equity_rows=control.equity_rows,
                    created_at=validation.get("created_at"),
                    finished_at=now,
                )
        except Exception as analytics_exc:
            db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].update_one(
                {"id": str(validation_id)},
                {"$set": {"analytics_failure_message": str(analytics_exc), "updated_at": _utc_now()}},
            )
    except Exception as exc:  # background job must persist diagnostics instead of losing the exception
        now = _utc_now()
        db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].update_one(
            {"id": str(validation_id)},
            {
                "$set": {
                    "status": "failed",
                    "stage": "Failed",
                    "updated_at": now,
                    "finished_at": now,
                    "failure_message": str(exc),
                }
            },
        )


def start_temporal_rotation_quality_validation(
    db: Any,
    research_id: str,
    request: TemporalRotationQualityValidationRequest,
    *,
    actor_email: str | None,
    start_thread: bool = True,
) -> dict[str, Any]:
    research = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one({"id": str(research_id)})
    if not research:
        raise TemporalRotationQualityNotFound(f"Temporal Rotation Quality research {research_id} was not found.")
    if str(research.get("status") or "") != "completed":
        raise TemporalRotationQualityConflict("Rotation Quality validation/certification requires a completed research execution.")

    from ..infrastructure.persistence.mongo_repository import (
        JOBS_COLLECTION,
        MODEL_TUNING_RUNS_COLLECTION,
        TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
    )
    from .system_settings import get_system_settings

    runtime_settings = get_system_settings(db)
    if not bool(runtime_settings["training"]["enabled"]):
        raise TemporalRotationQualityConflict("Model training is disabled in System Settings.")

    active_temporal = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running", "stop_requested"]}}, {"_id": 0, "id": 1}
    )
    if active_temporal:
        raise TemporalRotationQualityConflict(
            f"Wait for Temporal Intelligence {active_temporal.get('id', 'unknown')} to finish before starting validation."
        )
    active_backtest = db[JOBS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}}, {"_id": 0, "id": 1}
    )
    if active_backtest:
        raise TemporalRotationQualityConflict("Wait for the active Simulation Backtest to finish before starting validation.")
    active_tuning = db[MODEL_TUNING_RUNS_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running", "stop_requested"]}}, {"_id": 0, "id": 1}
    )
    if active_tuning:
        raise TemporalRotationQualityConflict("Wait for the active Model Tuning campaign to finish before starting validation.")

    active_research = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 0, "id": 1},
    )
    if active_research:
        raise TemporalRotationQualityConflict(
            f"Wait for Rotation Quality research {active_research.get('id', 'unknown')} to finish before starting validation/certification."
        )

    active = db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].find_one(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 0, "id": 1, "research_id": 1},
    )
    if active:
        raise TemporalRotationQualityConflict(
            f"Wait for Rotation Quality validation {active.get('id', 'unknown')} to finish before starting another validation."
        )

    candidates_by_id = {
        str(item.get("candidate_id")): item
        for item in (research.get("candidates") or [])
        if isinstance(item, dict) and item.get("candidate_id")
    }
    frozen_candidates: list[dict[str, Any]] = []
    for candidate_id in request.candidate_ids:
        item = candidates_by_id.get(str(candidate_id))
        if item is None:
            raise TemporalRotationQualityNotFound(
                f"Candidate {candidate_id} does not exist in research {research_id}."
            )
        if item.get("drawdown_trigger") is None or item.get("rotation_score_tolerance") is None:
            raise TemporalRotationQualityConflict(f"Candidate {candidate_id} does not contain frozen Rotation Quality parameters.")
        frozen_candidates.append(
            {
                "candidate_id": str(candidate_id),
                "drawdown_trigger": float(item["drawdown_trigger"]),
                "rotation_score_tolerance": float(item["rotation_score_tolerance"]),
                "challenger_quality_floor": (
                    float(item["challenger_quality_floor"])
                    if item.get("challenger_quality_floor") is not None else None
                ),
                "research_ending_capital": item.get("ending_capital"),
                "research_capital_lift_vs_control": item.get("capital_lift_vs_control"),
            }
        )

    run_kind = str(request.kind or "validation")
    validation_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + f"-rq-{run_kind}-" + uuid.uuid4().hex[:8]
    now = _utc_now()
    document = {
        "id": validation_id,
        "research_id": str(research_id),
        "source_run_id": str(research.get("source_run_id") or ""),
        "status": "queued",
        "stage": "Queued",
        "progress": 0.0,
        "kind": run_kind,
        "fold_count": int(request.fold_count),
        "required_fold_wins": int(request.resolved_required_fold_wins()),
        "minimum_capital_lift": float(request.minimum_capital_lift),
        "minimum_sharpe_delta": float(request.minimum_sharpe_delta),
        "minimum_max_drawdown_delta": float(request.minimum_max_drawdown_delta),
        "candidate_ids": list(request.candidate_ids),
        "frozen_candidates": frozen_candidates,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "actor_email": actor_email,
        "control": None,
        "candidates": [],
        "passing_candidate_count": 0,
        "best_validated_candidate": None,
        "validation_policy": None,
        "failure_message": None,
    }
    db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].insert_one(deepcopy(document))
    if start_thread:
        threading.Thread(
            target=_run_temporal_rotation_quality_validation,
            args=(db, validation_id),
            daemon=True,
        ).start()
    return _public_validation_document(document) or {}


def get_temporal_rotation_quality_validation(db: Any, research_id: str, validation_id: str) -> dict[str, Any]:
    document = db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION].find_one(
        {"id": str(validation_id), "research_id": str(research_id)}
    )
    if not document:
        raise TemporalRotationQualityNotFound(
            f"Temporal Rotation Quality validation {validation_id} was not found for research {research_id}."
        )
    return _public_validation_document(document) or {}


def list_temporal_rotation_quality_validations(
    db: Any,
    research_id: str,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    research = db[TEMPORAL_ROTATION_QUALITY_RESEARCH_COLLECTION].find_one(
        {"id": str(research_id)}, {"_id": 0, "id": 1}
    )
    if not research:
        raise TemporalRotationQualityNotFound(f"Temporal Rotation Quality research {research_id} was not found.")
    cursor = (
        db[TEMPORAL_ROTATION_QUALITY_VALIDATION_COLLECTION]
        .find({"research_id": str(research_id)})
        .sort("created_at", -1)
        .limit(int(limit))
    )
    return [_public_validation_document(item) or {} for item in cursor]
