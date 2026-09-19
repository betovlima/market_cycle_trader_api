from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.engine.capital_rotation import (
    _build_walk_forward_folds,
    _fold_performance,
    prepare_rotation_panel,
    run_rotation_models,
)
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    create_client,
    get_database,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest
from market_cycle_trader_api.services.model_tuning import _SEARCH_SPACE
from market_cycle_trader_api.services.model_tuning_probability import (
    champion_gate_evaluation,
    evolve_probability_search,
    initial_probability_state,
    propose_champion_probability_candidate,
    propose_unified_space_filling_candidate,
    unified_caro_next_mode,
)
from scripts.research_point_in_time_corporate_actions import (
    _apply_slippage,
    _calculate_reference_fees,
    _latest_job,
    _read_bars,
    _split_normalize,
    _utc,
)


RAW_COLLECTION = "alpaca_market_bars_raw_20260919"
CA_COLLECTION = "alpaca_corporate_actions_full_20260919"
DEFAULT_OUTPUT = "output/raw_split_unified_caro"


def _corporate_actions(collection: Any, symbol: str) -> list[dict[str, Any]]:
    return list(
        collection.find(
            {
                "$or": [
                    {"symbol": symbol},
                    {"source_symbol": symbol},
                    {"old_symbol": symbol},
                    {"new_symbol": symbol},
                    {"acquirer_symbol": symbol},
                    {"acquiree_symbol": symbol},
                ]
            },
            {"_id": 0},
        ).sort([("process_date", 1), ("ex_date", 1), ("effective_date", 1)])
    )


def _structural_identity_issue(symbol: str, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized = str(symbol).strip().upper()
    for action in actions:
        action_type = str(action.get("action_type") or "")
        if action_type in {"stock_merger", "stock_and_cash_merger", "cash_merger"}:
            acquiree = str(action.get("acquiree_symbol") or "").strip().upper()
            acquirer = str(action.get("acquirer_symbol") or "").strip().upper()
            if acquiree == normalized and acquirer and acquirer != normalized:
                return {
                    "reason": "structural_identity_change",
                    "action_type": action_type,
                    "process_date": action.get("process_date"),
                    "effective_date": action.get("effective_date"),
                    "acquiree_symbol": acquiree,
                    "acquirer_symbol": acquirer,
                }
    return None


def _candidate_config(
    base: BacktestExecutionRequest,
    settings: dict[str, Any],
) -> BacktestExecutionRequest:
    research_settings = deepcopy(base.research_model_settings)
    lightgbm = deepcopy(research_settings.get("lightgbm") or {})
    for name, value in settings.items():
        lightgbm[str(name)] = value
    research_settings["lightgbm"] = lightgbm
    return base.model_copy(update={"research_model_settings": research_settings})


def _metrics(
    result: Any,
    folds: list[dict[str, Any]],
    initial_capital: float,
) -> dict[str, Any]:
    fold_rows = _fold_performance(result.predictions, folds, initial_capital)
    worst_fold_return = (
        min(float(row["strategy_return"]) for row in fold_rows)
        if fold_rows
        else None
    )
    return {
        "ending_capital": float(result.metrics.get("strategy_ending_capital") or 0.0),
        "sharpe": float(result.metrics.get("strategy_sharpe") or 0.0),
        "maximum_drawdown": float(result.metrics.get("strategy_maximum_drawdown") or 0.0),
        "cagr": float(result.metrics.get("strategy_cagr") or 0.0),
        "worst_fold_return": worst_fold_return,
        "folds": fold_rows,
        "eligible": True,
    }


def _run_candidate(
    *,
    label: str,
    frames: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    folds: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    print(f"[caro] starting {label}", flush=True)
    results = run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=lambda p, stage, completed: print(
            f"[caro] {label} progress={p:.1f}% completed={completed} stage={stage}",
            flush=True,
        ),
    )
    if not results:
        raise RuntimeError(f"{label} returned no result.")
    result = results[0]
    metrics = _metrics(result, folds, float(config.initial_capital))
    print(
        f"[caro] completed {label} capital={metrics['ending_capital']:,.2f} "
        f"sharpe={metrics['sharpe']:.4f} maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold={metrics['worst_fold_return']:.4%}",
        flush=True,
    )
    return result, metrics


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Unified CARO recalibration on a frozen RAW Alpaca snapshot with "
            "local split normalization and structural corporate-action guards."
        )
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--raw-collection", default=RAW_COLLECTION)
    parser.add_argument("--corporate-actions-collection", default=CA_COLLECTION)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-dividend-features",
        action="store_true",
        help=(
            "Reserved for a later controlled campaign. v10.8.64 defaults to "
            "RAW+split price features only."
        ),
    )
    args = parser.parse_args()

    if args.include_dividend_features:
        raise ValueError(
            "v10.8.64 calibrates the canonical RAW+split price architecture only. "
            "Dividend-feature tuning must be run as a separate campaign."
        )
    if int(args.candidate_count) < 4:
        raise ValueError("--candidate-count must be at least 4.")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "campaign_checkpoint.json"

    client = create_client()
    try:
        db = get_database(client)
        job = _latest_job(db, args.job_id)
        request = BacktestExecutionRequest.model_validate(job["request"])
        start = _utc(request.start_date).normalize()
        end = _utc(request.analysis_end_date or request.end_date).normalize()
        end_exclusive = end + pd.Timedelta(days=1)

        raw_collection = db[str(args.raw_collection)]
        ca_collection = db[str(args.corporate_actions_collection)]

        frames: dict[str, pd.DataFrame] = {}
        exclusions: list[dict[str, Any]] = []
        split_diagnostics: list[dict[str, Any]] = []

        for position, symbol in enumerate(request.assets, start=1):
            raw = _read_bars(
                raw_collection,
                symbol=symbol,
                interval=request.timeframe,
                feed=request.alpaca_historical_feed,
                adjustment="raw",
                start=start,
                end_exclusive=end_exclusive,
            )
            actions = _corporate_actions(ca_collection, symbol)
            issue = _structural_identity_issue(symbol, actions)
            if issue is not None:
                exclusions.append({"symbol": symbol, **issue})
                print(
                    f"[data] {position}/{len(request.assets)} {symbol} excluded "
                    f"reason={issue['reason']} action={issue['action_type']} "
                    f"effective={issue.get('effective_date')}",
                    flush=True,
                )
                continue
            if raw.empty:
                exclusions.append({"symbol": symbol, "reason": "missing_raw_history"})
                continue

            reconstructed, applied = _split_normalize(raw, actions)
            frames[symbol] = reconstructed
            split_diagnostics.append(
                {
                    "symbol": symbol,
                    "raw_rows": len(raw),
                    "corporate_actions": len(actions),
                    "splits_applied": len(applied),
                }
            )
            print(
                f"[data] {position}/{len(request.assets)} {symbol} "
                f"rows={len(raw)} actions={len(actions)} splits={len(applied)}",
                flush=True,
            )

        if len(frames) < 2:
            raise RuntimeError("Fewer than two structurally valid assets remain.")

        eligible = list(frames)
        anchors = [item for item in request.calendar_anchor_assets if item in frames]
        if len(anchors) < 2:
            raise RuntimeError(
                "Structural corporate-action exclusions removed too many calendar anchors."
            )
        references = [item for item in request.research_reference_assets if item in frames]
        if len(references) < 2:
            references = list(anchors)
        reference_set = set(references)
        candidates = [
            item
            for item in request.research_candidate_assets
            if item in frames and item not in reference_set
        ]

        base_config = request.model_copy(
            update={
                "assets": eligible,
                "calendar_anchor_assets": anchors,
                "research_reference_assets": references,
                "research_candidate_assets": candidates,
                "market_data_provider": "alpaca",
                "alpaca_adjustment": "split",
                "research_market_data_mode": "database_only",
                "expected_market_data_signature_sha256": None,
                "research_market_data_snapshot_id": None,
            }
        )

        prepared, common_dates = prepare_rotation_panel(frames, base_config)
        folds = _build_walk_forward_folds(common_dates, base_config)

        base_lightgbm = deepcopy(base_config.research_model_settings.get("lightgbm") or {})
        base_tuning_values = {
            spec["name"]: base_lightgbm[spec["name"]]
            for spec in _SEARCH_SPACE
        }

        baseline_result, baseline_metrics = _run_candidate(
            label="CONTROL",
            frames=frames,
            config=base_config,
            folds=folds,
        )
        baseline_result.predictions.to_csv(output_dir / "control_predictions.csv", index=True)
        baseline_result.trades.to_csv(output_dir / "control_trades.csv", index=False)

        control_settings_hash = hashlib.sha256(
            json.dumps(
                base_tuning_values,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        control_observation = {
            "candidate_id": 0,
            "kind": "control",
            "is_control": True,
            "settings": deepcopy(base_tuning_values),
            "settings_hash": control_settings_hash,
            "status": "completed",
            "metrics": deepcopy(baseline_metrics),
            "champion_gate_passed": True,
        }

        document: dict[str, Any] = {
            "method": "champion_probability",
            "candidate_count": int(args.candidate_count),
            "seed": int(args.seed),
            "search_space": [dict(item) for item in _SEARCH_SPACE],
            "base_tuning_values": base_tuning_values,
            "base_model_values": base_tuning_values,
            "prior_observations": [control_observation],
            "candidates": [],
            "probability_config": {
                "minimum_exploration_trials": min(24, len(_SEARCH_SPACE) + 2),
                "min_capital_improvement": 0.0,
                "sharpe_tolerance": 0.05,
                "drawdown_tolerance": 0.03,
                "min_worst_fold_return": 0.0,
                "candidate_pool_size": 2048,
                "exploration_weight": 0.15,
                "initial_exploration_fraction": 0.45,
                "minimum_exploration_fraction": 0.20,
                "stagnation_recovery_trials": 4,
            },
            "probability_state": initial_probability_state([control_observation]),
            "probability_anchor": {
                "source": "control",
                "candidate_id": 0,
                "settings_hash": control_settings_hash,
                "settings": deepcopy(base_tuning_values),
                "metrics": deepcopy(baseline_metrics),
            },
            "baseline_execution": {
                "metrics": baseline_metrics,
                "settings": base_tuning_values,
            },
        }

        checkpoint = {
            "schema_version": 1,
            "api_version": "10.8.64",
            "source_job_id": job.get("id"),
            "raw_collection": str(args.raw_collection),
            "corporate_actions_collection": str(args.corporate_actions_collection),
            "eligible_assets": eligible,
            "excluded_assets": exclusions,
            "baseline_metrics": baseline_metrics,
            "baseline_settings": base_tuning_values,
            "control_observation_in_surrogate": True,
            "control_settings_hash": control_settings_hash,
            "candidates": [],
        }
        _checkpoint(checkpoint_path, checkpoint)

        champion_result = baseline_result
        champion_metrics = baseline_metrics
        champion_settings = base_tuning_values
        champion_candidate_id: int | None = None

        for iteration in range(1, int(args.candidate_count) + 1):
            policy = unified_caro_next_mode(document)
            if policy["mode"] == "space_filling":
                candidate = propose_unified_space_filling_candidate(document)
            else:
                candidate = propose_champion_probability_candidate(document)

            candidate_id = int(candidate["candidate_id"])
            settings = dict(candidate["settings"])
            config = _candidate_config(base_config, settings)
            result, metrics = _run_candidate(
                label=f"CANDIDATE_{candidate_id}_{candidate['kind']}",
                frames=frames,
                config=config,
                folds=folds,
            )

            gate = champion_gate_evaluation(document, metrics)
            candidate["status"] = "completed"
            candidate["metrics"] = metrics
            candidate["champion_gate"] = gate
            candidate["champion_gate_passed"] = bool(gate["passed"])
            candidate["iteration"] = iteration
            document["candidates"].append(candidate)

            evolution = evolve_probability_search(
                document,
                candidate,
                metrics,
                gate,
            )
            document["probability_state"] = evolution["state"]
            if evolution.get("probability_anchor") is not None:
                document["probability_anchor"] = evolution["probability_anchor"]
                champion_result = result
                champion_metrics = metrics
                champion_settings = settings
                champion_candidate_id = candidate_id

            checkpoint["candidates"].append(
                {
                    "candidate_id": candidate_id,
                    "kind": candidate["kind"],
                    "settings": settings,
                    "metrics": metrics,
                    "champion_gate_passed": bool(gate["passed"]),
                    "proposal": candidate.get("proposal"),
                }
            )
            checkpoint["probability_state"] = document["probability_state"]
            checkpoint["probability_anchor"] = document.get("probability_anchor")
            checkpoint["champion_candidate_id"] = champion_candidate_id
            checkpoint["champion_metrics"] = champion_metrics
            checkpoint["champion_settings"] = champion_settings
            _checkpoint(checkpoint_path, checkpoint)

            print(
                f"[caro] iteration={iteration}/{args.candidate_count} "
                f"mode={policy['mode']} candidate={candidate_id} "
                f"promoted={bool(gate['passed'])}",
                flush=True,
            )

        champion_result.predictions.to_csv(output_dir / "champion_predictions.csv", index=True)
        champion_result.trades.to_csv(output_dir / "champion_trades.csv", index=False)

        rows = []
        for item in checkpoint["candidates"]:
            metrics = item["metrics"]
            row = {
                "candidate_id": item["candidate_id"],
                "kind": item["kind"],
                "champion_gate_passed": item["champion_gate_passed"],
                "ending_capital": metrics["ending_capital"],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "worst_fold_return": metrics["worst_fold_return"],
                **item["settings"],
            }
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "candidates.csv", index=False)
        pd.DataFrame(split_diagnostics).to_csv(output_dir / "data_diagnostics.csv", index=False)
        pd.DataFrame(exclusions).to_csv(output_dir / "excluded_assets.csv", index=False)

        summary = {
            "schema_version": 1,
            "api_version": "10.8.64",
            "experiment": "raw-split-unified-caro-v2",
            "source_job_id": job.get("id"),
            "raw_collection": str(args.raw_collection),
            "corporate_actions_collection": str(args.corporate_actions_collection),
            "configured_asset_count": len(request.assets),
            "eligible_asset_count": len(eligible),
            "excluded_assets": exclusions,
            "candidate_count": int(args.candidate_count),
            "search_space": [dict(item) for item in _SEARCH_SPACE],
            "baseline": {
                "settings": base_tuning_values,
                "metrics": baseline_metrics,
            },
            "champion": {
                "candidate_id": champion_candidate_id,
                "settings": champion_settings,
                "metrics": champion_metrics,
                "improvement_vs_baseline": (
                    champion_metrics["ending_capital"] / baseline_metrics["ending_capital"] - 1.0
                    if baseline_metrics["ending_capital"] > 0
                    else None
                ),
            },
            "probability_state": document["probability_state"],
            "methodology": {
                "prices": "frozen RAW Alpaca OHLCV",
                "split_handling": "local forward/reverse split normalization",
                "dividend_features": False,
                "structural_identity_guard": True,
                "tuning": "existing Unified CARO space-filling + probabilistic refinement",
                "control_in_surrogate_training": True,
                "control_as_initial_probability_anchor": True,
                "hyperparameter_search_only": True,
            },
        }
        _checkpoint(output_dir / "summary.json", summary)

        print("")
        print("[done] RAW+split Unified CARO completed")
        print(f"[done] baseline_capital={baseline_metrics['ending_capital']:,.2f}")
        print(f"[done] champion_capital={champion_metrics['ending_capital']:,.2f}")
        print(f"[done] champion_candidate={champion_candidate_id}")
        print(f"[done] output={output_dir}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
