#!/usr/bin/env python
"""
Market Cycle Trader — local Temporal Rotation Quality research.

Read-only experiment over a frozen Temporal Intelligence export ZIP.

Hypothesis
----------
When the simulated strategy is already in drawdown, reject an original Temporal
rotation when the challenger's entry_rank_score is materially worse than the
simulated incumbent's entry_rank_score.

No future/realized columns are used to make the decision. Future interval
returns are used only after the decision to replay/evaluate the candidate.

This tool does NOT write to MongoDB, does NOT call Alpaca, does NOT modify a
Strategy, and does NOT deploy anything.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DRAWDOWN_TRIGGERS = [
    -0.03,
    -0.04,
    -0.05,
    -0.06,
    -0.07,
    -0.08,
    -0.10,
    -0.12,
]

ROTATION_SCORE_TOLERANCES = [
    -0.150,
    -0.125,
    -0.100,
    -0.075,
    -0.050,
    -0.025,
     0.000,
]

REQUIRED_FILES = {
    "temporal_intelligence_multi_horizon.csv",
    "temporal_intelligence_multi_horizon_equity_curve.csv",
    "temporal_intelligence_multi_horizon_daily_assets.csv",
    "temporal_intelligence_multi_horizon_folds.csv",
    "temporal_intelligence_summary.csv",
}


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


def _safe_float(value: Any) -> float:
    return float(value) if value is not None and not pd.isna(value) else float("nan")


def load_export(export_zip: Path) -> ReplayInputs:
    if not export_zip.exists():
        raise FileNotFoundError(f"Export ZIP not found: {export_zip}")

    with zipfile.ZipFile(export_zip) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED_FILES - names)
        if missing:
            raise RuntimeError(
                "The ZIP is not the required Temporal Intelligence export. "
                f"Missing: {', '.join(missing)}"
            )

        summary = pd.read_csv(archive.open("temporal_intelligence_summary.csv"))
        multi = pd.read_csv(archive.open("temporal_intelligence_multi_horizon.csv"))
        equity = pd.read_csv(
            archive.open("temporal_intelligence_multi_horizon_equity_curve.csv")
        )
        daily_assets = pd.read_csv(
            archive.open("temporal_intelligence_multi_horizon_daily_assets.csv")
        )
        folds = pd.read_csv(
            archive.open("temporal_intelligence_multi_horizon_folds.csv")
        )

    equity["decision_timestamp"] = pd.to_datetime(
        equity["decision_timestamp"], utc=True
    )
    daily_assets["timestamp"] = pd.to_datetime(
        daily_assets["timestamp"], utc=True
    )

    return_map = (
        daily_assets
        .set_index(["fold_id", "timestamp", "symbol"])["open_to_open_return"]
        .astype(float)
        .to_dict()
    )
    score_map = (
        daily_assets
        .set_index(["fold_id", "timestamp", "symbol"])["entry_rank_score"]
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


def replay(
    data: ReplayInputs,
    *,
    candidate_id: str,
    drawdown_trigger: float | None,
    rotation_score_tolerance: float | None,
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
        fold_returns: list[float] = []
        fold_drawdowns: list[float] = []

        for row in fold_frame.itertuples(index=False):
            timestamp = row.decision_timestamp
            original_target = str(row.target_symbol)
            equity_before = equity
            drawdown_before = equity_before / peak - 1.0

            chosen_target = original_target
            blocked = False
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
                incumbent_score = data.score_map.get(
                    (int(fold_id), timestamp, current_symbol), float("nan")
                )
                challenger_score = data.score_map.get(
                    (int(fold_id), timestamp, original_target), float("nan")
                )

                if np.isfinite(incumbent_score) and np.isfinite(challenger_score):
                    score_advantage = challenger_score - incumbent_score
                    if score_advantage < rotation_score_tolerance:
                        chosen_target = current_symbol
                        blocked = True

            key = (int(fold_id), timestamp, chosen_target)
            if key not in data.return_map:
                raise RuntimeError(
                    "Missing frozen market replay return for "
                    f"fold={fold_id}, timestamp={timestamp}, symbol={chosen_target}"
                )

            chosen_return = float(data.return_map[key])
            original_return = float(
                data.return_map[(int(fold_id), timestamp, original_target)]
            )

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
        fold_sharpe = (
            float(fold_rets.mean() / fold_rets.std(ddof=1) * math.sqrt(252))
            if len(fold_rets) > 1 and fold_rets.std(ddof=1) > 0
            else float("nan")
        )

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
                    1
                    for item in blocked_rows
                    if item["candidate_id"] == candidate_id
                    and item["fold_id"] == int(fold_id)
                ),
            }
        )

    stitched_returns = pd.Series(all_returns, dtype="float64")
    overall_ending = 10_000.0
    for fold in fold_rows:
        overall_ending *= fold["ending_capital"] / 10_000.0

    decision_days = len(stitched_returns)
    cagr = (
        (overall_ending / 10_000.0) ** (252.0 / decision_days) - 1.0
        if decision_days > 0
        else float("nan")
    )
    sharpe = (
        float(
            stitched_returns.mean()
            / stitched_returns.std(ddof=1)
            * math.sqrt(252)
        )
        if len(stitched_returns) > 1 and stitched_returns.std(ddof=1) > 0
        else float("nan")
    )
    max_drawdown = min(
        (float(fold["max_drawdown"]) for fold in fold_rows),
        default=0.0,
    )

    positive_immediate = sum(
        max(0.0, float(row["immediate_incremental_dollars"]))
        for row in blocked_rows
    )
    negative_immediate = sum(
        max(0.0, -float(row["immediate_incremental_dollars"]))
        for row in blocked_rows
    )

    metrics = {
        "candidate_id": candidate_id,
        "drawdown_trigger": drawdown_trigger,
        "rotation_score_tolerance": rotation_score_tolerance,
        "initial_capital": 10_000.0,
        "ending_capital": overall_ending,
        "total_return": overall_ending / 10_000.0 - 1.0,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "decision_days": decision_days,
        "switch_count": int(sum(row["switch_count"] for row in fold_rows)),
        "blocked_rotations": len(blocked_rows),
        "blocks_improving_next_interval": int(
            sum(bool(row["block_improved_next_interval"]) for row in blocked_rows)
        ),
        "blocked_original_negative_intervals": int(
            sum(bool(row["original_interval_was_negative"]) for row in blocked_rows)
        ),
        "immediate_loss_avoided_dollars": positive_immediate,
        "immediate_profit_missed_dollars": negative_immediate,
        "immediate_net_rotation_benefit_dollars": (
            positive_immediate - negative_immediate
        ),
    }

    return ReplayResult(
        metrics=metrics,
        fold_rows=fold_rows,
        equity_rows=equity_rows,
        blocked_rows=blocked_rows,
    )


def monthly_returns(equity_rows: pd.DataFrame) -> pd.DataFrame:
    if equity_rows.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "month",
                "start_equity",
                "ending_equity",
                "monthly_return",
            ]
        )

    frame = equity_rows.copy()
    frame["decision_timestamp"] = pd.to_datetime(
        frame["decision_timestamp"], utc=True
    )
    frame["month"] = frame["decision_timestamp"].dt.strftime("%Y-%m")

    rows: list[dict[str, Any]] = []
    for (candidate_id, fold_id, month), group in frame.groupby(
        ["candidate_id", "fold_id", "month"], sort=True
    ):
        group = group.sort_values("decision_timestamp")
        rows.append(
            {
                "candidate_id": candidate_id,
                "fold_id": int(fold_id),
                "month": month,
                "start_equity": float(group.iloc[0]["equity_before"]),
                "ending_equity": float(group.iloc[-1]["strategy_equity"]),
                "monthly_return": (
                    float(group.iloc[-1]["strategy_equity"])
                    / float(group.iloc[0]["equity_before"])
                    - 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the local Drawdown-Adaptive Rotation Quality grid over a "
            "frozen Temporal Intelligence export ZIP."
        )
    )
    parser.add_argument(
        "--export-zip",
        required=True,
        type=Path,
        help="Temporal Intelligence export ZIP.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/temporal_rotation_quality"),
        help="Directory for research outputs.",
    )
    parser.add_argument(
        "--focus-month",
        default="2026-06",
        help="Month for detailed diagnostic output (YYYY-MM).",
    )
    parser.add_argument(
        "--control-tolerance-usd",
        type=float,
        default=1.0,
        help="Maximum allowed replay-vs-export Control capital difference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen Temporal export: {args.export_zip}")
    data = load_export(args.export_zip)

    run_id = str(data.summary.iloc[0]["run_id"])
    exported_control_capital = float(data.multi.iloc[0]["ending_capital"])

    control = replay(
        data,
        candidate_id="CONTROL",
        drawdown_trigger=None,
        rotation_score_tolerance=None,
    )
    replayed_control_capital = float(control.metrics["ending_capital"])
    control_difference = replayed_control_capital - exported_control_capital

    print(f"Run ID: {run_id}")
    print(
        "Control export: "
        f"US$ {exported_control_capital:,.2f}"
    )
    print(
        "Control replay: "
        f"US$ {replayed_control_capital:,.2f}"
    )
    print(
        "Control difference: "
        f"US$ {control_difference:,.6f}"
    )

    if abs(control_difference) > args.control_tolerance_usd:
        raise RuntimeError(
            "CONTROL REPLAY FAILED. The frozen replay does not reproduce the "
            f"exported Temporal capital within US$ {args.control_tolerance_usd:.2f}. "
            "Do not trust candidate results."
        )

    control_fold_map = {
        int(row["fold_id"]): float(row["ending_capital"])
        for row in control.fold_rows
    }

    candidate_results: list[ReplayResult] = []
    index = 1
    total_candidates = len(DRAWDOWN_TRIGGERS) * len(
        ROTATION_SCORE_TOLERANCES
    )
    print(f"Evaluating {total_candidates} deterministic candidates...")

    for trigger in DRAWDOWN_TRIGGERS:
        for tolerance in ROTATION_SCORE_TOLERANCES:
            candidate_id = f"RQ-{index:03d}"
            result = replay(
                data,
                candidate_id=candidate_id,
                drawdown_trigger=trigger,
                rotation_score_tolerance=tolerance,
            )

            fold_map = {
                int(row["fold_id"]): float(row["ending_capital"])
                for row in result.fold_rows
            }
            all_folds_beat_control = all(
                fold_map[fold_id] > control_fold_map[fold_id]
                for fold_id in control_fold_map
            )

            result.metrics["capital_lift_vs_control"] = (
                result.metrics["ending_capital"] / replayed_control_capital - 1.0
            )
            result.metrics["sharpe_delta_vs_control"] = (
                result.metrics["sharpe"] - control.metrics["sharpe"]
            )
            result.metrics["max_drawdown_delta_vs_control"] = (
                result.metrics["max_drawdown"] - control.metrics["max_drawdown"]
            )
            result.metrics["switch_delta_vs_control"] = (
                result.metrics["switch_count"] - control.metrics["switch_count"]
            )
            result.metrics["all_folds_beat_control"] = all_folds_beat_control
            result.metrics["robust_vs_control"] = bool(
                result.metrics["ending_capital"] > replayed_control_capital
                and result.metrics["sharpe"] >= control.metrics["sharpe"]
                and result.metrics["max_drawdown"]
                >= control.metrics["max_drawdown"]
                and all_folds_beat_control
            )

            for fold_id in sorted(control_fold_map):
                result.metrics[f"fold_{fold_id}_ending_capital"] = fold_map[
                    fold_id
                ]
                result.metrics[f"fold_{fold_id}_lift_vs_control"] = (
                    fold_map[fold_id] / control_fold_map[fold_id] - 1.0
                )

            candidate_results.append(result)
            index += 1

    control.metrics["capital_lift_vs_control"] = 0.0
    control.metrics["sharpe_delta_vs_control"] = 0.0
    control.metrics["max_drawdown_delta_vs_control"] = 0.0
    control.metrics["switch_delta_vs_control"] = 0
    control.metrics["all_folds_beat_control"] = True
    control.metrics["robust_vs_control"] = True
    for fold_id in sorted(control_fold_map):
        control.metrics[f"fold_{fold_id}_ending_capital"] = control_fold_map[
            fold_id
        ]
        control.metrics[f"fold_{fold_id}_lift_vs_control"] = 0.0

    candidates_frame = pd.DataFrame(
        [control.metrics] + [item.metrics for item in candidate_results]
    )

    robust_candidates = candidates_frame[
        (candidates_frame["candidate_id"] != "CONTROL")
        & (candidates_frame["robust_vs_control"] == True)  # noqa: E712
    ]
    if not robust_candidates.empty:
        best_candidate_id = str(
            robust_candidates.sort_values(
                ["ending_capital", "sharpe"],
                ascending=[False, False],
            ).iloc[0]["candidate_id"]
        )
        best_selection_reason = (
            "highest ending capital among candidates that beat Control capital, "
            "do not reduce Sharpe, do not worsen MaxDD, and beat Control in every fold"
        )
    else:
        non_control = candidates_frame[
            candidates_frame["candidate_id"] != "CONTROL"
        ]
        best_candidate_id = str(
            non_control.sort_values(
                ["ending_capital", "sharpe"],
                ascending=[False, False],
            ).iloc[0]["candidate_id"]
        )
        best_selection_reason = (
            "no candidate passed the robust gate; selected highest ending capital "
            "for diagnostic purposes only"
        )

    best = next(
        item
        for item in candidate_results
        if item.metrics["candidate_id"] == best_candidate_id
    )

    fold_frame = pd.DataFrame(
        control.fold_rows
        + [
            row
            for result in candidate_results
            for row in result.fold_rows
        ]
    )
    for candidate_id, group in fold_frame.groupby("candidate_id"):
        if candidate_id == "CONTROL":
            continue
        for idx in group.index:
            fold_id = int(fold_frame.loc[idx, "fold_id"])
            fold_frame.loc[idx, "control_ending_capital"] = control_fold_map[
                fold_id
            ]
            fold_frame.loc[idx, "lift_vs_control"] = (
                float(fold_frame.loc[idx, "ending_capital"])
                / control_fold_map[fold_id]
                - 1.0
            )

    control_equity = pd.DataFrame(control.equity_rows)
    best_equity = pd.DataFrame(best.equity_rows)
    best_blocked = pd.DataFrame(best.blocked_rows)

    best_monthly = monthly_returns(best_equity)
    control_monthly = monthly_returns(control_equity)
    monthly_compare = control_monthly.merge(
        best_monthly,
        on=["fold_id", "month"],
        how="outer",
        suffixes=("_control", "_best"),
    )
    monthly_compare["monthly_return_delta"] = (
        monthly_compare["monthly_return_best"]
        - monthly_compare["monthly_return_control"]
    )

    focus_best = best_equity[
        pd.to_datetime(
            best_equity["decision_timestamp"], utc=True
        ).dt.strftime("%Y-%m")
        == args.focus_month
    ].copy()
    focus_control = control_equity[
        pd.to_datetime(
            control_equity["decision_timestamp"], utc=True
        ).dt.strftime("%Y-%m")
        == args.focus_month
    ][
        [
            "fold_id",
            "decision_timestamp",
            "original_target_symbol",
            "interval_return",
            "equity_before",
            "strategy_equity",
            "strategy_drawdown",
        ]
    ].copy()
    focus_control = focus_control.rename(
        columns={
            "original_target_symbol": "control_target_symbol",
            "interval_return": "control_interval_return",
            "equity_before": "control_equity_before",
            "strategy_equity": "control_strategy_equity",
            "strategy_drawdown": "control_strategy_drawdown",
        }
    )
    focus_compare = focus_best.merge(
        focus_control,
        on=["fold_id", "decision_timestamp"],
        how="left",
    )

    surface = (
        candidates_frame[candidates_frame["candidate_id"] != "CONTROL"]
        .pivot(
            index="drawdown_trigger",
            columns="rotation_score_tolerance",
            values="ending_capital",
        )
        .sort_index(ascending=False)
    )

    candidates_frame = candidates_frame.sort_values(
        ["ending_capital", "sharpe"],
        ascending=[False, False],
    )

    candidates_frame.to_csv(
        output_dir / "temporal_rotation_quality_candidates.csv",
        index=False,
    )
    fold_frame.to_csv(
        output_dir / "temporal_rotation_quality_folds.csv",
        index=False,
    )
    surface.to_csv(
        output_dir / "temporal_rotation_quality_surface.csv"
    )
    best_equity.to_csv(
        output_dir / "temporal_rotation_quality_best_equity.csv",
        index=False,
    )
    best_blocked.to_csv(
        output_dir / "temporal_rotation_quality_best_blocked_rotations.csv",
        index=False,
    )
    monthly_compare.to_csv(
        output_dir / "temporal_rotation_quality_best_monthly_comparison.csv",
        index=False,
    )
    focus_compare.to_csv(
        output_dir / f"temporal_rotation_quality_best_{args.focus_month.replace('-', '_')}.csv",
        index=False,
    )

    best_row = candidates_frame[
        candidates_frame["candidate_id"] == best_candidate_id
    ].iloc[0]

    manifest = {
        "experiment": "drawdown_adaptive_rotation_quality_gate",
        "research_only": True,
        "source_run_id": run_id,
        "source_export_zip": str(args.export_zip),
        "decision_features": [
            "simulated strategy drawdown before decision",
            "entry_rank_score of simulated incumbent",
            "entry_rank_score of original Temporal target",
        ],
        "future_information_used_for_decision": False,
        "grid": {
            "drawdown_triggers": DRAWDOWN_TRIGGERS,
            "rotation_score_tolerances": ROTATION_SCORE_TOLERANCES,
            "candidate_count": total_candidates,
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
            "candidate_id": best_candidate_id,
            "selection_reason": best_selection_reason,
            "drawdown_trigger": _safe_float(best_row["drawdown_trigger"]),
            "rotation_score_tolerance": _safe_float(
                best_row["rotation_score_tolerance"]
            ),
            "ending_capital": float(best_row["ending_capital"]),
            "capital_lift_vs_control": float(
                best_row["capital_lift_vs_control"]
            ),
            "sharpe": float(best_row["sharpe"]),
            "max_drawdown": float(best_row["max_drawdown"]),
            "switch_count": int(best_row["switch_count"]),
            "blocked_rotations": int(best_row["blocked_rotations"]),
            "all_folds_beat_control": bool(
                best_row["all_folds_beat_control"]
            ),
            "robust_vs_control": bool(best_row["robust_vs_control"]),
        },
        "focus_month": args.focus_month,
        "outputs": [
            "temporal_rotation_quality_candidates.csv",
            "temporal_rotation_quality_folds.csv",
            "temporal_rotation_quality_surface.csv",
            "temporal_rotation_quality_best_equity.csv",
            "temporal_rotation_quality_best_blocked_rotations.csv",
            "temporal_rotation_quality_best_monthly_comparison.csv",
            f"temporal_rotation_quality_best_{args.focus_month.replace('-', '_')}.csv",
        ],
    }

    (output_dir / "temporal_rotation_quality_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Research completed.")
    print(
        "Best candidate: "
        f"{best_candidate_id} | "
        f"DD trigger {float(best_row['drawdown_trigger']):.1%} | "
        f"score tolerance {float(best_row['rotation_score_tolerance']):.3f}"
    )
    print(
        "Ending capital: "
        f"US$ {float(best_row['ending_capital']):,.2f} "
        f"({float(best_row['capital_lift_vs_control']):+.2%} vs Control)"
    )
    print(
        "Sharpe: "
        f"{float(best_row['sharpe']):.6f} | "
        "MaxDD: "
        f"{float(best_row['max_drawdown']):.2%} | "
        "Switches: "
        f"{int(best_row['switch_count'])}"
    )
    print(
        "Robust gate: "
        f"{'PASS' if bool(best_row['robust_vs_control']) else 'FAIL'}"
    )
    print(f"Outputs: {output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
