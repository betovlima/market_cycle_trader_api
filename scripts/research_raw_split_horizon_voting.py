from __future__ import annotations

import argparse
from copy import deepcopy
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
    prepare_rotation_panel,
    run_rotation_models,
)
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    create_client,
    get_database,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest
from scripts.research_point_in_time_corporate_actions import (
    _apply_slippage,
    _calculate_reference_fees,
    _latest_job,
    _read_bars,
    _split_normalize,
    _utc,
)
from scripts.research_raw_split_unified_caro import (
    CA_COLLECTION,
    RAW_COLLECTION,
    _corporate_actions,
    _metrics,
    _structural_identity_issue,
)


DEFAULT_OUTPUT = "output/raw_split_horizon_voting_consensus"


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def _run(
    *,
    label: str,
    frames: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    folds: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    print(f"[horizon-voting] starting {label}", flush=True)
    results = run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=lambda p, stage, completed: print(
            f"[horizon-voting] {label} progress={p:.1f}% "
            f"completed={completed} stage={stage}",
            flush=True,
        ),
        technical_log_callback=lambda message: print(
            f"[technical] {label} {message}",
            flush=True,
        ),
    )
    if not results:
        raise RuntimeError(f"{label} returned no result.")
    result = results[0]
    metrics = _metrics(result, folds, float(config.initial_capital))
    metrics.update(
        {
            key: value
            for key, value in result.metrics.items()
            if str(key).startswith("horizon_voting_")
        }
    )
    buy_hold = float(metrics.get("buy_hold_ending_capital") or 0.0)
    ratio = metrics.get("strategy_vs_buy_hold_capital_ratio")
    print(
        f"[horizon-voting] completed {label} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"buy_hold={buy_hold:,.2f} "
        f"vs_buy_hold={(f'{float(ratio):.3f}x' if ratio is not None else 'n/a')} "
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold={metrics['worst_fold_return']:.4%}",
        flush=True,
    )
    return result, metrics


def _comparison_row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": label,
        "ending_capital": metrics.get("ending_capital"),
        "return": metrics.get("strategy_return"),
        "cagr": metrics.get("cagr"),
        "sharpe": metrics.get("sharpe"),
        "maximum_drawdown": metrics.get("maximum_drawdown"),
        "worst_fold_return": metrics.get("worst_fold_return"),
        "buy_hold_ending_capital": metrics.get("buy_hold_ending_capital"),
        "buy_hold_return": metrics.get("buy_hold_return"),
        "buy_hold_cagr": metrics.get("buy_hold_cagr"),
        "buy_hold_sharpe": metrics.get("buy_hold_sharpe"),
        "buy_hold_maximum_drawdown": metrics.get("buy_hold_maximum_drawdown"),
        "strategy_vs_buy_hold_capital_ratio": metrics.get(
            "strategy_vs_buy_hold_capital_ratio"
        ),
        "strategy_vs_buy_hold_excess_capital": metrics.get(
            "strategy_vs_buy_hold_excess_capital"
        ),
        "strategy_vs_buy_hold_excess_return": metrics.get(
            "strategy_vs_buy_hold_excess_return"
        ),
        "simulation_total_seconds": metrics.get("simulation_total_seconds"),
        "simulation_policy_seconds": metrics.get("simulation_policy_seconds"),
        "simulation_accounting_seconds": metrics.get(
            "simulation_accounting_seconds"
        ),
        "horizon_voting_decisions": metrics.get("horizon_voting_decisions"),
        "horizon_voting_changed_base_actions": metrics.get(
            "horizon_voting_changed_base_actions"
        ),
        "horizon_voting_change_rate": metrics.get(
            "horizon_voting_change_rate"
        ),
        "horizon_voting_consensus_accepts": metrics.get(
            "horizon_voting_consensus_accepts"
        ),
        "horizon_voting_cash_overrides": metrics.get(
            "horizon_voting_cash_overrides"
        ),
        "horizon_voting_blocked_switches": metrics.get(
            "horizon_voting_blocked_switches"
        ),
        "horizon_voting_average_winner_weight": metrics.get(
            "horizon_voting_average_winner_weight"
        ),
        "horizon_voting_average_cash_vote_weight": metrics.get(
            "horizon_voting_average_cash_vote_weight"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A/B test of the canonical RAW+split MCT Control against a "
            "weighted multi-horizon voting consensus guard."
        )
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--raw-collection", default=RAW_COLLECTION)
    parser.add_argument("--corporate-actions-collection", default=CA_COLLECTION)
    parser.add_argument(
        "--minimum-consensus-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--no-cash-override",
        action="store_true",
        help="Do not let a weighted CASH majority override an asset decision.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not 0.0 < float(args.minimum_consensus_weight) <= 1.0:
        raise ValueError(
            "--minimum-consensus-weight must be in (0, 1]."
        )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_client()
    try:
        db = get_database(client)
        job = _latest_job(db, args.job_id)
        request = BacktestExecutionRequest.model_validate(job["request"])
        start = _utc(request.start_date).normalize()
        end = _utc(
            request.analysis_end_date or request.end_date
        ).normalize()
        end_exclusive = end + pd.Timedelta(days=1)

        raw_collection = db[str(args.raw_collection)]
        ca_collection = db[str(args.corporate_actions_collection)]

        frames: dict[str, pd.DataFrame] = {}
        exclusions: list[dict[str, Any]] = []
        data_diagnostics: list[dict[str, Any]] = []

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
                    f"[data] {position}/{len(request.assets)} "
                    f"{symbol} excluded reason={issue['reason']}",
                    flush=True,
                )
                continue
            if raw.empty:
                exclusions.append(
                    {"symbol": symbol, "reason": "missing_raw_history"}
                )
                continue
            reconstructed, applied = _split_normalize(raw, actions)
            frames[symbol] = reconstructed
            data_diagnostics.append(
                {
                    "symbol": symbol,
                    "raw_rows": len(raw),
                    "corporate_actions": len(actions),
                    "splits_applied": len(applied),
                }
            )

        eligible = list(frames)
        anchors = [
            symbol
            for symbol in request.calendar_anchor_assets
            if symbol in frames
        ]
        references = [
            symbol
            for symbol in request.research_reference_assets
            if symbol in frames
        ]
        if len(references) < 2:
            references = list(anchors)
        reference_set = set(references)
        candidates = [
            symbol
            for symbol in request.research_candidate_assets
            if symbol in frames and symbol not in reference_set
        ]

        base_settings = deepcopy(request.research_model_settings)
        base_lightgbm = deepcopy(base_settings.get("lightgbm") or {})
        base_lightgbm["early_stopping_enabled"] = False
        base_settings["lightgbm"] = base_lightgbm
        base_settings["horizon_voting"] = {
            "enabled": False,
        }

        base_config = request.model_copy(
            update={
                "research_model_settings": base_settings,
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

        if bool(
            getattr(base_config, "optimized_allocation_enabled", False)
        ):
            raise ValueError(
                "Horizon voting v1 must run on the canonical single-position "
                "rotation policy, not optimized allocation."
            )

        _, common_dates = prepare_rotation_panel(frames, base_config)
        folds = _build_walk_forward_folds(common_dates, base_config)

        control_result, control_metrics = _run(
            label="CONTROL",
            frames=frames,
            config=base_config,
            folds=folds,
        )

        voting_settings = deepcopy(base_settings)
        voting_settings["horizon_voting"] = {
            "enabled": True,
            "minimum_consensus_weight": float(
                args.minimum_consensus_weight
            ),
            "cash_override_enabled": not bool(args.no_cash_override),
        }
        voting_config = base_config.model_copy(
            update={
                "research_model_settings": voting_settings,
            }
        )

        voting_result, voting_metrics = _run(
            label="HORIZON_VOTING",
            frames=frames,
            config=voting_config,
            folds=folds,
        )

        control_result.predictions.to_csv(
            output_dir / "control_predictions.csv",
            index=True,
        )
        control_result.trades.to_csv(
            output_dir / "control_trades.csv",
            index=False,
        )
        voting_result.predictions.to_csv(
            output_dir / "horizon_voting_predictions.csv",
            index=True,
        )
        voting_result.trades.to_csv(
            output_dir / "horizon_voting_trades.csv",
            index=False,
        )

        voting_columns = [
            column
            for column in voting_result.predictions.columns
            if str(column).startswith("horizon_voting_")
        ]
        if voting_columns:
            voting_result.predictions[voting_columns].to_csv(
                output_dir / "horizon_voting_decisions.csv",
                index=True,
            )

        comparison = pd.DataFrame(
            [
                _comparison_row("CONTROL", control_metrics),
                _comparison_row("HORIZON_VOTING", voting_metrics),
            ]
        )
        comparison.to_csv(
            output_dir / "strategy_comparison.csv",
            index=False,
        )
        pd.DataFrame(data_diagnostics).to_csv(
            output_dir / "data_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(exclusions).to_csv(
            output_dir / "excluded_assets.csv",
            index=False,
        )

        control_capital = float(control_metrics["ending_capital"])
        voting_capital = float(voting_metrics["ending_capital"])
        summary = {
            "schema_version": 1,
            "api_version": "10.8.73",
            "experiment": "raw-split-horizon-voting-consensus-v1",
            "source_job_id": job.get("id"),
            "raw_collection": str(args.raw_collection),
            "corporate_actions_collection": str(
                args.corporate_actions_collection
            ),
            "eligible_assets": eligible,
            "excluded_assets": exclusions,
            "methodology": {
                "control": (
                    "canonical weighted multi-horizon LightGBM utility policy"
                ),
                "challenger": (
                    "same Control policy plus independent per-horizon "
                    "LightGBM weighted consensus guard"
                ),
                "horizons": [
                    int(item)
                    for item in base_config.rotation_target_horizons
                ],
                "horizon_weights": [
                    float(item)
                    for item in base_config.rotation_target_horizon_weights
                ],
                "minimum_consensus_weight": float(
                    args.minimum_consensus_weight
                ),
                "cash_override_enabled": not bool(args.no_cash_override),
                "cash_vote_definition": (
                    "CASH wins a horizon when every finite asset utility "
                    "for that horizon is <= 0"
                ),
                "policy_rule": (
                    "Voting confirms the base asset, can veto a rotation, "
                    "or can override to CASH; it cannot jump directly to a "
                    "different asset in v1."
                ),
                "same_hyperparameters": True,
                "same_folds": True,
                "same_frozen_raw_split_snapshot": True,
                "same_cost_model": True,
                "same_buy_hold_benchmark": True,
                "oos_used_for_training_or_threshold_selection": False,
            },
            "control": control_metrics,
            "horizon_voting": voting_metrics,
            "comparison": {
                "capital_difference": voting_capital - control_capital,
                "capital_ratio": (
                    voting_capital / control_capital
                    if control_capital > 0
                    else None
                ),
                "cagr_difference": (
                    float(voting_metrics["cagr"])
                    - float(control_metrics["cagr"])
                ),
                "sharpe_difference": (
                    float(voting_metrics["sharpe"])
                    - float(control_metrics["sharpe"])
                ),
                "maximum_drawdown_difference": (
                    float(voting_metrics["maximum_drawdown"])
                    - float(control_metrics["maximum_drawdown"])
                ),
                "worst_fold_difference": (
                    float(voting_metrics["worst_fold_return"])
                    - float(control_metrics["worst_fold_return"])
                ),
            },
        }
        _checkpoint(output_dir / "summary.json", summary)

        print("")
        print("[done] horizon voting A/B completed", flush=True)
        print(
            f"[done] CONTROL={control_capital:,.2f}",
            flush=True,
        )
        print(
            f"[done] HORIZON_VOTING={voting_capital:,.2f}",
            flush=True,
        )
        print(
            f"[done] delta={voting_capital - control_capital:,.2f}",
            flush=True,
        )
        print(f"[done] output={output_dir}", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
