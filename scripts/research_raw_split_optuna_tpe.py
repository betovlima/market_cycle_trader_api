from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import optuna
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
from market_cycle_trader_api.services.model_tuning import _SEARCH_SPACE
from market_cycle_trader_api.services.model_tuning_optuna import (
    OPTUNA_TPE_MODEL,
    ask_optuna_candidate,
    create_optuna_tpe_study,
    default_startup_trials,
    optuna_study_diagnostics,
    tell_optuna_candidate,
)
from market_cycle_trader_api.services.model_tuning_probability import (
    champion_gate_evaluation,
)
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
    _candidate_config,
    _checkpoint,
    _corporate_actions,
    _metrics,
    _structural_identity_issue,
)


DEFAULT_OUTPUT = "output/raw_split_optuna_tpe"


def _run_candidate(
    *,
    label: str,
    frames: dict[str, pd.DataFrame],
    config: BacktestExecutionRequest,
    folds: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    print(f"[optuna] starting {label}", flush=True)
    results = run_rotation_models(
        frames,
        config,
        _calculate_reference_fees,
        _apply_slippage,
        progress_callback=lambda p, stage, completed: print(
            f"[optuna] {label} progress={p:.1f}% "
            f"completed={completed} stage={stage}",
            flush=True,
        ),
    )
    if not results:
        raise RuntimeError(f"{label} returned no result.")
    result = results[0]
    metrics = _metrics(result, folds, float(config.initial_capital))
    print(
        f"[optuna] completed {label} "
        f"capital={metrics['ending_capital']:,.2f} "
        f"sharpe={metrics['sharpe']:.4f} "
        f"maxdd={metrics['maximum_drawdown']:.4%} "
        f"worst_fold={metrics['worst_fold_return']:.4%}",
        flush=True,
    )
    return result, metrics


def _control_anchor(
    *,
    settings_hash: str,
    settings: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source": "control",
        "candidate_id": 0,
        "settings_hash": settings_hash,
        "settings": deepcopy(settings),
        "metrics": deepcopy(metrics),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna TPE baseline on the frozen RAW Alpaca snapshot with "
            "local split normalization and the same MCT Champion gate."
        )
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--raw-collection", default=RAW_COLLECTION)
    parser.add_argument("--corporate-actions-collection", default=CA_COLLECTION)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-trials", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-dividend-features",
        action="store_true",
        help=(
            "Reserved for a separate controlled campaign. v10.8.70 uses "
            "RAW+split price features only."
        ),
    )
    args = parser.parse_args()

    if args.include_dividend_features:
        raise ValueError(
            "v10.8.70 calibrates the canonical RAW+split price architecture only. "
            "Dividend-feature tuning must be run separately."
        )
    if int(args.candidate_count) < 4:
        raise ValueError("--candidate-count must be at least 4.")
    if args.startup_trials is not None and int(args.startup_trials) < 4:
        raise ValueError("--startup-trials must be at least 4.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

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
                exclusions.append(
                    {"symbol": symbol, "reason": "missing_raw_history"}
                )
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
        anchors = [
            item for item in request.calendar_anchor_assets
            if item in frames
        ]
        if len(anchors) < 2:
            raise RuntimeError(
                "Structural corporate-action exclusions removed too many "
                "calendar anchors."
            )

        references = [
            item for item in request.research_reference_assets
            if item in frames
        ]
        if len(references) < 2:
            references = list(anchors)

        reference_set = set(references)
        candidates = [
            item
            for item in request.research_candidate_assets
            if item in frames and item not in reference_set
        ]

        research_settings = deepcopy(request.research_model_settings)
        lightgbm_methodology = deepcopy(
            research_settings.get("lightgbm") or {}
        )
        lightgbm_methodology["early_stopping_enabled"] = False
        research_settings["lightgbm"] = lightgbm_methodology

        base_config = request.model_copy(
            update={
                "research_model_settings": research_settings,
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

        _, common_dates = prepare_rotation_panel(frames, base_config)
        folds = _build_walk_forward_folds(common_dates, base_config)

        base_lightgbm = deepcopy(
            base_config.research_model_settings.get("lightgbm") or {}
        )
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
        baseline_result.predictions.to_csv(
            output_dir / "control_predictions.csv",
            index=True,
        )
        baseline_result.trades.to_csv(
            output_dir / "control_trades.csv",
            index=False,
        )

        control_settings_hash = hashlib.sha256(
            json.dumps(
                base_tuning_values,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        study, distributions, fixed_constraint_thresholds = (
            create_optuna_tpe_study(
                search_space=_SEARCH_SPACE,
                base_tuning_values=base_tuning_values,
                baseline_metrics=baseline_metrics,
                seed=int(args.seed),
                startup_trials=(
                    int(args.startup_trials)
                    if args.startup_trials is not None
                    else None
                ),
            )
        )

        gate_document: dict[str, Any] = {
            "probability_config": {
                "min_capital_improvement": 0.0,
                "sharpe_tolerance": 0.05,
                "drawdown_tolerance": 0.03,
                "min_worst_fold_return": 0.0,
            },
            "probability_anchor": _control_anchor(
                settings_hash=control_settings_hash,
                settings=base_tuning_values,
                metrics=baseline_metrics,
            ),
            "baseline_execution": {
                "metrics": deepcopy(baseline_metrics),
                "settings": deepcopy(base_tuning_values),
            },
        }

        resolved_startup_trials = int(
            int(args.startup_trials)
            if args.startup_trials is not None
            else default_startup_trials(_SEARCH_SPACE)
        )

        checkpoint: dict[str, Any] = {
            "schema_version": 1,
            "api_version": "10.8.70",
            "experiment": "raw-split-optuna-tpe-v1",
            "optimizer": OPTUNA_TPE_MODEL,
            "source_job_id": job.get("id"),
            "raw_collection": str(args.raw_collection),
            "corporate_actions_collection": str(
                args.corporate_actions_collection
            ),
            "eligible_assets": eligible,
            "excluded_assets": exclusions,
            "baseline_metrics": baseline_metrics,
            "baseline_settings": base_tuning_values,
            "control_settings_hash": control_settings_hash,
            "control_seeded_into_optimizer": True,
            "candidate_count": int(args.candidate_count),
            "seed": int(args.seed),
            "startup_trials": resolved_startup_trials,
            "fixed_optimizer_constraints": fixed_constraint_thresholds,
            "champion_gate": deepcopy(
                gate_document["probability_config"]
            ),
            "candidates": [],
        }
        _checkpoint(checkpoint_path, checkpoint)

        champion_result = baseline_result
        champion_metrics = baseline_metrics
        champion_settings = deepcopy(base_tuning_values)
        champion_candidate_id: int | None = None

        for iteration in range(1, int(args.candidate_count) + 1):
            trial, settings = ask_optuna_candidate(
                study,
                distributions,
            )
            candidate_id = int(trial.number)

            config = _candidate_config(base_config, settings)
            result, metrics = _run_candidate(
                label=f"CANDIDATE_{candidate_id}_OPTUNA_TPE",
                frames=frames,
                config=config,
                folds=folds,
            )

            gate = champion_gate_evaluation(
                gate_document,
                metrics,
            )
            gate_passed = bool(gate["passed"])

            violations = tell_optuna_candidate(
                study,
                trial,
                metrics,
                thresholds=fixed_constraint_thresholds,
                champion_gate_passed=gate_passed,
            )

            if gate_passed:
                champion_result = result
                champion_metrics = metrics
                champion_settings = deepcopy(settings)
                champion_candidate_id = candidate_id
                gate_document["probability_anchor"] = {
                    "source": "optuna_tpe_candidate",
                    "candidate_id": candidate_id,
                    "settings": deepcopy(settings),
                    "metrics": deepcopy(metrics),
                }

            candidate_payload = {
                "candidate_id": candidate_id,
                "iteration": iteration,
                "kind": "optuna_tpe",
                "optuna_trial_number": int(trial.number),
                "settings": deepcopy(settings),
                "metrics": deepcopy(metrics),
                "objective_ending_capital": float(
                    metrics["ending_capital"]
                ),
                "optimizer_constraints": violations,
                "optimizer_feasible": all(
                    float(value) <= 0.0
                    for value in violations.values()
                ),
                "champion_gate": gate,
                "champion_gate_passed": gate_passed,
            }
            checkpoint["candidates"].append(candidate_payload)
            checkpoint["champion_candidate_id"] = champion_candidate_id
            checkpoint["champion_metrics"] = deepcopy(champion_metrics)
            checkpoint["champion_settings"] = deepcopy(champion_settings)
            checkpoint["optuna_study"] = optuna_study_diagnostics(study)
            _checkpoint(checkpoint_path, checkpoint)

            print(
                f"[optuna] iteration={iteration}/{args.candidate_count} "
                f"trial={trial.number} "
                f"feasible={candidate_payload['optimizer_feasible']} "
                f"promoted={gate_passed}",
                flush=True,
            )

        champion_result.predictions.to_csv(
            output_dir / "champion_predictions.csv",
            index=True,
        )
        champion_result.trades.to_csv(
            output_dir / "champion_trades.csv",
            index=False,
        )

        rows: list[dict[str, Any]] = []
        for item in checkpoint["candidates"]:
            metrics = item["metrics"]
            constraints = item.get("optimizer_constraints") or {}
            rows.append(
                {
                    "candidate_id": item["candidate_id"],
                    "iteration": item["iteration"],
                    "kind": item["kind"],
                    "optuna_trial_number": item[
                        "optuna_trial_number"
                    ],
                    "optimizer_feasible": item[
                        "optimizer_feasible"
                    ],
                    "champion_gate_passed": item[
                        "champion_gate_passed"
                    ],
                    "ending_capital": metrics["ending_capital"],
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "maximum_drawdown": metrics[
                        "maximum_drawdown"
                    ],
                    "worst_fold_return": metrics[
                        "worst_fold_return"
                    ],
                    "constraint_sharpe": constraints.get("sharpe"),
                    "constraint_maximum_drawdown": constraints.get(
                        "maximum_drawdown"
                    ),
                    "constraint_worst_fold_return": constraints.get(
                        "worst_fold_return"
                    ),
                    "validation_mae": metrics.get(
                        "validation_mae"
                    ),
                    "validation_rmse": metrics.get(
                        "validation_rmse"
                    ),
                    "train_mae": metrics.get("train_mae"),
                    "train_rmse": metrics.get("train_rmse"),
                    "generalization_gap_rmse": metrics.get(
                        "generalization_gap_rmse"
                    ),
                    **item["settings"],
                }
            )
        pd.DataFrame(rows).to_csv(
            output_dir / "candidates.csv",
            index=False,
        )
        pd.DataFrame(split_diagnostics).to_csv(
            output_dir / "data_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(exclusions).to_csv(
            output_dir / "excluded_assets.csv",
            index=False,
        )

        control_model_diagnostics = deepcopy(
            baseline_result.metrics.get(
                "lightgbm_fold_diagnostics"
            ) or []
        )
        champion_model_diagnostics = deepcopy(
            champion_result.metrics.get(
                "lightgbm_fold_diagnostics"
            ) or []
        )
        _checkpoint(
            output_dir / "control_model_diagnostics.json",
            {
                "folds": control_model_diagnostics,
                "summary": baseline_result.metrics.get(
                    "lightgbm_predictive_diagnostics"
                ) or {},
            },
        )
        _checkpoint(
            output_dir / "champion_model_diagnostics.json",
            {
                "folds": champion_model_diagnostics,
                "summary": champion_result.metrics.get(
                    "lightgbm_predictive_diagnostics"
                ) or {},
            },
        )

        feature_rows: list[dict[str, Any]] = []
        champion_feature_importance = (
            (
                champion_result.metrics.get(
                    "lightgbm_predictive_diagnostics"
                ) or {}
            ).get("feature_importance_gain")
            or {}
        )
        control_feature_importance = (
            (
                baseline_result.metrics.get(
                    "lightgbm_predictive_diagnostics"
                ) or {}
            ).get("feature_importance_gain")
            or {}
        )
        for feature in sorted(
            set(control_feature_importance)
            | set(champion_feature_importance)
        ):
            feature_rows.append(
                {
                    "feature": feature,
                    "control_importance_gain": (
                        control_feature_importance.get(
                            feature,
                            0.0,
                        )
                    ),
                    "champion_importance_gain": (
                        champion_feature_importance.get(
                            feature,
                            0.0,
                        )
                    ),
                }
            )
        pd.DataFrame(feature_rows).to_csv(
            output_dir / "feature_importance_gain.csv",
            index=False,
        )

        summary = {
            "schema_version": 1,
            "api_version": "10.8.70",
            "experiment": "raw-split-optuna-tpe-v1",
            "optimizer": OPTUNA_TPE_MODEL,
            "source_job_id": job.get("id"),
            "raw_collection": str(args.raw_collection),
            "corporate_actions_collection": str(
                args.corporate_actions_collection
            ),
            "asset_count": len(eligible),
            "eligible_assets": eligible,
            "excluded_assets": exclusions,
            "candidate_count": int(args.candidate_count),
            "seed": int(args.seed),
            "search_space": [
                dict(item) for item in _SEARCH_SPACE
            ],
            "methodology": {
                "market_data": (
                    "frozen RAW Alpaca bars + local split "
                    "normalization"
                ),
                "structural_identity_guard": True,
                "lightgbm_early_stopping": False,
                "optimizer": "Optuna TPESampler",
                "optimizer_mode": (
                    "sequential deterministic seeded ask/tell"
                ),
                "sampler": {
                    "model": OPTUNA_TPE_MODEL,
                    "seed": int(args.seed),
                    "multivariate": True,
                    "constant_liar": False,
                    "startup_trials": resolved_startup_trials,
                },
                "objective": "maximize ending_capital",
                "optimizer_constraints": {
                    "role": (
                        "fixed Control-relative robustness "
                        "constraints for TPE feasibility"
                    ),
                    "thresholds": fixed_constraint_thresholds,
                    "semantics": "<= 0 violation is feasible",
                },
                "champion_gate": {
                    "role": (
                        "MCT promotion rule; remains independent "
                        "from Optuna objective"
                    ),
                    **gate_document["probability_config"],
                },
                "control_seeded_into_optimizer": True,
                "same_search_space_as_v10_8_69": True,
                "same_candidate_budget_as_v10_8_69": (
                    int(args.candidate_count) == 24
                ),
                "oos_used_for_optimizer": False,
                "predictive_diagnostics_role": (
                    "informative_only"
                ),
            },
            "baseline": baseline_metrics,
            "champion_candidate_id": champion_candidate_id,
            "champion_metrics": champion_metrics,
            "champion_settings": champion_settings,
            "optuna_study": optuna_study_diagnostics(study),
        }
        _checkpoint(output_dir / "summary.json", summary)

        print("")
        print(
            "[done] RAW+split Optuna TPE campaign completed",
            flush=True,
        )
        print(
            f"[done] control_capital="
            f"{baseline_metrics['ending_capital']:,.2f}",
            flush=True,
        )
        print(
            f"[done] champion_capital="
            f"{champion_metrics['ending_capital']:,.2f}",
            flush=True,
        )
        print(
            f"[done] champion_candidate_id="
            f"{champion_candidate_id}",
            flush=True,
        )
        print(f"[done] output={output_dir}", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
