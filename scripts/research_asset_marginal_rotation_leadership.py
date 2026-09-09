from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_asset_rotation_leadership as research  # noqa: E402
import research_windows_file_io as file_io  # noqa: E402
from research_marginal_reproducibility import code_identity  # noqa: E402
from market_cycle_trader_api.services.asset_marginal_score_replay import (  # noqa: E402
    export_execution_rows, replay_policy_settings,
)

SCRIPT_VERSION = "asset-marginal-rotation-leadership-v2.0.0"
_ANALYSIS_FAILURES: dict[str, str] = {}
_REPLAY_CONFIG: dict[str, Any] = {}


def _install_replay_export() -> None:
    original_config = research._execution_config
    original_rows = research._prediction_rows

    def execution_config(*args: Any, **kwargs: Any):
        config = original_config(*args, **kwargs)
        _REPLAY_CONFIG.update({
            "policy_settings": replay_policy_settings(config),
            "maximum_label_horizon": max(int(h) for h in config.rotation_target_horizons),
        })
        return config

    def prediction_rows(symbol: str, frame: pd.DataFrame, fold: dict[str, Any], model: Any):
        rows = original_rows(symbol, frame, fold, model)
        return export_execution_rows(rows, frame, fold, _REPLAY_CONFIG["maximum_label_horizon"])

    research._execution_config = execution_config
    research._prediction_rows = prediction_rows


def _is_candidate_model_eligibility_failure(exc: Exception) -> bool:
    message = str(exc)
    return (
        "LightGBM did not fit a model" in message
        or ("utility rows are available" in message and "are required" in message)
    )


def _failed_aggregate(symbol: str, source: str, reason: str) -> dict[str, Any]:
    nan = float("nan")
    return {
        "symbol": symbol,
        "source": source,
        "fold_count": 0,
        "beat_buy_hold_fold_count": 0,
        "beat_buy_hold_fold_rate": 0.0,
        "median_fold_excess_return": nan,
        "mean_fold_excess_return": nan,
        "worst_fold_excess_return": nan,
        "best_fold_excess_return": nan,
        "fold_excess_std": nan,
        "compound_oos_timing_return": nan,
        "compound_oos_buy_hold_return": nan,
        "compound_oos_excess_return": nan,
        "median_timing_return": nan,
        "median_buy_hold_return": nan,
        "mean_drawdown_improvement": nan,
        "median_timing_sharpe": nan,
        "median_buy_hold_sharpe": nan,
        "mean_market_exposure": nan,
        "mean_upside_capture": nan,
        "mean_downside_avoidance": nan,
        "mean_return_while_in_market": nan,
        "mean_return_while_in_cash": nan,
        "total_buy_count": 0,
        "total_sell_count": 0,
        "mean_trade_return": nan,
        "mean_profitable_trade_rate": nan,
        "calibrated_switch_margin_mean": nan,
        "effective_switch_margin_mean": nan,
        "elapsed_seconds": 0.0,
        "analysis_status": "candidate_model_ineligible",
        "analysis_failure_reason": reason,
    }


def _placeholder_fold_rows(
    symbol: str,
    folds: list[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "symbol": symbol,
            "fold": int(fold["fold_id"]),
            "analysis_status": "candidate_model_ineligible",
            "analysis_failure_reason": reason,
        }
        for fold in folds
    ]


def _placeholder_prediction_rows(
    symbol: str,
    folds: list[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, fold in enumerate(folds):
        rows.append(
            {
                "fold": int(fold["fold_id"]),
                "timestamp": pd.Timestamp("1900-01-01", tz="UTC")
                + pd.Timedelta(days=offset),
                "symbol": symbol,
                "predicted_utility": np.nan,
                "realized_utility": np.nan,
                "forward_net_log_return": np.nan,
                "analysis_status": "candidate_model_ineligible",
                "analysis_failure_reason": reason,
            }
        )
    return rows


def _install_candidate_failure_isolation() -> None:
    original_analyse_asset = research._analyse_asset
    original_rank_predictions = research._rank_predictions
    original_leadership_qualification = research._leadership_qualification

    def resilient_analyse_asset(*args: Any, **kwargs: Any):
        symbol = str(args[0] if args else kwargs.get("symbol") or "unknown").upper()
        source = str(args[1] if len(args) > 1 else kwargs.get("source") or "")
        folds = list(args[5] if len(args) > 5 else kwargs.get("folds") or [])
        try:
            return original_analyse_asset(*args, **kwargs)
        except (RuntimeError, ValueError) as exc:
            if source != "candidate" or not _is_candidate_model_eligibility_failure(exc):
                raise
            reason = "model_not_trainable_in_required_walk_forward_fold: " + str(exc)
            _ANALYSIS_FAILURES[symbol] = reason
            research._log(
                f"{symbol}: CANDIDATE EXCLUDED - LightGBM cannot be trained with the required "
                f"usable rows in at least one walk-forward fold. reason={exc}"
            )
            return (
                _failed_aggregate(symbol, source, reason),
                _placeholder_fold_rows(symbol, folds, reason),
                _placeholder_prediction_rows(symbol, folds, reason),
            )

    def finite_rank_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
        frame = predictions.copy()
        numeric = pd.to_numeric(frame["predicted_utility"], errors="coerce")
        frame = frame.loc[np.isfinite(numeric)].copy()
        if frame.empty:
            raise RuntimeError(
                "No trainable assets produced finite OOS rotation predictions."
            )
        return original_rank_predictions(frame)

    def leadership_qualification(
        row: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        symbol = str(row.get("symbol") or "").upper()
        reason = _ANALYSIS_FAILURES.get(symbol)
        if reason:
            return False, [reason]
        return original_leadership_qualification(row)

    research._analyse_asset = resilient_analyse_asset
    research._rank_predictions = finite_rank_predictions
    research._leadership_qualification = leadership_qualification


def _install_long_path_safe_writes() -> None:
    research._write_json = file_io.write_json
    research._write_csv = file_io.write_csv


def main() -> int:
    args = research._parser().parse_args()
    identity = code_identity()
    if not args.no_resume:
        raise RuntimeError("v2 Leadership requires --no-resume (or the runner's --fresh-run) to export a complete execution tape.")
    _install_candidate_failure_isolation()
    _install_long_path_safe_writes()
    _install_replay_export()
    research.SCRIPT_VERSION = SCRIPT_VERSION
    result = int(research.main())
    output_dir = Path(args.output_dir or PROJECT_ROOT / "research_output" /
        f"asset_rotation_leadership_strategy_{args.strategy_sequence}_{pd.Timestamp(args.snapshot_end).date().isoformat()}")
    frozen = file_io.read_json(output_dir / "rotation_leadership_snapshot_frozen.json")
    predictions_path = output_dir / "leadership_predictions_raw.csv"
    with open(file_io.windows_long_path(predictions_path), "rb") as handle:
        predictions_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    with open(file_io.windows_long_path(output_dir / "asset_rotation_qualification.csv"), "rb") as handle:
        qualification_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    file_io.write_json(output_dir / "marginal_replay_contract.json", {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "code_identity": identity,
        **_REPLAY_CONFIG,
        "selection_end": frozen["snapshot_end"],
        "strategy_configuration_hash": frozen["strategy_configuration_hash"],
        "leadership_snapshot_sha256": frozen["decision_snapshot_sha256"],
        "predictions_sha256": predictions_hash,
        "qualification_sha256": qualification_hash,
        "margin_basis": "immutable Strategy configured margin; no OOS margin tuning",
        "execution_outcomes_used_as_action_features": False,
    })
    return result


if __name__ == "__main__":
    raise SystemExit(main())
