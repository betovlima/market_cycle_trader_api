"""Protected operational binding for the frozen TCC v1.0.6 Control policy.

This module deliberately reuses the vendored scientific engine instead of
reimplementing its model or decision policy. The MCT owns orchestration,
current market-data acquisition, account state and orders; the Control signal
contract remains the TCC v1.0.6 contract.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from ..core.config import TCC_CONTROL_OPERATIONAL_MODE
from ..tcc_v106_reference.config import (
    ASSETS,
    CONFIG as TCC_CONFIG,
    REFERENCE_ASSETS,
    build_control_config,
)
from ..tcc_v106_reference.execution import (
    apply_slippage as tcc_apply_slippage,
    calculate_reference_fees as tcc_calculate_reference_fees,
)
from ..tcc_v106_reference.research_challengers import run_research_challenger
from .live_lightgbm_signal import LiveLightGBMDecision
from .operational_control_preview import (
    TCC_CONTROL_SOURCE_COMMIT,
    build_control_shadow_decision,
)

TCC_CONTROL_EXPECTED_CHECKPOINT_CAPITAL = 10_094_316.30
TCC_CONTROL_EXPERIMENT_VERSION = "1.0.6"
TCC_CONTROL_REQUESTED_ASSETS = tuple(ASSETS)


def tcc_control_expected_lightgbm_settings() -> dict[str, Any]:
    config = build_control_config(TCC_CONFIG)
    settings = dict(config.research_model_settings or {})
    return dict(settings.get("lightgbm") or {})


def tcc_control_model_snapshot_issues(snapshot: Any) -> list[str]:
    raw = snapshot if isinstance(snapshot, dict) else {}
    issues: list[str] = []
    family = str(raw.get("family") or "")
    if family != "lightgbm_utility":
        issues.append(
            f"family: expected='lightgbm_utility', actual={family!r}"
        )
        return issues
    settings_snapshot = (
        raw.get("settings_snapshot")
        if isinstance(raw.get("settings_snapshot"), dict)
        else {}
    )
    actual = (
        settings_snapshot.get("lightgbm")
        if isinstance(settings_snapshot.get("lightgbm"), dict)
        else {}
    )
    expected = tcc_control_expected_lightgbm_settings()
    for key, expected_value in expected.items():
        if key == "early_stopping_enabled":
            actual_value = actual.get(key, False)
        else:
            actual_value = actual.get(key)
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual_value == expected_value
        if not matches:
            issues.append(
                f"lightgbm.{key}: expected={expected_value!r}, actual={actual_value!r}"
            )
    return issues


def assert_tcc_control_model_snapshot(snapshot: Any) -> None:
    issues = tcc_control_model_snapshot_issues(snapshot)
    if issues:
        raise ValueError(
            "TCC Control v1.0.6 model snapshot mismatch: "
            + "; ".join(issues)
        )


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def tcc_control_contract_issues(config: Any) -> list[str]:
    """Return deviations from the protected operational TCC Control contract."""
    issues: list[str] = []

    expected_exact: dict[str, Any] = {
        "strategy_mode": TCC_CONTROL_OPERATIONAL_MODE,
        "start_date": "2016-01-01",
        "timeframe": "1Day",
        "market_data_provider": "alpaca",
        "alpaca_historical_feed": "sip",
        # MCT operational acquisition is physically RAW; causal split
        # normalization is performed locally before the scientific engine.
        "alpaca_adjustment": "raw",
        "rotation_horizon_days": 40,
        "rotation_target_horizons": [5, 10, 20, 40, 60],
        "rotation_target_horizon_weights": [0.10, 0.15, 0.20, 0.30, 0.25],
        "rotation_movement_capture_weight": 0.35,
        "rotation_trend_persistence_weight": 0.20,
        "rotation_minimum_training_rows": 700,
        "rotation_walk_forward_enabled": True,
        "rotation_walk_forward_calibration_days": 126,
        "rotation_walk_forward_test_days": 504,
        "rotation_walk_forward_min_test_days": 126,
        "rotation_purge_days": 60,
        "rotation_downside_penalty": 0.20,
        "rotation_drawdown_penalty": 0.35,
        "rotation_min_holding_days": 2,
        "rotation_min_expected_edge": 0.001,
        "rotation_cash_threshold": 0.0,
        "rotation_switch_margin": 0.0005,
        "rotation_switch_margin_candidates": [0.0, 0.0025, 0.005, 0.01],
        "initial_capital": 10_000.0,
        "whole_shares": False,
        "slippage_bps": 0.0,
        "commission_rate": 0.0,
        "sec_fee_rate": 0.0000206,
        "taf_fee_per_share": 0.000195,
        "taf_fee_cap": 9.79,
        "cat_fee_per_share": 0.000003,
        "random_state": 42,
    }

    for name, expected in expected_exact.items():
        actual = _value(config, name)
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        elif isinstance(expected, list):
            actual_list = list(actual or [])
            if expected and isinstance(expected[0], float):
                matches = (
                    len(actual_list) == len(expected)
                    and all(
                        math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)
                        for left, right in zip(actual_list, expected, strict=True)
                    )
                )
            else:
                matches = actual_list == expected
        else:
            matches = actual == expected
        if not matches:
            issues.append(f"{name}: expected={expected!r}, actual={actual!r}")

    model_family = _value(config, "research_model_family")
    if model_family is not None and str(model_family) != "lightgbm_utility":
        issues.append(
            f"research_model_family: expected='lightgbm_utility', actual={model_family!r}"
        )
    protocol = _value(config, "research_market_data_protocol")
    if protocol is not None and str(protocol) != "raw_total_causal_v1":
        issues.append(
            "research_market_data_protocol: expected='raw_total_causal_v1', "
            f"actual={protocol!r}"
        )

    assets = tuple(str(item).upper() for item in (_value(config, "assets", []) or []))
    if assets != TCC_CONTROL_REQUESTED_ASSETS:
        issues.append(
            "assets: expected exact frozen 56-symbol TCC universe in original order"
        )

    end_date = _value(config, "end_date")
    if end_date not in {None, ""}:
        issues.append("end_date: operational TCC Control must remain open-ended")

    return issues


def assert_tcc_control_operational_contract(config: Any) -> None:
    issues = tcc_control_contract_issues(config)
    if issues:
        raise ValueError(
            "TCC Control v1.0.6 operational contract mismatch: "
            + "; ".join(issues)
        )


def build_tcc_control_replay_config(
    base_config: Any,
    *,
    eligible_assets: list[str] | tuple[str, ...],
) -> Any:
    """Build the scientific TCC config while allowing only the evaluation cutoff.

    All model/policy parameters come from the vendored v1.0.6 source. The MCT
    request can select the current historical cutoff, but cannot mutate the
    TCC Control behavior.
    """
    config = build_control_config(TCC_CONFIG, assets=tuple(eligible_assets))
    analysis_end = _value(base_config, "analysis_end_date")
    end_date = _value(base_config, "end_date")
    updates: dict[str, Any] = {}
    if analysis_end:
        updates["analysis_end_date"] = analysis_end
    if end_date:
        updates["end_date"] = end_date
    return config.model_copy(update=updates) if updates else config


def run_tcc_control_operational_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    base_config: Any,
    *,
    progress_callback: Callable[[float, str, int], None] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_detail_callback: Callable[[dict[str, Any]], None] | None = None,
    technical_log_callback: Callable[[str], None] | None = None,
) -> list[Any]:
    """Run the exact vendored TCC Control engine on MCT-provided audited bars."""
    eligible_assets = [
        symbol for symbol in TCC_CONTROL_REQUESTED_ASSETS
        if symbol in bars_by_symbol
    ]
    if len(eligible_assets) < 2:
        raise ValueError("TCC Control requires at least two eligible assets.")
    config = build_tcc_control_replay_config(
        base_config,
        eligible_assets=eligible_assets,
    )
    results = run_research_challenger(
        "lightgbm_utility",
        bars_by_symbol,
        config,
        tcc_calculate_reference_fees,
        tcc_apply_slippage,
        progress_callback=progress_callback,
        trade_callback=trade_callback,
        progress_detail_callback=progress_detail_callback,
        technical_log_callback=technical_log_callback,
    )
    for result in results:
        result.metrics["operational_contract"] = "tcc_v1.0.6_control"
        result.metrics["tcc_source_commit"] = TCC_CONTROL_SOURCE_COMMIT
        result.metrics["tcc_experiment_version"] = TCC_CONTROL_EXPERIMENT_VERSION
    return results


def _completed_session_from_anchors(
    bars_by_symbol: dict[str, pd.DataFrame],
) -> str:
    missing = [symbol for symbol in REFERENCE_ASSETS if symbol not in bars_by_symbol]
    if missing:
        raise ValueError(
            "TCC Control live input is missing reference assets: "
            + ", ".join(missing)
        )
    last_dates = []
    for symbol in REFERENCE_ASSETS:
        frame = bars_by_symbol[symbol]
        if frame is None or frame.empty:
            raise ValueError(f"TCC Control live input is empty for {symbol}.")
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        last_dates.append(index[-1].date())
    cutoff = min(last_dates)
    return cutoff.isoformat()


def build_live_tcc_control_decision(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Any,
    *,
    current_asset: str | None,
    holding_sessions: int,
) -> LiveLightGBMDecision:
    """Produce a Trader-compatible decision using the scientific TCC runtime."""
    assert_tcc_control_operational_contract(config)
    completed_session = _completed_session_from_anchors(bars_by_symbol)
    result = build_control_shadow_decision(
        bars_by_symbol,
        completed_session=completed_session,
        current_asset=current_asset,
        holding_sessions=holding_sessions,
    )

    utilities: dict[str, float] = {}
    for label, raw in (result.get("utilities") or {}).items():
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            utilities[str(label)] = value

    return LiveLightGBMDecision(
        decision_date=pd.Timestamp(result["decision_date"]),
        current_asset=str(result["current_asset"]),
        target_asset=str(result["target_asset"]),
        raw_best_asset=str(result["raw_best_asset"]),
        selected_utility=float(result["selected_utility"]),
        utilities=utilities,
        cash_edges={},
        opportunity_probability=None,
        opportunity_confidence=None,
        opportunity_threshold=None,
        opportunity_accepted=None,
        effective_switch_margin=float(result["effective_switch_margin"]),
        calibrated_candidate_margin=float(result["calibrated_candidate_margin"]),
        calibration_score=float(result["calibration_score"]),
        training_end=pd.Timestamp(result["training_end"]),
        calibration_start=pd.Timestamp(result["calibration_start"]),
        calibration_end=pd.Timestamp(result["calibration_end"]),
        final_fit_end=pd.Timestamp(result["final_fit_end"]),
        effective_compute_device="cpu",
        compute_fallback_reason=None,
        random_state=42,
    )
