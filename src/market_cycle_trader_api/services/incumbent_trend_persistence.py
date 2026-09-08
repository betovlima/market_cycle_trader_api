from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IncumbentTrendPersistenceSettings:
    """Research-only guard that protects a still-healthy incumbent from noisy challengers."""

    enabled: bool = True
    max_current_rank: int = 5
    max_challenger_gap_zscore: float = 1.0
    min_evidence_count: int = 4

    @classmethod
    def from_config(cls, config: Any) -> "IncumbentTrendPersistenceSettings":
        research = getattr(config, "research_model_settings", {}) or {}
        raw = (
            research.get("incumbent_trend_persistence", {})
            if isinstance(research, dict)
            else {}
        )
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            max_current_rank=max(1, int(raw.get("max_current_rank", 5))),
            max_challenger_gap_zscore=max(
                0.0,
                float(raw.get("max_challenger_gap_zscore", 1.0)),
            ),
            min_evidence_count=max(
                1,
                min(5, int(raw.get("min_evidence_count", 4))),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "max_current_rank": int(self.max_current_rank),
            "max_challenger_gap_zscore": float(self.max_challenger_gap_zscore),
            "min_evidence_count": int(self.min_evidence_count),
            "evidence_rule": "at_least_n_of_5_causal_uptrend_signals",
            "signals": [
                "return_5 > 0",
                "return_20 > 0",
                "ema_5_vs_20 > 0",
                "ema_slope_20_5 > 0",
                "channel_position_20 >= 0.5",
            ],
        }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _causal_evidence(row: pd.Series) -> dict[str, bool]:
    def value(name: str) -> float | None:
        return _finite_float(row.get(name))

    return {
        "return_5_positive": (value("return_5") or 0.0) > 0.0,
        "return_20_positive": (value("return_20") or 0.0) > 0.0,
        "ema_5_above_20": (value("ema_5_vs_20") or 0.0) > 0.0,
        "ema_20_slope_positive": (value("ema_slope_20_5") or 0.0) > 0.0,
        "upper_half_20d_channel": (value("channel_position_20") or 0.0) >= 0.5,
    }


def wrap_incumbent_trend_persistence_policy(
    base_policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    *,
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_diagnostics: dict[pd.Timestamp, dict[str, Any]],
    settings: IncumbentTrendPersistenceSettings,
) -> Callable[[pd.Timestamp, int, int], tuple[int, float]]:
    """Wrap only the final causal policy; model fitting and margin calibration stay unchanged.

    A rotation is suppressed only when all three conditions hold:
    1. the incumbent remains near the top of the current cross-sectional ranking;
    2. the challenger's score advantage is not more than a configurable number of
       same-date universe standard deviations;
    3. at least N of five causal price/trend signals still support continuation.

    No future target column is read by this guard.
    """

    def policy(
        timestamp: pd.Timestamp,
        current_position: int,
        holding_days: int,
    ) -> tuple[int, float]:
        target_position, target_score = base_policy(
            timestamp,
            current_position,
            holding_days,
        )
        key = pd.Timestamp(timestamp)
        diagnostic = decision_diagnostics.get(key)
        if diagnostic is None:
            return target_position, target_score

        diagnostic.update(
            {
                "incumbent_persistence_enabled": bool(settings.enabled),
                "incumbent_persistence_guard_applied": False,
                "incumbent_persistence_max_current_rank": int(settings.max_current_rank),
                "incumbent_persistence_max_challenger_gap_zscore": float(
                    settings.max_challenger_gap_zscore
                ),
                "incumbent_persistence_min_evidence_count": int(
                    settings.min_evidence_count
                ),
            }
        )

        if (
            not settings.enabled
            or current_position <= 0
            or target_position <= 0
            or target_position == current_position
            or diagnostic.get("decision_reason") != "ROTATE_TO_BEST_ASSET"
        ):
            return target_position, target_score

        current_symbol = symbols[current_position - 1]
        frame = frames.get(current_symbol)
        if frame is None or key not in frame.index:
            diagnostic["incumbent_persistence_reject_reason"] = "CURRENT_FRAME_UNAVAILABLE"
            return target_position, target_score

        row = frame.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        evidence = _causal_evidence(row)
        evidence_count = int(sum(bool(value) for value in evidence.values()))

        current_rank_raw = diagnostic.get("current_asset_rank")
        try:
            current_rank = int(current_rank_raw) if current_rank_raw is not None else None
        except (TypeError, ValueError):
            current_rank = None

        gap = _finite_float(diagnostic.get("best_vs_current_gap"))
        universe_std = _finite_float(diagnostic.get("universe_score_std"))
        gap_zscore = (
            float(gap / universe_std)
            if gap is not None and universe_std is not None and universe_std > 1e-12
            else None
        )

        rank_ok = current_rank is not None and current_rank <= settings.max_current_rank
        gap_ok = (
            gap_zscore is not None
            and gap_zscore <= settings.max_challenger_gap_zscore
        )
        evidence_ok = evidence_count >= settings.min_evidence_count

        diagnostic.update(
            {
                "incumbent_persistence_current_rank": current_rank,
                "incumbent_persistence_challenger_gap_zscore": gap_zscore,
                "incumbent_persistence_evidence_count": evidence_count,
                "incumbent_persistence_rank_ok": bool(rank_ok),
                "incumbent_persistence_gap_ok": bool(gap_ok),
                "incumbent_persistence_evidence_ok": bool(evidence_ok),
                **{
                    f"incumbent_persistence_{name}": bool(value)
                    for name, value in evidence.items()
                },
            }
        )

        if not (rank_ok and gap_ok and evidence_ok):
            diagnostic["incumbent_persistence_reject_reason"] = ",".join(
                name
                for name, passed in (
                    ("RANK", rank_ok),
                    ("CHALLENGER_GAP", gap_ok),
                    ("UPTREND_EVIDENCE", evidence_ok),
                )
                if not passed
            )
            return target_position, target_score

        current_score = _finite_float(diagnostic.get("current_score"))
        if current_score is None:
            diagnostic["incumbent_persistence_reject_reason"] = "CURRENT_SCORE_UNAVAILABLE"
            return target_position, target_score

        diagnostic.update(
            {
                "incumbent_persistence_guard_applied": True,
                "incumbent_persistence_reject_reason": None,
                "incumbent_persistence_blocked_asset": diagnostic.get("best_asset"),
                "final_action_asset": current_symbol,
                "final_action_score": current_score,
                "final_action_cash_edge": diagnostic.get("current_cash_edge"),
                "decision_reason": "INCUMBENT_TREND_PERSISTENCE_HOLD",
                "decision_is_rotation": False,
                "switch_margin_guard_applied": False,
                "q_final_action": current_score,
                "q_delta_final_vs_current": 0.0,
            }
        )
        return current_position, current_score

    return policy
