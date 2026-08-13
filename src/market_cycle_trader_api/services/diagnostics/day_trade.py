from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pymongo.database import Database

from .common import _safe_float

def _build_day_trade_diagnostics(
    frame: pd.DataFrame,
    trade_frame: pd.DataFrame,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    
    output: dict[str, Any] = {
        "rotation_summary": {},
        "rotation_diagnostics": [],
        "q_delta_buckets": [],
        "holding_distribution": [],
        "asset_performance": [],
        "session_performance": [],
    }

    rotations = pd.DataFrame()
    if not trade_frame.empty and "rotation_id" in trade_frame.columns:
        rotations = trade_frame.loc[
            trade_frame.get("action", pd.Series(index=trade_frame.index, dtype=object))
            .astype(str)
            .eq("BUY")
            & trade_frame["rotation_id"].notna()
        ].copy()

    rotation_rows: list[dict[str, Any]] = []
    if not rotations.empty:
        numeric_columns = [
            "q_current_position",
            "q_raw_best",
            "q_final_action",
            "q_delta_final_vs_current",
            "q_gap_best_vs_second",
        ]
        for horizon in (1, 2, 4, 8):
            numeric_columns.extend(
                [
                    f"from_return_after_{horizon}bar",
                    f"to_return_after_{horizon}bar",
                    f"rotation_advantage_{horizon}bar",
                ]
            )
        for column in numeric_columns:
            if column in rotations.columns:
                rotations[column] = pd.to_numeric(rotations[column], errors="coerce")

        for _, row in rotations.iterrows():
            item = {
                "timestamp": row.get("timestamp"),
                "decision_timestamp": row.get("decision_timestamp"),
                "from_asset": row.get("rotation_from_asset"),
                "to_asset": row.get("rotation_to_asset"),
                "q_current_position": _safe_float(row.get("q_current_position")),
                "q_target": _safe_float(row.get("q_final_action")),
                "q_delta": _safe_float(row.get("q_delta_final_vs_current")),
                "q_gap_best_vs_second": _safe_float(row.get("q_gap_best_vs_second")),
                "classification": row.get("rotation_classification") or "UNCLASSIFIED",
            }
            for horizon in (1, 2, 4, 8):
                item[f"from_return_{horizon}bar"] = _safe_float(
                    row.get(f"from_return_after_{horizon}bar")
                )
                item[f"to_return_{horizon}bar"] = _safe_float(
                    row.get(f"to_return_after_{horizon}bar")
                )
                item[f"advantage_{horizon}bar"] = _safe_float(
                    row.get(f"rotation_advantage_{horizon}bar")
                )
            rotation_rows.append(item)

        advantages_4 = pd.to_numeric(
            rotations.get("rotation_advantage_4bar"),
            errors="coerce",
        ).dropna()
        q_deltas = pd.to_numeric(
            rotations.get("q_delta_final_vs_current"),
            errors="coerce",
        ).dropna()
        classes = rotations.get(
            "rotation_classification",
            pd.Series(index=rotations.index, dtype=object),
        ).fillna("UNCLASSIFIED").astype(str)
        output["rotation_summary"] = {
            "rotation_count": int(len(rotations)),
            "good_rotations": int(classes.eq("GOOD_ROTATION").sum()),
            "bad_rotations": int(classes.eq("BAD_ROTATION").sum()),
            "neutral_rotations": int(classes.eq("NEUTRAL_ROTATION").sum()),
            "insufficient_future_bars": int(classes.eq("INSUFFICIENT_FUTURE_BARS").sum()),
            "average_rotation_advantage_4bar": (
                float(advantages_4.mean()) if not advantages_4.empty else None
            ),
            "median_rotation_advantage_4bar": (
                float(advantages_4.median()) if not advantages_4.empty else None
            ),
            "average_q_delta": float(q_deltas.mean()) if not q_deltas.empty else None,
            "median_q_delta": float(q_deltas.median()) if not q_deltas.empty else None,
        }

        def q_bucket(value: float) -> str:
            if value < 0.002:
                return "< 0.002"
            if value < 0.005:
                return "0.002–0.005"
            if value < 0.010:
                return "0.005–0.010"
            return ">= 0.010"

        bucket_frame = rotations.loc[
            pd.to_numeric(rotations.get("q_delta_final_vs_current"), errors="coerce").notna()
        ].copy()
        if not bucket_frame.empty:
            bucket_frame["_q_delta"] = pd.to_numeric(
                bucket_frame["q_delta_final_vs_current"], errors="coerce"
            )
            bucket_frame["_advantage_4"] = pd.to_numeric(
                bucket_frame.get("rotation_advantage_4bar"), errors="coerce"
            )
            bucket_frame["_bucket"] = bucket_frame["_q_delta"].map(q_bucket)
            ordered = ["< 0.002", "0.002–0.005", "0.005–0.010", ">= 0.010"]
            for bucket in ordered:
                part = bucket_frame.loc[bucket_frame["_bucket"] == bucket]
                if part.empty:
                    continue
                valid_adv = part["_advantage_4"].dropna()
                output["q_delta_buckets"].append(
                    {
                        "q_delta_bucket": bucket,
                        "rotations": int(len(part)),
                        "average_q_delta": float(part["_q_delta"].mean()),
                        "average_advantage_4bar": (
                            float(valid_adv.mean()) if not valid_adv.empty else None
                        ),
                        "good_share": (
                            float((valid_adv >= 0.002).mean()) if not valid_adv.empty else None
                        ),
                        "bad_share": (
                            float((valid_adv <= -0.002).mean()) if not valid_adv.empty else None
                        ),
                    }
                )

    output["rotation_diagnostics"] = sorted(
        rotation_rows,
        key=lambda item: (
            item.get("advantage_4bar")
            if item.get("advantage_4bar") is not None
            else 999.0
        ),
    )

    if not trade_frame.empty and "action" in trade_frame.columns:
        exits = trade_frame.loc[
            trade_frame["action"].astype(str).isin(["SELL", "FINAL_SELL"])
        ].copy()
        if not exits.empty:
            exits["holding_bars"] = pd.to_numeric(exits.get("holding_bars"), errors="coerce")
            exits["realized_pnl"] = pd.to_numeric(exits.get("realized_pnl"), errors="coerce")
            exits["position_return"] = pd.to_numeric(exits.get("position_return"), errors="coerce")

            def hold_bucket(value: float) -> str:
                if value <= 1:
                    return "1 bar"
                if value <= 2:
                    return "2 bars"
                if value <= 4:
                    return "3–4 bars"
                if value <= 8:
                    return "5–8 bars"
                return "9+ bars"

            valid_holds = exits.loc[exits["holding_bars"].notna()].copy()
            if not valid_holds.empty:
                valid_holds["_bucket"] = valid_holds["holding_bars"].map(hold_bucket)
                order = ["1 bar", "2 bars", "3–4 bars", "5–8 bars", "9+ bars"]
                total = len(valid_holds)
                for bucket in order:
                    part = valid_holds.loc[valid_holds["_bucket"] == bucket]
                    if part.empty:
                        continue
                    output["holding_distribution"].append(
                        {
                            "holding_bucket": bucket,
                            "closed_positions": int(len(part)),
                            "share": float(len(part) / total),
                            "average_realized_pnl": float(part["realized_pnl"].mean()),
                            "average_position_return": float(part["position_return"].mean()),
                        }
                    )

            if "asset" in exits.columns:
                for asset, part in exits.groupby(exits["asset"].astype(str)):
                    pnl = part["realized_pnl"].dropna()
                    returns = part["position_return"].dropna()
                    output["asset_performance"].append(
                        {
                            "asset": str(asset),
                            "closed_positions": int(len(part)),
                            "wins": int((part["realized_pnl"] > 0).sum()),
                            "win_rate": float((part["realized_pnl"] > 0).mean()),
                            "realized_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
                            "average_position_return": float(returns.mean()) if not returns.empty else None,
                            "average_holding_bars": float(part["holding_bars"].mean()),
                        }
                    )
                output["asset_performance"] = sorted(
                    output["asset_performance"],
                    key=lambda item: float(item.get("realized_pnl") or 0.0),
                )

    if not frame.empty:
        session_frame = frame.copy()
        session_frame["_session"] = (
            session_frame["timestamp"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        )
        initial_capital = float((metrics or {}).get("initial_capital", 0.0) or 0.0)
        previous_strategy = initial_capital if initial_capital > 0 else None
        previous_benchmark = initial_capital if initial_capital > 0 else None
        for session, part in session_frame.groupby("_session", sort=True):
            strategy_end = float(part.iloc[-1]["strategy_equity"])
            benchmark_end = float(part.iloc[-1]["buy_hold_equity"])
            strategy_start = previous_strategy or float(part.iloc[0]["strategy_equity"])
            benchmark_start = previous_benchmark or float(part.iloc[0]["buy_hold_equity"])
            actions = part.get("trade_action", pd.Series(index=part.index, dtype=object)).fillna("").astype(str)
            output["session_performance"].append(
                {
                    "session": session,
                    "strategy_starting_capital": strategy_start,
                    "strategy_ending_capital": strategy_end,
                    "strategy_return": strategy_end / strategy_start - 1.0 if strategy_start > 0 else None,
                    "benchmark_return": benchmark_end / benchmark_start - 1.0 if benchmark_start > 0 else None,
                    "excess_return": (
                        strategy_end / strategy_start - benchmark_end / benchmark_start
                        if strategy_start > 0 and benchmark_start > 0
                        else None
                    ),
                    "rotations": int(actions.eq("ROTATE").sum()),
                    "cash_bars": int(part.get("selected_asset", pd.Series(index=part.index, dtype=object)).fillna("").astype(str).eq("CASH").sum()),
                    "bars": int(len(part)),
                }
            )
            previous_strategy = strategy_end
            previous_benchmark = benchmark_end

    return output
