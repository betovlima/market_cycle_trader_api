"""Research-only marginal selection using one chronological account per universe.

Scores are already OOS. Future price columns are execution outcomes, never inputs
to the action. The full Strategy engine/model fitting is not run per candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import _execute_buy
from market_cycle_trader_api.engine.compound_rotation_backtest import (
    apply_slippage, calculate_reference_fees,
)
from .asset_marginal_rotation_contribution import _parse_utc_timestamps


POLICY_FIELDS = (
    "strategy_mode", "initial_capital", "rotation_cash_threshold",
    "rotation_min_expected_edge", "rotation_min_holding_days",
    "rotation_switch_margin", "fractional_shares", "slippage_bps",
    "commission_rate", "cat_fee_per_share", "sec_fee_rate",
    "taf_fee_per_share", "taf_fee_cap",
)
PRICE_COLUMNS = ("execution_open", "execution_close")
TIMESTAMP_COLUMNS = ("timestamp", "execution_timestamp", "training_label_last_timestamp")
REPLAY_COLUMNS = (
    "fold", "symbol", "predicted_utility", "decision_session_id",
    "execution_session_id", *TIMESTAMP_COLUMNS, *PRICE_COLUMNS,
)


def replay_policy_settings(config: Any) -> dict[str, Any]:
    values = {key: getattr(config, key) for key in POLICY_FIELDS}
    if values["strategy_mode"] != "COMPOUND_ROTATION_SWING_XGBOOST":
        raise ValueError("Cached score replay currently supports the single-position legacy Utility policy only.")
    for key, value in values.items():
        if key not in {"strategy_mode", "fractional_shares"} and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite replay configuration: {key}")
    if float(values["initial_capital"]) <= 0:
        raise ValueError("Replay initial capital must be positive.")
    return values


def export_execution_rows(
    rows: list[dict[str, Any]], frame: pd.DataFrame,
    fold: dict[str, Any], maximum_label_horizon: int,
) -> list[dict[str, Any]]:
    """Attach observed next-session execution prices to already computed scores."""
    last_label = int(fold["final_fit_end_index"]) - 1 + maximum_label_horizon
    label_timestamp = pd.Timestamp(frame.index[last_label])
    result = []
    for row in rows:
        now = pd.Timestamp(row["timestamp"])
        location = frame.index.get_loc(now)
        if not isinstance(location, (int, np.integer)) or location + 1 >= len(frame):
            raise ValueError("OOS row has no unique next execution session.")
        future = frame.iloc[location + 1]
        result.append({
            **row,
            "decision_session_id": int(location),
            "execution_session_id": int(location + 1),
            "execution_timestamp": pd.Timestamp(frame.index[location + 1]).isoformat(),
            "execution_open": float(future["open"]),
            "execution_close": float(future["close"]),
            "training_label_last_timestamp": label_timestamp.isoformat(),
        })
    return result


@dataclass
class ReplayResult:
    daily: pd.DataFrame
    ending_capital: float
    net_log_growth: float
    rotations: int
    fees: float


class ScoreReplayPanel:
    def __init__(self, predictions: pd.DataFrame, baseline_assets: list[str], selection_end: str):
        missing = sorted(set(REPLAY_COLUMNS) - set(predictions.columns))
        if missing:
            raise ValueError("A fresh v2 Leadership export is required; missing replay columns: " + ", ".join(missing))
        frame = predictions[list(REPLAY_COLUMNS)].copy()
        # v1 candidate-failure placeholders contain no predictions; they must not
        # define the common calendar or silently make a baseline unscoreable.
        frame = frame.loc[pd.to_numeric(frame.predicted_utility, errors="coerce").notna()].copy()
        frame["symbol"] = frame.symbol.astype(str).str.strip().str.upper()
        if frame.empty or frame.symbol.isin(["", "CASH", "NAN", "NONE"]).any():
            raise ValueError("Empty or invalid replay symbols.")
        for col in TIMESTAMP_COLUMNS:
            frame[col] = _parse_utc_timestamps(frame[col])
            if frame[col].isna().any():
                raise ValueError(f"Missing replay timestamp: {col}")
        for col in ("fold", "decision_session_id", "execution_session_id", "predicted_utility", *PRICE_COLUMNS):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite replay column: {col}")
        for col in ("fold", "decision_session_id", "execution_session_id"):
            if (frame[col] % 1 != 0).any():
                raise ValueError(f"Non-integral replay session/fold: {col}")
        if frame.duplicated(["timestamp", "symbol"]).any():
            raise ValueError("Duplicate OOS timestamp/symbol rows; no keep-last resolution is allowed.")
        cutoff = pd.Timestamp(selection_end).tz_localize("UTC") if pd.Timestamp(selection_end).tzinfo is None else pd.Timestamp(selection_end).tz_convert("UTC")
        if (frame.execution_timestamp.dt.normalize() > cutoff.normalize()).any():
            raise ValueError("Execution outcomes extend beyond selection_end.")
        if (frame.training_label_last_timestamp > frame.timestamp).any():
            raise ValueError("Training labels extend beyond their OOS decision.")
        if (frame.execution_timestamp <= frame.timestamp).any():
            raise ValueError("Execution must occur after the decision.")
        if (frame.execution_session_id != frame.decision_session_id + 1).any():
            raise ValueError("Execution must be on the next anchored session.")
        if (frame[list(PRICE_COLUMNS)] <= 0).any(axis=None):
            raise ValueError("Execution prices must be positive.")
        metadata = ["fold", "decision_session_id", "execution_session_id", "execution_timestamp"]
        if (frame.groupby("timestamp")[metadata].nunique() != 1).any(axis=None):
            raise ValueError("Assets disagree on the anchored execution calendar.")
        self.calendar = frame.drop_duplicates("timestamp").set_index("timestamp")[metadata].sort_index()
        if len(self.calendar) < 2:
            raise ValueError("Replay requires at least two execution sessions.")
        if not np.all(np.diff(self.calendar.decision_session_id) == 1):
            raise ValueError("The replay calendar has a missing/overlapping decision session.")
        if not np.array_equal(self.calendar.execution_timestamp.iloc[:-1].array, self.calendar.index[1:].array):
            raise ValueError("Execution/next-decision chain is broken.")
        self.scores = frame.pivot(index="timestamp", columns="symbol", values="predicted_utility").reindex(self.calendar.index)
        self.opens = frame.pivot(index="timestamp", columns="symbol", values="execution_open").reindex(self.scores.index, columns=self.scores.columns)
        self.closes = frame.pivot(index="timestamp", columns="symbol", values="execution_close").reindex(self.scores.index, columns=self.scores.columns)
        self.baseline = list(dict.fromkeys(s.strip().upper() for s in baseline_assets))
        if len(self.baseline) < 2:
            raise ValueError("At least two immutable baseline assets are required.")
        for symbol in self.baseline:
            if not self.complete(symbol):
                raise ValueError(f"Baseline {symbol} does not cover every OOS decision and execution.")

    def complete(self, symbol: str) -> bool:
        return symbol in self.scores and all(
            np.isfinite(frame[symbol].to_numpy(dtype=float)).all()
            for frame in (self.scores, self.opens, self.closes)
        )

    def replay(self, assets: list[str], settings: dict[str, Any]) -> ReplayResult:
        config = SimpleNamespace(**settings)
        replay_policy_settings(config)
        symbols = sorted(set(assets))  # Matches the Strategy tie-breaking order.
        if not symbols or any(not self.complete(s) for s in symbols):
            raise ValueError("Every replay asset must cover the immutable calendar.")
        scores = self.scores[symbols].to_numpy(dtype=float)
        opens = self.opens[symbols].to_numpy(dtype=float)
        closes = self.closes[symbols].to_numpy(dtype=float)
        cash = float(config.initial_capital)
        quantity, position, holding = 0.0, -1, 0
        fees_total, rotations = 0.0, 0
        equities, positions, reasons = [], [], []
        for i, utilities in enumerate(scores):
            # The decision uses scores and current account state only. Price
            # outcomes are read below, after the action has been determined.
            values = np.r_[0.0, utilities]
            best = int(np.argmax(values)) - 1
            best_value = float(values[best + 1])
            current_value = float(values[position + 1])
            if position >= 0 and holding < int(config.rotation_min_holding_days):
                target, reason = position, "MIN_HOLD_GUARD"
            elif best == -1 or best_value <= float(config.rotation_cash_threshold):
                target, reason = -1, "CASH_THRESHOLD"
            elif position == -1:
                accepted = best_value >= float(config.rotation_cash_threshold) + float(config.rotation_min_expected_edge)
                target, reason = (best, "ENTER_BEST_ASSET") if accepted else (-1, "MIN_EXPECTED_EDGE_GUARD")
            elif best == position:
                target, reason = position, "HOLD_CURRENT_BEST"
            elif best_value >= current_value + float(config.rotation_switch_margin):
                target, reason = best, "ROTATE_TO_BEST_ASSET"
            else:
                target, reason = position, "SWITCH_MARGIN_GUARD"
            if target != position:
                if position >= 0:
                    price = apply_slippage(float(opens[i, position]), "SELL", config)
                    fee = calculate_reference_fees("SELL", quantity, price, config)["total_fee"]
                    cash += quantity * price - fee
                    fees_total += fee
                if target >= 0:
                    quantity, price, buy_fees = _execute_buy(cash, float(opens[i, target]), config, calculate_reference_fees, apply_slippage)
                    cash -= quantity * price + buy_fees["total_fee"]
                    fees_total += buy_fees["total_fee"]
                else:
                    quantity = 0.0
                rotations += int(position >= 0 and target >= 0)
                position, holding = target, 1 if target >= 0 else 0
            elif position >= 0:
                holding += 1
            equity = cash + (quantity * closes[i, position] if position >= 0 else 0.0)
            if not math.isfinite(equity) or equity <= 0:
                raise ValueError("Replay account reached non-positive/non-finite equity.")
            equities.append(float(equity))
            positions.append(symbols[position] if position >= 0 else "CASH")
            reasons.append(reason)
        if position >= 0:
            price = apply_slippage(float(closes[-1, position]), "SELL", config)
            fee = calculate_reference_fees("SELL", quantity, price, config)["total_fee"]
            cash += quantity * price - fee
            fees_total += fee
            equities[-1] = cash
        if cash <= 0:
            raise ValueError("Final liquidation exhausted the replay account.")
        daily = self.calendar.reset_index().copy()
        daily["selected_asset"] = positions
        daily["decision_reason"] = reasons
        daily["strategy_equity"] = equities
        daily["net_log_return"] = np.diff(np.log(np.r_[config.initial_capital, equities]))
        # PCEA needs the policy-sufficient holding state to know when two paths
        # have truly reconverged. Exact holding counts above this threshold do not
        # affect future decisions, so export the configured threshold as metadata.
        daily.attrs["rotation_min_holding_days"] = int(config.rotation_min_holding_days)
        return ReplayResult(daily, float(cash), math.log(cash / config.initial_capital), rotations, fees_total)


@dataclass
class MarginalReplaySelection:
    baseline_assets: list[str]
    selected_candidates: list[str]
    final_assets: list[str]
    selected_steps: pd.DataFrame
    all_evaluations: pd.DataFrame
    selected_events: pd.DataFrame
    fold_evaluations: pd.DataFrame
    baseline_replay: ReplayResult
    final_replay: ReplayResult


def greedy_score_replay_selection(
    *, predictions: pd.DataFrame, baseline_assets: list[str], candidate_assets: list[str],
    settings: dict[str, Any], selection_end: str,
) -> MarginalReplaySelection:
    panel = ScoreReplayPanel(predictions, baseline_assets, selection_end)
    selected_assets = list(panel.baseline)
    current = baseline = panel.replay(selected_assets, settings)
    remaining = sorted({s.strip().upper() for s in candidate_assets} - set(selected_assets))
    selected, evaluations, steps, effects, folds = [], [], [], [], []
    while remaining:
        step = len(selected) + 1
        best = None
        for candidate in remaining:
            row = {"step": step, "candidate": candidate, "current_universe_size": len(selected_assets)}
            if not panel.complete(candidate):
                evaluations.append({**row, "status": "incomplete_oos_coverage", "marginal_net_log_growth": None})
                continue
            trial = panel.replay([*selected_assets, candidate], settings)
            delta = trial.net_log_growth - current.net_log_growth
            daily_delta = trial.daily.net_log_return - current.daily.net_log_return
            row.update({
                "status": "evaluated", "marginal_net_log_growth": float(delta),
                "current_ending_capital": current.ending_capital,
                "expanded_ending_capital": trial.ending_capital,
                "capital_delta": trial.ending_capital - current.ending_capital,
                "changed_position_sessions": int((trial.daily.selected_asset != current.daily.selected_asset).sum()),
                "rotation_delta": trial.rotations - current.rotations,
                "fee_delta": trial.fees - current.fees,
            })
            evaluations.append(row)
            for fold_id in trial.daily.fold.unique():
                mask = trial.daily.fold == fold_id
                folds.append({"step": step, "candidate": candidate, "fold": int(fold_id),
                              "sessions": int(mask.sum()), "marginal_net_log_growth": float(daily_delta[mask].sum())})
            if delta > 0.0 and (best is None or delta > best[0]["marginal_net_log_growth"]):
                best = row, trial
        if best is None:
            break
        row, trial = best
        candidate = row["candidate"]
        effect = trial.daily[["timestamp", "execution_timestamp", "fold"]].copy()
        effect["selection_step"], effect["selected_candidate"] = step, candidate
        effect["current_asset"] = current.daily.selected_asset
        effect["expanded_asset"] = trial.daily.selected_asset
        effect["marginal_net_log_return"] = trial.daily.net_log_return - current.daily.net_log_return
        effects.append(effect)
        selected.append(candidate)
        selected_assets.append(candidate)
        steps.append({**row, "selected": True, "resulting_universe_size": len(selected_assets)})
        remaining.remove(candidate)
        current = trial
    return MarginalReplaySelection(
        panel.baseline, selected, selected_assets,
        pd.DataFrame(steps, columns=list(steps[0]) if steps else ["step", "candidate", "marginal_net_log_growth", "selected"]),
        pd.DataFrame(evaluations, columns=list(dict.fromkeys(k for r in evaluations for k in r)) if evaluations else ["step", "candidate", "status", "marginal_net_log_growth"]),
        pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(columns=["timestamp", "selection_step", "selected_candidate", "marginal_net_log_return"]),
        pd.DataFrame(folds, columns=["step", "candidate", "fold", "sessions", "marginal_net_log_growth"]),
        baseline, current,
    )
