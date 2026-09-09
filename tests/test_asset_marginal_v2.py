from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from market_cycle_trader_api.services.asset_marginal_score_replay import (
    ScoreReplayPanel, export_execution_rows, greedy_score_replay_selection,
    replay_policy_settings,
)
from market_cycle_trader_api.engine.capital_rotation import (
    ROTATION_FEATURES, _cagr, _simulate_exact, _utility_policy,
)
from market_cycle_trader_api.engine.compound_rotation_backtest import (
    apply_slippage, calculate_reference_fees,
)
from market_cycle_trader_api.schemas.requests import BacktestExecutionRequest
import research_asset_marginal_rotation_contribution as selector
from research_marginal_reproducibility import market_data_hashes, verify_validation_pair


def fixture():
    values = json.loads((ROOT / "src/market_cycle_trader_api/parameterizations/winner-v1.13.2.json").read_text())
    values.update({
        "assets": ["AAA", "BBB", "CCC"], "start_date": "2020-01-01", "end_date": "2025-01-01",
        "analysis_start_date": "2024-01-01", "analysis_end_date": "2025-01-01",
        "strategy_mode": "COMPOUND_ROTATION_SWING_XGBOOST", "initial_capital": 10000,
        "whole_shares": False, "slippage_bps": 7,
        "commission_rate": 0.0003, "rotation_min_holding_days": 2,
        "rotation_switch_margin": 0.05, "rotation_min_expected_edge": 0.02,
        "calendar_anchor_assets": ["AAA", "BBB", "CCC"],
    })
    config = BacktestExecutionRequest.model_validate(values)
    dates = pd.bdate_range("2024-01-02", periods=91, tz="UTC")
    rng = np.random.default_rng(4421)
    frames, rows, scores = {}, [], {}
    for number, symbol in enumerate(sorted(config.assets)):
        close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.045, len(dates))))
        opens = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.015, len(dates)))
        frame = pd.DataFrame(0.0, index=dates, columns=ROTATION_FEATURES)
        frame["open"], frame["close"] = opens, close
        frame["high"], frame["low"] = np.maximum(opens, close) * 1.01, np.minimum(opens, close) * .99
        frame["volume"] = 1e6
        frames[symbol] = frame
        values = rng.normal(0.02, 0.1, len(dates))
        values[20:25] = -.1  # Exercise cash and minimum-hold decisions.
        values[40:45] = .1   # Exercise tied asset scores.
        scores[symbol] = pd.Series(values, index=dates)
        for i, date in enumerate(dates[:-1]):
            rows.append({"fold": 1 + i // 30, "timestamp": date.isoformat(), "symbol": symbol,
                         "predicted_utility": values[i], "decision_session_id": 100 + i,
                         "execution_session_id": 101 + i, "execution_timestamp": dates[i + 1].isoformat(),
                         "execution_open": opens[i + 1], "execution_close": close[i + 1],
                         "training_label_last_timestamp": dates[0].isoformat()})
    return config, dates, frames, pd.DataFrame(rows), scores


class CachedModel:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, features):
        return self.scores.loc[features.index].to_numpy()


class MarginalV2Tests(unittest.TestCase):
    def test_replay_matches_full_engine_positions_curve_fees_and_rotations(self):
        config, dates, frames, tape, scores = fixture()
        for fractional in (True, False):
            with self.subTest(fractional=fractional):
                config = config.model_copy(update={"whole_shares": not fractional})
                symbols = sorted(frames)
                models = {s: CachedModel(scores[s]) for s in symbols}
                policy = _utility_policy(models, frames, symbols, config, config.rotation_switch_margin)
                full = _simulate_exact("parity", policy, frames, symbols, dates, config,
                                       calculate_reference_fees, apply_slippage)
                result = ScoreReplayPanel(tape, ["AAA", "BBB"], str(dates[-1])).replay(symbols, replay_policy_settings(config))
                np.testing.assert_allclose(result.daily.strategy_equity, full.predictions.strategy_equity, rtol=1e-12)
                self.assertEqual(result.daily.selected_asset.tolist(), full.predictions.selected_asset.tolist())
                self.assertEqual(result.rotations, full.metrics["capital_rotations"])
                self.assertAlmostEqual(result.fees, full.metrics["total_transaction_fees"], places=10)
                self.assertAlmostEqual(result.daily.net_log_return.sum(), result.net_log_growth, places=12)
                self.assertAlmostEqual(full.metrics["strategy_cagr"],
                    (result.ending_capital / config.initial_capital) ** (365.25 / (dates[-1] - dates[1]).days) - 1, places=12)

    def test_future_prices_and_legacy_forward_labels_cannot_choose_an_action(self):
        config, dates, _, tape, _ = fixture()
        settings = replay_policy_settings(config)
        original = ScoreReplayPanel(tape, ["AAA", "BBB"], str(dates[-1])).replay(["AAA", "BBB", "CCC"], settings)
        changed = tape.copy()
        after = changed.decision_session_id >= 160
        changed.loc[after, "execution_close"] *= 1.3
        changed["forward_net_log_return"] = 1e15
        trial = ScoreReplayPanel(changed, ["AAA", "BBB"], str(dates[-1])).replay(["AAA", "BBB", "CCC"], settings)
        pd.testing.assert_series_equal(original.daily.selected_asset, trial.daily.selected_asset)
        np.testing.assert_array_equal(original.daily.strategy_equity[:60], trial.daily.strategy_equity[:60])

    def test_bad_calendar_future_labels_and_post_cutoff_prices_are_rejected(self):
        _, dates, _, tape, _ = fixture()
        for mutation, pattern in [
            (lambda p: pd.concat([p, p.iloc[:1]]), "Duplicate"),
            (lambda p: p.assign(training_label_last_timestamp=dates[5].isoformat()), "Training labels"),
            (lambda p: p.assign(execution_session_id=p.execution_session_id + 1), "next anchored"),
            (lambda p: p.loc[p.decision_session_id != 130], "missing/overlapping"),
            (lambda p: p.assign(execution_open=0), "positive"),
        ]:
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                ScoreReplayPanel(mutation(tape.copy()), ["AAA", "BBB"], str(dates[-1]))
        with self.assertRaisesRegex(ValueError, "selection_end"):
            ScoreReplayPanel(tape, ["AAA", "BBB"], str(dates[-2]))

    def test_incomplete_candidate_does_not_shrink_baseline_calendar(self):
        config, dates, _, tape, _ = fixture()
        tape = tape.loc[~((tape.symbol == "CCC") & (tape.decision_session_id == 123))]
        result = greedy_score_replay_selection(predictions=tape, baseline_assets=["AAA", "BBB"],
            candidate_assets=["CCC"], settings=replay_policy_settings(config), selection_end=str(dates[-1]))
        self.assertEqual(len(result.baseline_replay.daily), 90)
        self.assertEqual(result.selected_candidates, [])
        self.assertEqual(result.all_evaluations.iloc[0].status, "incomplete_oos_coverage")
        with self.assertRaisesRegex(ValueError, "Baseline CCC"):
            ScoreReplayPanel(tape, ["AAA", "CCC"], str(dates[-1]))

    def test_greedy_additions_recompute_the_entire_account_and_telescope(self):
        config, dates, _, tape, _ = fixture()
        # Complementary alternating opportunities plus a redundant duplicate.
        tape["execution_open"] = 100.0
        tape["execution_close"] = 100.0
        tape["predicted_utility"] = .1
        c = tape.symbol == "CCC"
        tape.loc[c, "predicted_utility"] = np.r_[np.full(45, .3), np.zeros(45)]
        clone = tape.loc[c].assign(symbol="DDD").copy()
        clone["predicted_utility"] = np.r_[np.zeros(45), np.full(45, .3)]
        for frame, mask, growth in ((tape, c, np.r_[np.full(45, .005), np.zeros(45)]),
                                    (clone, np.ones(len(clone), dtype=bool), np.r_[np.zeros(45), np.full(45, .005)])):
            closes = 100 * np.exp(np.cumsum(growth))
            frame.loc[mask, "execution_close"] = closes
            frame.loc[mask, "execution_open"] = np.r_[100, closes[:-1]]
        tape = pd.concat([tape, clone, clone.assign(symbol="EEE")], ignore_index=True)
        result = greedy_score_replay_selection(predictions=tape, baseline_assets=["AAA", "BBB"],
            candidate_assets=["EEE", "DDD", "CCC"], settings=replay_policy_settings(config), selection_end=str(dates[-1]))
        self.assertEqual(set(result.selected_candidates), {"CCC", "DDD"})
        expected = result.final_replay.net_log_growth - result.baseline_replay.net_log_growth
        self.assertAlmostEqual(result.selected_steps.marginal_net_log_growth.sum(), expected, places=12)
        self.assertAlmostEqual(result.selected_events.marginal_net_log_return.sum(), expected, places=12)
        self.assertTrue((result.selected_steps.marginal_net_log_growth > 0).all())

    def test_no_profitable_candidate_keeps_an_identical_account(self):
        config, dates, _, tape, _ = fixture()
        tape["execution_open"], tape["execution_close"] = 100.0, 100.0
        tape["predicted_utility"] = .1
        tape.loc[tape.symbol == "CCC", "predicted_utility"] = .5
        tape.loc[tape.symbol == "CCC", "execution_close"] = 90.0
        result = greedy_score_replay_selection(predictions=tape, baseline_assets=["AAA", "BBB"],
            candidate_assets=["CCC"], settings=replay_policy_settings(config), selection_end=str(dates[-1]))
        self.assertEqual(result.selected_candidates, [])
        self.assertEqual(result.final_replay.ending_capital, result.baseline_replay.ending_capital)

    def test_execution_export_uses_next_anchored_row_and_mature_training_labels(self):
        _, dates, frames, _, scores = fixture()
        original = [{"timestamp": dates[20].isoformat(), "symbol": "AAA", "fold": 1,
                     "predicted_utility": scores["AAA"].iloc[20]}]
        rows = export_execution_rows(original, frames["AAA"], {"final_fit_end_index": 15}, 5)
        self.assertEqual(rows[0]["execution_timestamp"], dates[21].isoformat())
        self.assertEqual(rows[0]["training_label_last_timestamp"], dates[19].isoformat())
        self.assertEqual(rows[0]["execution_open"], frames["AAA"].open.iloc[21])
        self.assertNotIn("execution_open", original[0])

    def test_old_tape_requires_explicit_legacy_method(self):
        _, dates, _, tape, _ = fixture()
        with self.assertRaisesRegex(ValueError, "fresh v2 Leadership export"):
            ScoreReplayPanel(tape.drop(columns=["execution_open"]), ["AAA", "BBB"], str(dates[-1]))
        config = fixture()[0].model_copy(update={"strategy_mode": "COMPOUND_ROTATION_SWING_RISK_OFF"})
        with self.assertRaisesRegex(ValueError, "single-position"):
            replay_policy_settings(config)

    def test_cagr_includes_the_first_session_change(self):
        curve = pd.Series([10386.04191780822, 9716.596436722744],
                          index=pd.to_datetime(["2025-09-05", "2026-09-04"], utc=True))
        self.assertAlmostEqual(_cagr(curve, 10000), -.028436281949587937, places=12)
        self.assertAlmostEqual(_cagr(curve), -.06467029662314394, places=12)

    def test_market_snapshot_mismatch_invalidates_the_final_comparison(self):
        _, _, frames, _, _ = fixture()
        hashes = market_data_hashes(frames)
        base = {key: "same" for key in ("initial_capital", "test_start", "test_end", "validation_sessions", "selection_end",
                "training_last_session", "calibration_first_session", "calibration_last_session", "final_fit_last_labeled_session")}
        base.update({key: False for key in ("selection_used_validation_period", "training_used_validation_period", "calibration_used_validation_period")})
        base.update(assets=["AAA", "BBB"], market_data_sha256_by_asset={s: hashes[s] for s in ["AAA", "BBB"]}, code_identity={"source": "same"})
        expanded = {**base, "assets": list(hashes), "market_data_sha256_by_asset": hashes}
        verify_validation_pair(base, expanded)
        altered = {s: f.copy() for s, f in frames.items()}
        altered["AAA"].iloc[-1, altered["AAA"].columns.get_loc("close")] += 1
        with self.assertRaisesRegex(ValueError, "different market prices"):
            verify_validation_pair(base, {**expanded, "market_data_sha256_by_asset": market_data_hashes(altered)})


if __name__ == "__main__":
    unittest.main()
