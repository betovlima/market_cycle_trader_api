"""Research-only TCC cutoff and independent Model Tuning selection regressions."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from market_cycle_trader_api.engine import tcc_v106_reference_backtest as reference
from market_cycle_trader_api.services import model_tuning, strategy_lab


class TCCExecutionCutoffTests(unittest.TestCase):
    def _request(self, source: str) -> SimpleNamespace:
        return SimpleNamespace(
            tcc_reference_input_source=source,
            analysis_end_date="2026-09-28",
            end_date=None,
        )

    def test_current_source_extends_both_variants_without_mutating_frozen_engine(self):
        scientific_end = reference.TCC_V106_CONFIG.analysis_end_date
        control, soft = reference._tcc_variant_configs({}, self._request("mct_current"))
        self.assertEqual(control.analysis_end_date, "2026-09-28")
        self.assertEqual(soft.analysis_end_date, "2026-09-28")
        self.assertEqual(reference.TCC_V106_CONFIG.analysis_end_date, scientific_end)

    def test_explicit_frozen_source_preserves_scientific_window(self):
        control, soft = reference._tcc_variant_configs({}, self._request("tcc_frozen_main"))
        self.assertEqual(control.analysis_end_date, "2026-09-17")
        self.assertEqual(soft.analysis_end_date, "2026-09-17")

    def test_completed_current_job_must_reach_requested_execution_session(self):
        def result(end: str):
            return SimpleNamespace(predictions=pd.DataFrame(
                {"strategy_equity": [1.0]},
                index=pd.to_datetime([end], utc=True),
            ))
        self.assertEqual(
            reference._verify_reference_execution_cutoff(
                result("2026-09-28T04:00:00Z"),
                expected_date="2026-09-28", variant="CONTROL",
            ),
            "2026-09-28",
        )
        with self.assertRaisesRegex(RuntimeError, "stale scientific execution dates"):
            reference._verify_reference_execution_cutoff(
                result("2026-09-16T04:00:00Z"),
                expected_date="2026-09-28", variant="SOFT",
            )


class SeparateTuningSelectionTests(unittest.TestCase):
    def test_existing_tuning_selection_is_not_synchronized_on_catalog_read(self):
        control = {
            "_id": "strategy-control",
            "revision": 4,
            "research_strategy_id": "reference-28",
            "model_tuning_strategy_id": "ordinary-11",
        }
        db = MagicMock()
        self.assertIs(
            strategy_lab._normalize_model_tuning_selection(db, control),
            control,
        )
        db.__getitem__.assert_not_called()

    def test_tuning_reads_tuning_selection_not_reference_research(self):
        with (
            patch.object(model_tuning, "get_strategy_control", return_value={
                "research_strategy_id": "reference-28",
                "model_tuning_strategy_id": "ordinary-11",
            }),
            patch.object(model_tuning, "get_strategy", return_value={"id": "ordinary-11"}) as get_strategy,
            patch.object(model_tuning, "get_strategy_model_snapshot", return_value={"family": "lightgbm_utility"}) as get_model,
        ):
            strategy, _, source = model_tuning._tuning_target_strategy(MagicMock())
        self.assertEqual(strategy["id"], "ordinary-11")
        self.assertEqual(source, "model_tuning_selection")
        get_strategy.assert_called_once_with(unittest.mock.ANY, "ordinary-11")
        get_model.assert_called_once_with(unittest.mock.ANY, "ordinary-11")

    def test_reference_research_selection_preserves_existing_tuning_and_winner(self):
        db = MagicMock()
        original_control = {"revision": 4}
        db[strategy_lab.STRATEGY_PROFILES_COLLECTION].find_one.return_value = {
            "_id": "reference-28",
            "backtest_engine_binding": strategy_lab.TCC_V106_BACKTEST_ENGINE_BINDING,
        }
        db[strategy_lab.STRATEGY_CONTROL_COLLECTION].find_one_and_update.return_value = {
            "revision": 5,
        }
        with (
            patch.object(strategy_lab, "_assert_no_active_backtest"),
            patch.object(strategy_lab, "ensure_strategy_catalog", return_value=original_control),
            patch.object(strategy_lab, "_normalize_catalog_roles"),
            patch.object(strategy_lab, "_control_response", return_value={"revision": 5}),
        ):
            strategy_lab.select_research_strategy(
                db, "reference-28",
                expected_control_revision=4,
                note="Research reference only",
                actor_email="admin@example.com",
            )
        operation = db[strategy_lab.STRATEGY_CONTROL_COLLECTION].find_one_and_update.call_args.args[1]
        changes = operation["$set"]
        self.assertEqual(changes["research_strategy_id"], "reference-28")
        self.assertNotIn("model_tuning_strategy_id", changes)
        self.assertNotIn("trader_winner_strategy_id", changes)

    def test_selecting_tuning_does_not_change_research_or_winner(self):
        db = MagicMock()
        db[strategy_lab.STRATEGY_PROFILES_COLLECTION].find_one.return_value = {
            "_id": "ordinary-11",
            "strategy_kind": "standard",
            "backtest_engine_binding": None,
        }
        db[strategy_lab.STRATEGY_CONTROL_COLLECTION].find_one_and_update.return_value = {
            "revision": 7,
        }
        with (
            patch.object(strategy_lab, "_assert_no_active_backtest"),
            patch.object(strategy_lab, "ensure_strategy_catalog", return_value={"revision": 6}),
            patch.object(strategy_lab, "_resolved_strategy_model_snapshot", return_value={"family": "lightgbm_utility"}),
            patch.object(strategy_lab, "_control_response", return_value={"revision": 7}),
        ):
            strategy_lab.select_model_tuning_strategy(
                db, "ordinary-11", expected_control_revision=6,
                note="Select ordinary tuning strategy", actor_email="admin@example.com",
            )
        changes = db[strategy_lab.STRATEGY_CONTROL_COLLECTION].find_one_and_update.call_args.args[1]["$set"]
        self.assertEqual(changes["model_tuning_strategy_id"], "ordinary-11")
        self.assertNotIn("research_strategy_id", changes)
        self.assertNotIn("trader_winner_strategy_id", changes)

    def test_tcc_reference_cannot_be_selected_for_generic_tuning(self):
        db = MagicMock()
        db[strategy_lab.STRATEGY_PROFILES_COLLECTION].find_one.return_value = {
            "_id": "reference-28",
            "backtest_engine_binding": strategy_lab.TCC_V106_BACKTEST_ENGINE_BINDING,
        }
        with (
            patch.object(strategy_lab, "_assert_no_active_backtest"),
            patch.object(strategy_lab, "ensure_strategy_catalog", return_value={"revision": 6}),
        ):
            with self.assertRaisesRegex(strategy_lab.StrategyLabConflict, "Research/Backtest-only"):
                strategy_lab.select_model_tuning_strategy(
                    db, "reference-28", expected_control_revision=6,
                    note="Attempt generic tuning", actor_email="admin@example.com",
                )


if __name__ == "__main__":
    unittest.main()
