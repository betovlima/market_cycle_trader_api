from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_exact_marginal_capital_search.py"
SPEC = importlib.util.spec_from_file_location("exact_search_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_cli)
reporting = audit_cli._reporting_module()


def candidate_rows(last_delta: float = 0.0) -> list[dict]:
    return reporting.annotate_preselector_ranks([
        {"symbol": "VSAT", "preselector_raw_score": 0.4, "evaluation_status": "completed", "ending_capital_delta_rate": -0.06},
        {"symbol": "RARE", "preselector_raw_score": 0.2, "evaluation_status": "completed", "ending_capital_delta_rate": -0.5},
        {"symbol": "LB", "preselector_raw_score": -0.1, "evaluation_status": "context_rejected"},
        {"symbol": "CCS", "preselector_raw_score": -0.3, "evaluation_status": "completed", "ending_capital_delta_rate": last_delta},
    ])


def write_export(path: Path, *, incomplete: bool = False) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("run/exact_search_manifest.json", json.dumps({"candidate_count": 4, "script_version": "test-v102"}))
        archive.writestr("run/baseline_history_integrity.csv", "symbol,history_complete\nBASE,True\n")
        if incomplete:
            return
        rows = candidate_rows()
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        archive.writestr("run/exact_candidate_evaluations.csv", buffer.getvalue())
        archive.writestr("run/exact_baseline.json", json.dumps({"metrics": {"ending_capital": 10000}}))
        archive.writestr("run/exact_search_summary.json", "{}")
        archive.writestr("run/preselector_recall.csv", "requested_cutoff,effective_cutoff\n100,3\n")


class ExactSearchQueueRegressionTests(unittest.TestCase):
    def test_rejected_rank_does_not_hide_later_positive(self) -> None:
        rows = candidate_rows(last_delta=0.25)
        result = {row["requested_cutoff"]: row for row in reporting.build_preselector_recall(rows, (3, 5))}
        self.assertEqual(result[3]["evaluations_in_top_k"], 2)
        self.assertEqual(result[3]["positive_recall"], 0.0)
        self.assertEqual(result[3]["uncompleted_candidates_in_top_k"], 1)
        self.assertEqual(result[5]["effective_cutoff"], 4)
        self.assertEqual(result[5]["positive_recall"], 1.0)
        self.assertEqual(result[5]["evaluations_in_top_k"], 3)
        self.assertEqual(result[5]["evaluation_fraction_of_ranked_queue"], 1.0)
        self.assertAlmostEqual(result[5]["positive_precision"], 1 / 3)
        self.assertEqual(result[4]["best_ending_capital_delta_rate_in_top_k"], 0.25)

    def test_no_positives_keeps_recall_undefined_and_includes_neutral(self) -> None:
        rows = candidate_rows()
        snapshot = json.dumps(rows, sort_keys=True)
        full = reporting.build_preselector_recall(rows, (100,))[-1]
        self.assertEqual(full["effective_cutoff"], 4)
        self.assertEqual(full["evaluations_in_top_k"], 3)
        self.assertEqual(full["evaluation_fraction_of_completed"], 1.0)
        self.assertEqual(full["best_ending_capital_delta_rate_in_top_k"], 0.0)
        self.assertIsNone(full["positive_recall"])
        self.assertEqual(json.dumps(rows, sort_keys=True), snapshot)

    def test_csv_ranks_preserve_gaps_and_failed_attempts(self) -> None:
        rows = [
            {"preselector_rank": "1", "evaluation_status": "failed"},
            {"preselector_rank": "7.0", "evaluation_status": "completed", "ending_capital_delta_rate": "0.5"},
        ]
        result = {row["requested_cutoff"]: row for row in reporting.build_preselector_recall(rows, (5, 100))}
        self.assertEqual(result[5]["evaluations_in_top_k"], 0)
        self.assertEqual(result[7]["positive_recall"], 1.0)
        self.assertEqual(result[100]["effective_cutoff"], 7)
        self.assertEqual(result[100]["ranked_queue_candidates"], 2)

    def test_unranked_positive_stays_in_denominator_after_rejection(self) -> None:
        rows = candidate_rows(last_delta=0.25)
        rows.append({"preselector_rank": None, "evaluation_status": "completed", "ending_capital_delta_rate": 0.5})
        full = reporting.build_preselector_recall(rows, (100,))[-1]
        self.assertEqual(full["positive_recall"], 0.5)
        self.assertEqual(full["unranked_exact_positive_candidates"], 1)

    def test_empty_export_has_no_economic_recall(self) -> None:
        row = reporting.build_preselector_recall([], (5,))[0]
        self.assertEqual(row["effective_cutoff"], 0)
        self.assertIsNone(row["positive_recall"])
        self.assertIsNone(row["positive_precision"])


class ExactSearchOfflineAuditTests(unittest.TestCase):
    def test_zip_cli_works_without_site_packages_and_preserves_input(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source.zip", root / "audit"
            write_export(source)
            original = source.read_bytes()
            process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--input", str(source), "--output-dir", str(output)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads((output / "exact_search_audit.json").read_text())
            self.assertEqual(result["status"], "complete_export")
            self.assertFalse(result["capital_results_recomputed"])
            self.assertEqual(result["baseline"]["metrics"]["ending_capital"], 10000)
            self.assertEqual(source.read_bytes(), original)
            with (output / "preselector_recall_corrected.csv").open(newline="") as handle:
                full = list(csv.DictReader(handle))[-1]
            self.assertEqual(full["evaluations_in_top_k"], "3")
            self.assertEqual(full["positive_recall"], "")

    def test_initial_artifacts_are_incomplete_not_an_economic_failure(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "source.zip", root / "audit"
            write_export(source, incomplete=True)
            process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--input", str(source), "--output-dir", str(output)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 2, process.stderr)
            result = json.loads((output / "exact_search_audit.json").read_text())
            self.assertEqual(result["status"], "incomplete_export")
            self.assertEqual(len(result["missing_files"]), 4)
            self.assertIsNone(result["baseline"])
            self.assertIsNone(result["candidate_summary"])
            self.assertFalse((output / "preselector_recall_corrected.csv").exists())

    def test_multiple_executions_in_one_zip_are_not_combined(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.zip"
            write_export(source, incomplete=True)
            with ZipFile(source, "a") as archive:
                archive.writestr("another_run/exact_search_manifest.json", '{"candidate_count": 4}')
            with self.assertRaisesRegex(ValueError, "exactly one execution"):
                audit_cli.audit_export(source, [100])

    def test_directory_input_preserves_missing_outcome_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_export(root / "source.zip")
            with ZipFile(root / "source.zip") as archive:
                archive.extractall(root / "export")
            manifest = root / "export/run/exact_search_manifest.json"
            manifest.write_text('{"candidate_count": 5, "script_version": "test"}')
            audit, recall = audit_cli.audit_export(root / "export", [100])
            self.assertEqual(audit["status"], "incomplete_export")
            self.assertEqual(audit["terminal_candidate_rows"], 4)
            self.assertIsNone(recall[-1]["positive_recall"])


if __name__ == "__main__":
    unittest.main()
