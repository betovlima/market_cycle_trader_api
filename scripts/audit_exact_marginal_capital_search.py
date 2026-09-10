"""Inspect exported exact-search artifacts and recalculate Top-K without a replay.

Uses only Python's standard library. It deliberately loads the pure reporting
module by path: importing services/__init__.py installs engine integrations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = (
    "exact_search_manifest.json",
    "baseline_history_integrity.csv",
    "exact_baseline.json",
    "exact_candidate_evaluations.csv",
    "preselector_recall.csv",
    "exact_search_summary.json",
)


def _reporting_module() -> Any:
    path = PROJECT_ROOT / "src/market_cycle_trader_api/services/exact_marginal_capital_search.py"
    spec = importlib.util.spec_from_file_location("exact_search_reporting_offline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reporting module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_export(source: Path) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}

    def add(name: str, data: bytes) -> None:
        if name in contents:
            raise ValueError(f"Multiple {name} files: provide exactly one execution.")
        contents[name] = data

    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name in EXPECTED_FILES:
                add(path.name, path.read_bytes())
    else:
        with ZipFile(source) as archive:
            for item in archive.infolist():
                name = Path(item.filename.replace("\\", "/")).name
                if not item.is_dir() and name in EXPECTED_FILES:
                    add(name, archive.read(item))
    if "exact_search_manifest.json" not in contents:
        raise ValueError("Missing exact_search_manifest.json; cannot identify the execution.")
    return contents


def audit_export(source: Path, cutoffs: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = source.resolve()
    contents = _read_export(source)
    manifest = json.loads(contents["exact_search_manifest.json"].decode("utf-8-sig"))
    missing = [name for name in EXPECTED_FILES if name not in contents]
    reporting = _reporting_module()
    rows: list[dict[str, Any]] = []
    if "exact_candidate_evaluations.csv" in contents:
        reader = csv.DictReader(io.StringIO(contents["exact_candidate_evaluations.csv"].decode("utf-8-sig")))
        required = {"symbol", "evaluation_status", "preselector_rank"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Candidate CSV is missing symbol, evaluation_status or preselector_rank.")
        rows = list(reader)

    issues: list[str] = []
    symbols = [str(row.get("symbol") or "").strip().upper() for row in rows]
    if len(set(symbols)) != len(symbols) or any(not symbol for symbol in symbols):
        issues.append("Candidate symbols are empty or duplicated.")
    expected_count = int(manifest["candidate_count"])
    terminal_statuses = {"completed", "history_rejected", "context_rejected", "failed"}
    terminal_count = sum(str(row.get("evaluation_status") or "").strip().lower() in terminal_statuses for row in rows)
    if len(rows) > expected_count:
        issues.append("More candidate rows than declared in the manifest.")
    for row in rows:
        if str(row.get("evaluation_status") or "").strip().lower() == "completed":
            if reporting.finite_float(row.get("ending_capital_delta_rate")) is None:
                issues.append(f"Completed candidate {row.get('symbol')} has no finite capital delta.")

    baseline = None
    if "exact_baseline.json" in contents:
        baseline = json.loads(contents["exact_baseline.json"].decode("utf-8-sig"))
    complete = not missing and terminal_count == expected_count and len(rows) == expected_count
    status = "invalid_export" if issues else "complete_export" if complete else "incomplete_export"
    if "exact_baseline.json" not in contents:
        evidence = (
            "No exported baseline result. The manifest and history-integrity table are written "
            "before ranker training and the baseline replay. The files cannot distinguish a "
            "running/interrupted execution from a partial ZIP; they do not identify an error."
        )
    elif terminal_count < expected_count:
        evidence = "A baseline result exists, but the export does not contain every declared candidate outcome."
    else:
        evidence = "Candidate outcomes are present; check missing_files and validation_issues for export completeness."

    recall = reporting.build_preselector_recall(rows, cutoffs=cutoffs) if rows and not issues else []
    audit = {
        "audit_version": "exact-marginal-capital-search-report-v1.0.4",
        "status": status,
        "source": str(source),
        "source_zip_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
        "source_files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())},
        "source_script_version": manifest.get("script_version"),
        "source_api_version": manifest.get("api_version"),
        "strategy_configuration_hash": manifest.get("strategy_configuration_hash"),
        "history_start": manifest.get("history_start"),
        "snapshot_end": manifest.get("snapshot_end"),
        "declared_candidate_count": expected_count,
        "exported_candidate_rows": len(rows),
        "terminal_candidate_rows": terminal_count,
        "missing_files": missing,
        "validation_issues": issues,
        "execution_evidence": evidence,
        "baseline": baseline,
        "candidate_summary": reporting.summarize_exact_search(rows) if rows and not issues else None,
        "preselector_recall": recall,
        "recall_population": "exported finite completed economic outcomes; pending/unexported outcomes are unknown",
        "recall_cutoff_basis": "original preselector ranks, including rejected/failed ranked candidates",
        "positive_recall_when_no_positives": None,
        "capital_results_recomputed": False,
        "source_files_modified": False,
        "market_data_or_database_accessed": False,
    }
    return audit, recall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="One exported ZIP or execution directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty audit directory.")
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[3, 5, 10, 20, 50, 100])
    args = parser.parse_args()
    if any(value <= 0 for value in args.cutoffs):
        parser.error("--cutoffs values must be positive integers.")
    try:
        audit, recall = audit_export(args.input, args.cutoffs)
        output = args.output_dir.resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("--output-dir must be new or empty; existing outputs are preserved.")
        output.mkdir(parents=True, exist_ok=True)
        (output / "exact_search_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        if recall:
            with (output / "preselector_recall_corrected.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(recall[0]))
                writer.writeheader()
                writer.writerows(recall)
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        parser.exit(1, f"Audit failed: {exc}\n")
    print(f"{audit['status']}: {audit['terminal_candidate_rows']}/{audit['declared_candidate_count']} candidate outcomes exported.")
    if audit["missing_files"]:
        print("Missing files: " + ", ".join(audit["missing_files"]))
    print(f"Audit written to {output}; original exports and capital results preserved.")
    return {"complete_export": 0, "incomplete_export": 2, "invalid_export": 1}[audit["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
