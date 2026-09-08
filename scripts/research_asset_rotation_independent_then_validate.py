from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_VERSION = "asset-rotation-independent-then-validate-v1.0.1"
DEFAULT_VALIDATION_SESSIONS = 252


def _log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{stamp}] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the candidate universe, select rotation-useful assets using only the past, "
            "then validate intelligent rotation versus equal-weight Buy & Hold on an untouched "
            "final chronological period."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--validation-sessions", type=int, default=DEFAULT_VALIDATION_SESSIONS)
    parser.add_argument("--universe-file", default=None)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    return parser


def _append(command: list[str], flag: str, value: object | None) -> None:
    if value is not None and str(value).strip():
        command.extend([flag, str(value)])


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _split(snapshot_end: str, validation_sessions: int) -> tuple[str, str, str]:
    count = int(validation_sessions)
    if count < 60:
        raise RuntimeError("Independent final validation requires at least 60 market sessions.")
    calendar = xcals.get_calendar("XNYS")
    snapshot = pd.Timestamp(snapshot_end).normalize()
    end_session = pd.Timestamp(calendar.date_to_session(snapshot, direction="previous"))
    first = pd.Timestamp(calendar.date_to_session("2016-01-01", direction="next"))
    sessions = pd.DatetimeIndex(calendar.sessions_in_range(first, end_session))
    if len(sessions) <= count + 700:
        raise RuntimeError(
            f"Not enough Strategy history to reserve {count} final validation sessions."
        )
    validation_start = pd.Timestamp(sessions[-count])
    selection_end = pd.Timestamp(sessions[-count - 1])
    return (
        selection_end.date().isoformat(),
        validation_start.date().isoformat(),
        end_session.date().isoformat(),
    )


def _default_universe_file(strategy_sequence: int, snapshot_end: str) -> Path:
    return (
        PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_leadership_strategy_{strategy_sequence}_{snapshot_end}"
        / "asset_history_integrity.csv"
    ).resolve()


def _candidate_universe(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(
            "Frozen candidate-universe file not found: "
            f"{path}. Run the completed rotation-leadership study first or pass --universe-file."
        )
    frame = pd.read_csv(path, usecols=lambda name: name in {"symbol", "source", "history_complete"})
    required = {"symbol", "source", "history_complete"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Universe file must contain symbol, source, and history_complete. "
            "Qualification results are intentionally not read."
        )
    candidates = frame.loc[
        (frame["source"].astype(str).str.lower() == "candidate")
        & frame["history_complete"].map(_truthy),
        "symbol",
    ]
    values = sorted(
        {
            str(item).strip().upper()
            for item in candidates
            if str(item).strip()
        }
    )
    if not values:
        raise RuntimeError("Frozen universe file contains no complete external candidates.")
    return values


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    if args.selection_only and args.validation_only:
        raise RuntimeError("Use either --selection-only or --validation-only, not both.")

    selection_end, validation_start, validation_end = _split(
        args.snapshot_end,
        args.validation_sessions,
    )
    universe_file = (
        Path(args.universe_file).resolve()
        if args.universe_file
        else _default_universe_file(args.strategy_sequence, validation_end)
    )
    candidates = _candidate_universe(universe_file)

    root = Path(
        args.output_dir
        or PROJECT_ROOT
        / "research_output"
        / (
            f"asset_rotation_independent_strategy_{args.strategy_sequence}_"
            f"{validation_start}_to_{validation_end}"
        )
    ).resolve()
    selection_dir = root / f"selection_through_{selection_end}"
    validation_dir = root / f"independent_validation_{validation_start}_to_{validation_end}"
    root.mkdir(parents=True, exist_ok=True)

    universe_record = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "strategy_sequence": int(args.strategy_sequence),
        "source_file": str(universe_file),
        "source_fields_used": ["symbol", "source", "history_complete"],
        "prior_qualification_columns_used": False,
        "external_candidate_count": len(candidates),
        "external_candidates": candidates,
        "external_candidates_sha256": _sha256_json(candidates),
        "selection_end": selection_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_sessions": int(args.validation_sessions),
        "benchmark": "equal_weight_buy_and_hold_same_frozen_universe",
        "scientific_question": (
            "Using only information available before the final validation period, does causal intelligent "
            "rotation across the frozen assets beat buying and holding the same frozen universe?"
        ),
    }
    _write_json(root / "independent_validation_design.json", universe_record)

    _log("INDEPENDENT FINAL-PERIOD TEST")
    _log(
        f"Candidate universe frozen from history-integrity only: external={len(candidates)}; "
        "previous qualification results are NOT read."
    )
    _log(
        f"Selection data ends at {selection_end}; untouched final validation is "
        f"{validation_start} -> {validation_end} ({args.validation_sessions} XNYS sessions)."
    )

    python = sys.executable
    frozen = selection_dir / "rotation_leadership_snapshot_frozen.json"

    if not args.validation_only:
        selection = [
            python,
            str(PROJECT_ROOT / "scripts" / "research_asset_rotation_leadership_v13.py"),
            "--strategy-sequence",
            str(args.strategy_sequence),
            "--snapshot-end",
            selection_end,
            "--workers",
            str(args.workers),
            "--output-dir",
            str(selection_dir),
            "--candidate-symbols",
            *candidates,
        ]
        _append(selection, "--strategy-id", args.strategy_id)
        _append(selection, "--mongo-uri", args.mongo_uri)
        _append(selection, "--database", args.database)
        _append(selection, "--env-file", args.env_file)
        if args.no_resume:
            selection.append("--no-resume")

        print(
            "=== PHASE 1: SELECT ASSETS USING ONLY PRE-VALIDATION HISTORY ===",
            flush=True,
        )
        subprocess.run(selection, check=True, cwd=PROJECT_ROOT)

    if args.selection_only:
        _log(f"Selection frozen at: {frozen}")
        return 0

    if not frozen.exists():
        raise RuntimeError(
            f"Frozen pre-validation selection not found: {frozen}. "
            "Run without --validation-only first."
        )

    validation = [
        python,
        str(PROJECT_ROOT / "scripts" / "research_asset_rotation_independent_validation_v101.py"),
        "--strategy-sequence",
        str(args.strategy_sequence),
        "--selection-end",
        selection_end,
        "--validation-start",
        validation_start,
        "--validation-end",
        validation_end,
        "--frozen-snapshot",
        str(frozen),
        "--output-dir",
        str(validation_dir),
    ]
    _append(validation, "--strategy-id", args.strategy_id)
    _append(validation, "--mongo-uri", args.mongo_uri)
    _append(validation, "--database", args.database)
    _append(validation, "--env-file", args.env_file)

    print(
        "=== PHASE 2: UNTOUCHED FINAL PERIOD — INTELLIGENT ROTATION VS BUY & HOLD ===",
        flush=True,
    )
    subprocess.run(validation, check=True, cwd=PROJECT_ROOT)
    print("=== INDEPENDENT FINAL-PERIOD TEST COMPLETED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
