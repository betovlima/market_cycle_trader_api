from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_asset_rotation_independent_then_validate as independent  # noqa: E402

SCRIPT_VERSION = "asset-marginal-rotation-contribution-v1.0.0"
DEFAULT_VALIDATION_SESSIONS = 252


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select marginally complementary assets using only pre-validation data, then compare expanded vs immutable baseline on an untouched final period.")
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _validation_command(*, args: argparse.Namespace, history_start: str, selection_end: str, validation_start: str, validation_end: str, frozen_snapshot: Path, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "research_asset_rotation_independent_validation_v103.py"),
        "--strategy-sequence", str(args.strategy_sequence),
        "--history-start", history_start,
        "--selection-end", selection_end,
        "--validation-start", validation_start,
        "--validation-end", validation_end,
        "--frozen-snapshot", str(frozen_snapshot),
        "--output-dir", str(output_dir),
    ]
    _append(command, "--strategy-id", args.strategy_id)
    _append(command, "--mongo-uri", args.mongo_uri)
    _append(command, "--database", args.database)
    _append(command, "--env-file", args.env_file)
    return command


def _metric(payload: dict[str, Any], name: str) -> float | None:
    try:
        return float(payload.get(name))
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = _parser().parse_args()
    if args.selection_only and args.validation_only:
        raise RuntimeError("Use either --selection-only or --validation-only, not both.")
    history_start = pd.Timestamp(args.history_start).date().isoformat()
    selection_end, validation_start, validation_end = independent._split(history_start, args.snapshot_end, args.validation_sessions)
    universe_file = Path(args.universe_file).resolve() if args.universe_file else independent._default_universe_file(args.strategy_sequence, validation_end)
    candidates = independent._candidate_universe(universe_file)
    root = Path(args.output_dir or PROJECT_ROOT / "research_output" / f"asset_marginal_rotation_strategy_{args.strategy_sequence}_{validation_start}_to_{validation_end}").resolve()
    leadership_dir = root / f"leadership_selection_through_{selection_end}"
    marginal_dir = root / f"marginal_selection_through_{selection_end}"
    baseline_validation_dir = root / f"baseline_validation_{validation_start}_to_{validation_end}"
    expanded_validation_dir = root / f"expanded_validation_{validation_start}_to_{validation_end}"
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "marginal_rotation_independent_design.json", {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "strategy_sequence": int(args.strategy_sequence),
        "history_start": history_start,
        "selection_end": selection_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_sessions": int(args.validation_sessions),
        "source_candidate_universe_file": str(universe_file),
        "source_candidate_count": len(candidates),
        "selection_uses_full_strategy_backtest": False,
        "selection_uses_validation_period": False,
        "baseline_universe_is_immutable": True,
        "final_primary_comparison": "expanded Strategy ending capital versus immutable baseline Strategy ending capital on the same untouched final period",
    })
    expanded_snapshot = marginal_dir / "marginal_rotation_contribution_snapshot_frozen.json"
    baseline_snapshot = marginal_dir / "marginal_rotation_baseline_snapshot_frozen.json"
    if not args.validation_only:
        leadership = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "research_asset_rotation_leadership_v13.py"),
            "--strategy-sequence", str(args.strategy_sequence),
            "--snapshot-end", selection_end,
            "--workers", str(args.workers),
            "--output-dir", str(leadership_dir),
            "--candidate-symbols", *candidates,
        ]
        _append(leadership, "--strategy-id", args.strategy_id)
        _append(leadership, "--mongo-uri", args.mongo_uri)
        _append(leadership, "--database", args.database)
        _append(leadership, "--env-file", args.env_file)
        if args.no_resume:
            leadership.append("--no-resume")
        print("=== PHASE 1A: PRE-VALIDATION LEADERSHIP ===", flush=True)
        subprocess.run(leadership, check=True, cwd=PROJECT_ROOT)
        print("=== PHASE 1B: GREEDY MARGINAL ROTATION CONTRIBUTION ===", flush=True)
        subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "research_asset_marginal_rotation_contribution.py"),
            "--leadership-output-dir", str(leadership_dir),
            "--output-dir", str(marginal_dir),
        ], check=True, cwd=PROJECT_ROOT)
    if not expanded_snapshot.exists() or not baseline_snapshot.exists():
        raise RuntimeError("Marginal selection snapshots are missing. Run phase 1 first or remove --validation-only.")
    if args.selection_only:
        print(f"Expanded snapshot: {expanded_snapshot}", flush=True)
        print(f"Baseline snapshot: {baseline_snapshot}", flush=True)
        return 0
    print("=== PHASE 2A: UNTOUCHED BASELINE VALIDATION ===", flush=True)
    subprocess.run(_validation_command(args=args, history_start=history_start, selection_end=selection_end, validation_start=validation_start, validation_end=validation_end, frozen_snapshot=baseline_snapshot, output_dir=baseline_validation_dir), check=True, cwd=PROJECT_ROOT)
    print("=== PHASE 2B: UNTOUCHED EXPANDED-UNIVERSE VALIDATION ===", flush=True)
    subprocess.run(_validation_command(args=args, history_start=history_start, selection_end=selection_end, validation_start=validation_start, validation_end=validation_end, frozen_snapshot=expanded_snapshot, output_dir=expanded_validation_dir), check=True, cwd=PROJECT_ROOT)
    baseline_result = json.loads((baseline_validation_dir / "independent_validation_result.json").read_text(encoding="utf-8"))
    expanded_result = json.loads((expanded_validation_dir / "independent_validation_result.json").read_text(encoding="utf-8"))
    baseline_capital = _metric(baseline_result, "strategy_ending_capital")
    expanded_capital = _metric(expanded_result, "strategy_ending_capital")
    if baseline_capital is None or expanded_capital is None:
        raise RuntimeError("Independent validation results are missing strategy_ending_capital.")
    ratio = expanded_capital / baseline_capital - 1.0 if baseline_capital > 0 else None
    comparison = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "baseline_ending_capital": baseline_capital,
        "expanded_ending_capital": expanded_capital,
        "capital_delta": expanded_capital - baseline_capital,
        "capital_delta_ratio": ratio,
        "baseline_cagr": _metric(baseline_result, "strategy_cagr"),
        "expanded_cagr": _metric(expanded_result, "strategy_cagr"),
        "baseline_sharpe": _metric(baseline_result, "strategy_sharpe"),
        "expanded_sharpe": _metric(expanded_result, "strategy_sharpe"),
        "baseline_maximum_drawdown": _metric(baseline_result, "strategy_maximum_drawdown"),
        "expanded_maximum_drawdown": _metric(expanded_result, "strategy_maximum_drawdown"),
        "baseline_rotations": _metric(baseline_result, "capital_rotations"),
        "expanded_rotations": _metric(expanded_result, "capital_rotations"),
        "marginal_expansion_beat_baseline": bool(expanded_capital > baseline_capital),
        "selection_used_validation_period": False,
    }
    _write_json(root / "marginal_rotation_independent_comparison.json", comparison)
    print("\n=== MARGINAL ROTATION INDEPENDENT RESULT ===", flush=True)
    print(f"Baseline final capital: ${baseline_capital:,.2f}", flush=True)
    print(f"Expanded final capital: ${expanded_capital:,.2f}", flush=True)
    if ratio is not None:
        print(f"Expanded vs baseline: {ratio * 100:+.2f}%", flush=True)
    print("RESULT: PASS - marginally selected assets improved the untouched-period baseline." if expanded_capital > baseline_capital else "RESULT: FAIL - marginally selected assets did not improve the untouched-period baseline.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
