from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_VERSION = "asset-rotation-leadership-then-backtest-v1.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run intrinsic-timing plus rotation-leadership qualification from scratch, "
            "freeze the resulting universe, then execute exactly one full Strategy Backtest."
        )
    )
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--mongo-uri", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-symbols", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-job-id", default=None)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def _append(command: list[str], flag: str, value: object | None) -> None:
    if value is not None and str(value).strip():
        command.extend([flag, str(value)])


def main() -> int:
    args = _parser().parse_args()
    python = sys.executable
    default_output = (
        PROJECT_ROOT
        / "research_output"
        / f"asset_rotation_leadership_strategy_{args.strategy_sequence}_{args.snapshot_end}"
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output.resolve()

    qualification = [
        python,
        str(PROJECT_ROOT / "scripts" / "research_asset_rotation_leadership.py"),
        "--strategy-sequence",
        str(args.strategy_sequence),
        "--snapshot-end",
        str(args.snapshot_end),
        "--workers",
        str(args.workers),
        "--output-dir",
        str(output_dir),
    ]
    _append(qualification, "--strategy-id", args.strategy_id)
    _append(qualification, "--mongo-uri", args.mongo_uri)
    _append(qualification, "--database", args.database)
    _append(qualification, "--env-file", args.env_file)
    if args.candidate_symbols:
        qualification.append("--candidate-symbols")
        qualification.extend(str(item) for item in args.candidate_symbols)
    if args.no_resume:
        qualification.append("--no-resume")

    print(
        f"=== PHASE 1: INTRINSIC TIMING + ROTATION LEADERSHIP ({SCRIPT_VERSION}) ===",
        flush=True,
    )
    subprocess.run(qualification, check=True, cwd=PROJECT_ROOT)

    if args.qualification_only:
        print(
            "Qualification-only requested. Frozen universe created; full Strategy Backtest not run.",
            flush=True,
        )
        return 0

    frozen = output_dir / "rotation_leadership_snapshot_frozen.json"
    final_backtest = [
        python,
        str(PROJECT_ROOT / "scripts" / "research_asset_timing_final_backtest.py"),
        "--strategy-sequence",
        str(args.strategy_sequence),
        "--snapshot-end",
        str(args.snapshot_end),
        "--frozen-snapshot",
        str(frozen),
        "--output-dir",
        str(output_dir),
    ]
    _append(final_backtest, "--strategy-id", args.strategy_id)
    _append(final_backtest, "--mongo-uri", args.mongo_uri)
    _append(final_backtest, "--database", args.database)
    _append(final_backtest, "--env-file", args.env_file)
    _append(final_backtest, "--baseline-job-id", args.baseline_job_id)

    print(
        "=== PHASE 2: ONE FULL STRATEGY BACKTEST OF THE FROZEN RESEARCH UNIVERSE ===",
        flush=True,
    )
    subprocess.run(final_backtest, check=True, cwd=PROJECT_ROOT)
    print("=== EXPERIMENT COMPLETED ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
