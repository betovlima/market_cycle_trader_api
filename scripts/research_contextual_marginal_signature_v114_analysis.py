from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.14"
EXPERIMENT_NAME = "contextual_marginal_signature_direct_effect_prevalence"
TOLERANCE = 1e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the prevalence and diversity of direct marginal effects from a "
            "frozen-switch-margin v1.0.14 campaign before fitting any path-signature model."
        )
    )
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _fresh(path: Path) -> None:
    resolved = path.resolve()
    if "research_output" not in resolved.parts or not resolved.name.startswith("contextual_marginal_signature_"):
        raise RuntimeError(f"Refusing to delete unexpected output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> int:
    args = _parser().parse_args()
    frozen_dir = Path(args.frozen_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.fresh_run:
        _fresh(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_path = frozen_dir / "trace_aggregate_dataset.csv"
    manifest_path = frozen_dir / "trace_manifest.json"
    if not aggregate_path.exists() or not manifest_path.exists():
        raise RuntimeError("Frozen campaign must contain trace_aggregate_dataset.csv and trace_manifest.json.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(aggregate_path)
    required = {"decision_date", "universe_name", "candidate", "delta_log_capital"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError("Aggregate dataset is missing columns: " + ", ".join(missing))

    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise")
    frame["candidate"] = frame["candidate"].astype(str).str.upper()
    frame["delta_log_capital"] = pd.to_numeric(frame["delta_log_capital"], errors="coerce")
    if frame["delta_log_capital"].isna().any():
        raise RuntimeError("Aggregate dataset contains non-finite delta_log_capital values.")
    frame["direct_nonzero"] = frame["delta_log_capital"].abs() > TOLERANCE
    frame["direct_positive"] = frame["delta_log_capital"] > TOLERANCE
    frame["direct_negative"] = frame["delta_log_capital"] < -TOLERANCE
    frame["period"] = frame["decision_date"].dt.year.map(lambda year: "validation_2026" if int(year) >= 2026 else "development_pre2026")

    candidate_rows: list[dict[str, Any]] = []
    for candidate, group in frame.groupby("candidate", sort=True):
        nonzero = group[group["direct_nonzero"]]
        candidate_rows.append({
            "candidate": candidate,
            "rows": int(len(group)),
            "nonzero_rows": int(len(nonzero)),
            "positive_rows": int(group["direct_positive"].sum()),
            "negative_rows": int(group["direct_negative"].sum()),
            "nonzero_rate": float(len(nonzero) / len(group)) if len(group) else 0.0,
            "mean_direct_log": float(group["delta_log_capital"].mean()),
            "mean_abs_direct_log": float(group["delta_log_capital"].abs().mean()),
            "nonzero_dates": int(nonzero["decision_date"].nunique()),
            "nonzero_universes": int(nonzero["universe_name"].nunique()),
        })
    candidate_summary = pd.DataFrame(candidate_rows)
    candidate_summary.to_csv(output_dir / "direct_effect_candidate_summary.csv", index=False)

    episode = (
        frame.groupby(["decision_date", "candidate"], as_index=False)
        .agg(
            universe_rows=("universe_name", "size"),
            nonzero_universe_rows=("direct_nonzero", "sum"),
            positive_universe_rows=("direct_positive", "sum"),
            negative_universe_rows=("direct_negative", "sum"),
            max_abs_direct_log=("delta_log_capital", lambda values: float(pd.Series(values).abs().max())),
        )
        .sort_values(["decision_date", "candidate"])
    )
    episode.to_csv(output_dir / "direct_effect_episode_summary.csv", index=False)

    nonzero = frame[frame["direct_nonzero"]]
    period_rows: list[dict[str, Any]] = []
    for period, group in frame.groupby("period", sort=True):
        nz = group[group["direct_nonzero"]]
        period_rows.append({
            "period": period,
            "rows": int(len(group)),
            "nonzero_rows": int(len(nz)),
            "nonzero_candidates": int(nz["candidate"].nunique()),
            "nonzero_dates": int(nz["decision_date"].nunique()),
            "nonzero_universes": int(nz["universe_name"].nunique()),
        })

    enough_rows = len(nonzero) >= 12
    enough_candidates = nonzero["candidate"].nunique() >= 3
    enough_dates = nonzero["decision_date"].nunique() >= 3
    has_validation_effect = bool((nonzero["decision_date"].dt.year >= 2026).any()) if not nonzero.empty else False
    proceed = bool(enough_rows and enough_candidates and enough_dates and has_validation_effect)

    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "source_script_version": manifest.get("script_version"),
        "market_snapshot_hash": manifest.get("market_snapshot_hash"),
        "strategy_configuration_hash": manifest.get("strategy_configuration_hash"),
        "rows": int(len(frame)),
        "dates": int(frame["decision_date"].nunique()),
        "candidates": int(frame["candidate"].nunique()),
        "universes": int(frame["universe_name"].nunique()),
        "nonzero_direct_rows": int(len(nonzero)),
        "positive_direct_rows": int(frame["direct_positive"].sum()),
        "negative_direct_rows": int(frame["direct_negative"].sum()),
        "nonzero_direct_rate": float(len(nonzero) / len(frame)) if len(frame) else 0.0,
        "nonzero_direct_candidates": int(nonzero["candidate"].nunique()),
        "nonzero_direct_dates": int(nonzero["decision_date"].nunique()),
        "nonzero_direct_universes": int(nonzero["universe_name"].nunique()),
        "period_summary": period_rows,
        "candidate_summary": candidate_rows,
        "predictive_signature_dataset_ready": proceed,
        "readiness_checks": {
            "at_least_12_nonzero_rows": bool(enough_rows),
            "at_least_3_nonzero_candidates": bool(enough_candidates),
            "at_least_3_nonzero_dates": bool(enough_dates),
            "at_least_one_2026_nonzero_effect": bool(has_validation_effect),
        },
        "decision_rule": (
            "Proceed to chronological M0 versus M0+LogSig2 predictive testing only if direct effects are "
            "not confined to one candidate/date: at least 12 nonzero rows, at least 3 candidates, at least "
            "3 decision dates, and at least one nonzero 2026 effect. Otherwise expand the frozen-margin "
            "dataset before fitting a predictive model."
        ),
    }
    _write_json(output_dir / "direct_effect_prevalence_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
