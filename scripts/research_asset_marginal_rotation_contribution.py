from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from market_cycle_trader_api.services.asset_marginal_rotation_contribution import greedy_marginal_rotation_selection  # noqa: E402

SCRIPT_VERSION = "asset-marginal-rotation-contribution-v1.0.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select external assets by positive OOS marginal rotation contribution without a full backtest per candidate.")
    parser.add_argument("--leadership-output-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _candidate_pool(qualification: pd.DataFrame, baseline: set[str]) -> list[str]:
    if "symbol" not in qualification.columns:
        raise RuntimeError("asset_rotation_qualification.csv is missing symbol.")
    frame = qualification.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if "qualified" in frame.columns:
        accepted = _truthy(frame["qualified"])
    elif "leadership_qualified" in frame.columns:
        accepted = _truthy(frame["leadership_qualified"])
    else:
        raise RuntimeError("asset_rotation_qualification.csv needs qualified or leadership_qualified.")
    return sorted({symbol for symbol in frame.loc[accepted, "symbol"].tolist() if symbol and symbol not in baseline})


def _snapshot(source: dict[str, Any], baseline: list[str], selected: list[str], pool: list[str], steps_hash: str, *, control: bool) -> dict[str, Any]:
    final_assets = list(baseline) if control else [*baseline, *selected]
    frozen = dict(source)
    frozen.pop("decision_snapshot_sha256", None)
    frozen.update({
        "schema_version": 6,
        "script_version": SCRIPT_VERSION,
        "selection_rule": (
            "immutable Strategy baseline control; no external asset added"
            if control
            else "immutable Strategy baseline plus greedy positive OOS marginal rotation contribution at predicted leadership-displacement event starts"
        ),
        "baseline_universe_immutable": True,
        "full_strategy_backtest_used_for_selection": False,
        "intrinsic_timing_used_for_selection": False,
        "same_asset_buy_hold_used_for_selection": False,
        "rotation_opportunity_used_for_selection": True,
        "marginal_rotation_contribution_used_for_selection": not control,
        "manual_candidate_acceptance_margin": False,
        "original_assets": list(baseline),
        "qualified_assets": final_assets,
        "retained_existing_assets": list(baseline),
        "removed_existing_assets": [],
        "added_candidate_assets": [] if control else list(selected),
        "marginal_candidate_pool": [] if control else list(pool),
        "marginal_rejected_candidates": [] if control else [s for s in pool if s not in set(selected)],
        "selection_steps_sha256": None if control else steps_hash,
        "qualified_assets_sha256": _sha256_json(final_assets),
    })
    frozen["decision_snapshot_sha256"] = _sha256_json(frozen)
    return frozen


def main() -> int:
    args = _parser().parse_args()
    leadership_dir = Path(args.leadership_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else leadership_dir / "marginal_rotation_contribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = leadership_dir / "rotation_leadership_snapshot_frozen.json"
    qualification_path = leadership_dir / "asset_rotation_qualification.csv"
    predictions_path = leadership_dir / "leadership_predictions_raw.csv"
    for path in (snapshot_path, qualification_path, predictions_path):
        if not path.exists():
            raise RuntimeError(f"Required leadership artifact not found: {path}")
    source = json.loads(snapshot_path.read_text(encoding="utf-8"))
    baseline = list(dict.fromkeys(str(x).strip().upper() for x in source.get("original_assets") or [] if str(x).strip()))
    if len(baseline) < 2:
        raise RuntimeError("Leadership snapshot does not contain a valid original baseline universe.")
    qualification = pd.read_csv(qualification_path)
    candidates = _candidate_pool(qualification, set(baseline))
    result = greedy_marginal_rotation_selection(
        predictions=pd.read_csv(predictions_path), baseline_assets=baseline, candidate_assets=candidates
    )
    _write_csv(output_dir / "marginal_rotation_selection_steps.csv", result.selected_steps)
    _write_csv(output_dir / "marginal_rotation_candidate_evaluations.csv", result.all_evaluations)
    _write_csv(output_dir / "marginal_rotation_selected_events.csv", result.selected_events)
    step_records = result.selected_steps.replace({np.nan: None}).to_dict(orient="records")
    steps_hash = _sha256_json(step_records)
    expanded = _snapshot(source, result.baseline_assets, result.selected_candidates, candidates, steps_hash, control=False)
    control = _snapshot(source, result.baseline_assets, result.selected_candidates, candidates, steps_hash, control=True)
    expanded_path = output_dir / "marginal_rotation_contribution_snapshot_frozen.json"
    control_path = output_dir / "marginal_rotation_baseline_snapshot_frozen.json"
    _write_json(expanded_path, expanded)
    _write_json(control_path, control)
    _write_json(output_dir / "marginal_rotation_contribution_summary.json", {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "baseline_asset_count": len(result.baseline_assets),
        "candidate_pool_count": len(candidates),
        "selected_candidate_count": len(result.selected_candidates),
        "final_asset_count": len(result.final_assets),
        "selected_candidates": result.selected_candidates,
        "final_assets": result.final_assets,
        "selection_steps_sha256": steps_hash,
        "full_backtest_per_candidate": False,
        "manual_acceptance_threshold": False,
        "economic_indifference_point": 0.0,
        "expanded_snapshot": str(expanded_path),
        "baseline_snapshot": str(control_path),
    })
    print("\n=== MARGINAL ROTATION CONTRIBUTION V1 ===", flush=True)
    print(f"Baseline assets: {len(result.baseline_assets)}", flush=True)
    print(f"Leadership-qualified candidates: {len(candidates)}", flush=True)
    print(f"Selected candidates: {len(result.selected_candidates)}", flush=True)
    for row in result.selected_steps.to_dict(orient="records"):
        print(f"step={int(row['step'])} add={row['candidate']} events={int(row['displacement_event_count'])} marginal_log_sum={float(row['marginal_forward_net_log_return_sum']):+.6f} universe={int(row['resulting_universe_size'])}", flush=True)
    print(f"Expanded frozen snapshot: {expanded_path}", flush=True)
    print(f"Baseline frozen snapshot: {control_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
