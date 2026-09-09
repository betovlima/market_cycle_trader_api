"""Offline audit of v1/v2 validation exports. Does not fit/select any model."""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_windows_file_io as file_io

SCRIPT_VERSION = "asset-marginal-rotation-export-audit-v2.0.0"


def _read(source: Path, suffix: str) -> bytes:
    if source.is_dir():
        matches = [p for p in source.rglob(Path(suffix).name) if p.as_posix().endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one artifact ending in {suffix}; found {len(matches)}.")
        with open(file_io.windows_long_path(matches[0]), "rb") as handle:
            return handle.read()
    with ZipFile(file_io.windows_long_path(source)) as archive:
        matches = [i for i in archive.infolist() if i.filename.endswith(suffix) and not i.is_dir()]
        if len(matches) != 1:
            raise ValueError(f"Expected one ZIP member ending in {suffix}; found {len(matches)}.")
        return archive.read(matches[0])  # Read only; never extract member paths.


def audit_export(source: Path) -> tuple[dict, pd.DataFrame]:
    frames, metrics, results = {}, {}, {}
    for name in ("baseline", "expanded"):
        prefix = f"val_{name}/independent_validation_"
        metric = json.loads(_read(source, prefix + "result.json"))
        frame = pd.read_csv(io.BytesIO(_read(source, prefix + "predictions.csv")))
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, format="mixed", errors="raise")
        frame = frame.sort_values("timestamp").set_index("timestamp")
        if frame.empty or frame.index.has_duplicates or frame.index.isna().any():
            raise ValueError(f"Invalid {name} execution calendar.")
        initial = float(metric["initial_capital"])
        equity = pd.to_numeric(frame.strategy_equity, errors="raise").to_numpy(dtype=float)
        if not math.isfinite(initial) or initial <= 0 or not np.isfinite(equity).all() or (equity <= 0).any():
            raise ValueError("Equity and initial capital must be finite and positive.")
        ending = float(metric["strategy_ending_capital"])
        if not math.isclose(ending, equity[-1], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"{name} final capital disagrees with predictions.")
        for key, actual in (("test_start", frame.index[0]), ("test_end", frame.index[-1])):
            if pd.Timestamp(metric[key]) != actual:
                raise ValueError(f"{name} {key} disagrees with execution dates.")
        years = max((frame.index[-1] - frame.index[0]).days / 365.25, 1 / 365.25)
        if not math.isclose(float(metric["test_calendar_years"]), years, rel_tol=1e-10):
            raise ValueError("Reported elapsed years disagree with execution dates.")
        frame["net_log_return"] = np.diff(np.log(np.r_[initial, equity]))
        frames[name], metrics[name] = frame, metric
        results[name] = {
            "initial_capital": initial, "ending_capital": ending,
            "total_return": ending / initial - 1,
            "reported_cagr": metric.get("strategy_cagr"),
            "cagr_from_initial_capital": math.expm1(math.log(ending / initial) / years),
            "reported_maximum_drawdown": metric.get("strategy_maximum_drawdown"),
            "rotations": metric.get("capital_rotations"),
        }
    if not frames["baseline"].index.equals(frames["expanded"].index):
        raise ValueError("Validation calendars differ; no intersection-only comparison is allowed.")
    if results["baseline"]["initial_capital"] != results["expanded"]["initial_capital"]:
        raise ValueError("Initial capitals differ.")
    if not set(metrics["baseline"]["assets"]).issubset(metrics["expanded"]["assets"]):
        raise ValueError("Expanded universe removed immutable baseline assets.")
    a, b = frames["baseline"], frames["expanded"]
    daily = pd.DataFrame({
        "timestamp": a.index, "baseline_asset": a.selected_asset.to_numpy(),
        "expanded_asset": b.selected_asset.to_numpy(),
        "baseline_equity": a.strategy_equity.to_numpy(), "expanded_equity": b.strategy_equity.to_numpy(),
        "marginal_net_log_return": (b.net_log_return - a.net_log_return).to_numpy(),
    })
    daily["position_changed"] = daily.baseline_asset != daily.expanded_asset
    ending_a, ending_b = results["baseline"]["ending_capital"], results["expanded"]["ending_capital"]
    delta = math.log(ending_b / ending_a)
    if not math.isclose(daily.marginal_net_log_return.sum(), delta, abs_tol=1e-12):
        raise ValueError("Daily contribution failed the final-capital reconciliation.")
    report = {
        "script_version": SCRIPT_VERSION, "audit_only": True,
        "selection_or_training_performed": False,
        "period": {"start": str(a.index[0]), "end": str(a.index[-1]), "sessions": len(a)},
        **results,
        "added_assets": sorted(set(metrics["expanded"]["assets"]) - set(metrics["baseline"]["assets"])),
        "capital_delta": ending_b - ending_a, "capital_delta_ratio": ending_b / ending_a - 1,
        "marginal_net_log_growth": delta,
        "changed_position_sessions": int(daily.position_changed.sum()),
        "identical_top1_sessions": int((a.top_1_asset == b.top_1_asset).sum()),
        "accounting_reconciled": True,
        "attribution_scope": "daily account differences include fees and prior-position effects; they are not standalone asset returns",
        "validation_scope": "retrospective audit of an observed period; no new independent performance claim",
    }
    return report, daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Research ZIP or extracted directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report, daily = audit_export(Path(args.input))
    output = Path(args.output_dir)
    file_io.write_json(output / "marginal_rotation_export_audit.json", report)
    file_io.write_csv(output / "marginal_rotation_daily_account_differences.csv", daily)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
