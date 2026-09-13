from __future__ import annotations

import argparse
import json
import math
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.13"
EXPERIMENT_NAME = "contextual_marginal_signature_logsignature_level2_sanity"
CHANNELS = ("time", "candidate", "universe", "dispersion")
TOLERANCE = 1e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "No-replay sanity test for a four-channel level-2 log-signature representation. "
            "The experiment verifies that candidate x universe path interactions survive "
            "cross-universe differencing before any predictive model is fitted."
        )
    )
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--decomposition", required=True)
    parser.add_argument("--window-sessions", type=int, default=60)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _fresh_output(path: Path) -> None:
    resolved = path.resolve()
    if "research_output" not in resolved.parts or not resolved.name.startswith("contextual_marginal_signature_"):
        raise RuntimeError(f"Refusing to delete unexpected output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _load_manifest(frozen_dir: Path) -> dict[str, Any]:
    path = frozen_dir / "trace_manifest.json"
    if not path.exists():
        raise RuntimeError(f"Missing trace manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Trace manifest must be a JSON object.")
    return payload


def _load_close_panel(frozen_dir: Path) -> pd.DataFrame:
    path = frozen_dir / "market_snapshot.csv.gz"
    if not path.exists():
        raise RuntimeError(f"Missing frozen market snapshot: {path}")
    table = pd.read_csv(path)
    required = {"symbol", "timestamp", "close"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError("Frozen snapshot is missing columns: " + ", ".join(missing))
    table["symbol"] = table["symbol"].astype(str).str.upper()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    table["close"] = pd.to_numeric(table["close"], errors="coerce")
    return table.pivot(index="timestamp", columns="symbol", values="close").sort_index()


def _universe_map(manifest: dict[str, Any]) -> dict[str, list[str]]:
    universes: dict[str, list[str]] = {}
    for item in manifest.get("universes") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        assets = [str(symbol).strip().upper() for symbol in item.get("assets") or [] if str(symbol).strip()]
        if name and assets:
            universes[name] = assets
    if not universes:
        raise RuntimeError("Trace manifest contains no universes.")
    return universes


def _level2_logsignature(path: np.ndarray, channels: tuple[str, ...] = CHANNELS) -> dict[str, float]:
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(channels) or len(values) < 2:
        raise ValueError("Path must have shape (n>=2, number_of_channels).")
    if not np.isfinite(values).all():
        raise ValueError("Path contains non-finite values.")

    features: dict[str, float] = {}
    increments = values[-1] - values[0]
    for index, channel in enumerate(channels):
        features[f"logsig1__{channel}"] = float(increments[index])

    # At level 2, the free-Lie/log-signature coefficient for a channel pair is
    # the signed Levy area 1/2 * integral(x_i dx_j - x_j dx_i).  For a
    # piecewise-linear path this is exactly the polygonal shoelace sum below.
    for left, right in combinations(range(len(channels)), 2):
        area = 0.5 * np.sum(
            values[:-1, left] * values[1:, right]
            - values[:-1, right] * values[1:, left]
        )
        features[f"logsig2__{channels[left]}__{channels[right]}"] = float(area)
    return features


def _joint_path(
    close: pd.DataFrame,
    *,
    candidate: str,
    universe_assets: list[str],
    decision_date: str,
    window_sessions: int,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    decision = pd.Timestamp(decision_date)
    decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
    eligible = close.index[close.index <= decision]
    if len(eligible) < window_sessions:
        raise RuntimeError(
            f"Not enough pre-decision sessions for {decision.date()}: "
            f"need={window_sessions}, available={len(eligible)}"
        )
    sessions = pd.DatetimeIndex(eligible[-window_sessions:])
    candidate = candidate.upper()
    required = [candidate, *universe_assets]
    missing = [symbol for symbol in required if symbol not in close.columns]
    if missing:
        raise RuntimeError(f"Market snapshot is missing symbols: {missing}")

    candidate_close = close.loc[sessions, candidate].astype(float)
    universe_close = close.loc[sessions, universe_assets].astype(float)
    if candidate_close.isna().any() or universe_close.isna().any().any():
        raise RuntimeError(f"Missing close data for {decision.date()}/{candidate}")
    if (candidate_close <= 0).any() or (universe_close <= 0).any().any():
        raise RuntimeError(f"Non-positive close data for {decision.date()}/{candidate}")

    candidate_path = np.log(candidate_close / float(candidate_close.iloc[0]))
    universe_paths = np.log(universe_close.divide(universe_close.iloc[0], axis="columns"))
    universe_path = universe_paths.mean(axis=1)
    dispersion_path = universe_paths.std(axis=1, ddof=0)
    time_path = np.linspace(0.0, 1.0, len(sessions))

    values = np.column_stack(
        [
            time_path,
            candidate_path.to_numpy(dtype=float),
            universe_path.to_numpy(dtype=float),
            dispersion_path.to_numpy(dtype=float),
        ]
    )
    if np.max(np.abs(values[0])) > TOLERANCE:
        raise RuntimeError("Joint path must start at the fixed zero base-point.")
    return values, sessions


def _feature_columns() -> list[str]:
    columns = [f"logsig1__{channel}" for channel in CHANNELS]
    columns.extend(
        f"logsig2__{CHANNELS[left]}__{CHANNELS[right]}"
        for left, right in combinations(range(len(CHANNELS)), 2)
    )
    return columns


def main() -> int:
    args = _parser().parse_args()
    frozen_dir = Path(args.frozen_dir).resolve()
    decomposition_path = Path(args.decomposition).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.fresh_run:
        _fresh_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if int(args.window_sessions) < 2:
        raise RuntimeError("--window-sessions must be at least 2.")
    if not decomposition_path.exists():
        raise RuntimeError(f"Missing frozen-margin decomposition: {decomposition_path}")

    manifest = _load_manifest(frozen_dir)
    universes = _universe_map(manifest)
    close = _load_close_panel(frozen_dir)
    decomposition = pd.read_csv(decomposition_path)
    required_columns = {"decision_date", "universe_name", "candidate", "frozen_marginal_log"}
    missing_columns = sorted(required_columns.difference(decomposition.columns))
    if missing_columns:
        raise RuntimeError("Decomposition is missing columns: " + ", ".join(missing_columns))

    rows: list[dict[str, Any]] = []
    for record in decomposition.to_dict(orient="records"):
        universe_name = str(record["universe_name"])
        if universe_name not in universes:
            raise RuntimeError(f"Unknown universe in decomposition: {universe_name}")
        candidate = str(record["candidate"]).strip().upper()
        assets = universes[universe_name]
        if candidate in assets:
            raise RuntimeError(f"Candidate {candidate} is already in universe {universe_name}.")
        path, sessions = _joint_path(
            close,
            candidate=candidate,
            universe_assets=assets,
            decision_date=str(record["decision_date"]),
            window_sessions=int(args.window_sessions),
        )
        features = _level2_logsignature(path)
        rows.append(
            {
                "decision_date": str(record["decision_date"]),
                "universe_name": universe_name,
                "candidate": candidate,
                "window_start": sessions[0].date().isoformat(),
                "window_end": sessions[-1].date().isoformat(),
                "window_sessions": int(len(sessions)),
                "frozen_marginal_log": float(record["frozen_marginal_log"]),
                **features,
            }
        )

    features = pd.DataFrame(rows)
    feature_columns = _feature_columns()
    features.to_csv(output_dir / "path_logsignature_level2_features.csv", index=False)

    sensitivity_rows: list[dict[str, Any]] = []
    grouped = features.groupby(["decision_date", "candidate"], sort=True)
    for (decision_date, candidate), group in grouped:
        for feature in feature_columns:
            values = pd.to_numeric(group[feature], errors="coerce")
            sensitivity_rows.append(
                {
                    "decision_date": decision_date,
                    "candidate": candidate,
                    "feature": feature,
                    "universe_count": int(len(values)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "range": float(values.max() - values.min()),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "path_logsignature_universe_sensitivity.csv", index=False)

    sensitivity_summary: list[dict[str, Any]] = []
    for feature, group in sensitivity.groupby("feature", sort=False):
        ranges = group["range"].abs()
        sensitivity_summary.append(
            {
                "feature": feature,
                "candidate_date_groups": int(len(group)),
                "nonzero_universe_range_groups": int((ranges > TOLERANCE).sum()),
                "mean_universe_range": float(ranges.mean()),
                "max_universe_range": float(ranges.max()),
            }
        )

    candidate_summary: list[dict[str, Any]] = []
    for candidate, group in features.groupby("candidate", sort=True):
        target = pd.to_numeric(group["frozen_marginal_log"], errors="coerce").fillna(0.0)
        candidate_summary.append(
            {
                "candidate": candidate,
                "rows": int(len(group)),
                "nonzero_frozen_target_rows": int((target.abs() > TOLERANCE).sum()),
                "mean_abs_frozen_target": float(target.abs().mean()),
            }
        )

    by_feature = {row["feature"]: row for row in sensitivity_summary}
    candidate_l1 = by_feature["logsig1__candidate"]
    candidate_universe_area = by_feature["logsig2__candidate__universe"]
    representation_non_cancellation_confirmed = (
        int(candidate_l1["nonzero_universe_range_groups"]) == 0
        and int(candidate_universe_area["nonzero_universe_range_groups"]) > 0
    )

    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "source_script_version": manifest.get("script_version"),
        "market_snapshot_hash": manifest.get("market_snapshot_hash"),
        "strategy_configuration_hash": manifest.get("strategy_configuration_hash"),
        "window_sessions": int(args.window_sessions),
        "channels": list(CHANNELS),
        "feature_count": int(len(feature_columns)),
        "rows": int(len(features)),
        "candidate_date_groups": int(features.groupby(["decision_date", "candidate"]).ngroups),
        "nonzero_frozen_target_rows": int((features["frozen_marginal_log"].abs() > TOLERANCE).sum()),
        "representation_non_cancellation_confirmed": bool(representation_non_cancellation_confirmed),
        "candidate_first_order_invariant_across_universes": bool(
            int(candidate_l1["nonzero_universe_range_groups"]) == 0
        ),
        "candidate_universe_level2_nonzero_groups": int(
            candidate_universe_area["nonzero_universe_range_groups"]
        ),
        "candidate_universe_level2_group_count": int(
            candidate_universe_area["candidate_date_groups"]
        ),
        "candidate_summary": candidate_summary,
        "feature_sensitivity": sensitivity_summary,
        "decision_rule": (
            "This is a representation test, not a predictive validation. Continue to a broader chronological "
            "campaign only if candidate first-order terms remain invariant when U changes while at least one "
            "candidate x universe level-2 term changes. Do not infer predictive power from the current sparse "
            "frozen target."
        ),
    }
    _write_json(output_dir / "path_logsignature_level2_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
