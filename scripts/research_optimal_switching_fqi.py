from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_asset_rotation_universe_scale as universe_scale  # noqa: E402
from market_cycle_trader_api.engine.optimal_switching_research import (  # noqa: E402
    run_optimal_switching_fqi,
)
from market_cycle_trader_api.services.optimal_switching_fqi import (  # noqa: E402
    OptimalSwitchingFQISettings,
)
from market_cycle_trader_api.services.research_universe_baseline import (  # noqa: E402
    FrozenUniverseScaleCandidateLoader,
    resolve_universe_scale_baseline,
)

SCRIPT_VERSION = "optimal-switching-fqi-v1.0.0"
FQI_SETTINGS = OptimalSwitchingFQISettings(
    enabled=True,
    action_top_k=5,
    bellman_iterations=20,
)


def _preparse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--analysis-start", required=True)
    parser.add_argument("--analysis-end", required=True)
    parser.add_argument("--baseline-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--finalize-only", action="store_true")
    values, _ = parser.parse_known_args(argv)
    return values


def _strip_custom(argv: list[str], names: set[str]) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        matched = next(
            (
                name
                for name in names
                if token == name or token.startswith(name + "=")
            ),
            None,
        )
        if matched is None:
            output.append(token)
            index += 1
            continue
        if token == matched and matched not in {"--finalize-only"}:
            index += 2
        else:
            index += 1
    return output


def _default_output_dir(pre: argparse.Namespace) -> Path:
    return (
        PROJECT_ROOT
        / "research_output"
        / (
            f"optimal_switching_fqi_strategy_{pre.strategy_sequence}_"
            f"{pre.analysis_start}_to_{pre.analysis_end}"
        )
    ).resolve()


def _patch_snapshot() -> None:
    original = universe_scale._immutable_model_snapshot

    def snapshot(strategy: dict[str, Any]):
        family, model_settings, _old_hash = original(strategy)
        merged = deepcopy(model_settings)
        merged["optimal_switching_fqi"] = FQI_SETTINGS.as_dict()
        digest = hashlib.sha256(
            json.dumps(
                merged,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        return family, merged, digest

    universe_scale._immutable_model_snapshot = snapshot


def _metric(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row.get(name)
    return None


def _comparison(
    baseline_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    baseline = json.loads(
        (baseline_dir / "universe_scale_summary.json").read_text(
            encoding="utf-8"
        )
    )
    variant = json.loads(
        (output_dir / "universe_scale_summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_by_size = {
        int(row["universe_size"]): row
        for row in list(baseline.get("series") or [])
    }
    rows: list[dict[str, Any]] = []

    for current in list(variant.get("series") or []):
        size = int(current["universe_size"])
        control = baseline_by_size.get(size)
        if control is None:
            continue

        action_counts: dict[str, int] = {}
        predictions_path = output_dir / f"universe_{size}_predictions.csv"
        if predictions_path.exists():
            predictions = pd.read_csv(
                predictions_path,
                usecols=lambda name: name == "fqi_selected_action",
            )
            if "fqi_selected_action" in predictions.columns:
                action_counts = {
                    str(key): int(value)
                    for key, value in predictions[
                        "fqi_selected_action"
                    ].fillna("UNKNOWN").value_counts().items()
                }

        baseline_capital = float(
            _metric(
                control,
                "strategy_ending_capital",
                "ending_capital",
            )
            or 0.0
        )
        variant_capital = float(
            _metric(
                current,
                "strategy_ending_capital",
                "ending_capital",
            )
            or 0.0
        )
        rows.append(
            {
                "universe_size": size,
                "baseline_ending_capital": baseline_capital,
                "fqi_ending_capital": variant_capital,
                "capital_delta": variant_capital - baseline_capital,
                "capital_delta_ratio": (
                    variant_capital / baseline_capital - 1.0
                    if baseline_capital > 0
                    else None
                ),
                "baseline_cagr": _metric(
                    control, "strategy_cagr", "cagr"
                ),
                "fqi_cagr": _metric(
                    current, "strategy_cagr", "cagr"
                ),
                "baseline_sharpe": _metric(
                    control, "strategy_sharpe", "sharpe"
                ),
                "fqi_sharpe": _metric(
                    current, "strategy_sharpe", "sharpe"
                ),
                "baseline_maximum_drawdown": _metric(
                    control,
                    "maximum_drawdown",
                    "strategy_max_drawdown",
                    "max_drawdown",
                ),
                "fqi_maximum_drawdown": _metric(
                    current,
                    "maximum_drawdown",
                    "strategy_max_drawdown",
                    "max_drawdown",
                ),
                "baseline_rotations": _metric(
                    control, "capital_rotations", "rotation_count"
                ),
                "fqi_rotations": _metric(
                    current, "capital_rotations", "rotation_count"
                ),
                "baseline_cash_days": _metric(control, "cash_days"),
                "fqi_cash_days": _metric(current, "cash_days"),
                "fqi_hold_decisions": action_counts.get("HOLD", 0),
                "fqi_rotate_decisions": action_counts.get("ROTATE", 0),
                "fqi_enter_decisions": action_counts.get("ENTER", 0),
                "fqi_cash_decisions": (
                    action_counts.get("CASH", 0)
                    + action_counts.get("EXIT_TO_CASH", 0)
                ),
            }
        )

    if not rows:
        raise RuntimeError(
            "No matching baseline universe sizes were found for comparison."
        )
    return pd.DataFrame(rows).sort_values("universe_size")


def _finalize(
    *,
    baseline_dir: Path,
    baseline_source: Path,
    output_dir: Path,
) -> Path:
    manifest_path = output_dir / "universe_scale_manifest.json"
    summary_path = output_dir / "universe_scale_summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        raise RuntimeError(
            "Optimal Switching output is incomplete; expected "
            "universe_scale_manifest.json and universe_scale_summary.json."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "optimal_switching_fitted_q_iteration",
            "script_version": SCRIPT_VERSION,
            "baseline_artifact_source": str(baseline_source),
            "frozen_universe_reused": True,
            "optimal_switching_fqi": FQI_SETTINGS.as_dict(),
            "model_target_changed": False,
            "lightgbm_opportunity_hyperparameters_changed": False,
            "decision_policy_changed": True,
            "test_period_used_for_fqi_training": False,
        }
    )
    universe_scale._write_json(manifest_path, manifest)

    comparison = _comparison(baseline_dir, output_dir)
    universe_scale._write_csv(
        output_dir / "optimal_switching_fqi_comparison.csv",
        comparison,
    )
    universe_scale._write_json(
        output_dir / "optimal_switching_fqi_comparison.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "algorithm": "Fitted Q Iteration / approximate optimal switching",
            "hypothesis": (
                "A sequential action-value policy can distinguish HOLD, "
                "ROTATE and CASH better than an instantaneous utility ranking."
            ),
            "settings": FQI_SETTINGS.as_dict(),
            "series": comparison.to_dict(orient="records"),
        },
    )

    archive_path = universe_scale._archive_output_dir(output_dir)
    print("\n=== OPTIMAL SWITCHING FQI COMPARISON ===", flush=True)
    for row in comparison.to_dict(orient="records"):
        delta = row.get("capital_delta_ratio")
        delta_text = (
            f"{float(delta) * 100:+.2f}%"
            if delta is not None
            else "n/a"
        )
        universe_scale._log(
            f"{int(row['universe_size'])} assets: "
            f"baseline=${float(row['baseline_ending_capital']):,.2f}; "
            f"FQI=${float(row['fqi_ending_capital']):,.2f}; "
            f"delta={delta_text}; "
            f"cash_days={int(row.get('fqi_cash_days') or 0)}."
        )
    universe_scale._log(f"Optimal Switching artifacts: {output_dir}")
    universe_scale._log(f"Optimal Switching archive: {archive_path}")
    return archive_path


def main() -> int:
    original_argv = list(sys.argv[1:])
    pre = _preparse(original_argv)
    baseline_dir, baseline_temp, baseline_source = (
        resolve_universe_scale_baseline(
            project_root=PROJECT_ROOT,
            strategy_sequence=pre.strategy_sequence,
            analysis_start=pre.analysis_start,
            analysis_end=pre.analysis_end,
            requested=pre.baseline_output_dir,
        )
    )
    output_dir = (
        Path(pre.output_dir).resolve()
        if pre.output_dir
        else _default_output_dir(pre)
    )

    try:
        if not pre.finalize_only:
            _patch_snapshot()
            universe_scale.CandidateUniverseLoader = (
                FrozenUniverseScaleCandidateLoader
            )
            universe_scale.run_rotation_models = (
                run_optimal_switching_fqi
            )
            universe_scale.SCRIPT_VERSION = SCRIPT_VERSION

            seed_file = (
                baseline_dir
                / "universe_candidate_history_integrity.csv"
            )
            forwarded = _strip_custom(
                original_argv,
                {"--baseline-output-dir", "--finalize-only"},
            )
            if "--seed-universe-file" not in forwarded:
                forwarded.extend(
                    ["--seed-universe-file", str(seed_file)]
                )
            if "--output-dir" not in forwarded:
                forwarded.extend(["--output-dir", str(output_dir)])
            sys.argv = [sys.argv[0], *forwarded]

            status = universe_scale.main()
            if status != 0:
                return int(status)

        _finalize(
            baseline_dir=baseline_dir,
            baseline_source=baseline_source,
            output_dir=output_dir,
        )
        return 0
    finally:
        if baseline_temp is not None:
            baseline_temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
