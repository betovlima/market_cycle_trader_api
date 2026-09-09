from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import research_optimal_switching_fqi as base
from market_cycle_trader_api.services.optimal_switching_fqi import (
    OptimalSwitchingFQISettings,
)

SCRIPT_VERSION = "optimal-switching-position-aware-v1.1.0"
POSITION_AWARE_SETTINGS = OptimalSwitchingFQISettings(
    enabled=True,
    action_top_k=5,
    bellman_iterations=20,
    position_aware_state=True,
)

_ORIGINAL_COMPARISON = base._comparison
_ORIGINAL_FINALIZE = base._finalize


def _default_output_dir(pre) -> Path:
    return (
        base.PROJECT_ROOT
        / "research_output"
        / (
            f"optimal_switching_position_aware_strategy_{pre.strategy_sequence}_"
            f"{pre.analysis_start}_to_{pre.analysis_end}"
        )
    ).resolve()


def _comparison(baseline_dir: Path, output_dir: Path) -> pd.DataFrame:
    frame = _ORIGINAL_COMPARISON(baseline_dir, output_dir)

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
    variant_by_size = {
        int(row["universe_size"]): row
        for row in list(variant.get("series") or [])
    }

    for index, row in frame.iterrows():
        size = int(row["universe_size"])
        control = baseline_by_size.get(size, {})
        current = variant_by_size.get(size, {})
        frame.at[index, "baseline_maximum_drawdown"] = base._metric(
            control,
            "strategy_maximum_drawdown",
            "maximum_drawdown",
            "strategy_max_drawdown",
            "max_drawdown",
        )
        frame.at[index, "fqi_maximum_drawdown"] = base._metric(
            current,
            "strategy_maximum_drawdown",
            "maximum_drawdown",
            "strategy_max_drawdown",
            "max_drawdown",
        )

    return frame


def _finalize(
    *,
    baseline_dir: Path,
    baseline_source: Path,
    output_dir: Path,
) -> Path:
    archive_path = _ORIGINAL_FINALIZE(
        baseline_dir=baseline_dir,
        baseline_source=baseline_source,
        output_dir=output_dir,
    )

    manifest_path = output_dir / "universe_scale_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": (
                "optimal_switching_fitted_q_iteration_position_aware"
            ),
            "script_version": SCRIPT_VERSION,
            "position_aware_state": True,
            "position_state_source": (
                "causal holding history reconstructed from the current "
                "position and pre-decision market data"
            ),
            "calibration_state_sampling": (
                "grounded baseline-policy trajectories"
            ),
            "fqi_v1_0_result_used_for_training": False,
        }
    )
    base.universe_scale._write_json(manifest_path, manifest)

    comparison_json_path = (
        output_dir / "optimal_switching_fqi_comparison.json"
    )
    if comparison_json_path.exists():
        comparison_payload = json.loads(
            comparison_json_path.read_text(encoding="utf-8")
        )
        comparison_payload.update(
            {
                "script_version": SCRIPT_VERSION,
                "algorithm": (
                    "Position-Aware Fitted Q Iteration / "
                    "approximate optimal switching"
                ),
                "hypothesis": (
                    "Explicit causal history of the currently held position "
                    "improves the sequential decision between HOLD, ROTATE "
                    "and CASH without changing the LightGBM opportunity model."
                ),
            }
        )
        base.universe_scale._write_json(
            comparison_json_path,
            comparison_payload,
        )

    archive_path = base.universe_scale._archive_output_dir(output_dir)
    return archive_path


def main() -> int:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.FQI_SETTINGS = POSITION_AWARE_SETTINGS
    base._default_output_dir = _default_output_dir
    base._comparison = _comparison
    base._finalize = _finalize
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
