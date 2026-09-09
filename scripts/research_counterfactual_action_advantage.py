from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_asset_rotation_universe_scale as universe_scale  # noqa: E402
from market_cycle_trader_api.engine.counterfactual_action_advantage_research import (  # noqa: E402
    run_counterfactual_action_advantage,
)
from market_cycle_trader_api.services.counterfactual_action_advantage import (  # noqa: E402
    CounterfactualActionAdvantageSettings,
)
from market_cycle_trader_api.services.research_universe_baseline import (  # noqa: E402
    FrozenUniverseScaleCandidateLoader,
    resolve_universe_scale_baseline,
)

SCRIPT_VERSION = "counterfactual-action-advantage-v2.0.0"
CAA_SETTINGS = CounterfactualActionAdvantageSettings(enabled=True)


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
            f"counterfactual_action_advantage_strategy_{pre.strategy_sequence}_"
            f"{pre.analysis_start}_to_{pre.analysis_end}"
        )
    ).resolve()


def _patch_snapshot() -> None:
    original = universe_scale._immutable_model_snapshot

    def snapshot(strategy: dict[str, Any]):
        family, model_settings, _old_hash = original(strategy)
        merged = deepcopy(model_settings)
        merged["counterfactual_action_advantage"] = CAA_SETTINGS.as_dict()
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


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _research_source_sha256() -> str:
    digest = hashlib.sha256()
    relative_paths = [
        "scripts/research_counterfactual_action_advantage.py",
        "src/market_cycle_trader_api/engine/counterfactual_action_advantage_research.py",
        "src/market_cycle_trader_api/services/counterfactual_action_advantage.py",
        "src/market_cycle_trader_api/services/optimal_switching_state.py",
    ]
    for relative in relative_paths:
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _equity_max_drawdown(path: Path) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(
        path,
        usecols=lambda name: name in {"strategy_equity"},
    )
    if "strategy_equity" not in frame.columns:
        return None
    equity = pd.to_numeric(frame["strategy_equity"], errors="coerce").dropna()
    if equity.empty:
        return None
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def _decision_statistics(predictions_path: Path) -> dict[str, Any]:
    if not predictions_path.exists():
        return {}
    columns = {
        "decision_reason",
        "current_asset",
        "best_asset",
        "final_action_asset",
        "caa_selected_advantage",
        "caa_cash_advantage",
        "caa_best_candidate_advantage",
    }
    frame = pd.read_csv(
        predictions_path,
        usecols=lambda name: name in columns,
    )
    result: dict[str, Any] = {}
    if "decision_reason" in frame.columns:
        counts = frame["decision_reason"].fillna("UNKNOWN").value_counts()
        result.update(
            {
                "caa_hold_decisions": int(counts.get("CAA_HOLD", 0)),
                "caa_rotate_decisions": int(counts.get("CAA_ROTATE", 0)),
                "caa_enter_decisions": int(counts.get("CAA_ENTER", 0)),
                "caa_exit_to_cash_decisions": int(counts.get("CAA_CASH", 0)),
                "caa_stay_cash_decisions": int(counts.get("CAA_STAY_CASH", 0)),
            }
        )
    required = {"current_asset", "best_asset", "final_action_asset"}
    if required.issubset(frame.columns):
        valid = frame[list(required)].dropna()
        changed = valid["current_asset"] != valid["best_asset"]
        held = valid["final_action_asset"] == valid["current_asset"]
        to_cash = valid["final_action_asset"] == "CASH"
        result["top1_chase_avoided_days"] = int((changed & held).sum())
        result["cash_instead_of_top1_days"] = int((changed & to_cash).sum())
    for column in (
        "caa_selected_advantage",
        "caa_cash_advantage",
        "caa_best_candidate_advantage",
    ):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            result[f"mean_{column}"] = (
                float(values.mean()) if not values.empty else None
            )
    return result


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

        baseline_capital = float(
            _metric(control, "strategy_ending_capital", "ending_capital")
            or 0.0
        )
        variant_capital = float(
            _metric(current, "strategy_ending_capital", "ending_capital")
            or 0.0
        )
        stats = _decision_statistics(
            output_dir / f"universe_{size}_predictions.csv"
        )
        baseline_maxdd = _metric(
            control,
            "strategy_maximum_drawdown",
            "maximum_drawdown",
            "strategy_max_drawdown",
            "max_drawdown",
        )
        if baseline_maxdd is None:
            baseline_maxdd = _equity_max_drawdown(
                baseline_dir / f"universe_{size}_predictions.csv"
            )
        variant_maxdd = _metric(
            current,
            "strategy_maximum_drawdown",
            "maximum_drawdown",
            "strategy_max_drawdown",
            "max_drawdown",
        )
        if variant_maxdd is None:
            variant_maxdd = _equity_max_drawdown(
                output_dir / f"universe_{size}_predictions.csv"
            )

        rows.append(
            {
                "universe_size": size,
                "baseline_ending_capital": baseline_capital,
                "caa_ending_capital": variant_capital,
                "capital_delta": variant_capital - baseline_capital,
                "capital_delta_ratio": (
                    variant_capital / baseline_capital - 1.0
                    if baseline_capital > 0
                    else None
                ),
                "baseline_cagr": _metric(control, "strategy_cagr", "cagr"),
                "caa_cagr": _metric(current, "strategy_cagr", "cagr"),
                "baseline_sharpe": _metric(control, "strategy_sharpe", "sharpe"),
                "caa_sharpe": _metric(current, "strategy_sharpe", "sharpe"),
                "baseline_maximum_drawdown": baseline_maxdd,
                "caa_maximum_drawdown": variant_maxdd,
                "baseline_rotations": _metric(
                    control, "capital_rotations", "rotation_count"
                ),
                "caa_rotations": _metric(
                    current, "capital_rotations", "rotation_count"
                ),
                "baseline_cash_days": _metric(control, "cash_days"),
                "caa_cash_days": _metric(current, "cash_days"),
                **stats,
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
            "Counterfactual Action Advantage output is incomplete; expected "
            "universe_scale_manifest.json and universe_scale_summary.json."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "counterfactual_action_advantage",
            "script_version": SCRIPT_VERSION,
            "baseline_artifact_source": str(baseline_source),
            "frozen_universe_reused": True,
            "counterfactual_action_advantage": CAA_SETTINGS.as_dict(),
            "model_target_changed": False,
            "lightgbm_opportunity_hyperparameters_changed": False,
            "decision_policy_changed": True,
            "test_period_used_for_action_advantage_training": False,
            "action_advantage_label_tail_purged": True,
            "action_advantage_label_horizon_sessions": 1,
            "git_commit_at_execution": _git_commit(),
            "research_source_sha256": _research_source_sha256(),
        }
    )
    universe_scale._write_json(manifest_path, manifest)

    comparison = _comparison(baseline_dir, output_dir)
    universe_scale._write_csv(
        output_dir / "counterfactual_action_advantage_comparison.csv",
        comparison,
    )
    universe_scale._write_json(
        output_dir / "counterfactual_action_advantage_comparison.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "algorithm": "Supervised Counterfactual Action Advantage",
            "hypothesis": (
                "A learned next-decision incremental advantage can decide "
                "whether changing from the current position to CASH or the "
                "current LightGBM Top-1 adds economic value, without "
                "hand-written switch, cash or minimum-hold thresholds."
            ),
            "settings": CAA_SETTINGS.as_dict(),
            "series": comparison.replace({np.nan: None}).to_dict(orient="records"),
        },
    )

    archive_path = universe_scale._archive_output_dir(output_dir)
    print("\n=== COUNTERFACTUAL ACTION ADVANTAGE COMPARISON ===", flush=True)
    for row in comparison.to_dict(orient="records"):
        delta = row.get("capital_delta_ratio")
        delta_text = (
            f"{float(delta) * 100:+.2f}%"
            if delta is not None and np.isfinite(delta)
            else "n/a"
        )
        universe_scale._log(
            f"{int(row['universe_size'])} assets: "
            f"baseline=${float(row['baseline_ending_capital']):,.2f}; "
            f"CAA=${float(row['caa_ending_capital']):,.2f}; "
            f"delta={delta_text}; "
            f"rotations={int(row.get('caa_rotations') or 0)}; "
            f"cash_days={int(row.get('caa_cash_days') or 0)}."
        )
    universe_scale._log(f"Counterfactual Action Advantage artifacts: {output_dir}")
    universe_scale._log(f"Counterfactual Action Advantage archive: {archive_path}")
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
                run_counterfactual_action_advantage
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
