from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from market_cycle_trader_api.engine import research_challengers  # noqa: E402
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (  # noqa: E402
    get_alpaca_credentials,
)
from market_cycle_trader_api.services.asset_universe_scale_candidates import (  # noqa: E402
    CandidateUniverseLoader,
)
from market_cycle_trader_api.services.incumbent_trend_persistence import (  # noqa: E402
    IncumbentTrendPersistenceSettings,
    wrap_incumbent_trend_persistence_policy,
)

SCRIPT_VERSION = "incumbent-trend-persistence-v1.0.0"
PERSISTENCE_SETTINGS = IncumbentTrendPersistenceSettings(
    enabled=True,
    max_current_rank=5,
    max_challenger_gap_zscore=1.0,
    min_evidence_count=4,
)


def _bool_mask(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


class FrozenBaselineCandidateLoader(CandidateUniverseLoader):
    """Reload exactly the external assets accepted by the baseline experiment.

    This deliberately does not discover substitutes. If one frozen asset can no
    longer be reproduced, the experiment fails instead of silently changing the
    universe being compared.
    """

    def prepare(
        self,
        required_external: int,
        *,
        seed_file: Path | None,
    ) -> tuple[dict[str, pd.DataFrame], list[str], list[dict[str, Any]], int]:
        if required_external <= 0:
            self.log("Frozen candidate reload skipped: Strategy control only.")
            return {}, [], [], 0
        if seed_file is None or not seed_file.exists():
            raise RuntimeError(
                "Incumbent persistence research requires the baseline "
                "universe_candidate_history_integrity.csv file."
            )

        seed_frame = pd.read_csv(seed_file)
        required_columns = {"symbol", "source", "history_complete", "accepted"}
        if not required_columns.issubset(seed_frame.columns):
            missing = sorted(required_columns.difference(seed_frame.columns))
            raise RuntimeError(
                "Baseline candidate integrity file is missing: " + ", ".join(missing)
            )

        candidate_mask = seed_frame["source"].astype(str).str.lower().eq("candidate")
        complete_mask = _bool_mask(seed_frame["history_complete"])
        accepted_mask = _bool_mask(seed_frame["accepted"])
        frozen = [
            str(value).strip().upper()
            for value in seed_frame.loc[
                candidate_mask & complete_mask & accepted_mask,
                "symbol",
            ]
            if str(value).strip()
        ]
        frozen = list(dict.fromkeys(frozen))
        if len(frozen) < required_external:
            raise RuntimeError(
                f"Baseline contains only {len(frozen)} accepted external assets; "
                f"{required_external} are required."
            )
        frozen = frozen[:required_external]
        self.log(
            f"Frozen baseline universe: revalidating exactly {len(frozen)} external assets; "
            "no discovery or substitution is allowed."
        )

        credentials = get_alpaca_credentials(self.db)
        accepted_frames: dict[str, pd.DataFrame] = {}
        diagnostics: list[dict[str, Any]] = []
        failed: list[tuple[str, Any]] = []

        for start in range(0, len(frozen), self.workers):
            batch = frozen[start : start + self.workers]
            completed: dict[str, tuple[pd.DataFrame | None, dict[str, Any]]] = {}
            with ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="incumbent-history",
            ) as executor:
                futures = {
                    executor.submit(self._evaluate, symbol, credentials): symbol
                    for symbol in batch
                }
                for future in as_completed(futures):
                    symbol, frame, row = future.result()
                    completed[symbol] = (frame, row)

            for symbol in batch:
                frame, row = completed[symbol]
                diagnostics.append(row)
                if frame is None or not row.get("accepted"):
                    failed.append(
                        (
                            symbol,
                            row.get("failed_quality_checks")
                            or row.get("error")
                            or "not_reproducible",
                        )
                    )
                    continue
                accepted_frames[symbol] = frame
                self.log(
                    f"Frozen candidate {len(accepted_frames)}/{len(frozen)} - {symbol}: "
                    f"history={row.get('observed_rows')}, source={row.get('data_source')}."
                )

        if failed:
            sample = "; ".join(f"{symbol}: {reason}" for symbol, reason in failed[:10])
            raise RuntimeError(
                "Frozen baseline universe could not be reproduced exactly. "
                f"Failures={len(failed)}. {sample}"
            )
        return accepted_frames, frozen, diagnostics, len(frozen)


def _preparse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--analysis-start", required=True)
    parser.add_argument("--analysis-end", required=True)
    parser.add_argument("--baseline-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    values, _ = parser.parse_known_args(argv)
    return values


def _strip_custom_option(argv: list[str], name: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name:
            index += 2
            continue
        if token.startswith(name + "="):
            index += 1
            continue
        output.append(token)
        index += 1
    return output


def _patch_lightgbm_snapshot() -> None:
    original = universe_scale._immutable_model_snapshot

    def snapshot(strategy: dict[str, Any]):
        family, model_settings, _settings_hash = original(strategy)
        merged = deepcopy(model_settings)
        merged["incumbent_trend_persistence"] = PERSISTENCE_SETTINGS.as_dict()
        digest = hashlib.sha256(
            json.dumps(merged, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        return family, merged, digest

    universe_scale._immutable_model_snapshot = snapshot


def _patch_final_policy_only() -> None:
    original = research_challengers._utility_policy

    def research_policy(*args: Any, **kwargs: Any):
        base_policy = original(*args, **kwargs)
        diagnostics = kwargs.get("decision_diagnostics")
        if diagnostics is None:
            # Calibration remains byte-for-byte equivalent to the baseline logic.
            return base_policy
        if len(args) < 4:
            return base_policy
        frames = args[1]
        symbols = args[2]
        config = args[3]
        settings = IncumbentTrendPersistenceSettings.from_config(config)
        if not settings.enabled:
            return base_policy
        return wrap_incumbent_trend_persistence_policy(
            base_policy,
            frames=frames,
            symbols=symbols,
            decision_diagnostics=diagnostics,
            settings=settings,
        )

    research_challengers._utility_policy = research_policy


def _comparison(
    baseline_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    baseline_path = baseline_dir / "universe_scale_summary.json"
    variant_path = output_dir / "universe_scale_summary.json"
    if not baseline_path.exists():
        raise RuntimeError(f"Baseline summary not found: {baseline_path}")
    if not variant_path.exists():
        raise RuntimeError(f"Persistence summary not found: {variant_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    baseline_by_size = {
        int(row["universe_size"]): row for row in list(baseline.get("series") or [])
    }
    rows: list[dict[str, Any]] = []

    for current in list(variant.get("series") or []):
        size = int(current["universe_size"])
        control = baseline_by_size.get(size)
        if control is None:
            continue
        predictions_path = output_dir / f"universe_{size}_predictions.csv"
        guard_count = 0
        if predictions_path.exists():
            predictions = pd.read_csv(
                predictions_path,
                usecols=lambda name: name == "incumbent_persistence_guard_applied",
            )
            if "incumbent_persistence_guard_applied" in predictions.columns:
                guard_count = int(
                    _bool_mask(predictions["incumbent_persistence_guard_applied"]).sum()
                )

        baseline_capital = float(control.get("strategy_ending_capital") or 0.0)
        variant_capital = float(current.get("strategy_ending_capital") or 0.0)
        rows.append(
            {
                "universe_size": size,
                "baseline_ending_capital": baseline_capital,
                "persistence_ending_capital": variant_capital,
                "capital_delta": variant_capital - baseline_capital,
                "capital_delta_ratio": (
                    variant_capital / baseline_capital - 1.0
                    if baseline_capital > 0
                    else None
                ),
                "baseline_cagr": control.get("strategy_cagr"),
                "persistence_cagr": current.get("strategy_cagr"),
                "baseline_sharpe": control.get("strategy_sharpe"),
                "persistence_sharpe": current.get("strategy_sharpe"),
                "baseline_maximum_drawdown": control.get("maximum_drawdown"),
                "persistence_maximum_drawdown": current.get("maximum_drawdown"),
                "baseline_rotations": control.get("capital_rotations"),
                "persistence_rotations": current.get("capital_rotations"),
                "persistence_holds_applied": guard_count,
            }
        )
    return pd.DataFrame(rows).sort_values("universe_size")


def main() -> int:
    original_argv = list(sys.argv[1:])
    pre = _preparse(original_argv)
    baseline_dir = (
        Path(pre.baseline_output_dir).resolve()
        if pre.baseline_output_dir
        else (
            PROJECT_ROOT
            / "research_output"
            / (
                f"asset_rotation_universe_scale_strategy_{pre.strategy_sequence}_"
                f"{pre.analysis_start}_to_{pre.analysis_end}"
            )
        ).resolve()
    )
    seed_file = baseline_dir / "universe_candidate_history_integrity.csv"
    if not seed_file.exists():
        raise RuntimeError(
            "Baseline universe artifacts are required before this experiment. "
            f"Missing: {seed_file}"
        )

    output_dir = (
        Path(pre.output_dir).resolve()
        if pre.output_dir
        else (
            PROJECT_ROOT
            / "research_output"
            / (
                f"incumbent_trend_persistence_strategy_{pre.strategy_sequence}_"
                f"{pre.analysis_start}_to_{pre.analysis_end}"
            )
        ).resolve()
    )

    _patch_lightgbm_snapshot()
    _patch_final_policy_only()
    universe_scale.CandidateUniverseLoader = FrozenBaselineCandidateLoader
    universe_scale.SCRIPT_VERSION = SCRIPT_VERSION

    forwarded = _strip_custom_option(original_argv, "--baseline-output-dir")
    if "--seed-universe-file" not in forwarded:
        forwarded.extend(["--seed-universe-file", str(seed_file)])
    if "--output-dir" not in forwarded:
        forwarded.extend(["--output-dir", str(output_dir)])
    sys.argv = [sys.argv[0], *forwarded]

    status = universe_scale.main()
    if status != 0:
        return int(status)

    manifest_path = output_dir / "universe_scale_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment": "incumbent_trend_persistence_vs_base_rotation",
            "baseline_output_dir": str(baseline_dir),
            "baseline_seed_file": str(seed_file),
            "frozen_universe_reused": True,
            "incumbent_trend_persistence": PERSISTENCE_SETTINGS.as_dict(),
            "calibration_policy_changed": False,
            "model_target_changed": False,
            "lightgbm_hyperparameters_changed": False,
        }
    )
    universe_scale._write_json(manifest_path, manifest)

    comparison = _comparison(baseline_dir, output_dir)
    universe_scale._write_csv(
        output_dir / "incumbent_persistence_comparison.csv",
        comparison,
    )
    universe_scale._write_json(
        output_dir / "incumbent_persistence_comparison.json",
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "hypothesis": (
                "Protect a still-healthy incumbent when a challenger is only moderately "
                "better on the same-date cross-sectional score scale."
            ),
            "settings": PERSISTENCE_SETTINGS.as_dict(),
            "series": comparison.to_dict(orient="records"),
        },
    )

    archive_path = universe_scale._archive_output_dir(output_dir)
    print("\n=== INCUMBENT TREND PERSISTENCE COMPARISON ===", flush=True)
    for row in comparison.to_dict(orient="records"):
        delta = row.get("capital_delta_ratio")
        delta_text = f"{float(delta) * 100:+.2f}%" if delta is not None else "n/a"
        universe_scale._log(
            f"{int(row['universe_size'])} assets: "
            f"baseline=${float(row['baseline_ending_capital']):,.2f}; "
            f"persistence=${float(row['persistence_ending_capital']):,.2f}; "
            f"delta={delta_text}; holds={int(row['persistence_holds_applied'])}."
        )
    universe_scale._log(f"Persistence artifacts: {output_dir}")
    universe_scale._log(f"Persistence archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
