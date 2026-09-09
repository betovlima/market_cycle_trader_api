from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import pandas as pd

from ..infrastructure.persistence.mongo_repository import get_alpaca_credentials
from .asset_universe_scale_candidates import CandidateUniverseLoader


BASELINE_REQUIRED_FILES = (
    "universe_candidate_history_integrity.csv",
    "universe_scale_summary.json",
)


def _bool_mask(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _archive_member(archive: zipfile.ZipFile, filename: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if not name.endswith("/") and Path(name).name == filename
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Baseline archive must contain exactly one {filename}; "
            f"found={len(matches)}."
        )
    return matches[0]


def resolve_universe_scale_baseline(
    *,
    project_root: Path,
    strategy_sequence: int,
    analysis_start: str,
    analysis_end: str,
    requested: str | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None, Path]:
    default_dir = (
        project_root
        / "research_output"
        / (
            f"asset_rotation_universe_scale_strategy_{strategy_sequence}_"
            f"{analysis_start}_to_{analysis_end}"
        )
    ).resolve()
    selected = Path(requested).resolve() if requested else default_dir

    if selected.suffix.lower() == ".zip":
        baseline_dir = selected.with_suffix("")
        archive_path = selected
    else:
        baseline_dir = selected
        archive_path = selected.parent / f"{selected.name}.zip"

    missing = [
        name
        for name in BASELINE_REQUIRED_FILES
        if not (baseline_dir / name).exists()
    ]
    if not missing:
        return baseline_dir, None, baseline_dir

    if not archive_path.exists():
        missing_paths = ", ".join(
            str(baseline_dir / name) for name in missing
        )
        raise RuntimeError(
            "Universe-scale baseline artifacts are required. "
            f"Missing: {missing_paths}. "
            f"Baseline archive also not found: {archive_path}"
        )

    temporary = tempfile.TemporaryDirectory(
        prefix="mct_universe_scale_baseline_"
    )
    materialized = Path(temporary.name).resolve()
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        for filename in BASELINE_REQUIRED_FILES:
            member = _archive_member(archive, filename)
            (materialized / filename).write_bytes(
                archive.read(member)
            )
    return materialized, temporary, archive_path


class FrozenUniverseScaleCandidateLoader(CandidateUniverseLoader):
    """Revalidate exactly the external assets accepted by the baseline run."""

    def prepare(
        self,
        required_external: int,
        *,
        seed_file: Path | None,
    ) -> tuple[
        dict[str, pd.DataFrame],
        list[str],
        list[dict[str, Any]],
        int,
    ]:
        if required_external <= 0:
            self.log("Frozen candidate reload skipped: Strategy control only.")
            return {}, [], [], 0
        if seed_file is None or not seed_file.exists():
            raise RuntimeError(
                "Frozen universe research requires "
                "universe_candidate_history_integrity.csv."
            )

        seed = pd.read_csv(seed_file)
        required = {"symbol", "source", "history_complete", "accepted"}
        if not required.issubset(seed.columns):
            missing = sorted(required.difference(seed.columns))
            raise RuntimeError(
                "Baseline candidate integrity file is missing: "
                + ", ".join(missing)
            )

        mask = (
            seed["source"].astype(str).str.lower().eq("candidate")
            & _bool_mask(seed["history_complete"])
            & _bool_mask(seed["accepted"])
        )
        frozen = [
            str(value).strip().upper()
            for value in seed.loc[mask, "symbol"]
            if str(value).strip()
        ]
        frozen = list(dict.fromkeys(frozen))
        if len(frozen) < required_external:
            raise RuntimeError(
                f"Baseline contains only {len(frozen)} accepted external "
                f"assets; {required_external} are required."
            )
        frozen = frozen[:required_external]
        self.log(
            f"Frozen baseline universe: revalidating exactly "
            f"{len(frozen)} external assets; no discovery or substitution."
        )

        credentials = get_alpaca_credentials(self.db)
        accepted_frames: dict[str, pd.DataFrame] = {}
        diagnostics: list[dict[str, Any]] = []
        failures: list[tuple[str, Any]] = []

        for start in range(0, len(frozen), self.workers):
            batch = frozen[start : start + self.workers]
            completed: dict[
                str,
                tuple[pd.DataFrame | None, dict[str, Any]],
            ] = {}
            with ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="frozen-universe-history",
            ) as executor:
                futures = {
                    executor.submit(
                        self._evaluate, symbol, credentials
                    ): symbol
                    for symbol in batch
                }
                for future in as_completed(futures):
                    symbol, frame, row = future.result()
                    completed[symbol] = (frame, row)

            for symbol in batch:
                frame, row = completed[symbol]
                diagnostics.append(row)
                if frame is None or not row.get("accepted"):
                    failures.append(
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
                    f"Frozen candidate "
                    f"{len(accepted_frames)}/{len(frozen)} - {symbol}: "
                    f"history={row.get('observed_rows')}, "
                    f"source={row.get('data_source')}."
                )

        if failures:
            sample = "; ".join(
                f"{symbol}: {reason}"
                for symbol, reason in failures[:10]
            )
            raise RuntimeError(
                "Frozen baseline universe could not be reproduced exactly. "
                f"Failures={len(failures)}. {sample}"
            )
        return accepted_frames, frozen, diagnostics, len(frozen)
