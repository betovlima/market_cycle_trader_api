from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import research_contextual_marginal_signature_v111 as base

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.11.1"
EXPERIMENT_NAME = base.EXPERIMENT_NAME
WINDOWS_SAFE_PATH_LIMIT = 240


def _assert_path_budget(path: Path) -> None:
    """Fail early with a clear message before Windows raises FileNotFoundError.

    The classic Win32 path limit is 260 characters when long-path support is not
    enabled. Keep research trace files comfortably below that threshold so the
    experiment remains portable across Windows installations.
    """
    if os.name == "nt" and len(str(path.resolve())) > WINDOWS_SAFE_PATH_LIMIT:
        raise RuntimeError(
            "Trace output path is too long for a portable Windows run "
            f"({len(str(path.resolve()))} characters): {path}. "
            "Use a shorter --output-dir."
        )


def _compact_save_capture(base_path: Path, results: list[Any]) -> dict[str, dict[str, Any]]:
    """Persist RotationRunResult traces without deep backend directories.

    v1.0.11 used:
      <base>/<backend_name>/prediction_columns.json

    which can exceed the legacy Win32 MAX_PATH limit when repository, output,
    universe and backend names are all long. v1.0.11.1 keeps the logical backend
    key in memory and in trace_index.json, but stores files using short prefixes:
      <base>/b00_p.csv
      <base>/b00_t.csv
      <base>/b00_m.json
      <base>/b00_s.txt
      <base>/b00_c.json
    """
    base_path.mkdir(parents=True, exist_ok=True)
    captured: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        key = base._backend_key(result, index)
        prefix = f"b{index:02d}"
        predictions = base._prediction_frame(result)
        trades = base._trade_frame(result)

        paths = {
            "predictions": base_path / f"{prefix}_p.csv",
            "trades": base_path / f"{prefix}_t.csv",
            "metrics": base_path / f"{prefix}_m.json",
            "summary": base_path / f"{prefix}_s.txt",
            "columns": base_path / f"{prefix}_c.json",
        }
        for path in paths.values():
            _assert_path_budget(path)

        base._write_frame(paths["predictions"], predictions)
        base._write_frame(paths["trades"], trades)
        base._write_json(paths["metrics"], getattr(result, "metrics", {}) or {})
        paths["summary"].write_text(str(getattr(result, "summary", "") or ""), encoding="utf-8")
        base._write_json(paths["columns"], list(predictions.columns))

        captured[key] = {
            "predictions": predictions,
            "trades": trades,
            "metrics": dict(getattr(result, "metrics", {}) or {}),
        }
        index_rows.append(
            {
                "backend_key": key,
                "backend": str(getattr(result, "backend", "") or ""),
                "index": index,
                "prefix": prefix,
                "files": {name: path.name for name, path in paths.items()},
            }
        )

    index_path = base_path / "trace_index.json"
    _assert_path_budget(index_path)
    base._write_json(
        index_path,
        {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "storage_layout": "compact_windows_safe",
            "backends": index_rows,
        },
    )
    return captured


def main() -> int:
    # Preserve the v1.0.11 scientific protocol and replace only the physical trace
    # storage layout. Patching the module globals is intentional: base.main resolves
    # both names at execution time.
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._save_capture = _compact_save_capture
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
