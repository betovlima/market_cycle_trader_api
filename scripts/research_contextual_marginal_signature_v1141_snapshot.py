from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

import research_contextual_marginal_signature_v111 as base

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.14.1"
EXPERIMENT_NAME = "contextual_marginal_signature_direct_effect_snapshot_expansion"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand an existing frozen contextual-signature market snapshot only with symbols "
            "missing from a later campaign. Existing symbol histories are preserved exactly; "
            "Yahoo is queried only for missing symbols."
        )
    )
    parser.add_argument("--base-snapshot", required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--universe-spec", required=True)
    parser.add_argument("--cases-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def _safe_fresh(path: Path) -> None:
    resolved = path.resolve()
    if "research_output" not in resolved.parts or not resolved.name.startswith("contextual_marginal_signature_"):
        raise RuntimeError(f"Refusing to delete unexpected output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _snapshot_table_from_path(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError("Base snapshot is missing columns: " + ", ".join(missing))
    table = table.loc[:, ["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()
    table["symbol"] = table["symbol"].astype(str).str.upper().str.strip()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    for column in ["open", "high", "low", "close", "volume"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise RuntimeError("Base snapshot contains non-numeric OHLCV values.")
    return table.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _required_symbols(universe_spec: Path, cases_spec: Path) -> tuple[list[str], list[str], list[str]]:
    universe_order, cases = base._load_cases(cases_spec)
    universes = base._load_universe_spec(universe_spec, universe_order)
    universe_symbols = base._normalize_symbols([symbol for item in universes for symbol in item["assets"]])
    candidate_symbols = base._normalize_symbols([candidate for case in cases for candidate in case["candidates"]])
    required = base._normalize_symbols([*universe_symbols, *candidate_symbols])
    return required, universe_symbols, candidate_symbols


def _merge_snapshot(existing: pd.DataFrame, additions: pd.DataFrame, required_symbols: list[str]) -> pd.DataFrame:
    existing_symbols = set(existing["symbol"].astype(str).str.upper())
    overlap = sorted(existing_symbols.intersection(set(additions["symbol"].astype(str).str.upper())))
    if overlap:
        raise RuntimeError("Expansion attempted to overwrite frozen symbols: " + ", ".join(overlap))
    merged = pd.concat([existing, additions], ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged["symbol"] = merged["symbol"].astype(str).str.upper()
    duplicate = merged.duplicated(subset=["symbol", "timestamp"], keep=False)
    if bool(duplicate.any()):
        sample = merged.loc[duplicate, ["symbol", "timestamp"]].head(10).to_dict(orient="records")
        raise RuntimeError(f"Expanded snapshot contains duplicate symbol/timestamp rows: {sample}")
    missing = sorted(set(required_symbols).difference(set(merged["symbol"])))
    if missing:
        raise RuntimeError("Expanded snapshot still does not provide: " + ", ".join(missing))
    return merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    base_snapshot_path = Path(args.base_snapshot).resolve()
    universe_spec = Path(args.universe_spec).resolve()
    cases_spec = Path(args.cases_spec).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.fresh_run:
        _safe_fresh(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_snapshot_path.exists():
        raise RuntimeError(f"Base snapshot does not exist: {base_snapshot_path}")

    history_start = base._normalize_date(args.history_start)
    snapshot_end = base._normalize_date(args.snapshot_end)
    required, universe_symbols, candidate_symbols = _required_symbols(universe_spec, cases_spec)

    existing = _snapshot_table_from_path(base_snapshot_path)
    existing_hash = base._snapshot_hash(existing)
    existing_symbols = sorted(set(existing["symbol"]))
    missing_symbols = sorted(set(required).difference(existing_symbols))

    print(f"[{SCRIPT_VERSION}] base_snapshot={base_snapshot_path}", flush=True)
    print(f"[{SCRIPT_VERSION}] existing_symbols={len(existing_symbols)} required_symbols={len(required)}", flush=True)
    print(f"[{SCRIPT_VERSION}] missing_symbols={','.join(missing_symbols) if missing_symbols else 'none'}", flush=True)

    if missing_symbols:
        downloaded_frames = base._load_yahoo_frames(missing_symbols, history_start, snapshot_end)
        additions = base._snapshot_table(downloaded_frames)
    else:
        additions = existing.iloc[0:0].copy()

    expanded = _merge_snapshot(existing, additions, required)

    # Scientific invariant: rows for every pre-existing frozen symbol must remain exactly unchanged.
    preserved = expanded[expanded["symbol"].isin(existing_symbols)].reset_index(drop=True)
    preserved_hash = base._snapshot_hash(preserved)
    if preserved_hash != existing_hash:
        raise RuntimeError(
            "Existing frozen histories changed during snapshot expansion; refusing to write mixed snapshot."
        )

    output_snapshot = output_dir / "market_snapshot.csv.gz"
    expanded.to_csv(output_snapshot, index=False, compression="gzip")
    final_hash = base._snapshot_hash(expanded)

    manifest = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "base_snapshot": str(base_snapshot_path),
        "base_snapshot_hash": existing_hash,
        "preserved_existing_snapshot_hash": preserved_hash,
        "final_snapshot_hash": final_hash,
        "history_start": history_start.date().isoformat(),
        "snapshot_end": snapshot_end.date().isoformat(),
        "existing_symbol_count": len(existing_symbols),
        "required_symbol_count": len(required),
        "universe_symbols": universe_symbols,
        "candidate_symbols": candidate_symbols,
        "added_symbols": missing_symbols,
        "preserved_existing_histories": True,
        "data_source_for_added_symbols": "Yahoo Finance via yfinance, auto_adjust=True, threads=False",
        "output_snapshot": str(output_snapshot),
    }
    _write_json(output_dir / "snapshot_expansion_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
