from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

import research_contextual_marginal_signature_v111 as base

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.14.2"
EXPERIMENT_NAME = "contextual_marginal_signature_alpaca_snapshot"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-sequence", type=int, default=10)
    parser.add_argument("--strategy-id", default=None)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--snapshot-end", required=True)
    parser.add_argument("--universe-spec", required=True)
    parser.add_argument("--cases-spec", required=True)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fresh-run", action="store_true")
    return parser


def _fresh(path: Path) -> None:
    root = (base.PROJECT_ROOT / "research_output").resolve()
    resolved = path.resolve()
    resolved.relative_to(root)
    if not resolved.name.startswith("contextual_marginal_signature_"):
        raise RuntimeError(f"Unexpected output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _campaign(path_universe: Path, path_cases: Path):
    universe_order, cases = base._load_cases(path_cases)
    universes = base._load_universe_spec(path_universe, universe_order)
    candidates = sorted({candidate for case in cases for candidate in case["candidates"]})
    symbols = base._normalize_symbols([s for u in universes for s in u["assets"]] + candidates)
    return symbols, universes, cases


def main() -> int:
    args = _parser().parse_args()
    base.load_project_environment(args.env_file)
    from market_cycle_trader_api.engine import market_data
    from market_cycle_trader_api.infrastructure.persistence import mongo_repository
    from market_cycle_trader_api.schemas.requests import BacktestRequest

    output = Path(args.output_dir).resolve()
    if args.fresh_run:
        _fresh(output)
    output.mkdir(parents=True, exist_ok=True)

    history_start = base._normalize_date(args.history_start)
    snapshot_end = base._normalize_date(args.snapshot_end)
    symbols, universes, cases = _campaign(Path(args.universe_spec), Path(args.cases_spec))

    client = mongo_repository.create_client()
    try:
        db = mongo_repository.get_database(client)
        strategy = base._strategy_document(db, args.strategy_sequence, args.strategy_id)
        stored = base._configuration(strategy)
        if base._normalize_date(stored.get("start_date")) != history_start:
            raise RuntimeError("--history-start must match the Strategy start date.")
        missing = sorted(set(symbols).difference(base._normalize_symbols(list(stored.get("assets") or []))))
        if missing:
            raise RuntimeError(f"Campaign symbols are not present in the Strategy: {missing}")

        config = BacktestRequest.model_validate(stored).model_copy(update={
            "end_date": snapshot_end.date().isoformat(),
            "research_market_data_mode": "backtest_bootstrap_missing",
            "mongo_cache_enabled": True,
            "market_data_require_complete_history": True,
        })
        frames = {}
        sources = []
        for index, symbol in enumerate(symbols, 1):
            base._log(f"[{index}/{len(symbols)}] Alpaca-origin market data: {symbol}")
            frame = market_data.validate_and_clean_bars(market_data.load_market_bars(symbol, config), config)
            frames[symbol] = frame
            provenance = dict(frame.attrs.get("market_data_provenance") or {})
            sources.append({
                "symbol": symbol,
                "rows": len(frame),
                "first": pd.Timestamp(frame.index.min()).isoformat(),
                "last": pd.Timestamp(frame.index.max()).isoformat(),
                "access": provenance.get("research_access_path"),
                "bootstrap_rows": int(provenance.get("cache_bootstrap_rows") or 0),
            })

        snapshot = base._snapshot_table(frames)
        snapshot_path = output / "market_snapshot.csv.gz"
        snapshot.to_csv(snapshot_path, index=False, compression="gzip")
        manifest = {
            "schema_version": 1,
            "script_version": SCRIPT_VERSION,
            "experiment": EXPERIMENT_NAME,
            "status": "completed",
            "strategy_sequence": int(strategy.get("strategy_sequence") or args.strategy_sequence),
            "history_start": history_start.date().isoformat(),
            "snapshot_end": snapshot_end.date().isoformat(),
            "market_snapshot_hash": base._snapshot_hash(snapshot),
            "market_snapshot_path": str(snapshot_path),
            "market_origin": "Alpaca",
            "timeframe": str(config.timeframe),
            "feed": str(config.alpaca_historical_feed),
            "adjustment": str(config.alpaca_adjustment),
            "universes": universes,
            "cases": cases,
            "symbols": sources,
        }
        _write_json(output / "alpaca_mongo_snapshot_manifest.json", manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
