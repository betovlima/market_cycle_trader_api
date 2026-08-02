from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment  # noqa: E402

load_project_environment()

from market_cycle_trader_api.engine.market_data import (  # noqa: E402
    load_market_bars,
    validate_and_clean_bars,
)
from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (  # noqa: E402
    create_client,
    get_database,
    get_settings,
)
from market_cycle_trader_api.schemas.requests import BacktestRequest  # noqa: E402


def main() -> int:
    client = create_client()
    try:
        db = get_database(client)
        config = BacktestRequest.model_validate(get_settings(db))

        results: list[dict[str, object]] = []
        failures: list[dict[str, str]] = []
        for symbol in config.assets:
            try:
                frame = validate_and_clean_bars(load_market_bars(symbol, config), config)
                provenance = dict(frame.attrs.get("market_data_provenance", {}))
                results.append(
                    {
                        "symbol": symbol,
                        "rows": int(len(frame)),
                        "first_timestamp": frame.index.min().isoformat(),
                        "last_timestamp": frame.index.max().isoformat(),
                        "history_complete": bool(provenance.get("history_complete")),
                        "provider": provenance.get("provider")
                        or provenance.get("effective_provider"),
                        "historical_feed": provenance.get("historical_feed"),
                        "live_feed": provenance.get("live_feed"),
                        "adjustment": provenance.get("adjustment"),
                        "history_backfill_provider": provenance.get(
                            "history_backfill_provider"
                        ),
                        "history_backfill_rows": int(
                            provenance.get("history_backfill_rows") or 0
                        ),
                    }
                )
            except Exception as exc:
                failures.append({"symbol": symbol, "error": str(exc)})

        payload = {
            "mode": "active_strategy_market_history_bootstrap",
            "configuration_source": "backtest_settings/default",
            "requested_start": config.start_date,
            "provider": config.market_data_provider,
            "historical_feed": config.alpaca_historical_feed,
            "live_feed": config.alpaca_live_feed,
            "history_backfill_enabled": config.market_data_history_backfill_enabled,
            "history_backfill_provider": config.market_data_history_backfill_provider,
            "results": results,
            "failures": failures,
            "complete": not failures and len(results) == len(config.assets),
        }
        print(json.dumps(payload, indent=2, default=str))
        if failures:
            print(
                "Market-history bootstrap failed for the active API-managed strategy.",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
