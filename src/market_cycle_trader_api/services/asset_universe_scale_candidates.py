from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..infrastructure.market_data.alpaca import download_stock_bars
from ..infrastructure.persistence.mongo_repository import get_alpaca_credentials
from . import asset_discovery_market as market
from .asset_discovery_behavior import behavior_risk_profile

ASSET_DISCOVERY_SETTINGS_COLLECTION = "asset_discovery_settings"

# Only the settings consumed by this research eligibility check.
DEFAULT_ELIGIBILITY_SETTINGS: dict[str, Any] = {
    "min_price": 5.0,
    "min_median_dollar_volume": 5_000_000.0,
    "min_nonzero_volume_ratio": 0.95,
    "behavior_lookback_sessions": 756,
    "behavior_min_sessions": 63,
    "behavior_max_downside_tail_1pct": 0.12,
    "behavior_max_gap_downside_tail_1pct": 0.10,
    "behavior_max_annualized_volatility": 0.90,
    "behavior_max_drawdown": 0.75,
    "behavior_max_single_day_loss": 0.30,
    "behavior_max_single_gap_loss": 0.25,
    "behavior_max_10_session_loss": 0.35,
}


class CandidateUniverseLoader:
    """Prepare external research assets without writing their history to MongoDB."""

    def __init__(
        self,
        *,
        db: Any,
        collection: Any,
        configuration: dict[str, Any],
        configuration_hash: str,
        identity: dict[str, str],
        history_start: pd.Timestamp,
        snapshot_end: pd.Timestamp,
        expected_sessions: pd.DatetimeIndex,
        baseline_assets: list[str],
        workers: int,
        max_candidate_scans: int,
        log: Callable[[str], None],
    ) -> None:
        self.db = db
        self.collection = collection
        self.configuration = configuration
        self.configuration_hash = configuration_hash
        self.identity = identity
        self.history_start = history_start
        self.snapshot_end = snapshot_end
        self.expected_sessions = expected_sessions
        self.baseline_set = set(baseline_assets)
        self.workers = max(1, workers)
        self.max_candidate_scans = max(1, max_candidate_scans)
        self.log = log
        self.settings = self._settings()

    def prepare(
        self,
        required_external: int,
        *,
        seed_file: Path | None,
    ) -> tuple[dict[str, pd.DataFrame], list[str], list[dict[str, Any]], int]:
        if required_external <= 0:
            self.log(
                "Candidate preparation skipped: the requested universe is the Strategy control only."
            )
            return {}, [], [], 0

        seed = [s for s in self._seed_symbols(seed_file) if s not in self.baseline_set]
        if seed:
            self.log(
                f"Candidate seed: {len(seed)} prior full-history symbols will be revalidated; "
                "no previous ranking is read."
            )

        discovered = market.discover_alpaca_symbols()
        seed_set = set(seed)
        remaining = [
            s for s in discovered if s not in self.baseline_set and s not in seed_set
        ]
        remaining.sort(key=self._priority)
        order = list(dict.fromkeys([*seed, *remaining]))
        scan_limit = min(
            len(order),
            max(required_external, self.max_candidate_scans),
        )
        credentials = get_alpaca_credentials(self.db)

        accepted_frames: dict[str, pd.DataFrame] = {}
        accepted_symbols: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        scanned = 0

        while len(accepted_symbols) < required_external:
            if scanned >= scan_limit:
                if scan_limit >= len(order):
                    break
                previous_limit = scan_limit
                scan_limit = len(order)
                self.log(
                    f"Initial candidate scan budget {previous_limit} exhausted with "
                    f"{len(accepted_symbols)}/{required_external} qualified. "
                    "Continuing through the remaining deterministic candidate universe instead of aborting."
                )

            batch = order[scanned : min(scan_limit, scanned + self.workers)]
            if not batch:
                break
            scanned += len(batch)

            completed: dict[str, tuple[pd.DataFrame | None, dict[str, Any]]] = {}
            with ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="universe-history",
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
                if frame is not None and row.get("accepted"):
                    accepted_frames[symbol] = frame
                    accepted_symbols.append(symbol)
                    self.log(
                        f"Candidate accepted {len(accepted_symbols)}/{required_external} - "
                        f"{symbol}: history={row.get('observed_rows')}, "
                        f"median$vol={float(row.get('median_dollar_volume_63d') or 0):,.0f}, "
                        f"source={row.get('data_source')}."
                    )
                else:
                    reason = (
                        row.get("failed_quality_checks")
                        or row.get("error")
                        or "incomplete_history"
                    )
                    self.log(f"Candidate rejected - {symbol}: {reason}.")
                if len(accepted_symbols) >= required_external:
                    break

        if len(accepted_symbols) < required_external:
            raise RuntimeError(
                f"Could only qualify {len(accepted_symbols)} external assets after "
                f"scanning the available deterministic universe ({scanned} symbols); "
                f"{required_external} are required."
            )
        return accepted_frames, accepted_symbols, diagnostics, scanned

    def _settings(self) -> dict[str, Any]:
        document = self.db[ASSET_DISCOVERY_SETTINGS_COLLECTION].find_one(
            {"_id": "default"}
        ) or {}
        stored = (
            document.get("settings")
            if isinstance(document.get("settings"), dict)
            else {}
        )
        return {**DEFAULT_ELIGIBILITY_SETTINGS, **stored}

    def _seed_symbols(self, path: Path | None) -> list[str]:
        if path is None or not path.exists():
            return []
        frame = pd.read_csv(
            path,
            usecols=lambda name: name in {"symbol", "source", "history_complete"},
        )
        required = {"symbol", "source", "history_complete"}
        if not required.issubset(frame.columns):
            raise RuntimeError(
                "Seed universe must contain symbol, source and history_complete. "
                "Qualification/ranking columns are intentionally ignored."
            )
        complete = (
            frame["history_complete"]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )
        rows = frame.loc[
            (frame["source"].astype(str).str.lower() == "candidate") & complete,
            "symbol",
        ]
        return sorted({str(v).strip().upper() for v in rows if str(v).strip()})

    def _priority(self, symbol: str) -> str:
        material = (
            f"{self.configuration_hash}|"
            f"{self.snapshot_end.date().isoformat()}|{symbol.upper()}"
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def _evaluate(
        self,
        symbol: str,
        credentials: dict[str, str],
    ) -> tuple[str, pd.DataFrame | None, dict[str, Any]]:
        started = time.perf_counter()
        source = "mongo_read_only"
        try:
            frame = self._cached_frame(symbol)
            history = self._history(symbol, frame)
            if not history["history_complete"]:
                source = "alpaca_ram_only"
                frame = self._download(symbol, credentials)
                history = self._history(symbol, frame)

            quality = (
                self._quality(frame)
                if history["history_complete"]
                else {
                    "market_quality_passed": False,
                    "failed_quality_checks": ["full_strategy_history"],
                }
            )
            accepted = bool(
                history["history_complete"] and quality["market_quality_passed"]
            )
            row = {
                **history,
                **quality,
                "source": "candidate",
                "accepted": accepted,
                "data_source": source,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            return symbol, frame if accepted else None, row
        except Exception as exc:
            return symbol, None, {
                "symbol": symbol,
                "source": "candidate",
                "history_complete": False,
                "market_quality_passed": False,
                "accepted": False,
                "data_source": source,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": float(time.perf_counter() - started),
            }

    def _cached_frame(self, symbol: str) -> pd.DataFrame:
        start = self.history_start.tz_localize("UTC").to_pydatetime()
        end = (self.snapshot_end + pd.Timedelta(days=1)).tz_localize(
            "UTC"
        ).to_pydatetime()
        rows = list(
            self.collection.find(
                {
                    "symbol": symbol,
                    **self.identity,
                    "timestamp": {"$gte": start, "$lt": end},
                },
                {
                    "_id": 0,
                    "timestamp": 1,
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "volume": 1,
                    "vwap": 1,
                    "trade_count": 1,
                },
            ).sort("timestamp", 1)
        )
        return self._clean(rows)

    def _download(
        self,
        symbol: str,
        credentials: dict[str, str],
    ) -> pd.DataFrame:
        frame = download_stock_bars(
            api_key_id=credentials["api_key_id"],
            secret_key=credentials["secret_key"],
            symbol=symbol,
            timeframe=str(self.configuration.get("timeframe") or "1Day"),
            start=self.history_start.tz_localize("UTC").to_pydatetime(),
            end=(self.snapshot_end + pd.Timedelta(days=1))
            .tz_localize("UTC")
            .to_pydatetime(),
            feed=str(self.configuration.get("alpaca_historical_feed") or "sip"),
            adjustment=str(self.configuration.get("alpaca_adjustment") or "all"),
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        result = frame.copy()
        result.index = pd.to_datetime(result.index, utc=True)
        result = result[~result.index.duplicated(keep="last")].sort_index()
        return self._clean(result.reset_index().to_dict(orient="records"))

    @staticmethod
    def _clean(rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        timestamp = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
        frame[timestamp] = pd.to_datetime(frame[timestamp], utc=True)
        frame = frame.set_index(timestamp).sort_index()
        required = ["open", "high", "low", "close", "volume"]
        if any(column not in frame.columns for column in required):
            return pd.DataFrame()
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        frame = frame[(frame[["open", "high", "low", "close"]] > 0).all(axis=1)]
        frame = frame[frame["volume"] >= 0]
        return frame[~frame.index.duplicated(keep="last")]

    def _history(self, symbol: str, frame: pd.DataFrame) -> dict[str, Any]:
        if frame is None or frame.empty:
            return {
                "symbol": symbol,
                "history_complete": False,
                "observed_rows": 0,
                "expected_sessions": int(len(self.expected_sessions)),
                "missing_sessions": int(len(self.expected_sessions)),
                "actual_start": None,
                "actual_end": None,
            }
        observed = pd.DatetimeIndex(
            pd.to_datetime(frame.index, utc=True).normalize().tz_localize(None)
        ).unique()
        missing = self.expected_sessions.difference(observed)
        return {
            "symbol": symbol,
            "history_complete": bool(len(missing) == 0),
            "observed_rows": int(len(frame)),
            "expected_sessions": int(len(self.expected_sessions)),
            "missing_sessions": int(len(missing)),
            "actual_start": observed.min().date().isoformat(),
            "actual_end": observed.max().date().isoformat(),
            "missing_sample": ",".join(
                item.date().isoformat() for item in missing[:5]
            ),
        }

    def _quality(self, frame: pd.DataFrame) -> dict[str, Any]:
        recent = frame.tail(min(63, len(frame)))
        close = pd.to_numeric(recent.get("close"), errors="coerce")
        volume = pd.to_numeric(recent.get("volume"), errors="coerce")
        latest_close = (
            float(close.dropna().iloc[-1]) if not close.dropna().empty else float("nan")
        )
        dollar_volume = (close * volume).dropna()
        median_dollar_volume = (
            float(dollar_volume.median()) if not dollar_volume.empty else 0.0
        )
        nonzero_volume_ratio = float((volume > 0).mean()) if len(volume) else 0.0
        behavior = behavior_risk_profile(frame, self.settings)
        checks = {
            "price_ready": bool(
                np.isfinite(latest_close)
                and latest_close >= float(self.settings["min_price"])
            ),
            "liquidity_ready": bool(
                median_dollar_volume
                >= float(self.settings["min_median_dollar_volume"])
            ),
            "volume_quality_ready": bool(
                nonzero_volume_ratio
                >= float(self.settings["min_nonzero_volume_ratio"])
            ),
            "behavior_ready": bool(
                not behavior.get("sample_ready") or behavior.get("passed")
            ),
        }
        return {
            "market_quality_passed": bool(all(checks.values())),
            "latest_close": latest_close,
            "median_dollar_volume_63d": median_dollar_volume,
            "nonzero_volume_ratio": nonzero_volume_ratio,
            "behavior_passed": behavior.get("passed"),
            "behavior_reason_codes": list(behavior.get("reason_codes") or []),
            "failed_quality_checks": [
                name for name, passed in checks.items() if not passed
            ],
        }
