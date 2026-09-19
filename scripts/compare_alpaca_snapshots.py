from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.core.environment import load_project_environment

load_project_environment()

from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
    ALPACA_MARKET_BARS_COLLECTION,
    JOBS_COLLECTION,
    create_client,
    get_database,
)


OHLCV = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
DEFAULT_FRESH_COLLECTION = "alpaca_market_bars_fresh_20260919"


def _latest_job(db: Any, job_id: str | None) -> dict[str, Any]:
    if job_id:
        job = db[JOBS_COLLECTION].find_one({"id": str(job_id)})
    else:
        job = db[JOBS_COLLECTION].find_one(
            {"internal_job": {"$ne": True}, "status": "completed"},
            sort=[("finished_at", -1), ("created_at", -1)],
        )
    if job is None or not isinstance(job.get("request"), dict):
        raise RuntimeError("No completed Backtest job with an immutable request snapshot was found.")
    return job


def _read(
    collection: Any,
    *,
    symbol: str,
    interval: str,
    feed: str,
    adjustment: str,
) -> pd.DataFrame:
    rows = list(
        collection.find(
            {
                "symbol": symbol,
                "interval": interval,
                "feed": feed,
                "adjustment": adjustment,
            },
            {"_id": 0, "timestamp": 1, **{column: 1 for column in OHLCV}},
        ).sort("timestamp", 1)
    )
    if not rows:
        return pd.DataFrame(columns=list(OHLCV))
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    frame.index = frame.index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in OHLCV:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=list(OHLCV))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the preserved Alpaca cache against a fresh Alpaca API snapshot."
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument(
        "--old-collection",
        default=ALPACA_MARKET_BARS_COLLECTION,
    )
    parser.add_argument(
        "--fresh-collection",
        default=DEFAULT_FRESH_COLLECTION,
    )
    parser.add_argument(
        "--output-dir",
        default="output/alpaca_old_vs_fresh",
    )
    args = parser.parse_args()

    if args.old_collection == args.fresh_collection:
        raise ValueError("Old and fresh collections must be different.")

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = create_client()
    try:
        db = get_database(client)
        job = _latest_job(db, args.job_id)
        request = job["request"]
        assets = [str(item).strip().upper() for item in request.get("assets", []) if str(item).strip()]
        timeframe = str(request.get("timeframe") or "1Day")
        feed = str(request.get("alpaca_historical_feed") or "sip").lower()
        adjustment = str(request.get("alpaca_adjustment") or "all").lower()

        old = db[str(args.old_collection)]
        fresh = db[str(args.fresh_collection)]

        summaries: list[dict[str, Any]] = []
        details: list[pd.DataFrame] = []

        for position, symbol in enumerate(assets, start=1):
            left = _read(old, symbol=symbol, interval=timeframe, feed=feed, adjustment=adjustment)
            right = _read(fresh, symbol=symbol, interval=timeframe, feed=feed, adjustment=adjustment)

            common = left.index.intersection(right.index).sort_values()
            only_old = left.index.difference(right.index)
            only_fresh = right.index.difference(left.index)

            row: dict[str, Any] = {
                "symbol": symbol,
                "old_rows": int(len(left)),
                "fresh_rows": int(len(right)),
                "common_rows": int(len(common)),
                "old_only_rows": int(len(only_old)),
                "fresh_only_rows": int(len(only_fresh)),
            }

            if len(common):
                l = left.loc[common, list(OHLCV)].astype(float)
                r = right.loc[common, list(OHLCV)].astype(float)
                exact = np.ones(len(common), dtype=bool)
                material = pd.Series(False, index=common)
                detail = pd.DataFrame(index=common)
                detail.index.name = "timestamp"
                detail["symbol"] = symbol

                for column in OHLCV:
                    exact &= np.equal(l[column].to_numpy(), r[column].to_numpy())
                    denominator = l[column].abs().replace(0.0, np.nan)
                    rel = ((r[column] - l[column]).abs() / denominator).replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    threshold = 1e-4 if column in PRICE_COLUMNS else 0.01
                    material = material | (rel.fillna(0.0) > threshold)
                    detail[f"old_{column}"] = l[column]
                    detail[f"fresh_{column}"] = r[column]
                    detail[f"{column}_rel_diff"] = rel
                    row[f"{column}_mean_rel_diff"] = (
                        float(rel.dropna().mean()) if rel.notna().any() else None
                    )
                    row[f"{column}_max_rel_diff"] = (
                        float(rel.dropna().max()) if rel.notna().any() else None
                    )

                row["exact_match_rows"] = int(exact.sum())
                row["exact_match_rate"] = float(exact.mean())
                row["material_difference_rows"] = int(material.sum())
                row["material_difference_rate"] = float(material.mean())
                detail["exact_match"] = exact
                detail["material_difference"] = material
                changed = detail[(~detail["exact_match"]) | detail["material_difference"]].reset_index()
                if not changed.empty:
                    details.append(changed)
            else:
                row["exact_match_rows"] = 0
                row["exact_match_rate"] = None
                row["material_difference_rows"] = 0
                row["material_difference_rate"] = None

            summaries.append(row)
            print(
                f"[compare-alpaca] {position}/{len(assets)} {symbol} "
                f"old={len(left)} fresh={len(right)} common={len(common)}",
                flush=True,
            )

        per_asset = pd.DataFrame(summaries)
        per_asset.to_csv(output_dir / "per_asset_equivalence.csv", index=False)

        changed = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
        changed.to_csv(output_dir / "changed_rows.csv", index=False)

        total_common = int(per_asset["common_rows"].sum())
        total_exact = int(per_asset["exact_match_rows"].sum())
        total_material = int(per_asset["material_difference_rows"].sum())
        summary = {
            "source_job_id": job.get("id"),
            "old_collection": str(args.old_collection),
            "fresh_collection": str(args.fresh_collection),
            "asset_count": len(assets),
            "timeframe": timeframe,
            "feed": feed,
            "adjustment": adjustment,
            "total_common_rows": total_common,
            "total_exact_match_rows": total_exact,
            "exact_match_rate": (total_exact / total_common if total_common else None),
            "total_material_difference_rows": total_material,
            "material_difference_rate": (
                total_material / total_common if total_common else None
            ),
            "assets_with_material_differences": int(
                (per_asset["material_difference_rows"] > 0).sum()
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("", flush=True)
        print("[compare-alpaca] completed", flush=True)
        print(f"[compare-alpaca] output={output_dir}", flush=True)
        print(
            f"[compare-alpaca] exact_match_rate="
            f"{(summary['exact_match_rate'] or 0.0):.6f}",
            flush=True,
        )
        print(
            f"[compare-alpaca] material_difference_rate="
            f"{(summary['material_difference_rate'] or 0.0):.6f}",
            flush=True,
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
