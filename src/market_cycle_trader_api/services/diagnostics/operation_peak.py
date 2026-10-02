from __future__ import annotations

import io
import math
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
POST_EXIT_HORIZONS = (5, 10, 20)

OPERATION_PEAK_FIELDS = (
    "sequence",
    "asset",
    "walk_forward_fold",
    "reason",
    "rotation_id",
    "action",
    "entry_timestamp",
    "exit_timestamp",
    "entry_price",
    "exit_price",
    "position_return_pct",
    "holding_bars",
    "peak_price_while_held",
    "peak_timestamp_while_held",
    "peak_source",
    "trough_price_while_held",
    "trough_timestamp_while_held",
    "exit_distance_from_peak_pct",
    "exit_peak_proximity_pct",
    "peak_capture_pct",
    "max_runup_pct",
    "max_drawdown_from_entry_pct",
    "sessions_from_peak_to_exit",
    "calendar_days_from_peak_to_exit",
    "post_exit_peak_5d_pct",
    "post_exit_peak_5d_timestamp",
    "post_exit_available_sessions_5d",
    "post_exit_peak_10d_pct",
    "post_exit_peak_10d_timestamp",
    "post_exit_available_sessions_10d",
    "post_exit_peak_20d_pct",
    "post_exit_peak_20d_timestamp",
    "post_exit_available_sessions_20d",
)


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _frame(frames: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame | None:
    source = frames.get(str(symbol or "").strip().upper())
    if source is None or source.empty or not isinstance(source.index, pd.DatetimeIndex):
        return None
    result = source.copy()
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def _series_extreme(
    series: pd.Series,
    *,
    find_max: bool,
) -> tuple[float | None, pd.Timestamp | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None, None
    index = clean.idxmax() if find_max else clean.idxmin()
    return float(clean.loc[index]), pd.Timestamp(index)


def _sessions_between(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    return int(((index > start) & (index <= end)).sum())


def _operation_metrics(
    row: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    symbol = str(row.get("asset") or "").strip().upper()
    action = str(row.get("action") or "").strip().upper()
    frame = _frame(frames, symbol)
    entry_at = _timestamp(row.get("entry_timestamp"))
    exit_at = _timestamp(row.get("timestamp"))
    entry_price = _number(row.get("entry_price"))
    exit_price = _number(row.get("execution_price"))
    if (
        frame is None
        or entry_at is None
        or exit_at is None
        or entry_price is None
        or exit_price is None
        or entry_price <= 0.0
        or exit_price <= 0.0
        or exit_at < entry_at
    ):
        return {}

    # Normal rotations execute at the exit-session open. Therefore the session's
    # later high/low is not part of the held position. FINAL_SELL liquidates at
    # the final close, so its full final session legitimately belongs to holding.
    if action == "FINAL_SELL":
        held = frame.loc[(frame.index >= entry_at) & (frame.index <= exit_at)]
    else:
        held = frame.loc[(frame.index >= entry_at) & (frame.index < exit_at)]

    high, high_at = _series_extreme(held.get("high", pd.Series(dtype=float)), find_max=True)
    low, low_at = _series_extreme(held.get("low", pd.Series(dtype=float)), find_max=False)
    peak_source = "session_high" if high is not None else None

    # For a normal SELL the execution happens at the exit open. It can itself be
    # the best price obtained while the position was still owned.
    if action != "FINAL_SELL":
        if high is None or exit_price > high:
            high = float(exit_price)
            high_at = exit_at
            peak_source = "exit_execution"
        if low is None or exit_price < low:
            low = float(exit_price)
            low_at = exit_at

    # Defensive fallback for sparse data.
    if high is None:
        high = max(entry_price, exit_price)
        high_at = exit_at if exit_price >= entry_price else entry_at
        peak_source = "execution_fallback"
    if low is None:
        low = min(entry_price, exit_price)
        low_at = exit_at if exit_price <= entry_price else entry_at

    distance_pct = 100.0 * (exit_price / high - 1.0) if high > 0.0 else None
    proximity_pct = 100.0 * exit_price / high if high > 0.0 else None
    runup_pct = 100.0 * (high / entry_price - 1.0)
    drawdown_pct = 100.0 * (low / entry_price - 1.0)
    capture_pct = (
        100.0 * (exit_price - entry_price) / (high - entry_price)
        if high > entry_price
        else None
    )

    result: dict[str, Any] = {
        "operation_peak_analysis_schema_version": SCHEMA_VERSION,
        "peak_price_while_held": high,
        "peak_timestamp_while_held": high_at,
        "peak_source": peak_source,
        "trough_price_while_held": low,
        "trough_timestamp_while_held": low_at,
        "exit_distance_from_peak_pct": distance_pct,
        "exit_peak_proximity_pct": proximity_pct,
        "peak_capture_pct": capture_pct,
        "max_runup_pct": runup_pct,
        "max_drawdown_from_entry_pct": drawdown_pct,
        "sessions_from_peak_to_exit": (
            _sessions_between(frame.index, high_at, exit_at)
            if high_at is not None
            else None
        ),
        "calendar_days_from_peak_to_exit": (
            int((exit_at.normalize() - high_at.normalize()).days)
            if high_at is not None
            else None
        ),
    }

    # A normal SELL happens at the open, so the remainder of that same session
    # is already missed opportunity. FINAL_SELL happens at the close; only later
    # sessions are post-exit.
    post_exit = (
        frame.loc[frame.index > exit_at]
        if action == "FINAL_SELL"
        else frame.loc[frame.index >= exit_at]
    )
    for horizon in POST_EXIT_HORIZONS:
        window = post_exit.iloc[:horizon]
        future_peak, future_peak_at = _series_extreme(
            window.get("high", pd.Series(dtype=float)),
            find_max=True,
        )
        result[f"post_exit_available_sessions_{horizon}d"] = int(len(window))
        result[f"post_exit_peak_{horizon}d_pct"] = (
            100.0 * (future_peak / exit_price - 1.0)
            if future_peak is not None and exit_price > 0.0
            else None
        )
        result[f"post_exit_peak_{horizon}d_timestamp"] = future_peak_at

    return result


def enrich_operation_peak_trades(
    trades: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add hindsight-only peak diagnostics without affecting trading decisions."""
    if trades is None or trades.empty:
        return trades.copy() if isinstance(trades, pd.DataFrame) else pd.DataFrame()

    rows = trades.to_dict(orient="records")
    for row in rows:
        if str(row.get("action") or "").upper() not in {"SELL", "FINAL_SELL"}:
            continue
        row.update(_operation_metrics(row, frames))

    result = pd.DataFrame(rows)
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    if "entry_timestamp" in result.columns:
        result["entry_timestamp"] = pd.to_datetime(
            result["entry_timestamp"], utc=True, errors="coerce"
        )
    return result


def operation_peak_rows(
    trades: pd.DataFrame | Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(trades, pd.DataFrame):
        source = trades.to_dict(orient="records")
    else:
        source = [dict(row) for row in trades]

    output: list[dict[str, Any]] = []
    for row in source:
        if str(row.get("action") or "").upper() not in {"SELL", "FINAL_SELL"}:
            continue
        if row.get("operation_peak_analysis_schema_version") is None:
            continue
        position_return = _number(row.get("position_return"))
        normalized = {
            "sequence": row.get("sequence"),
            "asset": str(row.get("asset") or "").strip().upper(),
            "walk_forward_fold": row.get("walk_forward_fold"),
            "reason": row.get("reason"),
            "rotation_id": row.get("rotation_id"),
            "action": row.get("action"),
            "entry_timestamp": row.get("entry_timestamp"),
            "exit_timestamp": row.get("timestamp"),
            "entry_price": _number(row.get("entry_price")),
            "exit_price": _number(row.get("execution_price")),
            "position_return_pct": (
                100.0 * position_return if position_return is not None else None
            ),
            "holding_bars": row.get("holding_bars"),
        }
        for field in OPERATION_PEAK_FIELDS:
            if field not in normalized:
                normalized[field] = row.get(field)
        output.append(normalized)
    return output


def _stat(values: pd.Series, name: str) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    if name == "mean":
        return float(clean.mean())
    if name == "median":
        return float(clean.median())
    if name == "p25":
        return float(clean.quantile(0.25))
    if name == "p75":
        return float(clean.quantile(0.75))
    raise ValueError(name)


def operation_peak_summary(
    trades_or_rows: pd.DataFrame | Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(trades_or_rows, pd.DataFrame):
        rows = operation_peak_rows(trades_or_rows)
    else:
        candidate = [dict(row) for row in trades_or_rows]
        if candidate and "exit_timestamp" in candidate[0]:
            rows = candidate
        else:
            rows = operation_peak_rows(candidate)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []

    summaries: list[dict[str, Any]] = []
    for asset, group in frame.groupby("asset", sort=True):
        distance = pd.to_numeric(
            group["exit_distance_from_peak_pct"], errors="coerce"
        ).dropna()
        summaries.append(
            {
                "asset": asset,
                "n_operations": int(len(group)),
                "mean_exit_distance_from_peak_pct": _stat(
                    group["exit_distance_from_peak_pct"], "mean"
                ),
                "median_exit_distance_from_peak_pct": _stat(
                    group["exit_distance_from_peak_pct"], "median"
                ),
                "p25_exit_distance_from_peak_pct": _stat(
                    group["exit_distance_from_peak_pct"], "p25"
                ),
                "p75_exit_distance_from_peak_pct": _stat(
                    group["exit_distance_from_peak_pct"], "p75"
                ),
                "median_exit_peak_proximity_pct": _stat(
                    group["exit_peak_proximity_pct"], "median"
                ),
                "mean_peak_capture_pct": _stat(group["peak_capture_pct"], "mean"),
                "median_peak_capture_pct": _stat(
                    group["peak_capture_pct"], "median"
                ),
                "median_max_runup_pct": _stat(group["max_runup_pct"], "median"),
                "median_max_drawdown_from_entry_pct": _stat(
                    group["max_drawdown_from_entry_pct"], "median"
                ),
                "median_sessions_from_peak_to_exit": _stat(
                    group["sessions_from_peak_to_exit"], "median"
                ),
                "median_post_exit_peak_5d_pct": _stat(
                    group["post_exit_peak_5d_pct"], "median"
                ),
                "median_post_exit_peak_10d_pct": _stat(
                    group["post_exit_peak_10d_pct"], "median"
                ),
                "median_post_exit_peak_20d_pct": _stat(
                    group["post_exit_peak_20d_pct"], "median"
                ),
                "exit_within_1pct_of_peak_rate": (
                    float((distance.abs() <= 1.0).mean()) if not distance.empty else None
                ),
                "exit_within_2pct_of_peak_rate": (
                    float((distance.abs() <= 2.0).mean()) if not distance.empty else None
                ),
                "exit_within_5pct_of_peak_rate": (
                    float((distance.abs() <= 5.0).mean()) if not distance.empty else None
                ),
                "exit_within_10pct_of_peak_rate": (
                    float((distance.abs() <= 10.0).mean()) if not distance.empty else None
                ),
            }
        )
    return summaries


def operation_peak_metadata() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": "hindsight diagnostics only; never used by model or policy",
        "normal_sell_peak_window": (
            "entry session through the session before exit, plus the exit execution price "
            "at the exit-session open"
        ),
        "final_sell_peak_window": (
            "entry session through the final liquidation session inclusive because "
            "FINAL_SELL executes at the final close"
        ),
        "post_exit_window": (
            "normal SELL includes the exit session because the position was sold at its open; "
            "FINAL_SELL starts on the next session"
        ),
        "peak_capture_definition": (
            "100 * (exit_price - entry_price) / "
            "(peak_price_while_held - entry_price); null when no positive run-up occurred"
        ),
        "exit_distance_definition": (
            "100 * (exit_price / peak_price_while_held - 1); 0% means exit at the peak"
        ),
        "post_exit_horizons_sessions": list(POST_EXIT_HORIZONS),
    }


def _chunks(values: list[str], page_size: int) -> list[list[str]]:
    return [
        values[index : index + page_size]
        for index in range(0, len(values), page_size)
    ]


def peak_distance_boxplot_figures(
    operation_rows: list[dict[str, Any]],
    *,
    page_size: int = 20,
) -> list[Any]:
    frame = pd.DataFrame(operation_rows)
    if frame.empty:
        return []
    frame["exit_distance_from_peak_pct"] = pd.to_numeric(
        frame["exit_distance_from_peak_pct"], errors="coerce"
    )
    frame = frame.dropna(subset=["asset", "exit_distance_from_peak_pct"])
    if frame.empty:
        return []

    order = (
        frame.groupby("asset")["exit_distance_from_peak_pct"]
        .median()
        .sort_values()
        .index.tolist()
    )
    figures: list[Any] = []
    pages = _chunks(order, max(1, int(page_size)))
    for page_number, assets in enumerate(pages, start=1):
        data = [
            frame.loc[
                frame["asset"] == asset,
                "exit_distance_from_peak_pct",
            ].to_numpy(dtype=float)
            for asset in assets
        ]
        figure, axis = plt.subplots(
            figsize=(12, max(6.0, 0.42 * len(assets) + 1.5))
        )
        axis.boxplot(data, vert=False, labels=assets, showfliers=False)
        axis.axvline(0.0, linewidth=1.0)
        axis.set_xlabel("Distância da saída ao topo (%) — 0% = saída no topo")
        axis.set_title(
            f"Distância ao topo por ativo — página {page_number}/{len(pages)}"
        )
        axis.grid(True, axis="x", alpha=0.25)
        figure.tight_layout()
        figures.append(figure)
    return figures


def metric_bar_figures(
    summary_rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    xlabel: str,
    page_size: int = 20,
    ascending: bool = True,
) -> list[Any]:
    frame = pd.DataFrame(summary_rows)
    if frame.empty or metric not in frame:
        return []
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame.dropna(subset=["asset", metric]).sort_values(
        metric, ascending=ascending
    )
    if frame.empty:
        return []

    figures: list[Any] = []
    records = frame.to_dict(orient="records")
    pages = [
        records[index : index + max(1, int(page_size))]
        for index in range(0, len(records), max(1, int(page_size)))
    ]
    for page_number, page in enumerate(pages, start=1):
        labels = [str(row["asset"]) for row in page]
        values = [float(row[metric]) for row in page]
        figure, axis = plt.subplots(
            figsize=(12, max(6.0, 0.42 * len(labels) + 1.5))
        )
        axis.barh(labels, values)
        axis.set_xlabel(xlabel)
        axis.set_title(f"{title} — página {page_number}/{len(pages)}")
        axis.grid(True, axis="x", alpha=0.25)
        figure.tight_layout()
        figures.append(figure)
    return figures


def distance_vs_post_exit_scatter_figure(
    summary_rows: list[dict[str, Any]],
    *,
    top_n: int = 25,
) -> Any | None:
    frame = pd.DataFrame(summary_rows)
    required = {
        "asset",
        "n_operations",
        "median_exit_distance_from_peak_pct",
        "median_post_exit_peak_10d_pct",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None

    for field in (
        "n_operations",
        "median_exit_distance_from_peak_pct",
        "median_post_exit_peak_10d_pct",
    ):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame = (
        frame.dropna(
            subset=[
                "asset",
                "n_operations",
                "median_exit_distance_from_peak_pct",
                "median_post_exit_peak_10d_pct",
            ]
        )
        .sort_values("n_operations", ascending=False)
        .head(max(1, int(top_n)))
    )
    if frame.empty:
        return None

    sizes = 35.0 + 7.0 * frame["n_operations"].clip(lower=1.0)
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.scatter(
        frame["median_exit_distance_from_peak_pct"],
        frame["median_post_exit_peak_10d_pct"],
        s=sizes,
        alpha=0.7,
    )
    for _, row in frame.iterrows():
        axis.annotate(
            str(row["asset"]),
            (
                float(row["median_exit_distance_from_peak_pct"]),
                float(row["median_post_exit_peak_10d_pct"]),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(0.0, linewidth=1.0)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xlabel("Mediana da distância da saída ao topo (%)")
    axis.set_ylabel("Mediana da alta máxima nos 10 pregões após a saída (%)")
    axis.set_title(
        f"Proximidade do topo vs alta após a saída — {len(frame)} ativos mais operados"
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure


def figure_png_bytes(figure: Any) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()
