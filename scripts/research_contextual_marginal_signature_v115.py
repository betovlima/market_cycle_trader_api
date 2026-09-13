from __future__ import annotations

import argparse
import json
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import research_contextual_marginal_signature_v103 as v103
import research_contextual_marginal_signature_v113 as v113

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.15"
EXPERIMENT_NAME = "contextual_marginal_signature_predictive_screen"
TOLERANCE = 1e-12
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
STATIC_FEATURES = tuple(v103.MODEL_FEATURES)
LEVEL2_FEATURES = tuple(
    f"logsig2__{v113.CHANNELS[i]}__{v113.CHANNELS[j]}"
    for i, j in combinations(range(len(v113.CHANNELS)), 2)
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chronological M0 versus M0+LogSig2 predictive screen.")
    p.add_argument("--frozen-dir", required=True)
    p.add_argument("--window-sessions", type=int, default=60)
    p.add_argument("--validation-start", default="2026-01-01")
    p.add_argument("--market-proxy", default="SPY")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fresh-run", action="store_true")
    return p


def _fresh(path: Path) -> None:
    resolved = path.resolve()
    if "research_output" not in resolved.parts or not resolved.name.startswith("contextual_marginal_signature_"):
        raise RuntimeError(f"Refusing to delete unexpected output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _spearman(frame: pd.DataFrame, pred: str) -> float | None:
    if frame["delta_log_capital"].nunique() < 2 or frame[pred].nunique() < 2:
        return None
    value = frame["delta_log_capital"].rank().corr(frame[pred].rank())
    return None if pd.isna(value) else float(value)


def _metrics(frame: pd.DataFrame, pred: str) -> dict[str, Any]:
    contexts = []
    for (date, universe), group in frame.groupby(["decision_date", "universe_name"], sort=True):
        top = group.loc[group[pred].idxmax()]
        target = float(top["delta_log_capital"])
        contexts.append({
            "decision_date": pd.Timestamp(date).date().isoformat(),
            "universe_name": universe,
            "spearman": _spearman(group, pred),
            "top1_candidate": str(top["candidate"]),
            "top1_direct_log": target,
            "top1_positive": target > TOLERANCE,
            "top1_negative": target < -TOLERANCE,
        })
    scores = [row["spearman"] for row in contexts if row["spearman"] is not None]
    return {
        "pooled_spearman": _spearman(frame, pred),
        "mean_context_spearman": float(np.mean(scores)) if scores else None,
        "mae": float(mean_absolute_error(frame["delta_log_capital"], frame[pred])),
        "rmse": float(mean_squared_error(frame["delta_log_capital"], frame[pred]) ** 0.5),
        "top1_positive_contexts": sum(row["top1_positive"] for row in contexts),
        "top1_negative_contexts": sum(row["top1_negative"] for row in contexts),
        "mean_top1_direct_log": float(np.mean([row["top1_direct_log"] for row in contexts])),
        "contexts": contexts,
    }


def _tune(development: pd.DataFrame, features: list[str]) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for alpha in ALPHAS:
        folds = []
        for date in sorted(development["decision_date"].unique()):
            train = development[development["decision_date"] != date]
            test = development[development["decision_date"] == date].copy()
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[features], train["delta_log_capital"])
            test["prediction"] = model.predict(test[features])
            folds.append(test)
        oof = pd.concat(folds, ignore_index=True)
        metrics = _metrics(oof, "prediction")
        rows.append({"alpha": alpha, **{k: metrics[k] for k in ("pooled_spearman", "mean_context_spearman", "mae", "rmse")}})
    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row["pooled_spearman"] if row["pooled_spearman"] is not None else -999.0),
            float(row["mean_context_spearman"] if row["mean_context_spearman"] is not None else -999.0),
            -float(row["mae"]),
        )
    return float(max(rows, key=key)["alpha"]), rows


def _load_dataset(frozen: Path, window: int, proxy: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((frozen / "trace_manifest.json").read_text(encoding="utf-8"))
    aggregate = pd.read_csv(frozen / "trace_aggregate_dataset.csv")
    aggregate["decision_date"] = pd.to_datetime(aggregate["decision_date"])
    aggregate["candidate"] = aggregate["candidate"].astype(str).str.upper()
    aggregate["delta_log_capital"] = pd.to_numeric(aggregate["delta_log_capital"], errors="raise")
    nonzero = aggregate[aggregate["delta_log_capital"].abs() > TOLERANCE]
    readiness = {
        "rows": len(nonzero) >= 12,
        "candidates": nonzero["candidate"].nunique() >= 3,
        "dates": nonzero["decision_date"].nunique() >= 3,
        "validation": bool((nonzero["decision_date"].dt.year >= 2026).any()),
    }
    if not all(readiness.values()):
        raise RuntimeError(f"Direct-effect dataset is not ready: {readiness}")

    universe_map = {item["name"]: [str(x).upper() for x in item["assets"]] for item in manifest["universes"]}
    snapshot = pd.read_csv(frozen / "market_snapshot.csv.gz")
    snapshot["timestamp"] = pd.to_datetime(snapshot["timestamp"], utc=True)
    snapshot["symbol"] = snapshot["symbol"].astype(str).str.upper()
    raw = {
        symbol: group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        for symbol, group in snapshot.groupby("symbol", sort=False)
    }
    configuration = json.loads((frozen / "strategy_configuration.json").read_text(encoding="utf-8"))
    config = v103.BacktestRequest.model_validate(configuration)
    rotation = {symbol: v103.rotation.build_rotation_frame(frame.copy(), config) for symbol, frame in raw.items()}
    close = v113._load_close_panel(frozen)

    rows = []
    for item in aggregate.to_dict(orient="records"):
        universe = str(item["universe_name"])
        candidate = str(item["candidate"])
        decision = pd.Timestamp(item["decision_date"])
        static = v103.build_feature_snapshot(
            candidate=candidate,
            seed_assets=universe_map[universe],
            rotation_frames=rotation,
            raw_frames=raw,
            decision=decision,
            market_proxy=proxy,
        )
        path, _ = v113._joint_path(
            close,
            candidate=candidate,
            universe_assets=universe_map[universe],
            decision_date=decision.date().isoformat(),
            window_sessions=window,
        )
        logsig = v113._level2_logsignature(path)
        rows.append({
            "decision_date": decision,
            "universe_name": universe,
            "candidate": candidate,
            "delta_log_capital": float(item["delta_log_capital"]),
            **{name: float(static[name]) for name in STATIC_FEATURES},
            **{name: float(logsig[name]) for name in LEVEL2_FEATURES},
        })
    return pd.DataFrame(rows), {"manifest": manifest, "readiness": readiness, "nonzero": len(nonzero)}


def _passes(m0: dict[str, Any], m1: dict[str, Any]) -> bool:
    keys = ("pooled_spearman", "mean_context_spearman")
    if any(m0[k] is None or m1[k] is None for k in keys):
        return False
    return bool(
        m1["pooled_spearman"] > 0.0
        and m1["pooled_spearman"] > m0["pooled_spearman"]
        and m1["mean_context_spearman"] > 0.0
        and m1["mean_context_spearman"] > m0["mean_context_spearman"]
        and m1["top1_negative_contexts"] <= m0["top1_negative_contexts"]
        and m1["top1_positive_contexts"] >= m0["top1_positive_contexts"]
        and m1["mean_top1_direct_log"] >= m0["mean_top1_direct_log"]
    )


def main() -> int:
    args = _parser().parse_args()
    frozen = Path(args.frozen_dir).resolve()
    output = Path(args.output_dir).resolve()
    if args.fresh_run:
        _fresh(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset, source = _load_dataset(frozen, int(args.window_sessions), str(args.market_proxy).upper())
    boundary = pd.Timestamp(args.validation_start)
    development = dataset[dataset["decision_date"] < boundary].copy()
    validation = dataset[dataset["decision_date"] >= boundary].copy()
    if development["decision_date"].nunique() < 3 or validation.empty:
        raise RuntimeError("Chronological split is too small.")

    model_features = {
        "M0_static_context": list(STATIC_FEATURES),
        "M1_static_plus_logsig2": [*STATIC_FEATURES, *LEVEL2_FEATURES],
    }
    summaries = {}
    predictions = validation[["decision_date", "universe_name", "candidate", "delta_log_capital"]].copy()
    cv_rows = []
    for name, features in model_features.items():
        alpha, cv = _tune(development, features)
        cv_rows.extend({"model": name, **row} for row in cv)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(development[features], development["delta_log_capital"])
        predictions[name] = model.predict(validation[features])
        summaries[name] = {"feature_count": len(features), "alpha": alpha, "validation": _metrics(predictions, name)}

    m0 = summaries["M0_static_context"]["validation"]
    m1 = summaries["M1_static_plus_logsig2"]["validation"]
    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "source_script_version": source["manifest"].get("script_version"),
        "market_snapshot_hash": source["manifest"].get("market_snapshot_hash"),
        "strategy_configuration_hash": source["manifest"].get("strategy_configuration_hash"),
        "window_sessions": int(args.window_sessions),
        "validation_start": boundary.date().isoformat(),
        "development_rows": len(development),
        "validation_rows": len(validation),
        "nonzero_direct_rows": source["nonzero"],
        "readiness": source["readiness"],
        "models": summaries,
        "incremental": {
            "pooled_spearman_gain": None if m0["pooled_spearman"] is None or m1["pooled_spearman"] is None else m1["pooled_spearman"] - m0["pooled_spearman"],
            "mean_context_spearman_gain": None if m0["mean_context_spearman"] is None or m1["mean_context_spearman"] is None else m1["mean_context_spearman"] - m0["mean_context_spearman"],
            "top1_positive_context_gain": m1["top1_positive_contexts"] - m0["top1_positive_contexts"],
            "top1_negative_context_gain": m1["top1_negative_contexts"] - m0["top1_negative_contexts"],
            "mean_top1_direct_log_gain": m1["mean_top1_direct_log"] - m0["mean_top1_direct_log"],
        },
        "incremental_path_signal_screen_passed": _passes(m0, m1),
        "decision_rule": "M1 must be positive and improve on M0 in pooled and within-context Spearman, without worsening Top-1 sign quality or mean Top-1 direct contribution. This is a screen, not an independent blinded replication.",
    }
    dataset.to_csv(output / "predictive_screen_dataset.csv", index=False)
    predictions.to_csv(output / "predictive_screen_predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output / "ridge_development_cv.csv", index=False)
    _write_json(output / "predictive_screen_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
