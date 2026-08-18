from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import qmc

from market_cycle_trader_api.services.temporal_cash_reentry_counterfactual import (
    DEFAULT_TIMING_SETTINGS,
    compile_absolute_opportunity_context,
    replay_compiled_absolute_opportunity_reentry_gate,
)
from market_cycle_trader_api.services.temporal_policy_replay import _timestamp_key


SEARCH_SPACE = (
    ("absolute_entry_threshold", 0.45, 0.75, "float"),
    ("absolute_exit_discount", 0.05, 0.30, "float"),
    ("cash_reentry_premium", 0.00, 0.10, "float"),
    ("minimum_risk_safety", 0.00, 0.30, "float"),
    ("minimum_horizon_agreement", 0.00, 0.75, "float"),
    ("reentry_confirmation_sessions", 1.0, 3.0, "int"),
)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict(orient="records")


def _observations_from_asset_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _timestamp_key(row.get("timestamp"))
        symbol = str(row.get("symbol") or "").strip()
        if not key or not symbol:
            continue
        payload = grouped.setdefault(key, {"fold_id": None, "rows_by_symbol": {}})
        payload["rows_by_symbol"][symbol] = dict(row)
        if payload["fold_id"] is None and row.get("fold_id") is not None:
            payload["fold_id"] = int(row["fold_id"])
    return grouped


def _timing_settings_from_summary(summary: pd.DataFrame | None) -> dict[str, float]:
    settings = dict(DEFAULT_TIMING_SETTINGS)
    if summary is None or summary.empty:
        return settings
    row = summary.iloc[0]
    for name in settings:
        if name in summary.columns and pd.notna(row.get(name)):
            settings[name] = float(row[name])
    return settings


def load_export_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        required = {
            "temporal_intelligence_multi_horizon_daily_assets.csv",
            "temporal_intelligence_winner_reference_daily.csv",
        }
        if not required.issubset(names):
            missing = ", ".join(sorted(required - names))
            raise ValueError(f"Temporal export ZIP is missing: {missing}")
        assets = pd.read_csv(io.BytesIO(archive.read("temporal_intelligence_multi_horizon_daily_assets.csv")))
        winner = pd.read_csv(io.BytesIO(archive.read("temporal_intelligence_winner_reference_daily.csv")))
        folds = pd.read_csv(io.BytesIO(archive.read("temporal_intelligence_multi_horizon_folds.csv"))) if "temporal_intelligence_multi_horizon_folds.csv" in names else pd.DataFrame()
        summary = pd.read_csv(io.BytesIO(archive.read("temporal_intelligence_multi_horizon.csv"))) if "temporal_intelligence_multi_horizon.csv" in names else None
        manifest = json.loads(archive.read("temporal_intelligence_manifest.json").decode("utf-8")) if "temporal_intelligence_manifest.json" in names else {}
    request = manifest.get("request") if isinstance(manifest.get("request"), dict) else {}
    initial_capital = float(request.get("initial_capital") or 10_000.0)
    one_side_cost = max(0.0, float(request.get("slippage_bps") or 0.0) / 10_000.0) + max(0.0, float(request.get("commission_rate") or 0.0))
    winner_fold_returns = {
        int(row.fold_id): float(row.winner_reference_return)
        for row in folds.itertuples()
        if hasattr(row, "fold_id") and hasattr(row, "winner_reference_return") and pd.notna(row.winner_reference_return)
    }
    run = manifest.get("run") if isinstance(manifest.get("run"), dict) else {}
    return {
        "source": str(path),
        "source_run_id": str(run.get("id") or run.get("run_id") or "export_zip"),
        "observations": _observations_from_asset_rows(_records(assets)),
        "winner_rows": _records(winner),
        "initial_capital": initial_capital,
        "one_side_cost": one_side_cost,
        "winner_fold_returns": winner_fold_returns,
        "timing_settings": _timing_settings_from_summary(summary),
    }


def load_mongo(source_run_id: str | None) -> dict[str, Any]:
    # Standalone research tools do not pass through the API application bootstrap.
    # Load the same project .env before importing mongo_repository, because that
    # module captures MONGO_URL / MONGO_DATABASE at import time.
    from market_cycle_trader_api.core.environment import load_project_environment

    loaded_env = load_project_environment()
    if loaded_env:
        print(f"Environment loaded from: {loaded_env[0]}")
    else:
        print("Warning: no project .env file was found; using process environment only.")

    # Lazy imports keep --export-zip usable in lightweight research environments.
    from market_cycle_trader_api.infrastructure.persistence.mongo_repository import (
        TEMPORAL_INTELLIGENCE_RUNS_COLLECTION,
        create_client,
        get_database,
    )
    from market_cycle_trader_api.services.temporal_policy_tuning import _load_artifact_rows, _load_observations

    client = create_client()
    try:
        db = get_database(client)
        if source_run_id:
            run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one({"id": str(source_run_id)})
        else:
            run = db[TEMPORAL_INTELLIGENCE_RUNS_COLLECTION].find_one(
                {"status": "completed"}, sort=[("finished_at", -1), ("created_at", -1)]
            )
        if run is None:
            raise ValueError("Completed Temporal Intelligence source run was not found in MongoDB.")
        run_id = str(run["id"])
        observations = _load_observations(db, run_id)
        winner_rows = _load_artifact_rows(db, run_id, "winner_reference_daily")
        if not observations or not winner_rows:
            raise ValueError("Frozen Temporal observations/winner reference are incomplete.")
        request = run.get("request") if isinstance(run.get("request"), dict) else {}
        result = run.get("result") if isinstance(run.get("result"), dict) else {}
        capital = result.get("multi_horizon_capital") if isinstance(result.get("multi_horizon_capital"), dict) else {}
        timing = dict(DEFAULT_TIMING_SETTINGS)
        for name in timing:
            if capital.get(name) is not None:
                timing[name] = float(capital[name])
        fold_rows = result.get("multi_horizon_fold_metrics") if isinstance(result.get("multi_horizon_fold_metrics"), list) else []
        winner_fold_returns = {
            int(item.get("fold_id")): float(item.get("winner_reference_return") or 0.0)
            for item in fold_rows if item.get("fold_id") is not None
        }
        return {
            "source": "mongo",
            "source_run_id": run_id,
            "observations": observations,
            "winner_rows": winner_rows,
            "initial_capital": float(request.get("initial_capital") or 10_000.0),
            "one_side_cost": max(0.0, float(request.get("slippage_bps") or 0.0) / 10_000.0) + max(0.0, float(request.get("commission_rate") or 0.0)),
            "winner_fold_returns": winner_fold_returns,
            "timing_settings": timing,
        }
    finally:
        client.close()


def settings_from_unit(unit: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (name, low, high, kind) in enumerate(SEARCH_SPACE):
        value = float(low) + float(unit[index]) * (float(high) - float(low))
        if kind == "int":
            result[name] = int(np.clip(round(value), int(low), int(high)))
        else:
            result[name] = round(value, 6)
    return result


def _candidate_summary(candidate_id: int, replay: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    metrics = replay["metrics"]
    control_metrics = control["metrics"]
    capital = float(metrics["ending_capital"])
    control_capital = float(control_metrics["ending_capital"])
    sharpe = float(metrics["sharpe"])
    maxdd = float(metrics["maximum_drawdown"])
    worst = metrics.get("worst_fold_return")
    robust = bool(metrics.get("eligible")) and sharpe >= float(control_metrics["sharpe"]) - 0.05 and maxdd >= float(control_metrics["maximum_drawdown"]) - 0.03
    gate = robust and capital >= control_capital * 1.03
    row = {
        "candidate_id": int(candidate_id),
        "robust": robust,
        "beats_control_3pct_gate": gate,
        "ending_capital": capital,
        "capital_delta_vs_control": capital - control_capital,
        "capital_lift_vs_control": capital / control_capital - 1.0,
        "cagr": float(metrics["cagr"]),
        "sharpe": sharpe,
        "maximum_drawdown": maxdd,
        "worst_fold_return": worst,
        "capital_rotations": int(metrics["capital_rotations"]),
        "market_exposure": float(metrics["market_exposure"]),
        "cash_days": int(metrics["cash_days"]),
        "loss_avoided_by_cash_usd": float(metrics["loss_avoided_by_cash_usd"]),
        "profit_missed_by_cash_usd": float(metrics["profit_missed_by_cash_usd"]),
        "net_cash_edge_usd": float(metrics["net_cash_edge_usd"]),
        "cash_intervention_sessions": int(metrics["cash_intervention_sessions"]),
    }
    for name in ("absolute_entry_threshold", "absolute_exit_discount", "absolute_exit_threshold", "cash_reentry_premium", "minimum_risk_safety", "minimum_horizon_agreement", "reentry_confirmation_sessions"):
        row[name] = replay["settings"].get(name)
    for fold in metrics.get("folds") or []:
        fid = int(fold["fold_id"])
        row[f"fold_{fid}_return"] = float(fold["strategy_return"])
        row[f"fold_{fid}_max_drawdown"] = float(fold["maximum_drawdown"])
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Temporal Absolute Opportunity + CASH Re-entry counterfactual research.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-run-id", help="Completed local Temporal Intelligence run id in MongoDB.")
    source.add_argument("--export-zip", type=Path, help="Temporal Intelligence export ZIP (offline mode).")
    parser.add_argument("--trials", type=int, default=500, help="Latin-Hypercube counterfactual candidates. Default: 500.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focus-month", default="2026-06")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    data = load_export_zip(args.export_zip) if args.export_zip else load_mongo(args.source_run_id)
    output = args.output_dir or Path("research") / f"temporal_cash_reentry_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(parents=True, exist_ok=True)

    compiled = compile_absolute_opportunity_context(
        data["observations"], data["winner_rows"], timing_settings=data["timing_settings"]
    )
    control = replay_compiled_absolute_opportunity_reentry_gate(
        compiled, initial_capital=data["initial_capital"], one_side_cost=data["one_side_cost"],
        winner_fold_returns=data["winner_fold_returns"], gate_enabled=False,
    )
    sampler = qmc.LatinHypercube(d=len(SEARCH_SPACE), seed=int(args.seed))
    points = sampler.random(n=int(args.trials))

    summaries: list[dict[str, Any]] = []
    replays: dict[int, dict[str, Any]] = {}
    control_row = _candidate_summary(0, control, control)
    control_row["is_control"] = True
    summaries.append(control_row)
    replays[0] = control

    for candidate_id, unit in enumerate(points, start=1):
        settings = settings_from_unit(unit)
        replay = replay_compiled_absolute_opportunity_reentry_gate(
            compiled, initial_capital=data["initial_capital"], one_side_cost=data["one_side_cost"],
            gate_settings=settings, winner_fold_returns=data["winner_fold_returns"], gate_enabled=True,
        )
        row = _candidate_summary(candidate_id, replay, control)
        row["is_control"] = False
        summaries.append(row)
        replays[candidate_id] = replay

    ranked = sorted(
        summaries,
        key=lambda row: (
            1 if bool(row["robust"]) else 0,
            float(row["ending_capital"]),
            float(row["sharpe"]),
            float(row["maximum_drawdown"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    best = ranked[0]
    best_id = int(best["candidate_id"])
    best_replay = replays[best_id]

    write_csv(output / "temporal_cash_reentry_candidates.csv", ranked)
    write_csv(output / "temporal_cash_reentry_best_equity.csv", list(best_replay["equity"]))
    write_csv(output / "temporal_cash_reentry_best_intervals.csv", list(best_replay["intervals"]))
    write_csv(output / "temporal_cash_reentry_monthly_attribution.csv", list(best_replay["monthly_attribution"]))
    focus_rows = [
        row for row in best_replay["intervals"]
        if pd.Timestamp(row["next_execution_date"] or row["decision_date"]).strftime("%Y-%m") == str(args.focus_month)
    ]
    write_csv(output / f"temporal_cash_reentry_focus_{str(args.focus_month).replace('-', '_')}.csv", focus_rows)

    manifest = {
        "experiment": "temporal_absolute_opportunity_cash_reentry_counterfactual_v1",
        "read_only": True,
        "source": data["source"],
        "source_temporal_run_id": data["source_run_id"],
        "trials": int(args.trials),
        "seed": int(args.seed),
        "focus_month": str(args.focus_month),
        "timing_settings": data["timing_settings"],
        "search_space": [
            {"name": name, "min": low, "max": high, "type": kind}
            for name, low, high, kind in SEARCH_SPACE
        ],
        "control": control_row,
        "best": best,
        "absolute_opportunity_weights": best_replay["settings"].get("absolute_opportunity_weights"),
        "selection": "robustness gate first, then ending capital; robust requires all folds positive, Sharpe >= control-0.05, MaxDD >= control-0.03",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (output / "temporal_cash_reentry_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    focus_month = next((row for row in best_replay["monthly_attribution"] if row["month"] == str(args.focus_month)), None)
    print(f"Source Temporal run: {data['source_run_id']}")
    print(f"Control ending capital: {control_row['ending_capital']:,.2f}")
    print(f"Best candidate: #{best_id} | ending capital: {best['ending_capital']:,.2f} | lift: {100*best['capital_lift_vs_control']:.2f}%")
    print(f"Best Sharpe: {best['sharpe']:.4f} | MaxDD: {100*best['maximum_drawdown']:.2f}% | cash days: {best['cash_days']}")
    print(f"Cash attribution: avoided {best['loss_avoided_by_cash_usd']:,.2f} | missed {best['profit_missed_by_cash_usd']:,.2f} | net {best['net_cash_edge_usd']:,.2f}")
    if focus_month:
        print(f"{args.focus_month}: avoided {focus_month['loss_avoided_by_cash_usd']:,.2f} | missed {focus_month['profit_missed_by_cash_usd']:,.2f} | net {focus_month['net_cash_edge_usd']:,.2f} | interventions {focus_month['cash_intervention_sessions']}")
    print(f"Results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
