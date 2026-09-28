from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from typing import Any, Callable

import pandas as pd

from ..core.environment import load_project_environment

load_project_environment()

from ..infrastructure.persistence.mongo_repository import (
    JOBS_COLLECTION,
    bson_value,
    create_client,
    ensure_database,
    get_database,
    replace_comparison,
    replace_run_result,
)
from ..schemas.requests import BacktestExecutionRequest
from ..services.reproducibility import build_reproducibility_manifest
from .market_data import validate_and_clean_bars
from .tcc_frozen_reference_source import (
    FROZEN_SOURCE,
    FROZEN_TCC_END,
    FROZEN_TCC_MAIN_SHA256,
    FROZEN_TCC_MAIN_COMMIT,
    FROZEN_TCC_START,
    frozen_tcc_root_from_environment,
    load_frozen_tcc_main_symbol,
    selected_tcc_reference_input_source,
    validate_frozen_tcc_main,
)
from .research_market_data import (
    StructuralResearchAssetExclusion,
    effective_research_config,
    load_research_market_bars,
)
from ..tcc_v106_reference.capital_rotation import (
    _build_walk_forward_folds,
    _fold_performance,
    prepare_rotation_panel,
    run_rotation_models,
)
from ..tcc_v106_reference.config import (
    ASSETS as TCC_V106_ASSETS,
    CONFIG as TCC_V106_CONFIG,
    SOFT_HORIZON_CONSENSUS_PENALTY,
    build_control_config,
    build_soft_config,
)
from ..tcc_v106_reference.execution import (
    apply_slippage,
    calculate_reference_fees,
)

TCC_SOURCE_REPOSITORY = "betovlima/tcc_mba_usp_data_science_analytics"
TCC_SOURCE_TAG = "v1.0.6"
TCC_SOURCE_COMMIT = "c0d71772092f0c26c9f28f0211b9933e9c396b95"
REFERENCE_ENGINE_ID = "tcc-v1.0.6-verbatim"
DATA_CONTRACT = "mct-current-alpaca-data+tcc-v1.0.6-engine"


def configure_console_utf8() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def load_mct_request(db: Any, job_id: str) -> BacktestExecutionRequest:
    job = db[JOBS_COLLECTION].find_one({"id": job_id}, {"_id": 0, "request": 1})
    if job is None:
        raise ValueError(f"Backtest job not found: {job_id}")
    return BacktestExecutionRequest.model_validate(job.get("request") or {})


def emit_progress(percent: float, stage: str, completed_runs: int = 0) -> None:
    safe_stage = str(stage).replace("|", "/").strip()
    print(
        f"JOB_PROGRESS|{float(percent):.1f}|{int(completed_runs)}|{safe_stage}",
        flush=True,
    )


def emit_progress_detail(detail: dict[str, Any]) -> None:
    safe = {
        key: value
        for key, value in detail.items()
        if key in {
            "run_index",
            "run_count",
            "fold_index",
            "fold_count",
            "phase",
            "trained_models",
            "total_models",
            "device",
        }
    }
    print(
        "JOB_DETAIL|"
        + json.dumps(safe, ensure_ascii=False, separators=(",", ":"), default=str),
        flush=True,
    )


def emit_research_technical(message: str) -> None:
    safe_message = str(message).replace("\n", " ").replace("\r", " ").strip()
    if safe_message:
        print(f"RESEARCH_TECH|{safe_message}", flush=True)


def _load_mct_market_frames(
    source_config: BacktestExecutionRequest,
) -> tuple[
    dict[str, pd.DataFrame],
    list[dict[str, Any]],
    BacktestExecutionRequest,
]:
    effective_config = effective_research_config(source_config)
    expected_assets = tuple(TCC_V106_ASSETS)
    configured_assets = tuple(str(item) for item in effective_config.assets)
    if set(configured_assets) != set(expected_assets):
        missing = sorted(set(expected_assets).difference(configured_assets))
        extra = sorted(set(configured_assets).difference(expected_assets))
        raise RuntimeError(
            "TCC v1.0.6 reference API requires the exact 56-asset universe. "
            f"missing={missing or 'none'} extra={extra or 'none'}"
        )

    source = selected_tcc_reference_input_source(effective_config)
    frozen_root = None
    frozen_manifest = None
    if source == FROZEN_SOURCE:
        if (
            str(effective_config.start_date) != FROZEN_TCC_START
            or str(effective_config.analysis_end_date or effective_config.end_date)
            != FROZEN_TCC_END
        ):
            raise ValueError(
                "Frozen TCC reference input requires exactly the pinned "
                "2016-01-01 through 2026-09-17 research window."
            )
        frozen_root = frozen_tcc_root_from_environment()
        frozen_manifest = validate_frozen_tcc_main(
            frozen_root, assets=expected_assets,
        )

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    exclusions: list[dict[str, Any]] = []
    anchors = set(effective_config.calendar_anchor_assets)
    total = len(expected_assets)

    for position, symbol in enumerate(expected_assets, start=1):
        emit_progress(
            2.0 + 13.0 * ((position - 1) / max(1, total)),
            f"TCC v1.0.6 loading market data {position}/{total} — {symbol}",
        )
        try:
            asset_config = (
                effective_config
                if symbol in anchors
                else effective_config.model_copy(
                    update={"market_data_require_complete_history": False}
                )
            )
            if frozen_manifest is not None and frozen_root is not None:
                raw = load_frozen_tcc_main_symbol(
                    frozen_root, symbol, frozen_manifest,
                )
            else:
                raw = load_research_market_bars(symbol, asset_config)
            bars_by_symbol[symbol] = validate_and_clean_bars(raw, asset_config)
        except StructuralResearchAssetExclusion as exc:
            exclusions.append(dict(exc.details))

    excluded = {
        str(item.get("symbol") or "").strip().upper()
        for item in exclusions
    }
    missing = [
        symbol
        for symbol in expected_assets
        if symbol not in bars_by_symbol and symbol not in excluded
    ]
    if missing:
        raise RuntimeError(
            "TCC v1.0.6 reference API could not load required assets: "
            + ", ".join(missing)
        )
    return bars_by_symbol, exclusions, effective_config


def _tcc_variant_configs(
    frames: dict[str, pd.DataFrame],
) -> tuple[Any, Any]:
    eligible = tuple(symbol for symbol in TCC_V106_ASSETS if symbol in frames)
    base = TCC_V106_CONFIG
    anchors = tuple(
        symbol
        for symbol in base.calendar_anchor_assets
        if symbol in frames
    )
    references = tuple(
        symbol
        for symbol in base.research_reference_assets
        if symbol in frames
    )
    reference_set = set(references)
    candidates = tuple(
        symbol
        for symbol in base.research_candidate_assets
        if symbol in frames and symbol not in reference_set
    )
    prepared = base.model_copy(
        update={
            "assets": eligible,
            "calendar_anchor_assets": anchors,
            "research_reference_assets": references,
            "research_candidate_assets": candidates,
        }
    )
    control = build_control_config(prepared, assets=eligible)
    soft = build_soft_config(
        prepared,
        assets=eligible,
        penalty_strength=SOFT_HORIZON_CONSENSUS_PENALTY,
    )
    return control, soft


def _variant_progress(
    label: str,
    start: float,
    end: float,
) -> Callable[[float, str, int], None]:
    span = max(0.0, float(end) - float(start))

    def callback(percent: float, stage: str, completed: int) -> None:
        local = max(0.0, min(100.0, float(percent))) / 100.0
        emit_progress(
            float(start) + span * local,
            f"TCC v1.0.6 {label} — {stage}",
            completed,
        )

    return callback


def _run_variant(
    *,
    label: str,
    backend: str,
    frames: dict[str, pd.DataFrame],
    config: Any,
    progress_start: float,
    progress_end: float,
) -> Any:
    results = run_rotation_models(
        frames,
        config,
        calculate_reference_fees,
        apply_slippage,
        progress_callback=_variant_progress(
            label,
            progress_start,
            progress_end,
        ),
        progress_detail_callback=emit_progress_detail,
        technical_log_callback=emit_research_technical,
    )
    if not results:
        raise RuntimeError(f"TCC v1.0.6 {label} returned no result.")
    result = results[0]
    result.backend = backend
    return result


def _fold_rows(
    result: Any,
    frames: dict[str, pd.DataFrame],
    config: Any,
) -> list[dict[str, Any]]:
    _, common_dates = prepare_rotation_panel(frames, config)
    folds = _build_walk_forward_folds(common_dates, config)
    return _fold_performance(
        result.predictions,
        folds,
        float(config.initial_capital),
    )


def _comparison_row(result: Any, label: str) -> dict[str, Any]:
    metrics = result.metrics
    return bson_value(
        {
            "symbol": "PORTFOLIO",
            "portfolio_rotation": True,
            "backend": result.backend,
            "model_family": "lightgbm_utility",
            "strategy_label": label,
            "strategy_mode": metrics.get("strategy_mode"),
            "strategy_ending_capital": metrics.get("strategy_ending_capital"),
            "strategy_return": metrics.get("strategy_return"),
            "strategy_cagr": metrics.get("strategy_cagr"),
            "strategy_sharpe": metrics.get("strategy_sharpe"),
            "strategy_maximum_drawdown": metrics.get(
                "strategy_maximum_drawdown"
            ),
            "buy_hold_ending_capital": metrics.get(
                "buy_hold_ending_capital"
            ),
            "buy_hold_return": metrics.get("buy_hold_return"),
            "capital_rotations": metrics.get("capital_rotations"),
            "effective_switch_margin": metrics.get(
                "effective_switch_margin"
            ),
            "reference_engine_id": REFERENCE_ENGINE_ID,
            "reference_source_tag": TCC_SOURCE_TAG,
            "reference_source_commit": TCC_SOURCE_COMMIT,
            "data_contract": DATA_CONTRACT,
        }
    )


def run_reference_job(
    job_id: str,
    request: BacktestExecutionRequest,
    db: Any,
) -> tuple[list[dict[str, Any]], BacktestExecutionRequest]:
    emit_progress(1.0, "Preparing TCC v1.0.6 reference engine")
    frames, exclusions, effective_mct_config = _load_mct_market_frames(request)
    control_config, soft_config = _tcc_variant_configs(frames)

    reproducibility = build_reproducibility_manifest(
        effective_mct_config,
        frames,
    )
    input_source = selected_tcc_reference_input_source(effective_mct_config)
    frozen_tcc_used = input_source == FROZEN_SOURCE
    data_contract = (
        "pinned-tcc-main-raw-csv+tcc-v1.0.6-engine"
        if frozen_tcc_used else DATA_CONTRACT
    )
    provenance = {
        "reference_engine_id": REFERENCE_ENGINE_ID,
        "reference_source_repository": TCC_SOURCE_REPOSITORY,
        "reference_source_tag": TCC_SOURCE_TAG,
        "reference_source_commit": TCC_SOURCE_COMMIT,
        "reference_engine_code": "verbatim-vendored-copy",
        "data_contract": data_contract,
        "mct_data_transport_only": not frozen_tcc_used,
        "tcc_frozen_snapshot_used": frozen_tcc_used,
        "input_source": input_source,
        "tcc_frozen_snapshot_sha256": (
            FROZEN_TCC_MAIN_SHA256 if frozen_tcc_used else None
        ),
        "tcc_frozen_snapshot_source_commit": (
            FROZEN_TCC_MAIN_COMMIT if frozen_tcc_used else None
        ),
        "structural_exclusions": deepcopy(exclusions),
        "eligible_asset_count": int(len(frames)),
        "configured_asset_count": int(len(TCC_V106_ASSETS)),
        "eligible_total_rows": int(
            sum(len(frame) for frame in frames.values())
        ),
        "market_data_signature_sha256": reproducibility.get(
            "market_data_signature_sha256"
        ),
    }

    emit_progress(16.0, "Running TCC v1.0.6 Control")
    control = _run_variant(
        label="CONTROL",
        backend="tcc_v106_control",
        frames=frames,
        config=control_config,
        progress_start=16.0,
        progress_end=53.0,
    )
    control.metrics.update(deepcopy(provenance))
    control.metrics["walk_forward_folds"] = _fold_rows(
        control,
        frames,
        control_config,
    )
    control.summary += (
        "\n\nTCC V1.0.6 REFERENCE API\n"
        f"Source tag: {TCC_SOURCE_TAG}\n"
        f"Source commit: {TCC_SOURCE_COMMIT}\n"
        f"Data contract: {data_contract}\n"
        f"TCC frozen snapshot used: {str(frozen_tcc_used).lower()}\n"
    )

    emit_progress(54.0, "Running TCC v1.0.6 Soft Horizon Consensus")
    soft = _run_variant(
        label="SOFT_HORIZON_CONSENSUS",
        backend="tcc_v106_soft",
        frames=frames,
        config=soft_config,
        progress_start=54.0,
        progress_end=91.0,
    )
    soft.metrics.update(deepcopy(provenance))
    soft.metrics["walk_forward_folds"] = _fold_rows(
        soft,
        frames,
        soft_config,
    )
    soft.summary += (
        "\n\nTCC V1.0.6 REFERENCE API\n"
        f"Source tag: {TCC_SOURCE_TAG}\n"
        f"Source commit: {TCC_SOURCE_COMMIT}\n"
        f"Data contract: {data_contract}\n"
        f"TCC frozen snapshot used: {str(frozen_tcc_used).lower()}\n"
    )

    batch_size = int(effective_mct_config.mongo_write_batch_size)
    for result in (control, soft):
        emit_progress(
            92.0 if result is control else 95.0,
            f"Saving {result.backend}",
            2,
        )
        replace_run_result(
            db,
            job_id=job_id,
            symbol="PORTFOLIO",
            backend=result.backend,
            metrics=result.metrics,
            summary=result.summary,
            predictions=result.predictions,
            trades=result.trades,
            batch_size=batch_size,
        )

    comparison = [
        _comparison_row(control, "TCC v1.0.6 Control"),
        _comparison_row(soft, "TCC v1.0.6 Soft Horizon Consensus"),
    ]
    emit_progress(99.0, "Finalizing TCC v1.0.6 comparison", 2)
    return comparison, effective_mct_config


def main() -> None:
    configure_console_utf8()
    args = parse_args()
    client = create_client()
    try:
        db = get_database(client)
        ensure_database(db)
        request = load_mct_request(db, args.job_id)
        print(
            "TCC v1.0.6 reference engine running inside MCT. "
            "Local simulation only.",
            flush=True,
        )
        print(
            f"TCC source: {TCC_SOURCE_REPOSITORY}@{TCC_SOURCE_TAG} "
            f"({TCC_SOURCE_COMMIT})",
            flush=True,
        )
        print(
            "Data source: MCT current Alpaca research snapshot; "
            "the TCC frozen CSV snapshot is not loaded.",
            flush=True,
        )
        comparison, effective_config = run_reference_job(
            args.job_id,
            request,
            db,
        )
        replace_comparison(
            db,
            job_id=args.job_id,
            comparison=comparison,
            failures=[],
            effective_config=bson_value(
                {
                    **effective_config.model_dump(mode="python"),
                    "reference_engine_id": REFERENCE_ENGINE_ID,
                    "reference_source_tag": TCC_SOURCE_TAG,
                    "reference_source_commit": TCC_SOURCE_COMMIT,
                    "data_contract": DATA_CONTRACT,
                }
            ),
        )
        if not comparison:
            raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
