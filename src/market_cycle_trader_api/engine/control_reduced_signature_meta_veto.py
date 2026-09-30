"""v10.8.53 reduced-signature one-shot fail-safe meta-veto research.

The v10.8.44 Liquidity-Aware Control remains the default policy. A fixed
Logistic Regression (C=0.25) trained only on prior-fold v10.8.49 rollout
labels may veto one asset-to-asset Control rotation by holding the incumbent
for one decision.

Protocol frozen before capital evaluation:
- exactly the nine v10.8.50 features;
- fold 1 always Control-only;
- fold 2 learns only fold 1;
- fold 3 learns only folds 1+2;
- prior data split chronologically 70/30 for train/calibration;
- enable only with >=20 calibration samples, BA>=0.52 and AUC>=0.52;
- veto only when P(ROTATE better)<=0.35;
- CASH transitions are never modified;
- dynamic features use the actual meta-trajectory state at decision time;
- labels must be fully matured before the test-fold start;
- after a veto, the next policy decision is Control-only (one-shot semantics).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..tcc_v106_reference import research_challengers as scientific
from ..tcc_v106_reference.capital_rotation import _utility_policy as frozen_utility_policy
from .control_execution_feasibility import simulate_feasible_control
from .control_liquidity_policy import _CapitalAwareUtilityCache
from .control_reduced_rollout_signature import REDUCED_FEATURES

MODE = "MCT_RESEARCH_REDUCED_SIGNATURE_META_VETO_ONE_SHOT_V2"
MODEL_C = 0.25
TRAIN_FRACTION = 0.70
MIN_CALIBRATION_SAMPLES = 20
MIN_CALIBRATION_BALANCED_ACCURACY = 0.52
MIN_CALIBRATION_ROC_AUC = 0.52
VETO_PROBABILITY_MAX = 0.35
RANDOM_SEED = 20260930
LIQUIDITY_LOOKBACK = 20
PARTICIPATION_RATE = 0.10
CAPITAL_WEIGHT_FLOOR = 1e-6


@dataclass(frozen=True)
class FoldModel:
    enabled: bool
    model: Pipeline | None
    disable_reason: str | None


def _make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=MODEL_C,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])


def _chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("decision_date").reset_index(drop=True)
    if len(ordered) <= MIN_CALIBRATION_SAMPLES:
        return ordered.iloc[0:0], ordered.iloc[0:0]
    cut = int(math.floor(len(ordered) * TRAIN_FRACTION))
    cut = min(cut, len(ordered) - MIN_CALIBRATION_SAMPLES)
    if cut <= 0:
        return ordered.iloc[0:0], ordered.iloc[0:0]
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _capital_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    raw = pd.to_numeric(
        frame["delta_capital_fraction"], errors="coerce",
    ).abs().fillna(0.0).to_numpy(dtype=float)
    raw = np.maximum(raw, CAPITAL_WEIGHT_FLOOR)
    mean = float(np.mean(raw))
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("Capital-weighted training requires positive finite weights.")
    return raw / mean


def train_fold_models(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    *,
    capital_weighted: bool = False,
) -> tuple[dict[int, FoldModel], list[dict[str, Any]]]:
    required = {"decision_date", "rollout_end_date", "source_fold_id", "rotate_better", *REDUCED_FEATURES}
    if capital_weighted:
        required.add("delta_capital_fraction")
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"v10.8.53 source dataset missing columns: {missing}")
    source = dataset.copy()
    source["decision_date"] = pd.to_datetime(source["decision_date"], utc=True)
    source["rollout_end_date"] = pd.to_datetime(source["rollout_end_date"], utc=True)
    source = source.sort_values("decision_date").reset_index(drop=True)

    models: dict[int, FoldModel] = {}
    reports: list[dict[str, Any]] = []
    for fold in folds:
        fid = int(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        if test_start.tzinfo is None:
            test_start = test_start.tz_localize("UTC")
        else:
            test_start = test_start.tz_convert("UTC")
        prior = source.loc[
            (source["source_fold_id"].astype(int) < fid)
            & (source["rollout_end_date"] < test_start)
        ].copy()
        if fid == 1:
            models[fid] = FoldModel(False, None, "NO_PRIOR_OOS_FOLD")
            reports.append({
                "fold_id": fid,
                "eligible_prior_rows": 0,
                "training_rows": 0,
                "calibration_rows": 0,
                "model_enabled": False,
                "disable_reason": "NO_PRIOR_OOS_FOLD",
                "test_start": test_start.isoformat(),
            })
            continue

        training, calibration = _chronological_split(prior)
        if training.empty or len(calibration) < MIN_CALIBRATION_SAMPLES:
            models[fid] = FoldModel(False, None, "INSUFFICIENT_PRIOR_ROWS")
            reports.append({
                "fold_id": fid,
                "eligible_prior_rows": int(len(prior)),
                "training_rows": int(len(training)),
                "calibration_rows": int(len(calibration)),
                "model_enabled": False,
                "disable_reason": "INSUFFICIENT_PRIOR_ROWS",
                "test_start": test_start.isoformat(),
            })
            continue

        model = _make_model()
        training_weights = (
            _capital_sample_weights(training) if capital_weighted else None
        )
        training_fit = (
            {"model__sample_weight": training_weights}
            if training_weights is not None else {}
        )
        model.fit(
            training[list(REDUCED_FEATURES)],
            training["rotate_better"].astype(int),
            **training_fit,
        )
        probability = model.predict_proba(calibration[list(REDUCED_FEATURES)])[:, 1]
        prediction = (probability >= 0.50).astype(int)
        y_cal = calibration["rotate_better"].astype(int)
        ba = float(balanced_accuracy_score(y_cal, prediction))
        auc = (
            float(roc_auc_score(y_cal, probability))
            if y_cal.nunique() == 2 else float("nan")
        )
        enabled = bool(
            len(calibration) >= MIN_CALIBRATION_SAMPLES
            and math.isfinite(ba) and ba >= MIN_CALIBRATION_BALANCED_ACCURACY
            and math.isfinite(auc) and auc >= MIN_CALIBRATION_ROC_AUC
        )
        final_model = None
        reason = None
        if enabled:
            final_model = _make_model()
            prior_weights = (
                _capital_sample_weights(prior) if capital_weighted else None
            )
            final_fit = (
                {"model__sample_weight": prior_weights}
                if prior_weights is not None else {}
            )
            final_model.fit(
                prior[list(REDUCED_FEATURES)],
                prior["rotate_better"].astype(int),
                **final_fit,
            )
        else:
            reason = "CALIBRATION_GATE_FAILED"
        models[fid] = FoldModel(enabled, final_model, reason)
        reports.append({
            "fold_id": fid,
            "eligible_prior_rows": int(len(prior)),
            "training_rows": int(len(training)),
            "calibration_rows": int(len(calibration)),
            "final_training_rows": int(len(prior)) if enabled else 0,
            "calibration_positive_rate": float(y_cal.mean()),
            "calibration_balanced_accuracy": ba,
            "calibration_roc_auc": auc if math.isfinite(auc) else None,
            "model_enabled": enabled,
            "disable_reason": reason,
            "veto_probability_max": VETO_PROBABILITY_MAX,
            "test_start": test_start.isoformat(),
            "capital_weighted_training": bool(capital_weighted),
            "weight_definition": (
                "abs(delta_capital_fraction)/mean(abs(delta_capital_fraction))"
                if capital_weighted else "uniform"
            ),
        })
    return models, reports


def _loc(frame: pd.DataFrame, date: pd.Timestamp) -> int:
    idx = frame.index.get_indexer([pd.Timestamp(date)])
    return int(idx[0]) if len(idx) == 1 and int(idx[0]) >= 0 else -1


def _prior_median_volume(frame: pd.DataFrame, loc: int) -> float:
    prior = pd.to_numeric(
        frame["volume"].iloc[max(0, loc - LIQUIDITY_LOOKBACK):loc],
        errors="coerce",
    )
    prior = prior[np.isfinite(prior) & (prior >= 0)]
    return float(prior.median()) if len(prior) else float("nan")


def build_dynamic_feature_row(
    *,
    frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    incumbent_asset: str,
    candidate_asset: str,
    shares: int,
    equity: float,
) -> pd.DataFrame:
    incumbent = frames[incumbent_asset]
    candidate = frames[candidate_asset]
    iloc = _loc(incumbent, date)
    cloc = _loc(candidate, date)
    if iloc < 0 or cloc < 0:
        raise ValueError("Meta-veto decision date missing from frozen panel.")
    if not math.isfinite(float(equity)) or float(equity) <= 0:
        raise ValueError("Meta-veto requires positive known decision equity.")

    prior_volume = _prior_median_volume(incumbent, iloc)
    incumbent_close = float(incumbent["close"].iloc[iloc])
    estimated_capacity = (
        math.floor(prior_volume * PARTICIPATION_RATE)
        if math.isfinite(prior_volume) else float("nan")
    )
    capacity_ratio = (
        estimated_capacity * incumbent_close / float(equity)
        if math.isfinite(float(estimated_capacity)) else float("nan")
    )
    values = {
        "incumbent__return_20": float(incumbent["return_20"].iloc[iloc]),
        "incumbent__return_60": float(incumbent["return_60"].iloc[iloc]),
        "incumbent__ema_distance_20": float(
            incumbent["ema_distance_20"].iloc[iloc]
        ),
        "incumbent__ema_distance_50": float(
            incumbent["ema_distance_50"].iloc[iloc]
        ),
        "incumbent__rsi_14": float(incumbent["rsi_14"].iloc[iloc]),
        "incumbent__channel_position_50": float(
            incumbent["channel_position_50"].iloc[iloc]
        ),
        "incumbent_capacity_equity_ratio": float(capacity_ratio),
        "state_shares": float(shares),
        "candidate__channel_position_50": float(
            candidate["channel_position_50"].iloc[cloc]
        ),
    }
    if tuple(values) != tuple(REDUCED_FEATURES):
        raise RuntimeError("v10.8.53 dynamic feature order drifted from v10.8.50.")
    return pd.DataFrame([values], columns=list(REDUCED_FEATURES))


def run_reduced_signature_meta_veto_pair(
    bars: dict[str, pd.DataFrame],
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    *,
    source_dataset: pd.DataFrame,
    capital_weighted: bool = False,
    progress_callback: Callable[[float, str, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    (
        frames, _common_dates, symbols, folds, _all_decision_dates,
        _decision_to_fold, decision_metadata,
    ) = scientific._build_execution_context(bars, config)
    if len(symbols) != 55 or len(folds) != 3:
        raise ValueError("v10.8.53 expects frozen 55-asset / 3-fold Control context.")

    fold_models, training_reports = train_fold_models(
        source_dataset, folds, capital_weighted=capital_weighted,
    )

    original_runner = scientific._run_lightgbm
    original_policy = original_runner.__globals__.get("_utility_policy")
    original_simulator = original_runner.__globals__.get("_simulate_exact")
    if original_policy is not frozen_utility_policy or original_simulator is not scientific._simulate_exact:
        raise RuntimeError("Frozen Control runner bindings changed unexpectedly.")

    account: dict[str, Any] = {"enabled": True, "audit": {}}
    captured: dict[str, Any] = {}
    audit: list[dict[str, Any]] = []

    def prepare_account(date, position, holding, cash, shares, equity):
        account.update({
            "decision_timestamp": pd.Timestamp(date),
            "position": int(position),
            "holding_days": int(holding),
            "cash": float(cash),
            "shares": int(shares),
            "equity": float(equity),
        })

    def policy_wrapper(models, panel, labels, settings, switch_margin, **kwargs):
        utility_cache = kwargs.get("utility_cache")
        if utility_cache is None:
            return original_policy(
                models, panel, labels, settings, switch_margin, **kwargs,
            )
        liquidity_cache = _CapitalAwareUtilityCache(
            utility_cache, panel, labels, account,
        )
        base_kwargs = dict(kwargs)
        base_kwargs["utility_cache"] = liquidity_cache
        return original_policy(
            models, panel, labels, settings, switch_margin, **base_kwargs,
        )

    def simulator_wrapper(*args, **kwargs):
        if captured:
            raise ValueError("v10.8.53 expected exactly one scheduled OOS simulator call.")
        backend, scheduled, panel, labels, dates, settings, fees, slip = args[:8]
        metadata = kwargs.get("decision_metadata") or decision_metadata

        baseline = simulate_feasible_control(
            backend, scheduled, panel, labels, dates, settings, fees, slip,
            **{**kwargs, "decision_prepare": prepare_account},
        )
        captured["liquidity_baseline"] = baseline

        force_control_next = [False]

        def meta_policy(
            timestamp: pd.Timestamp,
            current_position: int,
            holding_days: int,
        ) -> tuple[int, float]:
            control_target, control_score = scheduled(
                timestamp, current_position, holding_days,
            )
            key = pd.Timestamp(timestamp)
            fid = int((metadata or {}).get(key, {}).get("fold_id") or 0)
            fold_model = fold_models.get(fid, FoldModel(False, None, "UNKNOWN_FOLD"))
            incumbent = labels[current_position - 1] if current_position > 0 else "CASH"
            candidate = labels[control_target - 1] if control_target > 0 else "CASH"
            probability = None
            veto = False
            reason = "CONTROL_DEFAULT"

            if force_control_next[0]:
                force_control_next[0] = False
                reason = "CONTROL_AFTER_ONE_SHOT_VETO"
            elif (
                fold_model.enabled
                and current_position > 0
                and control_target > 0
                and current_position != control_target
            ):
                if pd.Timestamp(account.get("decision_timestamp")) != key:
                    raise ValueError("Meta-veto account state is not aligned to decision date.")
                feature_row = build_dynamic_feature_row(
                    frames=panel,
                    date=key,
                    incumbent_asset=incumbent,
                    candidate_asset=candidate,
                    shares=int(account["shares"]),
                    equity=float(account["equity"]),
                )
                probability = float(
                    fold_model.model.predict_proba(feature_row)[:, 1][0]
                )
                if probability <= VETO_PROBABILITY_MAX:
                    veto = True
                    force_control_next[0] = True
                    reason = "VETO_LOW_ROTATE_PROBABILITY"
                else:
                    reason = "MODEL_PASS"
            elif not fold_model.enabled:
                reason = fold_model.disable_reason or "MODEL_DISABLED"
            elif current_position == 0 or control_target == 0:
                reason = "CASH_TRANSITION_NOT_ELIGIBLE"
            else:
                reason = "NO_ASSET_ROTATION"

            final_target = int(current_position) if veto else int(control_target)
            audit.append({
                "decision_date": key,
                "fold_id": fid,
                "incumbent_asset": incumbent,
                "control_target_asset": candidate,
                "final_target_asset": (
                    labels[final_target - 1] if final_target > 0 else "CASH"
                ),
                "probability_rotate_better": probability,
                "model_enabled": bool(fold_model.enabled),
                "veto_applied": veto,
                "reason": reason,
                "state_shares": int(account.get("shares") or 0),
                "state_equity": float(account.get("equity") or 0.0),
                "capital_weighted_training": bool(capital_weighted),
            })
            return final_target, float(control_score)

        captured["meta_veto"] = simulate_feasible_control(
            backend,
            meta_policy,
            panel,
            labels,
            dates,
            settings,
            fees,
            slip,
            **{**kwargs, "decision_prepare": prepare_account},
        )
        return baseline

    isolated_globals = dict(original_runner.__globals__)
    isolated_globals["_utility_policy"] = policy_wrapper
    isolated_globals["_simulate_exact"] = simulator_wrapper
    isolated = FunctionType(
        original_runner.__code__, isolated_globals, original_runner.__name__,
        original_runner.__defaults__, original_runner.__closure__,
    )
    isolated.__kwdefaults__ = dict(original_runner.__kwdefaults__ or {})
    result = isolated(
        bars, config, fee_calculator, slippage,
        progress_callback=progress_callback,
        trade_callback=None,
        progress_detail_callback=None,
        technical_log_callback=None,
    )
    if (
        len(result) != 1
        or result[0] is not captured.get("liquidity_baseline")
        or set(captured) != {"liquidity_baseline", "meta_veto"}
    ):
        raise ValueError("v10.8.53 replay did not produce baseline and meta paths.")
    if (
        original_runner.__globals__.get("_utility_policy") is not original_policy
        or original_runner.__globals__.get("_simulate_exact") is not original_simulator
    ):
        raise RuntimeError("Frozen TCC module was mutated by v10.8.53.")
    return captured, training_reports, audit
