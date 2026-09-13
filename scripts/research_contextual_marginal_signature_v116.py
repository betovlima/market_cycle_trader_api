from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import research_contextual_marginal_signature_v1144 as frozen
from market_cycle_trader_api.engine import research_challengers

base = frozen.base
diagnostics = frozen.diagnostics
storage = frozen.storage
progress = frozen.progress
sound = frozen.sound

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.16"
EXPERIMENT_NAME = "contextual_marginal_signature_paired_action_advantage"
LEGACY_MODE = "COMPOUND_ROTATION_SWING_XGBOOST"
TOLERANCE = 1e-12

_ORIGINAL_PARSER = base._parser
_ORIGINAL_SIMULATE_EXACT = research_challengers._simulate_exact


def _parser() -> argparse.ArgumentParser:
    parser = _ORIGINAL_PARSER()
    parser.add_argument(
        "--source-direct-dir",
        required=True,
        help=(
            "Completed v1.0.14.4 direct-effect campaign used as the normal-policy control. "
            "Its normal challenger ending capital is the denominator of the paired action advantage."
        ),
    )
    return parser


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _trace_prediction_path(root: Path, decision_date: str, universe: str, candidate: str) -> Path:
    trace_dir = root / "traces" / decision_date / universe / candidate
    index_path = trace_dir / "trace_index.json"
    if not index_path.exists():
        raise RuntimeError(f"Missing trace index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    backends = list(payload.get("backends") or [])
    if len(backends) != 1:
        raise RuntimeError(
            f"Action-advantage protocol currently requires exactly one captured backend; "
            f"trace={trace_dir}, backends={len(backends)}."
        )
    files = dict(backends[0].get("files") or {})
    name = str(files.get("predictions") or "").strip()
    if not name:
        raise RuntimeError(f"Trace index has no predictions file: {index_path}")
    return trace_dir / name


def _first_selected_asset(root: Path, decision_date: str, universe: str, candidate: str) -> tuple[str, int]:
    path = _trace_prediction_path(root, decision_date, universe, candidate)
    frame = pd.read_csv(path)
    if frame.empty or "selected_asset" not in frame.columns:
        raise RuntimeError(f"Prediction trace has no first selected asset: {path}")
    selected = str(frame.iloc[0]["selected_asset"] or "CASH").strip().upper()
    return selected, int(len(frame))


def _forced_first_action_simulator(
    backend: str,
    policy: Callable[[pd.Timestamp, int, int], tuple[int, float]],
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    decision_dates: pd.DatetimeIndex,
    config: Any,
    fee_calculator: Callable,
    slippage: Callable,
    decision_metadata: dict[pd.Timestamp, dict[str, Any]] | None = None,
    policy_decision_diagnostics: dict[pd.Timestamp, dict[str, Any]] | None = None,
    trade_callback: Callable[[dict[str, Any]], None] | None = None,
    *,
    model_label: str = "LightGBM Utility",
    method_line: str | None = None,
):
    context = dict(frozen._CURRENT_CONTEXT or {})
    candidate = str(context.get("candidate") or "").strip().upper()
    if not candidate:
        return _ORIGINAL_SIMULATE_EXACT(
            backend,
            policy,
            frames,
            symbols,
            decision_dates,
            config,
            fee_calculator,
            slippage,
            decision_metadata=decision_metadata,
            policy_decision_diagnostics=policy_decision_diagnostics,
            trade_callback=trade_callback,
            model_label=model_label,
            method_line=method_line,
        )

    if str(config.strategy_mode) != LEGACY_MODE:
        raise RuntimeError(
            "v1.0.16 paired rollout is intentionally restricted to the stateless legacy rotation policy; "
            f"observed strategy_mode={config.strategy_mode}. Stateful policy modes require an explicit "
            "hidden-state intervention protocol before forced-action rollouts are scientifically valid."
        )
    if candidate not in symbols:
        raise RuntimeError(f"Forced candidate {candidate} is not in simulator symbols.")
    if len(decision_dates) < 2:
        raise RuntimeError("Paired action rollout requires at least one decision-to-execution transition.")

    candidate_position = int(symbols.index(candidate) + 1)
    first_decision = _utc(decision_dates[0])
    forced = False
    policy_action_before_force: str | None = None

    def forced_policy(timestamp: pd.Timestamp, current_position: int, holding_days: int) -> tuple[int, float]:
        nonlocal forced, policy_action_before_force
        ts = _utc(timestamp)
        if not forced and ts == first_decision:
            normal_target, _normal_score = policy(timestamp, current_position, holding_days)
            policy_action_before_force = "CASH" if int(normal_target) <= 0 else str(symbols[int(normal_target) - 1])
            forced = True

            diag = (policy_decision_diagnostics or {}).get(pd.Timestamp(timestamp))
            if isinstance(diag, dict):
                diag["research_forced_first_action"] = True
                diag["research_policy_action_before_force"] = policy_action_before_force
                diag["research_forced_action_asset"] = candidate
                diag["final_action_asset"] = candidate
                diag["final_action_score"] = None
                diag["decision_reason"] = "RESEARCH_FORCE_CANDIDATE_FIRST_ACTION"

            # The score is diagnostic only in _simulate_exact.  The capital path is
            # determined by the forced target position and subsequent policy calls.
            return candidate_position, 0.0
        return policy(timestamp, current_position, holding_days)

    result = _ORIGINAL_SIMULATE_EXACT(
        backend,
        forced_policy,
        frames,
        symbols,
        decision_dates,
        config,
        fee_calculator,
        slippage,
        decision_metadata=decision_metadata,
        policy_decision_diagnostics=policy_decision_diagnostics,
        trade_callback=trade_callback,
        model_label=model_label,
        method_line=method_line,
    )
    if not forced:
        raise RuntimeError(f"Forced first action was never applied for candidate={candidate}.")

    result.metrics["research_forced_first_action"] = True
    result.metrics["research_forced_action_asset"] = candidate
    result.metrics["research_policy_action_before_force"] = policy_action_before_force
    result.metrics["research_action_advantage_protocol"] = (
        "force candidate on the first decision, then resume the same frozen-margin legacy policy "
        "from the resulting simulator state"
    )
    return result


def _validate_source(source: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest_path = source / "trace_manifest.json"
    output_manifest_path = output / "trace_manifest.json"
    if not source_manifest_path.exists() or not output_manifest_path.exists():
        raise RuntimeError("Both source and forced campaign must contain trace_manifest.json.")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
    if str(source_manifest.get("script_version")) != "contextual-marginal-signature-v1.0.14.4":
        raise RuntimeError(
            "--source-direct-dir must be the completed v1.0.14.4 fold-aware direct-effect campaign."
        )
    for key in ("market_snapshot_hash", "strategy_configuration_hash"):
        if source_manifest.get(key) != output_manifest.get(key):
            raise RuntimeError(
                f"Paired rollout source mismatch for {key}: "
                f"source={source_manifest.get(key)}, forced={output_manifest.get(key)}"
            )
    return source_manifest, output_manifest


def _postprocess(source: Path, output: Path) -> None:
    source_manifest, output_manifest = _validate_source(source, output)
    source_frame = pd.read_csv(source / "trace_aggregate_dataset.csv")
    forced_frame = pd.read_csv(output / "trace_aggregate_dataset.csv")
    keys = ["decision_date", "universe_name", "candidate"]
    if source_frame.duplicated(keys).any() or forced_frame.duplicated(keys).any():
        raise RuntimeError("Paired rollout aggregate contains duplicate candidate/context keys.")

    source_columns = keys + [
        "candidate_ending_capital",
        "baseline_ending_capital",
        "horizon_end",
    ]
    forced_columns = keys + ["candidate_ending_capital"]
    merged = source_frame[source_columns].merge(
        forced_frame[forced_columns],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_policy", "_forced"),
        indicator=True,
    )
    if not bool((merged["_merge"] == "both").all()):
        bad = merged.loc[merged["_merge"] != "both", keys + ["_merge"]]
        raise RuntimeError(f"Paired rollout key mismatch:\n{bad.to_string(index=False)}")
    merged = merged.drop(columns=["_merge"])

    rows: list[dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        date = str(record["decision_date"])
        universe = str(record["universe_name"])
        candidate = str(record["candidate"]).strip().upper()
        policy_capital = float(record["candidate_ending_capital_policy"])
        forced_capital = float(record["candidate_ending_capital_forced"])
        if policy_capital <= 0 or forced_capital <= 0:
            raise RuntimeError(f"Non-positive paired ending capital for {date}/{universe}/{candidate}.")

        normal_first, normal_execution_rows = _first_selected_asset(source, date, universe, candidate)
        forced_first, forced_execution_rows = _first_selected_asset(output, date, universe, candidate)
        if forced_first != candidate:
            raise RuntimeError(
                f"Forced-action invariant failed for {date}/{universe}/{candidate}: first selected={forced_first}."
            )
        if normal_execution_rows != forced_execution_rows:
            raise RuntimeError(
                f"Paired rollout length mismatch for {date}/{universe}/{candidate}: "
                f"policy={normal_execution_rows}, forced={forced_execution_rows}."
            )

        advantage_log = float(math.log(forced_capital / policy_capital))
        rows.append(
            {
                "decision_date": date,
                "horizon_end": record["horizon_end"],
                "universe_name": universe,
                "candidate": candidate,
                "policy_ending_capital": policy_capital,
                "forced_action_ending_capital": forced_capital,
                "action_advantage_log": advantage_log,
                "action_advantage_rate": float(forced_capital / policy_capital - 1.0),
                "policy_first_selected_asset": normal_first,
                "policy_first_selected_candidate": bool(normal_first == candidate),
                "forced_first_selected_asset": forced_first,
                "execution_transitions": int(forced_execution_rows),
                "decision_points": int(forced_execution_rows + 1),
                "source_direct_delta_log_capital": float(
                    math.log(policy_capital / float(record["baseline_ending_capital"]))
                ),
            }
        )

    dataset = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    dataset.to_csv(output / "action_advantage_dataset.csv", index=False)

    nonzero = dataset[dataset["action_advantage_log"].abs() > TOLERANCE]
    summary = {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "completed",
        "source_direct_script_version": source_manifest.get("script_version"),
        "market_snapshot_hash": output_manifest.get("market_snapshot_hash"),
        "strategy_configuration_hash": output_manifest.get("strategy_configuration_hash"),
        "rows": int(len(dataset)),
        "dates": int(dataset["decision_date"].nunique()),
        "universes": int(dataset["universe_name"].nunique()),
        "candidates": int(dataset["candidate"].nunique()),
        "nonzero_action_advantage_rows": int(len(nonzero)),
        "positive_action_advantage_rows": int((dataset["action_advantage_log"] > TOLERANCE).sum()),
        "negative_action_advantage_rows": int((dataset["action_advantage_log"] < -TOLERANCE).sum()),
        "policy_already_selected_candidate_rows": int(dataset["policy_first_selected_candidate"].sum()),
        "execution_transitions_values": sorted(int(value) for value in dataset["execution_transitions"].unique()),
        "decision_points_values": sorted(int(value) for value in dataset["decision_points"].unique()),
        "horizon_definition": (
            "decision_points includes the causal decision point and the subsequent execution sessions; "
            "execution_transitions is the number of next-open transitions actually simulated."
        ),
        "target_definition": (
            "action_advantage_log = log(W_forced_candidate_then_same_policy / W_same_policy_without_force). "
            "The candidate is available in both arms; switch-margin calibration is frozen fold-by-fold to "
            "the v1.0.14.4 universe baseline."
        ),
        "next_step": (
            "Use this paired action-advantage target for a controlled model-capacity benchmark. "
            "Compare linear, tree and neural estimators under the same chronological split before "
            "expanding signature depth or architecture complexity."
        ),
    }
    (output / "action_advantage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    # This version changes the target, not the market data or strategy family.
    # v1.0.14.4 remains the normal-policy control.  v1.0.16 reruns the same
    # challenger contexts with only the first decision forced to the candidate.
    base._parser = _parser
    research_challengers._simulate_exact = _forced_first_action_simulator

    diagnostics.SCRIPT_VERSION = SCRIPT_VERSION
    diagnostics.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    storage.SCRIPT_VERSION = SCRIPT_VERSION
    progress.SCRIPT_VERSION = SCRIPT_VERSION
    sound.SCRIPT_VERSION = SCRIPT_VERSION
    frozen.SCRIPT_VERSION = SCRIPT_VERSION
    frozen.EXPERIMENT_NAME = EXPERIMENT_NAME

    exit_code = int(frozen.main())
    if exit_code != 0:
        return exit_code

    args = _parser().parse_args()
    source = Path(args.source_direct_dir).resolve()
    output = Path(args.output_dir).resolve()
    _postprocess(source, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
