from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import research_contextual_marginal_signature_v1 as base  # noqa: E402

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.1"


def _utc_index(values: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return one sorted UTC-aware index regardless of the source timezone policy."""
    return pd.DatetimeIndex(pd.to_datetime(pd.DatetimeIndex(values), utc=True)).sort_values()


def _horizon_end(
    sessions: pd.DatetimeIndex,
    decision_session: pd.Timestamp,
    horizon_sessions: int,
) -> pd.Timestamp:
    """Resolve the future horizon without mixing tz-naive and tz-aware timestamps."""
    ordered = _utc_index(sessions)
    decision = base._utc_timestamp(decision_session)
    position = int(ordered.searchsorted(decision, side="left"))
    if position >= len(ordered) or pd.Timestamp(ordered[position]) != decision:
        raise RuntimeError(f"Decision session is not in the expected calendar: {decision.date()}")
    end_position = position + max(1, int(horizon_sessions)) - 1
    if end_position >= len(ordered):
        raise RuntimeError(f"Horizon exceeds snapshot for decision session {decision.date()}")
    return pd.Timestamp(ordered[end_position])


def install_v101() -> None:
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._horizon_end = _horizon_end


if __name__ == "__main__":
    install_v101()
    base._log(
        "Contextual Marginal Signature v1.0.1 enabled: calendar sessions are normalized to UTC before "
        "horizon search. The experiment definition, exact capital labels and feature set are unchanged."
    )
    raise SystemExit(base.main())
