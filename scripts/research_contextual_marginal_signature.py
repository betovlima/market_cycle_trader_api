from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import research_contextual_signature_campaign_runner as _campaign_runner

CAMPAIGN_VERSION = "contextual-marginal-signature-v1.0.17.5"
TOURNAMENT_VERSION = "contextual-marginal-signature-v1.0.20.0"
SCRIPT_VERSION = CAMPAIGN_VERSION

# ---------------------------------------------------------------------------
# Spyder runner
# ---------------------------------------------------------------------------
# This is intentionally the only file a researcher needs to open and run in
# Spyder.  The scientific implementation remains in the existing project
# modules so that the CLI, tests and historical research contracts stay
# unchanged.
#
# v1.0.20 scientific boundary:
# - NO new ML model is trained here.
# - Existing campaign traces may contain decisions produced by the frozen
#   Strategy/LightGBM pipeline, but this phase only reads those persisted
#   traces from local MongoDB.
# - The analysis is retrospective mechanism decomposition: exact counts,
#   forward/log returns, capital deltas, path divergence and table-improvement
#   diagnostics.
# - There are NO p-value/significance claims and NO predictive claim in this
#   phase.  Predictive ML is a later experiment only if the mechanism is first
#   observed retrospectively.
#
# In Spyder, edit only this block if necessary and press Run (F5).
SPYDER_CONFIG: dict[str, Any] = {
    "enabled": True,
    "phase": "tournament",
    "strategy_sequence": 10,
    "strategy_id": None,
    "env_file": str(Path(__file__).resolve().parents[1] / ".env"),
    "mongo_uri": None,
    "database": None,
}


# Preserve the stable public surface used by tests and research helpers.
for _export_name in dir(_campaign_runner):
    if not _export_name.startswith("_"):
        globals().setdefault(_export_name, getattr(_campaign_runner, _export_name))

_impl = _campaign_runner._impl
_campaign_runner.SCRIPT_VERSION = CAMPAIGN_VERSION
_campaign_runner._impl.SCRIPT_VERSION = CAMPAIGN_VERSION
_campaign_runner._impl.prev.SCRIPT_VERSION = CAMPAIGN_VERSION


def _running_in_spyder() -> bool:
    """Return True only for an active Spyder kernel/session."""
    if os.getenv("SPYDER_KERNEL_ID") or os.getenv("SPYDER_ARGS"):
        return True
    return any(name.startswith("spyder_kernels") for name in sys.modules)


def _spyder_argv() -> list[str]:
    phase = str(SPYDER_CONFIG.get("phase") or "tournament").strip().lower()
    if phase not in {"campaign", "tournament"}:
        raise SystemExit("SPYDER_CONFIG['phase'] must be campaign or tournament")

    argv = [
        sys.argv[0] if sys.argv else "research_contextual_marginal_signature.py",
        "--phase",
        phase,
        "--strategy-sequence",
        str(int(SPYDER_CONFIG.get("strategy_sequence") or 10)),
    ]

    optional = (
        ("strategy_id", "--strategy-id"),
        ("env_file", "--env-file"),
        ("mongo_uri", "--mongo-uri"),
        ("database", "--database"),
    )
    for key, flag in optional:
        value = SPYDER_CONFIG.get(key)
        if value is not None and str(value).strip():
            argv.extend([flag, str(value)])
    return argv


def _consume_phase_argument(argv: list[str]) -> str:
    phase = "campaign"
    if "--phase" not in argv:
        return phase
    index = argv.index("--phase")
    if index + 1 >= len(argv):
        raise SystemExit("--phase requires campaign or tournament")
    phase = str(argv[index + 1]).strip().lower()
    del argv[index:index + 2]
    if phase not in {"campaign", "tournament"}:
        raise SystemExit("--phase must be campaign or tournament")
    return phase


def main() -> int:
    if bool(SPYDER_CONFIG.get("enabled")) and _running_in_spyder():
        sys.argv[:] = _spyder_argv()
        print(
            "[spyder] Contextual Marginal Signature | "
            f"phase={SPYDER_CONFIG['phase']} | "
            f"strategy={SPYDER_CONFIG['strategy_sequence']} | "
            "source=local MongoDB",
            flush=True,
        )

    phase = _consume_phase_argument(sys.argv)
    if phase == "tournament":
        import research_contextual_signature_tournament as tournament

        return int(tournament.main(script_version=TOURNAMENT_VERSION))
    return int(_campaign_runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
