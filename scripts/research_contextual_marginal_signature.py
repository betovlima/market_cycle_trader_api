from __future__ import annotations

import sys

import research_contextual_signature_campaign_runner as _campaign_runner

CAMPAIGN_VERSION = "contextual-marginal-signature-v1.0.17.5"
TOURNAMENT_VERSION = "contextual-marginal-signature-v1.0.20.0"
SCRIPT_VERSION = CAMPAIGN_VERSION

# Preserve the stable public surface used by tests and research helpers.
for _export_name in dir(_campaign_runner):
    if not _export_name.startswith("_"):
        globals().setdefault(_export_name, getattr(_campaign_runner, _export_name))

_impl = _campaign_runner._impl
_campaign_runner.SCRIPT_VERSION = CAMPAIGN_VERSION
_campaign_runner._impl.SCRIPT_VERSION = CAMPAIGN_VERSION
_campaign_runner._impl.prev.SCRIPT_VERSION = CAMPAIGN_VERSION


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
    phase = _consume_phase_argument(sys.argv)
    if phase == "tournament":
        import research_contextual_signature_tournament as tournament

        return int(tournament.main(script_version=TOURNAMENT_VERSION))
    return int(_campaign_runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
