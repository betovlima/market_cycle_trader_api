from __future__ import annotations

import research_contextual_marginal_signature_v112 as frozen

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.14.2"
EXPERIMENT_NAME = "contextual_marginal_signature_direct_effect_prevalence"


def main() -> int:
    frozen.SCRIPT_VERSION = SCRIPT_VERSION
    frozen.EXPERIMENT_NAME = EXPERIMENT_NAME
    return frozen.main()


if __name__ == "__main__":
    raise SystemExit(main())
