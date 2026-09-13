from __future__ import annotations

import research_contextual_marginal_signature_v112 as frozen

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.14"
EXPERIMENT_NAME = "contextual_marginal_signature_direct_effect_prevalence"


def main() -> int:
    # Preserve the v1.0.12 frozen-switch-margin mechanism exactly. This version
    # changes only the campaign identity/case design so we can estimate how often
    # direct candidate effects occur across candidates and separated regimes.
    frozen.SCRIPT_VERSION = SCRIPT_VERSION
    frozen.EXPERIMENT_NAME = EXPERIMENT_NAME
    return frozen.main()


if __name__ == "__main__":
    raise SystemExit(main())
