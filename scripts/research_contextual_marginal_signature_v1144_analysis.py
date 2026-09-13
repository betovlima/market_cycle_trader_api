from __future__ import annotations

import research_contextual_marginal_signature_v114_analysis as analysis

SCRIPT_VERSION = "contextual-marginal-signature-v1.0.14.4"
EXPERIMENT_NAME = "contextual_marginal_signature_direct_effect_prevalence"


def main() -> int:
    analysis.SCRIPT_VERSION = SCRIPT_VERSION
    analysis.EXPERIMENT_NAME = EXPERIMENT_NAME
    return analysis.main()


if __name__ == "__main__":
    raise SystemExit(main())
