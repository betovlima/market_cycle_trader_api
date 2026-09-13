from __future__ import annotations

# Stable user-facing entrypoint. Internal research versions may evolve, but the
# command, output folder and ZIP names remain unchanged.
import research_contextual_marginal_signature_v117 as _impl
from research_contextual_marginal_signature_v117 import *  # noqa: F401,F403

# v1.0.17 originally included calendar anchors before the locked champion had
# any executable out-of-sample fold.  Strategy #10 starts in 2016, but its
# protected walk-forward protocol needs feature warm-up + training + purge +
# calibration before the first executable OOS session.  The campaign therefore
# starts safely inside the executable window instead of asking the engine to
# simulate a period for which no champion policy exists.
SCRIPT_VERSION = "contextual-marginal-signature-v1.0.17.1"
DECISION_DATES = (
    "2020-08-03", "2020-11-02",
    "2021-02-01", "2021-05-03", "2021-08-02", "2021-11-01",
    "2022-02-01", "2022-05-02", "2022-08-01", "2022-11-01",
    "2023-02-01", "2023-05-01", "2023-08-01", "2023-11-01",
    "2024-02-01", "2024-05-01", "2024-08-01", "2024-11-01",
    "2025-02-03", "2025-05-01", "2025-08-01",
    "2026-02-02", "2026-06-01",
)

_impl.SCRIPT_VERSION = SCRIPT_VERSION
_impl.DECISION_DATES = DECISION_DATES
main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
