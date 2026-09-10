from __future__ import annotations

from typing import Any

import research_exact_marginal_capital_search as base
import research_exact_marginal_capital_search_v103 as previous

SCRIPT_VERSION = "exact-marginal-capital-search-v1.0.4"


def _manifest_contract_v104(**kwargs: Any) -> dict[str, Any]:
    payload = dict(previous._manifest_contract_v103(**kwargs))
    payload.update({
        "schema_version": 5,
        "script_version": SCRIPT_VERSION,
        "preselector_recall_schema_version": 2,
        "preselector_recall_cutoff_basis": "original queue ranks, including uncompleted evaluations",
        "note": (
            "v1.0.4 corrects Top-K recall with rejected/failed candidates in the ranked queue. "
            "It preserves v1.0.3 identity checks and the existing economic judge."
        ),
    })
    return payload


def install_v104() -> None:
    previous.install_v103()
    base.SCRIPT_VERSION = SCRIPT_VERSION
    base._manifest_contract = _manifest_contract_v104


if __name__ == "__main__":
    install_v104()
    raise SystemExit(base.main())
