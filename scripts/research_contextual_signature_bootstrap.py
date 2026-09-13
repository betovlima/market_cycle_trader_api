from __future__ import annotations

import importlib
import sys
from types import ModuleType

# The current scientific protocol was built incrementally across several
# historical modules.  Those implementation files now have stable,
# non-versioned names, but their internal imports still reference the old module
# names.  Load them in dependency order and register compatibility aliases in
# memory so the public source tree no longer needs versioned Python filenames.
_MODULE_CHAIN = (
    ("research_contextual_marginal_signature_v111", "research_contextual_signature_base"),
    ("research_contextual_marginal_signature_v1111", "research_contextual_signature_trace"),
    ("research_contextual_marginal_signature_v1112", "research_contextual_signature_progress"),
    ("research_contextual_marginal_signature_v1113", "research_contextual_signature_sound"),
    ("research_contextual_marginal_signature_v1114", "research_contextual_signature_diagnostics"),
    ("research_contextual_marginal_signature_v1144", "research_contextual_signature_frozen_schedule"),
    ("research_contextual_marginal_signature_v116", "research_contextual_signature_paired_action"),
    ("research_contextual_marginal_signature_v117", "research_contextual_signature_temporal"),
    ("research_contextual_marginal_signature_v1172", "research_contextual_signature_campaign"),
)


def load_campaign_module() -> ModuleType:
    loaded: ModuleType | None = None
    for legacy_name, stable_name in _MODULE_CHAIN:
        module = importlib.import_module(stable_name)
        sys.modules[legacy_name] = module
        loaded = module
    if loaded is None:
        raise RuntimeError("Contextual signature module chain is empty.")
    return loaded
