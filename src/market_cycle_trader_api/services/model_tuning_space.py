from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

ENTRY_THRESHOLD = "opportunity_utility_entry_threshold"
EXIT_THRESHOLD = "opportunity_utility_exit_threshold"


def sample_value(spec: dict[str, Any], unit_value: float) -> Any:
    low = float(spec["min"])
    high = float(spec["max"])
    value = low + float(unit_value) * (high - low)
    if spec["type"] == "integer":
        return int(round(value))
    return round(value, int(spec.get("precision") or 8))


def normalize_tuning_values(
    values: dict[str, Any],
    search_space: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Normalize deterministic structural constraints without collapsing samples onto a boundary."""
    active_space = list(search_space)
    normalized = deepcopy(values)
    names = {str(item.get("name") or "") for item in active_space}

    if {"max_depth", "num_leaves"}.issubset(names):
        depth = int(normalized.get("max_depth") or 0)
        if depth > 0 and "num_leaves" in normalized:
            normalized["num_leaves"] = min(int(normalized["num_leaves"]), 2 ** depth)
            normalized["num_leaves"] = max(2, int(normalized["num_leaves"]))

    # This is intentionally only a final safety guard. New sampling is generated
    # directly inside the valid EXIT <= ENTRY domain by settings_from_unit_point().
    if {ENTRY_THRESHOLD, EXIT_THRESHOLD}.issubset(names):
        entry = float(normalized[ENTRY_THRESHOLD])
        exit_ = float(normalized[EXIT_THRESHOLD])
        if exit_ > entry:
            normalized[EXIT_THRESHOLD] = entry
    return normalized


def settings_from_unit_point(
    base_values: dict[str, Any],
    search_space: Sequence[dict[str, Any]] | Iterable[dict[str, Any]],
    point: Sequence[float],
) -> dict[str, Any]:
    """Map a unit-cube point to settings while sampling constrained pairs in-domain.

    ENTRY is sampled first. EXIT then uses its own Latin-Hypercube coordinate but
    is drawn directly from [exit_min, min(exit_max, entry)]. This avoids the old
    projection `exit = entry`, which accumulated invalid samples on the diagonal.
    """
    active_space = list(search_space)
    if len(active_space) != len(point):
        raise ValueError("Search-space dimension and unit-point dimension must match.")

    values = deepcopy(base_values)
    unit_by_name = {
        str(spec["name"]): float(unit_value)
        for spec, unit_value in zip(active_space, point, strict=True)
    }
    spec_by_name = {str(spec["name"]): spec for spec in active_space}

    # First sample every parameter except EXIT using its ordinary marginal.
    for spec in active_space:
        name = str(spec["name"])
        if name == EXIT_THRESHOLD and ENTRY_THRESHOLD in spec_by_name:
            continue
        values[name] = sample_value(spec, unit_by_name[name])

    # Then sample EXIT conditionally inside the triangular valid domain.
    if EXIT_THRESHOLD in spec_by_name and ENTRY_THRESHOLD in spec_by_name:
        exit_spec = spec_by_name[EXIT_THRESHOLD]
        entry = float(values[ENTRY_THRESHOLD])
        low = float(exit_spec["min"])
        high = min(float(exit_spec["max"]), entry)
        if high < low:
            raise ValueError(
                f"Invalid tuning domain: {EXIT_THRESHOLD} minimum {low} exceeds sampled {ENTRY_THRESHOLD} {entry}."
            )
        raw = low + unit_by_name[EXIT_THRESHOLD] * (high - low)
        values[EXIT_THRESHOLD] = round(raw, int(exit_spec.get("precision") or 8))

    return normalize_tuning_values(values, active_space)
