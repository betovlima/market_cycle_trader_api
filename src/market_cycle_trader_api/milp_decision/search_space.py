from __future__ import annotations

from typing import Any

from .config import DEFAULT_CONFIGURATION, SEARCH_CANDIDATE_COUNT, SEARCH_LEVELS

_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0 / float(base)
    value = int(index)
    while value > 0:
        result += fraction * float(value % base)
        value //= base
        fraction /= float(base)
    return result


def _signature(configuration: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(configuration[key] for key in SEARCH_LEVELS)


def configurations(count: int = SEARCH_CANDIDATE_COUNT) -> list[dict[str, Any]]:
    target = max(1, int(count))
    rows: list[dict[str, Any]] = [{"candidate_id": "baseline", "configuration": dict(DEFAULT_CONFIGURATION)}]
    seen = {_signature(DEFAULT_CONFIGURATION)}
    sequence = 1
    while len(rows) < target:
        configuration: dict[str, Any] = {}
        for position, (key, levels) in enumerate(SEARCH_LEVELS.items()):
            value = _radical_inverse(sequence, _BASES[position])
            index = min(len(levels) - 1, int(value * len(levels)))
            configuration[key] = levels[index]
        signature = _signature(configuration)
        if signature not in seen:
            seen.add(signature)
            rows.append({
                "candidate_id": f"design-{len(rows):03d}",
                "configuration": configuration,
            })
        sequence += 1
    return rows
