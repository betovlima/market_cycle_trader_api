from __future__ import annotations

import math
from typing import Any

from .errors import MilpDecisionError


def solve_binary_one_hot(alternatives: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    feasible = [item for item in alternatives if bool(item.get("eligible", True))]
    if not feasible:
        raise MilpDecisionError("MILP has no feasible decision alternative.")
    ordered = sorted(feasible, key=lambda item: (-float(item["objective"]), str(item["symbol"])))
    best: dict[str, Any] | None = None
    best_objective = -math.inf
    explored = 0
    pruned = 0

    def branch(index: int, selected: dict[str, Any] | None) -> None:
        nonlocal best, best_objective, explored, pruned
        explored += 1
        if selected is not None:
            upper_bound = float(selected["objective"])
        elif index < len(ordered):
            upper_bound = max(float(item["objective"]) for item in ordered[index:])
        else:
            return
        if upper_bound <= best_objective + 1e-15:
            pruned += 1
            return
        if selected is not None:
            if float(selected["objective"]) > best_objective + 1e-15:
                best = selected
                best_objective = float(selected["objective"])
            return
        branch(index + 1, ordered[index])
        branch(index + 1, None)

    branch(0, None)
    if best is None:
        raise MilpDecisionError("MILP solver did not find a feasible binary solution.")
    return best, {"nodes_explored": explored, "nodes_pruned": pruned}
