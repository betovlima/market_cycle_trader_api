from __future__ import annotations

import math
from typing import Any

import numpy as np


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def equity_metrics(values: list[float], initial_capital: float) -> dict[str, Any]:
    if not values or initial_capital <= 0:
        return {}
    equity = np.asarray(values, dtype=float)
    returns = np.empty(len(equity), dtype=float)
    returns[0] = equity[0] / float(initial_capital) - 1.0
    if len(equity) > 1:
        returns[1:] = np.divide(equity[1:], equity[:-1], out=np.ones(len(equity) - 1), where=equity[:-1] != 0.0) - 1.0
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = float(np.mean(returns) / volatility * math.sqrt(252.0)) if volatility > 1e-12 else None
    ending = float(equity[-1])
    years = max(len(equity) / 252.0, 1.0 / 252.0)
    cagr = (ending / float(initial_capital)) ** (1.0 / years) - 1.0 if ending > 0 else -1.0
    return {
        "initial_capital": float(initial_capital),
        "ending_capital": ending,
        "total_return": ending / float(initial_capital) - 1.0,
        "cagr": finite(cagr),
        "sharpe": finite(sharpe),
        "max_drawdown": finite(float(np.min(drawdown))),
        "equity_sessions": int(len(equity)),
    }


def metric_delta(challenger: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("ending_capital", "total_return", "cagr", "sharpe", "max_drawdown"):
        left = finite(challenger.get(key))
        right = finite(control.get(key))
        result[key] = (left - right) if left is not None and right is not None else None
    base = finite(control.get("ending_capital"))
    candidate = finite(challenger.get("ending_capital"))
    result["ending_capital_rate"] = (candidate / base - 1.0) if candidate is not None and base not in {None, 0.0} else None
    return result
