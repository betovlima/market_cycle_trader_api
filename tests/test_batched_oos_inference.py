from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from market_cycle_trader_api.engine.capital_rotation import (
    ROTATION_FEATURES,
    _model_utilities,
    _precompute_model_utilities,
)


class _LinearModel:
    def __init__(self, scale: float, bias: float) -> None:
        self.scale = float(scale)
        self.bias = float(bias)
        self.calls = 0

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self.calls += 1
        first = pd.to_numeric(
            frame[ROTATION_FEATURES[0]],
            errors="coerce",
        ).to_numpy(dtype=float)
        return self.bias + self.scale * first


def _frames() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(
        [
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ],
        utc=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    for offset, symbol in enumerate(("AAA", "BBB"), start=1):
        frame = pd.DataFrame(
            0.0,
            index=index,
            columns=ROTATION_FEATURES,
        )
        frame[ROTATION_FEATURES[0]] = (
            np.arange(len(index), dtype=float) + float(offset)
        )
        frame["open"] = 100.0 + float(offset)
        frame["close"] = 101.0 + float(offset)
        frames[symbol] = frame
    return frames


def test_batched_utility_cache_matches_scalar_predictions() -> None:
    frames = _frames()
    dates = pd.DatetimeIndex(
        list(frames["AAA"].index[:3])
    )
    scalar_models = {
        "AAA": _LinearModel(0.10, 0.01),
        "BBB": _LinearModel(-0.05, 0.20),
    }
    batch_models = {
        "AAA": _LinearModel(0.10, 0.01),
        "BBB": _LinearModel(-0.05, 0.20),
    }
    config = SimpleNamespace()

    expected = {
        pd.Timestamp(date): _model_utilities(
            scalar_models,
            frames,
            ["AAA", "BBB"],
            pd.Timestamp(date),
            config,
        )
        for date in dates
    }

    cache, profile = _precompute_model_utilities(
        batch_models,
        frames,
        ["AAA", "BBB"],
        dates,
        config,
    )

    for date in dates:
        actual = _model_utilities(
            batch_models,
            frames,
            ["AAA", "BBB"],
            pd.Timestamp(date),
            config,
            utility_cache=cache,
        )
        assert np.allclose(
            actual,
            expected[pd.Timestamp(date)],
            rtol=0.0,
            atol=1e-15,
        )

    assert batch_models["AAA"].calls == 1
    assert batch_models["BBB"].calls == 1
    assert scalar_models["AAA"].calls == len(dates)
    assert scalar_models["BBB"].calls == len(dates)
    assert profile["cache_predict_calls"] == 2
    assert profile["cache_session_count"] == 3
    assert profile["cache_predicted_rows"] == 6


def test_cached_lookup_does_not_call_model_again() -> None:
    frames = _frames()
    model = _LinearModel(0.20, 0.0)
    models = {"AAA": model}
    dates = pd.DatetimeIndex(
        list(frames["AAA"].index[:3])
    )
    config = SimpleNamespace()

    cache, _ = _precompute_model_utilities(
        models,
        frames,
        ["AAA"],
        dates,
        config,
    )
    calls_after_build = model.calls

    for date in dates:
        _model_utilities(
            models,
            frames,
            ["AAA"],
            pd.Timestamp(date),
            config,
            utility_cache=cache,
        )

    assert model.calls == calls_after_build
