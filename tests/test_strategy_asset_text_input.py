from pathlib import Path
import json

import pytest
from pydantic import ValidationError

from market_cycle_trader_api.schemas.requests import BacktestRequest, normalize_assets_input
from market_cycle_trader_api.schemas.strategy_lab import StrategyUpdateRequest


def _configuration() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "market_cycle_trader_api"
        / "parameterizations"
        / "winner-v1.13.2.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_plain_asset_text_builds_normalized_unique_backend_list() -> None:
    assert normalize_assets_input(" nvda, AAPL  msft\nNVDA ; brk.b ") == [
        "NVDA",
        "AAPL",
        "MSFT",
        "BRK.B",
    ]


def test_legacy_json_style_paste_is_tolerated_by_backend_parser() -> None:
    assert normalize_assets_input('["NVDA", "AAPL", "MSFT"]') == ["NVDA", "AAPL", "MSFT"]


def test_strategy_update_constructs_assets_on_backend() -> None:
    configuration = _configuration()
    configuration.pop("assets", None)
    request = StrategyUpdateRequest(
        expected_revision=1,
        configuration=configuration,
        assets_input="NVDA, AAPL, MSFT",
        name="Test strategy",
        description="",
        note="Update asset universe",
    )

    built = request.build_configuration()

    assert isinstance(built, BacktestRequest)
    assert built.assets == ["NVDA", "AAPL", "MSFT"]
    assert "assets" not in request.configuration


def test_strategy_update_rejects_invalid_asset_text() -> None:
    configuration = _configuration()
    configuration.pop("assets", None)
    with pytest.raises(ValidationError):
        StrategyUpdateRequest(
            expected_revision=1,
            configuration=configuration,
            assets_input="NVDA, AAPL, BAD/SYMBOL",
            name="Test strategy",
            description="",
            note="Update asset universe",
        )


def test_legacy_strategy_update_with_assets_array_remains_supported() -> None:
    configuration = _configuration()
    request = StrategyUpdateRequest(
        expected_revision=1,
        configuration=configuration,
        name="Legacy compatible strategy",
        description="",
        note="Keep old client compatibility",
    )

    built = request.build_configuration()

    assert built.assets == configuration["assets"]
