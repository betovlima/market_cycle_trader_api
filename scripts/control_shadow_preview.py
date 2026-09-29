"""Run a read-only frozen-data Control shadow decision from the PyCharm terminal.

Examples (PowerShell, API repository root):
  $env:PYTHONPATH = "src"
  python scripts/control_shadow_preview.py --frozen-root "C:/path/to/tcc/dados/pesquisa" --completed-session 2026-09-17 --current-asset CASH

No Alpaca connection, MongoDB reads/writes, Strategy selection, paper plan or
orders occur. This script only creates the specified local JSON output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from market_cycle_trader_api.engine.operational_control_preview import (
    build_control_shadow_decision,
)
from market_cycle_trader_api.engine.tcc_frozen_reference_source import (
    FROZEN_TCC_MAIN_SHA256,
    load_frozen_tcc_main_symbol,
    validate_frozen_tcc_main,
)
from market_cycle_trader_api.tcc_v106_reference.config import ASSETS
from market_cycle_trader_api.engine.research_market_data import (
    StructuralResearchAssetExclusion,
)


def run(
    *,
    frozen_root: Path,
    completed_session: str,
    current_asset: str,
    holding_sessions: int,
    output: Path,
) -> dict:
    frozen_root = frozen_root.expanduser().resolve(strict=True)
    manifest = validate_frozen_tcc_main(frozen_root, assets=ASSETS)
    frames = {}
    exclusions = []
    for symbol in ASSETS:
        try:
            frames[symbol] = load_frozen_tcc_main_symbol(frozen_root, symbol, manifest)
        except StructuralResearchAssetExclusion as exc:
            exclusions.append({"symbol": symbol, "reason": str(exc)})
    preview = build_control_shadow_decision(
        frames,
        completed_session=completed_session,
        current_asset=current_asset,
        holding_sessions=holding_sessions,
    )
    preview["source_validation"] = "sha256_verified_frozen_tcc_main_raw_sip_and_corporate_actions"
    preview["input_audit"]["snapshot_sha256"] = FROZEN_TCC_MAIN_SHA256
    preview["input_audit"]["structural_exclusions"] = exclusions
    preview["input_audit"]["source_kind"] = "frozen_scientific_reference_not_current_alpaca"
    preview["input_audit"]["eligible_assets"] = len(frames)
    preview["order_eligible"] = False
    preview["order_submission"] = "never"
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(preview, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return preview


def main() -> int:
    parser = argparse.ArgumentParser(description="TCC Control no-order shadow decision")
    parser.add_argument("--frozen-root", type=Path, required=True, help="Verified TCC dados/pesquisa root")
    parser.add_argument("--completed-session", required=True, help="Completed YYYY-MM-DD XNYS session")
    parser.add_argument("--current-asset", default="CASH", help="CASH or a position symbol")
    parser.add_argument("--holding-sessions", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("output/control_shadow_preview.json"))
    args = parser.parse_args()
    result = run(
        frozen_root=args.frozen_root,
        completed_session=args.completed_session,
        current_asset=args.current_asset,
        holding_sessions=args.holding_sessions,
        output=args.output,
    )
    print(json.dumps({
        "status": result["status"],
        "decision_date": result["decision_date"],
        "current_asset": result["current_asset"],
        "target_asset": result["target_asset"],
        "effective_switch_margin": result["effective_switch_margin"],
        "output": str(args.output),
        "order_submission": "never",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
