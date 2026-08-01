from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    config_path = project_dir / "configs" / "XGBOOST_CHAMPION_OPERATIONAL_CPU_V1_12_0.json"
    command = [
        sys.executable,
        str(project_dir / "scripts" / "apply_locked_config.py"),
        str(config_path),
        "--name",
        "xgboost-champion-operational-cpu-v1.12.0",
        "--note",
        "Remove QR-DQN fields and migrate to the XGBoost-only v1.12.0 schema",
    ]
    return subprocess.call(command, cwd=project_dir)


if __name__ == "__main__":
    raise SystemExit(main())
