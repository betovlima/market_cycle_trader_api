"""Identity of the TCC U67 v1.21.0 source migrated into MCT.

This module records the exact scientific source. It does not make the strategy
Trader-eligible by itself.
"""
from __future__ import annotations

from .configuracao import ASSETS as U56_ASSETS

SOURCE_REPOSITORY = "betovlima/tcc_mba_usp_data_science_analytics"
SOURCE_BRANCH = "main"
SOURCE_COMMIT = "4b5f16030afa8b850790747bb0e3e3063d233e79"
REPRODUCTION_VERSION = "1.21.0"
EXECUTION_SCHEMA = "u67-control-reproduction-v1"
EXPECTED_ENDING_CAPITAL = 58_557_157.67496595

U59_ADDITIONS = ("COLB", "AMS", "FOXF")
U67_ADDITIONS = (
    "THO",
    "WDAY",
    "EXR",
    "XEL",
    "SBFG",
    "PAYX",
    "MUX",
    "SXC",
)
U67_REQUESTED_ASSETS = (
    *tuple(U56_ASSETS),
    *U59_ADDITIONS,
    *U67_ADDITIONS,
)

SOURCE_BLOBS = {
    "engine/__init__.py": "792590b40120dc8eb093e6a999f1126a5e3f69c1",
    "engine/configuracao.py": "7be43f34581a950682a0952bc0395ce247e8f8f7",
    "engine/diagnosticos.py": "3cf8910eb659b26fe88f45bdfbda5e25a94408c3",
    "engine/execucao.py": "6cb7798e821ed0b6414262fb762ec20e67b58263",
    "engine/modelo_lightgbm.py": "116b5edd9eaeed59e071c7f1e741563a05005443",
    "engine/rotacao.py": "7277ba0d49723f15333348a5bd4aa000eca4face",
}
