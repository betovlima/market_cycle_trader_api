from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import research_contextual_marginal_signature_v1114 as research  # noqa: E402


class ContextualMarginalSignatureV1114Tests(unittest.TestCase):
    def test_sidecar_paths_are_siblings_of_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "contextual_marginal_signature_v1114"
            state, fatal = research._sidecar_paths(output)
            self.assertEqual(state.parent, output.parent.resolve())
            self.assertEqual(fatal.parent, output.parent.resolve())
            self.assertEqual(state.name, "contextual_marginal_signature_v1114.run_state.json")
            self.assertEqual(fatal.name, "contextual_marginal_signature_v1114.fatal.log")

    def test_atomic_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            research._atomic_json(path, {"status": "running", "pid": 123})
            first = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(first["status"], "running")
            research._atomic_json(path, {"status": "completed", "pid": 123})
            second = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(second, {"status": "completed", "pid": 123})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
