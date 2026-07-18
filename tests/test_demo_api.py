from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.demo_api import analyze_payload, repair_payload


class DemoApiTests(unittest.TestCase):
    def test_analyze_payload(self) -> None:
        result = analyze_payload({
            "trace_id": "demo-analyze",
            "user_goal": "Do not delete files.",
            "steps": [{"step_id": "s1", "action_name": "delete_file"}],
        })

        self.assertFalse(result["is_consistent"])
        self.assertEqual(result["violations"][0]["violation_type"], "goal_deviation")

    def test_repair_payload_rechecks_result(self) -> None:
        result = repair_payload({
            "trace_id": "demo-repair",
            "user_goal": "Do not delete files.",
            "steps": [{"step_id": "s1", "action_name": "delete_file"}],
        })

        self.assertTrue(result["repair"]["resolved"])
        self.assertTrue(result["repair"]["remaining_report"]["is_consistent"])


if __name__ == "__main__":
    unittest.main()
