from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.claims import extract_step_claims
from llm_logic_guard.models import LogicTrace, TraceStep
from llm_logic_guard.pipeline import LogicGuardPipeline


class ClaimTests(unittest.TestCase):
    def test_extracts_structured_claim_with_source(self) -> None:
        claims = extract_step_claims(
            TraceStep(step_id="s1", role="tool", tool_result="notes.md does not exist.")
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].subject, "notes.md")
        self.assertEqual(claims[0].predicate, "file_exists")
        self.assertEqual(claims[0].polarity, "negative")
        self.assertEqual(claims[0].source, "tool_result")

    def test_different_files_do_not_contradict(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "different-files",
            "user_goal": "Inspect files.",
            "steps": [
                {"step_id": "s1", "role": "assistant", "content": "a.txt exists."},
                {"step_id": "s2", "role": "assistant", "content": "b.txt does not exist."},
            ],
        })

        self.assertTrue(LogicGuardPipeline().run(trace).is_consistent)

    def test_state_change_invalidates_previous_file_claim(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "state-change",
            "user_goal": "Delete a.txt.",
            "steps": [
                {"step_id": "s1", "role": "assistant", "content": "a.txt exists."},
                {
                    "step_id": "s2",
                    "role": "tool_call",
                    "action_name": "delete_file",
                    "action_args": {"filename": "a.txt"},
                    "tool_result": "Deleted a.txt",
                },
                {"step_id": "s3", "role": "assistant", "content": "a.txt does not exist."},
            ],
        })

        report = LogicGuardPipeline().run(trace)
        self.assertFalse(any(v.violation_type == "self_contradiction" for v in report.violations))

    def test_failed_state_change_does_not_invalidate_claim(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "failed-state-change",
            "user_goal": "Inspect a.txt.",
            "steps": [
                {"step_id": "s1", "role": "assistant", "content": "a.txt exists."},
                {
                    "step_id": "s2",
                    "role": "tool_call",
                    "action_name": "delete_file",
                    "action_args": {"filename": "a.txt"},
                    "tool_result": "Delete failed",
                },
                {"step_id": "s3", "role": "assistant", "content": "a.txt does not exist."},
            ],
        })

        report = LogicGuardPipeline().run(trace)
        self.assertTrue(any(v.violation_type == "self_contradiction" for v in report.violations))


if __name__ == "__main__":
    unittest.main()
