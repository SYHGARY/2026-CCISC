from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.models import LogicTrace
from llm_logic_guard.pipeline import LogicGuardPipeline
from llm_logic_guard.repair import LogicRepairer


class RepairTests(unittest.TestCase):
    def test_blocks_and_resolves_irreversible_action(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "repair-block",
            "user_goal": "Only read a.txt. Do not delete it.",
            "steps": [
                {"step_id": "s1", "role": "tool_call", "action_name": "delete_file"},
            ],
        })

        result = LogicRepairer().apply(trace)

        self.assertTrue(result.resolved)
        self.assertEqual(result.applied_interventions, ["block"])
        self.assertIsNone(result.repaired_trace.steps[0].action_name)
        self.assertEqual(result.repaired_trace.steps[0].metadata["blocked_action"], "delete_file")

    def test_revises_failed_tool_success_claim(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "repair-final",
            "user_goal": "Read report.txt.",
            "steps": [
                {
                    "step_id": "s1",
                    "role": "tool_call",
                    "action_name": "read_file",
                    "tool_result": "File not found: report.txt",
                },
                {"step_id": "s2", "role": "final", "content": "I successfully read report.txt."},
            ],
        })

        result = LogicRepairer().apply(trace)

        self.assertTrue(result.resolved)
        self.assertEqual(result.applied_interventions, ["revise"])
        self.assertIn("File not found", result.repaired_trace.steps[1].content)
        self.assertEqual(
            result.repaired_trace.steps[1].metadata["original_content"],
            "I successfully read report.txt.",
        )

    def test_soft_contradiction_remains_for_self_check(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "repair-self-check",
            "user_goal": "Check a.txt.",
            "steps": [
                {"step_id": "s1", "role": "assistant", "content": "a.txt exists."},
                {"step_id": "s2", "role": "assistant", "content": "a.txt does not exist."},
            ],
        })

        result = LogicRepairer().apply(trace, LogicGuardPipeline().run(trace))

        self.assertFalse(result.resolved)
        self.assertEqual(result.applied_interventions, [])

    def test_human_confirmation_does_not_execute_proposed_action(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "repair-confirm",
            "user_goal": "Do not send data externally.",
            "steps": [
                {
                    "step_id": "s1",
                    "role": "tool_call",
                    "action_name": "send_email",
                    "action_args": {"to": "external@example.com", "body": "data"},
                },
            ],
        })

        result = LogicRepairer().apply(trace)

        self.assertTrue(result.resolved)
        self.assertEqual(result.applied_interventions, ["human_confirm"])
        self.assertEqual(
            result.repaired_trace.steps[0].action_name,
            "request_human_confirmation",
        )

    def test_cascading_repair_revises_final_after_block(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "repair-cascade",
            "user_goal": "Only read a.txt. Do not delete it.",
            "steps": [
                {"step_id": "s1", "role": "tool_call", "action_name": "delete_file"},
                {"step_id": "s2", "role": "final", "content": "I successfully deleted a.txt."},
            ],
        })

        result = LogicRepairer().apply(trace)

        self.assertTrue(result.resolved)
        self.assertEqual(result.applied_interventions, ["block", "revise"])
        self.assertIn("blocked before execution", result.repaired_trace.steps[1].content)


if __name__ == "__main__":
    unittest.main()
