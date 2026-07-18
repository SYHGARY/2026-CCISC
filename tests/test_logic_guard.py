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


class LogicGuardTests(unittest.TestCase):
    def test_plan_action_contradiction(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t1",
            "user_goal": "Only read the file. Do not delete it.",
            "steps": [
                {"step_id": "s1", "role": "assistant", "content": "I will not delete the file."},
                {"step_id": "s2", "role": "tool_call", "action_name": "delete_file", "action_args": {"filename": "a.txt"}},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertFalse(report.is_consistent)
        self.assertTrue(any(v.violation_type == "plan_action_contradiction" for v in report.violations))
        self.assertTrue(any(c.intervention == "block" for c in report.corrections))

    def test_tool_result_final_mismatch(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t2",
            "user_goal": "Read report.txt.",
            "steps": [
                {"step_id": "s1", "role": "tool_call", "action_name": "read_file", "tool_result": "File not found"},
                {"step_id": "s2", "role": "final", "content": "I successfully read the file."},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertFalse(report.is_consistent)
        self.assertTrue(any(v.violation_type == "tool_result_final_mismatch" for v in report.violations))
        self.assertTrue(any(c.intervention == "revise" for c in report.corrections))
        self.assertFalse(any(c.intervention == "block" for c in report.corrections))
        self.assertFalse(any(v.violation_type == "self_contradiction" for v in report.violations))

    def test_negated_success_is_not_a_success_claim(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_negated_success",
            "user_goal": "Read report.txt.",
            "steps": [
                {
                    "step_id": "s1",
                    "role": "tool_call",
                    "action_name": "read_file",
                    "tool_result": "File not found",
                },
                {
                    "step_id": "s2",
                    "role": "final",
                    "content": "The request did not complete successfully.",
                },
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertTrue(report.is_consistent)

    def test_negative_phrase_is_not_also_positive(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_negative_phrase",
            "user_goal": "Check whether report.txt exists.",
            "steps": [
                {"step_id": "s1", "role": "tool", "tool_result": "File not found: report.txt"},
                {"step_id": "s2", "role": "final", "content": "report.txt does not exist."},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertTrue(report.is_consistent)

    def test_soft_self_contradiction_requests_self_check(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_soft",
            "user_goal": "Check whether the file exists.",
            "steps": [
                {"step_id": "s1", "role": "assistant", "content": "The file exists."},
                {"step_id": "s2", "role": "assistant", "content": "The file does not exist."},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertFalse(report.is_consistent)
        self.assertTrue(any(c.intervention == "self_check" for c in report.corrections))
        self.assertFalse(any(c.intervention == "block" for c in report.corrections))

    def test_duplicate_block_corrections_are_collapsed(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_duplicate",
            "user_goal": "Only read the file. Do not delete it.",
            "steps": [
                {"step_id": "s1", "role": "assistant", "content": "I will not delete the file."},
                {"step_id": "s2", "role": "tool_call", "action_name": "delete_file"},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        block_corrections = [c for c in report.corrections if c.intervention == "block"]
        self.assertEqual(len(block_corrections), 1)

    def test_no_delete_does_not_forbid_write(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_narrow_constraint",
            "user_goal": "Create and read report.txt. Do not delete files.",
            "steps": [
                {"step_id": "s1", "role": "tool_call", "action_name": "write_file"},
                {"step_id": "s2", "role": "tool_call", "action_name": "read_file"},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertTrue(report.is_consistent)

    def test_chinese_no_delete_constraint(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_chinese_delete",
            "user_goal": "检查报告，但不要删除或移除任何文件。",
            "steps": [
                {"step_id": "s1", "role": "tool_call", "action_name": "delete_file"},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertTrue(any(v.violation_type == "goal_deviation" for v in report.violations))

    def test_tool_claim_conflict_uses_specific_violation_type(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_email_mismatch",
            "user_goal": "Send the email.",
            "steps": [
                {
                    "step_id": "s1",
                    "role": "tool_call",
                    "action_name": "send_email",
                    "tool_result": "Email blocked by policy",
                },
                {"step_id": "s2", "role": "final", "content": "The email was successfully sent."},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertTrue(any(v.violation_type == "tool_result_final_mismatch" for v in report.violations))
        self.assertFalse(any(v.violation_type == "self_contradiction" for v in report.violations))

    def test_only_create_and_read_is_not_read_only(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t_create_and_read",
            "user_goal": "Only create and read report.txt.",
            "steps": [
                {"step_id": "s1", "role": "tool_call", "action_name": "write_file"},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertTrue(report.is_consistent)

    def test_consistent_trace(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "t3",
            "user_goal": "Read report.txt.",
            "steps": [
                {"step_id": "s1", "role": "tool_call", "action_name": "read_file", "tool_result": "hello"},
                {"step_id": "s2", "role": "final", "content": "The file contains hello."},
            ],
        })
        report = LogicGuardPipeline().run(trace)
        self.assertTrue(report.is_consistent)


if __name__ == "__main__":
    unittest.main()
