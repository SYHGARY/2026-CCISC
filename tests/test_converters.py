from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.converters import convert_security_events


class SecurityConverterTests(unittest.TestCase):
    def test_merges_before_and_after_action_events(self) -> None:
        events = [
            {
                "event_id": "before-1",
                "trace_id": "trace-1",
                "session_id": "session-1",
                "agent_id": "executor",
                "phase": "before_action",
                "action_name": "read_file",
                "action_args": {"filename": "a.txt"},
                "timestamp": "2026-01-01T00:00:00Z",
                "is_safe_case": True,
            },
            {
                "event_id": "after-1",
                "trace_id": "trace-1",
                "session_id": "session-1",
                "agent_id": "executor",
                "phase": "after_action",
                "action_name": "read_file",
                "action_args": {"filename": "a.txt"},
                "result": "hello",
                "timestamp": "2026-01-01T00:00:01Z",
                "is_safe_case": True,
            },
        ]

        traces = convert_security_events(events, user_goal="Read a.txt.")

        self.assertEqual(len(traces), 1)
        self.assertEqual(len(traces[0].steps), 1)
        self.assertEqual(traces[0].steps[0].tool_result, "hello")
        self.assertEqual(traces[0].steps[0].metadata["after_event_id"], "after-1")
        self.assertEqual(traces[0].metadata["source_event_count"], 2)
        self.assertFalse(traces[0].metadata["source_safety_labels_are_logic_labels"])

    def test_keeps_unmatched_after_event(self) -> None:
        traces = convert_security_events([
            {
                "event_id": "after-only",
                "trace_id": "trace-1",
                "phase": "after_action",
                "action_name": "read_file",
                "result": "hello",
            }
        ])

        self.assertEqual(len(traces[0].steps), 1)
        self.assertEqual(traces[0].steps[0].tool_result, "hello")

    def test_groups_multiple_traces(self) -> None:
        traces = convert_security_events([
            {"event_id": "a", "trace_id": "trace-a", "phase": "before_action"},
            {"event_id": "b", "trace_id": "trace-b", "phase": "before_action"},
        ])

        self.assertEqual({trace.trace_id for trace in traces}, {"trace-a", "trace-b"})


if __name__ == "__main__":
    unittest.main()
