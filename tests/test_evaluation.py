from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.evaluation import evaluate_reports
from llm_logic_guard.models import AnalysisReport, Violation


class EvaluationTests(unittest.TestCase):
    def test_type_level_metrics(self) -> None:
        reports = [
            AnalysisReport(
                trace_id="t1",
                is_consistent=False,
                violations=[
                    Violation("v1", "self_contradiction", "medium", "s1", "message"),
                    Violation("v2", "goal_deviation", "high", "s2", "message"),
                ],
                corrections=[],
            ),
            AnalysisReport(trace_id="t2", is_consistent=True, violations=[], corrections=[]),
        ]
        metrics = evaluate_reports(
            reports,
            {
                "t1": {"self_contradiction", "tool_result_final_mismatch"},
                "t2": set(),
            },
        )
        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_positives, 1)
        self.assertEqual(metrics.false_negatives, 1)
        self.assertEqual(metrics.true_negatives, 3)
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertEqual(metrics.false_positive_rate, 0.25)
        self.assertEqual(metrics.trace_accuracy, 0.5)
        self.assertAlmostEqual(metrics.macro_f1, 1 / 3)
        self.assertEqual(metrics.per_type["self_contradiction"]["support"], 1)


if __name__ == "__main__":
    unittest.main()
