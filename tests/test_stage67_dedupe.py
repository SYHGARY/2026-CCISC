from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.hybrid import _dedupe
from llm_logic_guard.models import Violation


def _violation(vid: str, *, step_id: str, vtype: str, spec_id=None, confidence=0.75) -> Violation:
    return Violation(
        violation_id=vid,
        violation_type=vtype,
        severity="medium",
        step_id=step_id,
        message="test",
        confidence=confidence,
        spec_id=spec_id,
    )


class DedupeTests(unittest.TestCase):
    def test_distinct_semantic_conflicts_on_same_step_are_kept(self) -> None:
        # One step contradicting two different prior steps: distinct findings,
        # both must survive dedupe (regression for the (step,type,"") collision).
        a = _violation(
            "t:s3:semantic:s1", step_id="s3", vtype="semantic_contradiction", confidence=0.9
        )
        b = _violation(
            "t:s3:semantic:s2", step_id="s3", vtype="semantic_contradiction", confidence=0.8
        )
        result = _dedupe([a, b])
        self.assertEqual({v.violation_id for v in result}, {a.violation_id, b.violation_id})

    def test_identical_semantic_violation_id_still_dedupes_to_highest_confidence(self) -> None:
        low = _violation(
            "t:s3:semantic:s1", step_id="s3", vtype="semantic_contradiction", confidence=0.6
        )
        high = _violation(
            "t:s3:semantic:s1", step_id="s3", vtype="semantic_contradiction", confidence=0.95
        )
        result = _dedupe([low, high])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.95)

    def test_spec_engine_violations_still_dedupe_on_spec_id(self) -> None:
        first = _violation(
            "id-a", step_id="s1", vtype="policy", spec_id="untrusted_boundary", confidence=0.7
        )
        second = _violation(
            "id-b", step_id="s1", vtype="policy", spec_id="untrusted_boundary", confidence=0.9
        )
        result = _dedupe([first, second])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
