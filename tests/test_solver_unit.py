"""Unit tests for the structural constraint solver (core IP).

These pin solver behaviour in isolation, complementing the integration coverage
in test_stage3_consistency. They verify value-conflict detection, temporal-scope
partitioning, and untrusted-permission-grant detection independent of the full
hybrid pipeline.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.models import Claim
from llm_logic_guard.solver import StructuredConstraintSolver


def _claim(
    subject: str,
    predicate: str,
    polarity: str,
    *,
    trust: float = 0.9,
    temporal_scope: str = "current",
    event_source: str = "user",
    step_id: str = "s",
) -> Claim:
    return Claim(
        subject=subject,
        predicate=predicate,
        polarity=polarity,
        evidence=f"{predicate}:{subject}={polarity}",
        step_id=step_id,
        source=event_source,
        trust=trust,
        provenance={"temporal_scope": temporal_scope, "event_source": event_source},
    )


class ConstraintSolverPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = StructuredConstraintSolver()

    def test_opposite_polarity_same_key_and_scope_is_inconsistent(self) -> None:
        left = _claim("report.txt", "file_exists", "positive")
        right = _claim("report.txt", "file_exists", "negative")
        self.assertFalse(self.solver.check_claim_pair(left, right).consistent)

    def test_same_polarity_is_consistent(self) -> None:
        left = _claim("report.txt", "file_exists", "positive")
        right = _claim("report.txt", "file_exists", "positive")
        self.assertTrue(self.solver.check_claim_pair(left, right).consistent)

    def test_different_keys_are_consistent(self) -> None:
        left = _claim("report.txt", "file_exists", "positive")
        right = _claim("secret.txt", "file_exists", "negative")
        self.assertTrue(self.solver.check_claim_pair(left, right).consistent)

    def test_different_temporal_scope_is_not_a_conflict(self) -> None:
        past = _claim("report.txt", "file_exists", "negative", temporal_scope="past")
        current = _claim("report.txt", "file_exists", "positive", temporal_scope="current")
        self.assertTrue(self.solver.check_claim_pair(past, current).consistent)


class ConstraintSolverClaimsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.solver = StructuredConstraintSolver()

    def test_value_conflict_detected_across_claim_list(self) -> None:
        claims = [
            _claim("report.txt", "file_exists", "positive", step_id="s1"),
            _claim("report.txt", "file_exists", "negative", step_id="s2"),
        ]
        issues = self.solver.check_claims(claims)
        self.assertTrue(any(i.issue_type == "value_conflict" for i in issues))

    def test_invalidated_claim_is_skipped(self) -> None:
        earlier = _claim("report.txt", "file_exists", "positive", step_id="s1")
        earlier.invalidated_by = "s2"
        later = _claim("report.txt", "file_exists", "negative", step_id="s3")
        issues = self.solver.check_claims([earlier, later])
        self.assertFalse(any(i.issue_type == "value_conflict" for i in issues))

    def test_untrusted_permission_grant_flagged_by_source(self) -> None:
        claim = _claim(
            "external_send",
            "permission_grant",
            "positive",
            trust=0.9,
            event_source="memory",
        )
        issues = self.solver.check_claims([claim])
        self.assertTrue(
            any(i.issue_type == "untrusted_permission_grant" for i in issues)
        )

    def test_untrusted_permission_grant_flagged_by_low_trust(self) -> None:
        claim = _claim(
            "external_send",
            "permission_grant",
            "positive",
            trust=0.4,
            event_source="user",
        )
        issues = self.solver.check_claims([claim])
        self.assertTrue(
            any(i.issue_type == "untrusted_permission_grant" for i in issues)
        )

    def test_trusted_permission_grant_is_clean(self) -> None:
        claim = _claim(
            "external_send",
            "permission_grant",
            "positive",
            trust=0.95,
            event_source="user",
        )
        issues = self.solver.check_claims([claim])
        self.assertFalse(
            any(i.issue_type == "untrusted_permission_grant" for i in issues)
        )


if __name__ == "__main__":
    unittest.main()
