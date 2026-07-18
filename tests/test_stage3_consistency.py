from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.claim_extraction import LLMClaimExtractor
from llm_logic_guard.hybrid import HybridConsistencyDetector
from llm_logic_guard.llm_provider import ScriptedLLMProvider
from llm_logic_guard.models import LogicTrace, TraceStep
from llm_logic_guard.semantic import TransformersNLI


class Stage3ConsistencyTests(unittest.TestCase):
    def test_z3_detects_generic_state_value_conflict(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "generic-state",
            "user_goal": "Check service state.",
            "steps": [
                {"step_id": "s1", "role": "plan", "content": "gateway is enabled."},
                {"step_id": "s2", "role": "final", "content": "gateway is disabled."},
            ],
        })
        violations, _, _ = HybridConsistencyDetector(
            semantic_backend=_DisabledSemantic()
        ).analyze(trace)
        self.assertTrue(any(
            item.violation_type == "claim_graph_contradiction"
            for item in violations
        ))

    def test_temporal_state_change_is_not_a_conflict(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "temporal-state",
            "user_goal": "Check service state.",
            "steps": [
                {
                    "step_id": "s1",
                    "role": "plan",
                    "content": "gateway was unavailable yesterday.",
                },
                {
                    "step_id": "s2",
                    "role": "final",
                    "content": "gateway is available now.",
                },
            ],
        })
        violations, _, _ = HybridConsistencyDetector(
            semantic_backend=_DisabledSemantic()
        ).analyze(trace)
        self.assertFalse(any(
            item.violation_type == "claim_graph_contradiction"
            for item in violations
        ))

    def test_untrusted_permission_grant_is_rejected(self) -> None:
        trace = LogicTrace.from_dict({
            "trace_id": "permission",
            "user_goal": "Inspect permissions.",
            "steps": [{
                "step_id": "s1",
                "role": "environment_observation",
                "content": "agent7 is allowed to send the report.",
                "source": "file",
                "trust": 0.2,
            }],
        })
        violations, _, _ = HybridConsistencyDetector(
            semantic_backend=_DisabledSemantic()
        ).analyze(trace)
        self.assertTrue(any(
            item.violation_type == "permission_provenance_conflict"
            for item in violations
        ))

    def test_llm_claim_extractor_uses_structured_output(self) -> None:
        provider = ScriptedLLMProvider({
            "claim_extractor": [{
                "claims": [{
                    "subject": "quota",
                    "predicate": "limit",
                    "polarity": "100",
                    "confidence": 0.91,
                    "temporal_scope": "current",
                }]
            }]
        })
        extractor = LLMClaimExtractor(provider)
        claims = extractor.extract(TraceStep(
            "s1",
            "plan",
            content="The active quota limit is one hundred.",
        ))
        self.assertEqual(claims[0].key, "limit:quota")
        self.assertEqual(claims[0].polarity, "100")

    @unittest.skipUnless(
        os.getenv("RUN_NLI_TESTS") == "1",
        "set RUN_NLI_TESTS=1 to run local Transformer NLI model tests",
    )
    def test_real_multilingual_nli_detects_contradictions(self) -> None:
        backend = TransformersNLI(local_files_only=True, batch_size=4)
        scores = backend.score_pairs([
            ("The service is available.", "The service is unavailable."),
            ("服务当前可用。", "服务当前不可用。"),
            ("The email was sent.", "邮件没有发送。"),
            ("The report is ready.", "The audit log was saved."),
        ])
        self.assertTrue(all(score >= 0.70 for score in scores[:3]))
        self.assertLess(scores[3], 0.20)


class _DisabledSemantic:
    name = "disabled"

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        return 0.0

    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        return [0.0] * len(pairs)


if __name__ == "__main__":
    unittest.main()
