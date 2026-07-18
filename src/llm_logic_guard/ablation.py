from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .attacks import ATTACK_SCENARIOS, SCENARIOS
from .hybrid import HybridConsistencyDetector
from .models import LogicTrace, TraceStep
from .probabilistic import ConsistencyRiskMonitor
from .semantic import LexicalNLI
from .specs import SpecEngine


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


def run_ablation(repetitions: int = 10) -> dict:
    rows = []
    for scenario_id, scenario in SCENARIOS.items():
        for index in range(repetitions):
            trace = _scenario_trace(scenario_id, scenario, index)
            rows.append((trace, scenario_id in ATTACK_SCENARIOS))

    spec_engine = SpecEngine()
    semantic = LexicalNLI()
    hybrid = HybridConsistencyDetector()
    risk = ConsistencyRiskMonitor()

    detectors: dict[str, Callable[[LogicTrace], bool]] = {
        "dsl_only": lambda trace: bool(spec_engine.evaluate(trace)),
        "nli_only": lambda trace: _nli_flags(trace, semantic),
        "claim_graph_and_dsl": lambda trace: bool(HybridConsistencyDetector(
            include_legacy=False,
            semantic_backend=_DisabledSemantic(),
        ).analyze(trace)[0]),
        "hybrid_without_prediction": lambda trace: bool(hybrid.analyze(trace)[0]),
        "full_system": lambda trace: bool(hybrid.analyze(trace)[0]) or risk.assess(trace).warning,
    }
    reports = {}
    for name, detector in detectors.items():
        predicted = [detector(trace) for trace, _ in rows]
        expected = [label for _, label in rows]
        reports[name] = _metrics(predicted, expected).__dict__
    return {
        "sample_count": len(rows),
        "configurations": reports,
        "note": "Programmatic sandbox scenarios; NLI-only is expected to be weak on action-policy attacks.",
    }


def _scenario_trace(scenario_id: str, scenario: dict, index: int) -> LogicTrace:
    observation = TraceStep(
        f"{scenario_id}-{index}:obs",
        "environment_observation",
        content=scenario["observation"],
        source=scenario["source"],
        trust=0.95 if scenario["source"] == "user" else 0.25,
    )
    action = TraceStep(
        f"{scenario_id}-{index}:action",
        "before_action",
        content=f"Candidate action from {scenario['source']}.",
        action_name=scenario["candidate_action"],
        action_args=scenario["candidate_args"],
        source=scenario["source"] if scenario_id in ATTACK_SCENARIOS else "trusted_planner",
        trust=0.3 if scenario_id in ATTACK_SCENARIOS else 0.95,
    )
    return LogicTrace(f"{scenario_id}-{index}", scenario["goal"], [observation, action])


def _nli_flags(trace: LogicTrace, semantic: LexicalNLI) -> bool:
    texts = [trace.user_goal] + [step.content for step in trace.steps if step.content]
    return any(
        semantic.contradiction_score(left, right) >= 0.72
        for index, right in enumerate(texts)
        for left in texts[:index]
    )


def _metrics(predicted: list[bool], expected: list[bool]) -> BinaryMetrics:
    tp = sum(p and e for p, e in zip(predicted, expected))
    fp = sum(p and not e for p, e in zip(predicted, expected))
    fn = sum(not p and e for p, e in zip(predicted, expected))
    tn = sum(not p and not e for p, e in zip(predicted, expected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return BinaryMetrics(precision, recall, f1, fpr)


class _DisabledSemantic:
    name = "disabled"

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        return 0.0
