from __future__ import annotations

from .corrector import LogicCorrector
from .hybrid import HybridConsistencyDetector
from .models import AnalysisReport, LogicTrace
from .probabilistic import ConsistencyRiskMonitor


class LogicGuardPipeline:
    def __init__(
        self,
        *,
        detector: HybridConsistencyDetector | None = None,
        risk_monitor: ConsistencyRiskMonitor | None = None,
    ) -> None:
        self.detector = detector or HybridConsistencyDetector()
        self.corrector = LogicCorrector()
        self.risk_monitor = risk_monitor or ConsistencyRiskMonitor()

    def run(
        self,
        trace: LogicTrace,
        *,
        assess_risk: bool = True,
        focus_step_id: str | None = None,
    ) -> AnalysisReport:
        violations, claims, edges = self.detector.analyze(trace, focus_step_id=focus_step_id)
        corrections = self.corrector.propose(trace, violations)
        risk = self.risk_monitor.assess(trace) if (assess_risk and trace.steps) else None
        return AnalysisReport(
            trace_id=trace.trace_id,
            is_consistent=not violations,
            violations=violations,
            corrections=corrections,
            claims=claims,
            claim_edges=edges,
            risk=risk,
            detector_versions={
                "hybrid": "logicguard-hybrid.v1",
                "semantic": self.detector.semantic.name,
                "specs": "logicguard-specs.v1",
                "probabilistic": "logicguard-dtmc.v2",
            },
        )
