from .detector import LogicConsistencyDetector
from .corrector import LogicCorrector
from .converters import convert_security_events
from .hybrid import HybridConsistencyDetector
from .models import (
    AnalysisReport,
    AttackRunResult,
    Claim,
    ConsistencySpec,
    GuardDecision,
    LogicTrace,
    RepairResult,
    RiskAssessment,
    TraceEvent,
    TraceStep,
    Violation,
)
from .llm_provider import DeepSeekProvider, DeterministicOfficeProvider
from .claim_extraction import LLMClaimExtractor
from .probabilistic import ConsistencyRiskMonitor, DTMCModel
from .semantic import LexicalNLI, TransformersNLI
from .repair import LogicRepairer
from .storage import TraceStore


def __getattr__(name: str):
    if name == "OfficeAgentService":
        try:
            from .office_agent import OfficeAgentService
        except ImportError as exc:
            raise ImportError(
                "OfficeAgentService requires the optional LangGraph runtime. "
                "Install it with: python -m pip install -e \".[langgraph]\""
            ) from exc
        return OfficeAgentService
    raise AttributeError(name)

__all__ = [
    "AnalysisReport",
    "AttackRunResult",
    "Claim",
    "ConsistencyRiskMonitor",
    "ConsistencySpec",
    "DeepSeekProvider",
    "DeterministicOfficeProvider",
    "DTMCModel",
    "GuardDecision",
    "HybridConsistencyDetector",
    "LogicConsistencyDetector",
    "LogicCorrector",
    "LLMClaimExtractor",
    "LogicRepairer",
    "LogicTrace",
    "LexicalNLI",
    "OfficeAgentService",
    "RepairResult",
    "RiskAssessment",
    "TraceEvent",
    "TraceStep",
    "TraceStore",
    "TransformersNLI",
    "Violation",
    "convert_security_events",
]
