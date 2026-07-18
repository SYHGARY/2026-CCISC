from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventPhase(str, Enum):
    USER_INPUT = "user_input"
    PLAN = "plan"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    ENVIRONMENT_OBSERVATION = "environment_observation"
    BEFORE_ACTION = "before_action"
    AFTER_ACTION = "after_action"
    FINAL_ANSWER = "final_answer"


class GuardAction(str, Enum):
    ALLOW = "allow"
    AUDIT = "audit"
    CONFIRM = "confirm"
    DENY = "deny"
    REPLAN = "replan"


@dataclass
class TraceStep:
    step_id: str
    role: str
    content: str = ""
    action_name: str | None = None
    action_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_id: str = "agent"
    source: str = "agent"
    trust: float = 0.7
    parent_event_id: str | None = None
    security_labels: list[str] = field(default_factory=list)

    @property
    def phase(self) -> str:
        aliases = {
            "assistant": EventPhase.PLAN.value,
            "tool_call": EventPhase.BEFORE_ACTION.value,
            "tool": EventPhase.AFTER_ACTION.value,
            "final": EventPhase.FINAL_ANSWER.value,
            "assistant_final": EventPhase.FINAL_ANSWER.value,
        }
        return aliases.get(self.role.lower(), self.role.lower())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceStep":
        metadata = dict(data.get("metadata") or {})
        return cls(
            step_id=str(data.get("step_id") or data.get("event_id") or "unknown"),
            role=str(data.get("role") or data.get("phase") or "assistant"),
            content=str(data.get("content") or data.get("message") or data.get("thought") or ""),
            action_name=data.get("action_name"),
            action_args=dict(data.get("action_args") or data.get("args") or {}),
            tool_result=data.get("tool_result") or data.get("tool_result_summary") or data.get("result"),
            metadata=metadata,
            timestamp=str(data.get("timestamp") or metadata.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            agent_id=str(data.get("agent_id") or metadata.get("agent_id") or "agent"),
            source=str(data.get("source") or metadata.get("source") or "agent"),
            trust=float(data.get("trust", metadata.get("trust", 0.7))),
            parent_event_id=data.get("parent_event_id") or metadata.get("parent_event_id"),
            security_labels=list(data.get("security_labels") or metadata.get("security_labels") or []),
        )


TraceEvent = TraceStep


@dataclass
class LogicTrace:
    trace_id: str
    user_goal: str
    steps: list[TraceStep]
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    status: str = "running"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogicTrace":
        trace_id = str(data.get("trace_id") or data.get("case_id") or "trace_unknown")
        return cls(
            trace_id=trace_id,
            user_goal=str(data.get("user_goal") or data.get("task") or ""),
            steps=[TraceStep.from_dict(step) for step in data.get("steps", data.get("events", []))],
            metadata=dict(data.get("metadata") or {}),
            session_id=str(data.get("session_id") or trace_id),
            status=str(data.get("status") or "running"),
        )


@dataclass
class Claim:
    subject: str
    predicate: str
    polarity: str
    evidence: str
    step_id: str
    source: str
    confidence: float = 0.75
    trust: float = 0.7
    timestamp: str = ""
    valid_from_step: str = ""
    invalidated_by: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.predicate}:{self.subject}"


@dataclass
class ClaimEdge:
    source_claim: str
    target_claim: str
    relation: str
    confidence: float = 1.0


@dataclass
class ConsistencySpec:
    spec_id: str
    description: str
    trigger_phases: list[str] = field(default_factory=list)
    kind: str = "invariant"
    condition: dict[str, Any] = field(default_factory=dict)
    enforcement: str = GuardAction.AUDIT.value
    severity: str = "medium"
    enabled: bool = True


@dataclass
class Violation:
    violation_id: str
    violation_type: str
    severity: str
    step_id: str
    message: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.75
    reversible: bool = True
    recommended_intervention: str = "self_check"
    detector: str = "legacy"
    spec_id: str | None = None
    attack_stage: str | None = None


@dataclass
class Correction:
    correction_type: str
    target_step_id: str
    recommendation: str
    intervention: str = "self_check"
    confidence: float = 0.75
    replacement_action: str | None = None
    replacement_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    state: str
    probability: float
    pac_upper: float
    threshold: float
    horizon: int
    warning: bool
    state_count: int = 0
    features: dict[str, bool | str | float] = field(default_factory=dict)
    reason: str = ""
    pac_epsilon: float = 0.0
    model_version: str = ""
    calibrated: bool = False
    fallback: bool = False


@dataclass
class GuardDecision:
    trace_id: str
    event_id: str
    action: str
    reasons: list[str]
    violations: list[Violation] = field(default_factory=list)
    risk: RiskAssessment | None = None
    alternative_action: str | None = None
    alternative_args: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action in {
            GuardAction.CONFIRM.value,
            GuardAction.DENY.value,
            GuardAction.REPLAN.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    trace_id: str
    is_consistent: bool
    violations: list[Violation]
    corrections: list[Correction]
    claims: list[Claim] = field(default_factory=list)
    claim_edges: list[ClaimEdge] = field(default_factory=list)
    risk: RiskAssessment | None = None
    detector_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairResult:
    trace_id: str
    resolved: bool
    applied_interventions: list[str]
    repaired_trace: LogicTrace
    remaining_report: AnalysisReport

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttackRunResult:
    experiment_id: str
    scenario_id: str
    attack_type: str
    defended: bool
    attack_succeeded: bool
    task_completed: bool
    trace_id: str
    decision: str
    warning_step: int | None
    unsafe_step: int | None
    latency_ms: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
