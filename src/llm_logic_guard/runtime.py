from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .models import GuardAction, GuardDecision, LogicTrace, TraceStep
from .pipeline import LogicGuardPipeline
from .probabilistic import ConsistencyRiskMonitor
from .storage import TraceStore


class GuardRuntime:
    def __init__(
        self,
        *,
        store: TraceStore | None = None,
        risk_monitor: ConsistencyRiskMonitor | None = None,
        pipeline: LogicGuardPipeline | None = None,
    ) -> None:
        self.store = store or TraceStore()
        if pipeline is not None:
            # Shared, read-only pipeline injected by a batch caller. Reuse its
            # risk monitor so runtime and pipeline agree on the same model.
            self.pipeline = pipeline
            self.risk_monitor = risk_monitor or pipeline.risk_monitor
        else:
            self.risk_monitor = risk_monitor or ConsistencyRiskMonitor(
                Path("outputs/logicguard_dtmc.json")
            )
            self.pipeline = LogicGuardPipeline(risk_monitor=self.risk_monitor)

    def create_trace(
        self,
        user_goal: str,
        *,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LogicTrace:
        trace_id = trace_id or f"trace-{uuid.uuid4().hex[:12]}"
        trace = LogicTrace(
            trace_id=trace_id,
            session_id=trace_id,
            user_goal=user_goal,
            steps=[],
            metadata=dict(metadata or {}),
        )
        self.store.create_trace(trace)
        return trace

    def evaluate_event(self, trace_id: str, event: TraceStep) -> GuardDecision:
        trace = self.store.get_trace(trace_id)
        if trace is None:
            raise KeyError(f"unknown trace: {trace_id}")
        # Build the candidate trace by sharing the existing step objects and
        # appending the new event. The detector, corrector, and risk monitor
        # treat steps as read-only, so this avoids a full asdict/from_dict deep
        # copy on every event (the per-event hot path).
        candidate_trace = LogicTrace(
            trace_id=trace.trace_id,
            user_goal=trace.user_goal,
            steps=[*trace.steps, event],
            metadata=trace.metadata,
            session_id=trace.session_id,
            status=trace.status,
        )
        # The runtime computes its own risk below via assess(trace, event); skip
        # the pipeline's internal risk assessment to avoid a duplicate call.
        # focus_step_id restricts semantic pairing to the new event, whose
        # violations are the only ones the runtime keeps.
        report = self.pipeline.run(
            candidate_trace,
            assess_risk=False,
            focus_step_id=event.step_id,
        )
        event_violations = [v for v in report.violations if v.step_id == event.step_id]
        risk = self.risk_monitor.assess(trace, event)
        action = _select_action(
            event_violations,
            risk.warning,
            risk.features,
            # Only honor a human_confirmed flag from a trusted source; a forged
            # flag on an untrusted (file/web/tool/memory) event is ignored.
            human_confirmed=(
                bool(event.metadata.get("human_confirmed"))
                and event.source in {"user", "system", "trusted_planner"}
            ),
        )
        reasons = [v.message for v in event_violations]
        if risk.warning:
            reasons.append(risk.reason)
        if not reasons:
            reasons.append("未发现违反一致性规范或高风险轨迹。")
        alternative_action, alternative_args = safer_alternative(event, action)
        decision = GuardDecision(
            trace_id=trace_id,
            event_id=event.step_id,
            action=action,
            reasons=reasons,
            violations=event_violations,
            risk=risk,
            alternative_action=alternative_action,
            alternative_args=alternative_args,
        )
        self.store.append_event(trace_id, event)
        self.store.save_decision(decision)
        return decision

    def post_event(self, trace_id: str, event: TraceStep) -> GuardDecision:
        return self.evaluate_event(trace_id, event)

    def repair(self, trace_id: str, event_id: str | None = None) -> dict[str, Any]:
        trace = self.store.get_trace(trace_id)
        if trace is None:
            raise KeyError(f"unknown trace: {trace_id}")
        decisions = self.store.get_decisions(trace_id)
        target = next(
            (
                item for item in reversed(decisions)
                if event_id is None or item["event_id"] == event_id
            ),
            None,
        )
        if target is None:
            raise KeyError("no decision available for repair")
        event = next(step for step in trace.steps if step.step_id == target["event_id"])
        alternative_action, alternative_args = safer_alternative(event, GuardAction.REPLAN.value)
        repaired = TraceStep(
            step_id=f"{event.step_id}:replan",
            role="plan",
            content=_repair_message(event, alternative_action),
            action_name=alternative_action,
            action_args=alternative_args,
            source="trusted_planner",
            trust=0.95,
            parent_event_id=event.step_id,
            security_labels=["replanned"],
        )
        decision = self.evaluate_event(trace_id, repaired)
        return {
            "original_event_id": event.step_id,
            "repaired_event": asdict(repaired),
            "decision": decision.to_dict(),
            "verified": decision.action in {GuardAction.ALLOW.value, GuardAction.AUDIT.value},
        }

    def guarded_call(
        self,
        trace_id: str,
        event: TraceStep,
        function: Callable[..., Any],
    ) -> dict[str, Any]:
        decision = self.evaluate_event(trace_id, event)
        if decision.blocked:
            return {"executed": False, "decision": decision.to_dict(), "result": None}
        result = function(**event.action_args)
        after = TraceStep(
            step_id=f"{event.step_id}:result",
            role="after_action",
            content="",
            action_name=event.action_name,
            action_args=event.action_args,
            tool_result=str(result),
            source="tool",
            trust=0.95,
            parent_event_id=event.step_id,
        )
        post_decision = self.evaluate_event(trace_id, after)
        return {
            "executed": True,
            "decision": decision.to_dict(),
            "post_decision": post_decision.to_dict(),
            "result": result,
        }


def safer_alternative(event: TraceStep, decision: str) -> tuple[str | None, dict[str, Any]]:
    if decision not in {GuardAction.DENY.value, GuardAction.REPLAN.value, "block"}:
        return None, {}
    action = str(event.action_name or "").lower()
    args = dict(event.action_args)
    if action in {"send_email", "network_send", "http_post", "upload_file"}:
        return "save_local_draft", {
            "content": args.get("body") or args.get("data") or "External transmission removed.",
        }
    if action in {"delete_file", "remove_file"}:
        return "read_file", {"path": args.get("path") or args.get("filename") or ""}
    if action in {"execute_code", "run_code", "shell_exec"}:
        return "inspect_code", {"code": args.get("code") or args.get("command") or ""}
    if action == "memory_write":
        return "quarantine_memory", {"key": args.get("key"), "value": args.get("value")}
    return "request_safe_plan", {"blocked_action": event.action_name}


def _select_action(
    violations: list,
    warning: bool,
    features: dict[str, Any],
    *,
    human_confirmed: bool = False,
) -> str:
    interventions = {v.recommended_intervention for v in violations}
    if "deny" in interventions or "block" in interventions:
        return GuardAction.DENY.value
    if "confirm" in interventions or "human_confirm" in interventions:
        return GuardAction.CONFIRM.value
    if "replan" in interventions:
        return GuardAction.REPLAN.value
    if warning:
        if human_confirmed:
            return GuardAction.AUDIT.value
        if features.get("authorized_public_sandbox_external"):
            return GuardAction.AUDIT.value
        if (
            features.get("benign_preference_memory_write")
            and not features.get("unsafe_memory_write")
        ):
            return GuardAction.AUDIT.value
        risky_side_effect = any(
            bool(features.get(key))
            for key in (
                "external",
                "destructive",
                "memory_write",
                "irreversible",
                "sensitive_flow",
            )
        )
        inconsistent_action = features.get("action_consistent") is False
        if risky_side_effect or inconsistent_action:
            return GuardAction.REPLAN.value
        return GuardAction.AUDIT.value
    if violations:
        return GuardAction.AUDIT.value
    return GuardAction.ALLOW.value


def _repair_message(event: TraceStep, alternative: str | None) -> str:
    return (
        f"原候选动作 {event.action_name or event.phase} 未通过一致性验证。"
        f"保留用户目标与可信证据，改用 {alternative or '安全规划'}。"
    )
