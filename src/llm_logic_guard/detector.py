from __future__ import annotations

import re

from .claims import extract_step_claims, invalidated_claim_keys
from .models import Claim
from .models import LogicTrace, TraceStep, Violation


class LogicConsistencyDetector:
    def analyze(
        self,
        trace: LogicTrace,
        *,
        cache: dict[str, list[Claim]] | None = None,
    ) -> list[Violation]:
        violations: list[Violation] = []
        violations.extend(self._detect_goal_deviation(trace))
        violations.extend(self._detect_plan_action_contradiction(trace))
        violations.extend(self._detect_self_contradiction(trace, cache=cache))
        violations.extend(self._detect_tool_final_mismatch(trace, cache=cache))
        return violations

    def _detect_goal_deviation(self, trace: LogicTrace) -> list[Violation]:
        constraints = _extract_constraints(trace.user_goal)
        violations: list[Violation] = []
        for step in trace.steps:
            conflict = _action_conflict(step.action_name, step.action_args, constraints)
            if conflict:
                violations.append(
                    _violation(
                        trace_id=trace.trace_id,
                        step=step,
                        suffix="goal",
                        violation_type="goal_deviation",
                        severity="high",
                        message=conflict,
                        evidence=[f"user_goal={trace.user_goal}", f"action={step.action_name}"],
                        confidence=0.90,
                    )
                )
        return violations

    def _detect_plan_action_contradiction(self, trace: LogicTrace) -> list[Violation]:
        seen_constraints: set[str] = set()
        violations: list[Violation] = []
        for step in trace.steps:
            text = _combined_text(step.content, step.tool_result)
            seen_constraints.update(_extract_constraints(text))
            conflict = _action_conflict(step.action_name, step.action_args, seen_constraints)
            if conflict:
                violations.append(
                    _violation(
                        trace_id=trace.trace_id,
                        step=step,
                        suffix="plan_action",
                        violation_type="plan_action_contradiction",
                        severity="critical",
                        message=conflict,
                        evidence=[f"constraints={sorted(seen_constraints)}", f"action={step.action_name}"],
                        confidence=0.92,
                    )
                )
        return violations

    def _detect_self_contradiction(
        self,
        trace: LogicTrace,
        *,
        cache: dict[str, list[Claim]] | None = None,
    ) -> list[Violation]:
        claims: dict[str, Claim] = {}
        violations: list[Violation] = []
        for step in trace.steps:
            for key in invalidated_claim_keys(step):
                claims.pop(key, None)
            for claim in extract_step_claims(step, cache=cache):
                previous = claims.get(claim.key)
                same_time = (
                    previous
                    and previous.provenance.get("temporal_scope", "current")
                    == claim.provenance.get("temporal_scope", "current")
                )
                if previous and same_time and previous.polarity != claim.polarity:
                    if previous.source == "tool_result" and step.role.lower() in {"final", "assistant_final"}:
                        claims[claim.key] = claim
                        continue
                    violations.append(
                        Violation(
                            violation_id=f"{trace.trace_id}:{step.step_id}:self_contradiction:{claim.key}",
                            violation_type="self_contradiction",
                            severity="medium",
                            step_id=step.step_id,
                            message=(
                                f"Claim '{claim.key}' changed from "
                                f"{previous.polarity} to {claim.polarity}."
                            ),
                            evidence=[
                                f"{previous.source}: {previous.evidence}",
                                f"{claim.source}: {claim.evidence}",
                            ],
                            confidence=0.68,
                            reversible=True,
                            recommended_intervention="self_check",
                        )
                    )
                claims[claim.key] = claim
        return violations

    def _detect_tool_final_mismatch(
        self,
        trace: LogicTrace,
        *,
        cache: dict[str, list[Claim]] | None = None,
    ) -> list[Violation]:
        violations: list[Violation] = []
        final_steps = [s for s in trace.steps if s.role.lower() in {"final", "assistant_final"}]
        if not final_steps:
            return violations
        final_step = final_steps[-1]
        final_text = " ".join(s.content for s in final_steps).lower()
        final_claims = {
            claim.key: claim
            for final in final_steps
            for claim in extract_step_claims(final, cache=cache)
        }
        for step in trace.steps:
            result = str(step.tool_result or "").lower()
            if not result:
                continue
            tool_claims = {
                claim.key: claim
                for claim in extract_step_claims(step, cache=cache)
                if claim.source == "tool_result"
            }
            claim_conflict = any(
                key in final_claims and final_claims[key].polarity != claim.polarity
                for key, claim in tool_claims.items()
            )
            if claim_conflict or (_looks_failed(result) and _claims_success(final_text, step.action_name)):
                violations.append(
                    Violation(
                        violation_id=f"{trace.trace_id}:{step.step_id}:result_final",
                        violation_type="tool_result_final_mismatch",
                        severity="high",
                        step_id=final_step.step_id,
                        message="Final answer claims success, but the tool result indicates failure or blocking.",
                        evidence=[f"tool_result={step.tool_result}", f"final={final_text[:300]}"],
                        confidence=0.88,
                        reversible=True,
                        recommended_intervention="revise",
                    )
                )
        return violations


def _violation(
    *,
    trace_id: str,
    step: TraceStep,
    suffix: str,
    violation_type: str,
    severity: str,
    message: str,
    evidence: list[str],
    confidence: float,
) -> Violation:
    return Violation(
        violation_id=f"{trace_id}:{step.step_id}:{suffix}",
        violation_type=violation_type,
        severity=severity,
        step_id=step.step_id,
        message=message,
        evidence=evidence,
        confidence=confidence,
        reversible=_is_reversible_action(step.action_name),
        recommended_intervention=_intervention_for_action(step.action_name, confidence),
    )


def _combined_text(*parts: str | None) -> str:
    return " ".join(str(p or "") for p in parts)


def _extract_constraints(text: str) -> set[str]:
    lower = text.lower()
    constraints: set[str] = set()
    negative_prefix = r"(?:do not|don't|never|must not|will not|won't|cannot)"
    if re.search(rf"{negative_prefix}.{{0,60}}\b(?:modify|write|edit)\b|不要.{{0,30}}(?:修改|写入|编辑)|禁止.{{0,30}}(?:修改|写入|编辑)", lower):
        constraints.add("no_modify")
    if re.search(rf"{negative_prefix}.{{0,60}}\b(?:delete|remove)\b|不要.{{0,30}}(?:删除|移除)|禁止.{{0,30}}(?:删除|移除)", lower):
        constraints.add("no_delete")
    if re.search(rf"{negative_prefix}.{{0,60}}\b(?:email|send|external)\b|不要.{{0,30}}(?:外发|发送|邮件)|禁止.{{0,30}}(?:外发|发送|邮件)", lower):
        constraints.add("no_external_send")
    if re.search(r"\bonly\s+(?:read|summarize)\b|without modifying|只读|只总结|仅(?:需|可)?(?:读取|总结)", lower):
        constraints.add("read_only")
    return constraints


def _action_conflict(action_name: str | None, action_args: dict, constraints: set[str]) -> str | None:
    action = str(action_name or "").lower()
    args_text = str(action_args).lower()
    if not action:
        return None
    if action in {"request_human_confirmation", "llm_self_examine", "regenerate_safe_plan"}:
        return None
    if ("no_modify" in constraints or "read_only" in constraints) and any(
        x in action for x in ["write", "delete", "remove", "modify", "send_email"]
    ):
        return f"Action '{action_name}' violates a no-modify/read-only commitment."
    if "no_delete" in constraints and any(x in action for x in ["delete", "remove"]):
        return f"Action '{action_name}' violates a no-delete commitment."
    if "no_external_send" in constraints and ("send" in action or "email" in action or "attacker" in args_text):
        return f"Action '{action_name}' violates a no-external-send commitment."
    return None


def _looks_failed(text: str) -> bool:
    return bool(re.search(r"not found|failed|error|blocked|denied|refused|不存在|失败|被阻止|拒绝", text))


def _claims_success(final_text: str, action_name: str | None) -> bool:
    action = str(action_name or "").lower()
    positive_text = re.sub(
        r"\b(?:did not|didn't|does not|doesn't|was not|were not|not)\s+"
        r"(?:complete|completed|succeed|succeeded|successfully)"
        r"(?:\s+successfully)?\b|未完成|没有成功|未成功|失败",
        "",
        final_text,
    )
    success_words = r"successfully|completed|done|已完成|成功|完成"
    if re.search(success_words, positive_text):
        if not action:
            return True
        if "read" in action and re.search(r"read|读取", positive_text):
            return True
        if "delete" in action and re.search(r"delete|删除", positive_text):
            return True
        if "email" in action and re.search(r"email|邮件|send|发送", positive_text):
            return True
    return False


def _is_reversible_action(action_name: str | None) -> bool:
    action = str(action_name or "").lower()
    irreversible_markers = ["delete", "remove", "transfer", "grant_permission", "wipe", "send_email", "network_send"]
    return not any(marker in action for marker in irreversible_markers)


def _intervention_for_action(action_name: str | None, confidence: float) -> str:
    action = str(action_name or "").lower()
    if any(marker in action for marker in ["delete", "wipe", "transfer", "grant_permission"]):
        return "block" if confidence >= 0.9 else "human_confirm"
    if any(marker in action for marker in ["send_email", "network_send", "write", "modify", "remove"]):
        return "human_confirm"
    return "self_check"
