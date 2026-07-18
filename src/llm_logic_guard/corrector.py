from __future__ import annotations

from .models import Correction, LogicTrace, Violation


class LogicCorrector:
    def propose(self, trace: LogicTrace, violations: list[Violation]) -> list[Correction]:
        corrections: list[Correction] = []
        seen: set[tuple[str, str]] = set()
        for violation in violations:
            correction = self._one(trace, violation)
            key = (correction.target_step_id, correction.intervention)
            if key not in seen:
                corrections.append(correction)
                seen.add(key)
        return corrections

    def _one(self, trace: LogicTrace, violation: Violation) -> Correction:
        intervention = violation.recommended_intervention
        if intervention in {"block", "deny"}:
            return Correction(
                correction_type="block_irreversible_violation",
                intervention="deny" if intervention == "deny" else "block",
                confidence=violation.confidence,
                target_step_id=violation.step_id,
                recommendation=(
                    "Block this high-confidence irreversible contradiction. Restate the user goal, "
                    "preserve earlier commitments, and regenerate a safe plan before any further action."
                ),
                replacement_action="regenerate_safe_plan",
                replacement_args={"user_goal": trace.user_goal, "blocked_step": violation.step_id},
            )
        if intervention in {"human_confirm", "confirm"}:
            return Correction(
                correction_type="pause_for_human_confirmation",
                intervention="confirm" if intervention == "confirm" else "human_confirm",
                confidence=violation.confidence,
                target_step_id=violation.step_id,
                recommendation=(
                    "Pause before executing this state-changing action. Show the detected contradiction, "
                    "the proposed action, and ask for explicit human confirmation or a safer alternative."
                ),
                replacement_action="request_human_confirmation",
                replacement_args={"user_goal": trace.user_goal, "step": violation.step_id},
            )
        if intervention == "revise":
            return Correction(
                correction_type="revise_final_answer",
                intervention="revise",
                confidence=violation.confidence,
                target_step_id=violation.step_id,
                recommendation=(
                    "Revise the final answer so it matches the actual tool result. Do not claim success "
                    "unless the supporting observation exists."
                ),
            )
        if intervention == "replan":
            return Correction(
                correction_type="regenerate_consistent_plan",
                intervention="replan",
                confidence=violation.confidence,
                target_step_id=violation.step_id,
                recommendation=(
                    "Discard the unsafe candidate action, preserve trusted goals and evidence, "
                    "then generate a new plan that passes the full consistency guard."
                ),
                replacement_action="regenerate_safe_plan",
                replacement_args={
                    "user_goal": trace.user_goal,
                    "blocked_step": violation.step_id,
                    "evidence": violation.evidence,
                },
            )
        if intervention == "audit":
            return Correction(
                correction_type="audit_only",
                intervention="audit",
                confidence=violation.confidence,
                target_step_id=violation.step_id,
                recommendation="Record this low-confidence inconsistency for review without changing execution.",
            )
        return Correction(
            correction_type="request_self_check",
            intervention="self_check",
            confidence=violation.confidence,
            target_step_id=violation.step_id,
            recommendation=(
                "Ask the agent to compare the conflicting claims, cite the supporting evidence for each, "
                "and continue only after choosing the evidence-backed version."
            ),
            replacement_action="llm_self_examine",
            replacement_args={"violation_type": violation.violation_type, "evidence": violation.evidence},
        )
