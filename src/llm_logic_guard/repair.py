from __future__ import annotations

from dataclasses import asdict

from .models import AnalysisReport, LogicTrace, RepairResult, TraceStep
from .pipeline import LogicGuardPipeline


class LogicRepairer:
    def apply(self, trace: LogicTrace, report: AnalysisReport | None = None) -> RepairResult:
        pipeline = LogicGuardPipeline()
        source_report = report or pipeline.run(trace)
        repaired = LogicTrace.from_dict(asdict(trace))
        applied: list[str] = []
        remaining_report = source_report

        for _ in range(3):
            applied_this_round = 0
            for correction in remaining_report.corrections:
                step = _find_step(repaired, correction.target_step_id)
                if step is None:
                    continue
                if correction.intervention == "replan" and not step.action_name:
                    continue
                if correction.intervention in {"block", "deny", "replan"}:
                    _block_action(step)
                elif correction.intervention in {"human_confirm", "confirm"}:
                    _request_confirmation(step)
                elif correction.intervention == "revise":
                    _revise_final_answer(repaired, step)
                else:
                    continue
                applied.append(correction.intervention)
                applied_this_round += 1

            if applied_this_round == 0:
                break
            remaining_report = pipeline.run(repaired)
            if remaining_report.is_consistent:
                break

        return RepairResult(
            trace_id=trace.trace_id,
            resolved=remaining_report.is_consistent,
            applied_interventions=applied,
            repaired_trace=repaired,
            remaining_report=remaining_report,
        )


def _find_step(trace: LogicTrace, step_id: str) -> TraceStep | None:
    return next((step for step in trace.steps if step.step_id == step_id), None)


def _block_action(step: TraceStep) -> None:
    step.metadata["blocked_action"] = step.action_name
    step.metadata["blocked_action_args"] = step.action_args
    step.action_name = None
    step.action_args = {}
    step.tool_result = "Action blocked before execution."
    step.content = "The contradictory action was blocked before execution."


def _request_confirmation(step: TraceStep) -> None:
    proposed_action = step.action_name
    proposed_args = step.action_args
    step.metadata["proposed_action"] = proposed_action
    step.metadata["proposed_action_args"] = proposed_args
    step.action_name = "request_human_confirmation"
    step.action_args = {
        "proposed_action": proposed_action,
        "proposed_args": proposed_args,
    }
    step.tool_result = None
    step.content = "Execution paused pending explicit human confirmation."


def _revise_final_answer(trace: LogicTrace, final_step: TraceStep) -> None:
    failed_result = next(
        (
            step.tool_result
            for step in reversed(trace.steps)
            if step.tool_result and _looks_failed(step.tool_result)
        ),
        "The tool reported a failure.",
    )
    final_step.metadata["original_content"] = final_step.content
    final_step.content = f"The requested action did not complete successfully. Tool result: {failed_result}"


def _looks_failed(text: str) -> bool:
    lower = text.lower()
    return any(
        marker in lower
        for marker in [
            "not found",
            "failed",
            "error",
            "blocked",
            "denied",
            "refused",
            "不存在",
            "失败",
            "被阻止",
            "拒绝",
        ]
    )
