from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

from .models import LogicTrace, TraceStep


def convert_security_events(
    events: list[dict[str, Any]],
    user_goal: str = "",
) -> list[LogicTrace]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        trace_id = str(event.get("trace_id") or "trace_unknown")
        grouped[trace_id].append(event)

    traces: list[LogicTrace] = []
    for trace_id, trace_events in grouped.items():
        ordered = sorted(
            enumerate(trace_events),
            key=lambda item: (str(item[1].get("timestamp") or ""), item[0]),
        )
        steps: list[TraceStep] = []
        pending: list[TraceStep] = []

        for _, event in ordered:
            phase = str(event.get("phase") or "")
            if phase == "after_action":
                match = _find_pending_step(pending, event)
                if match is not None:
                    match.tool_result = _event_result(event)
                    match.metadata["after_event_id"] = event.get("event_id")
                    match.metadata["completed_at"] = event.get("timestamp")
                    match.metadata["source_is_safe_case"] = event.get(
                        "is_safe_case",
                        match.metadata.get("source_is_safe_case"),
                    )
                    pending.remove(match)
                    continue

            step = _event_to_step(event)
            steps.append(step)
            if phase == "before_action":
                pending.append(step)

        traces.append(
            LogicTrace(
                trace_id=trace_id,
                user_goal=user_goal,
                steps=steps,
                metadata={
                    "source_format": "security_event_jsonl",
                    "source_event_count": len(trace_events),
                    "source_safety_labels_are_logic_labels": False,
                },
            )
        )
    return traces


def trace_to_dict(trace: LogicTrace) -> dict[str, Any]:
    return asdict(trace)


def _event_to_step(event: dict[str, Any]) -> TraceStep:
    return TraceStep(
        step_id=str(event.get("event_id") or "unknown"),
        role=str(event.get("phase") or "tool_call"),
        content=str(event.get("content") or event.get("message") or ""),
        action_name=event.get("action_name"),
        action_args=dict(event.get("action_args") or {}),
        tool_result=_event_result(event),
        metadata={
            "session_id": event.get("session_id"),
            "agent_id": event.get("agent_id"),
            "started_at": event.get("timestamp"),
            "source_is_safe_case": event.get("is_safe_case"),
        },
    )


def _event_result(event: dict[str, Any]) -> str | None:
    result = event.get("result")
    return str(result) if result is not None else None


def _find_pending_step(
    pending: list[TraceStep],
    event: dict[str, Any],
) -> TraceStep | None:
    action_name = event.get("action_name")
    action_args = dict(event.get("action_args") or {})
    for step in reversed(pending):
        if step.action_name == action_name and step.action_args == action_args:
            return step
    return None
