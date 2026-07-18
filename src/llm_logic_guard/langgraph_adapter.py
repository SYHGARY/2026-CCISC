from __future__ import annotations

from typing import Any, Callable

from .models import TraceStep
from .runtime import GuardRuntime


class LangGraphLogicGuard:
    """Framework-compatible state-node wrapper; LangGraph is an optional dependency."""

    def __init__(self, runtime: GuardRuntime) -> None:
        self.runtime = runtime

    def wrap_tool_node(
        self,
        node: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        action_name: str,
        args_keys: tuple[str, ...] = (),
        source_key: str = "source",
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def guarded(state: dict[str, Any]) -> dict[str, Any]:
            trace_id = str(state["trace_id"])
            args = {key: state.get(key) for key in args_keys}
            event = TraceStep(
                step_id=str(state.get("event_id") or f"{trace_id}:{action_name}"),
                role="before_action",
                action_name=action_name,
                action_args=args,
                content=str(state.get("candidate_plan") or ""),
                source=str(state.get(source_key) or "trusted_planner"),
                trust=float(state.get("trust", 0.9)),
            )
            decision = self.runtime.evaluate_event(trace_id, event)
            if decision.blocked:
                return {
                    **state,
                    "guard_action": decision.action,
                    "guard_reasons": decision.reasons,
                    "replan_required": decision.action == "replan",
                    "proposed_alternative": decision.alternative_action,
                }
            output = node(state)
            return {**state, **output, "guard_action": decision.action}

        return guarded
