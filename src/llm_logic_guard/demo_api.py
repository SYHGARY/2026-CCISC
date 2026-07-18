from __future__ import annotations

from typing import Any

from .models import LogicTrace
from .pipeline import LogicGuardPipeline
from .repair import LogicRepairer


def analyze_payload(payload: dict[str, Any]) -> dict[str, Any]:
    trace = LogicTrace.from_dict(payload)
    return LogicGuardPipeline().run(trace).to_dict()


def repair_payload(payload: dict[str, Any]) -> dict[str, Any]:
    trace = LogicTrace.from_dict(payload)
    pipeline = LogicGuardPipeline()
    original_report = pipeline.run(trace)
    repair = LogicRepairer().apply(trace, original_report)
    return {
        "original_report": original_report.to_dict(),
        "repair": repair.to_dict(),
    }
