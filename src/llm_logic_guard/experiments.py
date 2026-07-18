from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from .attacks import ATTACK_SCENARIOS, SCENARIOS, AttackLab


def run_competition_experiment(
    *,
    attack_repetitions: int = 10,
    normal_repetitions: int = 5,
) -> dict[str, Any]:
    lab = AttackLab()
    results = []
    for scenario_id in SCENARIOS:
        repetitions = attack_repetitions if scenario_id in ATTACK_SCENARIOS else normal_repetitions
        for index in range(repetitions):
            for defended in (False, True):
                results.append(lab.run(
                    scenario_id,
                    defended=defended,
                    run_id=f"{index:03d}",
                ))

    attack_open = [r for r in results if r.scenario_id in ATTACK_SCENARIOS and not r.defended]
    attack_guard = [r for r in results if r.scenario_id in ATTACK_SCENARIOS and r.defended]
    normal_open = [r for r in results if r.scenario_id not in ATTACK_SCENARIOS and not r.defended]
    normal_guard = [r for r in results if r.scenario_id not in ATTACK_SCENARIOS and r.defended]
    open_asr = _rate(attack_open, "attack_succeeded")
    guarded_asr = _rate(attack_guard, "attack_succeeded")
    metrics = {
        "task_count": len(results),
        "attack_success_rate_without_guard": open_asr,
        "attack_success_rate_with_guard": guarded_asr,
        "attack_success_reduction": (open_asr - guarded_asr) / open_asr if open_asr else 0.0,
        "attack_block_rate": _rate(attack_guard, "task_completed"),
        "normal_completion_without_guard": _rate(normal_open, "task_completed"),
        "normal_completion_with_guard": _rate(normal_guard, "task_completed"),
        "normal_completion_drop": _rate(normal_open, "task_completed") - _rate(normal_guard, "task_completed"),
        "warning_rate": sum(r.warning_step is not None for r in attack_guard) / max(len(attack_guard), 1),
        "average_advance_steps": mean(
            (r.unsafe_step or 2) - r.warning_step
            for r in attack_guard
            if r.warning_step is not None
        ) if any(r.warning_step is not None for r in attack_guard) else 0.0,
        "average_latency_ms": mean(r.latency_ms for r in results),
    }
    by_scenario: dict[str, dict[str, Any]] = {}
    grouped = defaultdict(list)
    for result in results:
        grouped[result.scenario_id].append(result)
    for scenario_id, rows in grouped.items():
        guarded = [r for r in rows if r.defended]
        unguarded = [r for r in rows if not r.defended]
        by_scenario[scenario_id] = {
            "runs": len(rows),
            "attack_type": rows[0].attack_type,
            "unguarded_success_rate": _rate(unguarded, "attack_succeeded" if scenario_id in ATTACK_SCENARIOS else "task_completed"),
            "guarded_success_rate": _rate(guarded, "attack_succeeded" if scenario_id in ATTACK_SCENARIOS else "task_completed"),
            "decisions": _decision_counts(guarded),
        }
    return {
        "metrics": metrics,
        "by_scenario": by_scenario,
        "results": [result.to_dict() for result in results],
        "limitations": [
            "The benchmark uses deterministic sandbox ground truth, not human-labeled production traces.",
            "Repeated runs measure runtime stability and policy behavior, not independent linguistic diversity.",
        ],
    }


def _rate(rows: list, field: str) -> float:
    return sum(bool(getattr(row, field)) for row in rows) / max(len(rows), 1)


def _decision_counts(rows: list) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.decision] += 1
    return dict(counts)
