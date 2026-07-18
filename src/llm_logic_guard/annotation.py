from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .redteam import attack_catalog
from .stage3_evaluation import generate_stage3_benchmark


LABEL_SCHEMA = {
    "logic_consistency": ["consistent", "inconsistent", "uncertain"],
    "attack_family": [
        "none",
        "prompt_injection",
        "memory_poisoning",
        "environment_pollution",
    ],
    "violation_types": [
        "goal_deviation",
        "plan_action_contradiction",
        "claim_graph_contradiction",
        "semantic_contradiction",
        "permission_provenance_conflict",
        "tool_result_final_mismatch",
        "spec_violation",
    ],
    "recommended_action": ["allow", "audit", "confirm", "deny", "replan"],
}


def build_annotation_seed(limit: int = 120) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in attack_catalog():
        rows.append({
            "case_id": f"attack-{item['id']}",
            "source": "redteam_catalog",
            "task": item["goal"],
            "observable_events": [
                {
                    "phase": "environment_observation",
                    "content": item["normal_or_observed_input"],
                    "source": item["source"],
                },
                {
                    "phase": "before_action",
                    "action_name": item["candidate_action"],
                },
            ],
            "gold": {
                "logic_consistency": (
                    "inconsistent" if item["is_attack"] else "consistent"
                ),
                "attack_family": (
                    item["family"] if item["is_attack"] else "none"
                ),
                "recommended_action": item["expected_defense"],
            },
        })
    for item in generate_stage3_benchmark():
        rows.append({
            "case_id": f"logic-{item['id']}",
            "source": "stage3_consistency_benchmark",
            "task": "Check observable statements for consistency.",
            "observable_events": [
                {
                    "phase": "plan",
                    "content": item["premise"],
                    "source": "agent",
                },
                {
                    "phase": "final_answer",
                    "content": item["hypothesis"],
                    "source": "agent",
                },
            ],
            "gold": {
                "logic_consistency": (
                    "inconsistent" if item["inconsistent"] else "consistent"
                ),
                "attack_family": "none",
                "recommended_action": (
                    "replan" if item["inconsistent"] else "allow"
                ),
            },
        })
        if len(rows) >= limit:
            break
    return rows[:limit]


def write_annotation_packet(
    *,
    output_path: Path,
    manifest_path: Path,
    limit: int = 120,
) -> dict[str, Any]:
    rows = build_annotation_seed(limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "case_count": len(rows),
        "label_schema": LABEL_SCHEMA,
        "protocol": {
            "annotators": 2,
            "adjudication": (
                "Disagreements are resolved by a third pass using observable "
                "events only; hidden chain-of-thought is not labeled."
            ),
            "agreement_metrics": ["Cohen kappa", "exact-match rate"],
            "allowed_evidence": [
                "user task",
                "observable event content",
                "tool name and arguments",
                "tool result",
                "source and trust metadata",
            ],
        },
        "sources": {
            "redteam_catalog": sum(
                row["source"] == "redteam_catalog" for row in rows
            ),
            "stage3_consistency_benchmark": sum(
                row["source"] == "stage3_consistency_benchmark"
                for row in rows
            ),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
