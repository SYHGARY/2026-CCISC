from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = PROJECT_ROOT / "data" / "deepseek_candidate_pool_1000.json"
DEFAULT_SCHEMA = PROJECT_ROOT / "data" / "deepseek_extended_case_schema.json"
DEFAULT_CORE = PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
DEFAULT_AUDIT_JSON = PROJECT_ROOT / "outputs" / "stage19_candidate_pool_consistency_audit.json"
DEFAULT_AUDIT_MD = PROJECT_ROOT / "outputs" / "stage19_candidate_pool_consistency_audit.md"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_deepseek_candidate_pool import load_cases, load_core_ids, load_json, write_json


ATTACK_CATEGORIES = {
    "prompt_injection",
    "jailbreak",
    "tool_hijacking",
    "dangerous_code_execution",
    "file_access_overwrite",
    "sensitive_information_leakage",
    "memory_poisoning",
    "environment_pollution",
    "multi_agent_error_propagation",
}


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def case_surfaces(case: dict[str, Any]) -> list[str]:
    surfaces = case.get("attack_surface", [])
    if not isinstance(surfaces, list):
        return []
    return [str(surface) for surface in surfaces]


def build_consistency_audit(
    pool_path: Path = DEFAULT_POOL,
    schema_path: Path = DEFAULT_SCHEMA,
    core_path: Path = DEFAULT_CORE,
) -> dict[str, Any]:
    cases = load_cases(pool_path)
    schema = load_json(schema_path)
    allowed_categories = set(schema["properties"]["category"]["enum"])
    allowed_surfaces = set(schema["properties"]["attack_surface"]["items"]["enum"])
    core_ids = load_core_ids(core_path)

    sample_type_counts = Counter(str(case.get("sample_type")) for case in cases)
    category_counts = Counter(str(case.get("category")) for case in cases)
    primary_surface_counts: Counter[str] = Counter()
    attack_surface_assignment_counts: Counter[str] = Counter()
    attack_only_assignment_counts: Counter[str] = Counter()

    inconsistencies: list[dict[str, Any]] = []
    multi_surface_cases: list[dict[str, Any]] = []

    for case in cases:
        case_id = str(case.get("case_id", ""))
        sample_type = str(case.get("sample_type", ""))
        category = str(case.get("category", ""))
        surfaces = case_surfaces(case)

        if sample_type == "attack":
            primary_surface = category
        elif sample_type in {"normal", "hard_normal"}:
            primary_surface = sample_type
        else:
            primary_surface = "invalid"
        primary_surface_counts[primary_surface] += 1

        if len(surfaces) > 1:
            multi_surface_cases.append(
                {
                    "case_id": case_id,
                    "sample_type": sample_type,
                    "category": category,
                    "attack_surface": surfaces,
                    "extra_surface_count": len(surfaces) - 1,
                }
            )
        for surface in surfaces:
            attack_surface_assignment_counts[surface] += 1
            if sample_type == "attack":
                attack_only_assignment_counts[surface] += 1

        if category not in allowed_categories:
            inconsistencies.append({"case_id": case_id, "issue": "invalid_category", "category": category})
        invalid_surfaces = sorted(surface for surface in surfaces if surface not in allowed_surfaces)
        if invalid_surfaces:
            inconsistencies.append({"case_id": case_id, "issue": "invalid_attack_surface", "surfaces": invalid_surfaces})

        if sample_type == "attack":
            if category not in ATTACK_CATEGORIES:
                inconsistencies.append({"case_id": case_id, "issue": "attack_case_has_non_attack_category"})
            if not surfaces or any(surface in {"normal", "hard_normal"} for surface in surfaces):
                inconsistencies.append({"case_id": case_id, "issue": "attack_case_has_control_surface", "surfaces": surfaces})
            if category in ATTACK_CATEGORIES and category not in surfaces:
                inconsistencies.append({"case_id": case_id, "issue": "attack_category_missing_from_surfaces"})
        elif sample_type == "normal":
            if category != "normal" or surfaces != ["normal"]:
                inconsistencies.append(
                    {
                        "case_id": case_id,
                        "issue": "normal_case_mismatch",
                        "category": category,
                        "attack_surface": surfaces,
                    }
                )
        elif sample_type == "hard_normal":
            if category != "hard_normal" or surfaces != ["hard_normal"]:
                inconsistencies.append(
                    {
                        "case_id": case_id,
                        "issue": "hard_normal_case_mismatch",
                        "category": category,
                        "attack_surface": surfaces,
                    }
                )
        else:
            inconsistencies.append({"case_id": case_id, "issue": "invalid_sample_type", "sample_type": sample_type})

    case_ids = {str(case.get("case_id", "")) for case in cases}
    core_missing = sorted(core_ids - case_ids)
    assignment_total = sum(attack_surface_assignment_counts.values())
    attack_assignment_total = sum(attack_only_assignment_counts.values())
    primary_total = sum(primary_surface_counts.values())
    extra_surface_assignments = assignment_total - len(cases)
    attack_extra_surface_assignments = attack_assignment_total - sample_type_counts.get("attack", 0)

    data_consistent = not inconsistencies and len(cases) == 1000 and not core_missing
    audit = {
        "stage": "Stage 19",
        "artifact": "candidate_pool_consistency_audit",
        "pool_path": str(pool_path.relative_to(PROJECT_ROOT)),
        "total_count": len(cases),
        "expected_total_count": 1000,
        "total_count_matches_expected": len(cases) == 1000,
        "sample_type_counts": counter_dict(sample_type_counts),
        "category_counts": counter_dict(category_counts),
        "primary_attack_surface_counts": counter_dict(primary_surface_counts),
        "primary_attack_surface_count_sum": primary_total,
        "primary_attack_surface_count_sum_matches_total": primary_total == len(cases),
        "attack_surface_assignment_counts": counter_dict(attack_surface_assignment_counts),
        "attack_surface_assignment_count_sum": assignment_total,
        "attack_surface_assignment_count_sum_matches_total": assignment_total == len(cases),
        "attack_sample_attack_surface_assignment_counts": counter_dict(attack_only_assignment_counts),
        "attack_sample_attack_surface_assignment_count_sum": attack_assignment_total,
        "attack_sample_count": sample_type_counts.get("attack", 0),
        "attack_sample_surface_assignment_count_matches_attack_count": attack_assignment_total
        == sample_type_counts.get("attack", 0),
        "multi_surface_case_count": len(multi_surface_cases),
        "multi_surface_cases": multi_surface_cases,
        "extra_surface_assignments": extra_surface_assignments,
        "attack_extra_surface_assignments": attack_extra_surface_assignments,
        "inconsistency_count": len(inconsistencies),
        "inconsistencies": inconsistencies,
        "core_40": {
            "expected_count": len(core_ids),
            "present_count": len(core_ids) - len(core_missing),
            "missing_count": len(core_missing),
            "missing_ids": core_missing,
        },
        "stage18_750_vs_753_resolution": {
            "reported_attack_sample_count": sample_type_counts.get("attack", 0),
            "attack_surface_assignments_for_attack_samples": attack_assignment_total,
            "extra_assignments_from_multi_surface_core_cases": attack_extra_surface_assignments,
            "cause": "attack_surface is a multi-label array; three preserved core attack cases each carry one secondary attack surface.",
            "judgment": "reporting_ambiguity_not_data_corruption",
        },
        "data_consistent": data_consistent,
    }
    return audit


def markdown_table(mapping: dict[str, int], key_header: str) -> str:
    lines = [f"| {key_header} | count |", "| --- | ---: |"]
    for key, count in mapping.items():
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines)


def write_audit_reports(audit: dict[str, Any], json_path: Path, md_path: Path) -> None:
    write_json(json_path, audit)
    resolution = audit["stage18_750_vs_753_resolution"]
    lines = [
        "# Stage 19 Candidate Pool Consistency Audit",
        "",
        f"- total_count: `{audit['total_count']}`",
        f"- total_count_matches_expected: `{audit['total_count_matches_expected']}`",
        f"- data_consistent: `{audit['data_consistent']}`",
        f"- core_40_present: `{audit['core_40']['present_count']}` / `{audit['core_40']['expected_count']}`",
        "",
        "## Sample Type",
        "",
        markdown_table(audit["sample_type_counts"], "sample_type"),
        "",
        "## Primary Attack Surface",
        "",
        "Primary attack surface is one count per case: attack cases use `category`, while normal and hard-normal controls use their sample type.",
        "",
        markdown_table(audit["primary_attack_surface_counts"], "primary_attack_surface"),
        "",
        f"- primary_count_sum: `{audit['primary_attack_surface_count_sum']}`",
        f"- primary_count_sum_matches_total: `{audit['primary_attack_surface_count_sum_matches_total']}`",
        "",
        "## Multi-Label Attack Surface Assignments",
        "",
        "This is a multi-label count over the `attack_surface` array, so the sum can exceed the number of cases.",
        "",
        markdown_table(audit["attack_surface_assignment_counts"], "attack_surface"),
        "",
        f"- assignment_count_sum: `{audit['attack_surface_assignment_count_sum']}`",
        f"- assignment_count_sum_matches_total: `{audit['attack_surface_assignment_count_sum_matches_total']}`",
        f"- multi_surface_case_count: `{audit['multi_surface_case_count']}`",
        "",
        "## Stage 18 750 vs 753 Resolution",
        "",
        f"- attack_sample_count: `{resolution['reported_attack_sample_count']}`",
        f"- attack_surface_assignments_for_attack_samples: `{resolution['attack_surface_assignments_for_attack_samples']}`",
        f"- extra_assignments_from_multi_surface_core_cases: `{resolution['extra_assignments_from_multi_surface_core_cases']}`",
        f"- cause: {resolution['cause']}",
        f"- judgment: `{resolution['judgment']}`",
        "",
        "## Multi-Surface Cases",
        "",
    ]
    if audit["multi_surface_cases"]:
        for item in audit["multi_surface_cases"]:
            lines.append(f"- `{item['case_id']}`: {', '.join(item['attack_surface'])}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Inconsistencies",
            "",
        ]
    )
    if audit["inconsistencies"]:
        for item in audit["inconsistencies"]:
            lines.append(f"- `{item.get('case_id')}`: {item.get('issue')}")
    else:
        lines.append("- none")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_candidate_pool(
    pool_path: Path = DEFAULT_POOL,
    schema_path: Path = DEFAULT_SCHEMA,
    core_path: Path = DEFAULT_CORE,
    json_path: Path = DEFAULT_AUDIT_JSON,
    md_path: Path = DEFAULT_AUDIT_MD,
) -> dict[str, Any]:
    audit = build_consistency_audit(pool_path=pool_path, schema_path=schema_path, core_path=core_path)
    write_audit_reports(audit, json_path=json_path, md_path=md_path)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 18 candidate-pool consistency for Stage 19.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_AUDIT_MD)
    args = parser.parse_args()
    audit = audit_candidate_pool(
        pool_path=args.pool,
        schema_path=args.schema,
        core_path=args.core,
        json_path=args.json_output,
        md_path=args.md_output,
    )
    if not audit["data_consistent"]:
        print("Candidate pool consistency audit found errors.")
        return 1
    print("Candidate pool consistency audit passed.")
    print(audit["stage18_750_vs_753_resolution"]["cause"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
