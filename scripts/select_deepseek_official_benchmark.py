from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = PROJECT_ROOT / "data" / "deepseek_candidate_pool_1000.json"
DEFAULT_RULES = PROJECT_ROOT / "outputs" / "stage17_official_benchmark_selection_rules.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560.json"
DEFAULT_TRACE_JSON = PROJECT_ROOT / "outputs" / "stage19_official_benchmark_selection_trace.json"
DEFAULT_TRACE_MD = PROJECT_ROOT / "outputs" / "stage19_official_benchmark_selection_trace.md"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_deepseek_candidate_pool import (
    DESTRUCTIVE_PATTERNS,
    SENSITIVE_PATTERNS,
    load_cases,
    load_core_ids,
    load_json,
    normalize_prompt,
    write_json,
)


TARGET_SAMPLE_TYPE_COUNTS = {"attack": 400, "normal": 80, "hard_normal": 80}
TARGET_ATTACK_CATEGORY_COUNTS = {
    "prompt_injection": 55,
    "jailbreak": 45,
    "tool_hijacking": 50,
    "dangerous_code_execution": 50,
    "file_access_overwrite": 45,
    "sensitive_information_leakage": 55,
    "memory_poisoning": 40,
    "environment_pollution": 35,
    "multi_agent_error_propagation": 25,
}
RISK_WEIGHT = {"critical": 5, "high": 4, "medium": 3, "low": 2, "none": 1}


def text_blob(case: dict[str, Any]) -> str:
    return json.dumps(case, ensure_ascii=False, sort_keys=True)


def has_forbidden_pattern(case: dict[str, Any]) -> bool:
    blob = text_blob(case)
    return any(pattern.search(blob) for pattern in SENSITIVE_PATTERNS + DESTRUCTIVE_PATTERNS)


def is_schema_complete(case: dict[str, Any], required_fields: set[str]) -> bool:
    if not required_fields.issubset(case):
        return False
    if not case.get("success_criteria") or not case.get("failure_criteria"):
        return False
    if case.get("sample_type") == "hard_normal" and not str(case.get("boundary_reason", "")).strip():
        return False
    return True


def is_eligible(case: dict[str, Any], required_fields: set[str]) -> bool:
    tags = set(case.get("quality_tags", []))
    if "templated_low_quality" in tags:
        return False
    if not is_schema_complete(case, required_fields):
        return False
    if has_forbidden_pattern(case):
        return False
    return True


def selection_sort_key(case: dict[str, Any]) -> tuple[int, int, int, int, str]:
    requires_count = sum(
        1
        for field in ("requires_memory", "requires_file", "requires_code", "requires_external_send")
        if case.get(field)
    )
    return (
        int(bool(case.get("is_core_case"))),
        int(case.get("real_priority", 0)),
        RISK_WEIGHT.get(str(case.get("risk_level", case.get("expected_risk", "none"))), 0),
        requires_count,
        str(case.get("case_id", "")),
    )


def balanced_order(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_family[str(case.get("variant_family", "missing"))].append(case)
    queues: list[deque[dict[str, Any]]] = []
    for family in sorted(by_family):
        ordered = sorted(by_family[family], key=selection_sort_key, reverse=True)
        queues.append(deque(ordered))

    result: list[dict[str, Any]] = []
    while queues:
        next_round: list[deque[dict[str, Any]]] = []
        for queue in queues:
            if queue:
                result.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round
    return result


def count_by(cases: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(case.get(key)) for case in cases)


def add_case(case: dict[str, Any], selected: list[dict[str, Any]], selected_ids: set[str]) -> None:
    selected.append(case)
    selected_ids.add(str(case["case_id"]))


def select_official_benchmark(
    pool_path: Path = DEFAULT_POOL,
    rules_path: Path = DEFAULT_RULES,
    core_path: Path = PROJECT_ROOT / "data" / "deepseek_batch_cases.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = load_cases(pool_path)
    schema = load_json(PROJECT_ROOT / "data" / "deepseek_extended_case_schema.json")
    required_fields = set(schema["required"])
    core_ids = load_core_ids(core_path)
    _rules_text = rules_path.read_text(encoding="utf-8")

    eligible_cases = [case for case in cases if is_eligible(case, required_fields)]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    excluded_low_quality = len(cases) - len(eligible_cases)

    core_cases = [case for case in eligible_cases if str(case.get("case_id")) in core_ids]
    for case in sorted(core_cases, key=lambda item: str(item["case_id"])):
        add_case(case, selected, selected_ids)

    for category, target_count in TARGET_ATTACK_CATEGORY_COUNTS.items():
        current = sum(1 for case in selected if case.get("sample_type") == "attack" and case.get("category") == category)
        needed = target_count - current
        if needed < 0:
            raise ValueError(f"Core cases exceed target for attack category {category}: {current}>{target_count}")
        candidates = [
            case
            for case in eligible_cases
            if case.get("sample_type") == "attack"
            and case.get("category") == category
            and str(case.get("case_id")) not in selected_ids
        ]
        ordered = balanced_order(candidates)
        if len(ordered) < needed:
            raise ValueError(f"Not enough eligible cases for attack category {category}: need {needed}, have {len(ordered)}")
        for case in ordered[:needed]:
            add_case(case, selected, selected_ids)

    for sample_type in ("normal", "hard_normal"):
        target_count = TARGET_SAMPLE_TYPE_COUNTS[sample_type]
        current = sum(1 for case in selected if case.get("sample_type") == sample_type)
        needed = target_count - current
        if needed < 0:
            raise ValueError(f"Core cases exceed target for sample_type {sample_type}: {current}>{target_count}")
        candidates = [
            case
            for case in eligible_cases
            if case.get("sample_type") == sample_type and str(case.get("case_id")) not in selected_ids
        ]
        ordered = balanced_order(candidates)
        if len(ordered) < needed:
            raise ValueError(f"Not enough eligible {sample_type} cases: need {needed}, have {len(ordered)}")
        for case in ordered[:needed]:
            add_case(case, selected, selected_ids)

    selected_sample_counts = count_by(selected, "sample_type")
    selected_attack_counts = Counter(
        str(case.get("category")) for case in selected if case.get("sample_type") == "attack"
    )
    normalized_prompt_counts = Counter(normalize_prompt(str(case.get("user_prompt", ""))) for case in selected)

    trace = {
        "stage": "Stage 19",
        "artifact": "official_benchmark_selection_trace",
        "source_pool": str(pool_path.relative_to(PROJECT_ROOT)),
        "rules_source": str(rules_path.relative_to(PROJECT_ROOT)),
        "target_counts": {
            "sample_type": TARGET_SAMPLE_TYPE_COUNTS,
            "attack_category": TARGET_ATTACK_CATEGORY_COUNTS,
        },
        "stage17_rules_note": "Stage 17 documented 420/70/70, but Stage 19 user instruction supersedes it with 400/80/80.",
        "real_api_called": False,
        "experiments_run": False,
        "selection_policy": [
            "Preserve all eligible core 40 cases first.",
            "Fill attack categories to the Stage 19 target counts.",
            "Fill normal and hard-normal controls to 80 each.",
            "Within each bucket, round-robin by variant_family, then prefer core cases, higher real_priority, higher risk, and broader tool-path coverage.",
            "Reject missing-schema, sensitive-pattern, destructive-pattern, and templated_low_quality-tagged cases.",
        ],
        "candidate_count": len(cases),
        "eligible_candidate_count": len(eligible_cases),
        "excluded_low_quality_or_invalid_count": excluded_low_quality,
        "selected_count": len(selected),
        "selected_sample_type_counts": dict(sorted(selected_sample_counts.items())),
        "selected_attack_category_counts": dict(sorted(selected_attack_counts.items())),
        "selected_core_40_count": sum(1 for case in selected if str(case.get("case_id")) in core_ids),
        "missing_core_40_ids": sorted(core_ids - {str(case.get("case_id")) for case in selected}),
        "variant_family_counts": dict(sorted(count_by(selected, "variant_family").items())),
        "real_priority_counts": dict(sorted(count_by(selected, "real_priority").items())),
        "near_duplicate_prompt_group_count": sum(1 for count in normalized_prompt_counts.values() if count > 1),
    }

    output = {
        "metadata": {
            "stage": "Stage 19",
            "artifact": "deepseek_official_benchmark_560",
            "generated_by": "scripts/select_deepseek_official_benchmark.py",
            "source_pool": str(pool_path.relative_to(PROJECT_ROOT)),
            "real_api_called": False,
            "experiments_run": False,
            "target_sample_type_counts": TARGET_SAMPLE_TYPE_COUNTS,
            "target_attack_category_counts": TARGET_ATTACK_CATEGORY_COUNTS,
            "core_cases_preserved": trace["selected_core_40_count"],
            "selection_rules": str(rules_path.relative_to(PROJECT_ROOT)),
            "claim_boundary": "Official benchmark case selection only; no dry-run or real-model behavior has been evaluated.",
        },
        "cases": selected,
    }
    return output, trace


def markdown_table(mapping: dict[str, Any], key_header: str) -> str:
    lines = [f"| {key_header} | count |", "| --- | ---: |"]
    for key, value in mapping.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def write_selection_trace(trace: dict[str, Any], json_path: Path, md_path: Path) -> None:
    write_json(json_path, trace)
    lines = [
        "# Stage 19 Official Benchmark Selection Trace",
        "",
        f"- selected_count: `{trace['selected_count']}`",
        f"- candidate_count: `{trace['candidate_count']}`",
        f"- eligible_candidate_count: `{trace['eligible_candidate_count']}`",
        f"- selected_core_40_count: `{trace['selected_core_40_count']}`",
        f"- real_api_called: `{trace['real_api_called']}`",
        f"- experiments_run: `{trace['experiments_run']}`",
        f"- stage17_rules_note: {trace['stage17_rules_note']}",
        "",
        "## Sample Type",
        "",
        markdown_table(trace["selected_sample_type_counts"], "sample_type"),
        "",
        "## Attack Category",
        "",
        markdown_table(trace["selected_attack_category_counts"], "category"),
        "",
        "## Selection Policy",
        "",
    ]
    lines.extend(f"- {item}" for item in trace["selection_policy"])
    lines.extend(
        [
            "",
            "## Core 40",
            "",
            f"- missing_core_40_count: `{len(trace['missing_core_40_ids'])}`",
        ]
    )
    if trace["missing_core_40_ids"]:
        lines.extend(f"- `{case_id}`" for case_id in trace["missing_core_40_ids"])
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_official_benchmark(
    benchmark: dict[str, Any],
    trace: dict[str, Any],
    output_path: Path = DEFAULT_OUTPUT,
    trace_json: Path = DEFAULT_TRACE_JSON,
    trace_md: Path = DEFAULT_TRACE_MD,
) -> None:
    write_json(output_path, benchmark)
    write_selection_trace(trace, trace_json, trace_md)


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the Stage 19 official benchmark from the 1000-case pool.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--core", type=Path, default=PROJECT_ROOT / "data" / "deepseek_batch_cases.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace-json", type=Path, default=DEFAULT_TRACE_JSON)
    parser.add_argument("--trace-md", type=Path, default=DEFAULT_TRACE_MD)
    args = parser.parse_args()

    benchmark, trace = select_official_benchmark(pool_path=args.pool, rules_path=args.rules, core_path=args.core)
    write_official_benchmark(
        benchmark,
        trace,
        output_path=args.output,
        trace_json=args.trace_json,
        trace_md=args.trace_md,
    )
    print(f"Selected {trace['selected_count']} official benchmark cases at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
