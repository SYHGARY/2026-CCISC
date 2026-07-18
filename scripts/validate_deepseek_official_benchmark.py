from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560.json"
DEFAULT_CANDIDATE_POOL = PROJECT_ROOT / "data" / "deepseek_candidate_pool_1000.json"
DEFAULT_SCHEMA = PROJECT_ROOT / "data" / "deepseek_extended_case_schema.json"
DEFAULT_CORE = PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
DEFAULT_COVERAGE_JSON = PROJECT_ROOT / "outputs" / "stage19_official_benchmark_coverage_report.json"
DEFAULT_COVERAGE_MD = PROJECT_ROOT / "outputs" / "stage19_official_benchmark_coverage_report.md"
DEFAULT_QUALITY_JSON = PROJECT_ROOT / "outputs" / "stage19_official_benchmark_quality_report.json"
DEFAULT_QUALITY_MD = PROJECT_ROOT / "outputs" / "stage19_official_benchmark_quality_report.md"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from select_deepseek_official_benchmark import TARGET_ATTACK_CATEGORY_COUNTS, TARGET_SAMPLE_TYPE_COUNTS
from validate_deepseek_candidate_pool import (
    DESTRUCTIVE_PATTERNS,
    SENSITIVE_PATTERNS,
    load_cases,
    load_core_ids,
    load_json,
    normalize_prompt,
    write_json,
)


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def text_blob(case: dict[str, Any]) -> str:
    return json.dumps(case, ensure_ascii=False, sort_keys=True)


def find_pattern_hits(cases: list[dict[str, Any]], patterns: list[Any]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for case in cases:
        blob = text_blob(case)
        for pattern in patterns:
            if pattern.search(blob):
                hits.append({"case_id": str(case.get("case_id")), "pattern": pattern.pattern})
    return hits


def surface_counts(cases: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for case in cases:
        for surface in case.get("attack_surface", []):
            counts[str(surface)] += 1
    return counts


def primary_surface(case: dict[str, Any]) -> str:
    sample_type = str(case.get("sample_type"))
    if sample_type == "attack":
        return str(case.get("category"))
    return sample_type


def load_benchmark_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_cases(path)


def build_candidate_diff(benchmark_cases: list[dict[str, Any]], candidate_cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = {str(case.get("case_id")) for case in benchmark_cases}
    candidate_ids = {str(case.get("case_id")) for case in candidate_cases}
    candidate_sample_counts = Counter(str(case.get("sample_type")) for case in candidate_cases)
    benchmark_sample_counts = Counter(str(case.get("sample_type")) for case in benchmark_cases)
    candidate_attack_counts = Counter(str(case.get("category")) for case in candidate_cases if case.get("sample_type") == "attack")
    benchmark_attack_counts = Counter(str(case.get("category")) for case in benchmark_cases if case.get("sample_type") == "attack")
    sample_type_delta = {
        key: benchmark_sample_counts.get(key, 0) - candidate_sample_counts.get(key, 0)
        for key in sorted(set(candidate_sample_counts) | set(benchmark_sample_counts))
    }
    attack_category_delta = {
        key: benchmark_attack_counts.get(key, 0) - candidate_attack_counts.get(key, 0)
        for key in sorted(set(candidate_attack_counts) | set(benchmark_attack_counts))
    }
    return {
        "candidate_count": len(candidate_cases),
        "selected_count": len(benchmark_cases),
        "not_selected_count": len(candidate_ids - selected_ids),
        "selected_ratio": round(len(benchmark_cases) / len(candidate_cases), 4) if candidate_cases else 0.0,
        "all_selected_ids_from_candidate_pool": selected_ids.issubset(candidate_ids),
        "sample_type_delta_selected_minus_candidate": sample_type_delta,
        "attack_category_delta_selected_minus_candidate": attack_category_delta,
    }


def validate_official_benchmark(
    benchmark_path: Path = DEFAULT_BENCHMARK,
    candidate_pool_path: Path = DEFAULT_CANDIDATE_POOL,
    schema_path: Path = DEFAULT_SCHEMA,
    core_path: Path = DEFAULT_CORE,
    write_report_files: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cases = load_benchmark_cases(benchmark_path)
    candidate_cases = load_cases(candidate_pool_path)
    schema = load_json(schema_path)
    required = set(schema["required"])
    allowed_sample_types = set(schema["properties"]["sample_type"]["enum"])
    allowed_surfaces = set(schema["properties"]["attack_surface"]["items"]["enum"])
    allowed_categories = set(schema["properties"]["category"]["enum"])
    core_ids = load_core_ids(core_path)

    errors: list[str] = []
    case_ids = [str(case.get("case_id", "")) for case in cases]
    case_id_counts = Counter(case_ids)
    duplicate_case_ids = sorted(case_id for case_id, count in case_id_counts.items() if count > 1)
    prompt_counts = Counter(str(case.get("user_prompt", "")) for case in cases)
    duplicate_prompts = sorted(prompt for prompt, count in prompt_counts.items() if count > 1)
    normalized_prompt_counts = Counter(normalize_prompt(str(case.get("user_prompt", ""))) for case in cases)

    missing_fields: list[dict[str, Any]] = []
    invalid_sample_type_ids: list[str] = []
    invalid_category_ids: list[str] = []
    invalid_surface_entries: list[dict[str, str]] = []
    sample_category_surface_mismatches: list[dict[str, Any]] = []
    hard_normal_missing_boundary: list[str] = []
    low_quality_ids: list[str] = []

    for case in cases:
        case_id = str(case.get("case_id", ""))
        sample_type = str(case.get("sample_type", ""))
        category = str(case.get("category", ""))
        surfaces = [str(surface) for surface in case.get("attack_surface", [])]
        missing = sorted(required - set(case))
        if missing:
            missing_fields.append({"case_id": case_id, "missing": missing})
        if sample_type not in allowed_sample_types:
            invalid_sample_type_ids.append(case_id)
        if category not in allowed_categories:
            invalid_category_ids.append(case_id)
        for surface in surfaces:
            if surface not in allowed_surfaces:
                invalid_surface_entries.append({"case_id": case_id, "surface": surface})
        if sample_type == "attack" and (category not in surfaces or category in {"normal", "hard_normal"}):
            sample_category_surface_mismatches.append(
                {"case_id": case_id, "sample_type": sample_type, "category": category, "attack_surface": surfaces}
            )
        if sample_type == "normal" and (category != "normal" or surfaces != ["normal"]):
            sample_category_surface_mismatches.append(
                {"case_id": case_id, "sample_type": sample_type, "category": category, "attack_surface": surfaces}
            )
        if sample_type == "hard_normal":
            if category != "hard_normal" or surfaces != ["hard_normal"]:
                sample_category_surface_mismatches.append(
                    {"case_id": case_id, "sample_type": sample_type, "category": category, "attack_surface": surfaces}
                )
            if not str(case.get("boundary_reason", "")).strip():
                hard_normal_missing_boundary.append(case_id)
        if "templated_low_quality" in set(case.get("quality_tags", [])):
            low_quality_ids.append(case_id)

    sample_type_counts = Counter(str(case.get("sample_type")) for case in cases)
    category_counts = Counter(str(case.get("category")) for case in cases)
    attack_category_counts = Counter(str(case.get("category")) for case in cases if case.get("sample_type") == "attack")
    primary_surface_counts = Counter(primary_surface(case) for case in cases)
    attack_surface_assignment_counts = surface_counts(cases)
    risk_level_counts = Counter(str(case.get("risk_level", case.get("expected_risk", "missing"))) for case in cases)
    real_priority_counts = Counter(str(case.get("real_priority", "missing")) for case in cases)
    variant_family_counts = Counter(str(case.get("variant_family", "missing")) for case in cases)
    core_present = sorted(core_ids & set(case_ids))
    core_missing = sorted(core_ids - set(case_ids))
    sensitive_hits = find_pattern_hits(cases, SENSITIVE_PATTERNS)
    destructive_hits = find_pattern_hits(cases, DESTRUCTIVE_PATTERNS)
    candidate_diff = build_candidate_diff(cases, candidate_cases)

    if len(cases) != 560:
        errors.append(f"Expected 560 cases, found {len(cases)}.")
    for sample_type, expected_count in TARGET_SAMPLE_TYPE_COUNTS.items():
        if sample_type_counts.get(sample_type, 0) != expected_count:
            errors.append(f"Expected sample_type {sample_type}={expected_count}, found {sample_type_counts.get(sample_type, 0)}.")
    for category, expected_count in TARGET_ATTACK_CATEGORY_COUNTS.items():
        if attack_category_counts.get(category, 0) != expected_count:
            errors.append(f"Expected attack category {category}={expected_count}, found {attack_category_counts.get(category, 0)}.")
    if duplicate_case_ids:
        errors.append(f"Duplicate case_id values: {duplicate_case_ids[:10]}")
    if missing_fields:
        errors.append(f"Cases with missing fields: {len(missing_fields)}")
    if invalid_sample_type_ids:
        errors.append(f"Invalid sample_type cases: {invalid_sample_type_ids[:10]}")
    if invalid_category_ids:
        errors.append(f"Invalid category cases: {invalid_category_ids[:10]}")
    if invalid_surface_entries:
        errors.append(f"Invalid attack_surface entries: {invalid_surface_entries[:10]}")
    if sample_category_surface_mismatches:
        errors.append(f"sample_type/category/attack_surface mismatches: {len(sample_category_surface_mismatches)}")
    if core_missing:
        errors.append(f"Core cases missing: {len(core_missing)}")
    if sensitive_hits:
        errors.append(f"Sensitive-looking real key patterns found: {len(sensitive_hits)}")
    if destructive_hits:
        errors.append(f"Destructive command patterns found: {len(destructive_hits)}")
    if hard_normal_missing_boundary:
        errors.append(f"Hard-normal cases missing boundary_reason: {len(hard_normal_missing_boundary)}")
    if low_quality_ids:
        errors.append(f"Low-quality tagged cases selected: {len(low_quality_ids)}")
    if not candidate_diff["all_selected_ids_from_candidate_pool"]:
        errors.append("Official benchmark includes IDs not found in the candidate pool.")

    coverage = {
        "total_count": len(cases),
        "sample_type_counts": counter_dict(sample_type_counts),
        "category_counts": counter_dict(category_counts),
        "attack_category_counts": counter_dict(attack_category_counts),
        "primary_attack_surface_counts": counter_dict(primary_surface_counts),
        "attack_surface_assignment_counts": counter_dict(attack_surface_assignment_counts),
        "attack_surface_assignment_count_sum": sum(attack_surface_assignment_counts.values()),
        "risk_level_counts": counter_dict(risk_level_counts),
        "real_priority_counts": counter_dict(real_priority_counts),
        "variant_family_counts": counter_dict(variant_family_counts),
        "variant_family_unique_count": len(variant_family_counts),
        "core_40": {
            "expected_count": len(core_ids),
            "present_count": len(core_present),
            "missing_count": len(core_missing),
            "missing_ids": core_missing,
        },
        "normal_count": sample_type_counts.get("normal", 0),
        "hard_normal_count": sample_type_counts.get("hard_normal", 0),
        "hard_normal_boundary_reason_coverage": {
            "with_boundary_reason": sample_type_counts.get("hard_normal", 0) - len(hard_normal_missing_boundary),
            "missing_boundary_reason": len(hard_normal_missing_boundary),
        },
        "candidate_pool_diff": candidate_diff,
        "target_counts": {
            "sample_type": TARGET_SAMPLE_TYPE_COUNTS,
            "attack_category": TARGET_ATTACK_CATEGORY_COUNTS,
        },
    }
    quality = {
        "validation_passed": not errors,
        "duplicate_case_id_count": len(duplicate_case_ids),
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_prompt_count": len(duplicate_prompts),
        "duplicate_prompts_sample": duplicate_prompts[:20],
        "near_duplicate_prompt_group_count": sum(1 for count in normalized_prompt_counts.values() if count > 1),
        "missing_field_case_count": len(missing_fields),
        "missing_fields": missing_fields[:50],
        "low_quality_or_templated_sample_count": len(low_quality_ids),
        "low_quality_or_templated_case_ids": low_quality_ids,
        "hard_normal_missing_boundary_reason_count": len(hard_normal_missing_boundary),
        "hard_normal_missing_boundary_reason_case_ids": hard_normal_missing_boundary,
        "sample_category_surface_mismatch_count": len(sample_category_surface_mismatches),
        "sample_category_surface_mismatches": sample_category_surface_mismatches[:50],
        "sensitive_pattern_hit_count": len(sensitive_hits),
        "sensitive_pattern_hits": sensitive_hits,
        "destructive_pattern_hit_count": len(destructive_hits),
        "destructive_pattern_hits": destructive_hits,
        "errors": errors,
    }

    if write_report_files:
        write_reports(coverage, quality)
    return coverage, quality, errors


def markdown_table(mapping: dict[str, Any], key_header: str) -> str:
    lines = [f"| {key_header} | count |", "| --- | ---: |"]
    for key, value in mapping.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def write_reports(
    coverage: dict[str, Any],
    quality: dict[str, Any],
    coverage_json: Path = DEFAULT_COVERAGE_JSON,
    coverage_md: Path = DEFAULT_COVERAGE_MD,
    quality_json: Path = DEFAULT_QUALITY_JSON,
    quality_md: Path = DEFAULT_QUALITY_MD,
) -> None:
    write_json(coverage_json, coverage)
    write_json(quality_json, quality)

    coverage_lines = [
        "# Stage 19 Official Benchmark Coverage Report",
        "",
        f"- total_count: `{coverage['total_count']}`",
        f"- core_40_present: `{coverage['core_40']['present_count']}` / `{coverage['core_40']['expected_count']}`",
        f"- variant_family_unique_count: `{coverage['variant_family_unique_count']}`",
        f"- selected_ratio_from_candidate_pool: `{coverage['candidate_pool_diff']['selected_ratio']}`",
        "",
        "## Sample Type",
        "",
        markdown_table(coverage["sample_type_counts"], "sample_type"),
        "",
        "## Category",
        "",
        markdown_table(coverage["category_counts"], "category"),
        "",
        "## Primary Attack Surface",
        "",
        "Primary attack surface is one count per case. It is the distribution validated against the Stage 19 plan.",
        "",
        markdown_table(coverage["primary_attack_surface_counts"], "primary_attack_surface"),
        "",
        "## Multi-Label Attack Surface Assignments",
        "",
        "This count expands every item in the `attack_surface` array, so preserved multi-surface core cases add secondary assignments.",
        "",
        markdown_table(coverage["attack_surface_assignment_counts"], "attack_surface"),
        "",
        f"- attack_surface_assignment_count_sum: `{coverage['attack_surface_assignment_count_sum']}`",
        "",
        "## Risk Level",
        "",
        markdown_table(coverage["risk_level_counts"], "risk_level"),
        "",
        "## Real Priority",
        "",
        markdown_table(coverage["real_priority_counts"], "real_priority"),
        "",
        "## Controls And Core Cases",
        "",
        f"- normal_count: `{coverage['normal_count']}`",
        f"- hard_normal_count: `{coverage['hard_normal_count']}`",
        f"- hard_normal_missing_boundary_reason: `{coverage['hard_normal_boundary_reason_coverage']['missing_boundary_reason']}`",
        f"- core_40_missing_count: `{coverage['core_40']['missing_count']}`",
        "",
        "## Candidate Pool Difference",
        "",
        f"- candidate_count: `{coverage['candidate_pool_diff']['candidate_count']}`",
        f"- selected_count: `{coverage['candidate_pool_diff']['selected_count']}`",
        f"- not_selected_count: `{coverage['candidate_pool_diff']['not_selected_count']}`",
        f"- all_selected_ids_from_candidate_pool: `{coverage['candidate_pool_diff']['all_selected_ids_from_candidate_pool']}`",
    ]
    coverage_md.parent.mkdir(parents=True, exist_ok=True)
    coverage_md.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

    quality_lines = [
        "# Stage 19 Official Benchmark Quality Report",
        "",
        f"- validation_passed: `{quality['validation_passed']}`",
        f"- duplicate_case_id_count: `{quality['duplicate_case_id_count']}`",
        f"- duplicate_prompt_count: `{quality['duplicate_prompt_count']}`",
        f"- near_duplicate_prompt_group_count: `{quality['near_duplicate_prompt_group_count']}`",
        f"- missing_field_case_count: `{quality['missing_field_case_count']}`",
        f"- low_quality_or_templated_sample_count: `{quality['low_quality_or_templated_sample_count']}`",
        f"- hard_normal_missing_boundary_reason_count: `{quality['hard_normal_missing_boundary_reason_count']}`",
        f"- sample_category_surface_mismatch_count: `{quality['sample_category_surface_mismatch_count']}`",
        f"- sensitive_pattern_hit_count: `{quality['sensitive_pattern_hit_count']}`",
        f"- destructive_pattern_hit_count: `{quality['destructive_pattern_hit_count']}`",
        "",
        "## Errors",
        "",
    ]
    if quality["errors"]:
        quality_lines.extend(f"- {error}" for error in quality["errors"])
    else:
        quality_lines.append("- none")
    quality_md.parent.mkdir(parents=True, exist_ok=True)
    quality_md.write_text("\n".join(quality_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage 19 official benchmark.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--no-write-reports", action="store_true")
    args = parser.parse_args()

    coverage, _quality, errors = validate_official_benchmark(
        benchmark_path=args.benchmark,
        candidate_pool_path=args.candidate_pool,
        schema_path=args.schema,
        core_path=args.core,
        write_report_files=not args.no_write_reports,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Official benchmark validation passed.")
    print(f"total_count={coverage['total_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
