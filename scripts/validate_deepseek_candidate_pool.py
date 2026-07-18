from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = PROJECT_ROOT / "data" / "deepseek_candidate_pool_1000.json"
DEFAULT_SCHEMA = PROJECT_ROOT / "data" / "deepseek_extended_case_schema.json"
DEFAULT_DISTRIBUTION = PROJECT_ROOT / "outputs" / "stage17_candidate_pool_distribution.json"
DEFAULT_CORE = PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
DEFAULT_COVERAGE_JSON = PROJECT_ROOT / "outputs" / "stage18_candidate_pool_coverage_report.json"
DEFAULT_COVERAGE_MD = PROJECT_ROOT / "outputs" / "stage18_candidate_pool_coverage_report.md"
DEFAULT_QUALITY_JSON = PROJECT_ROOT / "outputs" / "stage18_candidate_pool_quality_report.json"
DEFAULT_QUALITY_MD = PROJECT_ROOT / "outputs" / "stage18_candidate_pool_quality_report.md"

SENSITIVE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
]

DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bdel\s+/s\s+/q\s+[a-z]:\\", re.IGNORECASE),
    re.compile(r"\bshutdown\s+/s\b", re.IGNORECASE),
    re.compile(r"\bmkfs\.", re.IGNORECASE),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_prompt(prompt: str) -> str:
    text = prompt.lower()
    text = re.sub(r"\d+", "<n>", text)
    text = re.sub(r"[a-z0-9_.+-]+@example\.test", "<email>", text)
    text = re.sub(r"case[_ -]?\d+", "case<n>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_cases(pool_path: Path = DEFAULT_POOL) -> list[dict[str, Any]]:
    payload = load_json(pool_path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    raise ValueError(f"Unsupported candidate pool shape in {pool_path}")


def load_core_ids(core_path: Path = DEFAULT_CORE) -> set[str]:
    core = load_json(core_path)
    ids: set[str] = set()
    for group in ("attack", "normal", "hard_normal"):
        for item in core.get(group, []):
            ids.add(item["id"])
    return ids


def schema_info(schema_path: Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    schema = load_json(schema_path)
    properties = schema["properties"]
    allowed_sample_types = set(properties["sample_type"]["enum"])
    allowed_surfaces = set(properties["attack_surface"]["items"]["enum"])
    allowed_categories = set(properties["category"]["enum"])
    return {
        "required": set(schema["required"]),
        "properties": set(properties),
        "allowed_sample_types": allowed_sample_types,
        "allowed_surfaces": allowed_surfaces,
        "allowed_categories": allowed_categories,
    }


def expected_counts(distribution_path: Path = DEFAULT_DISTRIBUTION) -> dict[str, Any]:
    distribution = load_json(distribution_path)
    return {
        "sample_type_counts": distribution["sample_type_counts"],
        "attack_category_counts": distribution["attack_category_counts"],
        "total": sum(distribution["sample_type_counts"].values()),
    }


def text_blob(case: dict[str, Any]) -> str:
    return json.dumps(case, ensure_ascii=False, sort_keys=True)


def find_sensitive_strings(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for case in cases:
        blob = text_blob(case)
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(blob):
                hits.append({"case_id": case.get("case_id"), "pattern": pattern.pattern})
    return hits


def find_destructive_strings(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for case in cases:
        blob = text_blob(case)
        for pattern in DESTRUCTIVE_PATTERNS:
            if pattern.search(blob):
                hits.append({"case_id": case.get("case_id"), "pattern": pattern.pattern})
    return hits


def build_reports(
    cases: list[dict[str, Any]],
    schema: dict[str, Any],
    expected: dict[str, Any],
    core_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    required = schema["required"]
    allowed_sample_types = schema["allowed_sample_types"]
    allowed_categories = schema["allowed_categories"]
    allowed_surfaces = schema["allowed_surfaces"]

    case_ids = [str(case.get("case_id", "")) for case in cases]
    case_id_counts = Counter(case_ids)
    duplicate_case_ids = sorted(case_id for case_id, count in case_id_counts.items() if count > 1)

    prompt_counts = Counter(str(case.get("user_prompt", "")) for case in cases)
    duplicate_prompts = sorted(prompt for prompt, count in prompt_counts.items() if count > 1)
    normalized_counts = Counter(normalize_prompt(str(case.get("user_prompt", ""))) for case in cases)
    near_duplicate_prompt_groups = sum(1 for count in normalized_counts.values() if count > 1)

    missing_fields: list[dict[str, Any]] = []
    invalid_sample_types: list[str] = []
    invalid_categories: list[str] = []
    invalid_surfaces: list[dict[str, Any]] = []
    hard_normal_missing_boundary = 0
    missing_success_criteria = 0
    missing_failure_criteria = 0
    templated_low_quality_count = 0

    for case in cases:
        case_id = str(case.get("case_id", ""))
        missing = sorted(required - set(case))
        if missing:
            missing_fields.append({"case_id": case_id, "missing": missing})

        sample_type = case.get("sample_type")
        if sample_type not in allowed_sample_types:
            invalid_sample_types.append(case_id)

        category = case.get("category")
        if category not in allowed_categories:
            invalid_categories.append(case_id)

        for surface in case.get("attack_surface", []):
            if surface not in allowed_surfaces:
                invalid_surfaces.append({"case_id": case_id, "surface": surface})

        if sample_type == "hard_normal" and not str(case.get("boundary_reason", "")).strip():
            hard_normal_missing_boundary += 1

        if not case.get("success_criteria"):
            missing_success_criteria += 1
        if not case.get("failure_criteria"):
            missing_failure_criteria += 1

        tags = set(case.get("quality_tags", []))
        if "templated_low_quality" in tags:
            templated_low_quality_count += 1

    sample_type_counts = Counter(str(case.get("sample_type")) for case in cases)
    category_counts = Counter(str(case.get("category")) for case in cases)
    attack_surface_counts: Counter[str] = Counter()
    risk_level_counts = Counter(str(case.get("risk_level", case.get("expected_risk", "missing"))) for case in cases)
    real_priority_counts = Counter(str(case.get("real_priority", "missing")) for case in cases)
    for case in cases:
        for surface in case.get("attack_surface", []):
            attack_surface_counts[str(surface)] += 1

    core_present = sorted(core_ids & set(case_ids))
    core_missing = sorted(core_ids - set(case_ids))
    sensitive_hits = find_sensitive_strings(cases)
    destructive_hits = find_destructive_strings(cases)

    if len(cases) != expected["total"]:
        errors.append(f"Expected {expected['total']} cases, found {len(cases)}.")
    for key, value in expected["sample_type_counts"].items():
        if sample_type_counts.get(key, 0) != value:
            errors.append(f"Expected sample_type {key}={value}, found {sample_type_counts.get(key, 0)}.")
    for key, value in expected["attack_category_counts"].items():
        if category_counts.get(key, 0) != value:
            errors.append(f"Expected attack category {key}={value}, found {category_counts.get(key, 0)}.")
    if duplicate_case_ids:
        errors.append(f"Duplicate case_id values: {duplicate_case_ids[:10]}")
    if missing_fields:
        errors.append(f"Cases with missing fields: {len(missing_fields)}")
    if invalid_sample_types:
        errors.append(f"Invalid sample_type cases: {invalid_sample_types[:10]}")
    if invalid_categories:
        errors.append(f"Invalid category cases: {invalid_categories[:10]}")
    if invalid_surfaces:
        errors.append(f"Invalid attack_surface entries: {invalid_surfaces[:10]}")
    if hard_normal_missing_boundary:
        errors.append(f"Hard-normal cases missing boundary_reason: {hard_normal_missing_boundary}")
    if missing_success_criteria:
        errors.append(f"Cases missing success_criteria: {missing_success_criteria}")
    if missing_failure_criteria:
        errors.append(f"Cases missing failure_criteria: {missing_failure_criteria}")
    if sensitive_hits:
        errors.append(f"Sensitive-looking real key patterns found: {len(sensitive_hits)}")
    if destructive_hits:
        errors.append(f"Destructive command patterns found: {len(destructive_hits)}")
    if core_missing:
        errors.append(f"Core cases missing from candidate pool: {len(core_missing)}")

    coverage = {
        "total_count": len(cases),
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "attack_surface_counts": dict(sorted(attack_surface_counts.items())),
        "risk_level_counts": dict(sorted(risk_level_counts.items())),
        "real_priority_counts": dict(sorted(real_priority_counts.items())),
        "core_40": {
            "expected_count": len(core_ids),
            "present_count": len(core_present),
            "missing_count": len(core_missing),
            "missing_ids": core_missing,
        },
    }
    quality = {
        "missing_field_case_count": len(missing_fields),
        "missing_fields": missing_fields[:50],
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_prompt_count": len(duplicate_prompts),
        "duplicate_prompts_sample": duplicate_prompts[:20],
        "near_duplicate_prompt_group_count": near_duplicate_prompt_groups,
        "hard_normal_missing_boundary_reason_count": hard_normal_missing_boundary,
        "missing_success_criteria_count": missing_success_criteria,
        "missing_failure_criteria_count": missing_failure_criteria,
        "templated_low_quality_sample_count": templated_low_quality_count,
        "sensitive_pattern_hits": sensitive_hits,
        "destructive_pattern_hits": destructive_hits,
        "invalid_sample_type_case_ids": invalid_sample_types,
        "invalid_category_case_ids": invalid_categories,
        "invalid_attack_surface_entries": invalid_surfaces,
        "validation_passed": not errors,
        "errors": errors,
    }
    return coverage, quality, errors


def markdown_table(counter: dict[str, Any], key_header: str, count_header: str = "count") -> str:
    lines = [f"| {key_header} | {count_header} |", "| --- | ---: |"]
    for key, value in counter.items():
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

    coverage_text = [
        "# Stage 18 Candidate Pool Coverage Report",
        "",
        f"- total_count: `{coverage['total_count']}`",
        f"- core_40_present: `{coverage['core_40']['present_count']}` / `{coverage['core_40']['expected_count']}`",
        "",
        "## Sample Type",
        "",
        markdown_table(coverage["sample_type_counts"], "sample_type"),
        "",
        "## Category",
        "",
        markdown_table(coverage["category_counts"], "category"),
        "",
        "## Attack Surface",
        "",
        markdown_table(coverage["attack_surface_counts"], "attack_surface"),
        "",
        "## Risk Level",
        "",
        markdown_table(coverage["risk_level_counts"], "risk_level"),
        "",
        "## Real Priority",
        "",
        markdown_table(coverage["real_priority_counts"], "real_priority"),
        "",
        "## Core 40",
        "",
        f"- missing_count: `{coverage['core_40']['missing_count']}`",
    ]
    coverage_md.parent.mkdir(parents=True, exist_ok=True)
    coverage_md.write_text("\n".join(coverage_text) + "\n", encoding="utf-8")

    quality_text = [
        "# Stage 18 Candidate Pool Quality Report",
        "",
        f"- validation_passed: `{quality['validation_passed']}`",
        f"- missing_field_case_count: `{quality['missing_field_case_count']}`",
        f"- duplicate_case_id_count: `{len(quality['duplicate_case_ids'])}`",
        f"- duplicate_prompt_count: `{quality['duplicate_prompt_count']}`",
        f"- near_duplicate_prompt_group_count: `{quality['near_duplicate_prompt_group_count']}`",
        f"- hard_normal_missing_boundary_reason_count: `{quality['hard_normal_missing_boundary_reason_count']}`",
        f"- missing_success_criteria_count: `{quality['missing_success_criteria_count']}`",
        f"- missing_failure_criteria_count: `{quality['missing_failure_criteria_count']}`",
        f"- templated_low_quality_sample_count: `{quality['templated_low_quality_sample_count']}`",
        f"- sensitive_pattern_hit_count: `{len(quality['sensitive_pattern_hits'])}`",
        f"- destructive_pattern_hit_count: `{len(quality['destructive_pattern_hits'])}`",
        "",
        "## Errors",
        "",
    ]
    if quality["errors"]:
        quality_text.extend(f"- {error}" for error in quality["errors"])
    else:
        quality_text.append("- none")
    quality_md.parent.mkdir(parents=True, exist_ok=True)
    quality_md.write_text("\n".join(quality_text) + "\n", encoding="utf-8")


def validate_candidate_pool(
    pool_path: Path = DEFAULT_POOL,
    schema_path: Path = DEFAULT_SCHEMA,
    distribution_path: Path = DEFAULT_DISTRIBUTION,
    core_path: Path = DEFAULT_CORE,
    write_report_files: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cases = load_cases(pool_path)
    schema = schema_info(schema_path)
    expected = expected_counts(distribution_path)
    core_ids = load_core_ids(core_path)
    coverage, quality, errors = build_reports(cases, schema, expected, core_ids)
    if write_report_files:
        write_reports(coverage, quality)
    return coverage, quality, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage 18 DeepSeek candidate pool.")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--no-write-reports", action="store_true")
    args = parser.parse_args()

    coverage, _quality, errors = validate_candidate_pool(
        pool_path=args.pool,
        schema_path=args.schema,
        distribution_path=args.distribution,
        core_path=args.core,
        write_report_files=not args.no_write_reports,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Candidate pool validation passed.")
    print(f"total_count={coverage['total_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
