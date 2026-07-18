from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
DEFAULT_COVERAGE_JSON = PROJECT_ROOT / "outputs" / "deepseek_dataset_coverage_report.json"
DEFAULT_COVERAGE_MD = PROJECT_ROOT / "outputs" / "deepseek_dataset_coverage_report.md"
DEFAULT_PLAN_JSON = PROJECT_ROOT / "outputs" / "deepseek_real_medium_plan.json"
DEFAULT_PLAN_MD = PROJECT_ROOT / "outputs" / "deepseek_real_medium_plan.md"

REQUIRED_ATTACK_SURFACES = (
    "prompt_injection",
    "jailbreak",
    "tool_hijacking",
    "dangerous_code_execution",
    "file_access_overwrite",
    "memory_poisoning",
    "environment_pollution",
    "multi_agent_error_propagation",
)

PREFERRED_ATTACK_CASES = (
    "atk_prompt_file_email_01",
    "atk_prompt_hidden_markdown",
    "atk_tool_hijack_api_delete",
    "atk_tool_hijack_description",
    "atk_code_exec_secret_read",
    "atk_code_exec_network_exfil",
    "atk_file_overwrite_report",
    "atk_file_secret_read",
    "atk_memory_poison_policy",
    "atk_memory_poison_permission",
    "atk_multi_agent_research_to_action",
    "atk_multi_agent_planner_override",
)

PREFERRED_NORMAL_CASES = (
    "norm_summarize_report",
    "norm_meeting_actions",
    "norm_calculate_total",
    "norm_write_summary",
    "norm_send_notice",
)

PREFERRED_HARD_NORMAL_CASES = (
    "hard_public_email",
    "hard_local_secret_policy",
    "hard_write_incident_note",
    "hard_arithmetic_sandbox",
    "hard_no_tool_policy_answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dataset coverage reports and a stratified DeepSeek real-medium plan."
    )
    parser.add_argument("--cases", default=str(DEFAULT_DATASET))
    parser.add_argument("--coverage-json", default=str(DEFAULT_COVERAGE_JSON))
    parser.add_argument("--coverage-md", default=str(DEFAULT_COVERAGE_MD))
    parser.add_argument("--plan-json", default=str(DEFAULT_PLAN_JSON))
    parser.add_argument("--plan-md", default=str(DEFAULT_PLAN_MD))
    return parser.parse_args()


def load_dataset(path: Path = DEFAULT_DATASET) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "attack": list(data.get("attack", [])),
        "normal": list(data.get("normal", [])),
        "hard_normal": list(data.get("hard_normal", [])),
    }


def flatten_cases(dataset: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_type in ("attack", "normal", "hard_normal"):
        for case in dataset.get(sample_type, []):
            item = dict(case)
            item["dataset"] = sample_type
            item["sample_type"] = str(item.get("sample_type") or sample_type)
            rows.append(item)
    return rows


def build_coverage(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = flatten_cases(dataset)
    sample_type_counts = Counter(item["dataset"] for item in rows)
    category_counts = Counter(str(item.get("category") or item["dataset"]) for item in rows)
    attack_surface_counts: Counter[str] = Counter()
    for item in rows:
        for surface in _surfaces(item):
            attack_surface_counts[surface] += 1

    boundary_counter = Counter(
        str(item.get("boundary_reason") or "missing")
        for item in dataset.get("hard_normal", [])
    )
    hard_normal_with_reason = sum(
        1 for item in dataset.get("hard_normal", []) if str(item.get("boundary_reason") or "").strip()
    )
    return {
        "dataset_path": str(DEFAULT_DATASET.relative_to(PROJECT_ROOT)),
        "total_count": len(rows),
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "attack_surface_counts": dict(sorted(attack_surface_counts.items())),
        "hard_normal_boundary_reason": {
            "total": len(dataset.get("hard_normal", [])),
            "with_reason": hard_normal_with_reason,
            "missing": len(dataset.get("hard_normal", [])) - hard_normal_with_reason,
            "unique_reason_count": len(boundary_counter),
            "reason_counts": dict(boundary_counter),
        },
    }


def build_medium_plan(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    index = {item["id"]: item for item in flatten_cases(dataset)}
    selected_ids = list(PREFERRED_ATTACK_CASES + PREFERRED_NORMAL_CASES + PREFERRED_HARD_NORMAL_CASES)
    missing = [case_id for case_id in selected_ids if case_id not in index]
    if missing:
        raise ValueError(f"medium plan references missing cases: {', '.join(missing)}")

    plan_cases = []
    for position, case_id in enumerate(selected_ids, start=1):
        item = index[case_id]
        plan_cases.append({
            "position": position,
            "case_id": case_id,
            "sample_type": item["dataset"],
            "category": str(item.get("category") or item["dataset"]),
            "attack_surface": _surfaces(item),
            "family": str(item.get("family") or ""),
            "selection_reason": selection_reason(item),
            "expected_focus_metrics": expected_focus_metrics(item),
            "must_run_real": True,
        })

    surface_counts: Counter[str] = Counter()
    for item in plan_cases:
        for surface in item["attack_surface"]:
            surface_counts[surface] += 1
    sample_type_counts = Counter(item["sample_type"] for item in plan_cases)
    missing_surfaces = [
        surface for surface in REQUIRED_ATTACK_SURFACES if surface not in surface_counts
    ]
    if missing_surfaces:
        raise ValueError(f"medium plan misses attack surfaces: {', '.join(missing_surfaces)}")
    if len(plan_cases) > 24:
        raise ValueError("medium plan must not exceed 24 cases")

    return {
        "plan_id": "deepseek-real-medium-stratified-v1",
        "description": (
            "Stratified real-medium candidate set for DeepSeek validation. "
            "This file is a plan only until explicitly run in real mode."
        ),
        "intended_mode": "real",
        "case_count": len(plan_cases),
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "attack_surface_counts": dict(sorted(surface_counts.items())),
        "required_attack_surfaces": list(REQUIRED_ATTACK_SURFACES),
        "missing_required_attack_surfaces": missing_surfaces,
        "real_run_status": "not_run",
        "cases": plan_cases,
    }


def selection_reason(item: dict[str, Any]) -> str:
    sample_type = str(item.get("dataset") or item.get("sample_type") or "")
    surfaces = ", ".join(_surfaces(item))
    if sample_type == "attack":
        return f"Selected to cover attack surface(s): {surfaces}."
    if sample_type == "normal":
        return "Selected as a normal office workflow control for false-positive measurement."
    return "Selected as a hard-normal boundary sample for over-blocking measurement."


def expected_focus_metrics(item: dict[str, Any]) -> list[str]:
    sample_type = str(item.get("dataset") or item.get("sample_type") or "")
    if sample_type == "attack":
        return [
            "attack_success_rate_before_guard",
            "attack_success_rate_after_guard",
            "blocked_attack_count",
            "repair_success_rate",
        ]
    if sample_type == "normal":
        return ["false_positive_rate_on_normal", "task_completion_rate"]
    return ["hard_normal_false_positive_rate", "task_completion_rate"]


def write_reports(
    *,
    coverage: dict[str, Any],
    plan: dict[str, Any],
    coverage_json: Path,
    coverage_md: Path,
    plan_json: Path,
    plan_md: Path,
) -> None:
    write_json(coverage_json, coverage)
    write_json(plan_json, plan)
    coverage_md.parent.mkdir(parents=True, exist_ok=True)
    coverage_md.write_text(render_coverage_md(coverage), encoding="utf-8")
    plan_md.parent.mkdir(parents=True, exist_ok=True)
    plan_md.write_text(render_plan_md(plan), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def render_coverage_md(coverage: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek Dataset Coverage Report",
        "",
        f"- dataset: `{coverage['dataset_path']}`",
        f"- total_count: `{coverage['total_count']}`",
        "",
        "## Sample Types",
        "",
        "| sample_type | count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {key} | {value} |"
        for key, value in coverage["sample_type_counts"].items()
    )
    lines.extend(["", "## Categories", "", "| category | count |", "| --- | ---: |"])
    lines.extend(
        f"| {key} | {value} |"
        for key, value in coverage["category_counts"].items()
    )
    lines.extend(["", "## Attack Surfaces", "", "| attack_surface | count |", "| --- | ---: |"])
    lines.extend(
        f"| {key} | {value} |"
        for key, value in coverage["attack_surface_counts"].items()
    )
    boundary = coverage["hard_normal_boundary_reason"]
    lines.extend([
        "",
        "## Hard-Normal Boundary Reasons",
        "",
        f"- total: `{boundary['total']}`",
        f"- with_reason: `{boundary['with_reason']}`",
        f"- missing: `{boundary['missing']}`",
        f"- unique_reason_count: `{boundary['unique_reason_count']}`",
        "",
        "| boundary_reason | count |",
        "| --- | ---: |",
    ])
    lines.extend(
        f"| {reason} | {count} |"
        for reason, count in boundary["reason_counts"].items()
    )
    lines.append("")
    return "\n".join(lines)


def render_plan_md(plan: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek Real-Medium Stratified Plan",
        "",
        f"- plan_id: `{plan['plan_id']}`",
        f"- intended_mode: `{plan['intended_mode']}`",
        f"- real_run_status: `{plan['real_run_status']}`",
        f"- case_count: `{plan['case_count']}`",
        f"- missing_required_attack_surfaces: `{plan['missing_required_attack_surfaces']}`",
        "",
        "## Sample Type Counts",
        "",
        "| sample_type | count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {key} | {value} |"
        for key, value in plan["sample_type_counts"].items()
    )
    lines.extend(["", "## Attack Surface Counts", "", "| attack_surface | count |", "| --- | ---: |"])
    lines.extend(
        f"| {key} | {value} |"
        for key, value in plan["attack_surface_counts"].items()
    )
    lines.extend([
        "",
        "## Cases",
        "",
        "| # | case_id | sample_type | category | attack_surface | must_run_real | selection_reason |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ])
    for item in plan["cases"]:
        lines.append(
            "| {position} | `{case_id}` | {sample_type} | {category} | {surfaces} | {must_run_real} | {reason} |".format(
                position=item["position"],
                case_id=item["case_id"],
                sample_type=item["sample_type"],
                category=item["category"],
                surfaces=", ".join(item["attack_surface"]),
                must_run_real=str(item["must_run_real"]).lower(),
                reason=item["selection_reason"],
            )
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "This is a plan and dry-run target. It is not a real-medium result until",
        "the same case list is explicitly run with `--mode real` after user approval.",
        "",
    ])
    return "\n".join(lines)


def _surfaces(item: dict[str, Any]) -> list[str]:
    surfaces = item.get("attack_surface") or [item.get("category") or item.get("dataset") or "unknown"]
    return [str(surface) for surface in surfaces]


def main() -> None:
    args = parse_args()
    dataset = load_dataset(resolve_project_path(args.cases))
    coverage = build_coverage(dataset)
    plan = build_medium_plan(dataset)
    write_reports(
        coverage=coverage,
        plan=plan,
        coverage_json=resolve_project_path(args.coverage_json),
        coverage_md=resolve_project_path(args.coverage_md),
        plan_json=resolve_project_path(args.plan_json),
        plan_md=resolve_project_path(args.plan_md),
    )
    print(json.dumps({
        "coverage_json": str(resolve_project_path(args.coverage_json)),
        "coverage_md": str(resolve_project_path(args.coverage_md)),
        "plan_json": str(resolve_project_path(args.plan_json)),
        "plan_md": str(resolve_project_path(args.plan_md)),
        "case_count": plan["case_count"],
        "sample_type_counts": plan["sample_type_counts"],
        "attack_surface_counts": plan["attack_surface_counts"],
    }, ensure_ascii=False, indent=2))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
