from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REAL300 = PROJECT_ROOT / "outputs" / "deepseek_real300_plan.json"
DEFAULT_PILOT30 = PROJECT_ROOT / "outputs" / "deepseek_real300_pilot30_plan.json"
REQUIRED_ATTACK_CATEGORIES = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 22 real300 and pilot30 plans.")
    parser.add_argument("--real300", default=str(DEFAULT_REAL300))
    parser.add_argument("--pilot30", default=str(DEFAULT_PILOT30))
    return parser.parse_args()


def validate(real300: dict[str, Any], pilot30: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    check_plan(real300, 300, {"attack": 220, "normal": 40, "hard_normal": 40}, errors, "real300")
    check_plan(pilot30, 30, {"attack": 20, "normal": 5, "hard_normal": 5}, errors, "pilot30")
    for name, plan in (("real300", real300), ("pilot30", pilot30)):
        categories = {
            str(item.get("category"))
            for item in plan.get("cases", [])
            if item.get("sample_type") == "attack"
        }
        missing = sorted(REQUIRED_ATTACK_CATEGORIES - categories)
        if missing:
            errors.append(f"{name} missing attack categories: {missing}")
        for item in plan.get("cases", []):
            if item.get("category") in {"jailbreak", "sensitive_information_leakage"}:
                if not item.get("success_criteria"):
                    errors.append(f"{name} {item.get('case_id')} missing success_criteria")
    return {
        "validation_passed": not errors,
        "errors": errors,
        "real300_case_count": real300.get("case_count"),
        "pilot30_case_count": pilot30.get("case_count"),
        "real300_distribution": real300.get("sample_type_counts"),
        "pilot30_distribution": pilot30.get("sample_type_counts"),
    }


def check_plan(
    plan: dict[str, Any],
    expected_count: int,
    expected_distribution: dict[str, int],
    errors: list[str],
    name: str,
) -> None:
    cases = list(plan.get("cases", []))
    if plan.get("case_count") != expected_count or len(cases) != expected_count:
        errors.append(f"{name} expected {expected_count} cases")
    if plan.get("sample_type_counts") != expected_distribution:
        errors.append(f"{name} distribution mismatch: {plan.get('sample_type_counts')}")
    ids = [str(item.get("case_id")) for item in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        errors.append(f"{name} duplicate case_id: {duplicates}")
    if any(str(item.get("real_batch") or "") == "" for item in cases):
        errors.append(f"{name} contains cases without real_batch")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    real300 = json.loads(resolve(args.real300).read_text(encoding="utf-8"))
    pilot30 = json.loads(resolve(args.pilot30).read_text(encoding="utf-8"))
    result = validate(real300, pilot30)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
