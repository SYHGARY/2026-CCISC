from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "deepseek_official560_dry_run.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_official560_residual_risk_cases.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_official560_residual_risk_cases.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List guarded residual-risk attack cases.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--md-output", default=str(DEFAULT_MD_OUTPUT))
    return parser.parse_args()


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    findings = [
        analyze_record(item)
        for item in report.get("records", [])
        if item.get("dataset") == "attack"
        and bool(item.get("guarded", {}).get("attack_success"))
    ]
    return {
        "input_experiment_id": report.get("experiment_id"),
        "mode": report.get("mode"),
        "provider": report.get("provider"),
        "model": report.get("model"),
        "residual_risk_count": len(findings),
        "case_ids": [item["id"] for item in findings],
        "surface_counts": count_surfaces(findings),
        "findings": findings,
    }


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    guarded = record.get("guarded", {})
    actions = [str(item) for item in guarded.get("tool_actions", [])]
    candidates = [
        str(item.get("action", ""))
        for item in guarded.get("candidate_actions", [])
        if item.get("action")
    ]
    return {
        "id": record.get("id"),
        "category": record.get("category", ""),
        "attack_surface": record.get("attack_surface", []),
        "guarded_status": guarded.get("status"),
        "task_completed": guarded.get("task_completed"),
        "intercepted": guarded.get("intercepted"),
        "repaired": guarded.get("repaired"),
        "tool_actions": actions,
        "candidate_actions": candidates,
        "initial_reason": classify_residual(actions, candidates, guarded),
        "attack_success_reason": guarded.get("attack_success_reason", ""),
        "attack_success_evidence": guarded.get("attack_success_evidence", ""),
    }


def classify_residual(actions: list[str], candidates: list[str], guarded: dict[str, Any]) -> str:
    combined = " ".join(actions + candidates).lower()
    if "send_email" in combined:
        return "external_send_or_exfiltration_still_executed"
    if "execute_code" in combined:
        return "unsafe_code_execution_still_succeeded"
    if "read_file" in combined:
        return "sensitive_or_unauthorized_file_read_still_succeeded"
    if "memory_write" in combined:
        return "unsafe_memory_write_still_succeeded"
    reason = str(guarded.get("attack_success_reason") or "")
    if reason.startswith("final_answer_leakage"):
        return "final_answer_sensitive_leakage"
    if reason.startswith("final_answer_policy_violation"):
        return "final_answer_policy_violation"
    if reason.startswith("final_answer_unsafe_instruction"):
        return "final_answer_unsafe_instruction"
    if guarded.get("attack_success"):
        return "metric_marked_attack_success_without_recognized_tool_action"
    return "unknown"


def count_surfaces(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in findings:
        surfaces = item.get("attack_surface") or [item.get("category", "unknown")]
        for surface in surfaces:
            key = str(surface)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek Official560 Residual Risk Cases",
        "",
        f"- experiment_id: `{analysis.get('input_experiment_id')}`",
        f"- mode: `{analysis.get('mode')}`",
        f"- provider: `{analysis.get('provider')}`",
        f"- residual_risk_count: `{analysis.get('residual_risk_count')}`",
        f"- surface_counts: `{analysis.get('surface_counts')}`",
        "",
        "## Findings",
        "",
    ]
    if not analysis.get("findings"):
        lines.append("- none")
    for item in analysis.get("findings", []):
        lines.extend([
            f"### {item.get('id')}",
            "",
            f"- attack_surface: `{item.get('attack_surface')}`",
            f"- initial_reason: `{item.get('initial_reason')}`",
            f"- tool_actions: `{item.get('tool_actions')}`",
            f"- candidate_actions: `{item.get('candidate_actions')}`",
            f"- attack_success_reason: `{item.get('attack_success_reason')}`",
            f"- attack_success_evidence: `{item.get('attack_success_evidence')}`",
            "",
        ])
    return "\n".join(lines)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    report = json.loads(resolve_path(args.input).read_text(encoding="utf-8"))
    analysis = analyze_report(report)

    json_output = resolve_path(args.json_output)
    md_output = resolve_path(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    md_output.write_text(render_markdown(analysis), encoding="utf-8")
    print(json.dumps({
        "json_output": str(json_output),
        "md_output": str(md_output),
        "residual_risk_count": analysis["residual_risk_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
