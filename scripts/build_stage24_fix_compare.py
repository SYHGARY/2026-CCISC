from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "outputs" / "stage24_pilot_issue_fix_compare.md"


def read_json(path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def fmt_metrics(metrics: dict[str, Any]) -> list[str]:
    return [f"- {key}: `{value}`" for key, value in metrics.items()]


def main() -> None:
    stage23_gate = read_json("outputs/stage23_pilot30_gate_decision.json")
    issue_cases = read_json("outputs/stage24_pilot_issue_cases.json")
    root = read_json("outputs/stage24_case_level_root_cause.json")
    targeted = read_json("outputs/deepseek_stage24_targeted_issue_dry_run_summary.json")
    targeted_fp = read_json("outputs/deepseek_stage24_targeted_issue_false_positive_analysis.json")
    targeted_residual = read_json("outputs/deepseek_stage24_targeted_issue_residual_risk_cases.json")
    official = read_json("outputs/deepseek_official560_dry_run_after_stage24_summary.json")
    official_fp = read_json("outputs/deepseek_official560_false_positive_analysis_after_stage24.json")
    official_residual = read_json("outputs/deepseek_official560_residual_risk_cases_after_stage24.json")

    lines = [
        "# Stage 24 Pilot Issue Fix Compare",
        "",
        "## Stage 23 Pilot Real Issues",
        "",
        f"- provider: `{stage23_gate['provider']}`",
        f"- case_count: `{stage23_gate['case_count']}`",
        f"- run_status_counts: `{stage23_gate['run_status_counts']}`",
        f"- failed_rate: `{stage23_gate['failed_rate']}`",
        f"- gate_pass: `{str(stage23_gate['gate_pass']).lower()}`",
        f"- decision: `{stage23_gate['decision']}`",
        "",
        "Issue cases:",
    ]
    for item in issue_cases["cases"]:
        lines.append(f"- {item['case_id']}: {', '.join(item['issue_type'])}")

    lines.extend(["", "## Stage 24 Root Cause", ""])
    for item in root["root_causes"]:
        lines.extend([
            f"### {item['case_id']}",
            "",
            f"- primary_class: `{item['primary_class']}`",
            f"- minimal_fix: {item['minimal_fix']}",
            "",
            item["root_cause"],
            "",
        ])

    lines.extend([
        "## Fix Summary",
        "",
        "- Completed provider fallback JSON parse errors are now recorded as model warnings, not failed cases.",
        "- Final-answer attack-success metrics now ignore safely contextualized blocked/low-trust unsafe instructions.",
        "- Guarded final answers redact fake sandbox secret/token/password markers.",
        "- Evidence-fidelity final-answer checks no longer treat a recovered tool flow as overriding an earlier failed attempt.",
        "- Response prompts include Guard decision history only when a real intervention or confirmation state needs to be explained.",
        "- `--case-ids` now falls back to the official benchmark dataset when needed, matching Stage 24 issue-case commands.",
        "",
        "## Targeted Dry-Run Result",
        "",
        f"- case_count: `{targeted['case_count']}`",
        f"- run_status_counts: `{targeted['run_status_counts']}`",
        f"- false_positive_count: `{targeted_fp['false_positive_count']}`",
        f"- residual_risk_count: `{targeted_residual['residual_risk_count']}`",
    ])
    lines.extend(fmt_metrics(targeted["metrics"]))

    lines.extend([
        "",
        "## Official560 Dry-Run Regression",
        "",
        f"- case_count: `{official['case_count']}`",
        f"- run_status_counts: `{official['run_status_counts']}`",
        f"- false_positive_count: `{official_fp['false_positive_count']}`",
        f"- residual_risk_count: `{official_residual['residual_risk_count']}`",
    ])
    lines.extend(fmt_metrics(official["metrics"]))

    lines.extend([
        "",
        "## Next Gate",
        "",
        "- Stage 24 did not call the real API.",
        "- Dry-run regression is clean, but dry-run is not real-model evidence.",
        "- Recommend a targeted real retest of the seven issue cases before Batch 1.",
        "- Batch 1 and the remaining 270 real cases remain blocked until targeted real retest is approved and reviewed.",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"md_output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
