from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_deepseek_batch_experiment import (
    collect_provider_warning_classifications,
    collect_reporting_label_classifications,
    is_metric_false_positive,
    is_provider_warning,
    provider_warning_affects_completion,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "deepseek_batch_dry_run.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_batch_summary.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_batch_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize DeepSeek batch experiment baseline-vs-guard metrics."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--md-output", default=str(DEFAULT_MD_OUTPUT))
    return parser.parse_args()


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    records = list(report.get("records", []))
    attacks = [item for item in records if item.get("dataset") == "attack"]
    normal = [item for item in records if item.get("dataset") == "normal"]
    hard_normal = [item for item in records if item.get("dataset") == "hard_normal"]
    guarded_runs = [item.get("guarded", {}) for item in records]
    all_runs = [item.get(side, {}) for item in records for side in ("baseline", "guarded")]

    failed_cases = []
    for item in records:
        for side in ("baseline", "guarded"):
            run = item.get(side, {})
            if (
                (item.get("failed") and side == "baseline" and run.get("error"))
                or run.get("status") == "failed"
                or provider_warning_affects_completion(run.get("error", ""))
                or non_fallback_errors(run.get("error", ""))
            ):
                failed_cases.append({
                    "id": item.get("id"),
                    "dataset": item.get("dataset"),
                    "side": side,
                    "status": run.get("status"),
                    "error": run.get("error", ""),
                    "provider_failure": provider_warning_affects_completion(run.get("error", "")),
                })

    summary = {
        "experiment_id": report.get("experiment_id"),
        "mode": report.get("mode"),
        "provider": report.get("provider"),
        "model": report.get("model"),
        "case_count": len(records),
        "run_status_counts": report.get("run_status_counts", {}),
        "dataset_counts": {
            "attack": len(attacks),
            "normal": len(normal),
            "hard_normal": len(hard_normal),
        },
        "metrics": {
            "attack_success_rate_before_guard": rate(
                item.get("baseline", {}).get("attack_success") for item in attacks
            ),
            "attack_success_rate_after_guard": rate(
                item.get("guarded", {}).get("attack_success") for item in attacks
            ),
            "blocked_attack_count": sum(
                bool(item.get("guarded", {}).get("intercepted")) for item in attacks
            ),
            "false_positive_rate_on_normal": rate(
                is_metric_false_positive(item) for item in normal
            ),
            "hard_normal_false_positive_rate": rate(
                is_metric_false_positive(item) for item in hard_normal
            ),
            "task_completion_rate": rate(run.get("task_completed") for run in guarded_runs),
            "repair_success_rate": repair_success_rate(attacks),
            "average_latency_ms": average(run.get("latency_ms") for run in all_runs),
        },
        "provider_warnings": collect_provider_warning_classifications(report),
        "reporting_labels": collect_reporting_label_classifications(report),
        "failed_cases": failed_cases,
    }
    return summary


def non_fallback_errors(error_text: str) -> list[str]:
    errors = [item.strip() for item in str(error_text or "").split(";") if item.strip()]
    return [item for item in errors if not is_provider_warning(item)]


def is_provider_fallback_warning(error_text: str) -> bool:
    text = str(error_text or "").lower()
    return "provider fallback:" in text and "json" in text


def repair_success_rate(attacks: list[dict[str, Any]]) -> float | None:
    repaired = [
        item for item in attacks
        if bool(item.get("guarded", {}).get("repaired"))
        or bool(item.get("guarded", {}).get("intercepted"))
    ]
    if not repaired:
        return None
    success = [
        item for item in repaired
        if not bool(item.get("guarded", {}).get("attack_success"))
    ]
    return round(len(success) / len(repaired), 6)


def rate(values: Any) -> float | None:
    items = [bool(value) for value in values]
    if not items:
        return None
    return round(sum(items) / len(items), 6)


def average(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return round(sum(items) / len(items), 3)


def render_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# DeepSeek Batch Experiment Summary",
        "",
        f"- experiment_id: `{summary.get('experiment_id')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- provider: `{summary.get('provider')}`",
        f"- model: `{summary.get('model')}`",
        f"- case_count: `{summary.get('case_count')}`",
        f"- run_status_counts: `{summary.get('run_status_counts')}`",
        f"- dataset_counts: `{summary.get('dataset_counts')}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: `{value}`")
    labels = summary.get("reporting_labels", {})
    lines.extend([
        "",
        "## Reporting Labels",
        "",
        f"- report_only: `{labels.get('report_only')}`",
        f"- affects_gate: `{labels.get('affects_gate')}`",
        f"- affects_fpr: `{labels.get('affects_fpr')}`",
        f"- label_counts: `{labels.get('label_counts', {})}`",
    ])
    lines.extend(["", "## Failed Cases", ""])
    failed = summary.get("failed_cases", [])
    if not failed:
        lines.append("- none")
    else:
        for item in failed:
            lines.append(
                f"- {item.get('id')} [{item.get('side')}]: "
                f"{item.get('status')} {item.get('error')}"
            )
    lines.append("")
    return "\n".join(lines)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    report = json.loads(input_path.read_text(encoding="utf-8"))
    summary = summarize_report(report)

    json_output = resolve_path(args.json_output)
    md_output = resolve_path(args.md_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_output.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({
        "input": str(input_path),
        "json_output": str(json_output),
        "md_output": str(md_output),
        "metrics": summary["metrics"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
