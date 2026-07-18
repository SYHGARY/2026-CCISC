from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SMALL_SUMMARY = PROJECT_ROOT / "outputs" / "deepseek_batch_real_small_after_norm_summary_fix_summary.json"
DEFAULT_MEDIUM_SUMMARY = PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium_summary.json"
DEFAULT_SMALL_REPORT = PROJECT_ROOT / "outputs" / "deepseek_batch_real_small_after_norm_summary_fix.json"
DEFAULT_MEDIUM_REPORT = PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium.json"
DEFAULT_CASES = PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_real_small_vs_medium_compare.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real-small and real-medium DeepSeek summaries."
    )
    parser.add_argument("--small-summary", default=str(DEFAULT_SMALL_SUMMARY))
    parser.add_argument("--medium-summary", default=str(DEFAULT_MEDIUM_SUMMARY))
    parser.add_argument("--small-report", default=str(DEFAULT_SMALL_REPORT))
    parser.add_argument("--medium-report", default=str(DEFAULT_MEDIUM_REPORT))
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--md-output", default=str(DEFAULT_MD_OUTPUT))
    return parser.parse_args()


def load_case_metadata(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    indexed: dict[str, dict[str, Any]] = {}
    for dataset, rows in data.items():
        for case in rows:
            indexed[str(case["id"])] = {**case, "dataset": dataset}
    return indexed


def coverage(report: dict[str, Any], metadata: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in report.get("records", []):
        case = metadata.get(str(record.get("id")), {})
        for surface in case.get("attack_surface") or [record.get("dataset", "unknown")]:
            counts[str(surface)] = counts.get(str(surface), 0) + 1
    return dict(sorted(counts.items()))


def render_compare(
    small_summary: dict[str, Any],
    medium_summary: dict[str, Any],
    small_coverage: dict[str, int],
    medium_coverage: dict[str, int],
) -> str:
    metric_keys = [
        "attack_success_rate_before_guard",
        "attack_success_rate_after_guard",
        "false_positive_rate_on_normal",
        "hard_normal_false_positive_rate",
        "task_completion_rate",
        "repair_success_rate",
        "average_latency_ms",
    ]
    lines = [
        "# DeepSeek Real Small vs Real Medium Compare",
        "",
        "| metric | real-small | real-medium | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    small_metrics = small_summary.get("metrics", {})
    medium_metrics = medium_summary.get("metrics", {})
    for key in metric_keys:
        small_value = small_metrics.get(key)
        medium_value = medium_metrics.get(key)
        lines.append(
            f"| {key} | {format_metric(small_value)} | {format_metric(medium_value)} | {format_delta(small_value, medium_value)} |"
        )
    lines.extend([
        "",
        "## Failed Cases",
        "",
        f"- real-small: {format_failed(small_summary.get('failed_cases', []))}",
        f"- real-medium: {format_failed(medium_summary.get('failed_cases', []))}",
        "",
        "## Attack Surface Coverage",
        "",
        "| attack_surface | real-small cases | real-medium cases |",
        "| --- | ---: | ---: |",
    ])
    for surface in sorted(set(small_coverage) | set(medium_coverage)):
        lines.append(
            f"| {surface} | {small_coverage.get(surface, 0)} | {medium_coverage.get(surface, 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def format_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


def format_delta(small_value: Any, medium_value: Any) -> str:
    if small_value is None or medium_value is None:
        return "n/a"
    try:
        return str(round(float(medium_value) - float(small_value), 6))
    except (TypeError, ValueError):
        return "n/a"


def format_failed(failed: list[dict[str, Any]]) -> str:
    if not failed:
        return "none"
    return ", ".join(str(item.get("id")) for item in failed)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    small_summary = json.loads(resolve_path(args.small_summary).read_text(encoding="utf-8"))
    medium_summary = json.loads(resolve_path(args.medium_summary).read_text(encoding="utf-8"))
    small_report = json.loads(resolve_path(args.small_report).read_text(encoding="utf-8"))
    medium_report = json.loads(resolve_path(args.medium_report).read_text(encoding="utf-8"))
    metadata = load_case_metadata(resolve_path(args.cases))

    md_output = resolve_path(args.md_output)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.write_text(
        render_compare(
            small_summary,
            medium_summary,
            coverage(small_report, metadata),
            coverage(medium_report, metadata),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"md_output": str(md_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
