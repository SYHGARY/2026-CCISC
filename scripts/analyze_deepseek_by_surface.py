from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_deepseek_batch_experiment import is_metric_false_positive

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "deepseek_official560_dry_run.json"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_official560_dry_run_by_surface.json"
DEFAULT_MD_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_official560_dry_run_by_surface.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DeepSeek metrics by attack surface.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--md-output", default=str(DEFAULT_MD_OUTPUT))
    return parser.parse_args()


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    surfaces: dict[str, list[dict[str, Any]]] = {}
    for record in report.get("records", []):
        for surface in record_surfaces(record):
            surfaces.setdefault(surface, []).append(record)

    return {
        "input_experiment_id": report.get("experiment_id"),
        "mode": report.get("mode"),
        "provider": report.get("provider"),
        "model": report.get("model"),
        "surface_count": len(surfaces),
        "surfaces": {
            surface: summarize_surface(records)
            for surface, records in sorted(surfaces.items())
        },
    }


def record_surfaces(record: dict[str, Any]) -> list[str]:
    raw = record.get("attack_surface") or [record.get("category") or record.get("dataset")]
    surfaces = raw if isinstance(raw, list) else [raw]
    return [str(item) for item in surfaces if str(item)]


def summarize_surface(records: list[dict[str, Any]]) -> dict[str, Any]:
    attacks = [item for item in records if item.get("dataset") == "attack"]
    controls = [item for item in records if item.get("dataset") in {"normal", "hard_normal"}]
    guarded_runs = [item.get("guarded", {}) for item in records]
    failed = [item for item in records if item.get("failed")]
    residual = [
        sample_record(item)
        for item in attacks
        if bool(item.get("guarded", {}).get("attack_success"))
    ]
    false_positive = [
        sample_record(item)
        for item in controls
        if is_metric_false_positive(item)
    ]
    return {
        "case_count": len(records),
        "attack_count": len(attacks),
        "control_count": len(controls),
        "baseline_asr": rate(item.get("baseline", {}).get("attack_success") for item in attacks),
        "guarded_asr": rate(item.get("guarded", {}).get("attack_success") for item in attacks),
        "blocked_count": sum(bool(item.get("guarded", {}).get("intercepted")) for item in attacks),
        "false_positive_count": len(false_positive),
        "task_completion_rate": rate(run.get("task_completed") for run in guarded_runs),
        "failed_cases": [sample_record(item) for item in failed],
        "residual_risk_samples": residual[:20],
        "false_positive_samples": false_positive[:20],
    }


def sample_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "dataset": record.get("dataset"),
        "category": record.get("category", ""),
        "attack_surface": record.get("attack_surface", []),
    }


def rate(values: Any) -> float | None:
    items = [bool(value) for value in values]
    if not items:
        return None
    return round(sum(items) / len(items), 6)


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek By-Surface Analysis",
        "",
        f"- experiment_id: `{analysis.get('input_experiment_id')}`",
        f"- mode: `{analysis.get('mode')}`",
        f"- provider: `{analysis.get('provider')}`",
        f"- surface_count: `{analysis.get('surface_count')}`",
        "",
        "| attack_surface | case_count | baseline_asr | guarded_asr | blocked | false_positive | completion | failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for surface, item in analysis.get("surfaces", {}).items():
        lines.append(
            f"| {surface} | {item['case_count']} | {item['baseline_asr']} | "
            f"{item['guarded_asr']} | {item['blocked_count']} | "
            f"{item['false_positive_count']} | {item['task_completion_rate']} | "
            f"{len(item['failed_cases'])} |"
        )
    lines.extend(["", "## Residual Risk Samples", ""])
    append_samples(lines, analysis, "residual_risk_samples")
    lines.extend(["", "## False Positive Samples", ""])
    append_samples(lines, analysis, "false_positive_samples")
    lines.append("")
    return "\n".join(lines)


def append_samples(lines: list[str], analysis: dict[str, Any], key: str) -> None:
    any_sample = False
    for surface, item in analysis.get("surfaces", {}).items():
        samples = item.get(key, [])
        if not samples:
            continue
        any_sample = True
        ids = ", ".join(str(sample.get("id")) for sample in samples[:10])
        lines.append(f"- {surface}: {ids}")
    if not any_sample:
        lines.append("- none")


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
        "surface_count": analysis["surface_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
