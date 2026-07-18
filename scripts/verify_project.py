from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(f"\n> {' '.join(command)}")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def read_json(path: str) -> dict:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    python = sys.executable
    run([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run([
        python,
        "scripts/evaluate.py",
        "--input",
        "data/evaluation_traces.jsonl",
        "--output",
        "outputs/evaluation_report.json",
    ])
    run([
        python,
        "scripts/evaluate.py",
        "--input",
        "data/adversarial_benchmark_v1.jsonl",
        "--output",
        "outputs/adversarial_benchmark_v1_report.json",
    ])
    run([
        python,
        "scripts/repair_traces.py",
        "--input",
        "data/sample_traces.jsonl",
        "--output",
        "outputs/repair_report.json",
    ])
    run([
        python,
        "scripts/repair_traces.py",
        "--input",
        "data/converted_security_demo.jsonl",
        "--output",
        "outputs/security_demo_repair_report.json",
    ])

    regression = read_json("outputs/evaluation_report.json")
    adversarial = read_json("outputs/adversarial_benchmark_v1_report.json")
    repair = read_json("outputs/repair_report.json")
    security_repair = read_json("outputs/security_demo_repair_report.json")

    repair_inconsistent = [
        item for item in repair["results"]
        if not item["original_report"]["is_consistent"]
    ]
    repair_resolved = [
        item for item in repair_inconsistent
        if item["repair"]["resolved"]
    ]
    security_resolved = all(
        item["repair"]["resolved"]
        for item in security_repair["results"]
        if not item["original_report"]["is_consistent"]
    )

    checks = {
        "regression_f1_at_least_0_95": regression["metrics"]["f1"] >= 0.95,
        "adversarial_macro_f1_at_least_0_90": adversarial["metrics"]["macro_f1"] >= 0.90,
        "sample_auto_repair_resolves_all_repairable_cases": (
            len(repair_inconsistent) == len(repair_resolved)
        ),
        "converted_security_demo_resolved": security_resolved,
    }
    summary = {
        "passed": all(checks.values()),
        "checks": checks,
        "test_command": f"{python} -m unittest discover -s tests -v",
        "regression_metrics": regression["metrics"],
        "adversarial_metrics": adversarial["metrics"],
        "sample_repair": {
            "inconsistent": len(repair_inconsistent),
            "resolved": len(repair_resolved),
        },
        "security_demo_resolved": security_resolved,
    }
    output_path = PROJECT_ROOT / "outputs" / "verification_summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nVerification summary: {output_path}")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
