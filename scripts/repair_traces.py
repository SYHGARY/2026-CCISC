from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.models import LogicTrace
from llm_logic_guard.pipeline import LogicGuardPipeline
from llm_logic_guard.repair import LogicRepairer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect, repair, and re-check LogicTrace JSONL.")
    parser.add_argument("--input", default="data/sample_traces.jsonl")
    parser.add_argument("--output", default="outputs/repair_report.json")
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    pipeline = LogicGuardPipeline()
    repairer = LogicRepairer()
    results = []
    for row in read_jsonl(input_path):
        trace = LogicTrace.from_dict(row)
        report = pipeline.run(trace)
        repair = repairer.apply(trace, report)
        results.append({
            "original_report": report.to_dict(),
            "repair": repair.to_dict(),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    inconsistent = sum(1 for result in results if not result["original_report"]["is_consistent"])
    resolved = sum(
        1
        for result in results
        if not result["original_report"]["is_consistent"] and result["repair"]["resolved"]
    )
    print(f"Analyzed traces: {len(results)}")
    print(f"Inconsistent traces: {inconsistent}")
    print(f"Automatically resolved: {resolved}")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
