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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM logic consistency detection and correction experiment.")
    parser.add_argument("--input", default="data/sample_traces.jsonl")
    parser.add_argument("--output", default="outputs/report.json")
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    pipeline = LogicGuardPipeline()
    reports = [pipeline.run(LogicTrace.from_dict(row)).to_dict() for row in read_jsonl(input_path)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(reports)
    inconsistent = sum(1 for report in reports if not report["is_consistent"])
    print(f"Analyzed traces: {total}")
    print(f"Inconsistent traces: {inconsistent}")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
