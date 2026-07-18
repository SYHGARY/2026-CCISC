from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.evaluation import evaluate_reports
from llm_logic_guard.corrector import LogicCorrector
from llm_logic_guard.detector import LogicConsistencyDetector
from llm_logic_guard.models import AnalysisReport, LogicTrace
from llm_logic_guard.pipeline import LogicGuardPipeline


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection against labeled JSONL traces.")
    parser.add_argument("--input", default="data/evaluation_traces.jsonl")
    parser.add_argument("--output", default="outputs/evaluation_report.json")
    parser.add_argument(
        "--mode",
        choices=("legacy", "hybrid"),
        default="legacy",
        help="Legacy reproduces the original four-label benchmark; hybrid evaluates all new detectors.",
    )
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_path = (PROJECT_ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    rows = read_jsonl(input_path)
    if args.mode == "hybrid":
        pipeline = LogicGuardPipeline()
        reports = [pipeline.run(LogicTrace.from_dict(row)) for row in rows]
    else:
        detector = LogicConsistencyDetector()
        corrector = LogicCorrector()
        reports = []
        for row in rows:
            trace = LogicTrace.from_dict(row)
            violations = detector.analyze(trace)
            reports.append(AnalysisReport(
                trace_id=trace.trace_id,
                is_consistent=not violations,
                violations=violations,
                corrections=corrector.propose(trace, violations),
                detector_versions={"legacy": "legacy_baseline.v1"},
            ))
    expected = {
        report.trace_id: set(row.get("expected_violation_types", []))
        for report, row in zip(reports, rows)
    }
    metrics = evaluate_reports(reports, expected)

    payload = {
        "dataset": str(input_path),
        "trace_count": len(rows),
        "mode": args.mode,
        "metrics": metrics.to_dict(),
        "reports": [report.to_dict() for report in reports],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Evaluated traces: {len(rows)}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"F1: {metrics.f1:.3f}")
    print(f"Macro F1: {metrics.macro_f1:.3f}")
    print(f"Trace accuracy: {metrics.trace_accuracy:.3f}")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
