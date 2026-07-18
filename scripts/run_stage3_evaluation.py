from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.stage3_evaluation import (
    generate_stage3_benchmark,
    run_stage3_evaluation,
    write_stage3_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 3 multilingual NLI/Z3 evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="data/stage3_consistency_benchmark.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/stage3_evaluation_report.json",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PROJECT_ROOT / args.dataset
    output = PROJECT_ROOT / args.output
    write_stage3_benchmark(generate_stage3_benchmark(), dataset)
    report = run_stage3_evaluation(
        dataset,
        use_transformer=not args.lexical_only,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = PROJECT_ROOT / "config" / "nli_config.json"
    config.write_text(
        json.dumps({
            "model": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            "backend": report["semantic_backend"],
            "threshold": report["nli_threshold"],
            "validation": report["validation"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Dataset saved: {dataset}")
    print(f"Report saved: {output}")
    print(f"NLI config saved: {config}")


if __name__ == "__main__":
    main()
