from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.probabilistic_training import (
    generate_trajectory_dataset,
    train_calibrate_evaluate,
    write_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and calibrate the LogicGuard DTMC/PAC monitor."
    )
    parser.add_argument(
        "--dataset",
        default="data/pro2guard_trajectories.jsonl",
    )
    parser.add_argument(
        "--model",
        default="outputs/logicguard_dtmc.json",
    )
    parser.add_argument(
        "--report",
        default="outputs/probabilistic_training_report.json",
    )
    parser.add_argument("--repetitions-per-family", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--target-epsilon", type=float, default=0.10)
    parser.add_argument("--maximum-fpr", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = PROJECT_ROOT / args.dataset
    model_path = PROJECT_ROOT / args.model
    report_path = PROJECT_ROOT / args.report
    rows = generate_trajectory_dataset(
        repetitions_per_family=args.repetitions_per_family
    )
    write_dataset(rows, dataset_path)
    report = train_calibrate_evaluate(
        dataset_path=dataset_path,
        model_path=model_path,
        horizon=args.horizon,
        confidence=args.confidence,
        target_epsilon=args.target_epsilon,
        maximum_false_positive_rate=args.maximum_fpr,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Dataset saved: {dataset_path}")
    print(f"Model saved: {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
