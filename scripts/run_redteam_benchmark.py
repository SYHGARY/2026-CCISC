from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.redteam import run_redteam_benchmark, write_redteam_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the expanded LogicGuard red-team benchmark."
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--catalog",
        default="data/redteam_attack_catalog.json",
    )
    parser.add_argument(
        "--report",
        default="outputs/redteam_benchmark_report.json",
    )
    parser.add_argument(
        "--replay",
        default="outputs/redteam_replay.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_redteam_benchmark(repetitions=args.repetitions)
    write_redteam_outputs(
        report,
        catalog_path=PROJECT_ROOT / args.catalog,
        report_path=PROJECT_ROOT / args.report,
        replay_path=PROJECT_ROOT / args.replay,
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Catalog saved: {PROJECT_ROOT / args.catalog}")
    print(f"Report saved: {PROJECT_ROOT / args.report}")
    print(f"Replay saved: {PROJECT_ROOT / args.replay}")


if __name__ == "__main__":
    main()
