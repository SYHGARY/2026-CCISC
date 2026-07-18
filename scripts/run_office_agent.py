from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.office_agent import OfficeAgentService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LogicGuard LangGraph office agent."
    )
    parser.add_argument("goal", help="User task for the office agent.")
    parser.add_argument(
        "--seed-file",
        action="append",
        default=[],
        metavar="PATH=CONTENT",
        help="Seed a sandbox file. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_files: dict[str, str] = {}
    for item in args.seed_file:
        if "=" not in item:
            raise SystemExit("--seed-file must use PATH=CONTENT")
        path, content = item.split("=", 1)
        seed_files[path] = content
    service = OfficeAgentService()
    try:
        result = service.run_task(args.goal, seed_files=seed_files)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if result.get("awaiting_confirmation"):
            print(
                "\nTask paused for confirmation. Resume through "
                "POST /api/v1/agents/tasks/{task_id}/confirm."
            )
    finally:
        service.close()


if __name__ == "__main__":
    main()
