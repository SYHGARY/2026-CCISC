from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.office_agent import OfficeAgentService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the normal-office LangGraph task suite."
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--approve-confirmations",
        action="store_true",
        help="Approve paused sandbox actions. Default is deny.",
    )
    parser.add_argument(
        "--output",
        default="outputs/office_task_suite_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = json.loads(
        (PROJECT_ROOT / "data" / "office_task_suite.json").read_text(
            encoding="utf-8"
        )
    )[: max(args.limit, 0)]
    service = OfficeAgentService()
    results: list[dict[str, Any]] = []
    try:
        for task in tasks:
            started = time.perf_counter()
            result = service.run_task(
                task["goal"],
                task_id=f"suite-{task['id']}",
                seed_files=task.get("seed_files", {}),
                context={"suite_id": task["id"], "category": task["category"]},
            )
            if result.get("awaiting_confirmation"):
                result = service.resume_task(
                    result["task_id"],
                    approved=args.approve_confirmations,
                    note="office suite policy",
                )
            actual_tools = [item["action"] for item in result.get("tool_results", [])]
            expected_tools = task.get("expected_tools", [])
            results.append({
                "id": task["id"],
                "category": task["category"],
                "status": result.get("status"),
                "provider": result.get("provider"),
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "tool_match": all(tool in actual_tools for tool in expected_tools),
                "errors": result.get("errors", []),
                "latency_ms": (time.perf_counter() - started) * 1000,
            })
    finally:
        service.close()

    report = {
        "task_count": len(results),
        "completed": sum(item["status"] == "completed" for item in results),
        "denied": sum(item["status"] == "denied" for item in results),
        "tool_match_rate": (
            sum(item["tool_match"] for item in results) / len(results)
            if results else 0.0
        ),
        "results": results,
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
