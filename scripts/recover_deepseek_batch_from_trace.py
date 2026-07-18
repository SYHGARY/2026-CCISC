from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_logic_guard.models import LogicTrace, TraceStep  # noqa: E402
from llm_logic_guard.storage import TraceStore  # noqa: E402
from run_deepseek_batch_experiment import (  # noqa: E402
    DANGEROUS_GUARD_ACTIONS,
    compact_guard_decisions,
    compact_text,
    extract_candidate_actions,
    failed_run_stub,
    is_attack_success,
    load_cases,
    mark_record_status,
    sanitize_for_output,
)

DEFAULT_TRACE_ROOT = PROJECT_ROOT / "outputs" / "deepseek_batch_runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "deepseek_batch_recovered.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a DeepSeek batch report from a checkpoint or trace DBs."
    )
    parser.add_argument("--checkpoint", default="", help="Existing checkpoint JSON to recover.")
    parser.add_argument(
        "--trace-root",
        default="",
        help="Experiment trace directory containing baseline/traces.db and guarded/traces.db.",
    )
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "data" / "deepseek_batch_cases.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--mode", default="real")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="")
    return parser.parse_args()


def recover_from_checkpoint(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["recovered_from_checkpoint"] = True
    report["recovery_source"] = str(path)
    return sanitize_for_output(report)


def recover_from_trace_root(
    trace_root: Path,
    *,
    cases_path: Path,
    mode: str,
    provider: str,
    model: str,
) -> dict[str, Any]:
    experiment_id = trace_root.name
    cases_by_id = {
        case["id"]: {**case, "dataset": dataset}
        for dataset, items in load_cases(cases_path).items()
        for case in items
    }
    pairs = discover_trace_pairs(trace_root, experiment_id)
    records: list[dict[str, Any]] = []
    for _, case_id in sorted(pairs):
        case = cases_by_id.get(case_id)
        if not case:
            continue
        record = {
            "id": case["id"],
            "dataset": case["dataset"],
            "family": case.get("family", ""),
            "goal": case.get("goal", ""),
            "baseline": recover_variant(trace_root, experiment_id, case, "baseline"),
            "guarded": recover_variant(trace_root, experiment_id, case, "guarded"),
        }
        records.append(mark_record_status(record))

    report = {
        "experiment_id": experiment_id,
        "mode": mode,
        "provider": provider,
        "model": model or provider,
        "dataset_filter": "recovered",
        "case_count": len(records),
        "limits": {
            "limit": 0,
            "attack_limit": None,
            "normal_limit": None,
            "hard_normal_limit": None,
        },
        "api_key_configured": False,
        "recovered_from_trace": True,
        "recovery_source": str(trace_root),
        "records": records,
    }
    return sanitize_for_output(report)


def discover_trace_pairs(trace_root: Path, experiment_id: str) -> set[tuple[int, str]]:
    baseline_db = trace_root / "baseline" / "traces.db"
    guarded_db = trace_root / "guarded" / "traces.db"
    baseline = trace_index(baseline_db, experiment_id, "baseline")
    guarded = trace_index(guarded_db, experiment_id, "guarded")
    return set(baseline) & set(guarded)


def trace_index(db_path: Path, experiment_id: str, variant: str) -> set[tuple[int, str]]:
    if not db_path.exists():
        return set()
    pattern = re.compile(
        rf"^office-{re.escape(experiment_id)}-(\d+)-(.+)-{re.escape(variant)}$"
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT trace_id FROM traces").fetchall()
    found: set[tuple[int, str]] = set()
    for (trace_id,) in rows:
        match = pattern.match(str(trace_id))
        if match:
            found.add((int(match.group(1)), match.group(2)))
    return found


def recover_variant(
    trace_root: Path,
    experiment_id: str,
    case: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    db_path = trace_root / variant / "traces.db"
    trace_id = find_trace_id(db_path, experiment_id, case["id"], variant)
    if not trace_id:
        return failed_run_stub("trace not found")

    store = TraceStore(db_path)
    try:
        trace = store.get_trace(trace_id)
        decisions = store.get_decisions(trace_id)
    finally:
        store.close()
    if trace is None:
        return failed_run_stub("trace not found")
    result = public_result_from_trace(trace, decisions)
    return {
        "status": result["status"],
        "task_completed": result["task_completed"],
        "attack_success": is_attack_success(case, result),
        "intercepted": any(
            item.get("action") in DANGEROUS_GUARD_ACTIONS for item in decisions
        ),
        "repaired": any(
            item.get("action") in DANGEROUS_GUARD_ACTIONS for item in decisions
        ),
        "latency_ms": None,
        "error": "",
        "trace_id": trace.trace_id,
        "tool_actions": [str(item.get("action", "")) for item in result["tool_results"]],
        "candidate_actions": extract_candidate_actions(result),
        "guard_decisions": compact_guard_decisions(decisions),
        "model_output_summary": [],
        "final_result": compact_text(result["final_answer"], 500),
        "repair_chain": [],
    }


def find_trace_id(db_path: Path, experiment_id: str, case_id: str, variant: str) -> str:
    if not db_path.exists():
        return ""
    pattern = re.compile(
        rf"^office-{re.escape(experiment_id)}-\d+-{re.escape(case_id)}-{re.escape(variant)}$"
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT trace_id FROM traces").fetchall()
    for (trace_id,) in rows:
        if pattern.match(str(trace_id)):
            return str(trace_id)
    return ""


def public_result_from_trace(
    trace: LogicTrace,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    tool_results: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []
    final_answer = ""
    for step in trace.steps:
        if step.action_name:
            if step.phase == "after_action":
                tool_results.append({
                    "action": step.action_name,
                    "args": step.action_args,
                    "result": step.tool_result or "",
                })
            elif step.phase == "before_action":
                llm_calls.append({
                    "role": step.agent_id,
                    "output": {
                        "action": step.action_name,
                        "args": step.action_args,
                    },
                })
        if step.phase == "final_answer":
            final_answer = step.content
    return {
        "status": "completed" if final_answer else "failed",
        "task_completed": bool(final_answer),
        "tool_results": tool_results,
        "guard_decisions": decisions,
        "llm_calls": llm_calls,
        "repair_chain": [],
        "errors": [],
        "final_answer": final_answer,
    }


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    if args.checkpoint:
        report = recover_from_checkpoint(resolve_path(args.checkpoint))
    elif args.trace_root:
        report = recover_from_trace_root(
            resolve_path(args.trace_root),
            cases_path=resolve_path(args.cases),
            mode=args.mode,
            provider=args.provider,
            model=args.model,
        )
    else:
        raise SystemExit("Provide --checkpoint or --trace-root.")

    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "case_count": report.get("case_count"),
        "recovered_from_checkpoint": report.get("recovered_from_checkpoint", False),
        "recovered_from_trace": report.get("recovered_from_trace", False),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
