from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
DEFAULT_REAL_MEDIUM = PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium.json"
DEFAULT_STAGE13 = PROJECT_ROOT / "outputs" / "deepseek_batch_real_stage13_targeted_retest.json"
DEFAULT_REAL_MEDIUM_SUMMARY = PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium_summary.json"
DEFAULT_STAGE13_SUMMARY = PROJECT_ROOT / "outputs" / "deepseek_batch_real_stage13_targeted_retest_summary.json"
DEFAULT_SELECTION_JSON = PROJECT_ROOT / "outputs" / "demo_stage14_case_selection.json"
DEFAULT_SELECTION_MD = PROJECT_ROOT / "outputs" / "demo_stage14_case_selection.md"
DEFAULT_GRAPH_JSON = PROJECT_ROOT / "outputs" / "demo_stage14_causal_graph.json"
DEFAULT_GRAPH_MD = PROJECT_ROOT / "outputs" / "demo_stage14_causal_graph.md"
DEFAULT_EVIDENCE_MD = PROJECT_ROOT / "outputs" / "demo_stage14_evidence_pack.md"

PREFERRED_CASE_IDS = {
    "prompt_or_jailbreak": ["atk_prompt_file_email_01", "atk_prompt_hidden_markdown"],
    "tool_hijacking": ["atk_tool_hijack_api_delete", "atk_tool_hijack_description"],
    "dangerous_code_execution": ["atk_code_exec_secret_read"],
    "hard_normal_boundary": ["hard_arithmetic_sandbox"],
}

DEMO_STAGE = "stage14_demo_evidence_pack"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline demo case selection and causal graph evidence."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--real-medium", default=str(DEFAULT_REAL_MEDIUM))
    parser.add_argument("--stage13", default=str(DEFAULT_STAGE13))
    parser.add_argument("--real-medium-summary", default=str(DEFAULT_REAL_MEDIUM_SUMMARY))
    parser.add_argument("--stage13-summary", default=str(DEFAULT_STAGE13_SUMMARY))
    parser.add_argument("--selection-json-output", default=str(DEFAULT_SELECTION_JSON))
    parser.add_argument("--selection-md-output", default=str(DEFAULT_SELECTION_MD))
    parser.add_argument("--graph-json-output", default=str(DEFAULT_GRAPH_JSON))
    parser.add_argument("--graph-md-output", default=str(DEFAULT_GRAPH_MD))
    parser.add_argument("--evidence-md-output", default=str(DEFAULT_EVIDENCE_MD))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_case_metadata(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    indexed: dict[str, dict[str, Any]] = {}
    for dataset, rows in data.items():
        for case in rows:
            indexed[str(case["id"])] = {**case, "dataset": dataset}
    return indexed


def index_records(report: dict[str, Any], source_label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in report.get("records", []):
        case_id = str(record.get("id", ""))
        if not case_id:
            continue
        indexed[case_id] = {**record, "source_report": source_label}
    return indexed


def build_demo_outputs(
    metadata: dict[str, dict[str, Any]],
    real_medium_report: dict[str, Any],
    stage13_report: dict[str, Any],
    real_medium_summary: dict[str, Any] | None = None,
    stage13_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = index_records(real_medium_report, "stage11_real_medium")
    records.update(index_records(stage13_report, "stage13_targeted_real_retest"))

    selected = select_demo_cases(metadata, records)
    causal_graph = build_causal_graph(selected)
    evidence = build_evidence_pack(selected, causal_graph, real_medium_summary, stage13_summary)
    return {
        "selection": selected,
        "causal_graph": causal_graph,
        "evidence_markdown": evidence,
    }


def select_demo_cases(
    metadata: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for slot, preferred_ids in PREFERRED_CASE_IDS.items():
        case_id = find_case_for_slot(slot, preferred_ids, metadata, records)
        record = records[case_id]
        meta = metadata.get(case_id, {})
        cases.append(build_case_summary(slot, meta, record))

    return {
        "stage": DEMO_STAGE,
        "real_api_used": False,
        "selection_count": len(cases),
        "selection_policy": {
            "prompt_or_jailbreak": "Prefer prompt-injection or jailbreak real-medium record.",
            "tool_hijacking": "Prefer tool-hijacking real-medium record.",
            "dangerous_code_execution": "Prefer atk_code_exec_secret_read from Stage 13 targeted retest.",
            "hard_normal_boundary": "Prefer hard_arithmetic_sandbox from Stage 13 targeted retest.",
        },
        "cases": cases,
    }


def find_case_for_slot(
    slot: str,
    preferred_ids: list[str],
    metadata: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> str:
    for case_id in preferred_ids:
        if case_id in records:
            return case_id

    for case_id, meta in metadata.items():
        if case_id not in records:
            continue
        surfaces = set(as_list(meta.get("attack_surface")))
        family = str(meta.get("family", ""))
        dataset = str(meta.get("dataset", ""))
        if slot == "prompt_or_jailbreak" and (
            "prompt_injection" in surfaces or "jailbreak" in surfaces
        ):
            return case_id
        if slot == "tool_hijacking" and ("tool_hijacking" in surfaces or family == "tool_hijacking"):
            return case_id
        if slot == "dangerous_code_execution" and "dangerous_code_execution" in surfaces:
            return case_id
        if slot == "hard_normal_boundary" and dataset == "hard_normal":
            return case_id
    raise ValueError(f"No record found for demo slot: {slot}")


def build_case_summary(slot: str, meta: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    case_id = str(record["id"])
    baseline = record.get("baseline", {})
    guarded = record.get("guarded", {})
    source_report = str(record.get("source_report", "unknown"))
    return {
        "slot": slot,
        "case_id": case_id,
        "sample_type": str(meta.get("sample_type") or record.get("dataset", "")),
        "family": str(meta.get("family") or record.get("family", "")),
        "attack_surface": as_list(meta.get("attack_surface") or [record.get("family", "")]),
        "source_report": source_report,
        "goal": str(record.get("goal") or meta.get("goal", "")),
        "why_suitable": why_suitable(slot, case_id, source_report),
        "baseline_behavior": summarize_run(baseline),
        "guarded_behavior": summarize_run(guarded),
        "intercepted": bool(guarded.get("intercepted")),
        "repaired": bool(guarded.get("repaired")),
        "display_highlight": display_highlight(slot, guarded, source_report),
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run.get("status"),
        "task_completed": bool(run.get("task_completed")),
        "attack_success": bool(run.get("attack_success")),
        "intercepted": bool(run.get("intercepted")),
        "repaired": bool(run.get("repaired")),
        "tool_actions": as_list(run.get("tool_actions")),
        "candidate_actions": compact_actions(as_list(run.get("candidate_actions"))),
        "guard_decisions": compact_decisions(as_list(run.get("guard_decisions"))),
        "repair_chain": compact_repairs(as_list(run.get("repair_chain"))),
        "trace_id": run.get("trace_id"),
    }


def compact_actions(actions: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    compacted = []
    for action in actions[:limit]:
        if not isinstance(action, dict):
            compacted.append({"action": str(action)})
            continue
        compacted.append({
            "role": action.get("role"),
            "action": action.get("action"),
            "args": action.get("args"),
        })
    return compacted


def compact_decisions(decisions: list[Any], limit: int = 8) -> list[dict[str, Any]]:
    compacted = []
    for decision in decisions[:limit]:
        if not isinstance(decision, dict):
            continue
        compacted.append({
            "action": decision.get("action"),
            "alternative_action": decision.get("alternative_action"),
            "reasons": as_list(decision.get("reasons"))[:3],
        })
    return compacted


def compact_repairs(repairs: list[Any], limit: int = 4) -> list[dict[str, Any]]:
    compacted = []
    for repair in repairs[:limit]:
        if not isinstance(repair, dict):
            continue
        compacted.append({
            "type": repair.get("type"),
            "reason": repair.get("reason"),
            "recheck_consistent": repair.get("recheck_consistent"),
            "discarded_instruction_count": repair.get("discarded_instruction_count"),
        })
    return compacted


def build_causal_graph(selection: dict[str, Any]) -> dict[str, Any]:
    graphs = []
    for case in selection.get("cases", []):
        graphs.append(build_case_graph(case))
    return {
        "stage": DEMO_STAGE,
        "real_api_used": False,
        "graph_count": len(graphs),
        "required_node_types": [
            "user_input",
            "planner_action_candidate",
            "tool_call",
            "guard_decision",
            "risk_feature_rule",
            "enforcement",
            "final_outcome",
        ],
        "graphs": graphs,
    }


def build_case_graph(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    guarded = case["guarded_behavior"]
    first_action = first_candidate_action(guarded)
    first_tool = ", ".join(guarded.get("tool_actions") or []) or "no tool call"
    decision = first_intervention(guarded) or first_decision(guarded)
    decision_action = str(decision.get("action") or "allow") if decision else "allow"
    alternative = str(decision.get("alternative_action") or "continue")
    rule_label = "; ".join(decision.get("reasons") or []) if decision else "No blocking rule fired"
    outcome = outcome_label(case)

    nodes = [
        node(case_id, "user_input", "User input", case.get("goal", "")),
        node(case_id, "planner_action_candidate", "Planner/action candidate", first_action),
        node(case_id, "tool_call", "Tool call", first_tool),
        node(case_id, "risk_feature_rule", "Risk feature/rule", rule_label),
        node(case_id, "guard_decision", "Guard decision", decision_action),
        node(case_id, "enforcement", "Enforcement", alternative),
        node(case_id, "final_outcome", "Final outcome", outcome),
    ]
    edges = [
        edge(case_id, "user_input", "planner_action_candidate", "User goal and context shape the proposed action."),
        edge(case_id, "planner_action_candidate", "tool_call", "The candidate becomes an auditable tool call or planned action."),
        edge(case_id, "tool_call", "risk_feature_rule", "Runtime metadata, provenance, and action type expose risk features."),
        edge(case_id, "risk_feature_rule", "guard_decision", "Rules and risk features produce allow/audit/replan/deny decisions."),
        edge(case_id, "guard_decision", "enforcement", "The decision selects enforcement such as continue, replan, inspect, or sanitize."),
        edge(case_id, "enforcement", "final_outcome", "The enforced path determines safe completion or blocked unsafe behavior."),
    ]
    return {
        "case_id": case_id,
        "slot": case.get("slot"),
        "source_report": case.get("source_report"),
        "nodes": nodes,
        "edges": edges,
    }


def node(case_id: str, node_type: str, label: str, detail: str) -> dict[str, str]:
    return {
        "id": f"{case_id}:{node_type}",
        "type": node_type,
        "label": label,
        "detail": str(detail),
    }


def edge(case_id: str, source: str, target: str, relation: str) -> dict[str, str]:
    return {
        "from": f"{case_id}:{source}",
        "to": f"{case_id}:{target}",
        "relation": relation,
    }


def first_candidate_action(run: dict[str, Any]) -> str:
    candidates = run.get("candidate_actions") or []
    if not candidates:
        return "no explicit candidate recorded"
    action = candidates[0]
    name = action.get("action") or "unknown"
    args = action.get("args")
    return f"{name} {json.dumps(args, ensure_ascii=False)}"


def first_decision(run: dict[str, Any]) -> dict[str, Any] | None:
    decisions = run.get("guard_decisions") or []
    return decisions[0] if decisions else None


def first_intervention(run: dict[str, Any]) -> dict[str, Any] | None:
    for decision in run.get("guard_decisions") or []:
        if decision.get("action") not in {None, "allow", "audit"}:
            return decision
    return None


def outcome_label(case: dict[str, Any]) -> str:
    guarded = case["guarded_behavior"]
    if guarded.get("attack_success"):
        return "unsafe outcome remained"
    if guarded.get("task_completed"):
        if case.get("sample_type") == "hard_normal" and not case.get("intercepted"):
            return "safe normal completion without false positive"
        return "safe task completion"
    return "unsafe action blocked before completion"


def why_suitable(slot: str, case_id: str, source_report: str) -> str:
    reasons = {
        "prompt_or_jailbreak": "Shows low-trust content instructions being separated from useful facts.",
        "tool_hijacking": "Shows poisoned tool/API output being audited before it can redirect the agent.",
        "dangerous_code_execution": "Shows the dangerous-code boundary and the Stage 13 metric/provenance fix on a real retest.",
        "hard_normal_boundary": "Shows that a safe arithmetic sandbox task can complete without over-blocking after Stage 13.",
    }
    return f"{reasons[slot]} Evidence source: {source_report}; case: {case_id}."


def display_highlight(slot: str, guarded: dict[str, Any], source_report: str) -> str:
    if slot == "prompt_or_jailbreak":
        return "Guard sanitizes untrusted instructions while preserving task facts."
    if slot == "tool_hijacking":
        return "Guard turns poisoned tool behavior into a replan/safe-plan decision."
    if slot == "dangerous_code_execution":
        return "Stage 13 real retest no longer reports guarded attack success for the code-exec case."
    if slot == "hard_normal_boundary":
        return "Stage 13 real retest completes safe arithmetic without a hard-normal false positive."
    return f"Evidence from {source_report}."


def build_evidence_pack(
    selection: dict[str, Any],
    causal_graph: dict[str, Any],
    real_medium_summary: dict[str, Any] | None,
    stage13_summary: dict[str, Any] | None,
) -> str:
    lines = [
        "# Stage 14 Demo Evidence Pack",
        "",
        "## One-Sentence Positioning",
        "",
        "LogicGuard is a runtime safety supervision and attack-chain self-healing defense system for tool-using LLM Agent applications.",
        "",
        "## Competition Alignment",
        "",
        "- Attack surface coverage: prompt injection, tool hijacking, dangerous code execution, file access, memory, environment, normal, and hard-normal cases.",
        "- Runtime audit: planner/action-candidate, tool-call/result, and final-answer phases are checked before unsafe effects are trusted.",
        "- Risk judgment: provenance, trust, tool type, DSL rules, semantic checks, and probabilistic signals feed `allow/audit/confirm/deny/replan` decisions.",
        "- Self-healing: unsafe trajectories are sanitized, replanned, or denied; repaired traces are preserved for root-cause explanation.",
        "- Real LLM evidence: Stage 11 used a 22-case real-medium DeepSeek plan; Stage 13 used a 2-case targeted real retest for the two repaired residuals.",
        "",
        "## System Flow",
        "",
        "```mermaid",
        "flowchart LR",
        "  A[User task and context] --> B[Planner/action candidate]",
        "  B --> C[Runtime audit]",
        "  C --> D{Guard decision}",
        "  D -->|allow/audit| E[Tool execution]",
        "  D -->|replan/deny/confirm| F[Self-healing enforcement]",
        "  F --> G[Safe alternative or blocked action]",
        "  E --> H[Final answer check]",
        "  G --> H",
        "  H --> I[Trace evidence and demo report]",
        "```",
        "",
        "## Four Demo Samples",
        "",
    ]
    for case in selection.get("cases", []):
        lines.extend(render_case_flow(case))

    lines.extend([
        "## Real-Medium Metrics",
        "",
        "Stage 11 real-medium was a selected 22-case stratified plan, not the full 40-case dataset.",
        "",
    ])
    lines.extend(render_metrics(real_medium_summary))
    lines.extend([
        "",
        "## Stage 13 Targeted Retest",
        "",
        "Stage 13 ran only `atk_code_exec_secret_read` and `hard_arithmetic_sandbox` with the real DeepSeek provider. It was not a 22-case real-medium rerun.",
        "",
    ])
    lines.extend(render_metrics(stage13_summary))
    lines.extend([
        "",
        "## Single Attack-Chain Causal Graph",
        "",
    ])
    lines.extend(render_mermaid_for_graph(causal_graph["graphs"][0]))
    lines.extend([
        "",
        "## Experiment Loop",
        "",
        "```mermaid",
        "flowchart LR",
        "  A[Dry-run controlled checks] --> B[Real-small validation]",
        "  B --> C[Real-medium 22-case plan]",
        "  C --> D[Case-level root cause]",
        "  D --> E[Stage 13 targeted real retest]",
        "  E --> F[Demo evidence pack and causal graph]",
        "```",
        "",
        "## Presentation Order",
        "",
        "1. Start from attack surfaces.",
        "2. Show runtime audit at the action boundary.",
        "3. Show the causal graph from user input to enforcement.",
        "4. Show self-healing or safe completion.",
        "5. Close with real-model evidence and strict scope boundaries.",
        "",
        "## Boundaries",
        "",
        "- This pack uses Stage 11 real-medium evidence plus Stage 13 two-case targeted retest evidence.",
        "- It is not a full 40-case real suite.",
        "- Stage 13 must not be reported as a new 22-case aggregate benchmark.",
        "- No real API was called to build Stage 14 materials.",
        "",
    ])
    return "\n".join(lines)


def render_case_flow(case: dict[str, Any]) -> list[str]:
    guarded = case["guarded_behavior"]
    baseline = case["baseline_behavior"]
    decisions = guarded.get("guard_decisions") or []
    first_guard = first_intervention(guarded) or first_decision(guarded) or {}
    return [
        f"### {case['case_id']}",
        "",
        f"- slot: `{case['slot']}`",
        f"- sample_type: `{case['sample_type']}`",
        f"- attack_surface: `{', '.join(case['attack_surface'])}`",
        f"- source_report: `{case['source_report']}`",
        f"- why suitable: {case['why_suitable']}",
        f"- baseline behavior: status `{baseline.get('status')}`, attack_success `{baseline.get('attack_success')}`, tools `{', '.join(baseline.get('tool_actions') or []) or 'none'}`",
        f"- guarded behavior: status `{guarded.get('status')}`, attack_success `{guarded.get('attack_success')}`, intercepted `{guarded.get('intercepted')}`, repaired `{guarded.get('repaired')}`, tools `{', '.join(guarded.get('tool_actions') or []) or 'none'}`",
        f"- first guard decision: `{first_guard.get('action', 'none')}` / alternative `{first_guard.get('alternative_action', 'none')}`",
        f"- display highlight: {case['display_highlight']}",
        "",
    ]


def render_metrics(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["- metrics unavailable"]
    metrics = summary.get("metrics", {})
    return [
        f"- mode: `{summary.get('mode')}`",
        f"- provider: `{summary.get('provider')}`",
        f"- case_count: `{summary.get('case_count')}`",
        f"- attack_success_rate_before_guard: `{metrics.get('attack_success_rate_before_guard')}`",
        f"- attack_success_rate_after_guard: `{metrics.get('attack_success_rate_after_guard')}`",
        f"- blocked_attack_count: `{metrics.get('blocked_attack_count')}`",
        f"- false_positive_rate_on_normal: `{metrics.get('false_positive_rate_on_normal')}`",
        f"- hard_normal_false_positive_rate: `{metrics.get('hard_normal_false_positive_rate')}`",
        f"- task_completion_rate: `{metrics.get('task_completion_rate')}`",
        f"- repair_success_rate: `{metrics.get('repair_success_rate')}`",
    ]


def render_selection_markdown(selection: dict[str, Any]) -> str:
    lines = [
        "# Stage 14 Demo Case Selection",
        "",
        f"- real_api_used_to_build_pack: `{selection.get('real_api_used')}`",
        f"- selection_count: `{selection.get('selection_count')}`",
        "",
        "| slot | case_id | sample_type | attack_surface | source | intercepted | repaired | display highlight |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in selection.get("cases", []):
        lines.append(
            "| {slot} | `{case_id}` | `{sample_type}` | `{surface}` | `{source}` | `{intercepted}` | `{repaired}` | {highlight} |".format(
                slot=case["slot"],
                case_id=case["case_id"],
                sample_type=case["sample_type"],
                surface=", ".join(case["attack_surface"]),
                source=case["source_report"],
                intercepted=case["intercepted"],
                repaired=case["repaired"],
                highlight=escape_table(case["display_highlight"]),
            )
        )
    lines.append("")
    for case in selection.get("cases", []):
        lines.extend(render_case_flow(case))
    return "\n".join(lines)


def render_graph_markdown(causal_graph: dict[str, Any]) -> str:
    lines = [
        "# Stage 14 Demo Causal Graph",
        "",
        f"- real_api_used_to_build_graph: `{causal_graph.get('real_api_used')}`",
        f"- graph_count: `{causal_graph.get('graph_count')}`",
        "",
        "## Required Node Types",
        "",
    ]
    for node_type in causal_graph.get("required_node_types", []):
        lines.append(f"- `{node_type}`")
    lines.extend(["", "## Mermaid Overview", ""])
    lines.extend(render_mermaid_for_graph(causal_graph["graphs"][0]))
    for graph in causal_graph.get("graphs", []):
        lines.extend(["", f"## {graph['case_id']}", ""])
        lines.append("| node | type | detail |")
        lines.append("| --- | --- | --- |")
        for item in graph.get("nodes", []):
            lines.append(f"| `{item['id']}` | `{item['type']}` | {escape_table(item['detail'])} |")
        lines.extend(["", "| from | to | relation |", "| --- | --- | --- |"])
        for item in graph.get("edges", []):
            lines.append(f"| `{item['from']}` | `{item['to']}` | {escape_table(item['relation'])} |")
    lines.append("")
    return "\n".join(lines)


def render_mermaid_for_graph(graph: dict[str, Any]) -> list[str]:
    case_id = graph["case_id"]
    return [
        "```mermaid",
        "flowchart LR",
        f"  A[\"{case_id}: user input\"] --> B[\"planner/action candidate\"]",
        "  B --> C[\"tool call\"]",
        "  C --> D[\"risk feature/rule\"]",
        "  D --> E[\"guard decision\"]",
        "  E --> F[\"enforcement\"]",
        "  F --> G[\"final outcome\"]",
        "```",
    ]


def write_outputs(
    selection: dict[str, Any],
    causal_graph: dict[str, Any],
    evidence_markdown: str,
    selection_json_output: Path,
    selection_md_output: Path,
    graph_json_output: Path,
    graph_md_output: Path,
    evidence_md_output: Path,
) -> None:
    for path in [
        selection_json_output,
        selection_md_output,
        graph_json_output,
        graph_md_output,
        evidence_md_output,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    selection_json_output.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selection_md_output.write_text(render_selection_markdown(selection), encoding="utf-8")
    graph_json_output.write_text(
        json.dumps(causal_graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    graph_md_output.write_text(render_graph_markdown(causal_graph), encoding="utf-8")
    evidence_md_output.write_text(evidence_markdown, encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> None:
    args = parse_args()
    metadata = load_case_metadata(resolve_path(args.cases))
    real_medium_report = load_json(resolve_path(args.real_medium))
    stage13_report = load_json(resolve_path(args.stage13))
    real_medium_summary = load_json(resolve_path(args.real_medium_summary))
    stage13_summary = load_json(resolve_path(args.stage13_summary))
    outputs = build_demo_outputs(
        metadata,
        real_medium_report,
        stage13_report,
        real_medium_summary,
        stage13_summary,
    )
    write_outputs(
        outputs["selection"],
        outputs["causal_graph"],
        outputs["evidence_markdown"],
        resolve_path(args.selection_json_output),
        resolve_path(args.selection_md_output),
        resolve_path(args.graph_json_output),
        resolve_path(args.graph_md_output),
        resolve_path(args.evidence_md_output),
    )
    print(json.dumps({
        "selection_json_output": str(resolve_path(args.selection_json_output)),
        "selection_md_output": str(resolve_path(args.selection_md_output)),
        "graph_json_output": str(resolve_path(args.graph_json_output)),
        "graph_md_output": str(resolve_path(args.graph_md_output)),
        "evidence_md_output": str(resolve_path(args.evidence_md_output)),
        "selection_count": outputs["selection"]["selection_count"],
        "graph_count": outputs["causal_graph"]["graph_count"],
        "real_api_used": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
