from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = PROJECT_ROOT / "outputs" / "demo_stage14_case_selection.json"
DEFAULT_GRAPH = PROJECT_ROOT / "outputs" / "demo_stage14_causal_graph.json"
DEFAULT_REAL_MEDIUM_SUMMARY = PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium_summary.json"
DEFAULT_STAGE13_SUMMARY = (
    PROJECT_ROOT / "outputs" / "deepseek_batch_real_stage13_targeted_retest_summary.json"
)
DEFAULT_VIEWER_DIR = PROJECT_ROOT / "outputs" / "demo_viewer"
DEFAULT_SCRIPT = PROJECT_ROOT / "outputs" / "demo_stage15_presentation_script.md"
DEFAULT_QA = PROJECT_ROOT / "outputs" / "demo_stage15_defense_qa.md"
DEFAULT_PPT = PROJECT_ROOT / "outputs" / "demo_stage15_ppt_outline.md"

PROJECT_TITLE = "LogicGuard Evidence Viewer"
POSITIONING = (
    "LogicGuard is a runtime safety supervision and attack-chain self-healing "
    "defense system for tool-using LLM Agent applications."
)
BENCHMARK_BOUNDARY = (
    "Evidence scope: Stage 11 used a selected 22-case real-medium plan; "
    "Stage 13 used a 2-case targeted real retest. This is not the full "
    "40-case benchmark."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build offline Stage 15 demo viewer and presentation materials."
    )
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument("--real-medium-summary", default=str(DEFAULT_REAL_MEDIUM_SUMMARY))
    parser.add_argument("--stage13-summary", default=str(DEFAULT_STAGE13_SUMMARY))
    parser.add_argument("--viewer-dir", default=str(DEFAULT_VIEWER_DIR))
    parser.add_argument("--presentation-script-output", default=str(DEFAULT_SCRIPT))
    parser.add_argument("--defense-qa-output", default=str(DEFAULT_QA))
    parser.add_argument("--ppt-outline-output", default=str(DEFAULT_PPT))
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_demo_data(
    selection: dict[str, Any],
    graph: dict[str, Any],
    real_medium_summary: dict[str, Any],
    stage13_summary: dict[str, Any],
) -> dict[str, Any]:
    graphs_by_id = {item["case_id"]: item for item in graph.get("graphs", [])}
    cases = []
    for case in selection.get("cases", []):
        cases.append(build_viewer_case(case, graphs_by_id.get(case["case_id"], {})))

    return {
        "stage": "stage15_demo_evidence_viewer",
        "project_title": PROJECT_TITLE,
        "positioning": POSITIONING,
        "real_api_used_to_build_viewer": False,
        "benchmark_boundary": BENCHMARK_BOUNDARY,
        "evidence_scope": {
            "real_medium": "Stage 11 selected 22-case real-medium plan",
            "targeted_retest": "Stage 13 two-case targeted real retest",
            "full_benchmark": "Not a 40-case full-suite benchmark",
        },
        "summary_metrics": {
            "real_medium": metric_summary(real_medium_summary),
            "stage13_targeted_retest": metric_summary(stage13_summary),
        },
        "case_count": len(cases),
        "cases": cases,
        "required_node_types": graph.get("required_node_types", []),
    }


def build_viewer_case(case: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    guarded = case.get("guarded_behavior", {})
    baseline = case.get("baseline_behavior", {})
    decision = first_intervention(guarded) or first_decision(guarded) or {}
    return {
        "case_id": case["case_id"],
        "slot": case.get("slot"),
        "sample_type": case.get("sample_type"),
        "attack_surface": case.get("attack_surface", []),
        "source_report": case.get("source_report"),
        "goal": case.get("goal"),
        "why_suitable": case.get("why_suitable"),
        "display_highlight": case.get("display_highlight"),
        "baseline_behavior": behavior_for_view(baseline),
        "guarded_behavior": behavior_for_view(guarded),
        "guard_decision": {
            "action": decision.get("action", "allow"),
            "alternative_action": decision.get("alternative_action") or "continue",
            "reasons": decision.get("reasons", []),
        },
        "enforcement": decision.get("alternative_action") or "continue",
        "self_healing_result": self_healing_result(case),
        "causal_graph": {
            "nodes": graph.get("nodes", []),
            "edges": graph.get("edges", []),
        },
    }


def behavior_for_view(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run.get("status"),
        "task_completed": bool(run.get("task_completed")),
        "attack_success": bool(run.get("attack_success")),
        "intercepted": bool(run.get("intercepted")),
        "repaired": bool(run.get("repaired")),
        "tool_actions": run.get("tool_actions") or [],
        "guard_decisions": run.get("guard_decisions") or [],
        "repair_chain": run.get("repair_chain") or [],
    }


def self_healing_result(case: dict[str, Any]) -> str:
    guarded = case.get("guarded_behavior", {})
    if guarded.get("repaired"):
        return "Repaired or replanned unsafe trajectory before final completion."
    if case.get("sample_type") == "hard_normal" and not guarded.get("intercepted"):
        return "Allowed safe hard-normal task without false-positive intervention."
    if guarded.get("intercepted"):
        return "Intervened at runtime and completed with a safe guarded outcome."
    return "Completed without repair because no blocking risk was detected."


def first_decision(run: dict[str, Any]) -> dict[str, Any] | None:
    decisions = run.get("guard_decisions") or []
    return decisions[0] if decisions else None


def first_intervention(run: dict[str, Any]) -> dict[str, Any] | None:
    for decision in run.get("guard_decisions") or []:
        if decision.get("action") not in {None, "allow", "audit"}:
            return decision
    return None


def metric_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": summary.get("mode"),
        "provider": summary.get("provider"),
        "model": summary.get("model"),
        "case_count": summary.get("case_count"),
        "dataset_counts": summary.get("dataset_counts"),
        "metrics": summary.get("metrics", {}),
        "failed_cases": summary.get("failed_cases", []),
    }


def write_outputs(
    data: dict[str, Any],
    viewer_dir: Path,
    presentation_script_output: Path,
    defense_qa_output: Path,
    ppt_outline_output: Path,
) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    (viewer_dir / "demo_data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (viewer_dir / "index.html").write_text(render_html(data), encoding="utf-8")
    (viewer_dir / "README.md").write_text(render_viewer_readme(data), encoding="utf-8")
    presentation_script_output.parent.mkdir(parents=True, exist_ok=True)
    presentation_script_output.write_text(render_presentation_script(data), encoding="utf-8")
    defense_qa_output.write_text(render_defense_qa(data), encoding="utf-8")
    ppt_outline_output.write_text(render_ppt_outline(data), encoding="utf-8")


def render_html(data: dict[str, Any]) -> str:
    cases = data["cases"]
    cards = "\n".join(render_case_card(case) for case in cases)
    metrics = render_metric_panels(data)
    embedded = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(data['project_title'])}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --accent: #0f766e;
      --accent-2: #7c2d12;
      --ok: #166534;
      --warn: #92400e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, "Microsoft YaHei", sans-serif;
      line-height: 1.55;
    }}
    header {{
      background: #102a43;
      color: white;
      padding: 28px 32px;
      border-bottom: 4px solid var(--accent);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0 0 10px; }}
    .subtitle {{ max-width: 920px; color: #d9e2ec; font-size: 16px; }}
    .scope {{
      margin-top: 14px;
      display: inline-block;
      padding: 8px 10px;
      border: 1px solid rgba(255,255,255,.35);
      border-radius: 6px;
      color: #f0f4f8;
      font-weight: 700;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 2px rgba(16,42,67,.06);
    }}
    .case-card {{ display: flex; flex-direction: column; gap: 12px; min-height: 100%; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .tag {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: #f8fafc;
    }}
    .highlight {{ color: var(--accent); font-weight: 700; }}
    .behavior {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .mini {{
      border-left: 3px solid var(--accent);
      padding: 8px 10px;
      background: #f8fafc;
      min-width: 0;
    }}
    .mini strong {{ display: block; margin-bottom: 4px; }}
    .decision {{ border-left-color: var(--accent-2); }}
    code, pre {{ font-family: Consolas, "Courier New", monospace; }}
    pre {{
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0b1720;
      color: #e6edf3;
      padding: 12px;
      border-radius: 6px;
      max-height: 380px;
    }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f7; }}
    .graph {{ margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .node, .edge {{ padding: 8px; border: 1px solid var(--line); border-radius: 6px; background: #fbfdff; }}
    .node-type {{ color: var(--accent); font-weight: 700; }}
    .footer-note {{ color: var(--muted); font-size: 13px; margin-top: 22px; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 640px) {{
      header {{ padding: 22px 18px; }}
      main {{ padding: 16px; }}
      .grid, .behavior, .graph {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 26px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(data['project_title'])}</h1>
    <p class="subtitle">{html.escape(data['positioning'])}</p>
    <div class="scope">{html.escape(data['benchmark_boundary'])}</div>
  </header>
  <main>
    <section>
      <h2>Demo Cases</h2>
      <div class="grid">
        {cards}
      </div>
    </section>
    <section>
      <h2>Real-Medium Metrics</h2>
      {metrics}
    </section>
    <section>
      <h2>Raw Demo Data</h2>
      <pre id="raw-data">{embedded}</pre>
      <p class="footer-note">This page is fully static and uses embedded data plus demo_data.json. No network request or model call is required.</p>
    </section>
  </main>
  <script>
    window.LOGICGUARD_DEMO_DATA = {json.dumps(data, ensure_ascii=False, indent=2)};
  </script>
</body>
</html>
"""


def render_case_card(case: dict[str, Any]) -> str:
    baseline = case["baseline_behavior"]
    guarded = case["guarded_behavior"]
    nodes = "\n".join(
        f"""<div class="node"><span class="node-type">{esc(node.get('type'))}</span><br>{esc(node.get('label'))}: {esc(node.get('detail'))}</div>"""
        for node in case["causal_graph"]["nodes"]
    )
    edges = "\n".join(
        f"""<div class="edge"><code>{esc(edge.get('from'))}</code> -> <code>{esc(edge.get('to'))}</code><br>{esc(edge.get('relation'))}</div>"""
        for edge in case["causal_graph"]["edges"]
    )
    surfaces = "".join(f'<span class="tag">{esc(surface)}</span>' for surface in case["attack_surface"])
    reasons = ", ".join(str(reason) for reason in case["guard_decision"].get("reasons", [])[:2])
    return f"""<article class="card case-card" id="{esc(case['case_id'])}">
  <div>
    <h3>{esc(case['case_id'])}</h3>
    <div class="meta">
      <span class="tag">{esc(case['sample_type'])}</span>
      {surfaces}
      <span class="tag">{esc(case['source_report'])}</span>
    </div>
  </div>
  <p class="highlight">{esc(case['display_highlight'])}</p>
  <p>{esc(case['goal'])}</p>
  <div class="behavior">
    <div class="mini">
      <strong>Baseline</strong>
      status: {esc(baseline['status'])}<br>
      attack_success: {esc(baseline['attack_success'])}<br>
      tools: {esc(', '.join(baseline['tool_actions']) or 'none')}
    </div>
    <div class="mini">
      <strong>Guarded</strong>
      status: {esc(guarded['status'])}<br>
      attack_success: {esc(guarded['attack_success'])}<br>
      intercepted: {esc(guarded['intercepted'])}; repaired: {esc(guarded['repaired'])}<br>
      tools: {esc(', '.join(guarded['tool_actions']) or 'none')}
    </div>
  </div>
  <div class="mini decision">
    <strong>Guard Decision / Enforcement / Self-Healing</strong>
    decision: {esc(case['guard_decision']['action'])};
    enforcement: {esc(case['enforcement'])}<br>
    repair result: {esc(case['self_healing_result'])}<br>
    reasons: {esc(reasons or 'none')}
  </div>
  <div class="graph">
    <div><strong>Causal Nodes</strong>{nodes}</div>
    <div><strong>Causal Edges</strong>{edges}</div>
  </div>
</article>"""


def render_metric_panels(data: dict[str, Any]) -> str:
    real_medium = data["summary_metrics"]["real_medium"]
    stage13 = data["summary_metrics"]["stage13_targeted_retest"]
    return "\n".join(
        [
            metric_table("Stage 11 Real-Medium: selected 22 cases", real_medium),
            metric_table("Stage 13 Targeted Retest: 2 cases", stage13),
        ]
    )


def metric_table(title: str, summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    rows = [
        ("mode", summary.get("mode")),
        ("provider", summary.get("provider")),
        ("case_count", summary.get("case_count")),
        ("attack_success_rate_before_guard", metrics.get("attack_success_rate_before_guard")),
        ("attack_success_rate_after_guard", metrics.get("attack_success_rate_after_guard")),
        ("blocked_attack_count", metrics.get("blocked_attack_count")),
        ("false_positive_rate_on_normal", metrics.get("false_positive_rate_on_normal")),
        ("hard_normal_false_positive_rate", metrics.get("hard_normal_false_positive_rate")),
        ("task_completion_rate", metrics.get("task_completion_rate")),
        ("repair_success_rate", metrics.get("repair_success_rate")),
    ]
    body = "\n".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in rows)
    return f"""<div class="panel">
  <h3>{esc(title)}</h3>
  <table><tbody>{body}</tbody></table>
</div>"""


def render_viewer_readme(data: dict[str, Any]) -> str:
    ids = ", ".join(f"`{case['case_id']}`" for case in data["cases"])
    return f"""# Stage 15 Evidence Viewer

Open `index.html` in a browser. The page is static and embeds the same data
written to `demo_data.json`, so it does not need a local server.

- real_api_used_to_build_viewer: `False`
- case_count: `{data['case_count']}`
- demo cases: {ids}
- evidence boundary: {data['benchmark_boundary']}

Files:

- `index.html`: static viewer page.
- `demo_data.json`: machine-readable demo data.
- `README.md`: this note.
"""


def render_presentation_script(data: dict[str, Any]) -> str:
    case_lines = "\n".join(render_case_talk_track(case) for case in data["cases"])
    return f"""# Stage 15 Presentation Script

## 3-Minute Version

1. Opening, 20s: LogicGuard is a runtime safety supervision and attack-chain
   self-healing defense for tool-using LLM Agents. The target problem is not
   only bad final text; it is unsafe actions before tools, files, APIs, code,
   email, or memory produce side effects.
2. System, 45s: show `outputs/demo_viewer/index.html`. Start at the project
   positioning, then point to the benchmark boundary. Explain that the Guard
   checks planner/action candidates, tool calls/results, and final answers,
   then decides `allow/audit/confirm/deny/replan`.
3. Evidence, 80s: click or scroll through the four cards. For prompt injection
   and tool hijacking, emphasize runtime replan and safe-plan enforcement. For
   dangerous code execution, emphasize Stage 13 targeted retest fixed the
   residual metric/provenance issue. For hard-normal arithmetic, emphasize that
   safe sandbox arithmetic now completes without over-blocking.
4. Metrics, 25s: Stage 11 is 22 selected real-medium cases: ASR before and
   after Guard were both `0.083333`, blocked attacks `6`, normal FPR `0.0`,
   hard-normal FPR `0.2`, task completion `0.954545`, repair success
   `0.888889`.
5. Boundary, 10s: Stage 13 is only a two-case targeted real retest, not a new
   22-case benchmark and not the full 40-case suite.

## 5-Minute Version

1. Open the viewer and read the one-sentence positioning.
2. Point to the boundary badge first: 22 real-medium cases plus 2 targeted
   retest cases, not a 40-case full benchmark.
3. Explain the action boundary: the system observes the action candidate before
   trusting a tool, API, file operation, code execution, external send, or
   memory write.
4. Walk through each card in this order: prompt injection, tool hijacking,
   dangerous code execution, hard-normal boundary.
5. In each card, look first at `Baseline`, then `Guarded`, then `Guard Decision
   / Enforcement / Self-Healing`, then the causal nodes/edges.
6. Use the metric panel to separate aggregate evidence from targeted evidence.
   Stage 11 is the selected real-medium aggregate. Stage 13 validates two
   repaired residuals only.
7. Close with limitation: the project has representative real-model evidence
   and a working runtime defense prototype, but it does not claim universal
   jailbreak robustness or production safety.

## 8-Minute Version

1. Problem framing, 60s: tool-using Agents fail at the action layer. Prompt
   filters see text, but they often miss the causal chain from low-trust input
   to a planned tool call, file mutation, API request, code execution, or
   memory update.
2. Architecture, 75s: planner/action candidate -> runtime audit -> Guard
   decision -> enforcement -> final answer check -> trace evidence. Stress
   that causal evidence is used to explain where the unsafe trajectory formed.
3. Demo navigation, 30s: open `outputs/demo_viewer/index.html`. Show title,
   positioning, and benchmark boundary.
4. Case walkthrough, 180s: use the four case tracks below.
5. Metrics, 75s: explain Stage 11 real-medium aggregate and Stage 13 targeted
   retest separately. Do not merge their metrics. ASR is the fraction of attack
   cases where the unsafe objective succeeds; FPR is normal or hard-normal
   behavior incorrectly intercepted as unsafe.
6. Innovation, 45s: runtime action audit, attack-chain causal graph, and
   self-healing/replan are the key differentiators from a prompt-only filter.
7. Limitations and next step, 15s: full 40-case real benchmark and richer UI/PPT
   production remain future work.

## Demo Case Talk Tracks

{case_lines}

## Metric Explanation

- `attack_success_rate_before_guard`: how often the attack objective succeeded
  in baseline execution.
- `attack_success_rate_after_guard`: how often the attack objective still
  succeeded after guarded execution.
- `blocked_attack_count`: attack cases where Guard made a blocking-class
  intervention such as replan, deny, confirm, or inspect.
- `false_positive_rate_on_normal`: normal tasks incorrectly intercepted.
- `hard_normal_false_positive_rate`: boundary-normal tasks incorrectly
  intercepted.
- `task_completion_rate`: whether the useful task still completed.
- `repair_success_rate`: whether replanning or repair produced a safe
  completion path.

## Stage 13 Boundary

Stage 13 ran only `atk_code_exec_secret_read` and `hard_arithmetic_sandbox` as
a targeted real retest. It validates the two Stage 11 residual fixes only. It
must not be reported as a new real-medium aggregate, a 22-case rerun, or a
40-case benchmark.
"""


def render_case_talk_track(case: dict[str, Any]) -> str:
    return f"""### {case['case_id']}

- Say: `{case['sample_type']}` sample, surface `{', '.join(case['attack_surface'])}`.
- Click/look: start with the card title and source report, then compare
  Baseline vs Guarded.
- Emphasize: {case['display_highlight']}
- Decision: Guard action `{case['guard_decision']['action']}` with enforcement
  `{case['enforcement']}`.
- Close: {case['self_healing_result']}
"""


def render_defense_qa(data: dict[str, Any]) -> str:
    return """# Stage 15 Defense Q&A

## 你的创新点是什么？

创新点是把 Agent 安全从“最终回答过滤”前移到运行时动作监督：在工具调用、文件访问、API 调用、代码执行、外部发送、记忆写入之前做审计，并保留攻击链因果证据，支持 replan/deny/inspect/sanitize 等自愈处置。

## 和普通 prompt filter 有什么区别？

普通 prompt filter 主要看输入或输出文本。LogicGuard 看的是完整执行轨迹：来源可信度、动作类型、工具参数、上下文传播、规则违反、概率风险和最终结果。它能指出“哪一步把低可信内容变成了危险动作”，而不只是给一句拒答。

## 为什么需要 runtime guard？

工具型 Agent 的风险经常发生在动作执行前，例如发送邮件、删除文件、调用 API 或执行代码。等最终回答生成后再过滤，副作用可能已经发生。runtime guard 的价值是把检查点放在副作用之前。

## 为什么要做因果图谱？

因果图谱把 user input、planner/action candidate、tool call、risk rule、guard decision、enforcement、final outcome 串起来。答辩时可以清楚解释攻击从哪里进入、在哪个动作边界被发现、为什么触发规则、最后如何修复。

## 为什么要区分 normal / hard-normal / attack？

attack 用来测防护是否有效；normal 用来测是否误伤普通任务；hard-normal 用来测边界正常任务是否被过度保守地拦截。没有 hard-normal，系统很容易为了安全指标好看而牺牲可用性。

## real-medium 为什么只有 22 条？

Stage 11 是一个分层选择的 real-medium 计划，目标是覆盖主要攻击面并控制真实 API 成本和延迟。它是比赛展示级别的 selected benchmark evidence，不是完整 40 条全量 benchmark。

## Stage 13 为什么不是新 benchmark？

Stage 13 只重测两个 Stage 11 residual case：`atk_code_exec_secret_read` 和 `hard_arithmetic_sandbox`。它验证这两个修复是否迁移到真实模型，不代表其余 38 条样本，也不能合并成新的 22 条 aggregate。

## 误报如何处理？

先看 case 级 trace，确认是规则太严、路由错误、环境/provenance 问题，还是指标定义错误。修复时优先加回归测试和窄修复，不降低外部发送、代码执行、文件修改、记忆写入等核心防线。

## 攻击成功率如何计算？

ASR 是攻击样本中攻击目标实际达成的比例。项目同时报告 before Guard 和 after Guard；如果危险动作被拦截、执行失败、或修复后没有实现攻击目标，就不应计为 guarded attack success。

## 系统局限性是什么？

当前证据来自 selected 22-case real-medium 和 2-case targeted retest，不是生产泛化证明。数据集仍需扩充 jailbreak、环境污染和持久记忆隔离样本；Viewer 还只是静态展示；完整 40-case real benchmark 需要单独审批成本和延迟。
"""


def render_ppt_outline(data: dict[str, Any]) -> str:
    return """# Stage 15 PPT Outline

## 1. Title: LogicGuard

Runtime safety supervision and attack-chain self-healing defense for tool-using LLM Agents.

## 2. Project Background And Competition Alignment

Map the competition topic to prompt injection, tool hijacking, code execution, file access, memory poisoning, environment pollution, and behavior monitoring.

## 3. Why Prompt-Only Filtering Is Not Enough

Unsafe behavior often appears as tool actions before final text is produced.

## 4. Attack Surface Coverage

Show dataset families and explain attack / normal / hard-normal split.

## 5. System Architecture

Planner/action candidate, runtime audit, Guard decision, enforcement, final answer check, trace evidence.

## 6. Runtime Audit Flow

Explain action-candidate, action-result, and final-answer checkpoints.

## 7. Guard Decision And Enforcement

Explain allow, audit, confirm, deny, replan, inspect, sanitize.

## 8. Attack-Chain Causal Graph

Use the Stage 14 node chain: user input -> planner/action candidate -> tool call -> risk feature/rule -> guard decision -> enforcement -> final outcome.

## 9. Self-Healing Repair Mechanism

Show how unsafe trajectories are sanitized or replanned while preserving useful task completion.

## 10. Real Experiment Design

Separate dry-run controlled checks, real-small validation, real-medium validation, and targeted retest.

## 11. Stage 11 Real-Medium Evidence

22 selected cases, provider real DeepSeek, normal FPR 0.0, hard-normal FPR 0.2, task completion 0.954545, repair success 0.888889.

## 12. Stage 13 Targeted Retest Boundary

Two repaired residual cases only; not a 22-case rerun and not the 40-case benchmark.

## 13. Four Demo Cases

Prompt injection, tool hijacking, dangerous code execution, and hard-normal arithmetic boundary.

## 14. Innovation, Limitations, And Next Step

Innovation: runtime action supervision, causal evidence, self-healing repair. Limitations: selected evidence, static viewer, incomplete full benchmark. Next step: formal PPT production or frontend demo polish.
"""


def esc(value: Any) -> str:
    return html.escape(str(value))


def main() -> None:
    args = parse_args()
    data = build_demo_data(
        load_json(resolve_path(args.selection)),
        load_json(resolve_path(args.graph)),
        load_json(resolve_path(args.real_medium_summary)),
        load_json(resolve_path(args.stage13_summary)),
    )
    write_outputs(
        data,
        resolve_path(args.viewer_dir),
        resolve_path(args.presentation_script_output),
        resolve_path(args.defense_qa_output),
        resolve_path(args.ppt_outline_output),
    )
    print(
        json.dumps(
            {
                "viewer_dir": str(resolve_path(args.viewer_dir)),
                "index_html": str(resolve_path(args.viewer_dir) / "index.html"),
                "demo_data_json": str(resolve_path(args.viewer_dir) / "demo_data.json"),
                "presentation_script": str(resolve_path(args.presentation_script_output)),
                "defense_qa": str(resolve_path(args.defense_qa_output)),
                "ppt_outline": str(resolve_path(args.ppt_outline_output)),
                "case_count": data["case_count"],
                "case_ids": [case["case_id"] for case in data["cases"]],
                "real_api_used_to_build_viewer": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
