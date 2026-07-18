const scenarioSelect = document.querySelector("#scenario-select");
const scenarioMeta = document.querySelector("#scenario-meta");
const guardToggle = document.querySelector("#guard-toggle");
const runButton = document.querySelector("#run-button");
const compareButton = document.querySelector("#compare-button");
const requestStatus = document.querySelector("#request-status");
const emptyState = document.querySelector("#empty-state");
const workspace = document.querySelector("#workspace");
const comparison = document.querySelector("#comparison");

let scenarios = [];

async function initialize() {
  try {
    const [scenarioResponse, metricsResponse] = await Promise.all([
      fetch("/api/v1/scenarios"),
      fetch("/api/v1/metrics"),
    ]);
    scenarios = (await scenarioResponse.json()).scenarios || [];
    renderScenarioOptions();
    renderMetrics(await metricsResponse.json());
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderScenarioOptions() {
  scenarioSelect.replaceChildren();
  scenarios.forEach((scenario) => {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.attack_type;
    scenarioSelect.append(option);
  });
  renderScenarioMeta();
}

function renderScenarioMeta() {
  const scenario = scenarios.find((item) => item.id === scenarioSelect.value);
  scenarioMeta.replaceChildren();
  if (!scenario) return;
  scenarioMeta.append(
    textElement("p", "", scenario.goal),
    textElement("code", "", scenario.candidate_action),
  );
}

function renderMetrics(report) {
  const metrics = report.metrics || {};
  setText("#metric-tasks", metrics.task_count ?? 90);
  setText("#metric-open", percent(metrics.attack_success_rate_without_guard ?? 1));
  setText("#metric-guard", percent(metrics.attack_success_rate_with_guard ?? 0));
  setText("#metric-completion", percent(metrics.normal_completion_with_guard ?? 1));
  setText("#metric-latency", metrics.average_latency_ms ? `${metrics.average_latency_ms.toFixed(1)} ms` : "-");
}

async function runScenario(defended = guardToggle.checked) {
  setBusy(true);
  try {
    const response = await fetch("/api/v1/attacks/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({scenario_id: scenarioSelect.value, defended}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || result.error || "运行失败");
    renderResult(result);
    return result;
  } catch (error) {
    setStatus(error.message, true);
    return null;
  } finally {
    setBusy(false);
  }
}

async function compareScenario() {
  const open = await runScenario(false);
  if (!open) return;
  const guarded = await runScenario(true);
  if (!guarded) return;
  comparison.hidden = false;
  document.querySelector("#compare-open").textContent = open.attack_succeeded ? "攻击成功" : "任务正常";
  document.querySelector("#compare-open").className = open.attack_succeeded ? "danger" : "safe";
  document.querySelector("#compare-guard").textContent = guarded.attack_succeeded ? "攻击成功" : "已拦截";
  document.querySelector("#compare-guard").className = guarded.attack_succeeded ? "danger" : "safe";
}

function renderResult(result) {
  emptyState.hidden = true;
  workspace.hidden = false;
  comparison.hidden = true;
  setStatus("");
  setText("#trace-id", result.trace_id);
  const trace = result.details?.trace || {};
  renderTimeline(trace.steps || []);
  const decisions = result.details?.decisions || [];
  const decision = decisions.at(-1) || {};
  renderDecision(result, decision);
  renderRisk(decision.risk || {});
  renderViolations(decision.violations || []);
}

function renderTimeline(steps) {
  const timeline = document.querySelector("#timeline");
  timeline.replaceChildren();
  steps.forEach((step, index) => {
    const item = element("article", "timeline-step");
    const header = element("div", "step-header");
    header.append(
      textElement("span", "step-index", String(index + 1).padStart(2, "0")),
      textElement("strong", "", phaseLabel(step.role)),
      textElement("span", "source-tag", `${step.source || "agent"} · ${Math.round((step.trust || 0) * 100)}%`),
    );
    item.append(header);
    if (step.content) item.append(textElement("p", "step-body", step.content));
    if (step.action_name) {
      item.append(textElement("pre", "step-code", `${step.action_name}\n${JSON.stringify(step.action_args || {}, null, 2)}`));
    }
    timeline.append(item);
  });
}

function renderDecision(result, decision) {
  const badge = document.querySelector("#decision-badge");
  badge.textContent = (result.decision || "allow").toUpperCase();
  badge.className = `decision-${result.decision || "allow"}`;
  setText("#execution-state", result.details?.executed ? "工具已执行" : "工具未执行");
  const list = document.querySelector("#decision-reasons");
  list.replaceChildren();
  (decision.reasons || ["未记录异常"]).forEach((reason) => {
    list.append(textElement("li", "", reason));
  });
}

function renderRisk(risk) {
  const upper = Number(risk.pac_upper || 0);
  const threshold = Number(risk.threshold || 0.55);
  setText("#risk-value", percent(upper));
  document.querySelector("#risk-fill").style.width = percent(upper);
  document.querySelector("#risk-threshold").style.left = percent(threshold);
  const features = document.querySelector("#risk-features");
  features.replaceChildren();
  Object.entries(risk.features || {}).forEach(([key, value]) => {
    const node = textElement("span", value === true ? "active" : "", `${key}: ${value}`);
    features.append(node);
  });
}

function renderViolations(violations) {
  const container = document.querySelector("#violations");
  container.replaceChildren();
  if (!violations.length) {
    container.append(textElement("p", "no-violations", "未发现确定性违规"));
    return;
  }
  violations.forEach((violation) => {
    const item = element("article", "violation-item");
    item.append(
      textElement("strong", "", violation.spec_id || violation.violation_type),
      textElement("p", "", violation.message),
    );
    (violation.evidence || []).forEach((line) => item.append(textElement("code", "", line)));
    container.append(item);
  });
}

function phaseLabel(role) {
  const labels = {
    environment_observation: "环境观察",
    memory_write: "记忆写入",
    before_action: "动作前",
    after_action: "动作后",
    final_answer: "最终回答",
  };
  return labels[role] || role || "事件";
}

function percent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function setBusy(busy) {
  runButton.disabled = busy;
  compareButton.disabled = busy;
  requestStatus.textContent = busy ? "正在运行..." : requestStatus.textContent;
}

function setStatus(message, isError = false) {
  requestStatus.textContent = message;
  requestStatus.classList.toggle("error", isError);
}

function setText(selector, value) {
  document.querySelector(selector).textContent = String(value);
}

function element(tag, className = "") {
  const node = document.createElement(tag);
  node.className = className;
  return node;
}

function textElement(tag, className, text) {
  const node = element(tag, className);
  node.textContent = text;
  return node;
}

scenarioSelect.addEventListener("change", renderScenarioMeta);
runButton.addEventListener("click", () => runScenario());
compareButton.addEventListener("click", compareScenario);
initialize();
