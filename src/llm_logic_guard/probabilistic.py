"""Probabilistic consistency monitor (DTMC + confidence-margin risk estimate).

Honest scope of the guarantee (read before citing this as a "PAC bound"):

The monitor abstracts each step into an 8-feature boolean state, estimates a
Laplace-smoothed DTMC over those states, and computes the K-step probability of
reaching the unsafe terminal ``__INCONSISTENT__`` (:func:`_reach_unsafe`). To that
probability it adds a per-state Hoeffding confidence margin ``epsilon`` derived
from that state's out-transition sample count.

This ``pac_upper`` value is a *risk score with a single-state confidence
correction*, NOT a proven PAC upper bound on the true K-step reachability. Three
limitations are deliberate simplifications and must not be overstated:

1. Only the *start* state's estimation error is added. ``_reach_unsafe`` is a
   multilinear function of the transition rows of every state on every K-step
   path; the error of those downstream rows is not propagated.
2. Hoeffding bounds the mean of ``n`` i.i.d. bounded samples. It is applied here
   to a product-of-estimates (the reachability polynomial), for which it is not
   a valid confidence bound.
3. Laplace smoothing (``alpha``) biases the estimate; the margin does not correct
   for it. Benign states carry a small residual risk purely from smoothing mass
   leaking to the unsafe terminal.

A rigorous bound would either (a) treat "reached unsafe within K from s" as a
single Bernoulli and Hoeffding-bound that episode-level frequency directly, or
(b) union-bound a multinomial concentration (e.g. Weissman) over all edges on all
K-step paths and propagate it through the recursion. The current code does
neither, so we report the quantity as a calibrated *risk estimate* and rely on
the calibrated decision threshold — not on the margin being a formal guarantee.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .models import LogicTrace, RiskAssessment, TraceStep


UNSAFE = "__INCONSISTENT__"
SAFE_TERMINAL = "__SAFE_TERMINAL__"
TERMINALS = {UNSAFE, SAFE_TERMINAL}
ABSTRACT_STATE_FEATURES = (
    "action_consistent",
    "evidence_supported",
    "goal_preserved",
    "instruction_polluted",
    "irreversible",
    "memory_trusted",
    "permission_granted",
    "sensitive_flow",
)


@dataclass
class DTMCModel:
    transitions: dict[str, dict[str, float]] = field(default_factory=dict)
    transition_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    valid_transitions: dict[str, list[str]] = field(default_factory=dict)
    state_counts: dict[str, int] = field(default_factory=dict)
    state_epsilons: dict[str, float] = field(default_factory=dict)
    risk_cache: dict[str, float] = field(default_factory=dict)
    horizon: int = 4
    alpha: float = 0.25
    pac_epsilon: float = 1.0
    confidence: float = 0.95
    # Meaningful target error for the sample-size sufficiency check. A ~0.4 margin
    # is near-vacuous (it certifies almost nothing), so we require a tighter 0.10.
    # This affects only ``required_state_visits`` / ``pac_sufficient`` reporting;
    # it does NOT change ``threshold``, ``risk_cache`` or any ``pac_upper`` score.
    target_epsilon: float = 0.10
    required_state_visits: int = 0
    pac_sufficient: bool = False
    threshold: float = 0.55
    calibration: dict[str, Any] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    version: str = "logicguard-dtmc.v2"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DTMCModel":
        transitions = {
            str(source): {
                str(target): float(probability)
                for target, probability in targets.items()
            }
            for source, targets in data.get("transitions", {}).items()
        }
        return cls(
            transitions=transitions,
            transition_counts={
                str(source): {
                    str(target): int(count)
                    for target, count in targets.items()
                }
                for source, targets in data.get("transition_counts", {}).items()
            },
            valid_transitions={
                str(source): [str(target) for target in targets]
                for source, targets in data.get("valid_transitions", {}).items()
            },
            state_counts={
                str(state): int(count)
                for state, count in data.get("state_counts", {}).items()
            },
            state_epsilons={
                str(state): float(value)
                for state, value in data.get("state_epsilons", {}).items()
            },
            risk_cache={
                str(state): float(value)
                for state, value in data.get("risk_cache", {}).items()
            },
            horizon=int(data.get("horizon", 4)),
            alpha=float(data.get("alpha", 0.25)),
            pac_epsilon=float(data.get("pac_epsilon", 1.0)),
            confidence=float(data.get("confidence", 0.95)),
            target_epsilon=float(data.get("target_epsilon", 0.10)),
            required_state_visits=int(data.get("required_state_visits", 0)),
            pac_sufficient=bool(data.get("pac_sufficient", False)),
            threshold=float(data.get("threshold", 0.55)),
            calibration=dict(data.get("calibration") or {}),
            training_summary=dict(data.get("training_summary") or {}),
            version=str(data.get("version", "logicguard-dtmc.v1")),
        )


class ConsistencyRiskMonitor:
    def __init__(
        self,
        model_path: Path | None = None,
        *,
        horizon: int = 4,
        threshold: float | None = None,
        minimum_state_count: int = 2,
        confidence: float = 0.95,
        alpha: float = 0.25,
        target_epsilon: float | None = None,
    ) -> None:
        self.model_path = model_path
        self.horizon = horizon
        self.threshold = 0.55 if threshold is None else threshold
        self.minimum_state_count = minimum_state_count
        self.confidence = confidence
        self.alpha = alpha
        self.target_epsilon = 0.10 if target_epsilon is None else target_epsilon
        self.model: DTMCModel | None = None
        if model_path and model_path.exists():
            self.model = DTMCModel.from_dict(
                json.loads(model_path.read_text(encoding="utf-8"))
            )
            self.horizon = self.model.horizon
            # Only inherit the saved target_epsilon when the caller did not
            # request one explicitly, so a retrain can tighten a stale value.
            if target_epsilon is None:
                self.target_epsilon = self.model.target_epsilon
            if threshold is None:
                self.threshold = self.model.threshold

    def train(self, episodes: Iterable[tuple[LogicTrace, bool]]) -> DTMCModel:
        rows = list(episodes)
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        state_counts: Counter[str] = Counter()
        observed_states: set[str] = set()
        unsafe_episodes = 0
        transition_total = 0

        for trace, unsafe in rows:
            states = _deduplicate_adjacent(
                [abstract_state(step, trace) for step in trace.steps]
            )
            if not states:
                continue
            unsafe_episodes += int(unsafe)
            observed_states.update(states)
            states.append(UNSAFE if unsafe else SAFE_TERMINAL)
            for source, target in zip(states, states[1:]):
                counts[source][target] += 1
                state_counts[source] += 1
                transition_total += 1

        valid_transitions = _infer_valid_transitions(observed_states, counts)
        transitions: dict[str, dict[str, float]] = {}
        for source, valid_targets in valid_transitions.items():
            denominator = state_counts[source] + self.alpha * len(valid_targets)
            if denominator == 0:
                continue
            transitions[source] = {
                target: (counts[source][target] + self.alpha) / denominator
                for target in valid_targets
            }

        state_epsilons = {
            state: _state_pac_epsilon(
                count,
                state_count=max(len(observed_states), 1),
                confidence=self.confidence,
            )
            for state, count in state_counts.items()
        }
        global_epsilon = max(state_epsilons.values(), default=1.0)
        required_state_visits = _required_state_visits(
            state_count=max(len(observed_states), 1),
            confidence=self.confidence,
            epsilon=self.target_epsilon,
        )
        pac_sufficient = bool(state_counts) and all(
            count >= required_state_visits
            for count in state_counts.values()
        )
        self.model = DTMCModel(
            transitions=transitions,
            transition_counts={
                source: dict(targets) for source, targets in counts.items()
            },
            valid_transitions={
                source: sorted(targets)
                for source, targets in valid_transitions.items()
            },
            state_counts=dict(state_counts),
            state_epsilons=state_epsilons,
            horizon=self.horizon,
            alpha=self.alpha,
            pac_epsilon=global_epsilon,
            confidence=self.confidence,
            target_epsilon=self.target_epsilon,
            required_state_visits=required_state_visits,
            pac_sufficient=pac_sufficient,
            threshold=self.threshold,
            training_summary={
                "episode_count": len(rows),
                "unsafe_episode_count": unsafe_episodes,
                "safe_episode_count": len(rows) - unsafe_episodes,
                "state_count": len(observed_states),
                "transition_count": transition_total,
                "minimum_state_visits": min(state_counts.values(), default=0),
                "required_state_visits": required_state_visits,
                "target_epsilon": self.target_epsilon,
                "confidence": self.confidence,
                "pac_sufficient": pac_sufficient,
            },
        )
        self.model.risk_cache = {
            state: self._reach_unsafe(state)
            for state in observed_states
        }
        self._save()
        return self.model

    def calibrate(
        self,
        episodes: Iterable[tuple[LogicTrace, bool]],
        *,
        maximum_false_positive_rate: float = 0.10,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("train or load a DTMC model before calibration")
        rows = list(episodes)
        scored = [
            (self.assess(trace).pac_upper, unsafe)
            for trace, unsafe in rows
            if trace.steps
        ]
        observed_scores = sorted(
            {round(score, 12) for score, _ in scored}
        )
        midpoints = {
            round((left + right) / 2.0, 12)
            for left, right in zip(observed_scores, observed_scores[1:])
        }
        candidates = sorted(
            set(observed_scores) | midpoints | {0.0, 1.0},
            reverse=True,
        )
        evaluations = [
            _classification_metrics(scored, threshold)
            for threshold in candidates
        ]
        feasible = [
            item
            for item in evaluations
            if item["false_positive_rate"] <= maximum_false_positive_rate
        ]
        pool = feasible or evaluations
        selected = max(
            pool,
            key=lambda item: (
                item["f1"],
                item["recall"],
                -item["false_positive_rate"],
                -item["threshold"],
            ),
        )
        self.threshold = float(selected["threshold"])
        self.model.threshold = self.threshold
        self.model.calibration = {
            **selected,
            "sample_count": len(scored),
            "maximum_false_positive_rate": maximum_false_positive_rate,
            "constraint_satisfied": bool(feasible),
        }
        self._save()
        return dict(self.model.calibration)

    def evaluate(
        self,
        episodes: Iterable[tuple[LogicTrace, bool]],
    ) -> dict[str, Any]:
        rows = [
            (self.assess(trace).pac_upper, unsafe)
            for trace, unsafe in episodes
            if trace.steps
        ]
        metrics = _classification_metrics(rows, self.threshold)
        return {
            **metrics,
            "sample_count": len(rows),
            "average_risk": (
                sum(score for score, _ in rows) / len(rows) if rows else 0.0
            ),
            "average_attack_risk": _mean(
                score for score, unsafe in rows if unsafe
            ),
            "average_normal_risk": _mean(
                score for score, unsafe in rows if not unsafe
            ),
        }

    def assess(
        self,
        trace: LogicTrace,
        candidate: TraceStep | None = None,
    ) -> RiskAssessment:
        step = candidate or (
            trace.steps[-1]
            if trace.steps
            else TraceStep("empty", "user_input")
        )
        features = state_features(step, trace)
        state = abstract_state(step, trace, features=features)
        count = self.model.state_counts.get(state, 0) if self.model else 0
        used_fallback = self.model is None or count < self.minimum_state_count

        probability = self._reach_unsafe(state)
        if used_fallback:
            probability = max(probability, _heuristic_risk(features))

        epsilon = self._epsilon_for(state)
        # NOTE: on the fallback path ``probability`` may come from the hand-weighted
        # heuristic rather than the DTMC, so ``epsilon`` (a sample-count confidence
        # term) is not a statistical bound on it — it acts only as a conservative
        # safety margin before comparison with the calibrated threshold.
        pac_upper = min(probability + epsilon, 1.0)
        warning = pac_upper >= self.threshold
        source = "稀疏状态回退" if used_fallback else "DTMC"
        reason = (
            f"{source}：K={self.horizon} 步违规可达概率 {probability:.3f}，"
            f"叠加单状态置信裕度后风险估计 {pac_upper:.3f}，阈值 {self.threshold:.3f}"
            f"（该裕度为单状态 Hoeffding 修正，非多步可达的严格 PAC 上界）。"
        )
        return RiskAssessment(
            state=state,
            probability=probability,
            pac_upper=pac_upper,
            threshold=self.threshold,
            horizon=self.horizon,
            warning=warning,
            state_count=count,
            features=features,
            reason=reason,
            pac_epsilon=epsilon,
            model_version=self.model.version if self.model else "heuristic-fallback",
            calibrated=bool(self.model and self.model.calibration),
            fallback=used_fallback,
        )

    def _reach_unsafe(self, start: str) -> float:
        if not self.model or start not in self.model.transitions:
            return 0.0
        cached = self.model.risk_cache.get(start)
        if cached is not None:
            return cached
        frontier = {start: 1.0}
        unsafe_probability = 0.0
        for _ in range(self.horizon):
            next_frontier: dict[str, float] = defaultdict(float)
            for source, mass in frontier.items():
                for target, probability in self.model.transitions.get(
                    source, {}
                ).items():
                    if target == UNSAFE:
                        unsafe_probability += mass * probability
                    elif target != SAFE_TERMINAL:
                        next_frontier[target] += mass * probability
            frontier = next_frontier
            if not frontier:
                break
        return min(unsafe_probability, 1.0)

    def _epsilon_for(self, state: str) -> float:
        """Per-state Hoeffding confidence margin for the risk estimate.

        This is the estimation-error term for the *start* state's transition row
        only; it is NOT a bound on the multi-step reachability. See the module
        docstring for why ``probability + epsilon`` is a risk estimate, not a PAC
        upper bound.
        """
        if not self.model:
            return 0.12
        return self.model.state_epsilons.get(
            state,
            self.model.pac_epsilon,
        )

    def _save(self) -> None:
        if not self.model_path or not self.model:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.write_text(
            json.dumps(self.model.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def state_features(
    step: TraceStep,
    trace: LogicTrace,
) -> dict[str, bool | str | float]:
    text = f"{step.content} {step.tool_result or ''} {step.action_args}".lower()
    goal = trace.user_goal.lower()
    action = str(step.action_name or "").lower()
    untrusted = (
        step.source in {"web", "file", "tool", "memory"}
        or step.trust < 0.6
    )
    injection = any(
        marker in text
        for marker in (
            "ignore previous",
            "system message",
            "admin approved",
            "系统指令",
            "忽略之前",
            "管理员已批准",
        )
    )
    sensitive = any(
        marker in text
        for marker in (
            "secret",
            "token",
            "password",
            "credential",
            "机密",
            "密钥",
        )
    )
    external = action in {
        "send_email",
        "network_send",
        "http_post",
        "upload_file",
    }
    destructive = action in {
        "delete_file",
        "remove_file",
        "execute_code",
        "shell_exec",
    }
    memory_write = action == "memory_write" or step.phase == "memory_write"
    evidence_failure = any(
        marker in str(step.tool_result or "").lower()
        for marker in ("failed", "blocked", "error", "失败", "阻止")
    )
    permission_granted = bool(
        step.metadata.get("human_confirmed")
        or step.source in {"user", "system", "trusted_planner"}
        and step.trust >= 0.8
        or _prior_authorized_action(trace, step)
    )
    unsafe_memory_write = memory_write and _unsafe_memory_write_content(step)
    benign_preference_memory_write = (
        memory_write
        and permission_granted
        and not unsafe_memory_write
        and _benign_preference_memory_write(step, trace)
    )
    public_recipient = _public_sandbox_recipient(step)
    sandbox_context = any(
        marker in f"{goal} {text}"
        for marker in ("sandbox", "example.com", "example.org", "example.net")
    )
    non_sensitive_content = not sensitive and any(
        marker in f"{goal} {text}"
        for marker in ("non-sensitive", "public", "sandbox")
    )
    goal_conflict = (
        external
        and any(marker in goal for marker in ("不得向外", "do not send", "local"))
    ) or (
        action in {"delete_file", "remove_file"}
        and any(marker in goal for marker in ("不得修改", "不得删除", "do not delete"))
    )
    return {
        "goal_preserved": not goal_conflict,
        "instruction_polluted": untrusted and injection,
        "memory_trusted": (
            not memory_write
            or not untrusted
            or benign_preference_memory_write
        ),
        "evidence_supported": not evidence_failure,
        "action_consistent": not goal_conflict,
        "permission_granted": permission_granted,
        "irreversible": destructive or external,
        "sensitive_flow": sensitive and external,
        "external": external,
        "destructive": destructive,
        "memory_write": memory_write,
        "benign_preference_memory_write": benign_preference_memory_write,
        "unsafe_memory_write": unsafe_memory_write,
        "public_recipient": public_recipient,
        "sandbox_context": sandbox_context,
        "non_sensitive_content": non_sensitive_content,
        "authorized_public_sandbox_external": (
            external
            and permission_granted
            and public_recipient
            and sandbox_context
            and non_sensitive_content
            and not sensitive
            and not goal_conflict
        ),
    }


def abstract_state(
    step: TraceStep,
    trace: LogicTrace,
    *,
    features: dict[str, bool | str | float] | None = None,
) -> str:
    if features is None:
        features = state_features(step, trace)
    return "|".join(
        f"{key}={int(bool(value))}"
        for key, value in ((key, features[key]) for key in ABSTRACT_STATE_FEATURES)
    )


def _prior_authorized_action(trace: LogicTrace, step: TraceStep) -> bool:
    action = str(step.action_name or "").lower()
    if step.phase != "after_action" or not action:
        return False
    args = dict(step.action_args or {})
    for previous in reversed(trace.steps):
        if previous.phase != "before_action":
            continue
        if str(previous.action_name or "").lower() != action:
            continue
        if dict(previous.action_args or {}) != args:
            continue
        return bool(
            previous.metadata.get("human_confirmed")
            or previous.source in {"user", "system", "trusted_planner"}
            and previous.trust >= 0.8
        )
    return False


def _public_sandbox_recipient(step: TraceStep) -> bool:
    if str(step.action_name or "").lower() not in {"send_email", "network_send"}:
        return False
    recipient = str(
        step.action_args.get("to")
        or step.action_args.get("recipient")
        or step.action_args.get("url")
        or ""
    ).lower()
    if any(marker in recipient for marker in ("attacker", "leak", "exfil")):
        return False
    return recipient.endswith(("@example.com", "@example.org", "@example.net"))


def _unsafe_memory_write_content(step: TraceStep) -> bool:
    text = json.dumps(step.action_args, ensure_ascii=False).lower()
    text = f"{text} {step.content} {step.tool_result or ''}".lower()
    unsafe_markers = (
        "api_key",
        "system_prompt",
        "tool_permission",
        "admin",
        "policy",
        "permission",
        "secret",
        "token",
        "password",
        "credential",
        "external",
        "attacker",
        "always send",
        "execute_code",
        "delete",
        "email secrets",
        "密钥",
        "令牌",
        "口令",
        "凭据",
        "策略",
        "权限",
        "外发",
    )
    return any(marker in text for marker in unsafe_markers)


def _benign_preference_memory_write(step: TraceStep, trace: LogicTrace) -> bool:
    args = dict(step.action_args or {})
    key = str(args.get("key") or "").strip().strip(" .:_-").lower()
    value = str(args.get("value") or "").strip().lower()
    combined = f"{trace.user_goal} {step.content} {key} {value}".lower()
    preference_keys = {
        "language",
        "locale",
        "timezone",
        "theme",
        "format",
        "preferred_language",
        "display_language",
        "preference",
    }
    preference_markers = (
        "preference",
        "preferred",
        "language",
        "locale",
        "timezone",
        "theme",
        "non-secret",
        "non sensitive",
        "non-sensitive",
        "偏好",
        "语言",
        "非敏感",
    )
    return key in preference_keys or any(marker in combined for marker in preference_markers)


def _infer_valid_transitions(
    states: set[str],
    observed: dict[str, Counter[str]],
) -> dict[str, set[str]]:
    valid: dict[str, set[str]] = {}
    for source in states:
        targets = set(observed.get(source, {}))
        for target in states:
            if _hamming_distance(source, target) <= 2:
                targets.add(target)
        targets.update(TERMINALS)
        valid[source] = targets
    return valid


def _hamming_distance(left: str, right: str) -> int:
    left_values = left.split("|")
    right_values = right.split("|")
    if len(left_values) != len(right_values):
        return max(len(left_values), len(right_values))
    return sum(a != b for a, b in zip(left_values, right_values))


def _heuristic_risk(features: dict[str, bool | str | float]) -> float:
    score = 0.02
    if features["instruction_polluted"]:
        score += 0.45
    if not features["goal_preserved"]:
        score += 0.30
    if not features["memory_trusted"]:
        score += 0.28
    if features.get("unsafe_memory_write"):
        score += 0.40
    if not features["evidence_supported"]:
        score += 0.18
    if features["irreversible"] and not features["permission_granted"]:
        score += 0.28
    if features["sensitive_flow"]:
        score += 0.35
    return min(score, 0.98)


def _state_pac_epsilon(
    n: int,
    *,
    state_count: int,
    confidence: float,
) -> float:
    if n <= 0:
        return 1.0
    delta = max((1.0 - confidence) / max(state_count, 1), 1e-12)
    return min(math.sqrt(math.log(2.0 / delta) / (2.0 * n)), 1.0)


def _required_state_visits(
    *,
    state_count: int,
    confidence: float,
    epsilon: float,
) -> int:
    delta = max((1.0 - confidence) / max(state_count, 1), 1e-12)
    return math.ceil(math.log(2.0 / delta) / (2.0 * epsilon * epsilon))


def _classification_metrics(
    scored: list[tuple[float, bool]],
    threshold: float,
) -> dict[str, float]:
    predicted = [score >= threshold for score, _ in scored]
    expected = [unsafe for _, unsafe in scored]
    tp = sum(p and e for p, e in zip(predicted, expected))
    fp = sum(p and not e for p, e in zip(predicted, expected))
    fn = sum(not p and e for p, e in zip(predicted, expected))
    tn = sum(not p and not e for p, e in zip(predicted, expected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _deduplicate_adjacent(states: list[str]) -> list[str]:
    result: list[str] = []
    for state in states:
        if not result or result[-1] != state:
            result.append(state)
    return result


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0
