from __future__ import annotations

from .claim_extraction import LLMClaimExtractor
from .claims import (
    AdditionalClaimExtractor,
    build_claim_graph,
    extract_step_claims,
)
from .detector import LogicConsistencyDetector
from .models import Claim, ClaimEdge, LogicTrace, Violation
from .semantic import (
    SemanticBackend,
    configured_semantic_threshold,
    create_semantic_backend,
)
from .solver import StructuredConstraintSolver
from .specs import SpecEngine


class HybridConsistencyDetector:
    def __init__(
        self,
        *,
        include_legacy: bool = True,
        spec_engine: SpecEngine | None = None,
        semantic_backend: SemanticBackend | None = None,
        semantic_threshold: float | None = None,
        claim_extractor: AdditionalClaimExtractor | None = None,
    ) -> None:
        self.include_legacy = include_legacy
        self.legacy = LogicConsistencyDetector()
        self.spec_engine = spec_engine or SpecEngine()
        self.semantic = semantic_backend or create_semantic_backend()
        self.semantic_threshold = (
            semantic_threshold
            if semantic_threshold is not None
            else configured_semantic_threshold(self.semantic.name)
        )
        self.solver = StructuredConstraintSolver()
        self.claim_extractor = (
            claim_extractor
            if claim_extractor is not None
            else LLMClaimExtractor.from_environment()
        )

    def analyze(
        self,
        trace: LogicTrace,
        *,
        focus_step_id: str | None = None,
    ) -> tuple[list[Violation], list[Claim], list[ClaimEdge]]:
        # One claim cache shared by every detector in this pass (keyed on
        # step_id). Collapses the redundant per-step re-extraction that the
        # claim graph, legacy detector and semantic anchors would each do.
        claim_cache: dict[str, list[Claim]] = {}
        claims, edges = build_claim_graph(trace, self.claim_extractor, cache=claim_cache)
        violations: list[Violation] = []
        if self.include_legacy:
            violations.extend(self.legacy.analyze(trace, cache=claim_cache))
        violations.extend(self.spec_engine.evaluate(trace))
        violations.extend(self._claim_conflicts(trace, claims))
        violations.extend(
            self._semantic_conflicts(trace, focus_step_id=focus_step_id, cache=claim_cache)
        )
        return _dedupe(violations), claims, edges

    def _claim_conflicts(self, trace: LogicTrace, claims: list[Claim]) -> list[Violation]:
        violations: list[Violation] = []
        for issue in self.solver.check_claims(claims):
            claim = issue.claim
            evidence = [claim.evidence]
            if issue.related:
                evidence.insert(0, issue.related.evidence)
            permission_issue = (
                issue.issue_type == "untrusted_permission_grant"
            )
            violations.append(Violation(
                violation_id=(
                    f"{trace.trace_id}:{claim.step_id}:constraint:"
                    f"{issue.issue_type}:{claim.key}"
                ),
                violation_type=(
                    "permission_provenance_conflict"
                    if permission_issue
                    else "claim_graph_contradiction"
                ),
                severity="critical" if permission_issue else "high",
                step_id=claim.step_id,
                message=(
                    "不可信来源不能授予工具权限。"
                    if permission_issue
                    else f"声明 {claim.key} 与同一时态的先前声明冲突。"
                ),
                evidence=evidence,
                confidence=issue.confidence,
                reversible=True,
                recommended_intervention=(
                    "replan" if issue.confidence >= 0.85 else "audit"
                ),
                detector=f"claim_graph+{self.solver.backend}",
                attack_stage="permission" if permission_issue else "evidence",
            ))
        return violations

    def _semantic_conflicts(
        self,
        trace: LogicTrace,
        *,
        focus_step_id: str | None = None,
        cache: dict[str, list[Claim]] | None = None,
    ) -> list[Violation]:
        visible = [
            step for step in trace.steps
            if step.content and step.phase in {"plan", "environment_observation", "final_answer"}
        ]
        # Precompute per-step temporal scope and semantic anchor keys once, so
        # pair eligibility does not re-extract claims O(N^2) times.
        scopes = [_temporal_scope(step.content) for step in visible]
        anchors = [
            _semantic_anchor_keys(step, _SEMANTIC_ANCHOR_PREDICATES, cache=cache)
            for step in visible
        ]
        violations: list[Violation] = []
        pair_rows = [
            (previous, current)
            for index, current in enumerate(visible)
            # When a focus step is given (per-event runtime path), only the
            # focus step is scored as `current`; its violations are identical
            # to the full pass filtered to that step_id.
            if focus_step_id is None or current.step_id == focus_step_id
            for prev_index, previous in enumerate(visible[:index])
            if _pair_eligible_cached(
                scopes[prev_index], scopes[index], anchors[prev_index], anchors[index]
            )
        ]
        pairs = [
            (previous.content, current.content)
            for previous, current in pair_rows
        ]
        if not pairs:
            scores = []
        elif hasattr(self.semantic, "score_pairs"):
            scores = self.semantic.score_pairs(pairs)
        else:
            scores = [
                self.semantic.contradiction_score(premise, hypothesis)
                for premise, hypothesis in pairs
            ]
        for (previous, current), score in zip(pair_rows, scores):
            if score < self.semantic_threshold:
                continue
            violations.append(Violation(
                violation_id=f"{trace.trace_id}:{current.step_id}:semantic:{previous.step_id}",
                violation_type="semantic_contradiction",
                severity="medium",
                step_id=current.step_id,
                message="自然语言声明与较早轨迹内容存在语义矛盾。",
                evidence=[previous.content[:300], current.content[:300]],
                confidence=score,
                reversible=True,
                recommended_intervention="audit" if score < 0.85 else "replan",
                detector=self.semantic.name,
                attack_stage=current.phase,
            ))
        return violations


def _dedupe(violations: list[Violation]) -> list[Violation]:
    selected: dict[tuple[str, str, str], Violation] = {}
    for violation in violations:
        # Spec-engine violations dedupe on their spec_id. Semantic / claim-graph
        # / legacy violations have no spec_id, so fall back to the unique
        # violation_id as the discriminator; otherwise distinct findings on the
        # same step (e.g. one step contradicting several prior steps) would
        # collapse to a single key and all but the highest-confidence one would
        # be silently dropped.
        discriminator = violation.spec_id or violation.violation_id
        key = (violation.step_id, violation.violation_type, discriminator)
        existing = selected.get(key)
        if existing is None or violation.confidence > existing.confidence:
            selected[key] = violation
    return list(selected.values())


def _temporal_scope(text: str) -> str:
    lower = text.lower()
    if any(
        marker in lower
        for marker in (
            " was ",
            " were ",
            "yesterday",
            "previously",
            "此前",
            "之前",
            "昨天",
            "曾经",
        )
    ):
        return "past"
    if any(
        marker in lower
        for marker in (
            " will ",
            "tomorrow",
            "later",
            "明天",
            "之后",
            "将会",
        )
    ):
        return "future"
    return "current"


def semantic_pair_eligible(
    previous,
    current,
) -> bool:
    if _temporal_scope(previous.content) != _temporal_scope(current.content):
        return False
    previous_keys = _semantic_anchor_keys(previous, _SEMANTIC_ANCHOR_PREDICATES)
    current_keys = _semantic_anchor_keys(current, _SEMANTIC_ANCHOR_PREDICATES)
    return _pair_eligible_cached(
        _temporal_scope(previous.content),
        _temporal_scope(current.content),
        previous_keys,
        current_keys,
    )


_SEMANTIC_ANCHOR_PREDICATES = {"entity_state", "file_exists", "email_sent", "action_safe"}


def _pair_eligible_cached(
    previous_scope: str,
    current_scope: str,
    previous_keys: set[str],
    current_keys: set[str],
) -> bool:
    if previous_scope != current_scope:
        return False
    if not previous_keys or not current_keys:
        return False
    return not previous_keys.isdisjoint(current_keys)


def _semantic_anchor_keys(
    step,
    predicates: set[str],
    *,
    cache: dict[str, list[Claim]] | None = None,
) -> set[str]:
    return {
        claim.key
        for claim in extract_step_claims(step, cache=cache)
        if claim.predicate in predicates
    }
