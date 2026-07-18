from __future__ import annotations

from dataclasses import dataclass

from .models import Claim

try:
    import z3
except ImportError:  # pragma: no cover
    z3 = None


@dataclass
class ConstraintResult:
    consistent: bool
    backend: str
    reason: str


@dataclass
class ConstraintIssue:
    issue_type: str
    claim: Claim
    related: Claim | None
    reason: str
    confidence: float


class StructuredConstraintSolver:
    """Checks value, temporal, provenance, and permission constraints."""

    @property
    def backend(self) -> str:
        return "z3" if z3 is not None else "deterministic_fallback"

    def check_claim_pair(
        self,
        left: Claim,
        right: Claim,
    ) -> ConstraintResult:
        if left.key != right.key:
            return ConstraintResult(
                True, self.backend, "different symbolic variables"
            )
        if not _same_temporal_scope(left, right):
            return ConstraintResult(
                True, self.backend, "different temporal scopes"
            )
        if left.polarity == right.polarity:
            return ConstraintResult(True, self.backend, "same assignment")
        if z3 is None:
            return ConstraintResult(
                False,
                self.backend,
                "different values assigned to one claim key",
            )
        variable = z3.String(_z3_name(left.key))
        solver = z3.Solver()
        solver.add(variable == z3.StringVal(left.polarity))
        solver.add(variable == z3.StringVal(right.polarity))
        consistent = solver.check() == z3.sat
        return ConstraintResult(
            consistent,
            self.backend,
            "satisfiable"
            if consistent
            else "UNSAT: incompatible values for one claim key",
        )

    def check_claims(self, claims: list[Claim]) -> list[ConstraintIssue]:
        issues: list[ConstraintIssue] = []
        active: dict[tuple[str, str], Claim] = {}
        for claim in claims:
            if claim.invalidated_by:
                continue
            temporal_scope = str(
                claim.provenance.get("temporal_scope", "current")
            )
            key = (claim.key, temporal_scope)
            previous = active.get(key)
            if previous:
                result = self.check_claim_pair(previous, claim)
                if not result.consistent:
                    issues.append(ConstraintIssue(
                        issue_type="value_conflict",
                        claim=claim,
                        related=previous,
                        reason=result.reason,
                        confidence=min(
                            max(previous.trust, claim.trust),
                            0.97,
                        ),
                    ))
            if (
                claim.predicate.startswith("permission_")
                and claim.polarity == "positive"
                and (
                    claim.trust < 0.8
                    or claim.provenance.get("event_source")
                    in {"file", "web", "tool", "memory"}
                )
            ):
                issues.append(ConstraintIssue(
                    issue_type="untrusted_permission_grant",
                    claim=claim,
                    related=None,
                    reason=(
                        "permission grant originates from an untrusted source"
                    ),
                    confidence=max(0.9, claim.confidence),
                ))
            active[key] = claim
        return issues


def _same_temporal_scope(left: Claim, right: Claim) -> bool:
    return (
        left.provenance.get("temporal_scope", "current")
        == right.provenance.get("temporal_scope", "current")
    )


def _z3_name(value: str) -> str:
    return "".join(
        char if char.isalnum() else "_" for char in value
    )
