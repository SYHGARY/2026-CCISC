from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .llm_provider import LLMProvider, provider_from_environment
from .models import Claim, TraceStep


@dataclass
class LLMClaimExtractor:
    provider: LLMProvider
    name: str = "llm_claim_extractor.v1"

    @classmethod
    def from_environment(cls) -> "LLMClaimExtractor | None":
        if os.getenv("LOGICGUARD_ENABLE_LLM_CLAIMS", "0") != "1":
            return None
        try:
            return cls(provider_from_environment())
        except Exception:
            return None

    def extract(self, step: TraceStep) -> list[Claim]:
        text = " ".join(
            part for part in (step.content, str(step.tool_result or "")) if part
        ).strip()
        if not text:
            return []
        response = self.provider.complete_json(
            role="claim_extractor",
            system=(
                "Extract only explicit observable claims. Return JSON with key "
                "'claims'. Each claim has subject, predicate, polarity, "
                "confidence, and temporal_scope. Do not infer hidden facts."
            ),
            user=json.dumps({
                "text": text,
                "event_source": step.source,
                "trust": step.trust,
            }, ensure_ascii=False),
            max_tokens=800,
        )
        claims: list[Claim] = []
        for item in response.get("claims", []):
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject") or "").strip().lower()
            predicate = str(item.get("predicate") or "").strip().lower()
            polarity = str(item.get("polarity") or "").strip().lower()
            if not subject or not predicate or not polarity:
                continue
            claims.append(Claim(
                subject=subject,
                predicate=predicate,
                polarity=polarity,
                evidence=text[:500],
                step_id=step.step_id,
                source="llm_extraction",
                confidence=_bounded_float(item.get("confidence"), 0.75),
                trust=step.trust,
                timestamp=step.timestamp,
                valid_from_step=step.step_id,
                provenance={
                    "agent_id": step.agent_id,
                    "event_source": step.source,
                    "temporal_scope": str(
                        item.get("temporal_scope") or "current"
                    ),
                    "extractor": self.name,
                },
            ))
        return claims


def _bounded_float(value: Any, default: float) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default
