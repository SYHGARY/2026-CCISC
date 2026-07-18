from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class SemanticBackend(Protocol):
    name: str

    def contradiction_score(self, premise: str, hypothesis: str) -> float: ...

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]: ...


@dataclass
class LexicalNLI:
    name: str = "lexical_nli.v2"

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        left = _normalize(premise)
        right = _normalize(hypothesis)
        if not left or not right:
            return 0.0
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        overlap = len(left_tokens & right_tokens) / max(
            len(left_tokens | right_tokens), 1
        )
        if overlap < 0.18:
            return 0.0
        left_neg = _has_negation(left)
        right_neg = _has_negation(right)
        if left_neg != right_neg:
            return min(0.58 + overlap * 0.38, 0.94)
        antonym_hits = sum(
            1
            for a, b in _ANTONYMS
            if (a in left and b in right) or (b in left and a in right)
        )
        if not antonym_hits:
            return 0.0
        return min(0.55 + 0.16 * antonym_hits + overlap * 0.2, 0.92)

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        return [
            self.contradiction_score(premise, hypothesis)
            for premise, hypothesis in pairs
        ]


class TransformersNLI:
    name = "mdeberta_v3_mnli_xnli.v1"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int = 8,
        local_files_only: bool | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        selected = model_name or os.getenv(
            "LOGICGUARD_NLI_MODEL",
            "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        )
        if local_files_only is None:
            local_files_only = (
                os.getenv("LOGICGUARD_NLI_LOCAL_ONLY", "0") == "1"
            )
        self.model_name = selected
        self.batch_size = batch_size
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            selected,
            local_files_only=local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            selected,
            local_files_only=local_files_only,
        )
        self._model.eval()
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model.to(self._device)
        labels = {
            int(index): str(label).lower()
            for index, label in self._model.config.id2label.items()
        }
        self._contradiction_index = next(
            (
                index
                for index, label in labels.items()
                if "contrad" in label
            ),
            2,
        )

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        return self.score_pairs([(premise, hypothesis)])[0]

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        scores: list[float] = []
        with self._torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                encoded = self._tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    padding=True,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                }
                logits = self._model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1)
                scores.extend(
                    float(value)
                    for value in probabilities[
                        :, self._contradiction_index
                    ].cpu()
                )
        return scores


def create_semantic_backend() -> SemanticBackend:
    if os.getenv("LOGICGUARD_ENABLE_TRANSFORMERS", "0") == "1":
        try:
            return TransformersNLI()
        except Exception:
            pass
    return LexicalNLI()


def configured_semantic_threshold(backend_name: str) -> float:
    environment = os.getenv("LOGICGUARD_NLI_THRESHOLD", "").strip()
    if environment:
        try:
            return float(environment)
        except ValueError:
            pass
    if backend_name.startswith("mdeberta"):
        path = (
            Path(__file__).resolve().parents[2]
            / "config"
            / "nli_config.json"
        )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return float(data.get("threshold", 0.72))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return 0.72


_ANTONYMS = (
    ("exists", "missing"),
    ("success", "failed"),
    ("allowed", "denied"),
    ("safe", "unsafe"),
    ("sent", "blocked"),
    ("open", "closed"),
    ("enabled", "disabled"),
    ("increase", "decrease"),
    ("存在", "不存在"),
    ("成功", "失败"),
    ("允许", "拒绝"),
    ("安全", "不安全"),
    ("开启", "关闭"),
    ("启用", "禁用"),
    ("增加", "减少"),
)


def _normalize(text: str) -> str:
    return " ".join(
        re.findall(r"[\w\u4e00-\u9fff.-]+", text.lower())
    )


def _has_negation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:not|never|no|cannot|can't|didn't|doesn't)\b"
            r"|不|未|没有|禁止",
            text,
        )
    )
