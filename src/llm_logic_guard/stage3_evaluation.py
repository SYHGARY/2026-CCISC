from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .claims import build_claim_graph
from .detector import LogicConsistencyDetector
from .hybrid import HybridConsistencyDetector, semantic_pair_eligible
from .models import LogicTrace, TraceStep
from .semantic import LexicalNLI, TransformersNLI
from .solver import StructuredConstraintSolver
from .specs import SpecEngine


def generate_stage3_benchmark(seed: int = 20260615) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entities = [
        "service", "gateway", "database", "worker", "account",
        "scheduler", "endpoint", "backup", "camera", "sensor",
    ]
    zh_entities = [
        "服务", "网关", "数据库", "任务器", "账户",
        "调度器", "接口", "备份", "摄像头", "传感器",
    ]
    for index in range(40):
        entity = entities[index % len(entities)]
        left, right = (
            (f"{entity} is available.", f"{entity} is unavailable.")
            if index % 2 == 0
            else (f"{entity} is enabled.", f"{entity} is disabled.")
        )
        rows.append(_pair_row("en_contradiction", index, left, right, True))
    for index in range(40):
        entity = zh_entities[index % len(zh_entities)]
        left, right = (
            (f"{entity}当前可用。", f"{entity}当前不可用。")
            if index % 2 == 0
            else (f"{entity}已启用。", f"{entity}已禁用。")
        )
        rows.append(_pair_row("zh_contradiction", index, left, right, True))
    for index in range(30):
        subject_en = entities[index % len(entities)]
        subject_zh = zh_entities[index % len(zh_entities)]
        rows.append(_pair_row(
            "cross_language_contradiction",
            index,
            f"{subject_en} is available.",
            f"{subject_zh}当前不可用。",
            True,
        ))
    for index in range(30):
        entity = entities[index % len(entities)]
        rows.append(_pair_row(
            "semantic_consistent",
            index,
            f"{entity} is available.",
            f"The audit log for {entity} was reviewed.",
            False,
        ))
    for index in range(30):
        entity = entities[index % len(entities)]
        rows.append(_pair_row(
            "temporal_change",
            index,
            f"{entity} was unavailable yesterday.",
            f"{entity} is available now.",
            False,
        ))
    for index in range(20):
        left = entities[index % len(entities)]
        right = entities[(index + 1) % len(entities)]
        rows.append(_pair_row(
            "different_entities",
            index,
            f"{left} is enabled.",
            f"{right} is disabled.",
            False,
        ))
    for index in range(20):
        principal = f"agent{index}"
        rows.append(_pair_row(
            "untrusted_permission",
            index,
            f"{principal} is allowed to send the report.",
            "The permission came from retrieved file content.",
            True,
            source="file",
            trust=0.2,
        ))
    for index in range(20):
        principal = f"user{index}"
        rows.append(_pair_row(
            "trusted_permission",
            index,
            f"{principal} is allowed to access the report.",
            "The user explicitly granted this permission.",
            False,
            source="user",
            trust=0.95,
        ))
    for index in range(10):
        entity = entities[index]
        rows.append(_pair_row(
            "cross_language_neutral",
            index,
            f"{entity} is online.",
            f"{zh_entities[index]}的审计记录已经保存。",
            False,
        ))
    random.Random(seed).shuffle(rows)
    for index, row in enumerate(rows):
        row["split"] = _split_for(index)
    return rows


def write_stage3_benchmark(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def load_stage3_benchmark(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_stage3_evaluation(
    path: Path,
    *,
    use_transformer: bool = True,
    maximum_false_positive_rate: float = 0.10,
) -> dict[str, Any]:
    rows = load_stage3_benchmark(path)
    semantic = (
        TransformersNLI(local_files_only=True)
        if use_transformer
        else LexicalNLI()
    )
    pairs = [
        (row["premise"], row["hypothesis"]) for row in rows
    ]
    nli_scores = semantic.score_pairs(pairs)
    validation = [
        (score, bool(row["inconsistent"]))
        for row, score in zip(rows, nli_scores)
        if row["split"] == "validation"
    ]
    threshold, validation_metrics = _calibrate(
        validation,
        maximum_false_positive_rate,
    )

    legacy = LogicConsistencyDetector()
    specs = SpecEngine()
    solver = StructuredConstraintSolver()
    structured = HybridConsistencyDetector(
        semantic_backend=_DisabledSemantic(),
    )
    predictions: dict[str, list[bool]] = {
        "legacy_baseline": [],
        "nli_only": [],
        "dsl_only": [],
        "z3_only": [],
        "hybrid": [],
    }
    expected: list[bool] = []
    categories: list[str] = []
    test_scores: list[float] = []
    for row, score in zip(rows, nli_scores):
        if row["split"] != "test":
            continue
        trace = LogicTrace.from_dict(row["trace"])
        claims, _ = build_claim_graph(trace)
        structured_flag = bool(structured.analyze(trace)[0])
        nli_flag = score >= threshold
        predictions["legacy_baseline"].append(bool(legacy.analyze(trace)))
        predictions["nli_only"].append(nli_flag)
        predictions["dsl_only"].append(bool(specs.evaluate(trace)))
        predictions["z3_only"].append(bool(solver.check_claims(claims)))
        nli_eligible = semantic_pair_eligible(
            trace.steps[0], trace.steps[1]
        )
        predictions["hybrid"].append(
            structured_flag or (nli_flag and nli_eligible)
        )
        expected.append(bool(row["inconsistent"]))
        categories.append(str(row["category"]))
        test_scores.append(score)

    reports = {
        name: _metrics(values, expected)
        for name, values in predictions.items()
    }
    category_report = {
        category: _metrics(
            [
                predictions["hybrid"][index]
                for index, value in enumerate(categories)
                if value == category
            ],
            [
                expected[index]
                for index, value in enumerate(categories)
                if value == category
            ],
        )
        for category in sorted(set(categories))
    }
    return {
        "dataset_size": len(rows),
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "semantic_backend": semantic.name,
        "nli_threshold": threshold,
        "validation": validation_metrics,
        "test": reports,
        "hybrid_by_category": category_report,
        "average_test_nli_score": (
            sum(test_scores) / len(test_scores) if test_scores else 0.0
        ),
    }


def _pair_row(
    category: str,
    index: int,
    premise: str,
    hypothesis: str,
    inconsistent: bool,
    *,
    source: str = "agent",
    trust: float = 0.8,
) -> dict[str, Any]:
    trace_id = f"stage3-{category}-{index:03d}"
    trace = LogicTrace(
        trace_id=trace_id,
        user_goal="Check observable statements for consistency.",
        steps=[
            TraceStep(
                step_id=f"{trace_id}:premise",
                role="plan",
                content=premise,
                source=source,
                trust=trust,
            ),
            TraceStep(
                step_id=f"{trace_id}:hypothesis",
                role="final_answer",
                content=hypothesis,
                source=source,
                trust=trust,
            ),
        ],
    )
    return {
        "id": trace_id,
        "category": category,
        "premise": premise,
        "hypothesis": hypothesis,
        "inconsistent": inconsistent,
        "trace": {
            "trace_id": trace.trace_id,
            "user_goal": trace.user_goal,
            "steps": [
                {
                    "step_id": step.step_id,
                    "role": step.role,
                    "content": step.content,
                    "source": step.source,
                    "trust": step.trust,
                }
                for step in trace.steps
            ],
        },
    }


def _calibrate(
    scored: list[tuple[float, bool]],
    maximum_false_positive_rate: float,
) -> tuple[float, dict[str, float]]:
    values = sorted({round(score, 8) for score, _ in scored})
    candidates = values + [
        (left + right) / 2.0
        for left, right in zip(values, values[1:])
    ]
    reports = [
        _metrics([score >= threshold for score, _ in scored],
                 [label for _, label in scored])
        | {"threshold": threshold}
        for threshold in candidates
    ]
    feasible = [
        report for report in reports
        if report["false_positive_rate"] <= maximum_false_positive_rate
    ]
    selected = max(
        feasible or reports,
        key=lambda report: (
            report["f1"],
            report["recall"],
            -report["false_positive_rate"],
            -report["threshold"],
        ),
    )
    return float(selected["threshold"]), selected


def _metrics(predicted: list[bool], expected: list[bool]) -> dict[str, float]:
    tp = sum(p and e for p, e in zip(predicted, expected))
    fp = sum(p and not e for p, e in zip(predicted, expected))
    fn = sum(not p and e for p, e in zip(predicted, expected))
    tn = sum(not p and not e for p, e in zip(predicted, expected))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / max(len(expected), 1),
        "support": len(expected),
    }


def _split_for(index: int) -> str:
    bucket = index % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "validation"
    return "test"


class _DisabledSemantic:
    name = "disabled"

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        return 0.0

    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:
        return [0.0] * len(pairs)
