from __future__ import annotations

from dataclasses import dataclass, field

from .models import AnalysisReport


@dataclass
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    trace_accuracy: float
    macro_f1: float
    per_type: dict[str, dict[str, int | float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "trace_accuracy": self.trace_accuracy,
            "macro_f1": self.macro_f1,
            "per_type": self.per_type,
        }


def evaluate_reports(
    reports: list[AnalysisReport],
    expected_by_trace: dict[str, set[str]],
) -> DetectionMetrics:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    exact_matches = 0
    label_universe = {
        violation_type
        for expected in expected_by_trace.values()
        for violation_type in expected
    }
    label_universe.update(
        violation.violation_type
        for report in reports
        for violation in report.violations
    )

    for report in reports:
        expected = expected_by_trace[report.trace_id]
        predicted = {violation.violation_type for violation in report.violations}
        true_positives += len(predicted & expected)
        false_positives += len(predicted - expected)
        false_negatives += len(expected - predicted)
        true_negatives += len(label_universe - predicted - expected)
        if predicted == expected:
            exact_matches += 1

    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    per_type = _per_type_metrics(reports, expected_by_trace, label_universe)
    return DetectionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        precision=precision,
        recall=recall,
        f1=_safe_divide(2 * precision * recall, precision + recall),
        false_positive_rate=_safe_divide(false_positives, false_positives + true_negatives),
        trace_accuracy=_safe_divide(exact_matches, len(reports)),
        macro_f1=_safe_divide(
            sum(float(metrics["f1"]) for metrics in per_type.values()),
            len(per_type),
        ),
        per_type=per_type,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _per_type_metrics(
    reports: list[AnalysisReport],
    expected_by_trace: dict[str, set[str]],
    label_universe: set[str],
) -> dict[str, dict[str, int | float]]:
    output: dict[str, dict[str, int | float]] = {}
    for violation_type in sorted(label_universe):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for report in reports:
            expected = violation_type in expected_by_trace[report.trace_id]
            predicted = any(
                violation.violation_type == violation_type
                for violation in report.violations
            )
            true_positives += int(expected and predicted)
            false_positives += int(not expected and predicted)
            false_negatives += int(expected and not predicted)
        precision = _safe_divide(true_positives, true_positives + false_positives)
        recall = _safe_divide(true_positives, true_positives + false_negatives)
        output[violation_type] = {
            "support": true_positives + false_negatives,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": _safe_divide(2 * precision * recall, precision + recall),
        }
    return output
