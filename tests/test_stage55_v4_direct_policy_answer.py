from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
SRC = PROJECT_ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_deepseek_batch_experiment
import summarize_deepseek_experiment


V1_DATASET = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560.json"
V2_DATASET = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560_v2_fixture_expectations.json"
V3_DATASET = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560_v3_fixture_expectations.json"
V4_DATASET = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560_v4_fixture_expectations.json"
STAGE52_REPORT = PROJECT_ROOT / "outputs" / "deepseek_stage52_batch2_retest_v3_real.json"
STAGE52_GATE = PROJECT_ROOT / "outputs" / "stage52_batch2_retest_v3_gate_decision.json"
TARGET_CASE_ID = "norm_direct_policy_answer_047"


def load_cases(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["case_id"]: item for item in data["cases"]}


def load_stage52_record() -> dict:
    report = json.loads(STAGE52_REPORT.read_text(encoding="utf-8"))
    return next(item for item in report["records"] if item["id"] == TARGET_CASE_ID)


class Stage55V4DirectPolicyAnswerTests(unittest.TestCase):
    def test_v4_fixture_is_independent_and_marks_only_target_case(self) -> None:
        v4_data = json.loads(V4_DATASET.read_text(encoding="utf-8"))

        self.assertEqual(len(json.loads(V1_DATASET.read_text(encoding="utf-8"))["cases"]), 560)
        self.assertEqual(len(json.loads(V2_DATASET.read_text(encoding="utf-8"))["cases"]), 560)
        self.assertEqual(len(json.loads(V3_DATASET.read_text(encoding="utf-8"))["cases"]), 560)
        self.assertEqual(len(v4_data["cases"]), 560)
        self.assertEqual(
            v4_data["metadata"]["fixture_expectation_version"],
            "official560_fixture_expectation_v4_candidate",
        )
        self.assertEqual(v4_data["metadata"]["revised_case_ids"], [TARGET_CASE_ID])

        marked = {
            case["case_id"]
            for case in v4_data["cases"]
            if case.get("fixture_expectation_version")
            == "official560_fixture_expectation_v4_candidate"
        }
        self.assertEqual(marked, {TARGET_CASE_ID})

    def test_v4_fixture_direct_answer_contract_does_not_overwrite_v1_to_v3(self) -> None:
        v1_case = load_cases(V1_DATASET)[TARGET_CASE_ID]
        v2_case = load_cases(V2_DATASET)[TARGET_CASE_ID]
        v3_case = load_cases(V3_DATASET)[TARGET_CASE_ID]
        v4_case = load_cases(V4_DATASET)[TARGET_CASE_ID]

        for case in (v1_case, v2_case, v3_case):
            self.assertNotEqual(
                case.get("fixture_expectation_version"),
                "official560_fixture_expectation_v4_candidate",
            )
        self.assertTrue(v4_case["direct_answer_only"])
        self.assertTrue(v4_case["side_effect_tools_disallowed"])
        self.assertEqual(v4_case["completion_evidence_source"], "final_answer_text")
        self.assertEqual(v4_case["allowed_actions"], ["final_answer"])
        self.assertIn("write_file", v4_case["disallowed_actions"])
        self.assertIn("read_file", v4_case["disallowed_actions"])
        self.assertIn("true_guard_overblock", v4_case["negative_reporting_labels"])
        self.assertIn("provider_parser_issue", v4_case["negative_reporting_labels"])

    def test_reporting_labels_for_stage52_target_case_are_report_only(self) -> None:
        record = load_stage52_record()

        self.assertTrue(run_deepseek_batch_experiment.is_metric_false_positive(record))
        labels = run_deepseek_batch_experiment.classify_reporting_labels(record)

        self.assertTrue(labels["report_only"])
        self.assertFalse(labels["affects_gate"])
        self.assertFalse(labels["affects_fpr"])
        self.assertEqual(labels["primary_label"], "fixture_ambiguity")
        self.assertIn("direct_answer_trajectory_drift", labels["secondary_labels"])
        self.assertIn("metric_evidence_gap", labels["optional_labels"])
        self.assertIn("true_guard_overblock", labels["negative_labels"])
        self.assertIn("provider_parser_issue", labels["negative_labels"])
        for definition in labels["definitions"].values():
            self.assertTrue(definition["report_only"])
            self.assertFalse(definition["affects_gate"])
            self.assertFalse(definition["affects_fpr"])

    def test_reporting_labels_do_not_change_stage52_metrics_or_failed_gate(self) -> None:
        report = json.loads(STAGE52_REPORT.read_text(encoding="utf-8"))
        gate = json.loads(STAGE52_GATE.read_text(encoding="utf-8"))

        before = summarize_deepseek_experiment.summarize_report(report)
        _ = run_deepseek_batch_experiment.collect_reporting_label_classifications(report)
        after = summarize_deepseek_experiment.summarize_report(report)

        self.assertEqual(before["metrics"], after["metrics"])
        self.assertEqual(before["metrics"]["false_positive_rate_on_normal"], 0.166667)
        self.assertFalse(gate["gate_pass"])
        self.assertTrue(str(gate["decision"]).startswith("FAILED"))
        self.assertIn(TARGET_CASE_ID, gate["false_positive_case_ids"])


if __name__ == "__main__":
    unittest.main()
