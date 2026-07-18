from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_deepseek_candidate_pool
import validate_deepseek_candidate_pool


class Stage18CandidatePoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool_path = PROJECT_ROOT / "data" / "deepseek_candidate_pool_1000.json"
        self.schema_path = PROJECT_ROOT / "data" / "deepseek_extended_case_schema.json"
        self.distribution_path = PROJECT_ROOT / "outputs" / "stage17_candidate_pool_distribution.json"

    def load_cases(self) -> list[dict[str, object]]:
        payload = json.loads(self.pool_path.read_text(encoding="utf-8"))
        return payload["cases"]

    def test_generated_pool_has_1000_cases_and_stage17_distribution(self) -> None:
        cases = self.load_cases()
        self.assertEqual(len(cases), 1000)

        sample_counts: dict[str, int] = {}
        category_counts: dict[str, int] = {}
        for case in cases:
            sample_counts[str(case["sample_type"])] = sample_counts.get(str(case["sample_type"]), 0) + 1
            category_counts[str(case["category"])] = category_counts.get(str(case["category"]), 0) + 1

        self.assertEqual(sample_counts["attack"], 750)
        self.assertEqual(sample_counts["normal"], 125)
        self.assertEqual(sample_counts["hard_normal"], 125)

        expected_attack_counts = json.loads(self.distribution_path.read_text(encoding="utf-8"))[
            "attack_category_counts"
        ]
        for category, expected_count in expected_attack_counts.items():
            self.assertEqual(category_counts[category], expected_count)

    def test_all_cases_have_required_and_recommended_fields(self) -> None:
        schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        required = set(schema["required"])
        recommended = {
            "quality_tags",
            "risk_level",
            "real_priority",
            "variant_family",
            "is_core_case",
            "source_dataset",
        }
        for case in self.load_cases():
            self.assertTrue(required.issubset(case), case["case_id"])
            self.assertTrue(recommended.issubset(case), case["case_id"])
            self.assertTrue(case["success_criteria"], case["case_id"])
            self.assertTrue(case["failure_criteria"], case["case_id"])
            if case["sample_type"] == "hard_normal":
                self.assertTrue(str(case["boundary_reason"]).strip(), case["case_id"])

    def test_validator_passes_and_reports_core_40_preserved(self) -> None:
        coverage, quality, errors = validate_deepseek_candidate_pool.validate_candidate_pool(
            pool_path=self.pool_path,
            write_report_files=False,
        )
        self.assertEqual(errors, [])
        self.assertTrue(quality["validation_passed"])
        self.assertEqual(coverage["core_40"]["present_count"], 40)
        self.assertEqual(coverage["core_40"]["missing_count"], 0)
        self.assertEqual(quality["duplicate_case_ids"], [])
        self.assertEqual(quality["duplicate_prompt_count"], 0)
        self.assertEqual(quality["sensitive_pattern_hits"], [])
        self.assertEqual(quality["destructive_pattern_hits"], [])

    def test_generation_function_creates_valid_pool_without_experiment_runner(self) -> None:
        pool = generate_deepseek_candidate_pool.build_candidate_pool(
            core_path=PROJECT_ROOT / "data" / "deepseek_batch_cases.json",
            distribution_path=self.distribution_path,
        )
        self.assertEqual(len(pool["cases"]), 1000)
        self.assertFalse(pool["metadata"]["real_api_called"])
        self.assertFalse(pool["metadata"]["experiments_run"])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "candidate_pool.json"
            temp_path.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
            _coverage, _quality, errors = validate_deepseek_candidate_pool.validate_candidate_pool(
                pool_path=temp_path,
                write_report_files=False,
            )
            self.assertEqual(errors, [])

    def test_stage18_scripts_do_not_import_experiment_runner(self) -> None:
        generator_source = (PROJECT_ROOT / "scripts" / "generate_deepseek_candidate_pool.py").read_text(
            encoding="utf-8"
        )
        validator_source = (PROJECT_ROOT / "scripts" / "validate_deepseek_candidate_pool.py").read_text(
            encoding="utf-8"
        )
        forbidden = [
            "run_deepseek_batch_experiment",
            "summarize_deepseek_experiment",
            "DEEPSEEK_API_KEY",
            "import httpx",
            "from httpx",
            "import requests",
            "from requests",
        ]
        for token in forbidden:
            self.assertNotIn(token, generator_source)
            self.assertNotIn(token, validator_source)


if __name__ == "__main__":
    unittest.main()
