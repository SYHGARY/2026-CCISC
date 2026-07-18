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

import audit_deepseek_candidate_pool_consistency
import select_deepseek_official_benchmark
import validate_deepseek_official_benchmark


class Stage19OfficialBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool_path = PROJECT_ROOT / "data" / "deepseek_candidate_pool_1000.json"
        self.rules_path = PROJECT_ROOT / "outputs" / "stage17_official_benchmark_selection_rules.md"
        self.schema_path = PROJECT_ROOT / "data" / "deepseek_extended_case_schema.json"
        self.core_path = PROJECT_ROOT / "data" / "deepseek_batch_cases.json"

    def core_ids(self) -> set[str]:
        payload = json.loads(self.core_path.read_text(encoding="utf-8"))
        ids: set[str] = set()
        for group in ("attack", "normal", "hard_normal"):
            ids.update(case["id"] for case in payload[group])
        return ids

    def test_candidate_pool_consistency_audit_explains_stage18_surface_mismatch(self) -> None:
        audit = audit_deepseek_candidate_pool_consistency.build_consistency_audit(
            pool_path=self.pool_path,
            schema_path=self.schema_path,
            core_path=self.core_path,
        )
        self.assertTrue(audit["data_consistent"])
        self.assertEqual(audit["sample_type_counts"]["attack"], 750)
        self.assertEqual(audit["attack_sample_attack_surface_assignment_count_sum"], 753)
        self.assertEqual(audit["attack_extra_surface_assignments"], 3)
        self.assertEqual(
            audit["stage18_750_vs_753_resolution"]["judgment"],
            "reporting_ambiguity_not_data_corruption",
        )

    def test_official_benchmark_selection_returns_560_cases_with_stage19_distribution(self) -> None:
        benchmark, trace = select_deepseek_official_benchmark.select_official_benchmark(
            pool_path=self.pool_path,
            rules_path=self.rules_path,
            core_path=self.core_path,
        )
        cases = benchmark["cases"]
        self.assertEqual(len(cases), 560)
        self.assertEqual(trace["selected_sample_type_counts"], {"attack": 400, "hard_normal": 80, "normal": 80})
        self.assertEqual(
            trace["selected_attack_category_counts"],
            {
                "dangerous_code_execution": 50,
                "environment_pollution": 35,
                "file_access_overwrite": 45,
                "jailbreak": 45,
                "memory_poisoning": 40,
                "multi_agent_error_propagation": 25,
                "prompt_injection": 55,
                "sensitive_information_leakage": 55,
                "tool_hijacking": 50,
            },
        )
        self.assertFalse(benchmark["metadata"]["real_api_called"])
        self.assertFalse(benchmark["metadata"]["experiments_run"])

    def test_core_40_are_preserved(self) -> None:
        benchmark, _trace = select_deepseek_official_benchmark.select_official_benchmark(
            pool_path=self.pool_path,
            rules_path=self.rules_path,
            core_path=self.core_path,
        )
        selected_ids = {case["case_id"] for case in benchmark["cases"]}
        self.assertTrue(self.core_ids().issubset(selected_ids))

    def test_validator_passes_for_selected_official_benchmark(self) -> None:
        benchmark, _trace = select_deepseek_official_benchmark.select_official_benchmark(
            pool_path=self.pool_path,
            rules_path=self.rules_path,
            core_path=self.core_path,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_path = Path(temp_dir) / "official.json"
            benchmark_path.write_text(json.dumps(benchmark, ensure_ascii=False), encoding="utf-8")
            coverage, quality, errors = validate_deepseek_official_benchmark.validate_official_benchmark(
                benchmark_path=benchmark_path,
                candidate_pool_path=self.pool_path,
                schema_path=self.schema_path,
                core_path=self.core_path,
                write_report_files=False,
            )
        self.assertEqual(errors, [])
        self.assertTrue(quality["validation_passed"])
        self.assertEqual(coverage["total_count"], 560)
        self.assertEqual(coverage["core_40"]["present_count"], 40)
        self.assertEqual(quality["hard_normal_missing_boundary_reason_count"], 0)
        self.assertEqual(quality["sensitive_pattern_hit_count"], 0)
        self.assertEqual(quality["destructive_pattern_hit_count"], 0)

    def test_stage19_scripts_do_not_import_experiment_runner_or_api_clients(self) -> None:
        forbidden = [
            "run_deepseek_batch_experiment",
            "summarize_deepseek_experiment",
            "DEEPSEEK_API_KEY",
            "import httpx",
            "from httpx",
            "import requests",
            "from requests",
        ]
        script_names = [
            "audit_deepseek_candidate_pool_consistency.py",
            "select_deepseek_official_benchmark.py",
            "validate_deepseek_official_benchmark.py",
        ]
        for script_name in script_names:
            source = (PROJECT_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, script_name)


if __name__ == "__main__":
    unittest.main()
