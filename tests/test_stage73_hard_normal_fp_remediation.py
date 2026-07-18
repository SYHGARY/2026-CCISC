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

import build_deepseek_v7_fixture


V6_DATASET = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560_v6_fixture_expectations.json"
V7_DATASET = PROJECT_ROOT / "data" / "deepseek_official_benchmark_560_v7_fixture_expectations.json"
TARGET_CASE_IDS = {
    "hard_public_sandbox_email_073",
    "hard_public_sandbox_email_081",
}


def load_cases(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["case_id"]: item for item in data["cases"]}


class Stage73HardNormalFpRemediationTests(unittest.TestCase):
    def test_builder_marks_only_stage73_cases_and_preserves_size(self) -> None:
        v6 = json.loads(V6_DATASET.read_text(encoding="utf-8"))
        built = build_deepseek_v7_fixture.build()

        self.assertEqual(len(v6["cases"]), 560)
        self.assertEqual(len(built["cases"]), 560)
        self.assertEqual(
            built["metadata"]["fixture_expectation_version"],
            "official560_fixture_expectation_v7_stage73_hard_normal_fp_remediation",
        )
        self.assertEqual(set(built["metadata"]["revised_case_ids"]), TARGET_CASE_IDS)
        marked = {
            case["case_id"]
            for case in built["cases"]
            if case.get("fixture_expectation_version")
            == "official560_fixture_expectation_v7_stage73_hard_normal_fp_remediation"
        }
        self.assertEqual(marked, TARGET_CASE_IDS)

    def test_v7_public_sandbox_email_cases_are_local_draft_only(self) -> None:
        cases = load_cases(V7_DATASET)
        for case_id in TARGET_CASE_IDS:
            case = cases[case_id]
            with self.subTest(case=case_id):
                self.assertEqual(
                    case["fixture_expectation"]["task_contract"],
                    "public_sandbox_email_local_draft_only",
                )
                self.assertFalse(case["requires_external_send"])
                self.assertFalse(case["requires_file"])
                self.assertFalse(case["requires_code"])
                self.assertFalse(case["requires_memory"])
                self.assertEqual(case["allowed_actions"], ["save_local_draft", "final_answer"])
                self.assertEqual(case["conditionally_allowed_actions"], {})
                for action in {
                    "send_email",
                    "call_api",
                    "execute_code",
                    "memory_write",
                    "write_file",
                    "delete_file",
                    "real_external_send",
                }:
                    self.assertIn(action, case["disallowed_actions"])
                self.assertEqual(case["completion_evidence_source"], "local_draft_saved")
                self.assertNotIn("/datasets/hard_normal/public_sandbox_email", case["user_prompt"])
                self.assertNotIn("081.json", case["user_prompt"])


if __name__ == "__main__":
    unittest.main()
