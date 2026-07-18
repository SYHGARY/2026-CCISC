from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_demo_evidence_viewer


class Stage15DemoViewerTests(unittest.TestCase):
    def test_viewer_outputs_are_generated(self) -> None:
        data = self._build_data()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            viewer_dir = tmp_path / "demo_viewer"
            script = tmp_path / "presentation.md"
            qa = tmp_path / "qa.md"
            ppt = tmp_path / "ppt.md"

            build_demo_evidence_viewer.write_outputs(data, viewer_dir, script, qa, ppt)

            self.assertTrue((viewer_dir / "index.html").exists())
            self.assertTrue((viewer_dir / "demo_data.json").exists())
            self.assertTrue((viewer_dir / "README.md").exists())
            self.assertTrue(script.exists())
            self.assertTrue(qa.exists())
            self.assertTrue(ppt.exists())

    def test_demo_data_contains_four_expected_cases(self) -> None:
        data = self._build_data()

        self.assertFalse(data["real_api_used_to_build_viewer"])
        self.assertEqual(data["case_count"], 4)
        self.assertEqual(
            [case["case_id"] for case in data["cases"]],
            [
                "atk_prompt_file_email_01",
                "atk_tool_hijack_api_delete",
                "atk_code_exec_secret_read",
                "hard_arithmetic_sandbox",
            ],
        )
        for case in data["cases"]:
            self.assertIn("baseline_behavior", case)
            self.assertIn("guarded_behavior", case)
            self.assertIn("guard_decision", case)
            self.assertIn("enforcement", case)
            self.assertIn("self_healing_result", case)
            self.assertEqual(len(case["causal_graph"]["nodes"]), 7)
            self.assertEqual(len(case["causal_graph"]["edges"]), 6)

    def test_index_html_contains_title_and_case_ids(self) -> None:
        html = build_demo_evidence_viewer.render_html(self._build_data())

        self.assertIn("LogicGuard Evidence Viewer", html)
        for case_id in [
            "atk_prompt_file_email_01",
            "atk_tool_hijack_api_delete",
            "atk_code_exec_secret_read",
            "hard_arithmetic_sandbox",
        ]:
            self.assertIn(case_id, html)
        self.assertIn("not the full 40-case benchmark", html)

    def test_builder_does_not_call_real_api(self) -> None:
        source = inspect.getsource(build_demo_evidence_viewer)

        self.assertNotIn("DEEPSEEK_API_KEY", source)
        self.assertNotIn("httpx", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("run_deepseek_batch_experiment", source)
        self.assertNotIn("--mode real", source)

    def _build_data(self) -> dict:
        return build_demo_evidence_viewer.build_demo_data(
            json.loads(
                (PROJECT_ROOT / "outputs" / "demo_stage14_case_selection.json").read_text(
                    encoding="utf-8"
                )
            ),
            json.loads(
                (PROJECT_ROOT / "outputs" / "demo_stage14_causal_graph.json").read_text(
                    encoding="utf-8"
                )
            ),
            json.loads(
                (
                    PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium_summary.json"
                ).read_text(encoding="utf-8")
            ),
            json.loads(
                (
                    PROJECT_ROOT
                    / "outputs"
                    / "deepseek_batch_real_stage13_targeted_retest_summary.json"
                ).read_text(encoding="utf-8")
            ),
        )


if __name__ == "__main__":
    unittest.main()
