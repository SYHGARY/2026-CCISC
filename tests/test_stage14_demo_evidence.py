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

import build_demo_causal_graph


class Stage14DemoEvidenceTests(unittest.TestCase):
    def test_build_demo_outputs_reads_existing_reports(self) -> None:
        metadata = build_demo_causal_graph.load_case_metadata(
            PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
        )
        real_medium = build_demo_causal_graph.load_json(
            PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium.json"
        )
        stage13 = build_demo_causal_graph.load_json(
            PROJECT_ROOT / "outputs" / "deepseek_batch_real_stage13_targeted_retest.json"
        )
        real_medium_summary = build_demo_causal_graph.load_json(
            PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium_summary.json"
        )
        stage13_summary = build_demo_causal_graph.load_json(
            PROJECT_ROOT / "outputs" / "deepseek_batch_real_stage13_targeted_retest_summary.json"
        )

        outputs = build_demo_causal_graph.build_demo_outputs(
            metadata,
            real_medium,
            stage13,
            real_medium_summary,
            stage13_summary,
        )

        selection = outputs["selection"]
        self.assertEqual(selection["selection_count"], 4)
        self.assertEqual(
            [case["case_id"] for case in selection["cases"]],
            [
                "atk_prompt_file_email_01",
                "atk_tool_hijack_api_delete",
                "atk_code_exec_secret_read",
                "hard_arithmetic_sandbox",
            ],
        )
        by_id = {case["case_id"]: case for case in selection["cases"]}
        self.assertEqual(
            by_id["atk_code_exec_secret_read"]["source_report"],
            "stage13_targeted_real_retest",
        )
        self.assertEqual(
            by_id["hard_arithmetic_sandbox"]["source_report"],
            "stage13_targeted_real_retest",
        )

    def test_causal_graph_contains_required_nodes_and_edges(self) -> None:
        outputs = self._build_outputs()
        graph = outputs["causal_graph"]

        self.assertEqual(graph["graph_count"], 4)
        required_types = set(graph["required_node_types"])
        for case_graph in graph["graphs"]:
            node_types = {node["type"] for node in case_graph["nodes"]}
            self.assertTrue(required_types.issubset(node_types))
            self.assertGreaterEqual(len(case_graph["edges"]), 6)
            self.assertTrue(
                any(edge["from"].endswith(":user_input") for edge in case_graph["edges"])
            )
            self.assertTrue(
                any(edge["to"].endswith(":final_outcome") for edge in case_graph["edges"])
            )

    def test_write_outputs_creates_selection_graph_and_evidence_files(self) -> None:
        outputs = self._build_outputs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            selection_json = tmp_path / "selection.json"
            selection_md = tmp_path / "selection.md"
            graph_json = tmp_path / "graph.json"
            graph_md = tmp_path / "graph.md"
            evidence_md = tmp_path / "evidence.md"

            build_demo_causal_graph.write_outputs(
                outputs["selection"],
                outputs["causal_graph"],
                outputs["evidence_markdown"],
                selection_json,
                selection_md,
                graph_json,
                graph_md,
                evidence_md,
            )

            self.assertTrue(selection_json.exists())
            self.assertTrue(selection_md.exists())
            self.assertTrue(graph_json.exists())
            self.assertTrue(graph_md.exists())
            self.assertTrue(evidence_md.exists())
            self.assertIn("nodes", graph_json.read_text(encoding="utf-8"))
            self.assertIn("Stage 14 Demo Evidence Pack", evidence_md.read_text(encoding="utf-8"))

    def test_builder_does_not_use_real_api_key(self) -> None:
        source = inspect.getsource(build_demo_causal_graph)
        self.assertNotIn("DEEPSEEK_API_KEY", source)
        self.assertNotIn("httpx", source)
        self.assertNotIn("DeepSeek", source.split("description=")[0])

    def _build_outputs(self) -> dict:
        metadata = build_demo_causal_graph.load_case_metadata(
            PROJECT_ROOT / "data" / "deepseek_batch_cases.json"
        )
        return build_demo_causal_graph.build_demo_outputs(
            metadata,
            build_demo_causal_graph.load_json(
                PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium.json"
            ),
            build_demo_causal_graph.load_json(
                PROJECT_ROOT / "outputs" / "deepseek_batch_real_stage13_targeted_retest.json"
            ),
            build_demo_causal_graph.load_json(
                PROJECT_ROOT / "outputs" / "deepseek_batch_real_medium_summary.json"
            ),
            build_demo_causal_graph.load_json(
                PROJECT_ROOT
                / "outputs"
                / "deepseek_batch_real_stage13_targeted_retest_summary.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
