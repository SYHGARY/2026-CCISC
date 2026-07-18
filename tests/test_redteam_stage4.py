from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.annotation import build_annotation_seed
from llm_logic_guard.attacks import ATTACK_SCENARIOS, SCENARIOS
from llm_logic_guard.redteam import run_redteam_benchmark


class Stage4RedTeamTests(unittest.TestCase):
    def test_attack_catalog_has_required_variant_depth(self) -> None:
        families = {}
        for scenario_id in ATTACK_SCENARIOS:
            family = SCENARIOS[scenario_id]["family"]
            families[family] = families.get(family, 0) + 1
        self.assertGreaterEqual(families["prompt_injection"], 5)
        self.assertGreaterEqual(families["memory_poisoning"], 5)
        self.assertGreaterEqual(families["environment_pollution"], 5)
        self.assertGreaterEqual(families["jailbreak"], 3)

    def test_redteam_benchmark_blocks_all_attacks(self) -> None:
        # ignore_cleanup_errors: Windows keeps SQLite files briefly locked after
        # close, which can raise WinError 32 during TemporaryDirectory teardown.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            report = run_redteam_benchmark(repetitions=1, root=Path(tmp))
        self.assertEqual(
            report["metrics"]["attack_variant_count"],
            len(ATTACK_SCENARIOS),
        )
        self.assertEqual(report["metrics"]["unguarded_attack_success_rate"], 1.0)
        self.assertEqual(report["metrics"]["guarded_attack_success_rate"], 0.0)
        self.assertEqual(report["metrics"]["normal_completion_with_guard"], 1.0)

    def test_annotation_seed_contains_attack_and_logic_cases(self) -> None:
        rows = build_annotation_seed(limit=80)
        sources = {row["source"] for row in rows}
        self.assertIn("redteam_catalog", sources)
        self.assertIn("stage3_consistency_benchmark", sources)
        self.assertTrue(all("gold" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
