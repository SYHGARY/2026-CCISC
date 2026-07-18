from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_logic_guard.probabilistic import ConsistencyRiskMonitor
from llm_logic_guard.probabilistic_training import (
    generate_trajectory_dataset,
    load_dataset,
    prefix_episode,
    train_calibrate_evaluate,
    write_dataset,
)


class ProbabilisticMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.dataset_path = root / "trajectories.jsonl"
        self.model_path = root / "model.json"
        write_dataset(
            generate_trajectory_dataset(repetitions_per_family=20),
            self.dataset_path,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_training_builds_normalized_valid_transition_model(self) -> None:
        splits = load_dataset(self.dataset_path)
        monitor = ConsistencyRiskMonitor(self.model_path)
        model = monitor.train(splits["train"])
        self.assertEqual(model.version, "logicguard-dtmc.v2")
        self.assertTrue(model.transitions)
        self.assertGreater(model.required_state_visits, 0)
        self.assertEqual(
            model.pac_sufficient,
            model.training_summary["pac_sufficient"],
        )
        for source, targets in model.transitions.items():
            self.assertAlmostEqual(sum(targets.values()), 1.0)
            self.assertEqual(
                set(targets),
                set(model.valid_transitions[source]),
            )

    def test_calibration_is_saved_and_respects_fpr_constraint(self) -> None:
        report = train_calibrate_evaluate(
            dataset_path=self.dataset_path,
            model_path=self.model_path,
        )
        self.assertLessEqual(
            report["calibration"]["false_positive_rate"],
            0.10,
        )
        self.assertGreaterEqual(report["test"]["f1"], 0.80)
        saved = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.assertEqual(
            saved["threshold"],
            report["calibration"]["threshold"],
        )

    def test_attack_prefix_has_higher_risk_than_normal_prefix(self) -> None:
        splits = load_dataset(self.dataset_path)
        monitor = ConsistencyRiskMonitor(self.model_path)
        monitor.train(splits["train"])
        monitor.calibrate(splits["validation"])
        attack = next(trace for trace, unsafe in splits["test"] if unsafe)
        normal = next(trace for trace, unsafe in splits["test"] if not unsafe)
        attack_risk = monitor.assess(prefix_episode(attack)).pac_upper
        normal_risk = monitor.assess(prefix_episode(normal)).pac_upper
        self.assertGreater(attack_risk, normal_risk)
        self.assertTrue(monitor.assess(prefix_episode(attack)).warning)

    def test_loaded_model_reuses_calibrated_threshold(self) -> None:
        splits = load_dataset(self.dataset_path)
        monitor = ConsistencyRiskMonitor(self.model_path)
        monitor.train(splits["train"])
        calibration = monitor.calibrate(splits["validation"])
        loaded = ConsistencyRiskMonitor(self.model_path)
        self.assertEqual(loaded.threshold, calibration["threshold"])
        self.assertTrue(loaded.model and loaded.model.risk_cache)


if __name__ == "__main__":
    unittest.main()
