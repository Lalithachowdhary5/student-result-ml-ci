import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class TestMLPipeline(unittest.TestCase):
    def test_training_pipeline_generates_expected_outputs(self):
        subprocess.run(
            [sys.executable, "train_model.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertTrue((ROOT / "student_results.csv").exists())
        self.assertTrue((ROOT / "student_result_model.pkl").exists())
        self.assertTrue((ROOT / "metrics.json").exists())

        with open(ROOT / "metrics.json", "r", encoding="utf-8") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.85)

    def test_quality_gate_passes_when_accuracy_meets_threshold(self):
        subprocess.run(
            [sys.executable, "train_model.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        result = subprocess.run(
            [sys.executable, "quality_gate.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("QUALITY GATE PASSED", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
