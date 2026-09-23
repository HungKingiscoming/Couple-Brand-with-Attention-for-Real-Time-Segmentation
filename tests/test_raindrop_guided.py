"""CPU-only tests for Raindrop-guided training orchestration."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raindrop_guided import split_guide_indices
from run_raindrop_guided import _training_command


class RaindropGuidedTests(unittest.TestCase):
    def test_guide_split_is_deterministic_and_disjoint(self):
        gradient, gate = split_guide_indices(1000, 192, 1/3, 42)
        repeated = split_guide_indices(1000, 192, 1/3, 42)
        self.assertEqual(len(gradient), 128)
        self.assertEqual(len(gate), 64)
        self.assertFalse(set(gradient).intersection(gate))
        self.assertEqual(gradient.tolist(), repeated[0].tolist())
        self.assertEqual(gate.tolist(), repeated[1].tolist())

    def test_only_guided_arm_enables_raindrop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                checkpoint=root / "start.pth", train_txt=root / "train.txt",
                val_txt=root / "val.txt", img_h=512, img_w=1024,
                batch_size=16, workers_per_gpu=2, epochs=1, lr=1e-5,
                weight_decay=0.0, seed=42, guide_samples=192,
                guide_gate_fraction=1/3, guide_batch_size=4,
                gradient_batches=4, pop_size=4, max_iter=1,
                factor_min=0.5, factor_max=1.5, min_ce_gain=1e-4)
            common = (args, root / "arch.json", root / "summary.json",
                      root / "checkpoints")
            guided = _training_command(*common, True)
            control = _training_command(*common, False)
            self.assertIn("--raindrop_guided", guided)
            self.assertNotIn("--raindrop_guided", control)
            self.assertEqual(guided[:control.index("--hpo_summary_json")],
                             control[:control.index("--hpo_summary_json")])


if __name__ == "__main__":
    unittest.main()
