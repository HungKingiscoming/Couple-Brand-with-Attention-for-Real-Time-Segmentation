"""CPU-only tests for class-aware Raindrop loss search."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raindrop_loss_search import (BASELINE_X, WEAK_CLASSES,
                                  _split_training_file, decode_policy,
                                  policy_score)


class RaindropLossSearchTests(unittest.TestCase):
    def test_baseline_policy_matches_existing_loss(self):
        policy = decode_policy(BASELINE_X)
        self.assertEqual(policy["class_weights"], [1.0] * 19)
        self.assertEqual(policy["ohem_keep_ratio"], 0.3)
        self.assertEqual(policy["dice_weight"], 0.5)
        self.assertEqual(policy["aux_weight"], 0.4)

    def test_weights_are_positive_and_mean_normalized(self):
        policy = decode_policy([2, 0.5, 1.5, 1, 1.8, 0.4, 0.6, 0.2])
        weights = np.asarray(policy["class_weights"])
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_score_rewards_weak_classes_but_protects_overall_miou(self):
        base_iou = np.full(19, 0.6)
        baseline = {"exit_code": 0, "best_miou": 0.6,
                    "best_per_class_iou": base_iou.tolist()}
        improved = base_iou.copy()
        improved[list(WEAK_CLASSES)] += 0.05
        good = {"exit_code": 0, "best_miou": 0.601,
                "best_per_class_iou": improved.tolist()}
        bad = {"exit_code": 0, "best_miou": 0.59,
               "best_per_class_iou": improved.tolist()}
        self.assertGreater(policy_score(good, baseline),
                           policy_score(baseline, baseline))
        self.assertLess(policy_score(bad, baseline),
                        policy_score(baseline, baseline))

    def test_proxy_files_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "all.txt"
            source.write_text("\n".join(f"sample-{i}" for i in range(100)),
                              encoding="utf-8")
            train, gate = _split_training_file(source, root, 60, 20, 42)
            train_rows = set(train.read_text(encoding="utf-8").splitlines())
            gate_rows = set(gate.read_text(encoding="utf-8").splitlines())
            self.assertEqual(len(train_rows), 60)
            self.assertEqual(len(gate_rows), 20)
            self.assertFalse(train_rows.intersection(gate_rows))


if __name__ == "__main__":
    unittest.main()
