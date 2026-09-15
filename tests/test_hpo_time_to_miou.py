import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo_time_to_miou import select_fastest, trial_candidates


class HpoTimeToMiouTests(unittest.TestCase):
    def test_trials_change_training_hyperparameters_not_architecture(self):
        trials = trial_candidates()
        self.assertGreaterEqual(len(trials), 4)
        self.assertEqual(trials[0], {"lr": 5e-4, "backbone_lr_factor": 0.1})
        self.assertEqual(len({tuple(sorted(t.items())) for t in trials}), len(trials))
        for trial in trials:
            self.assertEqual(set(trial), {"lr", "backbone_lr_factor"})

    def test_only_higher_iou_qualifies_then_fastest_wins(self):
        threshold = 0.6793
        rows = [
            {"id": 0, "target_reached": True, "best_miou": 0.6800,
             "time_to_target_sec": 900},
            {"id": 1, "target_reached": True, "best_miou": 0.6793,
             "time_to_target_sec": 100},
            {"id": 2, "target_reached": False, "best_miou": 0.7000,
             "time_to_target_sec": 50},
            {"id": 3, "target_reached": True, "best_miou": 0.6794,
             "time_to_target_sec": 600},
        ]
        self.assertEqual(select_fastest(rows, threshold)["id"], 3)
        self.assertIsNone(select_fastest(rows[1:3], threshold))


if __name__ == "__main__":
    unittest.main()
