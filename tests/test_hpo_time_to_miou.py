import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hpo_time_to_miou import (select_fastest, trial_candidates,
                              trial_training_args)


class HpoTimeToMiouTests(unittest.TestCase):
    def test_trials_change_fine_tuning_recipe_not_architecture(self):
        trials = trial_candidates()
        self.assertEqual(len(trials), 6)
        self.assertEqual(trials[0]["lr"], 1e-4)
        self.assertEqual(trials[0]["scheduler"], "cosine")
        self.assertEqual(trials[0]["weight_decay"], 1e-4)
        self.assertEqual(len({tuple(sorted(t.items())) for t in trials}), 6)
        for trial in trials:
            self.assertEqual(set(trial), {"lr", "backbone_lr_factor",
                                          "scheduler", "weight_decay", "scope",
                                          "lock_bn_stats", "aux_weight"})
            self.assertTrue(trial["lock_bn_stats"])
            self.assertNotIn("arch_json", trial)
            self.assertNotIn("img_h", trial)
        self.assertEqual({t["scheduler"] for t in trials}, {"cosine", "poly"})
        self.assertEqual({t["scope"] for t in trials},
                         {"all", "head_only", "attention_head"})

    def test_every_recorded_recipe_reaches_train_cli(self):
        for trial in trial_candidates():
            cli = trial_training_args(trial)
            self.assertIn("--lock_bn_stats", cli)
            self.assertEqual(cli[cli.index("--lr") + 1], str(trial["lr"]))
            self.assertEqual(cli[cli.index("--scheduler") + 1], trial["scheduler"])
            self.assertEqual(cli[cli.index("--weight_decay") + 1],
                             str(trial["weight_decay"]))
            self.assertEqual(cli[cli.index("--aux_weight") + 1],
                             str(trial["aux_weight"]))
            if trial["scope"] == "head_only":
                self.assertIn("--freeze_all_backbone", cli)
                self.assertNotIn("--freeze_backbone", cli)
                self.assertEqual(trial["aux_weight"], 0.0)
            elif trial["scope"] == "attention_head":
                self.assertIn("--freeze_backbone", cli)
                self.assertNotIn("--freeze_all_backbone", cli)
            else:
                self.assertNotIn("--freeze_backbone", cli)
                self.assertNotIn("--freeze_all_backbone", cli)

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
