import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch_config import apply_arch_config, load_arch_json, validate_arch_config


WINNER = {
    "dwsa_reduction": 16,
    "ppm_channels": 160,
    "sem_blocks_s4": 5,
    "det_blocks_s4": 4,
    "sem_blocks_s5": 7,
    "det_blocks_s5": 5,
    "sem_blocks_s6": 5,
    "det_blocks_s6": 4,
    "dropout_ratio": 0.1293045071203046,
}


class ArchConfigTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "backbone": {
                "channels": 32,
                "dwsa_reduction": 8,
                "ppm_channels": 128,
                "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
            },
            "head": {"dropout_ratio": 0.1},
            "loss": {"dice_weight": 0.5},
        }

    def test_search_output_is_read_and_merged_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best_gcnet_arch.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"best_proxy_miou": 0.529617, "best_config": WINNER}, f)
            original = copy.deepcopy(self.base)
            cfg = apply_arch_config(self.base, load_arch_json(path))

        self.assertEqual(self.base, original)
        self.assertEqual(cfg["backbone"]["num_blocks_per_stage"],
                         [4, 4, [5, 4], [7, 5], [5, 4]])
        self.assertEqual(cfg["backbone"]["dwsa_reduction"], 16)
        self.assertEqual(cfg["backbone"]["ppm_channels"], 160)
        self.assertEqual(cfg["head"]["dropout_ratio"], WINNER["dropout_ratio"])

    def test_bad_values_fail_before_gpu_work(self):
        for key, value in (("dwsa_reduction", 3), ("ppm_channels", 72),
                           ("sem_blocks_s4", 0), ("dropout_ratio", float("nan"))):
            bad = dict(WINNER)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_arch_config(bad)

    def test_unexpected_keys_and_non_dwsa_variant_fail(self):
        bad = dict(WINNER, unknown=1)
        with self.assertRaises(ValueError):
            validate_arch_config(bad)
        no_dwsa = copy.deepcopy(self.base)
        del no_dwsa["backbone"]["dwsa_reduction"]
        with self.assertRaises(ValueError):
            apply_arch_config(no_dwsa, WINNER)


if __name__ == "__main__":
    unittest.main()
