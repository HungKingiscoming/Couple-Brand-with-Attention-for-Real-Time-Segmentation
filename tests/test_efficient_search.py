import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efficient_search import baseline_arch, propose_architectures, select_for_promotion


BASE_CFG = {
    "backbone": {
        "dwsa_reduction": 8, "ppm_channels": 128,
        "num_blocks_per_stage": [4, 4, [5, 4], [5, 4], [2, 2]],
    },
    "head": {"dropout_ratio": 0.1},
}


class EfficientSearchTests(unittest.TestCase):
    def test_baseline_and_candidates_never_expand_depth_or_stage6(self):
        baseline = baseline_arch(BASE_CFG)
        self.assertEqual(baseline["sem_blocks_s6"], 2)
        candidates = propose_architectures(baseline, 12, 42)
        self.assertEqual(len(candidates), 12)
        self.assertEqual(candidates, propose_architectures(baseline, 12, 42))
        for arch in candidates:
            self.assertEqual(arch["sem_blocks_s6"], 2)
            self.assertEqual(arch["det_blocks_s6"], 2)
            self.assertEqual(arch["dropout_ratio"], 0.1)
            for key in ("sem_blocks_s4", "det_blocks_s4",
                        "sem_blocks_s5", "det_blocks_s5"):
                self.assertLessEqual(arch[key], baseline[key])
            self.assertLessEqual(arch["ppm_channels"], 128)

    def test_promotion_uses_quality_floor_then_latency(self):
        rows = [
            {"id": 1, "miou": 0.50, "latency_ms": 7.1},
            {"id": 2, "miou": 0.49, "latency_ms": 6.8},
            {"id": 3, "miou": 0.44, "latency_ms": 5.0},
        ]
        selected = select_for_promotion(rows, 0.51, 2, 0.02)
        self.assertEqual([r["id"] for r in selected], [2, 1])


if __name__ == "__main__":
    unittest.main()
