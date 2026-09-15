import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from efficient_search import (baseline_arch, parse_gpu_ids,
                              parallel_proxy_batch, propose_architectures,
                              select_for_promotion)


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

    def test_gpu_ids_for_t4x2_are_distinct_and_bounded(self):
        self.assertEqual(parse_gpu_ids("0"), [0])
        self.assertEqual(parse_gpu_ids("0,1"), [0, 1])
        for bad in ("", "0,0", "0,1,2", "x,1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_gpu_ids(bad)

    def test_parallel_results_keep_candidate_order_and_stage1_weights(self):
        rows = [
            {"id": "baseline", "arch": {"a": 0}, "latency_ms": 8.35,
             "train_params": 21_000_000},
            {"id": 1, "arch": {"a": 1}, "latency_ms": 7.9,
             "train_params": 19_000_000},
            {"id": 2, "arch": {"a": 2}, "latency_ms": 7.7,
             "train_params": 18_000_000},
        ]
        seen = []

        def fake_task(stage, row, prior, root):
            seen.append((stage, row["id"], prior))
            directory = Path(root) / f"candidate_{row['id']}"
            directory.mkdir(parents=True, exist_ok=True)
            ckpt = directory / f"stage{stage}.pth"
            ckpt.write_bytes(b"fake checkpoint")
            return dict(row, stage=f"proxy{stage}", miou=0.5,
                        checkpoint=str(ckpt))

        with tempfile.TemporaryDirectory() as tmp:
            with (ThreadPoolExecutor(max_workers=1) as gpu0,
                  ThreadPoolExecutor(max_workers=1) as gpu1):
                pools = [gpu0, gpu1]
                log = Path(tmp) / "results.jsonl"
                first = parallel_proxy_batch(1, rows, [None] * 3, pools,
                                             tmp, log, 0, 1e9, fake_task)
                self.assertEqual([r["id"] for r in first],
                                 ["baseline", 1, 2])
                second = parallel_proxy_batch(
                    2, rows[:2], [first[0]["checkpoint"],
                                 first[1]["checkpoint"]], pools,
                    tmp, log, 0, 1e9, fake_task)
                self.assertEqual([r["id"] for r in second], ["baseline", 1])
                self.assertIn((2, "baseline", first[0]["checkpoint"]), seen)
                count = len(seen)
                cached = parallel_proxy_batch(1, rows, [None] * 3, pools,
                                              tmp, log, 0, 1e9, fake_task)
                self.assertEqual(len(seen), count)
                self.assertEqual([r["id"] for r in cached],
                                 ["baseline", 1, 2])


if __name__ == "__main__":
    unittest.main()
