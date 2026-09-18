import os
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obl_de_rd import OBLAdaptiveRaindropOptimizer
from raindrop_hpo import LOW, HIGH, decode_candidate, proxy_cost
from hpo_time_to_miou import trial_training_args
import raindrop_hpo


class RaindropHpoTests(unittest.TestCase):
    def test_decode_keeps_model_fixed_and_produces_train_cli(self):
        for vector in (
                [-6, -0.5, -0.5, -0.5],
                [-4, 2.5, 1.5, 1.5],
                [-4.3, 1, 0, 1]):
            recipe = decode_candidate(vector)
            self.assertGreaterEqual(recipe["lr"], 1e-6)
            self.assertLessEqual(recipe["lr"], 1e-4)
            self.assertIn(recipe["scheduler"], ("cosine", "poly"))
            self.assertIn(recipe["scope"], ("head_only", "attention_head"))
            self.assertIn("--lock_bn_stats", trial_training_args(recipe))
            self.assertNotIn("arch_json", recipe)
        with self.assertRaises(ValueError):
            decode_candidate([0, 1, 2])

    def test_proxy_cost_is_not_flat_when_every_trial_misses_target(self):
        threshold = 0.6783018947437217
        def row(miou):
            return {"exit_code": 0, "best_miou": miou,
                    "elapsed_wall_sec": 2700}
        self.assertLess(proxy_cost(row(0.6774), threshold),
                        proxy_cost(row(0.6750), threshold))
        self.assertLess(proxy_cost(row(0.6784), threshold),
                        proxy_cost(row(0.6774), threshold))
        self.assertEqual(proxy_cost({"exit_code": 1, "best_miou": None},
                                    threshold), 1e6)

    def test_actual_raindrop_optimizer_calls_fitness(self):
        calls = []
        def fitness(population):
            calls.append(population.shape)
            return np.sum((population - np.array([-4.7, 1, 0, 1])) ** 2,
                          axis=1)
        optimizer = OBLAdaptiveRaindropOptimizer(
            obj_func=fitness, dim=4, lb=LOW, ub=HIGH,
            pop_size=4, max_iter=1, seed=42)
        best_x, best_cost = optimizer.optimize()
        self.assertEqual(best_x.shape, (4,))
        self.assertTrue(np.isfinite(best_cost))
        self.assertGreaterEqual(len(calls), 2)
        self.assertIn((8, 4), calls)  # OBL evaluates each initial opposite.

    def test_launcher_feeds_two_gpu_proxy_results_to_raindrop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("checkpoint.pth", "train.txt", "val.txt"):
                (root / name).write_bytes(b"fixture")
            seen = []

            def fake_run(params, stage, gpu_id, args, work_root,
                         threshold, deadline):
                seen.append((stage, gpu_id, params))
                return {"params": params, "stage": stage, "gpu_id": gpu_id,
                        "exit_code": 0, "elapsed_wall_sec": 2500,
                        "best_miou": 0.676 + params["lr"],
                        "target_reached": False,
                        "time_to_target_sec": None}

            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(device_count=lambda: 2))
            output_path = root / "result.json"
            argv = ["raindrop_hpo.py", "--checkpoint", str(root / "checkpoint.pth"),
                    "--train_txt", str(root / "train.txt"),
                    "--val_txt", str(root / "val.txt"),
                    "--pop_size", "4", "--max_iter", "1",
                    "--max_hours", "4", "--proxy_hours", "1",
                    "--work_dir", str(root / "runs"),
                    "--out_json", str(output_path)]
            with (mock.patch.object(raindrop_hpo, "_run_candidate", fake_run),
                  mock.patch.dict(sys.modules, {"torch": fake_torch}),
                  mock.patch.object(sys, "argv", argv)):
                raindrop_hpo.main()
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIsNone(result["best_candidate"])
            self.assertEqual(result["algorithm"], "OBL-ADE-RD")
            self.assertTrue(any(stage == "proxy" for stage, _, _ in seen))
            self.assertEqual(sum(stage == "full" for stage, _, _ in seen), 2)
            self.assertEqual({gpu for _, gpu, _ in seen}, {0, 1})


if __name__ == "__main__":
    unittest.main()
