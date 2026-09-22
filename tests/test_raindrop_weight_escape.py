"""CPU-only acceptance tests for the weight-space search guardrails."""

import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raindrop_weight_escape import (TARGETS, _stream_process, eligible,
                                    make_proxy_split)


class RaindropWeightEscapeTests(unittest.TestCase):
    def test_targets_are_existing_classifier_and_attention_weights(self):
        self.assertEqual(len(TARGETS), 5)
        self.assertIn("decode_head.cls_seg.weight", TARGETS)
        self.assertIn("decode_head.cls_seg.bias", TARGETS)
        self.assertEqual(sum("dwsa_stage" in name for name in TARGETS), 3)

    def test_winner_must_beat_original_and_matched_control(self):
        self.assertTrue(eligible(0.681, 0.6783, 0.679, True, 118,
                                 119.7, 0.02))
        self.assertFalse(eligible(0.6783, 0.6783, 0.677, True, 120,
                                  119.7, 0.02))
        self.assertFalse(eligible(0.679, 0.6783, 0.680, True, 120,
                                  119.7, 0.02))
        self.assertFalse(eligible(0.681, 0.6783, 0.679, False, 120,
                                  119.7, 0.02))
        self.assertFalse(eligible(0.681, 0.6783, 0.679, True, 115,
                                  119.7, 0.02))

    def test_proxy_search_and_gate_are_deterministic_and_disjoint(self):
        search, gate = make_proxy_split(1000, 512, 0.25, 42)
        repeated = make_proxy_split(1000, 512, 0.25, 42)
        self.assertEqual(len(search), 384)
        self.assertEqual(len(gate), 128)
        self.assertEqual(set(search).intersection(gate), set())
        self.assertEqual(search.tolist(), repeated[0].tolist())
        self.assertEqual(gate.tolist(), repeated[1].tolist())

    def test_train_output_is_visible_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            captured = StringIO()
            command = [sys.executable, "-u", "-c",
                       "import sys; print('epoch 1', end='\\r', flush=True); "
                       "print('epoch complete', flush=True)"]
            with redirect_stdout(captured):
                code = _stream_process(command, os.environ.copy(), Path(directory),
                                       log, time.monotonic() + 20, "unit")
            self.assertEqual(code, 0)
            self.assertIn("epoch complete", captured.getvalue(),
                          repr(log.read_text(encoding="utf-8")))
            self.assertIn("epoch complete", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
