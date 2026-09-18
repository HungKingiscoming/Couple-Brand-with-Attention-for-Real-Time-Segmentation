"""CPU-only acceptance tests for the weight-space search guardrails."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from raindrop_weight_escape import TARGETS, eligible


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


if __name__ == "__main__":
    unittest.main()
