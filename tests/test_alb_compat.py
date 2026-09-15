import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alb_compat import (pad_kwargs, grid_kwargs, dropout_kwargs, fog_kwargs,
                        compose_seed_kwargs)


class AlbCompatTests(unittest.TestCase):
    def test_modern_parameters_preserve_ignore_label_and_fog_intensity(self):
        def pad(fill=0, fill_mask=0):
            pass

        def grid(distort_range=None, fill=0, fill_mask=0):
            pass

        def dropout(num_holes_range=None, fill=0, fill_mask=None):
            pass

        def fog(fog_coef_range=None):
            pass

        self.assertEqual(pad_kwargs(pad)["fill_mask"], 255)
        self.assertEqual(grid_kwargs(grid)["fill_mask"], 255)
        self.assertEqual(dropout_kwargs(dropout)["fill_mask"], 255)
        self.assertEqual(fog_kwargs(fog)["fog_coef_range"], (0.05, 0.30))

    def test_legacy_parameters_still_work(self):
        def pad(value=0, mask_value=0):
            pass

        def grid(distort_limit=0.0, value=0, mask_value=0):
            pass

        def dropout(max_holes=0, mask_fill_value=None):
            pass

        def fog(fog_coef_lower=0.0, fog_coef_upper=1.0):
            pass

        self.assertEqual(pad_kwargs(pad)["mask_value"], 255)
        self.assertEqual(grid_kwargs(grid)["mask_value"], 255)
        self.assertEqual(dropout_kwargs(dropout)["mask_fill_value"], 255)
        self.assertEqual(fog_kwargs(fog)["fog_coef_upper"], 0.30)

    def test_unknown_api_fails_instead_of_silently_using_defaults(self):
        def unknown(p=0.5):
            pass

        for adapter in (pad_kwargs, grid_kwargs, dropout_kwargs, fog_kwargs):
            with self.subTest(adapter=adapter), self.assertRaises(RuntimeError):
                adapter(unknown)

    def test_seeded_compose_needs_seed_support(self):
        def modern_compose(transforms, seed=None):
            pass

        def legacy_compose(transforms):
            pass

        self.assertEqual(compose_seed_kwargs(modern_compose, 42), {"seed": 42})
        self.assertEqual(compose_seed_kwargs(legacy_compose, None), {})
        with self.assertRaises(RuntimeError):
            compose_seed_kwargs(legacy_compose, 42)


if __name__ == "__main__":
    unittest.main()
