"""Keep segmentation augmentation settings valid on Albumentations 1.x/2.x."""

import inspect


def _use_api(transform, modern_key, modern, legacy_key, legacy):
    params = inspect.signature(transform).parameters
    if modern_key in params:
        return modern
    if legacy_key in params:
        return legacy
    raise RuntimeError(
        f"Unsupported {transform.__name__} API: neither {modern_key!r} "
        f"nor {legacy_key!r} is accepted")


def pad_kwargs(transform):
    return _use_api(transform, "fill_mask", {"fill": 0, "fill_mask": 255},
                    "mask_value", {"value": 0, "mask_value": 255})


def grid_kwargs(transform):
    return _use_api(
        transform, "distort_range",
        {"distort_range": (-0.1, 0.1), "fill": 0, "fill_mask": 255},
        "distort_limit",
        {"distort_limit": 0.1, "value": 0, "mask_value": 255})


def dropout_kwargs(transform):
    return _use_api(
        transform, "num_holes_range",
        {"num_holes_range": (1, 4), "hole_height_range": (1, 24),
         "hole_width_range": (1, 24), "fill": 0, "fill_mask": 255},
        "max_holes",
        {"max_holes": 4, "max_height": 24, "max_width": 24,
         "fill_value": 0, "mask_fill_value": 255})


def fog_kwargs(transform):
    return _use_api(transform, "fog_coef_range",
                    {"fog_coef_range": (0.05, 0.30)},
                    "fog_coef_lower",
                    {"fog_coef_lower": 0.05, "fog_coef_upper": 0.30})


def compose_seed_kwargs(compose, seed):
    if seed is None:
        return {}
    if "seed" not in inspect.signature(compose).parameters:
        raise RuntimeError("Seeded HPO requires an Albumentations Compose API with seed support")
    return {"seed": seed}
