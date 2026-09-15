"""Validate and apply an architecture selected by search_architecture.py.

This module intentionally has no torch/numpy dependency so an exported search
result can be checked before a GPU training or evaluation run starts.
"""

import copy
import json
import math


_DEPTH_KEYS = (
    "sem_blocks_s4", "det_blocks_s4",
    "sem_blocks_s5", "det_blocks_s5",
    "sem_blocks_s6", "det_blocks_s6",
)
_REQUIRED_KEYS = {"dwsa_reduction", "ppm_channels", "dropout_ratio", *_DEPTH_KEYS}
_REDUCTION_CHOICES = {4, 8, 16, 32}
_CHANNEL_CHOICES = {64, 96, 128, 160, 192, 256}


def validate_arch_config(arch):
    """Return a normalized search config or raise on missing/invalid values."""
    if not isinstance(arch, dict):
        raise ValueError("architecture config must be a JSON object")
    missing = _REQUIRED_KEYS - arch.keys()
    extra = arch.keys() - _REQUIRED_KEYS
    if missing or extra:
        raise ValueError(f"architecture keys mismatch: missing={sorted(missing)}, "
                         f"unexpected={sorted(extra)}")

    normalized = {}
    for key, choices in (("dwsa_reduction", _REDUCTION_CHOICES),
                         ("ppm_channels", _CHANNEL_CHOICES)):
        value = arch[key]
        if type(value) is not int or value not in choices:
            raise ValueError(f"{key} must be one of {sorted(choices)}")
        normalized[key] = value
    for key in _DEPTH_KEYS:
        value = arch[key]
        if type(value) is not int or not 2 <= value <= 8:
            raise ValueError(f"{key} must be an integer from 2 through 8")
        normalized[key] = value

    dropout = arch["dropout_ratio"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise ValueError("dropout_ratio must be a number in [0, 0.3]")
    dropout = float(dropout)
    if not math.isfinite(dropout) or not 0 <= dropout <= 0.3:
        raise ValueError("dropout_ratio must be finite and in [0, 0.3]")
    normalized["dropout_ratio"] = dropout
    return normalized


def load_arch_json(path):
    """Read either search_architecture.py output or a bare best_config JSON."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "best_config" in payload:
        payload = payload["best_config"]
    return validate_arch_config(payload)


def apply_arch_config(base_cfg, arch):
    """Merge a validated candidate into ModelConfig.get_config() output."""
    arch = validate_arch_config(arch)
    cfg = copy.deepcopy(base_cfg)
    if "dwsa_reduction" not in cfg["backbone"]:
        raise ValueError("the selected architecture requires a DWSA model variant")
    cfg["backbone"]["dwsa_reduction"] = arch["dwsa_reduction"]
    cfg["backbone"]["ppm_channels"] = arch["ppm_channels"]
    blocks = cfg["backbone"]["num_blocks_per_stage"]
    blocks[2:] = [
        [arch["sem_blocks_s4"], arch["det_blocks_s4"]],
        [arch["sem_blocks_s5"], arch["det_blocks_s5"]],
        [arch["sem_blocks_s6"], arch["det_blocks_s6"]],
    ]
    cfg["head"]["dropout_ratio"] = arch["dropout_ratio"]
    return cfg
