"""
Search-space definition for applying ADE-RD to GCNet's architecture
hyperparameters (dwsa_reduction, ppm_channels, semantic/detail branch
depths at stage 4/5/6), as agreed:

  - The optimizer (ADE-RD) always works on a plain continuous vector
    x in [lb, ub]^d — this is what raindrop.py / ade_rd.py expect.
  - Each dimension is decoded into a valid integer / categorical value
    ONLY at evaluation time (i.e. right before building the model),
    exactly the way Le-Xuan Thang et al.'s EEFO paper handles their
    F1-F5 integer hyperparameters: search continuously, round once at
    the end of each candidate's evaluation.
  - `channels` (C) and the stem block counts are intentionally NOT part
    of the search space: changing them invalidates the pretrained
    backbone weights (see the shape-matching logic in
    load_pretrained_gcnet), which would turn every candidate evaluation
    into a from-scratch training run — far too expensive on Kaggle.

This file defines the *decoding*, a *config merge*, and a working
*fitness function* (`make_fitness_function`) that trains each candidate
via train.py's Segmentor/Trainer/load_pretrained_gcnet on a fixed random subset
of the given train/val split. `torch`/`train.py` imports are deferred to
inside `make_fitness_function`'s inner `fitness()`, so importing this
module itself (and using decode_candidate/build_model_config) never
requires torch or a GPU.
"""

import json
import os
import time

import numpy as np


# --------------------------------------------------------------------- #
# Search space definition
# --------------------------------------------------------------------- #
# Each entry: (name, low, high, kind, choices)
#   kind = "int"    -> round to nearest integer in [low, high]
#   kind = "choice" -> round to nearest INDEX in [low, high] = [0, len(choices)-1],
#                      then look up `choices[index]`
#   kind = "float"  -> continuous, used as-is (no rounding)
#
# `low`/`high` here are each parameter's true valid range -- NOT the
# search bounds the optimizer sees (those are widened by +/-0.5 below,
# see `_search_bounds`).
#
# "choice" decodes by nearest INDEX rather than nearest VALUE. The two
# discrete sets below are unevenly spaced (_REDUCTION_CHOICES has gaps of
# 4/8/16; _CHANNEL_CHOICES has gaps of 32 except a final gap of 64), so
# snapping a continuous x to the nearest *value* would implicitly hand
# whichever choice sits in the widest gap a proportionally larger capture
# region -- e.g. under nearest-value snapping, dwsa_reduction=32 would
# capture 4x more of the search space than dwsa_reduction=4 purely as an
# artifact of the encoding, not because 32 is actually a better prior.
# Snapping by index instead gives every discrete choice an equal-width
# capture region regardless of how the underlying values are spaced.
SEARCH_SPACE = [
    ("dwsa_reduction",     0,   3,   "choice", None),  # index into _REDUCTION_CHOICES
    ("ppm_channels",       0,   5,   "choice", None),  # index into _CHANNEL_CHOICES
    ("sem_blocks_s4",      2,   8,   "int",    None),  # semantic branch, stage 4
    ("det_blocks_s4",      2,   8,   "int",    None),  # detail   branch, stage 4
    ("sem_blocks_s5",      2,   8,   "int",    None),
    ("det_blocks_s5",      2,   8,   "int",    None),
    ("sem_blocks_s6",      2,   8,   "int",    None),
    ("det_blocks_s6",      2,   8,   "int",    None),
    ("dropout_ratio",      0.0, 0.3, "float",  None),  # continuous, no rounding needed
]

DIM = len(SEARCH_SPACE)

_REDUCTION_CHOICES = np.array([4, 8, 16, 32])
_CHANNEL_CHOICES = np.array([64, 96, 128, 160, 192, 256])
_CHOICES_BY_NAME = {
    "dwsa_reduction": _REDUCTION_CHOICES,
    "ppm_channels": _CHANNEL_CHOICES,
}


def _search_bounds(low, high, kind):
    """
    The optimizer searches a *widened* continuous range for discrete
    ("int"/"choice") dimensions: +/-0.5 around the true [low, high], so
    that round-to-nearest decoding gives every valid integer/index an
    equal-width capture region -- including the two boundary values,
    which would otherwise only get half the capture width of interior
    values (e.g. only x in [2.0, 2.5) rounds to 2, width 0.5, vs
    [2.5, 3.5) width 1.0 for 3).
    """
    if kind in ("int", "choice"):
        return low - 0.5, high + 0.5
    return low, high


LB = np.array([_search_bounds(lo, hi, kind)[0] for _, lo, hi, kind, _ in SEARCH_SPACE])
UB = np.array([_search_bounds(lo, hi, kind)[1] for _, lo, hi, kind, _ in SEARCH_SPACE])


def decode_candidate(x):
    """
    Decode one continuous search vector x (shape (DIM,)) into a config dict
    ready to be merged into ModelConfig.get_config()'s "backbone"/"head"
    sub-dicts. Rounding/snapping happens here, once, at evaluation time —
    the optimizer itself never sees anything but continuous numbers.
    """
    x = np.clip(x, LB, UB)
    cfg = {}
    for (name, lo, hi, kind, _), val in zip(SEARCH_SPACE, x):
        if kind in ("int", "choice"):
            snapped = int(np.clip(round(val), lo, hi))
            cfg[name] = int(_CHOICES_BY_NAME[name][snapped]) if kind == "choice" else snapped
        else:  # "float", used as-is
            cfg[name] = float(val)
    return cfg


def build_model_config(x, base_cfg):
    """
    Merge a decoded candidate into a full ModelConfig-style dict (as
    produced by train.py's ModelConfig.get_config()), without touching
    `channels` or the stem block counts (kept fixed to stay compatible
    with the pretrained backbone).
    """
    cand = decode_candidate(x)
    cfg = {
        "backbone": dict(base_cfg["backbone"]),
        "head":     dict(base_cfg["head"]),
        "loss":     dict(base_cfg["loss"]),
    }
    cfg["backbone"]["ppm_channels"]   = cand["ppm_channels"]
    cfg["backbone"]["dwsa_reduction"] = cand["dwsa_reduction"]

    # num_blocks_per_stage = [stem2, stem3, [sem_s4, det_s4], [sem_s5, det_s5], [sem_s6, det_s6]]
    # Only indices 2, 3, 4 (stage 4/5/6, the newly-added branches) are
    # touched; stem depths (indices 0, 1) are left exactly as in base_cfg.
    orig = base_cfg["backbone"]["num_blocks_per_stage"]
    cfg["backbone"]["num_blocks_per_stage"] = [
        orig[0],
        orig[1],
        [cand["sem_blocks_s4"], cand["det_blocks_s4"]],
        [cand["sem_blocks_s5"], cand["det_blocks_s5"]],
        [cand["sem_blocks_s6"], cand["det_blocks_s6"]],
    ]

    cfg["head"]["dropout_ratio"] = cand["dropout_ratio"]
    return cfg


# --------------------------------------------------------------------- #
# Fitness function scaffold
# --------------------------------------------------------------------- #

def _make_subset_txt(src_txt, fraction, out_path, seed):
    """
    Write a random `fraction` of the lines of `src_txt` to `out_path`, so
    proxy training/validation runs on a cheap subset instead of the full
    Cityscapes split. The caller creates this once and reuses it for every
    candidate, which makes comparisons fair and avoids repeatedly starting
    DataLoader workers. Re-evaluate finalists with another seed/full proxy
    to guard against overfitting this screening subset.
    """
    import random
    with open(src_txt, "r") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        # Fail here, immediately and clearly -- otherwise this silently
        # writes an empty subset file, and the real error only surfaces
        # much later as a cryptic "num_samples should be a positive
        # integer" from deep inside torch's DataLoader/RandomSampler,
        # nowhere near the actual cause (an empty/wrong --train_txt or
        # --val_txt path).
        raise ValueError(
            f"{src_txt!r} has no valid (non-blank) lines -- double-check "
            "that this path points to the correct train/val list file on "
            "Kaggle and that the file isn't empty."
        )
    rng = random.Random(seed)
    rng.shuffle(lines)
    n = max(1, int(round(len(lines) * fraction)))
    with open(out_path, "w") as f:
        f.writelines(lines[:n])


def _build_proxy_args(loss_config, proxy_epochs, save_dir,
                       num_classes, ignore_index, batch_size, lr,
                       fast_proxy=False):
    """
    Minimal args namespace covering every attribute Trainer/build_optimizer
    in train.py actually read (checked directly against train.py's source:
    Trainer.__init__/.train_epoch/.validate and build_optimizer). Fields
    with a `getattr(args, name, default)` call site in train.py are left
    out here and simply fall back to that default; everything else is
    required and set explicitly.
    """
    import types
    args = types.SimpleNamespace()
    args.loss_config = dict(loss_config)
    if fast_proxy:
        args.loss_config["dice_weight"] = 0.0
    args.ignore_index = ignore_index
    args.num_classes = num_classes
    args.use_amp = True
    args.save_dir = str(save_dir)
    args.batch_size = batch_size
    args.accumulation_steps = 1
    args.grad_clip = 5.0
    args.epochs = proxy_epochs
    args.aux_weight = 0.0 if fast_proxy else 0.4
    args.save_interval = proxy_epochs + 1  # never trigger train.py's periodic checkpoint
    args.lr = lr
    args.weight_decay = 1e-4
    args.backbone_lr_factor = 0.1
    args.dwsa_lr_factor = 0.5
    args.alpha_lr_factor = 0.1
    # Proxy runs prioritize throughput and do not need per-batch diagnostic
    # synchronization or one TensorBoard writer per candidate.
    args.gradient_check_interval = 0
    args.progress_interval = 50
    args.empty_cache_interval = 0
    args.enable_tensorboard = False
    return args


def make_fitness_function(base_cfg,
                           pretrained_weights_path,
                           train_txt,
                           val_txt,
                           proxy_epochs=4,
                           proxy_data_fraction=0.2,
                           num_classes=19,
                           batch_size=4,
                           lr=5e-4,
                           img_size=(512, 1024),
                           dataset_type="foggy",
                           device="cuda",
                           log_path=None,
                           proxy_num_workers=2,
                           proxy_seed=42,
                           fast_proxy=False):
    """
    Returns a `fitness(X)` function compatible with RaindropOptimizer /
    ADERaindropOptimizer's `obj_func` signature: takes an (N, d) array of
    candidates, returns an (N,) array of COSTS to MINIMIZE (so we return
    1 - mIoU, since RD/ADE-RD minimize by convention).

    Each candidate is trained for `proxy_epochs` epochs on one fixed random
    `proxy_data_fraction` subset of train_txt/val_txt, using the same
    Segmentor/Trainer/load_pretrained_gcnet building blocks as a full
    train.py run. `torch`/`train.py`/model imports are deferred to inside
    this function, so importing gcnet_search_space.py itself never
    requires torch (decode_candidate/build_model_config stay usable, and
    testable, without a GPU or even torch installed).

    If `log_path` is given, every candidate's result is appended to it as
    one JSON line (JSONL) the moment that candidate finishes evaluating --
    a proxy search can run for many hours across `pop_size * (max_iter+2)`
    candidates with nothing else saved until the very end, so if the
    process is killed partway (Kaggle session limit, disconnect, etc.)
    this log is the only way to recover which architectures were already
    tried and how they did. Safe to `tail -f` while the search is running.

    NOTE: this function trains N models per call (once per candidate row).
    Candidates within a call cannot share one GPU training run because
    each has a different architecture (different tensor shapes) — see the
    earlier discussion on why weight-sharing/supernets would be needed to
    avoid this, which was intentionally scoped out for this project.
    """

    if not 0 < proxy_data_fraction <= 1:
        raise ValueError("proxy_data_fraction must be in (0, 1]")
    if proxy_epochs < 1:
        raise ValueError("proxy_epochs must be at least 1")
    if batch_size < 1 or proxy_num_workers < 0:
        raise ValueError("batch_size must be positive and proxy_num_workers non-negative")

    def _file_identity(path):
        absolute = os.path.abspath(os.fspath(path))
        try:
            stat = os.stat(absolute)
            return {"path": absolute, "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns}
        except OSError:
            return {"path": absolute}

    proxy_signature = {
        "proxy_epochs": proxy_epochs,
        "proxy_data_fraction": proxy_data_fraction,
        "batch_size": batch_size,
        "lr": lr,
        "img_size": list(img_size),
        "dataset_type": dataset_type,
        "num_classes": num_classes,
        "fast_proxy": fast_proxy,
        "seed": proxy_seed,
        "checkpoint": _file_identity(pretrained_weights_path),
        "train_list": _file_identity(train_txt),
        "val_list": _file_identity(val_txt),
    }
    signature_json = json.dumps(proxy_signature, sort_keys=True)
    result_cache = {}
    shared = {}

    # Recover compatible completed evaluations when resuming a killed run.
    if log_path and os.path.isfile(log_path):
        with open(log_path, "r") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if json.dumps(row.get("proxy", {}), sort_keys=True) != signature_json:
                        continue
                    key = json.dumps(row["config"], sort_keys=True)
                    result_cache[key] = float(row["cost"])
                except (ValueError, KeyError, TypeError):
                    continue

    def _ensure_shared_resources(torch, create_dataloaders):
        if shared:
            return

        import atexit
        import shutil
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="ade_rd_proxy_shared_")
        atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
        train_txt_sub = os.path.join(tmp_dir, "train_subset.txt")
        val_txt_sub = os.path.join(tmp_dir, "val_subset.txt")
        _make_subset_txt(train_txt, proxy_data_fraction, train_txt_sub, seed=proxy_seed)
        _make_subset_txt(val_txt, proxy_data_fraction, val_txt_sub, seed=proxy_seed)

        train_loader, val_loader, _ = create_dataloaders(
            train_txt=train_txt_sub, val_txt=val_txt_sub,
            batch_size=batch_size, num_workers=proxy_num_workers,
            img_size=img_size, pin_memory=True,
            compute_class_weights=False, dataset_type=dataset_type,
            persistent_workers=proxy_num_workers > 0,
            prefetch_factor=2)

        # This file is identical for every candidate; deserialize it once.
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu",
                                weights_only=False)
        shared.update(tmp_dir=tmp_dir, train_loader=train_loader,
                      val_loader=val_loader, checkpoint=checkpoint)

    def fitness(X):
        import gc
        import tempfile

        import torch

        from model.backbone.model import GCNet
        from model.head.segmentation_head import GCNetHead
        from train import (Segmentor, Trainer, load_pretrained_gcnet,
                            build_optimizer)
        from data.custom import create_dataloaders

        _ensure_shared_resources(torch, create_dataloaders)
        costs = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            cfg = build_model_config(x, base_cfg)
            cfg_candidate_for_log = decode_candidate(x)
            cache_key = json.dumps(cfg_candidate_for_log, sort_keys=True)
            if cache_key in result_cache:
                costs[i] = result_cache[cache_key]
                print(f"Reusing cached candidate: mIoU={1.0-costs[i]:.4f}  "
                      f"config={cfg_candidate_for_log}")
                continue

            tmp_dir = tempfile.mkdtemp(prefix="ade_rd_proxy_")
            try:
                # Use the same initialization seed and data subset so fitness
                # differences primarily reflect architecture, not sampling noise.
                torch.manual_seed(proxy_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(proxy_seed)
                model = Segmentor(
                    GCNet(**cfg["backbone"]),
                    GCNetHead(**cfg["head"], num_classes=num_classes)
                ).to(device)
                # Shape mismatches from the changed stage-4/5/6 depths &
                # ppm_channels are skipped automatically by this function's
                # own shape check -- see train.py's load_pretrained_gcnet.
                load_pct = load_pretrained_gcnet(model, shared["checkpoint"])
                if load_pct < 1.0:
                    # stem_conv1/conv2/stage2/stage3 are never touched by
                    # the search, so a working load should ALWAYS match at
                    # least that much regardless of what stage-4/5/6 depths
                    # this candidate uses -- a near-0% match means the
                    # checkpoint didn't load at all (e.g. an unrecognized
                    # top-level key), which would silently turn every
                    # candidate's "proxy fine-tuning" into training from
                    # random init instead, making the whole search's cost
                    # signal meaningless. Fail loudly instead of burning
                    # GPU time on a broken proxy.
                    raise RuntimeError(
                        f"load_pretrained_gcnet matched only {load_pct:.2f}% of "
                        f"parameters from {pretrained_weights_path!r} -- this looks "
                        "like the checkpoint failed to load rather than a normal "
                        "architecture mismatch. Check the checkpoint's top-level "
                        "key (must be 'model', 'model_state_dict', 'state_dict', "
                        "or a flat state_dict) before continuing the search."
                    )

                proxy_args = _build_proxy_args(
                    loss_config=base_cfg["loss"], proxy_epochs=proxy_epochs,
                    save_dir=tmp_dir, num_classes=num_classes,
                    ignore_index=255, batch_size=batch_size, lr=lr,
                    fast_proxy=fast_proxy)
                optimizer = build_optimizer(model, proxy_args)
                trainer = Trainer(model=model, optimizer=optimizer, scheduler=None,
                                   device=device, args=proxy_args)

                for epoch in range(proxy_epochs):
                    trainer.train_epoch(shared["train_loader"], epoch)
                val_metrics = trainer.validate(shared["val_loader"], proxy_epochs - 1)
                costs[i] = 1.0 - val_metrics["miou"]
                result_cache[cache_key] = float(costs[i])

                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "config": cfg_candidate_for_log,
                            "miou": val_metrics["miou"],
                            "cost": float(costs[i]),
                            "backbone_load_pct": load_pct,
                            "proxy": proxy_signature,
                        }) + "\n")

                trainer.writer.close()
                del model, trainer, optimizer
            finally:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
                gc.collect()
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()

        return costs

    return fitness
