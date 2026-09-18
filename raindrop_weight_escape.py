"""Gradient -> bounded Raindrop weight escape -> gradient, with a matched control.

Raindrop searches ten coefficients, not every GCNet weight. Its fitness is
measured on a fixed, unaugmented subset of *training* images; the full
validation split is untouched until the two matched fine-tunes finish.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obl_de_rd import OBLAdaptiveRaindropOptimizer


TARGETS = (
    "decode_head.cls_seg.weight",
    "decode_head.cls_seg.bias",
    "backbone.dwsa_stage4.out_proj.weight",
    "backbone.dwsa_stage5.out_proj.weight",
    "backbone.dwsa_stage6.out_proj.weight",
)


def eligible(miou, baseline, control, params_equal, fps, baseline_fps,
             fps_tolerance):
    """Only the final, fused model can qualify; proxy scores never qualify."""
    return (math.isfinite(miou) and miou > max(baseline, control)
            and params_equal and math.isfinite(fps)
            and fps >= baseline_fps * (1 - fps_tolerance))


class BoundedDirections:
    """Fixed two-direction basis per tensor; all candidate edits are reversible."""

    def __init__(self, model, seed, relative_scale):
        import torch

        if not 0 < relative_scale <= 0.1:
            raise ValueError("relative_scale must be in (0, 0.1]")
        named = dict(model.named_parameters())
        missing = [name for name in TARGETS if name not in named]
        if missing:
            raise ValueError(f"Checkpoint model lacks target tensors: {missing}")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.groups = []
        self.dim = 2 * len(TARGETS)
        for name in TARGETS:
            param = named[name]
            base = param.detach().clone()
            rms = max(float(base.float().square().mean().sqrt()), 1e-3)
            directions = []
            for _ in range(2):
                direction = torch.randn(param.shape, generator=rng)
                direction /= direction.square().mean().sqrt().clamp_min(1e-8)
                directions.append(direction.to(param.device, dtype=param.dtype)
                                  * (relative_scale * rms))
            self.groups.append((param, base, directions))

    def apply(self, x):
        import torch

        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,) or not np.isfinite(x).all() or (abs(x) > 1).any():
            raise ValueError("Expected finite coefficients in [-1, 1]")
        with torch.no_grad():
            for group, (param, base, directions) in enumerate(self.groups):
                param.copy_(base + float(x[2 * group]) * directions[0]
                            + float(x[2 * group + 1]) * directions[1])

    def restore(self):
        import torch

        with torch.no_grad():
            for param, base, _ in self.groups:
                param.copy_(base)


def proxy_metrics(model, loader, device, num_classes=19):
    import torch
    import torch.nn.functional as F

    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64,
                            device=device)
    loss_sum = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            if labels.ndim == 4:
                labels = labels.squeeze(1)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                logits = F.interpolate(logits, size=labels.shape[-2:],
                                       mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, labels, ignore_index=255)
            prediction = logits.argmax(1)
            valid = (labels >= 0) & (labels < num_classes)
            cells = num_classes * labels[valid] + prediction[valid]
            confusion += torch.bincount(cells, minlength=num_classes ** 2).reshape(
                num_classes, num_classes)
            loss_sum += float(loss)
    true_positive = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - true_positive
    miou = float((true_positive / union.clamp_min(1)).mean())
    return miou, loss_sum / len(loader)


def search_perturbation(model, loader, device, args):
    basis = BoundedDirections(model, args.seed, args.relative_scale)
    baseline_miou, baseline_ce = proxy_metrics(model, loader, device)
    observations = {}

    def fitness(population):
        scores = []
        for vector in population:
            key = tuple(np.asarray(vector, dtype=float).round(8))
            if key not in observations:
                try:
                    basis.apply(vector)
                    miou, ce = proxy_metrics(model, loader, device)
                    # mIoU is primary; CE and perturbation size discourage
                    # large changes that only fit the small proxy split.
                    cost = -miou + 0.01 * max(0.0, ce - baseline_ce)
                    cost += 1e-4 * float(np.square(vector).mean())
                    observations[key] = {"proxy_miou": miou, "proxy_ce": ce,
                                         "cost": cost}
                finally:
                    basis.restore()
            scores.append(observations[key]["cost"])
        return np.asarray(scores, dtype=float)

    optimizer = OBLAdaptiveRaindropOptimizer(
        obj_func=fitness, dim=basis.dim,
        lb=-np.ones(basis.dim), ub=np.ones(basis.dim),
        pop_size=args.pop_size, max_iter=args.max_iter, seed=args.seed)
    best_x, _ = optimizer.optimize(verbose=True)
    best = observations[tuple(np.asarray(best_x).round(8))]
    accepted = (best["proxy_miou"] > baseline_miou + args.proxy_min_gain
                and best["proxy_ce"] <= baseline_ce * (1 + args.max_ce_increase))
    if accepted:
        basis.apply(best_x)
    return {"baseline_proxy_miou": baseline_miou,
            "baseline_proxy_ce": baseline_ce,
            "best_proxy": best, "coefficients": np.asarray(best_x).tolist(),
            "accepted_proxy": bool(accepted),
            "evaluations": len(observations),
            "history_best_cost": [float(v) for v in optimizer.history_best]}


def _train(checkpoint, gpu, args, out_dir, deadline):
    out_dir.mkdir(parents=True)
    summary = out_dir / "summary.json"
    log = out_dir / "train.log"
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
           "--pretrained_weights", str(checkpoint),
           "--arch_json", str(args.arch_json),
           "--train_txt", str(Path(args.train_txt).resolve()),
           "--val_txt", str(Path(args.val_txt).resolve()),
           "--model_variant", "fan_dwsa", "--dataset_type", "foggy",
           "--num_classes", "19", "--img_h", str(args.img_h),
           "--img_w", str(args.img_w), "--batch_size", str(args.batch_size),
           "--num_workers", str(args.workers_per_gpu), "--persistent_workers",
           "--epochs", str(args.epochs), "--freeze_backbone", "--lock_bn_stats",
           "--lr", str(args.lr), "--weight_decay", str(args.weight_decay),
           "--scheduler", "cosine", "--aux_weight", "0.4",
           "--gradient_check_interval", "0", "--seed", str(args.seed),
           "--hpo_summary_json", str(summary),
           "--save_dir", str(out_dir / "checkpoints")]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parent,
                                   env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    row = {"exit_code": process.returncode, "log": str(log),
           "elapsed_wall_sec": time.monotonic() - started,
           "checkpoint": str(out_dir / "checkpoints" / "best.pth")}
    if summary.is_file():
        row.update(json.loads(summary.read_text(encoding="utf-8")))
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Bounded Raindrop weight escape + matched SGD control")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--baseline_miou", type=float, default=0.6783018947437217)
    parser.add_argument("--gpu_ids", default="0,1")
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--proxy_batch_size", type=int, default=4)
    parser.add_argument("--proxy_samples", type=int, default=256)
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--relative_scale", type=float, default=0.02)
    parser.add_argument("--proxy_min_gain", type=float, default=0.0)
    parser.add_argument("--max_ce_increase", type=float, default=0.05)
    parser.add_argument("--pop_size", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=1)
    parser.add_argument("--fps_tolerance", type=float, default=0.0)
    parser.add_argument("--max_hours", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="raindrop_weight_escape_runs")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader, Subset
    from data.custom import CityscapesDataset, get_val_transforms
    import test as evaluation
    from train import ModelConfig
    from arch_config import apply_arch_config

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    if (len(gpu_ids) != 2 or len(set(gpu_ids)) != 2
            or any(x < 0 or x >= torch.cuda.device_count() for x in gpu_ids)):
        raise ValueError("Two distinct visible CUDA GPUs are required")
    if (args.pop_size < 4 or args.max_iter < 1 or args.epochs < 1
            or min(args.proxy_samples, args.proxy_batch_size, args.batch_size,
                   args.img_h, args.img_w, args.workers_per_gpu) < 1
            or not 0 < args.relative_scale <= 0.1
            or not 0 <= args.fps_tolerance <= 0.1
            or not 0 <= args.proxy_min_gain < 1
            or not 0 <= args.max_ce_increase <= 1
            or not 0 < args.baseline_miou < 1 or args.max_hours <= 0):
        raise ValueError("Invalid budget, search, or acceptance settings")
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_cfg = checkpoint_meta.get("model_config") or ModelConfig.get_config("fan_dwsa")
    del checkpoint_meta
    blocks = original_cfg["backbone"]["num_blocks_per_stage"]
    arch = {
        "dwsa_reduction": original_cfg["backbone"]["dwsa_reduction"],
        "ppm_channels": original_cfg["backbone"]["ppm_channels"],
        "sem_blocks_s4": blocks[2][0], "det_blocks_s4": blocks[2][1],
        "sem_blocks_s5": blocks[3][0], "det_blocks_s5": blocks[3][1],
        "sem_blocks_s6": blocks[4][0], "det_blocks_s6": blocks[4][1],
        "dropout_ratio": original_cfg["head"]["dropout_ratio"],
    }
    if apply_arch_config(ModelConfig.get_config("fan_dwsa"), arch) != original_cfg:
        raise ValueError("Checkpoint has a custom configuration this launcher cannot recreate")
    args.arch_json = root / "original_arch.json"
    args.arch_json.write_text(json.dumps(arch, indent=2), encoding="utf-8")
    deadline = time.monotonic() + args.max_hours * 3600
    device = torch.device(f"cuda:{gpu_ids[0]}")
    model, _ = evaluation.build_model("fan_dwsa", args.checkpoint,
                                     device, deploy=False)
    dataset = CityscapesDataset(
        txt_file=args.train_txt,
        transforms=get_val_transforms(img_size=(args.img_h, args.img_w)),
        img_size=(args.img_h, args.img_w), label_mapping="train_id",
        dataset_type="foggy")
    if len(dataset) < args.proxy_samples:
        raise ValueError("proxy_samples exceeds training set size")
    indices = np.sort(np.random.default_rng(args.seed).choice(
        len(dataset), size=args.proxy_samples, replace=False))
    (root / "proxy_indices.json").write_text(json.dumps(indices.tolist()), encoding="utf-8")
    loader = DataLoader(Subset(dataset, indices.tolist()),
                        batch_size=args.proxy_batch_size, shuffle=False,
                        num_workers=args.workers_per_gpu, pin_memory=True)
    proxy = search_perturbation(model, loader, device, args)
    print(f"Proxy: {proxy['baseline_proxy_miou']:.6f} -> "
          f"{proxy['best_proxy']['proxy_miou']:.6f}; "
          f"accepted={proxy['accepted_proxy']}", flush=True)
    if not proxy["accepted_proxy"]:
        output = {"best_candidate": None, "reason": "No safe proxy improvement",
                  "proxy": proxy}
        (root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        return

    candidate_path = root / "perturbed.pth"
    torch.save({"model": model.state_dict(),
                "model_config": original_cfg,
                "model_variant": "fan_dwsa", "best_miou": args.baseline_miou,
                "raindrop_escape": proxy}, candidate_path)
    del model, loader, dataset
    torch.cuda.empty_cache()

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        ordinary = pool.submit(_train, Path(args.checkpoint).resolve(), gpu_ids[1],
                               args, root / "ordinary", deadline)
        assisted = pool.submit(_train, candidate_path, gpu_ids[0],
                               args, root / "raindrop", deadline)
        ordinary = ordinary.result()
        assisted = assisted.result()

    result = {"baseline_miou": args.baseline_miou, "proxy": proxy,
              "ordinary": ordinary, "raindrop": assisted,
              "best_candidate": None}
    if (ordinary.get("exit_code") == 0 and assisted.get("exit_code") == 0
            and Path(ordinary["checkpoint"]).is_file()
            and Path(assisted["checkpoint"]).is_file()
            and time.monotonic() < deadline):
        # Validate all checkpoints with identical deploy/fuse rules.
        deployed = {}
        for label, path in (("original", args.checkpoint),
                            ("ordinary", ordinary["checkpoint"]),
                            ("raindrop", assisted["checkpoint"])):
            net, _ = evaluation.build_model("fan_dwsa", path, device, deploy=True)
            metrics = evaluation.validate(net, args.val_txt, args.img_h,
                                          args.img_w, args.batch_size,
                                          args.workers_per_gpu, device, True, None)
            deployed[label] = {"miou": float(metrics["miou_all_19"]),
                               "params": sum(p.numel() for p in net.parameters())}
            if label in ("original", "raindrop"):
                deployed[label]["benchmark"] = evaluation.benchmark(
                    net, args.img_h, args.img_w, device)
            del net
            torch.cuda.empty_cache()
        baseline = deployed["original"]
        control = deployed["ordinary"]
        candidate = deployed["raindrop"]
        result["deployed"] = deployed
        result["best_candidate"] = (assisted if eligible(
            candidate["miou"], max(args.baseline_miou, baseline["miou"]),
            control["miou"], candidate["params"] == baseline["params"],
            candidate["benchmark"]["fps_median"],
            baseline["benchmark"]["fps_median"], args.fps_tolerance)
            else None)
    (root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {root / 'result.json'}; winner={bool(result['best_candidate'])}",
          flush=True)


if __name__ == "__main__":
    main()
