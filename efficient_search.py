"""Budget-aware, baseline-centred GCNet architecture search for Kaggle.

This is deliberately separate from the unconstrained OBL-ADE-RD experiment:
the objective here is a smaller/faster model with approximately unchanged
validation quality, not maximum proxy mIoU at any cost.
"""

import argparse
import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path

from arch_config import apply_arch_config, validate_arch_config


def baseline_arch(base_cfg):
    stages = base_cfg["backbone"]["num_blocks_per_stage"]
    return {
        "dwsa_reduction": base_cfg["backbone"]["dwsa_reduction"],
        "ppm_channels": base_cfg["backbone"]["ppm_channels"],
        "sem_blocks_s4": stages[2][0], "det_blocks_s4": stages[2][1],
        "sem_blocks_s5": stages[3][0], "det_blocks_s5": stages[3][1],
        "sem_blocks_s6": stages[4][0], "det_blocks_s6": stages[4][1],
        "dropout_ratio": base_cfg["head"]["dropout_ratio"],
    }


def propose_architectures(base, count, seed):
    """One-block ablations first, then seeded mixed reductions; never expand."""
    if count < 1:
        raise ValueError("--n_candidates must be positive")
    depth_keys = ("det_blocks_s4", "det_blocks_s5", "sem_blocks_s4",
                  "sem_blocks_s5")
    seen = {json.dumps(base, sort_keys=True)}
    candidates = []

    def add(arch):
        validate_arch_config(arch)
        key = json.dumps(arch, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(arch)

    for key in depth_keys:
        if base[key] > 2:
            arch = dict(base)
            arch[key] -= 1
            add(arch)
    for ppm in (96, 64):
        if ppm < base["ppm_channels"]:
            arch = dict(base)
            arch["ppm_channels"] = ppm
            add(arch)

    rng = random.Random(seed)
    attempts = 0
    while len(candidates) < count and attempts < 1000:
        attempts += 1
        arch = dict(base)
        for key in depth_keys:
            arch[key] = rng.randint(2, base[key])
        arch["ppm_channels"] = rng.choice(
            [x for x in (64, 96, 128) if x <= base["ppm_channels"]])
        arch["dwsa_reduction"] = rng.choice(
            [x for x in (8, 16, 32) if x >= base["dwsa_reduction"]])
        add(arch)
    return candidates[:count]


def select_for_promotion(rows, baseline_miou, count, tolerance):
    """Preserve promising accuracy, then favour latency; keep fallbacks."""
    feasible = [r for r in rows if r["miou"] >= baseline_miou - tolerance]
    feasible.sort(key=lambda r: (r["latency_ms"], -r["miou"]))
    fallback = [r for r in rows if r not in feasible]
    fallback.sort(key=lambda r: (-r["miou"], r["latency_ms"]))
    return (feasible + fallback)[:count]


def _append_log(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _file_identity(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    with open(resolved, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "size": stat.st_size,
            "sha256": digest.hexdigest()}


def _manifest(args):
    return {
        "checkpoint": _file_identity(args.pretrained_weights),
        "train_txt": _file_identity(args.train_txt),
        "val_txt": _file_identity(args.val_txt),
        "variant": args.model_variant, "dataset_type": args.dataset_type,
        "num_classes": args.num_classes, "img_size": [args.img_h, args.img_w],
        "batch_size": args.batch_size, "proxy_num_workers": args.proxy_num_workers,
        "lr": args.lr, "seed": args.seed,
        "n_candidates": args.n_candidates, "n_promote": args.n_promote,
        "stage1": [args.stage1_fraction, args.stage1_epochs],
        "stage2": [args.stage2_fraction, args.stage2_extra_epochs],
        "proxy_tolerance": args.proxy_tolerance,
        "min_speedup": args.min_speedup,
        "latency_warmup": args.latency_warmup,
        "latency_repeat": args.latency_repeat,
    }


def _latency_ms(cfg, device, img_h, img_w, warmup, repeat):
    """Measure deploy architecture, with no checkpoint or training needed."""
    import torch
    from model.backbone.model import GCNet
    from model.head.segmentation_head import GCNetHead
    from train import Segmentor

    deploy_cfg = copy.deepcopy(cfg)
    deploy_cfg["backbone"]["deploy"] = True
    model = Segmentor(GCNet(**deploy_cfg["backbone"]),
                      GCNetHead(**deploy_cfg["head"], num_classes=19)).to(device)
    model.eval()
    inp = torch.randn(1, 3, img_h, img_w, device=device)
    times = []
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(inp)
            torch.cuda.synchronize(device)
            for _ in range(repeat):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                for _ in range(100):
                    model(inp)
                torch.cuda.synchronize(device)
                times.append((time.perf_counter() - start) * 10.0)
        return sorted(times)[len(times) // 2]
    finally:
        del model, inp
        gc.collect()
        torch.cuda.empty_cache()


def _build_model(cfg, num_classes, device):
    from model.backbone.model import GCNet
    from model.head.segmentation_head import GCNetHead
    from train import Segmentor
    return Segmentor(GCNet(**cfg["backbone"]),
                     GCNetHead(**cfg["head"], num_classes=num_classes)).to(device)


def _proxy_run(arch, base_cfg, checkpoint, loader, val_loader, args,
               epochs, prior_path, save_path, run_dir):
    import torch
    from gcnet_search_space import _build_proxy_args
    from train import Trainer, build_optimizer, load_pretrained_gcnet

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = apply_arch_config(base_cfg, arch)
    model = _build_model(cfg, args.num_classes, args.device)
    if prior_path is None:
        loaded = load_pretrained_gcnet(model, checkpoint)
        if loaded < 20.0:
            raise RuntimeError(f"Only {loaded:.1f}% of pretrained weights loaded")
    else:
        model.load_state_dict(torch.load(prior_path, map_location="cpu",
                                         weights_only=True), strict=True)
        loaded = 100.0

    proxy_args = _build_proxy_args(base_cfg["loss"], epochs, run_dir,
                                   args.num_classes, 255, args.batch_size,
                                   args.lr, fast_proxy=False)
    optimizer = build_optimizer(model, proxy_args)
    trainer = Trainer(model, optimizer, None, args.device, proxy_args)
    try:
        # Model construction consumes RNG differently at each depth. Reset
        # immediately before iterating shared DataLoaders for fair comparison.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        for epoch in range(epochs):
            trainer.train_epoch(loader, epoch)
        metrics = trainer.validate(val_loader, epochs - 1)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   save_path)
        return {"miou": float(metrics["miou"]), "load_pct": float(loaded)}
    finally:
        trainer.writer.close()
        del model, optimizer, trainer
        gc.collect()
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description="Budget-aware two-rung GCNet search")
    p.add_argument("--pretrained_weights", required=True)
    p.add_argument("--train_txt", required=True)
    p.add_argument("--val_txt", required=True)
    p.add_argument("--model_variant", default="fan_dwsa", choices=["fan_dwsa", "dwsa_only"])
    p.add_argument("--dataset_type", default="foggy", choices=["foggy", "normal"])
    p.add_argument("--num_classes", type=int, default=19)
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--proxy_num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_candidates", type=int, default=12)
    p.add_argument("--n_promote", type=int, default=3)
    p.add_argument("--stage1_fraction", type=float, default=0.10)
    p.add_argument("--stage1_epochs", type=int, default=1)
    p.add_argument("--stage2_fraction", type=float, default=0.20)
    p.add_argument("--stage2_extra_epochs", type=int, default=2)
    p.add_argument("--proxy_tolerance", type=float, default=0.02)
    p.add_argument("--min_speedup", type=float, default=0.03,
                   help="Minimum measured latency improvement before proxy training")
    p.add_argument("--latency_warmup", type=int, default=20)
    p.add_argument("--latency_repeat", type=int, default=3)
    p.add_argument("--max_hours", type=float, default=10.0,
                   help="Stop starting new candidate runs after this many hours")
    p.add_argument("--work_dir", default="efficient_search_runs")
    p.add_argument("--out_json", default="best_efficient_arch.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_classes != 19:
        raise ValueError("Latency screening currently supports the 19-class foggy task")
    if not args.device.startswith("cuda"):
        raise ValueError("Hardware latency screening requires CUDA")
    if not (0 < args.stage1_fraction <= args.stage2_fraction <= 1):
        raise ValueError("Require 0 < stage1_fraction <= stage2_fraction <= 1")
    if min(args.stage1_epochs, args.stage2_extra_epochs, args.n_promote) < 1:
        raise ValueError("Epochs and n_promote must be positive")
    if not (0 <= args.min_speedup < 1) or args.proxy_tolerance < 0:
        raise ValueError("Speedup/tolerance outside valid range")
    if args.max_hours <= 0:
        raise ValueError("--max_hours must be positive")
    started = time.perf_counter()

    import torch
    from data.custom import create_dataloaders
    from gcnet_search_space import _make_subset_txt
    from train import ModelConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run the search on a GPU notebook")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    base_cfg = ModelConfig.get_config(args.model_variant)
    baseline = baseline_arch(base_cfg)
    candidates = propose_architectures(baseline, args.n_candidates, args.seed)
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = _manifest(args)
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            previous = json.load(f)
        if previous != manifest:
            raise ValueError(f"Existing {root} belongs to different settings/data. "
                             "Use a new --work_dir to preserve that run.")
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    log_path = root / "results.jsonl"

    base_model = _build_model(base_cfg, args.num_classes, args.device)
    baseline_params = sum(p.numel() for p in base_model.parameters())
    del base_model
    torch.cuda.empty_cache()
    baseline_latency = _latency_ms(base_cfg, args.device, args.img_h, args.img_w,
                                   args.latency_warmup, args.latency_repeat)
    print(f"Baseline: {baseline_params/1e6:.2f}M train params, "
          f"{baseline_latency:.2f} ms deploy latency")
    checkpoint = torch.load(args.pretrained_weights, map_location="cpu",
                            weights_only=False)

    screened = []
    for i, arch in enumerate(candidates):
        if time.perf_counter() - started >= args.max_hours * 3600:
            print("Time budget reached during screening; saved log is available")
            return 3
        cfg = apply_arch_config(base_cfg, arch)
        model = _build_model(cfg, args.num_classes, args.device)
        params = sum(p.numel() for p in model.parameters())
        del model
        torch.cuda.empty_cache()
        if params > baseline_params:
            print(f"Skip candidate {i}: {params/1e6:.2f}M > baseline")
            continue
        latency = _latency_ms(cfg, args.device, args.img_h, args.img_w,
                              args.latency_warmup, args.latency_repeat)
        row = {"id": i, "arch": arch, "train_params": params,
               "latency_ms": latency, "stage": "screen"}
        _append_log(log_path, row)
        if latency <= baseline_latency * (1 - args.min_speedup):
            screened.append(row)
        else:
            print(f"Skip candidate {i}: {latency:.2f} ms has insufficient speedup")
    if not screened:
        print("No architecture passed size and latency gates; relax --min_speedup")
        return 2

    loaders = {}
    for stage, fraction in ((1, args.stage1_fraction), (2, args.stage2_fraction)):
        train_sub = root / f"train_stage{stage}.txt"
        val_sub = root / f"val_stage{stage}.txt"
        _make_subset_txt(args.train_txt, fraction, train_sub, args.seed)
        _make_subset_txt(args.val_txt, fraction, val_sub, args.seed)
        train_loader, val_loader, _ = create_dataloaders(
            train_txt=str(train_sub), val_txt=str(val_sub),
            batch_size=args.batch_size, num_workers=args.proxy_num_workers,
            img_size=(args.img_h, args.img_w), pin_memory=True,
            compute_class_weights=False, dataset_type=args.dataset_type,
            # Fresh worker seeds per candidate keep stochastic augmentations
            # comparable; persistent worker RNG drifts across candidates.
            persistent_workers=False,
            prefetch_factor=2)
        loaders[stage] = (train_loader, val_loader)

    def run(stage, row, prior=None):
        candidate_dir = root / f"candidate_{row['id']}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        save_path = candidate_dir / f"stage{stage}.pth"
        result_path = candidate_dir / f"stage{stage}.json"
        if save_path.exists() and result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                previous = json.load(f)
            if previous["arch"] == row["arch"]:
                print(f"Reusing candidate {row['id']} stage {stage}: "
                      f"mIoU={previous['miou']:.4f}")
                return dict(previous, latency_ms=row["latency_ms"],
                            train_params=row["train_params"])
        if time.perf_counter() - started >= args.max_hours * 3600:
            raise TimeoutError("Time budget reached; completed candidate stages "
                               "are saved and will be reused on rerun")
        epochs = args.stage1_epochs if stage == 1 else args.stage2_extra_epochs
        metrics = _proxy_run(row["arch"], base_cfg, checkpoint, *loaders[stage],
                             args, epochs, prior, save_path, candidate_dir)
        result = dict(row, stage=f"proxy{stage}", **metrics,
                      checkpoint=str(save_path))
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        _append_log(log_path, result)
        return result

    baseline_row = {"id": "baseline", "arch": baseline,
                    "train_params": baseline_params,
                    "latency_ms": baseline_latency}
    try:
        baseline_stage1 = run(1, baseline_row)
        first = [run(1, row) for row in screened]
        promoted = select_for_promotion(first, baseline_stage1["miou"],
                                         args.n_promote, args.proxy_tolerance)
        baseline_stage2 = run(2, baseline_row, baseline_stage1["checkpoint"])
        final = [run(2, row, row["checkpoint"]) for row in promoted]
    except TimeoutError as exc:
        print(exc)
        return 3
    viable = [r for r in final if r["miou"] >= baseline_stage2["miou"]
              - args.proxy_tolerance]
    if not viable:
        print("No candidate stayed within baseline proxy tolerance at stage 2")
        return 2
    best = min(viable, key=lambda r: (r["latency_ms"], -r["miou"]))
    output = {"best_config": best["arch"], "best_proxy_miou": best["miou"],
              "baseline_proxy_miou": baseline_stage2["miou"],
              "baseline_latency_ms": baseline_latency,
              "candidate_latency_ms": best["latency_ms"],
              "baseline_train_params": baseline_params,
              "candidate_train_params": best["train_params"],
              "candidate_proxy_checkpoint": best["checkpoint"],
              "note": "Proxy only. Full-train and test on all 1500 val images before claiming improvement."}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Selected candidate {best['id']}: {best['miou']:.4f} proxy mIoU, "
          f"{best['latency_ms']:.2f} ms; saved {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
