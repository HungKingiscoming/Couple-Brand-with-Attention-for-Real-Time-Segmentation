"""Run Raindrop-guided AdamW against a matched AdamW control on two GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raindrop_weight_escape import _stream_process, eligible


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matched two-GPU Raindrop-guided AdamW experiment")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--baseline_miou", type=float,
                        default=0.6783018947437217)
    parser.add_argument("--gpu_ids", default="0,1")
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--guide_samples", type=int, default=192)
    parser.add_argument("--guide_gate_fraction", type=float, default=1/3)
    parser.add_argument("--guide_batch_size", type=int, default=4)
    parser.add_argument("--gradient_batches", type=int, default=4)
    parser.add_argument("--pop_size", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=1)
    parser.add_argument("--factor_min", type=float, default=0.5)
    parser.add_argument("--factor_max", type=float, default=1.5)
    parser.add_argument("--min_ce_gain", type=float, default=1e-4)
    parser.add_argument("--fps_tolerance", type=float, default=0.0)
    parser.add_argument("--max_hours", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="raindrop_guided_runs")
    return parser.parse_args()


def _training_command(args, arch_json, summary, save_dir, guided):
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("train.py")),
        "--pretrained_weights", str(Path(args.checkpoint).resolve()),
        "--arch_json", str(arch_json),
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
        "--hpo_summary_json", str(summary), "--save_dir", str(save_dir),
    ]
    if guided:
        command.extend([
            "--raindrop_guided",
            "--raindrop_guide_samples", str(args.guide_samples),
            "--raindrop_guide_gate_fraction", str(args.guide_gate_fraction),
            "--raindrop_proxy_batch_size", str(args.guide_batch_size),
            "--raindrop_gradient_batches", str(args.gradient_batches),
            "--raindrop_pop_size", str(args.pop_size),
            "--raindrop_max_iter", str(args.max_iter),
            "--raindrop_factor_min", str(args.factor_min),
            "--raindrop_factor_max", str(args.factor_max),
            "--raindrop_min_ce_gain", str(args.min_ce_gain),
        ])
    return command


def _run_training(args, arch_json, root, gpu, guided, deadline):
    label = "raindrop_guided" if guided else "ordinary_adamw"
    out_dir = root / label
    out_dir.mkdir(parents=True)
    summary = out_dir / "summary.json"
    log = out_dir / "train.log"
    command = _training_command(
        args, arch_json, summary, out_dir / "checkpoints", guided)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.monotonic()
    print(f"Starting {label} on physical GPU {gpu}; log={log}", flush=True)
    exit_code = _stream_process(
        command, env, Path(__file__).resolve().parent, log, deadline,
        f"{label} GPU{gpu}")
    result = {
        "exit_code": exit_code,
        "elapsed_wall_sec": time.monotonic() - started,
        "log": str(log),
        "checkpoint": str(out_dir / "checkpoints" / "best.pth"),
    }
    if summary.is_file():
        result.update(json.loads(summary.read_text(encoding="utf-8")))
    return result


def main():
    args = parse_args()
    import torch
    import test as evaluation
    from arch_config import apply_arch_config
    from train import ModelConfig

    gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
    if (len(gpu_ids) != 2 or len(set(gpu_ids)) != 2
            or any(gpu < 0 or gpu >= torch.cuda.device_count()
                   for gpu in gpu_ids)):
        raise ValueError("Two distinct visible CUDA GPUs are required")
    if (args.epochs < 1 or args.batch_size < 1 or args.workers_per_gpu < 1
            or args.max_hours <= 0 or not 0 <= args.fps_tolerance <= 0.1):
        raise ValueError("Invalid training budget")
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)

    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_cfg = (metadata.get("model_config")
                    or ModelConfig.get_config("fan_dwsa"))
    del metadata
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
        raise ValueError("Checkpoint configuration cannot be recreated")
    arch_json = root / "original_arch.json"
    arch_json.write_text(json.dumps(arch, indent=2), encoding="utf-8")
    deadline = time.monotonic() + args.max_hours * 3600

    with ThreadPoolExecutor(max_workers=2) as pool:
        guided_future = pool.submit(
            _run_training, args, arch_json, root, gpu_ids[0], True, deadline)
        control_future = pool.submit(
            _run_training, args, arch_json, root, gpu_ids[1], False, deadline)
        guided = guided_future.result()
        control = control_future.result()

    result = {
        "baseline_miou": args.baseline_miou,
        "ordinary_adamw": control,
        "raindrop_guided": guided,
        "best_candidate": None,
    }
    guided_path = Path(guided["checkpoint"])
    control_path = Path(control["checkpoint"])
    if (guided.get("exit_code") == 0 and control.get("exit_code") == 0
            and guided_path.is_file() and control_path.is_file()
            and time.monotonic() < deadline):
        device = torch.device(f"cuda:{gpu_ids[0]}")
        deployed = {}
        for label, checkpoint in (
                ("original", args.checkpoint),
                ("ordinary_adamw", control_path),
                ("raindrop_guided", guided_path)):
            net, _ = evaluation.build_model(
                "fan_dwsa", checkpoint, device, deploy=True)
            metrics = evaluation.validate(
                net, args.val_txt, args.img_h, args.img_w, args.batch_size,
                args.workers_per_gpu, device, True, None)
            deployed[label] = {
                "miou": float(metrics["miou_all_19"]),
                "params": sum(param.numel() for param in net.parameters()),
            }
            if label in ("original", "raindrop_guided"):
                deployed[label]["benchmark"] = evaluation.benchmark(
                    net, args.img_h, args.img_w, device)
            del net
            torch.cuda.empty_cache()
        baseline = deployed["original"]
        ordinary = deployed["ordinary_adamw"]
        candidate = deployed["raindrop_guided"]
        result["deployed"] = deployed
        if eligible(
                candidate["miou"], max(args.baseline_miou, baseline["miou"]),
                ordinary["miou"], candidate["params"] == baseline["params"],
                candidate["benchmark"]["fps_median"],
                baseline["benchmark"]["fps_median"], args.fps_tolerance):
            result["best_candidate"] = guided
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {result_path}; winner={bool(result['best_candidate'])}",
          flush=True)


if __name__ == "__main__":
    main()
