"""Raindrop search for a class-aware segmentation training policy on T4x2."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from obl_de_rd import OBLAdaptiveRaindropOptimizer
from raindrop_weight_escape import _stream_process, eligible


CLASS_GROUPS = (
    (0, 1, 2, 8, 9, 10),       # large/static
    (3, 4, 5, 6, 7),           # thin/boundary
    (11, 12),                   # human
    (13,),                      # common vehicle
    (14, 15, 16, 17, 18),      # rare vehicle
)
WEAK_CLASSES = (3, 4, 5, 6, 12, 17)
COMMON_CLASSES = (0, 2, 8, 10, 13)
LOW = np.array([0.5] * 5 + [0.20, 0.20, 0.10], dtype=float)
HIGH = np.array([2.0] * 5 + [0.50, 0.80, 0.50], dtype=float)
BASELINE_X = np.array([1.0] * 5 + [0.30, 0.50, 0.40], dtype=float)


def decode_policy(vector):
    vector = np.clip(np.asarray(vector, dtype=float), LOW, HIGH)
    if vector.shape != (8,) or not np.isfinite(vector).all():
        raise ValueError("Policy must contain eight finite values")
    weights = np.ones(19, dtype=float)
    for value, indices in zip(vector[:5], CLASS_GROUPS):
        weights[list(indices)] = value
    weights /= weights.mean()
    return {
        "class_weights": weights.round(8).tolist(),
        "ohem_keep_ratio": round(float(vector[5]), 4),
        "dice_weight": round(float(vector[6]), 4),
        "aux_weight": round(float(vector[7]), 4),
    }


def policy_score(row, baseline=None):
    miou = row.get("best_miou")
    per_class = row.get("best_per_class_iou")
    if (row.get("exit_code") != 0 or miou is None or per_class is None
            or len(per_class) != 19):
        return -1e6
    per_class = np.asarray(per_class, dtype=float)
    weak = float(per_class[list(WEAK_CLASSES)].mean())
    common = float(per_class[list(COMMON_CLASSES)].mean())
    score = float(miou) + 0.15 * weak
    if baseline is not None:
        base_common = float(np.asarray(
            baseline["best_per_class_iou"])[list(COMMON_CLASSES)].mean())
        score -= 0.10 * max(0.0, base_common - common)
        score -= 100.0 * max(0.0, baseline["best_miou"] - float(miou))
    return score


def _key(policy):
    raw = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_training_file(path, root, train_samples, gate_samples, seed):
    lines = [line.strip() for line in Path(path).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if train_samples + gate_samples > len(lines):
        raise ValueError("Proxy train + gate samples exceed train.txt size")
    chosen = np.random.default_rng(seed).choice(
        len(lines), size=train_samples + gate_samples, replace=False)
    train_lines = [lines[i] for i in chosen[:train_samples]]
    gate_lines = [lines[i] for i in chosen[train_samples:]]
    train_path, gate_path = root / "proxy_train.txt", root / "proxy_gate.txt"
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    gate_path.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")
    return train_path, gate_path


def _run(policy, label, gpu, train_txt, val_txt, args, root, save_checkpoint,
         deadline):
    directory = root / label / _key(policy)
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / "result.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    weights_path = directory / "class_weights.json"
    weights_path.write_text(json.dumps(policy["class_weights"]), encoding="utf-8")
    summary = directory / "summary.json"
    log = directory / "train.log"
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("train.py")),
        "--pretrained_weights", str(Path(args.checkpoint).resolve()),
        "--train_txt", str(Path(train_txt).resolve()),
        "--val_txt", str(Path(val_txt).resolve()),
        "--model_variant", "fan_dwsa", "--dataset_type", "foggy",
        "--num_classes", "19", "--img_h", str(args.img_h),
        "--img_w", str(args.img_w), "--batch_size", str(args.batch_size),
        "--accumulation_steps", "1", "--num_workers", str(args.workers_per_gpu),
        "--persistent_workers", "--epochs", "1", "--freeze_backbone",
        "--lock_bn_stats", "--lr", str(args.lr), "--weight_decay", "0",
        "--scheduler", "cosine", "--use_class_weights",
        "--class_weights_file", str(weights_path),
        "--ohem_keep_ratio", str(policy["ohem_keep_ratio"]),
        "--dice_weight", str(policy["dice_weight"]),
        "--aux_weight", str(policy["aux_weight"]),
        "--gradient_check_interval", "0", "--seed", str(args.seed),
        "--hpo_summary_json", str(summary),
        "--save_dir", str(directory / "checkpoints"),
    ]
    if not save_checkpoint:
        command.append("--skip_checkpoint")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"Start {label} on GPU {gpu}: {_key(policy)}; "
          f"OHEM={policy['ohem_keep_ratio']}, Dice={policy['dice_weight']}, "
          f"Aux={policy['aux_weight']}", flush=True)
    started = time.monotonic()
    code = _stream_process(command, env, Path(__file__).resolve().parent,
                           log, deadline, f"{label} GPU{gpu}")
    row = {"policy": policy, "exit_code": code, "log": str(log),
           "elapsed_wall_sec": time.monotonic() - started,
           "checkpoint": str(directory / "checkpoints" / "best.pth")}
    if summary.is_file():
        row.update(json.loads(summary.read_text(encoding="utf-8")))
    cached.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"Finish {label} {_key(policy)}: mIoU={row.get('best_miou')}",
          flush=True)
    return row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Raindrop class-aware loss search on two T4 GPUs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--baseline_miou", type=float,
                        default=0.6783018947437217)
    parser.add_argument("--gpu_ids", default="0,1")
    parser.add_argument("--proxy_train_samples", type=int, default=1024)
    parser.add_argument("--proxy_gate_samples", type=int, default=256)
    parser.add_argument("--pop_size", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fps_tolerance", type=float, default=0.01)
    parser.add_argument("--max_hours", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="raindrop_loss_search_runs")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import test as evaluation

    gpus = [int(x) for x in args.gpu_ids.split(",")]
    if (len(gpus) != 2 or len(set(gpus)) != 2
            or any(x < 0 or x >= torch.cuda.device_count() for x in gpus)):
        raise ValueError("Two distinct CUDA GPUs are required")
    if args.pop_size < 4 or args.max_iter < 1 or args.max_hours <= 0:
        raise ValueError("Invalid search budget")
    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    proxy_train, proxy_gate = _split_training_file(
        args.train_txt, root, args.proxy_train_samples,
        args.proxy_gate_samples, args.seed)
    deadline = time.monotonic() + args.max_hours * 3600
    cache = {}

    with (ThreadPoolExecutor(max_workers=1) as pool0,
          ThreadPoolExecutor(max_workers=1) as pool1):
        pools = (pool0, pool1)

        def evaluate_policies(policies, label, train_txt, val_txt, save):
            unique = {_key(policy): policy for policy in policies}
            futures = {}
            for index, (key, policy) in enumerate(unique.items()):
                futures[key] = pools[index % 2].submit(
                    _run, policy, label, gpus[index % 2], train_txt, val_txt,
                    args, root, save, deadline)
            return {key: future.result() for key, future in futures.items()}

        baseline_policy = decode_policy(BASELINE_X)
        baseline_proxy = evaluate_policies(
            [baseline_policy], "proxy", proxy_train, proxy_gate, False)
        baseline_proxy = next(iter(baseline_proxy.values()))
        cache[_key(baseline_policy)] = baseline_proxy

        def fitness(population):
            policies = [decode_policy(x) for x in population]
            missing = [p for p in policies if _key(p) not in cache]
            cache.update(evaluate_policies(
                missing, "proxy", proxy_train, proxy_gate, False))
            return np.asarray([
                -policy_score(cache[_key(policy)], baseline_proxy)
                for policy in policies], dtype=float)

        optimizer = OBLAdaptiveRaindropOptimizer(
            obj_func=fitness, dim=8, lb=LOW, ub=HIGH,
            pop_size=args.pop_size, max_iter=args.max_iter, seed=args.seed)
        best_x, best_cost = optimizer.optimize(verbose=True)
        best_policy = decode_policy(best_x)

        full = evaluate_policies(
            [baseline_policy, best_policy], "full", args.train_txt,
            args.val_txt, True)

    control = full[_key(baseline_policy)]
    candidate = full[_key(best_policy)]
    output = {
        "algorithm": "OBL-ADE-RD class-aware loss search",
        "baseline_miou": args.baseline_miou,
        "baseline_policy": baseline_policy,
        "best_policy": best_policy,
        "optimizer_best_x": np.asarray(best_x).tolist(),
        "optimizer_best_cost": float(best_cost),
        "proxy_baseline": baseline_proxy,
        "proxy_results": list(cache.values()),
        "ordinary_control": control,
        "raindrop_candidate": candidate,
        "best_candidate": None,
    }
    if (control.get("exit_code") == 0 and candidate.get("exit_code") == 0
            and Path(control["checkpoint"]).is_file()
            and Path(candidate["checkpoint"]).is_file()):
        device = torch.device(f"cuda:{gpus[0]}")
        deployed = {}
        for label, checkpoint in (
                ("original", args.checkpoint),
                ("ordinary_control", control["checkpoint"]),
                ("raindrop_candidate", candidate["checkpoint"])):
            model, _ = evaluation.build_model(
                "fan_dwsa", checkpoint, device, deploy=True)
            metrics = evaluation.validate(
                model, args.val_txt, args.img_h, args.img_w, args.batch_size,
                args.workers_per_gpu, device, True, None)
            per_class = np.asarray(metrics["per_class_iou"], dtype=float)
            deployed[label] = {
                "miou": float(metrics["miou_all_19"]),
                "weak_class_miou": float(per_class[list(WEAK_CLASSES)].mean()),
                "per_class_iou": per_class.tolist(),
                "params": sum(p.numel() for p in model.parameters()),
            }
            if label in ("original", "raindrop_candidate"):
                deployed[label]["benchmark"] = evaluation.benchmark(
                    model, args.img_h, args.img_w, device)
            del model
            torch.cuda.empty_cache()
        original, ordinary, guided = (deployed["original"],
                                      deployed["ordinary_control"],
                                      deployed["raindrop_candidate"])
        output["deployed"] = deployed
        if eligible(guided["miou"], max(args.baseline_miou, original["miou"]),
                    ordinary["miou"], guided["params"] == original["params"],
                    guided["benchmark"]["fps_median"],
                    original["benchmark"]["fps_median"],
                    args.fps_tolerance) and (guided["weak_class_miou"]
                                             > ordinary["weak_class_miou"]):
            output["best_candidate"] = candidate
    result_path = root / "result.json"
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved {result_path}; winner={bool(output['best_candidate'])}",
          flush=True)


if __name__ == "__main__":
    main()
