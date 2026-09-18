"""Budgeted OBL-ADE-RD search over GCNet fine-tuning hyperparameters.

Architecture, image size, checkpoint and data split stay fixed. Raindrop
proposes configurations; train.py supplies the expensive measured fitness.
One-epoch full-data/full-validation proxies screen candidates, then the two
best are restarted for four epochs and checked with deploy-mode validation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpo_time_to_miou import trial_training_args
from obl_de_rd import OBLAdaptiveRaindropOptimizer


SCHEDULERS = ("cosine", "poly")
DECAYS = (0.0, 1e-4, 1e-2)
SCOPES = ("head_only", "attention_head")
LOW = np.array([-6.0, -0.5, -0.5, -0.5], dtype=float)
HIGH = np.array([-4.0, 2.5, 1.5, 1.5], dtype=float)


def decode_candidate(x):
    """Map a continuous four-dimensional raindrop to a valid train recipe."""
    x = np.clip(np.asarray(x, dtype=float), LOW, HIGH)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError("Candidate must have four finite dimensions")
    lr = round(10.0 ** x[0], 8)
    decay = DECAYS[int(np.clip(np.rint(x[1]), 0, len(DECAYS) - 1))]
    scheduler = SCHEDULERS[int(np.clip(np.rint(x[2]), 0, 1))]
    scope = SCOPES[int(np.clip(np.rint(x[3]), 0, 1))]
    return {
        "lr": lr, "backbone_lr_factor": 0.05,
        "scheduler": scheduler, "weight_decay": decay,
        "scope": scope, "lock_bn_stats": True,
        "aux_weight": 0.0 if scope == "head_only" else 0.4,
    }


def proxy_cost(row, threshold):
    """Keep failed trials distinct; feasibility is a hard priority."""
    miou = row.get("best_miou")
    if row.get("exit_code") != 0 or miou is None or not math.isfinite(miou):
        return 1e6
    hours = row["elapsed_wall_sec"] / 3600
    if miou > threshold:
        return hours
    return 100.0 + 1000.0 * (threshold - miou) + hours


def _file_id(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _recipe_key(params):
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _next_attempt(parent):
    i = 0
    while (parent / f"attempt_{i}").exists():
        i += 1
    path = parent / f"attempt_{i}"
    path.mkdir(parents=True)
    return path


def _stop_process(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
    except ProcessLookupError:
        pass


def _run_candidate(params, stage, gpu_id, args, root, threshold, deadline):
    key = _recipe_key(params)
    directory = root / stage / key
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        with result_path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        checkpoint_ok = (stage == "proxy" or not row.get("target_reached") or
                         Path(row.get("best_checkpoint") or "").is_file())
        if row.get("exit_code") == 0 and row.get("best_miou") is not None and checkpoint_ok:
            print(f"Reuse {stage} {key}: mIoU={row['best_miou']:.6f}", flush=True)
            return row
    if time.monotonic() >= deadline:
        return {"params": params, "stage": stage, "exit_code": None,
                "best_miou": None, "target_reached": False,
                "elapsed_wall_sec": 0, "not_started": True}

    attempt = _next_attempt(directory)
    summary = attempt / "summary.json"
    log = attempt / "train.log"
    save_dir = attempt / "checkpoints"
    command = [
        sys.executable, "-u", str(Path(__file__).resolve().parent / "train.py"),
        "--pretrained_weights", str(Path(args.checkpoint).resolve()),
        "--train_txt", str(Path(args.train_txt).resolve()),
        "--val_txt", str(Path(args.val_txt).resolve()),
        "--model_variant", "fan_dwsa", "--dataset_type", "foggy",
        "--num_classes", "19", "--img_h", str(args.img_h),
        "--img_w", str(args.img_w), "--batch_size", str(args.batch_size),
        "--accumulation_steps", "1", "--num_workers", str(args.workers_per_gpu),
        "--persistent_workers", "--epochs", str(args.full_epochs),
        *trial_training_args(params),
        "--seed", str(args.seed), "--gradient_check_interval", "0",
        "--target_miou", repr(threshold),
        "--hpo_summary_json", str(summary), "--save_dir", str(save_dir),
    ]
    if stage == "proxy":
        command += ["--max_trial_epochs", str(args.proxy_epochs),
                    "--skip_checkpoint"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Start {stage} {key} on GPU {gpu_id}: {params}; log={log}", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, cwd=Path(__file__).resolve().parent, env=environment,
            stdout=output, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"))
        try:
            process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _stop_process(process)
    row = {"params": params, "stage": stage, "gpu_id": gpu_id,
           "exit_code": process.returncode,
           "elapsed_wall_sec": time.monotonic() - started,
           "log": str(log)}
    if summary.exists():
        with summary.open(encoding="utf-8") as handle:
            row.update(json.load(handle))
    else:
        row.update(best_miou=None, target_reached=False,
                   time_to_target_sec=None, best_checkpoint=None)
    if row["exit_code"] != 0:
        row["target_reached"] = False
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
    print(f"Finish {stage} {key}: mIoU={row['best_miou']}, "
          f"reached={row['target_reached']}", flush=True)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="OBL-ADE-RD fine-tuning search on T4x2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_txt", required=True)
    parser.add_argument("--val_txt", required=True)
    parser.add_argument("--baseline_miou", type=float, default=0.6783018947437217)
    parser.add_argument("--min_gain", type=float, default=0.0)
    parser.add_argument("--pop_size", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=2)
    parser.add_argument("--proxy_epochs", type=int, default=1)
    parser.add_argument("--full_epochs", type=int, default=4)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--max_hours", type=float, default=10.0)
    parser.add_argument("--proxy_hours", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--img_w", type=int, default=1024)
    parser.add_argument("--workers_per_gpu", type=int, default=2)
    parser.add_argument("--gpu_ids", default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work_dir", default="raindrop_hpo_runs")
    parser.add_argument("--out_json", default="best_raindrop_hpo.json")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    gpu_ids = [int(part) for part in args.gpu_ids.split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2 or any(
            gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids):
        raise ValueError("Two distinct visible GPU IDs are required")
    if args.pop_size < 4 or args.max_iter < 1:
        raise ValueError("Use pop_size >= 4 and max_iter >= 1")
    if not 1 <= args.proxy_epochs < args.full_epochs:
        raise ValueError("Require 1 <= proxy_epochs < full_epochs")
    if not 1 <= args.finalists <= 2:
        raise ValueError("Only one or two full-run finalists fit the T4x2 budget")
    if args.max_hours <= 3 or not 0 < args.proxy_hours <= args.max_hours - 3:
        raise ValueError("Reserve at least three hours for finalists")
    if min(args.batch_size, args.img_h, args.img_w, args.workers_per_gpu) < 1:
        raise ValueError("Batch, resolution and worker count must be positive")
    if args.workers_per_gpu > 2:
        print("Clamping workers to two per GPU", flush=True)
        args.workers_per_gpu = 2
    threshold = args.baseline_miou + args.min_gain
    if not math.isfinite(threshold) or not 0 < threshold < 1 or args.min_gain < 0:
        raise ValueError("Invalid baseline mIoU/min_gain")

    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1, "checkpoint": _file_id(args.checkpoint),
        "train_txt": _file_id(args.train_txt), "val_txt": _file_id(args.val_txt),
        "threshold": threshold, "pop_size": args.pop_size,
        "max_iter": args.max_iter, "proxy_epochs": args.proxy_epochs,
        "full_epochs": args.full_epochs, "finalists": args.finalists,
        "batch_size": args.batch_size, "img_size": [args.img_h, args.img_w],
        "workers_per_gpu": args.workers_per_gpu, "gpu_ids": gpu_ids,
        "max_hours": args.max_hours, "proxy_hours": args.proxy_hours,
        "seed": args.seed, "space": {"low": LOW.tolist(),
                                     "high": HIGH.tolist(), "decays": list(DECAYS),
                                     "schedulers": list(SCHEDULERS),
                                     "scopes": list(SCOPES)},
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise ValueError("Existing work_dir has different settings; use a new one")
    else:
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    started = time.monotonic()
    proxy_deadline = started + args.proxy_hours * 3600
    final_deadline = started + args.max_hours * 3600
    proxy_rows = {}

    with (ThreadPoolExecutor(max_workers=1) as gpu0,
          ThreadPoolExecutor(max_workers=1) as gpu1):
        executors = (gpu0, gpu1)

        def evaluate_batch(items, stage, deadline):
            unique = {}
            for params in items:
                unique[_recipe_key(params)] = params
            futures = {}
            for index, (key, params) in enumerate(unique.items()):
                futures[key] = executors[index % 2].submit(
                    _run_candidate, params, stage, gpu_ids[index % 2],
                    args, root, threshold, deadline)
            return {key: future.result() for key, future in futures.items()}

        def fitness(X):
            decoded = [decode_candidate(x) for x in X]
            missing = [params for params in decoded
                       if _recipe_key(params) not in proxy_rows]
            proxy_rows.update(evaluate_batch(missing, "proxy", proxy_deadline))
            return np.asarray([
                proxy_cost(proxy_rows[_recipe_key(params)], threshold)
                for params in decoded], dtype=float)

        optimizer = OBLAdaptiveRaindropOptimizer(
            obj_func=fitness, dim=4, lb=LOW, ub=HIGH,
            pop_size=args.pop_size, max_iter=args.max_iter, seed=args.seed)
        print(f"Raindrop HPO: ~{args.pop_size * (args.max_iter + 2)} "
              f"proxy evaluations, then {args.finalists} full finalists; "
              f"mIoU must exceed {threshold:.8f}", flush=True)
        best_x, best_cost = optimizer.optimize(verbose=True)

        evaluated = [row for row in proxy_rows.values()
                     if row.get("exit_code") == 0 and
                     row.get("best_miou") is not None]
        evaluated.sort(key=lambda row: (
            not row.get("target_reached"),
            (row.get("time_to_target_sec") or row["elapsed_wall_sec"])
            if row.get("target_reached") else -row["best_miou"]))
        finalists = evaluated[:args.finalists]
        full_rows = evaluate_batch([row["params"] for row in finalists],
                                   "full", final_deadline)

    qualified = sorted(
        [row for row in full_rows.values()
         if row.get("exit_code") == 0 and row.get("target_reached") and
         row.get("best_miou", 0) > threshold and
         row.get("time_to_target_sec") is not None and
         row.get("best_checkpoint") and
         Path(row["best_checkpoint"]).is_file()],
        key=lambda row: row["time_to_target_sec"])
    winner = None
    if qualified and time.monotonic() < final_deadline:
        import test as test_module
        device = torch.device("cuda:0")
        for row in qualified:
            if time.monotonic() >= final_deadline:
                break
            model, _ = test_module.build_model(
                "fan_dwsa", row["best_checkpoint"], device, deploy=True)
            metrics = test_module.validate(
                model, str(Path(args.val_txt).resolve()), args.img_h, args.img_w,
                args.batch_size, args.workers_per_gpu, device, True, None)
            row["deploy_miou_all_19"] = float(metrics["miou_all_19"])
            del model
            torch.cuda.empty_cache()
            if row["deploy_miou_all_19"] > threshold:
                winner = row
                break

    output = {
        "baseline_miou": args.baseline_miou,
        "required_miou_strictly_above": threshold,
        "algorithm": "OBL-ADE-RD",
        "optimizer_best_x": np.asarray(best_x).tolist(),
        "optimizer_best_proxy_cost": float(best_cost),
        "history_best_proxy_cost": [float(value) for value in optimizer.history_best],
        "best_candidate": winner,
        "proxy_results": list(proxy_rows.values()),
        "full_results": list(full_rows.values()),
        "note": "Proxy scores do not certify mIoU; only full + deploy validation can win.",
    }
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    if winner:
        print(f"Qualified winner: {winner['params']}, "
              f"deploy mIoU={winner['deploy_miou_all_19']:.6f}, "
              f"train-to-target={winner['time_to_target_sec']/60:.1f} min", flush=True)
    else:
        print("No candidate exceeded the baseline on full deploy validation; "
              "keep the original checkpoint", flush=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
