"""Find the quickest fixed-GCNet fine-tuning recipe that beats full-val mIoU.

Two independent train.py trials run concurrently on Kaggle T4x2. Each trial
starts from the same checkpoint and stops at the first full-validation epoch
above the required all-19-class mIoU. No architecture or image size changes.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def trial_candidates():
    """Small baseline-centred search: only learning-rate allocation changes."""
    return [
        {"lr": 5e-4, "backbone_lr_factor": 0.10},  # existing recipe/control
        {"lr": 2e-4, "backbone_lr_factor": 0.10},
        {"lr": 1e-4, "backbone_lr_factor": 0.10},
        {"lr": 2e-4, "backbone_lr_factor": 0.05},
        {"lr": 1e-4, "backbone_lr_factor": 0.05},
        {"lr": 5e-4, "backbone_lr_factor": 0.05},
    ]


def select_fastest(results, threshold):
    reached = [row for row in results if row.get("target_reached")
               and row.get("best_miou", 0) > threshold
               and row.get("time_to_target_sec") is not None]
    return min(reached,
               key=lambda row: (row["time_to_target_sec"], -row["best_miou"])) if reached else None


def _identity(path):
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with open(resolved, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(resolved), "sha256": digest.hexdigest()}


def _next_attempt(trial_dir):
    attempt = 0
    while (trial_dir / f"attempt_{attempt}").exists():
        attempt += 1
    directory = trial_dir / f"attempt_{attempt}"
    directory.mkdir(parents=True)
    return directory


def parse_args():
    p = argparse.ArgumentParser(description="T4x2 time-to-higher-mIoU HPO")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_txt", required=True)
    p.add_argument("--val_txt", required=True)
    p.add_argument("--baseline_miou", type=float, default=0.6783018947437217)
    p.add_argument("--min_gain", type=float, default=0.001,
                   help="Require mIoU strictly above baseline plus this margin")
    p.add_argument("--max_epochs", type=int, default=4)
    p.add_argument("--max_hours", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument("--workers_per_gpu", type=int, default=2)
    p.add_argument("--gpu_ids", default="0,1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work_dir", default="hpo_time_to_miou_runs")
    p.add_argument("--out_json", default="best_hpo_time_to_miou.json")
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    try:
        gpu_ids = [int(part) for part in args.gpu_ids.split(",")]
    except ValueError as exc:
        raise ValueError("--gpu_ids must look like 0,1") from exc
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("This launcher needs two distinct T4 GPU IDs")
    if any(gpu_id < 0 or gpu_id >= torch.cuda.device_count() for gpu_id in gpu_ids):
        raise ValueError(f"Requested GPUs {gpu_ids}, visible: {torch.cuda.device_count()}")
    if args.min_gain < 0 or not 0 < args.baseline_miou + args.min_gain < 1:
        raise ValueError("Invalid baseline mIoU/minimum gain")
    if min(args.max_epochs, args.batch_size, args.workers_per_gpu) < 1:
        raise ValueError("Epochs, batch size and workers must be positive")
    if args.max_hours <= 0:
        raise ValueError("--max_hours must be positive")
    if args.workers_per_gpu > 2:
        print("T4x2 has four CPU cores; clamping workers to 2 per GPU")
        args.workers_per_gpu = 2

    root = Path(args.work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    trials = trial_candidates()
    threshold = args.baseline_miou + args.min_gain
    manifest = {
        "checkpoint": _identity(args.checkpoint),
        "train_txt": _identity(args.train_txt),
        "val_txt": _identity(args.val_txt),
        "threshold": threshold, "max_epochs": args.max_epochs,
        "batch_size": args.batch_size, "img_size": [args.img_h, args.img_w],
        "workers_per_gpu": args.workers_per_gpu, "gpu_ids": gpu_ids,
        "seed": args.seed, "trials": trials,
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            if json.load(f) != manifest:
                raise ValueError("Existing work_dir uses different data/settings; "
                                 "choose a new --work_dir to preserve it")
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    repo_dir = Path(__file__).resolve().parent
    train_script = repo_dir / "train.py"
    results = {}
    pending = []
    for i, params in enumerate(trials):
        trial_dir = root / f"trial_{i}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        result_path = trial_dir / "result.json"
        if result_path.exists():
            with open(result_path, encoding="utf-8") as f:
                cached = json.load(f)
            checkpoint_ok = (not cached.get("target_reached") or
                             Path(cached.get("best_checkpoint", "")).is_file())
            if cached.get("exit_code") == 0 and checkpoint_ok:
                results[i] = cached
                print(f"Reusing completed trial {i}")
            else:
                print(f"Retrying failed/missing-checkpoint trial {i}")
                pending.append(i)
        else:
            pending.append(i)

    started = time.perf_counter()
    deadline = started + args.max_hours * 3600
    active = {}
    stopping = {}

    def launch(i, gpu_id):
        params = trials[i]
        trial_dir = root / f"trial_{i}"
        attempt = _next_attempt(trial_dir)
        checkpoint_dir = attempt / "checkpoints"
        summary = attempt / "summary.json"
        log = attempt / "train.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, "-u", str(train_script),
            "--pretrained_weights", str(Path(args.checkpoint).resolve()),
            "--train_txt", str(Path(args.train_txt).resolve()),
            "--val_txt", str(Path(args.val_txt).resolve()),
            "--model_variant", "fan_dwsa", "--dataset_type", "foggy",
            "--num_classes", "19",
            "--img_h", str(args.img_h), "--img_w", str(args.img_w),
            "--batch_size", str(args.batch_size), "--accumulation_steps", "1",
            "--num_workers", str(args.workers_per_gpu), "--persistent_workers",
            "--epochs", str(args.max_epochs),
            "--lr", str(params["lr"]),
            "--backbone_lr_factor", str(params["backbone_lr_factor"]),
            "--seed", str(args.seed), "--gradient_check_interval", "0",
            "--target_miou", repr(threshold),
            "--hpo_summary_json", str(summary), "--save_dir", str(checkpoint_dir),
        ]
        log_handle = open(log, "w", encoding="utf-8")
        try:
            process = subprocess.Popen(cmd, cwd=repo_dir, env=env,
                                       stdout=log_handle,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != "nt"))
        except Exception:
            log_handle.close()
            raise
        active[gpu_id] = (i, process, log_handle, summary, log)
        print(f"Started trial {i} on T4 GPU {gpu_id}: {params}; log={log}")

    while pending or active:
        if time.perf_counter() >= deadline:
            for gpu_id, (_, process, _, _, _) in active.items():
                if process.poll() is not None:
                    continue
                try:
                    if gpu_id not in stopping:
                        if os.name == "nt":
                            process.terminate()
                        else:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        stopping[gpu_id] = time.perf_counter()
                        print(f"Time budget reached; stopping GPU {gpu_id} trial")
                    elif time.perf_counter() - stopping[gpu_id] > 30:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for gpu_id in gpu_ids:
            if gpu_id not in active and pending:
                if time.perf_counter() >= deadline:
                    break
                launch(pending.pop(0), gpu_id)
        completed = []
        for gpu_id, (i, process, handle, summary, log) in active.items():
            exit_code = process.poll()
            if exit_code is None:
                continue
            handle.close()
            row = {"id": i, "gpu_id": gpu_id, "params": trials[i],
                   "exit_code": exit_code, "log": str(log)}
            if summary.exists():
                with open(summary, encoding="utf-8") as f:
                    row.update(json.load(f))
            else:
                row.update(target_reached=False, best_miou=None,
                           time_to_target_sec=None)
            # A failed train process is never allowed to become a winner.
            if exit_code != 0:
                row["target_reached"] = False
            trial_dir = root / f"trial_{i}"
            with open(trial_dir / "result.json", "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            results[i] = row
            print(f"Finished trial {i} on GPU {gpu_id}: "
                  f"mIoU={row['best_miou']}, reached={row['target_reached']}")
            completed.append(gpu_id)
        for gpu_id in completed:
            del active[gpu_id]
        if not active and pending and time.perf_counter() >= deadline:
            print(f"Time budget reached; {len(pending)} trial(s) not started")
            break
        if active and not completed:
            time.sleep(2)

    ordered = [results[i] for i in sorted(results)]
    best = select_fastest(ordered, threshold)
    output = {
        "baseline_miou": args.baseline_miou,
        "required_miou_strictly_above": threshold,
        "best_candidate": best,
        "all_results": ordered,
        "not_started": pending,
        "note": "Time to first full-1500-val mIoU above threshold; one T4 per trial."
    }
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    if best:
        print(f"Fastest qualified trial {best['id']}: "
              f"{best['time_to_target_sec']/60:.1f} min, "
              f"mIoU={best['best_miou']:.6f}; saved {args.out_json}")
    else:
        print("No trial beat the checkpoint mIoU; keep the original checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
