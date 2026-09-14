"""
Run the OBL-ADE-RD architecture search over GCNet's stage-4/5/6 depths,
ppm_channels, dwsa_reduction and head dropout_ratio (see
gcnet_search_space.py for exactly which knobs and why the rest are fixed).

Usage (on Kaggle, with a GPU):
    python search_architecture.py \
        --pretrained_weights /kaggle/input/.../pretrained.pth \
        --train_txt /kaggle/.../train.txt \
        --val_txt   /kaggle/.../val.txt \
        --pop_size 8 --max_iter 20 \
        --proxy_epochs 4 --proxy_data_fraction 0.2

Each candidate is trained for `proxy_epochs` epochs on a fixed random
`proxy_data_fraction` subset of train/val (see
gcnet_search_space.make_fitness_function) -- this is a cheap PROXY signal
to rank candidates against each other, not a final accuracy number. After
the search finishes, take `best_config` (printed and saved to
--out_json) and run a full train.py training run with it to get the real
mIoU.

Cost note: OBL-ADE-RD evaluates 2 * pop_size candidates at initialization
(to pick the fitter of each opposite pair -- see obl_de_rd.py) and
pop_size candidates per iteration after that, so the total number of
proxy-trained models is roughly `pop_size * (max_iter + 2)`. Sanity-check
your GPU time budget with a single candidate first, e.g. by temporarily
setting --pop_size 1 --max_iter 1.

For a cheap first-stage screen, add `--fast_proxy`, use a smaller image
size/data fraction, then re-evaluate the finalists without `--fast_proxy`.
Compatible completed candidates in --log_path are reused on restart.
"""

import argparse
import json

import numpy as np

from gcnet_search_space import DIM, LB, UB, decode_candidate, make_fitness_function
from obl_de_rd import OBLAdaptiveRaindropOptimizer
from train import ModelConfig


def parse_args():
    p = argparse.ArgumentParser(description="OBL-ADE-RD search for GCNet architecture")
    # Data / weights
    p.add_argument("--pretrained_weights", required=True)
    p.add_argument("--train_txt", required=True)
    p.add_argument("--val_txt", required=True)
    p.add_argument("--model_variant", default="fan_dwsa",
                    choices=["fan_dwsa", "fan_only", "dwsa_only"])
    p.add_argument("--dataset_type", default="foggy", choices=["normal", "foggy"])
    p.add_argument("--num_classes", type=int, default=19)
    # Proxy evaluation budget
    p.add_argument("--proxy_epochs", type=int, default=4)
    p.add_argument("--proxy_data_fraction", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--proxy_num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=1024)
    # Optimizer budget
    p.add_argument("--pop_size", type=int, default=8)
    p.add_argument("--max_iter", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    # Output
    p.add_argument("--out_json", default="best_gcnet_arch.json")
    p.add_argument("--log_path", default="search_log.jsonl",
                    help="Every candidate's result is appended here as one JSON "
                         "line the moment it finishes, so progress survives even "
                         "if the process is killed before the search completes "
                         "(Kaggle session limit, disconnect, etc). tail -f this "
                         "file to watch the search live. Pass '' to disable.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fast_proxy", action="store_true",
                   help="Screen candidates faster by disabling Dice and the auxiliary loss. "
                        "Re-evaluate finalists without this flag before full training.")
    return p.parse_args()


def main():
    args = parse_args()

    # train.py enables these inside its main(), which is not executed when
    # architecture search imports Trainer. Enable the same fast paths here.
    import torch
    if str(args.device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_cfg = ModelConfig.get_config(variant=args.model_variant)

    fitness = make_fitness_function(
        base_cfg=base_cfg,
        pretrained_weights_path=args.pretrained_weights,
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        proxy_epochs=args.proxy_epochs,
        proxy_data_fraction=args.proxy_data_fraction,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=(args.img_h, args.img_w),
        dataset_type=args.dataset_type,
        device=args.device,
        log_path=(args.log_path or None),
        proxy_num_workers=args.proxy_num_workers,
        proxy_seed=args.seed,
        fast_proxy=args.fast_proxy,
    )

    opt = OBLAdaptiveRaindropOptimizer(
        obj_func=fitness, dim=DIM, lb=LB, ub=UB,
        pop_size=args.pop_size, max_iter=args.max_iter, seed=args.seed,
    )

    n_candidates = args.pop_size * (args.max_iter + 2)
    print(f"Starting search: pop_size={args.pop_size}, max_iter={args.max_iter}, "
          f"proxy_epochs={args.proxy_epochs}, proxy_data_fraction={args.proxy_data_fraction}")
    print(f"Proxy mode: {'FAST screening' if args.fast_proxy else 'full proxy loss'}; "
          f"workers={args.proxy_num_workers}")
    print(f"Total candidates to evaluate: ~{n_candidates}")
    if args.log_path:
        print(f"Live progress (safe even if this process gets killed): "
              f"tail -f {args.log_path}")
    best_x, best_cost = opt.optimize(verbose=True)

    best_config = decode_candidate(best_x)
    best_miou = 1.0 - best_cost

    print("\n" + "=" * 70)
    print(f"Best proxy mIoU found: {best_miou:.4f}")
    print(f"Best architecture: {best_config}")
    print("=" * 70)
    print("This is a PROXY result (short training, data subset) -- "
          "re-train the winning config with full epochs/data via train.py "
          "to get the real mIoU before drawing conclusions.")

    with open(args.out_json, "w") as f:
        json.dump({
            "best_proxy_miou": best_miou,
            "best_config": best_config,
            "best_x": np.asarray(best_x).tolist(),
            "history_best_cost": [float(c) for c in opt.history_best],
        }, f, indent=2)
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
