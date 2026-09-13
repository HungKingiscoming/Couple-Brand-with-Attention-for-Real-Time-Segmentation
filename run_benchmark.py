"""
Compare RD (baseline), ADE-RD, and the OBL-based improvements (see
obl_de_rd.py) on standard benchmark functions.

Usage:
    python run_benchmark.py

Runs N_RUNS independent trials per function per algorithm, reports mean/std
of the best cost found, and a Wilcoxon signed-rank test p-value (each
algorithm vs the RD baseline, paired by seed) to check whether the
difference is statistically significant (not just noise).

Algorithms compared, in increasing order of what they add on top of RD:
  RD          - baseline (raindrop.py)
  ADE-RD      - + adaptive diversity-based evaporation/reinjection (ade_rd.py)
  OBL-RD      - + opposition-based-learning initialization only (obl_de_rd.py)
  OBL+ADE-RD  - both of the above combined (obl_de_rd.py)

(obl_de_rd.py also defines DERaindropOptimizer and
OBLDEAdaptiveRaindropOptimizer, which additionally blend a
differential-evolution-style term into exploitation -- they are NOT
included here because, empirically, that term makes results worse on
most of these functions rather than better; see the module docstring
and class docstrings in obl_de_rd.py for the full comparison that led
to leaving it out of the recommended combination.)
"""

import numpy as np
from scipy.stats import wilcoxon

from raindrop import RaindropOptimizer
from ade_rd import ADERaindropOptimizer
from obl_de_rd import OBLRaindropOptimizer, OBLAdaptiveRaindropOptimizer
from benchmark_functions import BENCHMARKS

N_RUNS = 20        # independent trials per (function, algorithm) pair
POP_SIZE = 20
MAX_ITER = 200

ALGORITHMS = {
    "RD":         RaindropOptimizer,
    "ADE-RD":     ADERaindropOptimizer,
    "OBL-RD":     OBLRaindropOptimizer,
    "OBL+ADE-RD": OBLAdaptiveRaindropOptimizer,
}


def run_trials(optimizer_cls, func, lb, ub, dim, n_runs, **kwargs):
    results = []
    for run in range(n_runs):
        opt = optimizer_cls(
            obj_func=func, dim=dim, lb=lb, ub=ub,
            pop_size=POP_SIZE, max_iter=MAX_ITER,
            seed=run, **kwargs
        )
        _, best_cost = opt.optimize(verbose=False)
        results.append(best_cost)
    return np.array(results)


def _p_value(a, b):
    # Wilcoxon requires non-identical paired samples; guard against the
    # degenerate all-equal case (can happen on trivial sphere runs).
    try:
        return f"{wilcoxon(a, b)[1]:.4f}"
    except ValueError:
        return "n/a"


def main():
    header = f"{'Function':<12} " + " ".join(f"{name:<16}" for name in ALGORITHMS)
    print(header)
    print("-" * len(header))

    all_results = {}
    for fname, (func, lb, ub, dim) in BENCHMARKS.items():
        all_results[fname] = {
            name: run_trials(cls, func, lb, ub, dim, N_RUNS)
            for name, cls in ALGORITHMS.items()
        }
        row = " ".join(f"{all_results[fname][name].mean():<16.4e}" for name in ALGORITHMS)
        print(f"{fname:<12} {row}")

    print()
    print("Wilcoxon p-value vs RD baseline (paired by seed):")
    print(f"{'Function':<12} " + " ".join(f"{name:<12}" for name in ALGORITHMS if name != "RD"))
    for fname in BENCHMARKS:
        rd = all_results[fname]["RD"]
        row = " ".join(
            f"{_p_value(all_results[fname][name], rd):<12}"
            for name in ALGORITHMS if name != "RD"
        )
        print(f"{fname:<12} {row}")

    print()
    print("Mean improvement %% of OBL+ADE-RD vs RD, and vs ADE-RD:")
    for fname in BENCHMARKS:
        rd_mean = all_results[fname]["RD"].mean()
        ade_mean = all_results[fname]["ADE-RD"].mean()
        best_mean = all_results[fname]["OBL+ADE-RD"].mean()
        imp_rd = 100 * (rd_mean - best_mean) / (abs(rd_mean) + 1e-12)
        imp_ade = 100 * (ade_mean - best_mean) / (abs(ade_mean) + 1e-12)
        print(f"  {fname:<12} vs RD: {imp_rd:>7.2f}%   vs ADE-RD: {imp_ade:>7.2f}%")


if __name__ == "__main__":
    main()
