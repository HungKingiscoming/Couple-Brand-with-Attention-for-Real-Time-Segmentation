"""
Standard unconstrained benchmark functions, vectorized to accept an (N, d)
matrix of candidate solutions and return an (N,) array of fitness values.
All are minimization problems with known global minimum at x* (given below).

These are classic multimodal/unimodal test functions (not the exact
CEC-BC-2020 suite used in the original RD paper, which is not freely
redistributable) — they're used here purely to sanity-check that ADE-RD's
modifications actually help vs. hurt relative to the RD baseline, before
spending any real GPU time on the GCNet application.
"""

import numpy as np


def sphere(X):
    """Unimodal. Global min f(0,...,0) = 0. Typical bounds: [-100, 100]^d."""
    return np.sum(X ** 2, axis=1)


def rastrigin(X):
    """Highly multimodal. Global min f(0,...,0) = 0. Bounds: [-5.12, 5.12]^d."""
    d = X.shape[1]
    return 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=1)


def ackley(X):
    """Multimodal with a deep global basin. Global min f(0,...,0) = 0.
    Bounds: [-32.768, 32.768]^d."""
    d = X.shape[1]
    sum1 = np.sum(X ** 2, axis=1)
    sum2 = np.sum(np.cos(2 * np.pi * X), axis=1)
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + 20 + np.e


def rosenbrock(X):
    """Narrow curved valley, hard for local-only search. Global min
    f(1,...,1) = 0. Bounds: [-30, 30]^d (using a wide but common choice)."""
    x_i = X[:, :-1]
    x_ip1 = X[:, 1:]
    return np.sum(100 * (x_ip1 - x_i ** 2) ** 2 + (1 - x_i) ** 2, axis=1)


def griewank(X):
    """Multimodal, many widespread local optima. Global min
    f(0,...,0) = 0. Bounds: [-600, 600]^d."""
    d = X.shape[1]
    sum_term = np.sum(X ** 2, axis=1) / 4000.0
    idx = np.arange(1, d + 1)
    prod_term = np.prod(np.cos(X / np.sqrt(idx)), axis=1)
    return sum_term - prod_term + 1


BENCHMARKS = {
    # name: (function, lower_bound, upper_bound, recommended_dim)
    "sphere":     (sphere,     -100.0,    100.0,    30),
    "rastrigin":  (rastrigin,  -5.12,     5.12,     30),
    "ackley":     (ackley,     -32.768,   32.768,   30),
    "rosenbrock": (rosenbrock, -30.0,     30.0,     30),
    "griewank":   (griewank,   -600.0,    600.0,    30),
}
