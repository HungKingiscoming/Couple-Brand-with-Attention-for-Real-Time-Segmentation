"""
Raindrop Optimizer (RD) — baseline implementation.

Reference: "Raindrop optimizer: a novel physics-inspired metaheuristic for
global optimization" (Scientific Reports, 2025).

This is a from-scratch reimplementation based on the equations described in
the paper (initialization, rebound boundary handling, explore/exploit phase
split, splash/diversion, evaporation, convergence, overflow). Some notational
details in the original paper are terse (e.g. the exact definition of the
"local best" used in the diversion term); where ambiguous, a reasonable and
clearly-documented interpretation is used. This baseline exists mainly as a
faithful reference point to measure the improvement of ADE-RD (see
ade_rd.py) against.

The whole population is stored as a single (N, d) numpy matrix — every
update below is vectorized (no per-individual Python loop), so the internal
cost of the optimizer itself stays negligible even for large N. This does
NOT reduce the cost of the fitness function itself (see the accompanying
write-up on why matrix ops don't help with expensive DL fitness evaluation).

Three mechanisms are factored out into their own overridable methods —
`_initialize_population`, `_exploit`, and `_evaporate` — specifically so
subclasses (ADERaindropOptimizer and others in ade_rd.py) can replace just
one mechanism at a time without copy-pasting the rest of the loop (splash,
diversion, overflow escape stay shared and identical by construction).
"""

import numpy as np


class RaindropOptimizer:
    def __init__(self,
                 obj_func,
                 dim,
                 lb,
                 ub,
                 pop_size=20,
                 max_iter=100,
                 rb=0.5,
                 gamma=0.5,
                 evap_initial=0.10,
                 evap_final=0.30,
                 kappa=5,
                 seed=None):
        """
        Parameters
        ----------
        obj_func : callable
            Takes an (N, d) array of candidate positions, returns an
            (N,) array of fitness values (lower = better, minimization).
        dim : int
            Number of dimensions (search variables).
        lb, ub : float or array-like of shape (dim,)
            Lower / upper bounds per dimension.
        pop_size : int
            Number of raindrops (population size).
        max_iter : int
            Number of iterations.
        rb : float
            Rebound coefficient used when a position goes out of bounds.
        gamma : float
            Step size coefficient for the convergence (exploitation) move.
        evap_initial, evap_final : float
            Start / end evaporation fraction (fraction of population
            evaporated per iteration during exploration), linearly
            interpolated over iterations in the baseline.
        kappa : int
            Number of consecutive iterations a solution can repeat as the
            best before the overflow (stagnation-escape) mechanism fires.
        seed : int or None
            RNG seed for reproducibility.
        """
        self.f = obj_func
        self.d = dim
        self.lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)
        self.N = pop_size
        self.T = max_iter
        self.rb = rb
        self.gamma = gamma
        self.evap_initial = evap_initial
        self.evap_final = evap_final
        self.kappa = kappa
        self.rng = np.random.default_rng(seed)

        self.history_best = []      # best cost per iteration
        self.history_diversity = []  # diversity *entering* each iteration (diagnostic)

        self._levy_sigma_cache = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _clip_rebound(self, X):
        """
        Eq. (2): rebound handling instead of plain clipping.

        The violation past the boundary is normalized by the search range
        before squaring (`delta` below is in [0, 1]), so the rebound offset
        is always a bounded fraction (at most `rb`) of the domain span —
        for a small violation this is still the same soft quadratic bounce
        as before, but a large violation (a long Lévy jump, or ADE-RD's
        guided-reinjection radius, which can exceed the domain span) no
        longer overshoots past the opposite boundary and collapse into a
        hard clip; it saturates at "bounce back rb of the way across the
        domain" instead, which is what an unbounded quadratic in raw
        (unnormalized) violation units used to silently degrade into.
        """
        span = self.ub - self.lb
        below = X < self.lb
        above = X > self.ub
        eps = 1e-6
        if below.any():
            delta = np.minimum((self.lb - X) / span, 1.0)
            X = np.where(below,
                         self.lb + self.rb * span * delta ** 2 + eps,
                         X)
        if above.any():
            delta = np.minimum((X - self.ub) / span, 1.0)
            X = np.where(above,
                         self.ub - self.rb * span * delta ** 2 - eps,
                         X)
        # Final safety clip in case of floating-point edge cases at rb >= 1.
        return np.clip(X, self.lb, self.ub)

    def _levy_sigma(self, beta):
        """sigma_u depends only on beta, so cache it instead of recomputing
        math.gamma() on every call to `_levy` (every iteration, for every
        splashing individual)."""
        if beta not in self._levy_sigma_cache:
            self._levy_sigma_cache[beta] = (
                math_gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                (math_gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
            ) ** (1 / beta)
        return self._levy_sigma_cache[beta]

    def _levy(self, n, d, beta=1.5):
        """Standard Mantegna Lévy flight sampler, shape (n, d)."""
        sigma_u = self._levy_sigma(beta)
        u = self.rng.normal(0, sigma_u, size=(n, d))
        v = self.rng.normal(0, 1, size=(n, d))
        return u / (np.abs(v) ** (1 / beta) + 1e-12)

    def _diversity(self, X):
        """Mean per-dimension std, normalized by the search range in [0,1]."""
        span = (self.ub - self.lb)
        span[span == 0] = 1.0
        return float(np.mean(X.std(axis=0) / span))

    def _initialize_population(self):
        """
        Eq (1): uniform random initialization over [lb, ub].

        Returns (X, cost_or_None). `cost` is None here, so `optimize()`
        evaluates `self.f(X)` itself. Overridable (e.g. by an opposition-
        based-learning variant that must evaluate both each point and its
        mirror image to keep whichever of the pair is fitter) — such a
        variant can hand back the already-computed cost for the surviving
        half of its 2N evaluations instead of making `optimize()` pay for
        a redundant third evaluation of the same N points.
        """
        X = self.lb + self.rng.random((self.N, self.d)) * (self.ub - self.lb)
        return X, None

    def _exploit(self, exploit_idx, X, X_new, cost, frac, best_x):
        """
        Convergence (Eq. 15-16): move toward a good target. First half of
        the run: target = random draw from top-20% best. Second half:
        target = global best (sharper exploitation).

        Overridable (e.g. by a variant that blends in a differential-
        evolution-style difference vector between two other population
        members, which tracks curved/ridge-shaped landscapes such as
        Rosenbrock far better than a straight move-toward-target step).
        """
        if frac < 0.5:
            k = max(1, int(0.2 * self.N))
            top_idx = np.argsort(cost)[:k]
            targets = X[self.rng.choice(top_idx, size=len(exploit_idx))]
        else:
            targets = np.tile(best_x, (len(exploit_idx), 1))

        X_new[exploit_idx] = X[exploit_idx] + self.gamma * (targets - X[exploit_idx])
        return self._clip_rebound(X_new[exploit_idx])

    def _evaporate(self, idx, X, X_new, frac, D_t, centroid):
        """
        Baseline RD evaporation (Eq. 12-13): drop a fraction of the
        *explored* individuals — scheduled purely as a linear function of
        iteration progress `frac` — and re-initialize them uniformly at
        random over the whole domain.

        `D_t` (population diversity) and `centroid` are accepted but
        unused here; they exist so ADERaindropOptimizer can override this
        method with an adaptive-rate, centroid-repelling version (see
        ade_rd.py) while reusing everything else in `optimize()` verbatim.
        """
        evap = self.evap_initial + (self.evap_final - self.evap_initial) * frac
        n_evap = int(round(len(idx) * evap))
        if n_evap > 0:
            evap_idx = self.rng.choice(idx, size=n_evap, replace=False)
            X_new[evap_idx] = self.lb + self.rng.random((n_evap, self.d)) * (self.ub - self.lb)
        return X_new

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def optimize(self, verbose=False):
        X, cost = self._initialize_population()  # Eq (1)
        if cost is None:
            cost = self.f(X)

        best_idx = np.argmin(cost)
        best_x, best_cost = X[best_idx].copy(), cost[best_idx]
        repeat_count = 0
        prev_best_cost = best_cost

        for it in range(self.T):
            frac = it / max(self.T, 1)
            P = max(0.1, (1 - frac)) * self.rng.random()   # Eq (3)

            # Diversity/centroid measured *entering* this iteration (before
            # any move below). ADE-RD needs this value up front to drive its
            # adaptive evaporation and guided reinjection; the baseline logs
            # the same quantity (rather than diversity *after* the update,
            # as an earlier version of this file did) so that
            # `history_diversity` means the same thing, at the same point
            # in the loop, for both optimizers and is directly comparable.
            D_t = self._diversity(X)
            centroid = X.mean(axis=0)

            explore_mask = self.rng.random(self.N) < P     # per-individual decision
            X_new = X.copy()

            # -------------------- Exploration -------------------- #
            n_explore = explore_mask.sum()
            if n_explore > 0:
                idx = np.where(explore_mask)[0]
                use_splash = self.rng.random(n_explore) >= 0.5

                # Splash: Lévy-flight jump (Eq. 6, upper branch). `_levy`
                # already returns the full Mantegna Lévy step (u / |v|^(1/beta));
                # it is used directly here rather than multiplied by another
                # independent standard-normal sample, which would not be part
                # of the canonical Lévy-flight formula and would just inject
                # an extra, unmotivated source of randomness into the step size.
                splash_idx = idx[use_splash]
                if len(splash_idx) > 0:
                    step = self._levy(len(splash_idx), self.d)
                    X_new[splash_idx] = X[splash_idx] + P * step

                # Diversion: move along direction to a random better peer (Eq. 6, lower branch)
                divert_idx = idx[~use_splash]
                if len(divert_idx) > 0:
                    peers = self.rng.integers(0, self.N, size=len(divert_idx))
                    direction = X[peers] - X[divert_idx]
                    norm = np.linalg.norm(direction, axis=1, keepdims=True)
                    norm[norm == 0] = 1.0
                    unit_dir = direction / norm
                    shunt = self.rng.standard_normal((len(divert_idx), 1)) * (1 - frac)
                    X_new[divert_idx] = X[divert_idx] + unit_dir * shunt

                X_new[idx] = self._clip_rebound(X_new[idx])

                # Evaporation (Eq. 12-13) — see `_evaporate` (overridden by
                # ADERaindropOptimizer for the adaptive/guided version).
                X_new = self._evaporate(idx, X, X_new, frac, D_t, centroid)

            # -------------------- Exploitation -------------------- #
            exploit_idx = np.where(~explore_mask)[0]
            if len(exploit_idx) > 0:
                X_new[exploit_idx] = self._exploit(exploit_idx, X, X_new, cost, frac, best_x)

            # -------------------- Evaluate & select -------------------- #
            cost_new = self.f(X_new)
            improved = cost_new < cost
            X = np.where(improved[:, None], X_new, X)
            cost = np.where(improved, cost_new, cost)

            gen_best_idx = np.argmin(cost)
            if cost[gen_best_idx] < best_cost:
                best_cost = cost[gen_best_idx]
                best_x = X[gen_best_idx].copy()

            # Overflow / stagnation escape (Eq. 17)
            if best_cost >= prev_best_cost - 1e-12:
                repeat_count += 1
            else:
                repeat_count = 0
            prev_best_cost = best_cost

            if repeat_count >= self.kappa:
                k = min(10, self.N)
                scatter_idx = self.rng.choice(self.N, size=k, replace=False)
                offset = (self.ub - self.lb) * P * (0.5 - self.rng.random((k, self.d)))
                X[scatter_idx] = self._clip_rebound(X[scatter_idx] + offset)
                cost[scatter_idx] = self.f(X[scatter_idx])
                # The scatter step can occasionally land on a better point
                # than the current best purely by chance; unlike the main
                # explore/exploit update above, nothing would otherwise
                # notice. That matters most on the very last iteration,
                # where there is no next iteration left to pick it up via
                # `gen_best_idx` — without this check the improvement would
                # be silently discarded when `optimize()` returns.
                scatter_best = scatter_idx[np.argmin(cost[scatter_idx])]
                if cost[scatter_best] < best_cost:
                    best_cost = cost[scatter_best]
                    best_x = X[scatter_best].copy()
                repeat_count = 0

            self.history_best.append(best_cost)
            self.history_diversity.append(D_t)

            if verbose and (it % max(1, self.T // 10) == 0 or it == self.T - 1):
                print(f"  iter {it+1:>4}/{self.T}  best={best_cost:.6g}  "
                      f"diversity={D_t:.3f}  P={P:.3f}")

        return best_x, best_cost


# Small local re-implementation of math.gamma to avoid importing scipy just
# for the Levy-flight sigma constant.
def math_gamma(x):
    import math
    return math.gamma(x)
