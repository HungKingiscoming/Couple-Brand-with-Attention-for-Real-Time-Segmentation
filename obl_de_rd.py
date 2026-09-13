"""
Further improvements to RD/ADE-RD, borrowed from well-established techniques
in the broader metaheuristic-optimization literature rather than invented
from scratch — RD itself is too new (Scientific Reports, Sept 2025) for a
published "improved RD" variant to exist yet, so this applies two
widely-validated, independently-published hybridization techniques to the
same RD/ADE-RD framework already in raindrop.py / ade_rd.py:

1. Opposition-Based Learning (OBL) at initialization.
   Introduced by Tizhoosh (2005) and since reused across dozens of
   metaheuristic hybrids (OBL-GWO, OBL-PSO, Arctic Puffin Optimization
   w/ OBL, etc.): for every randomly sampled point x, also consider its
   mirror image about the domain center, x_opp = lb + ub - x, and keep
   whichever of the pair is fitter. This tends to produce a better-
   covering (and often already-fitter) starting population than plain
   uniform sampling, at the cost of one extra batch of fitness
   evaluations (2N instead of N) done ONCE, at initialization only.

   This targets the premature-convergence weakness that shows up most on
   landscapes with many evenly-spread local optima (Rastrigin, Ackley) —
   see benchmark_functions.py's docstrings — where a wider, more even
   initial spread lowers the chance the whole population starts out
   clustered near the same local basin.

2. A differential-evolution-style difference vector blended into the
   exploitation move.
   Storn & Price's DE (1997) mutation `x + F*(x_r1 - x_r2)` is a large
   part of *why* DE is well known to handle narrow, curved valleys
   (Rosenbrock-shaped landscapes) much better than algorithms that only
   move straight toward a single target point: as the population spreads
   out along a valley, the difference vector between two arbitrary
   members tends to point *along* the valley rather than across it, so
   the step naturally follows the ridge instead of overshooting across
   it. RD/ADE-RD's exploitation step (`_exploit` in raindrop.py) is a
   plain "move toward top-20%-best/global-best" — this mixin blends in a
   scaled DE-style difference term on top of that.

Both are implemented as small overrides of the two hooks raindrop.py
exposes for exactly this purpose (`_initialize_population`, `_exploit`),
so they can be tested in isolation (`OBLRaindropOptimizer`,
`DERaindropOptimizer`) or combined with each other and with ADE-RD's
adaptive evaporation (`OBLDEAdaptiveRaindropOptimizer`) via ordinary
multiple inheritance — none of them touch `_evaporate`, so they compose
with ADERaindropOptimizer without any copy-pasted loop code.
"""

import numpy as np
from raindrop import RaindropOptimizer
from ade_rd import ADERaindropOptimizer


class OppositionInitMixin:
    """Opposition-Based Learning (Tizhoosh, 2005) at initialization only."""

    def _initialize_population(self):
        X = self.lb + self.rng.random((self.N, self.d)) * (self.ub - self.lb)
        X_opp = self.lb + self.ub - X
        combined = np.vstack([X, X_opp])
        cost_combined = self.f(combined)
        order = np.argsort(cost_combined)[:self.N]
        # Hand back the already-computed cost for the surviving half so
        # optimize() doesn't pay for a redundant third evaluation of the
        # same N points (see RaindropOptimizer._initialize_population).
        return combined[order], cost_combined[order]


class DifferentialExploitMixin:
    """
    Blends a DE/rand/1-style difference vector into the exploitation move.

    `de_f` is DE's classic scale factor F (Storn & Price recommend
    F in [0.4, 1.0]; 0.5 is the standard default used here).
    """

    def __init__(self, *args, de_f=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.de_f = de_f

    def _exploit(self, exploit_idx, X, X_new, cost, frac, best_x):
        if frac < 0.5:
            k = max(1, int(0.2 * self.N))
            top_idx = np.argsort(cost)[:k]
            targets = X[self.rng.choice(top_idx, size=len(exploit_idx))]
        else:
            targets = np.tile(best_x, (len(exploit_idx), 1))

        n = len(exploit_idx)
        r1 = self.rng.integers(0, self.N, size=n)
        r2 = self.rng.integers(0, self.N, size=n)
        diff = X[r1] - X[r2]

        X_new[exploit_idx] = (
            X[exploit_idx]
            + self.gamma * (targets - X[exploit_idx])
            + self.de_f * diff
        )
        return self._clip_rebound(X_new[exploit_idx])


# --------------------------------------------------------------------- #
# Single-mechanism variants, for isolating each improvement's effect
# --------------------------------------------------------------------- #

class OBLRaindropOptimizer(OppositionInitMixin, RaindropOptimizer):
    """RD baseline + opposition-based-learning initialization ONLY."""
    pass


class DERaindropOptimizer(DifferentialExploitMixin, RaindropOptimizer):
    """RD baseline + differential-evolution-style exploitation ONLY."""
    pass


# --------------------------------------------------------------------- #
# Combined variant: ADE-RD's adaptive evaporation + both improvements
# --------------------------------------------------------------------- #

class OBLDEAdaptiveRaindropOptimizer(OppositionInitMixin, DifferentialExploitMixin,
                                      ADERaindropOptimizer):
    """
    ADE-RD (adaptive diversity-based evaporation + guided reinjection)
    plus opposition-based-learning init and DE-style exploitation. Every
    mechanism lives in its own overridable hook, so this class adds no
    loop code of its own — it is purely the composition of
    OppositionInitMixin._initialize_population,
    DifferentialExploitMixin._exploit, and
    ADERaindropOptimizer._evaporate on top of RaindropOptimizer.optimize().

    NOTE: empirically (see run_benchmark.py / the accompanying comparison),
    the DE-style exploitation term makes results *worse* on most of the
    5 benchmark functions rather than better -- the plain
    `gamma*(target-x)` move already handles convergence well, and adding
    `de_f*(x_r1 - x_r2)` on top of it just injects extra noise that
    outweighs its valley-following benefit here. This class is kept for
    reference/comparison, but OBLAdaptiveRaindropOptimizer (below, without
    the DE term) is the one that actually wins across the benchmark suite.
    """
    pass


class OBLAdaptiveRaindropOptimizer(OppositionInitMixin, ADERaindropOptimizer):
    """
    ADE-RD's adaptive diversity-based evaporation + guided reinjection,
    plus opposition-based-learning initialization -- WITHOUT the
    DE-style exploitation term (see the note on OBLDEAdaptiveRaindropOptimizer
    above for why it was dropped). This is the combination that performed
    best empirically across the benchmark suite.
    """
    pass
