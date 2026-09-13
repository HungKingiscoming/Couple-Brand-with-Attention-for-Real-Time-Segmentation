"""
ADE-RD: Adaptive Diversity-based Evaporation with Reinjection.

An improved variant of the Raindrop Optimizer (RD, 2025) that targets a
weakness the original authors explicitly acknowledge: the evaporation
schedule is a fixed linear function of iteration count, "which may not be
suitable for every objective-function landscape", and the dual-phase
explore/exploit transition can converge prematurely in highly multimodal
search spaces with dense local optima.

Two changes relative to the RD baseline (see raindrop.py):

1. Adaptive (nonlinear) evaporation rate.
   Instead of evap(t) depending only on iteration progress, it also depends
   on the *actual* population diversity D(t) — a normalized measure of how
   spread out the current population is (std per dimension / search range,
   averaged over dimensions):

        D(t) = (1/d) * sum_j  std(X[:, j]) / (ub_j - lb_j)          in [0, 1]

        evap(t) = evap_base(t) * (1 + lambda_evap * (1 - D(t)))

   When the population has collapsed (D(t) -> 0, a sign of premature
   convergence / stagnation), evap(t) is amplified — more individuals are
   evaporated exactly when the risk of being stuck is highest, instead of
   following a fixed schedule blind to the actual state of the search.

2. Guided reinjection instead of blind random re-initialization.
   The baseline RD simply re-initializes evaporated individuals uniformly
   at random over the whole domain. ADE-RD instead reinjects them away from
   the current population centroid, with a repulsion strength that also
   scales with (1 - D(t)):

        repulsion(t) = 1 + lambda_repel * (1 - D(t))
        x_new        = centroid + direction * radius
        radius        = repulsion(t) * rand() * 0.5*(ub - lb)

   where `direction` is a random unit vector. This deliberately pushes new
   individuals into presently under-explored regions instead of anywhere in
   the domain, which is more useful specifically when diversity has
   collapsed (uniform re-init would very likely just fall back near the
   existing cluster in a high-dimensional space).

Both changes live entirely in `_evaporate` (overriding
RaindropOptimizer._evaporate); `optimize()` itself is inherited verbatim
from the RD baseline, so the rest of the algorithm (splash/diversion,
convergence, overflow) is guaranteed to be identical to RD by construction
rather than by keeping two copies of the loop in sync — any performance
difference in the benchmark comparison is attributable specifically to
these two mechanisms.
"""

import numpy as np
from raindrop import RaindropOptimizer


class ADERaindropOptimizer(RaindropOptimizer):
    def __init__(self,
                 *args,
                 lambda_evap=1.0,
                 lambda_repel=1.5,
                 evap_min=0.05,
                 evap_max=0.6,
                 **kwargs):
        """
        Additional parameters (on top of RaindropOptimizer):

        lambda_evap : float
            Amplification strength of evaporation when diversity is low.
        lambda_repel : float
            Amplification strength of the reinjection repulsion radius when
            diversity is low.
        evap_min, evap_max : float
            Hard bounds on the adaptive evaporation fraction, so the
            population is never fully wiped out or left untouched.
        """
        super().__init__(*args, **kwargs)
        self.lambda_evap = lambda_evap
        self.lambda_repel = lambda_repel
        self.evap_min = evap_min
        self.evap_max = evap_max

    def _evaporate(self, idx, X, X_new, frac, D_t, centroid):
        # ---- (1) Adaptive evaporation rate ---- #
        evap_base = self.evap_initial + (self.evap_final - self.evap_initial) * frac
        evap = evap_base * (1 + self.lambda_evap * (1 - D_t))
        evap = float(np.clip(evap, self.evap_min, self.evap_max))
        n_evap = int(round(len(idx) * evap))

        if n_evap > 0:
            evap_idx = self.rng.choice(idx, size=n_evap, replace=False)

            # ---- (2) Guided reinjection, away from centroid ---- #
            repulsion = 1 + self.lambda_repel * (1 - D_t)
            raw_dir = self.rng.standard_normal((n_evap, self.d))
            norm = np.linalg.norm(raw_dir, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            unit_dir = raw_dir / norm
            radius = repulsion * self.rng.random((n_evap, 1)) * \
                0.5 * (self.ub - self.lb)
            X_new[evap_idx] = self._clip_rebound(centroid + unit_dir * radius)

        return X_new
