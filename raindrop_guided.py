"""Raindrop-guided AdamW learning-rate multipliers.

The outer Raindrop optimizer searches a small, meaningful space (one multiplier
per AdamW parameter group).  Fitness is the cross-entropy after a reversible
virtual AdamW step on a training-only calibration split.  The selected
multipliers then control every real AdamW step in the next epoch.
"""

import json
import math
from pathlib import Path

import numpy as np

from obl_de_rd import OBLAdaptiveRaindropOptimizer


def split_guide_indices(dataset_size, samples, gate_fraction, seed):
    if samples < 64 or samples > dataset_size:
        raise ValueError("raindrop_guide_samples must be between 64 and dataset size")
    if not 0.2 <= gate_fraction <= 0.8:
        raise ValueError("raindrop_guide_gate_fraction must be in [0.2, 0.8]")
    selected = np.random.default_rng(seed).choice(
        dataset_size, size=samples, replace=False)
    gate_size = int(round(samples * gate_fraction))
    return np.sort(selected[gate_size:]), np.sort(selected[:gate_size])


class RaindropAdamWGuide:
    """Select robust per-group AdamW step multipliers before each epoch."""

    def __init__(self, model, optimizer, gradient_loader, gate_loader, device,
                 pop_size=4, max_iter=1, factor_min=0.5, factor_max=1.5,
                 gradient_batches=4, min_ce_gain=1e-4, seed=42,
                 log_path=None):
        import torch

        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("Raindrop-guided training currently requires AdamW")
        if pop_size < 4 or max_iter < 1 or gradient_batches < 1:
            raise ValueError("Invalid Raindrop population or iteration budget")
        if not 0 < factor_min < 1 < factor_max <= 3:
            raise ValueError("Require 0 < factor_min < 1 < factor_max <= 3")
        if min_ce_gain < 0:
            raise ValueError("min_ce_gain must be non-negative")
        self.model = model
        self.optimizer = optimizer
        self.gradient_loader = gradient_loader
        self.gate_loader = gate_loader
        self.device = torch.device(device)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.gradient_batches = gradient_batches
        self.min_ce_gain = min_ce_gain
        self.seed = seed
        self.log_path = Path(log_path) if log_path else None
        self.group_names = [group.get("name", f"group_{i}")
                            for i, group in enumerate(optimizer.param_groups)]
        if len(set(self.group_names)) != len(self.group_names):
            raise ValueError("Optimizer parameter-group names must be unique")

    def _loss(self, loader, backward=False, max_batches=None):
        import torch
        import torch.nn.functional as F

        total = 0.0
        count = 0
        context = torch.enable_grad() if backward else torch.inference_mode()
        with context:
            for images, labels in loader:
                if max_batches is not None and count >= max_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()
                if labels.ndim == 4:
                    labels = labels.squeeze(1)
                with torch.autocast(device_type="cuda",
                                    enabled=self.device.type == "cuda"):
                    logits = self.model(images)
                    logits = F.interpolate(logits, size=labels.shape[-2:],
                                           mode="bilinear", align_corners=False)
                    loss = F.cross_entropy(logits, labels, ignore_index=255)
                if backward:
                    loss.backward()
                total += float(loss.detach())
                count += 1
        if count == 0:
            raise ValueError("Raindrop guidance loader is empty")
        return total / count

    def _predicted_updates(self):
        """Build the exact next AdamW direction without mutating optimizer state."""
        import torch

        updates = []
        for group in self.optimizer.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            group_updates = []
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach().float()
                state = self.optimizer.state.get(param, {})
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                exp_avg = (torch.zeros_like(grad) if exp_avg is None
                           else exp_avg.detach().float())
                exp_avg_sq = (torch.zeros_like(grad) if exp_avg_sq is None
                              else exp_avg_sq.detach().float())
                state_step = state.get("step", 0)
                if torch.is_tensor(state_step):
                    state_step = int(state_step.item())
                step = int(state_step) + 1
                next_avg = beta1 * exp_avg + (1 - beta1) * grad
                next_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()
                corrected_avg = next_avg / (1 - beta1 ** step)
                corrected_sq = next_sq / (1 - beta2 ** step)
                delta = -lr * corrected_avg / (corrected_sq.sqrt() + eps)
                if weight_decay:
                    delta = delta - lr * weight_decay * param.detach().float()
                group_updates.append((param, param.detach().clone(),
                                      delta.to(dtype=param.dtype)))
            updates.append(group_updates)
        return updates

    @staticmethod
    def _apply(updates, factors):
        import torch

        with torch.no_grad():
            for factor, group_updates in zip(factors, updates):
                for param, base, delta in group_updates:
                    param.copy_(base + float(factor) * delta)

    @staticmethod
    def _restore(updates):
        import torch

        with torch.no_grad():
            for group_updates in updates:
                for param, base, _ in group_updates:
                    param.copy_(base)

    def select(self, epoch):
        import torch

        was_training = self.model.training
        self.model.eval()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            self._loss(self.gradient_loader, backward=True,
                       max_batches=self.gradient_batches)
            used_batches = min(self.gradient_batches, len(self.gradient_loader))
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.div_(used_batches)
            updates = self._predicted_updates()
            if not all(updates):
                raise RuntimeError("At least one AdamW group has no proxy gradient")
            self.optimizer.zero_grad(set_to_none=True)
            observations = {}

            def evaluate(vector):
                key = tuple(np.asarray(vector, dtype=float).round(8))
                if key not in observations:
                    self._apply(updates, vector)
                    try:
                        ce = self._loss(self.gate_loader, backward=False)
                    finally:
                        self._restore(updates)
                    regularizer = 1e-5 * float(np.square(np.log(vector)).mean())
                    observations[key] = {"ce": ce, "cost": ce + regularizer}
                    print(f"  RD guide candidate {len(observations)}: CE={ce:.6f} "
                          f"factors={np.asarray(vector).round(3).tolist()}",
                          flush=True)
                return observations[key]["cost"]

            ones = np.ones(len(self.group_names), dtype=float)
            baseline_cost = evaluate(ones)

            def fitness(population):
                return np.asarray([evaluate(vector) for vector in population],
                                  dtype=float)

            optimizer = OBLAdaptiveRaindropOptimizer(
                obj_func=fitness, dim=len(self.group_names),
                lb=np.full(len(self.group_names), self.factor_min),
                ub=np.full(len(self.group_names), self.factor_max),
                pop_size=self.pop_size, max_iter=self.max_iter,
                seed=self.seed + int(epoch))
            best_x, best_cost = optimizer.optimize(verbose=True)
            best_key = tuple(np.asarray(best_x, dtype=float).round(8))
            best_ce = observations[best_key]["ce"]
            baseline_ce = observations[tuple(ones.round(8))]["ce"]
            accepted = (math.isfinite(best_ce)
                        and best_ce < baseline_ce - self.min_ce_gain
                        and best_cost < baseline_cost)
            factors = np.asarray(best_x if accepted else ones, dtype=float)
            record = {
                "epoch": int(epoch) + 1,
                "groups": self.group_names,
                "baseline_ce": baseline_ce,
                "best_ce": best_ce,
                "ce_gain": baseline_ce - best_ce,
                "accepted": bool(accepted),
                "selected_factors": factors.tolist(),
                "candidate_factors": np.asarray(best_x).tolist(),
                "evaluations": len(observations),
                "history_best_cost": [float(x) for x in optimizer.history_best],
            }
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record) + "\n")
            print(f"Raindrop guide epoch {epoch + 1}: CE {baseline_ce:.6f} -> "
                  f"{best_ce:.6f}; accepted={accepted}; "
                  f"multipliers={dict(zip(self.group_names, factors.round(3)))}",
                  flush=True)
            return dict(zip(self.group_names, factors.tolist())), record
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            if was_training:
                self.model.train()
