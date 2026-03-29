"""Gradual magnitude pruning callback.

Prunes weights with the smallest absolute magnitude on a configurable
schedule (one-shot, linear, or cubic).  Binary masks are enforced via
forward pre-hooks during training, then made permanent at train end.

Env vars:
    PRUNE_SPARSITY: Target fraction of weights to zero (default 0.3).
    PRUNE_SCHEDULE: "one_shot", "linear", or "cubic" (default "cubic").
    PRUNE_START_STEP: Step at which pruning begins (default 500).
    PRUNE_END_STEP: Step at which target sparsity is reached (default 15000).
    PRUNE_FREQUENCY: Steps between pruning updates (default 100).
    PRUNE_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low,lm_head").
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class MagnitudePruningCallback(TrainingCallback):
    """Gradual magnitude pruning with one-shot / linear / cubic schedules."""

    priority = 8

    def __init__(self, **kwargs: Any) -> None:
        self.target_sparsity = float(os.environ.get("PRUNE_SPARSITY", "0.3"))
        self.schedule = os.environ.get("PRUNE_SCHEDULE", "cubic")
        self.start_step = int(os.environ.get("PRUNE_START_STEP", "500"))
        self.end_step = int(os.environ.get("PRUNE_END_STEP", "15000"))
        self.frequency = int(os.environ.get("PRUNE_FREQUENCY", "100"))
        exclude_raw = os.environ.get("PRUNE_EXCLUDE_PATTERNS", "tok_emb,embed_low,lm_head")
        self.exclude_patterns = tuple(p.strip() for p in exclude_raw.split(",") if p.strip())

        # Populated during training
        self._masks: dict[str, Any] = {}       # name -> Tensor mask
        self._hooks: list[Any] = []             # RemovableHandle list
        self._layer_names: list[str] = []       # ordered prunable layer names

    # ------------------------------------------------------------------
    # Schedule helpers
    # ------------------------------------------------------------------

    def _current_sparsity(self, step: int) -> float:
        """Compute the sparsity target at *step* according to the schedule."""
        if step >= self.end_step or self.schedule == "one_shot":
            return self.target_sparsity
        if step < self.start_step:
            return 0.0

        t = (step - self.start_step) / max(self.end_step - self.start_step, 1)

        if self.schedule == "linear":
            return self.target_sparsity * t
        # cubic (default): s(t) = target * (1 - (1 - t)^3)
        return self.target_sparsity * (1.0 - (1.0 - t) ** 3)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, state: dict[str, Any]) -> None:
        model = state.get("model")
        if model is None:
            return

        import torch
        from crucible.training.compression_utils import iter_prunable_layers, apply_weight_mask

        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):
            mask = torch.ones_like(module.weight.data)
            self._masks[name] = mask
            self._layer_names.append(name)
            handle = apply_weight_mask(module, mask)
            self._hooks.append(handle)

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        # Only update masks at the configured frequency within the active window
        if step < self.start_step:
            return
        if step > self.end_step:
            return

        # One-shot: apply once at start_step only
        if self.schedule == "one_shot":
            if step != self.start_step:
                return
        else:
            if (step - self.start_step) % self.frequency != 0:
                return

        model = state.get("model")
        if model is None:
            return

        import torch

        sparsity = self._current_sparsity(step)
        if sparsity <= 0.0:
            return

        named_modules = dict(model.named_modules())

        for name in self._layer_names:
            module = named_modules.get(name)
            if module is None or not hasattr(module, "weight"):
                continue
            w = module.weight.data.abs()
            numel = w.numel()
            k = int(numel * sparsity)
            if k <= 0:
                continue

            threshold = torch.topk(w.flatten(), numel - k, largest=True).values[-1]
            new_mask = (w >= threshold).float()
            self._masks[name].copy_(new_mask)

        metrics["prune/sparsity_target"] = sparsity

    def on_train_end(self, state: dict[str, Any]) -> None:
        model = state.get("model")
        if model is None:
            return

        from crucible.training.compression_utils import make_masks_permanent

        # Remove hooks first
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Bake masks into weights
        make_masks_permanent(model, self._masks)
        self._masks.clear()


register_callback("pruning_magnitude", MagnitudePruningCallback, source="local")
