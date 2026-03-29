"""Wanda pruning callback.

One-shot pruning using the Wanda importance score:

    importance[i, j] = |W[i, j]| * ||X[:, j]||_2

During calibration steps, forward hooks accumulate per-layer input
activation column norms (running mean of ``||X[:, j]||_2``).  At the
target step the scores are computed, the lowest-scoring weights are
pruned per output row, hooks are removed, and masks are baked in.

Supports optional N:M structured sparsity (e.g. 2:4).

Env vars:
    WANDA_SPARSITY: Fraction of weights to prune (default 0.5).
    WANDA_CALIBRATION_STEPS: Steps to accumulate activation norms (default 50).
    WANDA_APPLY_AT_STEP: Step at which pruning fires (default 1000).
    WANDA_STRUCTURED: "1" to enable N:M structured sparsity (default "0").
    WANDA_NM_RATIO: N:M ratio string, e.g. "2:4" (default "2:4").
    WANDA_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low,lm_head").
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class WandaPruningCallback(TrainingCallback):
    """Wanda (Weights and Activations) one-shot pruning."""

    priority = 8

    def __init__(self, **kwargs: Any) -> None:
        self.sparsity = float(os.environ.get("WANDA_SPARSITY", "0.5"))
        self.calibration_steps = int(os.environ.get("WANDA_CALIBRATION_STEPS", "50"))
        self.apply_at_step = int(os.environ.get("WANDA_APPLY_AT_STEP", "1000"))
        self.use_structured = os.environ.get("WANDA_STRUCTURED", "0") == "1"
        nm_raw = os.environ.get("WANDA_NM_RATIO", "2:4")
        parts = nm_raw.split(":")
        self.nm_n = int(parts[0])
        self.nm_m = int(parts[1])
        exclude_raw = os.environ.get("WANDA_EXCLUDE_PATTERNS", "tok_emb,embed_low,lm_head")
        self.exclude_patterns = tuple(p.strip() for p in exclude_raw.split(",") if p.strip())

        self._act_hooks: list[Any] = []          # activation collection hooks
        self._act_norms: dict[str, Any] = {}     # name -> running sum of ||X[:, j]||_2
        self._act_counts: dict[str, int] = {}    # name -> number of accumulated batches
        self._layer_names: list[str] = []
        self._applied = False
        self._calibrating = False
        self._calibration_start: int = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Discover prunable layers and register activation hooks BEFORE torch.compile."""
        model = state.get("model")
        if model is None:
            return

        from crucible.training.compression_utils import iter_prunable_layers

        self._calibration_start = max(self.apply_at_step - self.calibration_steps, 0)

        norms_ref = self._act_norms
        counts_ref = self._act_counts

        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):
            self._layer_names.append(name)
            norms_ref[name] = None
            counts_ref[name] = 0

            # Register activation hooks now (before compile) so they're in the graph.
            # Gated by _calibrating flag to only collect during calibration window.
            def _make_hook(ln):
                def _hook(mod, inputs):
                    if not self._calibrating:
                        return
                    import torch
                    x = inputs[0] if isinstance(inputs, tuple) else inputs
                    if x is None:
                        return
                    with torch.no_grad():
                        x_2d = x.detach().reshape(-1, x.shape[-1])
                        col_norms = x_2d.norm(dim=0)
                        if norms_ref[ln] is None:
                            norms_ref[ln] = torch.zeros_like(col_norms)
                        norms_ref[ln] += col_norms
                        counts_ref[ln] += 1
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name))
            self._act_hooks.append(handle)

    def on_step_begin(self, step: int, state: dict[str, Any]) -> None:
        if self._applied:
            return
        # Enable calibration during the calibration window
        self._calibrating = self._calibration_start <= step < self.apply_at_step

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        if self._applied:
            return
        if step != self.apply_at_step:
            return

        model = state.get("model")
        if model is None:
            return

        import torch
        from crucible.training.compression_utils import make_masks_permanent

        named_modules = dict(model.named_modules())
        masks: dict[str, Any] = {}

        for name in self._layer_names:
            module = named_modules.get(name)
            if module is None or not hasattr(module, "weight"):
                continue

            W = module.weight.data  # (out_features, in_features)
            count = max(self._act_counts[name], 1)
            act_norm = self._act_norms[name]
            if act_norm is None:
                act_norm = torch.ones(W.shape[1], device=W.device)
            else:
                act_norm = act_norm / count  # mean activation norm

            # Wanda score: |W[i,j]| * ||X[:,j]||_2
            scores = W.abs() * act_norm.unsqueeze(0)

            if self.use_structured:
                mask = self._nm_prune(scores, W)
            else:
                mask = self._unstructured_prune(scores, W)

            masks[name] = mask

        # Remove activation hooks
        for handle in self._act_hooks:
            handle.remove()
        self._act_hooks.clear()

        # Bake masks permanently
        make_masks_permanent(model, masks)
        self._applied = True

        from crucible.training.compression_utils import CompressionMetrics
        sp = CompressionMetrics.sparsity(model)
        metrics["wanda/sparsity_overall"] = sp.get("sparsity/overall", 0.0)

    def _unstructured_prune(self, scores: Any, W: Any) -> Any:
        """Per-row unstructured pruning: prune lowest-scoring entries per row."""
        import torch
        out_features, in_features = W.shape
        k = int(in_features * self.sparsity)
        if k <= 0:
            return torch.ones_like(W)
        mask = torch.ones_like(W)
        for i in range(out_features):
            row_scores = scores[i]
            _, indices = torch.topk(row_scores, k, largest=False)
            mask[i, indices] = 0.0
        return mask

    def _nm_prune(self, scores: Any, W: Any) -> Any:
        """N:M structured sparsity: keep N largest per group of M."""
        import torch
        out_features, in_features = W.shape
        mask = torch.ones_like(W)
        n, m = self.nm_n, self.nm_m
        # Pad in_features to multiple of M if needed
        for i in range(out_features):
            for start in range(0, in_features, m):
                end = min(start + m, in_features)
                group_scores = scores[i, start:end]
                group_len = end - start
                keep = min(n, group_len)
                if keep >= group_len:
                    continue
                _, top_idx = torch.topk(group_scores, keep, largest=True)
                group_mask = torch.zeros(group_len, device=W.device)
                group_mask[top_idx] = 1.0
                mask[i, start:end] = group_mask
        return mask

    def on_train_end(self, state: dict[str, Any]) -> None:
        # Clean up any remaining hooks
        for handle in self._act_hooks:
            handle.remove()
        self._act_hooks.clear()


register_callback("pruning_wanda", WandaPruningCallback, source="local")
