"""Per-layer sensitivity analysis callback.

Accumulates per-layer importance scores during the first N training steps,
then writes a JSON report and stores results in ``state["sensitivity"]``
so downstream pruning callbacks can use sensitivity-informed sparsity
allocation instead of uniform pruning.

Env vars:
    SENSITIVITY_METHOD:      fisher | gradient | activation_norm (default: fisher)
    SENSITIVITY_STEPS:       Steps to accumulate (default: 100)
    SENSITIVITY_OUTPUT_PATH: Report output path (default: sensitivity_report.json)
"""
from __future__ import annotations

import json
import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class SensitivityAnalysisCallback(TrainingCallback):
    """Diagnostic pre-pass: score per-layer importance for pruning guidance."""

    priority = 3  # Run before QAT (5) and pruning (8)

    def __init__(self, **kwargs: Any) -> None:
        self.method = os.environ.get("SENSITIVITY_METHOD", "fisher")
        self.num_steps = int(os.environ.get("SENSITIVITY_STEPS", "100"))
        self.output_path = os.environ.get("SENSITIVITY_OUTPUT_PATH", "sensitivity_report.json")
        self._scores: dict[str, float] = {}
        self._step_count = 0
        self._done = False

    def on_train_begin(self, state: dict[str, Any]) -> None:
        import torch

        model = state.get("model")
        if model is None:
            return

        # Initialize accumulators for each parameter
        self._scores = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._scores[name] = 0.0

    def on_after_backward(self, step: int, state: dict[str, Any]) -> None:
        if self._done or step >= self.num_steps:
            return

        import torch

        model = state.get("model")
        if model is None:
            return

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if name not in self._scores:
                continue

            if self.method == "fisher":
                # Fisher information approximation: E[grad^2]
                self._scores[name] += float(param.grad.data.pow(2).sum().item())
            elif self.method == "gradient":
                # Gradient norm
                self._scores[name] += float(param.grad.data.norm().item())
            elif self.method == "activation_norm":
                # Use weight magnitude as proxy (no activation hooks needed)
                self._scores[name] += float(param.data.abs().mean().item())

        self._step_count += 1

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        if self._done or step < self.num_steps:
            return

        self._done = True
        self._finalize(state)

    def _finalize(self, state: dict[str, Any]) -> None:
        # Normalize by step count
        if self._step_count > 0:
            for name in self._scores:
                self._scores[name] /= self._step_count

        # Group by layer (aggregate param-level scores to module level)
        layer_scores: dict[str, float] = {}
        for name, score in self._scores.items():
            # Extract layer prefix (e.g., "blocks.0.attn.qkv.weight" -> "blocks.0")
            parts = name.split(".")
            # Find the block-level prefix
            layer_key = name
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:
                    layer_key = ".".join(parts[: i + 1])
                    break
            layer_scores[layer_key] = layer_scores.get(layer_key, 0.0) + score

        # Rank layers by importance
        sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)

        # Compute suggested per-layer sparsity (inverse importance)
        total_importance = sum(s for _, s in sorted_layers) or 1.0
        layers_report = []
        for rank, (name, importance) in enumerate(sorted_layers):
            normalized = importance / total_importance
            # More important layers get less sparsity
            suggested_sparsity = max(0.0, min(0.9, 1.0 - normalized * len(sorted_layers)))
            layers_report.append(
                {
                    "name": name,
                    "importance": round(importance, 6),
                    "normalized": round(normalized, 6),
                    "rank": rank + 1,
                    "suggested_sparsity": round(suggested_sparsity, 3),
                }
            )

        report = {
            "method": self.method,
            "steps_accumulated": self._step_count,
            "num_layers": len(layers_report),
            "layers": layers_report,
        }

        # Write report
        try:
            with open(self.output_path, "w") as f:
                json.dump(report, f, indent=2)
        except OSError:
            pass  # Non-fatal if we can't write

        # Store in state for downstream pruning callbacks
        state["sensitivity"] = {
            entry["name"]: entry["suggested_sparsity"] for entry in layers_report
        }
        state["sensitivity_report"] = report


register_callback("sensitivity_analysis", SensitivityAnalysisCallback, source="local")
