"""INT4 quantization-aware training callback.

Applies fake INT4 quantization with per-group scale factors via
straight-through estimator (STE).  Groups of ``GROUP_SIZE`` weights share
a single scale, giving finer granularity than per-row quantization.

Env vars:
    QAT_INT4_WARMUP_STEPS: Steps before enabling fake quant (default 1000).
    QAT_INT4_GROUP_SIZE: Number of weights per quantization group (default 128).
    QAT_INT4_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low,lm_head").
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


# ---------------------------------------------------------------------------
# Fake quantization (STE)
# ---------------------------------------------------------------------------

def fake_int4_quant(w, group_size: int = 128):
    """Straight-through estimator for int4 QAT -- gradient flows through w."""
    import torch

    if w.ndim != 2:
        return w
    rows, cols = w.shape

    if cols % group_size != 0:
        # Fallback to per-row if not divisible
        with torch.no_grad():
            scale = w.abs().amax(dim=1, keepdim=True) / 7.0
            scale = scale.clamp_min(1.0 / 7.0)
        q = torch.clamp(torch.round(w / scale), -8, 7)
        return w + (q * scale - w).detach()

    w_grouped = w.reshape(rows, -1, group_size)
    with torch.no_grad():
        scale = w_grouped.abs().amax(dim=-1, keepdim=True) / 7.0
        scale = scale.clamp_min(1.0 / 7.0)
    q = torch.clamp(torch.round(w_grouped / scale), -8, 7)
    dequant = (q * scale).reshape(rows, cols)
    return w + (dequant - w).detach()  # STE


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class QatInt4Callback(TrainingCallback):
    """INT4 QAT via per-group fake quantization with straight-through estimator."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_INT4_WARMUP_STEPS", "1000"))
        self.group_size = int(os.environ.get("QAT_INT4_GROUP_SIZE", "128"))
        exclude_raw = os.environ.get("QAT_INT4_EXCLUDE_PATTERNS", "tok_emb,embed_low,lm_head")
        self.exclude_patterns = tuple(p.strip() for p in exclude_raw.split(",") if p.strip())
        self._hooks: list[Any] = []
        self._enabled = False

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they're in the graph."""
        model = state.get("model")
        if model is None:
            return

        from crucible.training.compression_utils import iter_prunable_layers

        self._bit_assignments: dict[str, int] = {}
        gs = self.group_size

        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):

            def _make_hook(mod_name: str, group_sz: int):
                def _hook(mod, inputs):
                    if self._enabled:
                        mod.weight.data = fake_int4_quant(mod.weight.data, group_size=group_sz)
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name, gs))
            self._hooks.append(handle)
            self._bit_assignments[name] = 4

    def on_train_begin(self, state: dict[str, Any]) -> None:
        state["qat_bit_assignments"] = self._bit_assignments

    def on_step_begin(self, step: int, state: dict[str, Any]) -> None:
        if not self._enabled and step >= self.warmup_steps:
            self._enabled = True

    def on_train_end(self, state: dict[str, Any]) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._enabled = False


register_callback("qat_int4", QatInt4Callback, source="local")
