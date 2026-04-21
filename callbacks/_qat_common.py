"""Shared fake-quantization primitives for QAT callbacks.

Provides STE (straight-through estimator) fake quantization functions
for INT4 and INT8, plus a base callback class that handles the shared
warmup/hook/enable/cleanup lifecycle.

All three QAT callbacks (qat_int4, qat_int8, qat_mixed) import from
this module to avoid duplicating the STE math and lifecycle boilerplate.
"""
from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Fake quantization functions (STE)
# ---------------------------------------------------------------------------

def fake_int8_quant(w):
    """Straight-through estimator for int8 QAT -- gradient flows through w."""
    import torch

    if w.ndim != 2:
        return w
    with torch.no_grad():
        scale = w.abs().amax(dim=1, keepdim=True) / 127.0
        scale = scale.clamp_min(1.0 / 127.0)
    q = torch.clamp(torch.round(w / scale), -128, 127)
    return w + (q * scale - w).detach()  # STE


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


def get_quant_fn(bits: int):
    """Return the fake-quant function for the given bit-width."""
    if bits == 4:
        return fake_int4_quant
    # Default to int8 for 8-bit or any unrecognized value
    return fake_int8_quant


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_exclude_patterns(env_var: str, default: str = "tok_emb,embed_low,lm_head") -> tuple[str, ...]:
    """Parse comma-separated exclude patterns from an env var."""
    exclude_raw = os.environ.get(env_var, default)
    return tuple(p.strip() for p in exclude_raw.split(",") if p.strip())


class BaseQATCallback:
    """Mixin providing shared QAT lifecycle: warmup gating, hook management, cleanup.

    Subclasses must set:
        self.warmup_steps: int
        self.exclude_patterns: tuple[str, ...]
        self._hooks: list[Any]
        self._enabled: bool
        self._bit_assignments: dict[str, int]

    And call _register_quant_hooks_base() from on_model_ready().
    """

    def _register_quant_hooks_base(
        self,
        model: Any,
        quant_fn,
        bits: int,
        *,
        resolve_fn=None,
    ) -> None:
        """Register forward pre-hooks on prunable layers.

        Args:
            model: The nn.Module to instrument.
            quant_fn: The fake-quant function to apply (for uniform-bit callbacks).
            bits: The bit-width to record (for uniform-bit callbacks).
            resolve_fn: Optional callable(name) -> (quant_fn, bits) for per-layer
                        bit-width resolution (used by qat_mixed).
        """
        from crucible.training.compression_utils import iter_prunable_layers

        self._bit_assignments = {}

        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):
            if resolve_fn is not None:
                fn, b = resolve_fn(name)
            else:
                fn, b = quant_fn, bits

            def _make_hook(mod_name: str, func):
                def _hook(mod, inputs):
                    if self._enabled:
                        mod.weight.data = func(mod.weight.data)
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name, fn))
            self._hooks.append(handle)
            self._bit_assignments[name] = b

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
