"""Mixed-precision quantization-aware training callback.

Parses a per-module bit-width pattern from ``QAT_MIXED_PATTERN`` and applies
the matching fake-quantization function to each module during forward passes.
This lets different parts of the model run at different precisions (e.g.,
encoder at 8-bit, predictor at 4-bit).

Pattern format (``QAT_MIXED_PATTERN``):
    - Single value: ``"8"`` -- all modules at 8 bits (default).
    - Comma-separated: ``"encoder:8,predictor:4,lm_head:8"`` -- modules whose
      name contains the prefix get the specified bit-width.

Env vars:
    QAT_MIXED_PATTERN: Bit-width mapping (default "8").
    QAT_MIXED_WARMUP_STEPS: Steps before enabling fake quant (default 500).
    QAT_MIXED_EXCLUDE_PATTERNS: Comma-separated module name patterns to skip
        (default "tok_emb,embed_low").
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


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
    return w + (q * scale - w).detach()


def fake_int4_quant(w, group_size: int = 128):
    """Straight-through estimator for int4 QAT -- gradient flows through w."""
    import torch

    if w.ndim != 2:
        return w
    rows, cols = w.shape

    if cols % group_size != 0:
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
    return w + (dequant - w).detach()


def _get_quant_fn(bits: int):
    """Return the fake-quant function for the given bit-width."""
    if bits == 4:
        return fake_int4_quant
    # Default to int8 for 8-bit or any unrecognized value
    return fake_int8_quant


# ---------------------------------------------------------------------------
# Pattern parsing
# ---------------------------------------------------------------------------

def _parse_pattern(raw: str) -> dict[str, int] | int:
    """Parse QAT_MIXED_PATTERN into either a global bit-width or prefix map.

    Returns:
        int -- if the pattern is a single number (applied to all modules).
        dict -- mapping of name-prefix to bit-width.
    """
    raw = raw.strip()
    if ":" not in raw:
        return int(raw)

    mapping: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        prefix, _, bits_str = part.partition(":")
        mapping[prefix.strip()] = int(bits_str.strip())
    return mapping


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class QatMixedCallback(TrainingCallback):
    """Mixed-precision QAT with per-module bit-width assignment."""

    priority = 5  # Run early, before pruning callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.warmup_steps = int(os.environ.get("QAT_MIXED_WARMUP_STEPS", "500"))
        pattern_raw = os.environ.get("QAT_MIXED_PATTERN", "8")
        self.pattern = _parse_pattern(pattern_raw)
        exclude_raw = os.environ.get("QAT_MIXED_EXCLUDE_PATTERNS", "tok_emb,embed_low")
        self.exclude_patterns = tuple(p.strip() for p in exclude_raw.split(",") if p.strip())
        self._hooks: list[Any] = []
        self._enabled = False

    def _resolve_bits(self, module_name: str) -> int:
        """Determine the bit-width for a given module name."""
        if isinstance(self.pattern, int):
            return self.pattern
        for prefix, bits in self.pattern.items():
            if prefix in module_name:
                return bits
        # No match -- fall back to 8-bit
        return 8

    def on_model_ready(self, state: dict[str, Any]) -> None:
        """Register QAT hooks BEFORE torch.compile so they're in the graph."""
        model = state.get("model")
        if model is None:
            return

        from crucible.training.compression_utils import iter_prunable_layers

        self._bit_assignments: dict[str, int] = {}

        for name, module in iter_prunable_layers(model, exclude_patterns=self.exclude_patterns):
            bits = self._resolve_bits(name)
            quant_fn = _get_quant_fn(bits)

            def _make_hook(mod_name: str, fn):
                def _hook(mod, inputs):
                    if self._enabled:
                        mod.weight.data = fn(mod.weight.data)
                return _hook

            handle = module.register_forward_pre_hook(_make_hook(name, quant_fn))
            self._hooks.append(handle)
            self._bit_assignments[name] = bits

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


register_callback("qat_mixed", QatMixedCallback, source="local")
