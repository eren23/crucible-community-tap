"""Compression metrics tracking callback.

Reports sparsity, effective bits, compression ratio, and model size
alongside standard training metrics. Always include when using
compression plugins.

Env vars:
    COMPRESSION_METRICS_EVERY: Report frequency in steps (default: 100)
"""
from __future__ import annotations

import json
import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class CompressionMetricsCallback(TrainingCallback):
    """Track and report compression statistics during training."""

    priority = 95  # Run after all other callbacks

    def __init__(self, **kwargs: Any) -> None:
        self.report_every = int(os.environ.get("COMPRESSION_METRICS_EVERY", "100"))
        self._original_size: int = 0
        self._original_params: int = 0

    def on_train_begin(self, state: dict[str, Any]) -> None:
        from crucible.training.compression_utils import CompressionMetrics

        model = state.get("model")
        if model is None:
            return

        self._original_size = CompressionMetrics.model_size_bytes(model)
        self._original_params = sum(p.numel() for p in model.parameters())

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        if step % self.report_every != 0:
            return

        self._compute_metrics(metrics, state)

    def on_validation_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        self._compute_metrics(metrics, state)

    def on_train_end(self, state: dict[str, Any]) -> None:
        from crucible.training.compression_utils import CompressionMetrics

        model = state.get("model")
        if model is None:
            return

        bit_assignments = state.get("qat_bit_assignments", {})

        report = {
            "original_size_bytes": self._original_size,
            "original_params": self._original_params,
            "final_size_bytes": CompressionMetrics.model_size_bytes(model),
            "sparsity": CompressionMetrics.sparsity(model).get("sparsity/overall", 0.0),
            "nonzero_params": CompressionMetrics.nonzero_params(model),
            "effective_bits": CompressionMetrics.effective_bits(model, bit_assignments),
            "compression_ratio": CompressionMetrics.compression_ratio(
                self._original_size, model, bit_assignments
            ),
        }

        try:
            with open("compression_report.json", "w") as f:
                json.dump(report, f, indent=2)
        except OSError:
            pass

    def _compute_metrics(self, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        from crucible.training.compression_utils import CompressionMetrics

        model = state.get("model")
        if model is None:
            return

        bit_assignments = state.get("qat_bit_assignments", {})

        sparsity_info = CompressionMetrics.sparsity(model)
        metrics["sparsity"] = sparsity_info.get("sparsity/overall", 0.0)
        metrics["nonzero_params"] = CompressionMetrics.nonzero_params(model)
        metrics["effective_bits"] = CompressionMetrics.effective_bits(model, bit_assignments)
        metrics["model_size_mb"] = CompressionMetrics.model_size_bytes(model) / (1024 * 1024)
        metrics["compression_ratio"] = CompressionMetrics.compression_ratio(
            self._original_size, model, bit_assignments
        )


register_callback("compression_metrics", CompressionMetricsCallback, source="local")
