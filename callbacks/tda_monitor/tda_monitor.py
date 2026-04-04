"""TDA (Topological Data Analysis) monitor callback.

Computes persistent homology on latent representations during training
to track the topological structure of learned embeddings. Useful for
world models and representation learning where topological properties
(connected components, loops) reflect meaningful structure.

Requires the ``ripser`` package (``pip install ripser``).  Falls back
gracefully when ripser is not installed -- logs a single warning then
becomes a no-op.

Env vars:
    TDA_SAMPLE_SIZE: Number of latent vectors to analyze (default: 128).
                     TDA is O(n^3), so keep this small.
    TDA_MAX_DIM:     Maximum homology dimension (default: 1).
                     0 = connected components only, 1 = + loops, 2 = + voids.
    TDA_INTERVAL:    Compute every N validation steps (default: 100).
    TDA_THRESHOLD:   Persistence threshold for counting Betti numbers (default: 0.01).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback

logger = logging.getLogger(__name__)


class TDAMonitorCallback(TrainingCallback):
    """Persistent homology monitor for latent representations."""

    priority = 95  # Run late -- this is a monitoring callback

    def __init__(self, **kwargs: Any) -> None:
        self.sample_size = int(os.environ.get("TDA_SAMPLE_SIZE", "128"))
        self.max_dim = int(os.environ.get("TDA_MAX_DIM", "1"))
        self.interval = int(os.environ.get("TDA_INTERVAL", "100"))
        self.threshold = float(os.environ.get("TDA_THRESHOLD", "0.01"))

        self._ripser_available: bool | None = None  # None = not checked yet
        self._warned = False
        self._val_step_count = 0

    # ------------------------------------------------------------------
    # Lazy ripser import with one-time warning
    # ------------------------------------------------------------------

    def _check_ripser(self) -> bool:
        """Return True if ripser is importable, warn once if not."""
        if self._ripser_available is not None:
            return self._ripser_available
        try:
            import ripser as _ripser  # noqa: F401
            self._ripser_available = True
        except ImportError:
            self._ripser_available = False
            if not self._warned:
                logger.warning(
                    "tda_monitor: ripser not installed -- TDA metrics disabled. "
                    "Install with: pip install ripser"
                )
                self._warned = True
        return self._ripser_available

    # ------------------------------------------------------------------
    # Hook
    # ------------------------------------------------------------------

    def on_validation_end(
        self, step: int, metrics: dict[str, Any], state: dict[str, Any]
    ) -> None:
        self._val_step_count += 1

        # Only run every TDA_INTERVAL validation steps
        if self._val_step_count % self.interval != 0:
            return

        if not self._check_ripser():
            return

        latents = self._extract_latents(state, metrics)
        if latents is None:
            return

        tda_metrics = self._compute_tda(latents)
        if tda_metrics is not None:
            metrics.update(tda_metrics)

    # ------------------------------------------------------------------
    # Latent extraction
    # ------------------------------------------------------------------

    def _extract_latents(
        self, state: dict[str, Any], metrics: dict[str, Any]
    ) -> Any:
        """Get latent representations from the model.

        Tries multiple strategies:
        1. ``metrics["last_val_latents"]`` -- pre-computed by the training loop
        2. ``state["last_val_batch"]`` + ``model.encode()``
        3. ``state["last_val_batch"]`` + ``model.state_encoder()``

        Returns a numpy array of shape (N, D) or None.
        """
        import torch
        import numpy as np

        model = state.get("model")
        if model is None:
            return None

        # Strategy 1: latents already provided
        latents = metrics.get("last_val_latents")
        if latents is not None:
            if isinstance(latents, torch.Tensor):
                latents = latents.detach().cpu().numpy()
            return self._downsample(np.asarray(latents))

        # Need a batch to run the encoder
        batch = state.get("last_val_batch")
        if batch is None:
            return None

        # Strategy 2: model.encode()
        if hasattr(model, "encode"):
            try:
                with torch.no_grad():
                    z = model.encode(batch)
                if isinstance(z, torch.Tensor):
                    z = z.detach().cpu().numpy()
                return self._downsample(np.asarray(z).reshape(z.shape[0], -1))
            except Exception:
                logger.debug("tda_monitor: model.encode() failed, trying state_encoder")

        # Strategy 3: model.state_encoder attribute (e.g. world models)
        encoder = getattr(model, "state_encoder", None)
        if encoder is not None and callable(encoder):
            try:
                with torch.no_grad():
                    z = encoder(batch)
                if isinstance(z, torch.Tensor):
                    z = z.detach().cpu().numpy()
                return self._downsample(np.asarray(z).reshape(z.shape[0], -1))
            except Exception:
                logger.debug("tda_monitor: state_encoder() failed")

        # No encoder found -- skip silently
        return None

    def _downsample(self, points: Any) -> Any:
        """Downsample to at most ``self.sample_size`` points."""
        import numpy as np

        if len(points) <= self.sample_size:
            return points
        indices = np.random.choice(len(points), self.sample_size, replace=False)
        return points[indices]

    # ------------------------------------------------------------------
    # Persistent homology computation
    # ------------------------------------------------------------------

    def _compute_tda(self, latents: Any) -> dict[str, float] | None:
        """Compute persistence diagram and extract topological metrics.

        Returns a dict of ``tda/`` prefixed metrics, or None on failure.
        """
        import numpy as np

        try:
            from ripser import ripser
        except ImportError:
            return None

        if len(latents) < 3:
            # Not enough points for meaningful TDA
            return None

        try:
            result = ripser(latents, maxdim=self.max_dim, thresh=np.inf)
        except Exception as e:
            logger.debug("tda_monitor: ripser failed: %s", e)
            return None

        diagrams = result.get("dgms", [])
        if not diagrams:
            return None

        out: dict[str, float] = {}

        # Aggregate across all dimensions
        all_lifetimes: list[float] = []

        for dim, dgm in enumerate(diagrams):
            dgm = np.asarray(dgm)
            if len(dgm) == 0:
                if dim == 0:
                    out["tda/betti_0"] = 0.0
                    out["tda/mean_lifetime_h0"] = 0.0
                elif dim == 1:
                    out["tda/betti_1"] = 0.0
                    out["tda/mean_lifetime_h1"] = 0.0
                continue

            # Filter out infinite-death features for lifetime computation
            finite_mask = np.isfinite(dgm[:, 1])
            finite_dgm = dgm[finite_mask]

            if len(finite_dgm) > 0:
                lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
            else:
                lifetimes = np.array([])

            # Count features above persistence threshold (including infinite)
            # For Betti numbers: infinite features always count; finite features
            # count if their lifetime exceeds the threshold.
            n_infinite = int(np.sum(~finite_mask))
            n_persistent_finite = int(np.sum(lifetimes > self.threshold)) if len(lifetimes) > 0 else 0
            betti = n_infinite + n_persistent_finite

            if dim == 0:
                out["tda/betti_0"] = float(betti)
                if len(lifetimes) > 0:
                    out["tda/mean_lifetime_h0"] = float(np.mean(lifetimes))
                else:
                    out["tda/mean_lifetime_h0"] = 0.0
            elif dim == 1:
                out["tda/betti_1"] = float(betti)
                if len(lifetimes) > 0:
                    out["tda/mean_lifetime_h1"] = float(np.mean(lifetimes))
                else:
                    out["tda/mean_lifetime_h1"] = 0.0

            all_lifetimes.extend(lifetimes.tolist())

        # Global summary metrics
        if all_lifetimes:
            out["tda/total_persistence"] = float(sum(all_lifetimes))
            out["tda/max_lifetime"] = float(max(all_lifetimes))
        else:
            out["tda/total_persistence"] = 0.0
            out["tda/max_lifetime"] = 0.0

        return out


register_callback("tda_monitor", TDAMonitorCallback, source="local")
