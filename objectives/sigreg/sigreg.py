"""SIGReg: Sketched-Isotropic-Gaussian Regularizer.

Prevents representation collapse in JEPA models by enforcing that latent
embeddings follow an isotropic Gaussian N(0, I), using the Cramér-Wold
theorem. If all 1D marginal projections are Gaussian, the full distribution
is Gaussian.

Computes M random projections of the embedding batch, tests each against
N(0,1) via the Epps-Pulley statistic, and averages.

Compose with MSE prediction loss via CompositeObjective:
    CompositeObjective([(1.0, MSEObjective()), (lambda, SIGRegObjective())])

Env vars:
    SIGREG_PROJECTIONS: Number of random projections M (default: 128)

Expects predictions["embeddings"]: Tensor[B, D] — raw encoder output.
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.objectives import TrainingObjective, register_objective


class SIGRegObjective(TrainingObjective):
    """Cramér-Wold based Gaussian regularizer for JEPA latent spaces."""

    name = "sigreg"

    def __init__(self, num_projections: int | None = None, **kwargs: Any) -> None:
        self.num_projections = num_projections or int(
            os.environ.get("SIGREG_PROJECTIONS", "128")
        )
        # Cache random directions (regenerated per compute for different batch devices)
        self._directions = None
        self._directions_device = None

    def _get_directions(self, embed_dim: int, device) -> "torch.Tensor":
        """Sample M random unit-norm directions on S^{d-1}."""
        import torch

        if (
            self._directions is not None
            and self._directions.shape == (self.num_projections, embed_dim)
            and self._directions_device == device
        ):
            return self._directions

        # Sample from standard normal, normalize to unit sphere
        dirs = torch.randn(self.num_projections, embed_dim, device=device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self._directions = dirs
        self._directions_device = device
        return dirs

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute SIGReg loss on encoder embeddings.

        Args:
            predictions: Must contain ``"embeddings"`` key with shape [B, D]
            targets: Not used by SIGReg (regularizer only)

        Returns:
            ``{"loss": sigreg_scalar, "sigreg": sigreg_scalar}``
        """
        import torch

        embeddings = predictions.get("embeddings")
        if embeddings is None:
            raise KeyError(
                "SIGRegObjective requires 'embeddings' key in predictions "
                "(raw encoder output, shape [B, D])"
            )

        B, D = embeddings.shape
        if B < 2:
            zero = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            return {"loss": zero, "sigreg": zero.detach()}

        # Get random projection directions [M, D]
        directions = self._get_directions(D, embeddings.device)

        # Project embeddings onto each direction: [M, B]
        projections = torch.mm(directions, embeddings.t())  # [M, B]

        # Compute Epps-Pulley test statistic for each projection
        sigreg_loss = self._epps_pulley_batch(projections)

        return {"loss": sigreg_loss, "sigreg": sigreg_loss.detach()}

    def _epps_pulley_batch(self, projections: "torch.Tensor") -> "torch.Tensor":
        """Batch Epps-Pulley normality test across M projections.

        For each 1D projection h = {h_1, ..., h_n}, the test statistic is:

            EP(h) = (2/n) * sum_i exp(-h_i^2/2)
                  - (1/n^2) * sum_{i,j} exp(-(h_i - h_j)^2/4)
                  - sqrt(2)

        A perfect N(0,1) sample gives EP ≈ 0. Larger values indicate
        deviation from Gaussianity.

        Args:
            projections: [M, B] — M projections, each with B samples

        Returns:
            Scalar average test statistic (lower = more Gaussian)
        """
        import math

        import torch

        M, n = projections.shape

        # Standardize each projection (zero mean, unit variance)
        mean = projections.mean(dim=1, keepdim=True)
        std = projections.std(dim=1, keepdim=True).clamp_min(1e-8)
        h = (projections - mean) / std  # [M, B]

        # Term 1: (2/n) * sum_i exp(-h_i^2 / 2)
        term1 = (2.0 / n) * torch.exp(-0.5 * h.pow(2)).sum(dim=1)  # [M]

        # Term 2: (1/n^2) * sum_{i,j} exp(-(h_i - h_j)^2 / 4)
        # Efficient computation via kernel trick:
        # sum_{i,j} exp(-(h_i-h_j)^2/4) = sum_{i,j} exp(-h_i^2/4) * exp(-h_j^2/4) * exp(h_i*h_j/2)
        # But simpler: use the identity for Gaussian kernel
        # = n * E_i[exp(-h_i^2/4)]^2 ... no, just compute directly
        # For moderate B (≤ 512), direct computation is fine
        if n <= 512:
            # [M, B, 1] - [M, 1, B] -> [M, B, B]
            diff_sq = (h.unsqueeze(2) - h.unsqueeze(1)).pow(2)  # [M, B, B]
            term2 = (1.0 / (n * n)) * torch.exp(-0.25 * diff_sq).sum(dim=(1, 2))  # [M]
        else:
            # For larger batches, use random sampling approximation
            # Sample n_sample pairs
            n_sample = min(n * 10, 2048)
            idx_i = torch.randint(0, n, (M, n_sample), device=h.device)
            idx_j = torch.randint(0, n, (M, n_sample), device=h.device)
            h_i = torch.gather(h, 1, idx_i)  # [M, n_sample]
            h_j = torch.gather(h, 1, idx_j)  # [M, n_sample]
            diff_sq = (h_i - h_j).pow(2)
            term2 = torch.exp(-0.25 * diff_sq).mean(dim=1)  # [M]

        # Epps-Pulley statistic
        sqrt2 = math.sqrt(2.0)
        ep = term1 - term2 - sqrt2  # [M]

        # Average across projections, take absolute value (we want to minimize)
        return ep.abs().mean()

    def metric_names(self) -> list[str]:
        return ["sigreg"]


register_objective("sigreg", SIGRegObjective, source="local")
