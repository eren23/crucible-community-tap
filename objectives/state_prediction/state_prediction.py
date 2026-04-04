"""State prediction objective -- JEPA-style latent state prediction with variance regularization.

Designed for world models that produce predicted and target embeddings
in latent space. Combines MSE prediction loss with VICReg-style variance
hinge regularization to prevent representation collapse.

Expects:
    predictions: {"pred_embeddings": [B, T-1, D], "z_std": [D]}
    targets:     {"target_embeddings": [B, T-1, D]}

Returns:
    {"loss": scalar, "pred_loss": scalar, "var_reg": scalar}

Env vars:
    WM_VAR_WEIGHT:  Weight for variance regularization term (default: 0.1)
    WM_VAR_TARGET:  Target standard deviation threshold for hinge loss (default: 1.0)
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.objectives import TrainingObjective, register_objective


class StatePredictionObjective(TrainingObjective):
    """JEPA-style latent state prediction with variance regularization.

    Computes:
        pred_loss = MSE(pred_embeddings, target_embeddings)
        var_reg   = relu(var_target - z_std).mean()   (VICReg-style hinge)
        loss      = pred_loss + var_weight * var_reg
    """

    name = "state_prediction"

    def __init__(
        self,
        var_weight: float | None = None,
        var_target: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.var_weight = var_weight if var_weight is not None else float(
            os.environ.get("WM_VAR_WEIGHT", "0.1")
        )
        self.var_target = var_target if var_target is not None else float(
            os.environ.get("WM_VAR_TARGET", "1.0")
        )

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute state prediction loss with variance regularization.

        Args:
            predictions: Must contain:
                - ``"pred_embeddings"``: [B, T-1, D] predicted next-state embeddings
                - ``"z_std"``: [D] per-dimension standard deviation of encoder outputs
            targets: Must contain:
                - ``"target_embeddings"``: [B, T-1, D] target embeddings (from EMA encoder)

        Returns:
            dict with ``"loss"``, ``"pred_loss"``, ``"var_reg"`` keys.
        """
        import torch
        import torch.nn.functional as F

        pred_emb = predictions["pred_embeddings"]
        target_emb = targets["target_embeddings"]

        # Prediction loss: MSE in embedding space
        pred_loss = F.mse_loss(pred_emb, target_emb)

        # Variance regularization: hinge loss pushing per-dim std above var_target
        z_std = predictions["z_std"]  # [D]
        var_reg = torch.relu(self.var_target - z_std).mean()

        loss = pred_loss + self.var_weight * var_reg
        return {"loss": loss, "pred_loss": pred_loss, "var_reg": var_reg}

    def metric_names(self) -> list[str]:
        return ["pred_loss", "var_reg"]


register_objective("state_prediction", StatePredictionObjective, source="local")
