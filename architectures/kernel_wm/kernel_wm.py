"""KernelWorldModel — dense config encoder for GPU kernel migrations.

Uses a 2-layer MLP encoder that takes raw config vectors (41D one-hot +
normalized numericals) instead of token sequences. The LoopedPredictor
and EMA target from WorldModelBase are reused unchanged.

Env vars:
    KWM_INPUT_DIM:    Dense config vector dimension (default: 41)
    KWM_HIDDEN_DIM:   MLP hidden dimension (default: 256)
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor, nn

try:
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
except Exception:
    pass

from architectures.wm_base.wm_base import WorldModelBase, wm_base_kwargs_from_env


class DenseConfigEncoder(nn.Module):
    """MLP encoder for dense kernel config vectors.

    Input: [B, input_dim] float (one-hot categoricals + normalized numericals)
    Output: [B, model_dim] latent embedding
    """

    def __init__(self, input_dim: int = 41, model_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class KernelActionEncoder(nn.Module):
    """MLP encoder for 12D migration action vectors."""

    def __init__(self, action_dim: int = 12, model_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class KernelWorldModel(WorldModelBase):
    """World model for GPU kernel architecture migrations.

    Uses dense config vectors instead of token sequences. Subclasses
    WorldModelBase — inherits LoopedPredictor, EMA target, all losses.
    """

    def __init__(
        self,
        input_dim: int = 41,
        hidden_dim: int = 256,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim

        state_encoder = DenseConfigEncoder(
            input_dim=input_dim,
            model_dim=self.model_dim,
            hidden_dim=hidden_dim,
        )
        action_encoder = KernelActionEncoder(
            action_dim=self.action_dim,
            model_dim=self.model_dim,
        )
        self._build_common(state_encoder, action_encoder)

    def training_step(self, states, actions, **kwargs):
        """Bridge dense config format to WorldModelBase.forward().

        Args:
            states: [B, 2, input_dim] float — [before_config, after_config]
            actions: [B, 1, action_dim] float — migration action
        """
        return self.forward(states=states, actions=actions)


def kernel_wm_kwargs_from_env() -> dict[str, Any]:
    """Extract KernelWorldModel kwargs from env vars."""
    base = wm_base_kwargs_from_env(None)
    base["input_dim"] = int(os.environ.get("KWM_INPUT_DIM", "41"))
    base["hidden_dim"] = int(os.environ.get("KWM_HIDDEN_DIM", "256"))
    return base
