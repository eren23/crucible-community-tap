"""WorldModelBase -- domain-agnostic looped predictor backbone for world models.

Abstract base class for JEPA-style world models. Subclasses provide
domain-specific state and action encoders; the looped predictor and
EMA target encoder are shared infrastructure.

Key design: the predictor iterates a SINGLE transformer block multiple
times (weight-sharing / looping), following the LE-WM insight that depth
comes from loops, not extra parameters.

Subclass contract:
    1. Implement ``build_state_encoder(args) -> nn.Module``
       Input: domain-specific observation tensor
       Output: [B, D] latent embedding
    2. Implement ``build_action_encoder(args) -> nn.Module``
       Input: domain-specific action tensor
       Output: [B, D] latent embedding
    3. Register the subclass via ``register_model("my_wm", factory, source="local")``

Do NOT register wm_base itself -- it is abstract.

Env vars (all with sensible defaults):
    WM_MODEL_DIM:        Latent embedding dimension (default: 128)
    WM_NUM_LOOPS:        Iterations of the looped transformer block (default: 4)
    WM_NUM_HEADS:        Attention heads in predictor block (default: 4)
    WM_PREDICTOR_DEPTH:  Unique transformer blocks before looping (default: 2)
    WM_EMA_DECAY:        Target encoder EMA momentum (default: 0.996)
    WM_SIGREG_WEIGHT:    Inline variance regularization weight (default: 0.1, 0.0 to disable)
    WM_MLP_RATIO:        MLP hidden-dim ratio in transformer blocks (default: 4.0)
    WM_DROPOUT:          Dropout rate in predictor (default: 0.1)
    ACTION_DIM:          Action vector dimensionality (default: 4)
"""
from __future__ import annotations

import copy
import os
from abc import abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from crucible.models.base import CrucibleModel
except ImportError:
    # Standalone mode: minimal base matching CrucibleModel contract
    class CrucibleModel(nn.Module):  # type: ignore[no-redef]
        def training_step(self, **batch): return self.forward(**batch)
        def validation_step(self, **batch): return self.forward(**batch)
        @classmethod
        def modality(cls): return "generic"


# ---------------------------------------------------------------------------
# Looped Transformer Block
# ---------------------------------------------------------------------------

class LoopedTransformerBlock(nn.Module):
    """Single pre-norm transformer block designed for weight-shared looping.

    Architecture: LayerNorm -> MultiheadAttention -> residual
                  LayerNorm -> MLP(GELU) -> residual
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass. x: [B, S, D] (sequence of tokens)."""
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Looped Predictor
# ---------------------------------------------------------------------------

class LoopedPredictor(nn.Module):
    """Predicts z_next from (z_state, z_action) using weight-shared looping.

    The predictor concatenates the state and action embeddings into a
    2-token sequence, passes it through ``depth`` unique transformer blocks,
    each iterated ``num_loops`` times, then extracts the first token as the
    predicted next-state embedding.

    This is the core LE-WM innovation: depth from loops, not parameters.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int = 2,
        num_loops: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_loops = num_loops

        # Unique transformer blocks (each looped num_loops times)
        self.blocks = nn.ModuleList([
            LoopedTransformerBlock(dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, z_state: Tensor, z_action: Tensor) -> Tensor:
        """Predict next-state embedding.

        Args:
            z_state:  [B, D] -- current state embedding (detached from encoder)
            z_action: [B, D] -- action embedding

        Returns:
            [B, D] -- predicted next-state embedding
        """
        # Form 2-token sequence: [state, action]
        x = torch.stack([z_state, z_action], dim=1)  # [B, 2, D]

        # Pass through each block, looping num_loops times per block
        for block in self.blocks:
            for _ in range(self.num_loops):
                x = block(x)

        # Extract state token prediction
        x = self.norm(x[:, 0])  # [B, D]
        return x


# ---------------------------------------------------------------------------
# WorldModelBase
# ---------------------------------------------------------------------------

class WorldModelBase(CrucibleModel):
    """Abstract base for domain-agnostic world models.

    Provides:
      - EMA target encoder (deep copy of state_encoder, no-grad updates)
      - LoopedPredictor (weight-shared transformer blocks)
      - Inline variance regularization (SIGReg fallback)
      - Forward pass implementing the JEPA prediction loop

    Subclasses must implement:
      - ``build_state_encoder(args)`` -> nn.Module mapping observations to [B, D]
      - ``build_action_encoder(args)`` -> nn.Module mapping actions to [B, D]
    """

    def __init__(
        self,
        model_dim: int = 128,
        num_loops: int = 4,
        num_heads: int = 4,
        predictor_depth: int = 2,
        ema_decay: float = 0.996,
        sigreg_weight: float = 0.1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_dim: int = 4,
        lambda_pred: float = 1.0,
        lambda_dir: float = 0.5,
        lambda_mag: float = 0.1,
        lambda_cov: float = 0.01,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_loops = num_loops
        self.num_heads = num_heads
        self.predictor_depth = predictor_depth
        self.ema_decay = ema_decay
        self.sigreg_weight = sigreg_weight
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.action_dim = action_dim
        self.lambda_pred = lambda_pred
        self.lambda_dir = lambda_dir
        self.lambda_mag = lambda_mag
        self.lambda_cov = lambda_cov

    def _build_common(self, state_encoder: nn.Module, action_encoder: nn.Module) -> None:
        """Initialize shared components after subclass sets encoders.

        Subclass __init__ should call this after creating state_encoder
        and action_encoder:

            self.state_encoder = self.build_state_encoder(args)
            self.action_encoder = self.build_action_encoder(args)
            self._build_common(self.state_encoder, self.action_encoder)
        """
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

        # EMA target encoder: deep copy of state_encoder, frozen
        self.target_encoder = copy.deepcopy(state_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Looped predictor
        self.predictor = LoopedPredictor(
            dim=self.model_dim,
            num_heads=self.num_heads,
            depth=self.predictor_depth,
            num_loops=self.num_loops,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout_rate,
        )

        # Prediction projector (BYOL/VICReg pattern):
        # Projects predictions and targets through an MLP before MSE.
        # This prevents the encoder from collapsing to trivial solutions
        # where before ≈ after in latent space (common when inputs share
        # 90%+ of their tokens).
        proj_dim = self.model_dim * 2
        self.pred_projector = nn.Sequential(
            nn.Linear(self.model_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, self.model_dim),
        )
        self.target_projector = nn.Sequential(
            nn.Linear(self.model_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, self.model_dim),
        )

    @abstractmethod
    def build_state_encoder(self, args: Any) -> nn.Module:
        """Build the domain-specific state encoder.

        Must return an nn.Module that maps raw observations to
        [B, model_dim] embeddings.
        """
        ...

    @abstractmethod
    def build_action_encoder(self, args: Any) -> nn.Module:
        """Build the domain-specific action encoder.

        Must return an nn.Module that maps raw actions to
        [B, model_dim] embeddings.
        """
        ...

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Exponential moving average update of target encoder."""
        for p_target, p_online in zip(
            self.target_encoder.parameters(), self.state_encoder.parameters()
        ):
            p_target.data.mul_(self.ema_decay).add_(
                p_online.data, alpha=1.0 - self.ema_decay
            )

    def _compute_sigreg(self, z_flat: Tensor, num_proj: int = 128) -> tuple[Tensor, Tensor]:
        """SIGReg: Sketched Isotropic Gaussian Regularizer (LE-WM paper).

        Enforces embeddings follow N(0, I) via the Cramér-Wold theorem:
        if all 1D random projections are Gaussian, the full distribution is.
        Uses the Epps-Pulley test statistic for each projection.

        This is strictly stronger than VICReg variance/covariance reg — it
        controls the FULL distribution shape, not just marginal moments.

        Args:
            z_flat: [N, D] -- flattened encoder outputs
            num_proj: number of random projection directions M

        Returns:
            (sigreg_loss, z_std) where sigreg_loss is the scalar and
            z_std is [D] for monitoring.
        """
        import math

        N, D = z_flat.shape
        z_std = z_flat.std(dim=0)  # [D] for monitoring

        if N < 2:
            return torch.tensor(0.0, device=z_flat.device), z_std

        # Random unit-norm directions [M, D]
        dirs = torch.randn(num_proj, D, device=z_flat.device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)

        # Project embeddings: [M, N]
        proj = dirs @ z_flat.t()

        # Standardize each projection
        mean = proj.mean(dim=1, keepdim=True)
        std = proj.std(dim=1, keepdim=True).clamp_min(1e-8)
        h = (proj - mean) / std  # [M, N]

        # Epps-Pulley test: EP(h) = (2/n)Σ exp(-h²/2) - (1/n²)ΣΣ exp(-(hᵢ-hⱼ)²/4) - √2
        term1 = (2.0 / N) * torch.exp(-0.5 * h.pow(2)).sum(dim=1)  # [M]

        if N <= 512:
            diff_sq = (h.unsqueeze(2) - h.unsqueeze(1)).pow(2)  # [M, N, N]
            term2 = (1.0 / (N * N)) * torch.exp(-0.25 * diff_sq).sum(dim=(1, 2))
        else:
            # Random pair sampling for large batches
            n_samp = min(N * 10, 2048)
            idx_i = torch.randint(0, N, (num_proj, n_samp), device=h.device)
            idx_j = torch.randint(0, N, (num_proj, n_samp), device=h.device)
            diff_sq = (torch.gather(h, 1, idx_i) - torch.gather(h, 1, idx_j)).pow(2)
            term2 = torch.exp(-0.25 * diff_sq).mean(dim=1)

        ep = (term1 - term2 - math.sqrt(2.0)).abs().mean()
        return ep, z_std

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """JEPA prediction loop over a state-action trajectory.

        Args:
            states:  [B, T, *] -- sequence of T observations
            actions: [B, T-1, *] -- actions between consecutive states

        Returns:
            dict with keys:
                loss:              scalar training loss
                pred_loss:         MSE between predicted and target embeddings
                var_reg:           variance regularization term
                pred_embeddings:   [B, T-1, D] predicted next-state embeddings
                target_embeddings: [B, T-1, D] EMA target embeddings (no grad)
                z_std:             [D] per-dimension std of encoder outputs
        """
        B, T = states.shape[:2]
        num_transitions = T - 1

        # Encode all states through online encoder
        # Flatten temporal dim for encoding, then reshape back
        state_shape = states.shape[2:]
        flat_states = states.reshape(B * T, *state_shape)
        z_all = self.state_encoder(flat_states).reshape(B, T, -1)  # [B, T, D]

        # Encode actions
        action_shape = actions.shape[2:]
        flat_actions = actions.reshape(B * num_transitions, *action_shape)
        z_action = self.action_encoder(flat_actions).reshape(B, num_transitions, -1)  # [B, T-1, D]

        # Compute target embeddings via EMA encoder (no gradient)
        with torch.no_grad():
            flat_next_states = states[:, 1:].reshape(B * num_transitions, *state_shape)
            z_target = self.target_encoder(flat_next_states).reshape(B, num_transitions, -1)  # [B, T-1, D]

        # Predict next-state embeddings for each transition
        pred_list = []
        for t in range(num_transitions):
            z_pred_t = self.predictor(
                z_all[:, t].detach(),  # detach encoder from predictor path
                z_action[:, t],
            )
            pred_list.append(z_pred_t)

        pred_embeddings = torch.stack(pred_list, dim=1)  # [B, T-1, D]

        # ─── Loss A: Prediction loss ───
        # WM_USE_PROJECTOR=0 disables projectors — MSE on raw normalized embeddings.
        # With v2 delta losses providing direct geometric supervision, the projector
        # may sabotage by absorbing alignment gradient.
        pred_flat = pred_embeddings.reshape(-1, self.model_dim)
        target_flat = z_target.reshape(-1, self.model_dim)
        use_projector = os.environ.get("WM_USE_PROJECTOR", "1") == "1"
        if use_projector:
            loss_pred = F.mse_loss(
                F.normalize(self.pred_projector(pred_flat), dim=-1),
                F.normalize(self.target_projector(target_flat), dim=-1),
            )
        else:
            # Direct MSE on L2-normalized raw embeddings
            loss_pred = F.mse_loss(
                F.normalize(pred_flat, dim=-1),
                F.normalize(target_flat, dim=-1),
            )

        # ─── Delta-space losses (v2: transition geometry) ───
        # The edit displacement field should be smooth, calibrated, and
        # direction-aligned — this is the code-native geometry prior.
        z_current = z_all[:, :-1].detach().reshape(-1, self.model_dim)  # [N, D]
        delta_true = target_flat - z_current    # actual edit displacement
        delta_pred = pred_flat - z_current      # predicted edit displacement

        # Loss B: Delta direction alignment
        # "Predicted edit should point same direction as true edit"
        dt_norm = delta_true.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dp_norm = delta_pred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cos_sim = (delta_true / dt_norm * delta_pred / dp_norm).sum(dim=-1)
        loss_dir = 1.0 - cos_sim.mean()

        # Loss C: Delta magnitude calibration (log-space for numerical stability)
        # "Predicted edit size should match actual edit size"
        # log(ratio)^2 penalizes 2x-too-big same as 2x-too-small, scale-invariant
        loss_mag = (torch.log(dp_norm.squeeze(-1) + 1e-6)
                    - torch.log(dt_norm.squeeze(-1) + 1e-6)).pow(2).mean()

        # Loss D: Light covariance penalty (just prevent dimension collapse)
        z_flat = z_all.reshape(-1, self.model_dim)
        z_std = z_flat.std(dim=0)
        z_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(z_flat.shape[0] - 1, 1)
        loss_cov = cov.fill_diagonal_(0).pow(2).sum() / self.model_dim

        # ─── Optional static regularizers (VICReg/SIGReg) for baseline comparison ───
        # These run IN ADDITION to delta losses when enabled. For pure baselines,
        # set WM_LAMBDA_DIR=0, WM_LAMBDA_MAG=0, WM_LAMBDA_COV=0.
        reg_mode = os.environ.get("WM_REG_MODE", "none")
        static_reg_loss = torch.tensor(0.0, device=z_all.device)
        if reg_mode == "vicreg":
            # VICReg: variance hinge (>=1 std per dim) + covariance decorrelation
            var_hinge = F.relu(1.0 - z_std).mean()
            # Reuse loss_cov above (off-diagonal squared). Combine.
            static_reg_loss = var_hinge + 0.04 * loss_cov
        elif reg_mode == "sigreg":
            sig_loss, _ = self._compute_sigreg(z_flat)
            static_reg_loss = sig_loss

        # ─── Combined loss ───
        loss = (self.lambda_pred * loss_pred
                + self.lambda_dir * loss_dir
                + self.lambda_mag * loss_mag
                + self.lambda_cov * loss_cov
                + self.sigreg_weight * static_reg_loss)

        # Diagnostic: mean delta cosine sim and norm ratio
        delta_cos_sim = cos_sim.mean()
        delta_norm_ratio = (dp_norm / dt_norm).mean()

        # Update EMA target encoder
        if self.training:
            self._update_ema()

        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_dir": loss_dir,
            "loss_mag": loss_mag,
            "loss_cov": loss_cov,
            "loss_static_reg": static_reg_loss,
            "delta_cos_sim": delta_cos_sim,
            "delta_norm_ratio": delta_norm_ratio,
            "pred_embeddings": pred_embeddings,
            "target_embeddings": z_target,
            "z_std": z_std,
        }

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def metric_names(self) -> list[str]:
        return ["pred_loss", "var_reg"]

    def param_groups(self) -> list[dict[str, Any]]:
        """Separate encoder and predictor for different learning rates."""
        encoder_params = list(self.state_encoder.parameters())
        action_params = list(self.action_encoder.parameters())
        predictor_params = list(self.predictor.parameters())
        return [
            {"params": encoder_params, "name": "state_encoder"},
            {"params": action_params, "name": "action_encoder"},
            {"params": predictor_params, "name": "predictor"},
        ]

    @classmethod
    def modality(cls) -> str:
        return "world_model"

    def encode(self, states: Tensor) -> Tensor:
        """Encode observations to latent embeddings (for inference/planning).

        Args:
            states: [B, *] or [B, T, *]

        Returns:
            [B, D] or [B, T, D]
        """
        if states.dim() <= 2:
            # Single timestep: [B, *] -> [B, D]
            return self.state_encoder(states)
        # Sequence: [B, T, *] -> [B, T, D]
        B, T = states.shape[:2]
        state_shape = states.shape[2:]
        flat = states.reshape(B * T, *state_shape)
        return self.state_encoder(flat).reshape(B, T, -1)

    def predict_next(self, z_state: Tensor, action: Tensor) -> Tensor:
        """Predict next embedding from current embedding + action (for planning).

        Args:
            z_state: [B, D] -- current latent state
            action:  [B, *] -- raw action (will be encoded)

        Returns:
            [B, D] -- predicted next latent state
        """
        z_action = self.action_encoder(action)
        return self.predictor(z_state, z_action)


# ---------------------------------------------------------------------------
# Factory helper for subclasses
# ---------------------------------------------------------------------------

def wm_base_kwargs_from_env(args: Any | None = None) -> dict[str, Any]:
    """Extract WorldModelBase constructor kwargs from env vars / args namespace.

    Subclass factories should call this and pass the result to
    ``super().__init__(**kwargs)`` along with any domain-specific params.
    """
    def _get(attr: str, env_var: str, default: str) -> str:
        if args is not None:
            val = getattr(args, attr, None)
            if val is not None:
                return str(val)
        return os.environ.get(env_var, default)

    return {
        "model_dim": int(_get("model_dim", "WM_MODEL_DIM", "128")),
        "num_loops": int(_get("num_loops", "WM_NUM_LOOPS", "4")),
        "num_heads": int(_get("num_heads", "WM_NUM_HEADS", "4")),
        "predictor_depth": int(_get("predictor_depth", "WM_PREDICTOR_DEPTH", "2")),
        "ema_decay": float(_get("ema_decay", "WM_EMA_DECAY", "0.996")),
        "sigreg_weight": float(_get("sigreg_weight", "WM_SIGREG_WEIGHT", "0.1")),
        "mlp_ratio": float(_get("mlp_ratio", "WM_MLP_RATIO", "4.0")),
        "dropout": float(_get("dropout", "WM_DROPOUT", "0.1")),
        "action_dim": int(_get("action_dim", "ACTION_DIM", "4")),
        "lambda_pred": float(_get("lambda_pred", "WM_LAMBDA_PRED", "1.0")),
        "lambda_dir": float(_get("lambda_dir", "WM_LAMBDA_DIR", "0.5")),
        "lambda_mag": float(_get("lambda_mag", "WM_LAMBDA_MAG", "0.1")),
        "lambda_cov": float(_get("lambda_cov", "WM_LAMBDA_COV", "0.01")),
    }
