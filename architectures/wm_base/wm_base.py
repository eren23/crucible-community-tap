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

Rollout robustness (Phase 6 — all opt-in, default disabled):
    WM_NOISE_SIGMA:          Gaussian noise std on z_state during training (default: 0.0)
    WM_NOISE_ANNEAL_STEPS:   Steps to anneal noise sigma to 0 (default: 0 = constant)
    WM_SCHEDULED_ROLLOUT:    "1" to enable scheduled rollout in main loop (default: "0")
    WM_ROLLOUT_WARMUP:       Steps before rollout probability starts rising (default: 2000)
    WM_ROLLOUT_MAX_PROB:     Max probability of using predicted z (default: 0.5)
    WM_NORM_PROJECT:         "1" to project predictions to encoder norm (default: "0")
"""
from __future__ import annotations

import copy
import math
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

class KoopmanPredictor(nn.Module):
    """Linear dynamics predictor: z_{t+1} = (I + U·V^T)·z_t + W·z_action + bias.

    Inspired by Koopman operator theory: represent nonlinear dynamics as a
    linear operator in a lifted space.  When the state encoder is frozen
    (pre-trained), the encoder's feature space IS the lifted space, so a
    low-rank linear transition should capture the dynamics.

    ~8K params at dim=128, rank=16 vs LoopedPredictor's ~500K.
    """

    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.dim = dim
        self.rank = rank
        # Low-rank transition: K = I + U @ V.T
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(dim, rank) * 0.01)
        # Action projection
        self.W = nn.Linear(dim, dim)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_state: Tensor, z_action: Tensor) -> Tensor:
        """Predict next-state embedding via linear Koopman operator.

        Args:
            z_state:  [B, D] -- current state embedding
            z_action: [B, D] -- action embedding

        Returns:
            [B, D] -- predicted next-state embedding
        """
        # Low-rank residual: z + U @ V.T @ z
        delta = z_state @ self.V          # [B, rank]
        delta = delta @ self.U.T          # [B, dim]
        z_next = z_state + delta + self.W(z_action) + self.bias
        return self.norm(z_next)


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
# DiffEncoder — bottleneck encoder for code diffs (DeltaTok-inspired)
# ---------------------------------------------------------------------------

class DiffEncoder(nn.Module):
    """Bottleneck encoder that compresses diff tokens into a single delta vector.

    Inspired by DeltaTok (CVPR 2026): a learnable z_token attends to the diff
    token sequence, then only the z_token is extracted. This forces the output
    to encode ONLY the change, not the full state.

    Architecture:
        diff_tokens [B, S] -> embedding [B, S, D]
        prepend z_token -> [B, S+1, D]
        LoopedTransformerBlock x num_loops
        extract z_token at position 0 -> [B, D]
    """

    def __init__(
        self,
        vocab_size: int = 700,
        dim: int = 128,
        num_heads: int = 4,
        num_loops: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_enc = nn.Embedding(max_seq_len + 1, dim)  # +1 for z_token
        self.z_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.block = LoopedTransformerBlock(dim, num_heads, mlp_ratio, dropout)
        self.num_loops = num_loops
        self.norm = nn.LayerNorm(dim)

    def forward(self, diff_tokens: Tensor) -> Tensor:
        """Encode diff tokens to a single delta vector.

        Args:
            diff_tokens: [B, S] int64 — tokenized unified diff

        Returns:
            [B, D] — delta embedding (the change vector)
        """
        B, S = diff_tokens.shape
        h = self.embedding(diff_tokens)  # [B, S, D]
        # Positional encoding
        positions = torch.arange(S, device=diff_tokens.device)
        h = h + self.pos_enc(positions + 1)  # +1 to reserve 0 for z_token
        # Prepend learnable bottleneck z_token
        z = self.z_token.expand(B, -1, -1) + self.pos_enc(torch.zeros(1, dtype=torch.long, device=diff_tokens.device))
        h = torch.cat([z, h], dim=1)  # [B, S+1, D]
        # Looped transformer
        for _ in range(self.num_loops):
            h = self.block(h)
        # Extract z_token (position 0) — this IS the delta
        return self.norm(h[:, 0])  # [B, D]


# ---------------------------------------------------------------------------
# LogCosh loss (from DeltaTok — smooth near 0, robust to outliers)
# ---------------------------------------------------------------------------

def logcosh_loss(pred: Tensor, target: Tensor) -> Tensor:
    """LogCosh loss: smooth L2 near zero, L1 for large values.

    From DeltaTok (CVPR 2026): handles small deltas gracefully (smooth gradient
    near zero unlike L1), while being robust to large deltas (unlike L2 which
    squares the error).

    Formula: mean(|x| + softplus(-2|x|) - log(2))
    """
    diff = pred - target
    abs_diff = diff.abs()
    return (abs_diff + F.softplus(-2.0 * abs_diff) - math.log(2.0)).mean()


# ---------------------------------------------------------------------------
# Early Layer Fusion (Phase 8 — geometry-informed readout)
# ---------------------------------------------------------------------------

class EarlyLayerFusion(nn.Module):
    """Fuse encoder intermediate outputs via learnable scalar weights.

    Phase 0 showed early loop iterations have 12x more delta signal than
    the final output. This module selects specific loop outputs and
    computes a weighted sum with learned (softmax-normalized) scalars.

    Args:
        num_layers: number of loop iterations to fuse
        dim: embedding dimension (for optional LayerNorm)
    """

    def __init__(self, num_layers: int, dim: int):
        super().__init__()
        # Raw logits; softmax at forward time so weights sum to 1
        self.weight_logits = nn.Parameter(torch.zeros(num_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, intermediates: list[Tensor]) -> Tensor:
        """Fuse a list of [B, D] tensors into a single [B, D] output."""
        weights = torch.softmax(self.weight_logits, dim=0)
        fused = torch.zeros_like(intermediates[0])
        for w, z in zip(weights, intermediates):
            fused = fused + w * z
        return self.norm(fused)


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
        ema_decay: float = 0.99999,
        sigreg_weight: float = 0.1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_dim: int = 4,
        lambda_pred: float = 1.0,
        lambda_dir: float = 0.5,
        lambda_mag: float = 0.1,
        lambda_cov: float = 0.01,
        # ── Composition losses (opt-in, default disabled) ──
        rollout_steps: int = 0,
        lambda_rollout: float = 0.0,
        rollout_loss_decay: float = 0.5,
        lambda_path_consistency: float = 0.0,
        # ── Contrastive retrieval loss (Phase 4, opt-in) ──
        # Pulls deltas with similar action vectors closer in latent space.
        # Supervised contrastive (Khosla et al. 2020) with action cosine
        # as the supervision signal.
        lambda_contrast: float = 0.0,
        contrast_temperature: float = 0.07,
        contrast_pos_threshold: float = 0.9,
        # ── Rollout robustness (Phase 6, opt-in) ──
        noise_sigma: float = 0.0,
        noise_anneal_steps: int = 0,
        scheduled_rollout: bool = False,
        rollout_warmup_steps: int = 2000,
        rollout_max_prob: float = 0.5,
        norm_project: bool = False,
        # ── Delta-native mode (Phase 7, DeltaTok-inspired) ──
        delta_mode: bool = False,
        diff_vocab_size: int = 700,
        diff_encoder_loops: int = 4,
        freeze_diff_encoder: bool = False,
        predictor_dropout: float | None = None,
        # ── Early-layer readout (Phase 8, geometry-informed) ──
        # Read from early loop iterations where delta signal is 12x stronger.
        # When enabled, encoder must support return_intermediates=True.
        early_readout: bool = False,
        early_readout_loops: tuple[int, ...] = (1, 2, 3),  # 1-indexed loop iterations
        # ── Auxiliary loop losses (Phase 9, deep supervision) ──
        # Apply contrastive or predictive loss at intermediate loop outputs.
        # Prevents gradient attenuation that causes 12x delta signal compression.
        aux_loops: tuple[int, ...] = (),  # 1-indexed loop indices, e.g. (1, 2, 3)
        lambda_aux: tuple[float, ...] | float = 0.3,  # per-loop weight(s) or single weight
        aux_type: str = "contrast",  # "contrast" or "pred"
        # ── Dense skip connections (Phase 9) ──
        # Add z_initial residual at each encoder loop to preserve input signal.
        dense_skip: bool = False,
        # ── State-space delta prediction (Phase 8) ──
        # Predict target_encoder(x_{t+1}) - target_encoder(x_t) directly.
        # Different from delta_mode which uses a separate diff_encoder.
        delta_statespace: bool = False,
        # Fix 1: Use normalized MSE on deltas (strip magnitude, match baseline fairness)
        delta_normalize_loss: bool = False,
        # Fix 2: Residual predictor — pred output is z_t + correction, delta = pred - z_t
        delta_residual: bool = False,
        # ── Koopman predictor (Phase 11) ──
        predictor_type: str = "looped",  # "looped" or "koopman"
        koopman_rank: int = 16,
        # ── Frozen encoder mode (Phase 11, DeltaTok-inspired) ──
        freeze_encoder: bool = False,
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
        self.rollout_steps = rollout_steps
        self.lambda_rollout = lambda_rollout
        self.rollout_loss_decay = rollout_loss_decay
        self.lambda_path_consistency = lambda_path_consistency
        self.lambda_contrast = lambda_contrast
        self.contrast_temperature = contrast_temperature
        self.contrast_pos_threshold = contrast_pos_threshold
        # Phase 6: rollout robustness
        self.delta_mode = delta_mode
        self.diff_vocab_size = diff_vocab_size
        self.diff_encoder_loops = diff_encoder_loops
        self.freeze_diff_encoder = freeze_diff_encoder
        self.predictor_dropout = predictor_dropout
        self.early_readout = early_readout
        self.early_readout_loops = early_readout_loops
        self.aux_loops = aux_loops
        if isinstance(lambda_aux, (int, float)):
            self.lambda_aux = tuple(float(lambda_aux) for _ in aux_loops) if aux_loops else ()
        else:
            self.lambda_aux = tuple(float(x) for x in lambda_aux)
        self.aux_type = aux_type
        self.dense_skip = dense_skip
        self.delta_statespace = delta_statespace
        self.delta_normalize_loss = delta_normalize_loss
        self.delta_residual = delta_residual
        self._delta_logged = False
        self.predictor_type = predictor_type
        self.koopman_rank = koopman_rank
        self.freeze_encoder = freeze_encoder
        self.noise_sigma = noise_sigma
        self.noise_anneal_steps = noise_anneal_steps
        self.scheduled_rollout = scheduled_rollout
        self.rollout_warmup_steps = rollout_warmup_steps
        self.rollout_max_prob = rollout_max_prob
        self.norm_project = norm_project

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
        # In delta_mode, target comes from diff_encoder instead — EMA is optional
        self.target_encoder = copy.deepcopy(state_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Phase 11: freeze state encoder — only predictor + action encoder train
        if self.freeze_encoder:
            self.state_encoder.requires_grad_(False)
            self.state_encoder.train(False)
            n_frozen = sum(p.numel() for p in self.state_encoder.parameters())
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[WMBase] FROZEN encoder ({n_frozen:,} params frozen, {n_trainable:,} trainable)")

        # Delta-native mode: build diff encoder for encoding code diffs
        self.diff_encoder: nn.Module | None = None
        if self.delta_mode:
            self.diff_encoder = DiffEncoder(
                vocab_size=self.diff_vocab_size,
                dim=self.model_dim,
                num_heads=self.num_heads,
                num_loops=self.diff_encoder_loops,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout_rate,
            )
            if self.freeze_diff_encoder:
                self.diff_encoder.requires_grad_(False)
                self.diff_encoder.eval()  # PyTorch nn.Module.eval(), not Python eval()
                n_params = sum(p.numel() for p in self.diff_encoder.parameters())
                print(f"[WMBase] diff_encoder FROZEN ({n_params} params)")

        # Early-layer fusion (Phase 8)
        self.layer_fusion: EarlyLayerFusion | None = None
        if self.early_readout:
            self.layer_fusion = EarlyLayerFusion(
                num_layers=len(self.early_readout_loops),
                dim=self.model_dim,
            )
            print(f"[WMBase] early_readout ENABLED: loops={self.early_readout_loops}")

        if self.aux_loops:
            print(f"[WMBase] aux_loops ENABLED: loops={self.aux_loops}, "
                  f"lambda={self.lambda_aux}, type={self.aux_type}")

        if self.dense_skip:
            print("[WMBase] dense_skip ENABLED: z_initial residual at each encoder loop")

        # Predictor: looped (default) or koopman (Phase 11)
        _pred_dropout = self.predictor_dropout if self.predictor_dropout is not None else self.dropout_rate
        if self.predictor_type == "koopman":
            self.predictor = KoopmanPredictor(
                dim=self.model_dim,
                rank=self.koopman_rank,
            )
            print(f"[WMBase] KoopmanPredictor: dim={self.model_dim}, rank={self.koopman_rank}, "
                  f"params={sum(p.numel() for p in self.predictor.parameters()):,}")
        else:
            self.predictor = LoopedPredictor(
                dim=self.model_dim,
                num_heads=self.num_heads,
                depth=self.predictor_depth,
                num_loops=self.num_loops,
                mlp_ratio=self.mlp_ratio,
                dropout=_pred_dropout,
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

        # Step counter for scheduled rollout / noise annealing
        self.register_buffer("_train_step", torch.tensor(0, dtype=torch.long))

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen diff_encoder in eval regardless of model.train() calls
        if self.freeze_diff_encoder and self.diff_encoder is not None:
            self.diff_encoder.eval()  # PyTorch nn.Module.eval()
        return self

    # ── Phase 6: Rollout robustness helpers ──

    def _get_rollout_prob(self) -> float:
        """Compute scheduled rollout probability based on training step."""
        step = self._train_step.item()
        if step < self.rollout_warmup_steps:
            return 0.0
        # Linear ramp from 0 to max_prob over rollout_warmup_steps after warmup
        ramp_steps = max(self.rollout_warmup_steps, 1)
        progress = min((step - self.rollout_warmup_steps) / ramp_steps, 1.0)
        return self.rollout_max_prob * progress

    def _get_current_sigma(self) -> float:
        """Compute current noise sigma (optionally annealed)."""
        if self.noise_anneal_steps <= 0:
            return self.noise_sigma
        step = self._train_step.item()
        progress = min(step / self.noise_anneal_steps, 1.0)
        return self.noise_sigma * (1.0 - progress)

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
        if self.freeze_encoder:
            return  # frozen encoder: target stays at its init snapshot
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

        # Early-layer readout: extract intermediates from encoder loop iterations
        # Also extract intermediates when aux_loops is set (for deep supervision)
        need_intermediates = (
            (self.early_readout and self.layer_fusion is not None)
            or bool(self.aux_loops)
        )
        self._last_intermediates = None
        if need_intermediates:
            result = self.state_encoder(flat_states, return_intermediates=True)
            z_final, intermediates = result
            self._last_intermediates = intermediates  # cache for aux loss
            if self.early_readout and self.layer_fusion is not None:
                selected = [intermediates[i - 1] for i in self.early_readout_loops]
                z_all = self.layer_fusion(selected).reshape(B, T, -1)  # [B, T, D]
            else:
                z_all = z_final.reshape(B, T, -1)  # [B, T, D]
        else:
            z_all = self.state_encoder(flat_states).reshape(B, T, -1)  # [B, T, D]

        # Encode actions
        action_shape = actions.shape[2:]
        flat_actions = actions.reshape(B * num_transitions, *action_shape)
        z_action = self.action_encoder(flat_actions).reshape(B, num_transitions, -1)  # [B, T-1, D]

        # Compute target embeddings
        # State-space delta mode: target = target_enc(next) - target_enc(current)
        # Delta mode: target = diff_encoder(diff_tokens)
        # Standard mode: target = EMA encoder(next_states)
        diff_tokens = kwargs.get("diff_tokens")
        use_delta = self.delta_mode and diff_tokens is not None and self.diff_encoder is not None

        if not self._delta_logged:
            print(f"[WMBase] use_delta={use_delta} (delta_mode={self.delta_mode}, "
                  f"diff_tokens={'present' if diff_tokens is not None else 'None'}, "
                  f"diff_encoder={'built' if self.diff_encoder is not None else 'None'}, "
                  f"frozen={self.freeze_diff_encoder}), "
                  f"delta_statespace={self.delta_statespace}, "
                  f"early_readout={self.early_readout}")
            self._delta_logged = True

        if use_delta:
            diff_shape = diff_tokens.shape[2:]
            flat_diffs = diff_tokens[:, :num_transitions].reshape(B * num_transitions, *diff_shape)
            z_target = self.diff_encoder(flat_diffs).reshape(B, num_transitions, -1)  # [B, T-1, D]
            if self.freeze_diff_encoder:
                z_target = z_target.detach()
        else:
            with torch.no_grad():
                flat_next_states = states[:, 1:].reshape(B * num_transitions, *state_shape)
                if self.early_readout and self.layer_fusion is not None:
                    # Target also uses early-layer fusion (same loop iterations)
                    tgt_result = self.target_encoder(flat_next_states, return_intermediates=True)
                    _, tgt_intermediates = tgt_result
                    tgt_selected = [tgt_intermediates[i - 1] for i in self.early_readout_loops]
                    z_target = self.layer_fusion(tgt_selected).reshape(B, num_transitions, -1)
                else:
                    z_target = self.target_encoder(flat_next_states).reshape(B, num_transitions, -1)  # [B, T-1, D]

        # Predict next-state embeddings for each transition
        # Phase 6: optionally inject noise and/or use scheduled rollout
        pred_list = []
        use_noise = self.training and self.noise_sigma > 0
        use_sched = self.training and self.scheduled_rollout and num_transitions > 1
        current_sigma = self._get_current_sigma() if use_noise else 0.0
        rollout_prob = self._get_rollout_prob() if use_sched else 0.0
        # Pre-compute encoder norm for norm projection
        if self.norm_project:
            with torch.no_grad():
                _enc_norms = z_all.reshape(-1, self.model_dim).norm(dim=-1)
                _enc_mean_norm = _enc_norms.mean()

        for t in range(num_transitions):
            # Select input: teacher-forced or rolled-out
            if t > 0 and use_sched and rollout_prob > 0:
                if torch.rand(1).item() < rollout_prob:
                    z_input = pred_list[t - 1].detach()
                else:
                    z_input = z_all[:, t].detach()
            else:
                z_input = z_all[:, t].detach()

            # Noise injection: simulate off-manifold inputs
            if use_noise and current_sigma > 0:
                z_input = z_input + torch.randn_like(z_input) * current_sigma

            z_pred_t = self.predictor(z_input, z_action[:, t])

            # Norm projection: keep predictions on the encoder norm ball
            if self.norm_project:
                pred_norm = z_pred_t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                z_pred_t = z_pred_t * (_enc_mean_norm / pred_norm)

            pred_list.append(z_pred_t)

        pred_embeddings = torch.stack(pred_list, dim=1)  # [B, T-1, D]

        # ─── Loss A: Prediction loss ───
        pred_flat = pred_embeddings.reshape(-1, self.model_dim)
        target_flat = z_target.reshape(-1, self.model_dim)

        z_current = z_all[:, :-1].detach().reshape(-1, self.model_dim)  # [N, D]

        if self.delta_statespace:
            # State-space delta mode (Phase 8): predict state-space deltas
            with torch.no_grad():
                z_current_target = z_target.reshape(-1, self.model_dim)  # target_enc(x_{t+1})
                flat_current_states = states[:, :-1].reshape(B * num_transitions, *state_shape)
                if self.early_readout and self.layer_fusion is not None:
                    cur_result = self.target_encoder(flat_current_states, return_intermediates=True)
                    _, cur_intermediates = cur_result
                    cur_selected = [cur_intermediates[i - 1] for i in self.early_readout_loops]
                    z_current_target_prev = self.layer_fusion(cur_selected)
                else:
                    z_current_target_prev = self.target_encoder(flat_current_states)
            delta_true = z_current_target.reshape(-1, self.model_dim) - z_current_target_prev.detach()

            # Fix 2: Residual predictor — interpret pred as z_t + correction
            # so delta_pred = pred - z_current (predictor works near identity)
            if self.delta_residual:
                delta_pred = pred_flat - z_current
            else:
                delta_pred = pred_flat

            # Fix 1: Normalized loss — strip magnitude, match baseline fairness
            if self.delta_normalize_loss:
                loss_pred = F.mse_loss(
                    F.normalize(delta_pred, dim=-1),
                    F.normalize(delta_true, dim=-1),
                )
            else:
                loss_pred = logcosh_loss(delta_pred, delta_true)
        elif use_delta:
            # Delta mode (Phase 7): target from diff_encoder
            loss_pred = logcosh_loss(pred_flat, target_flat)
            delta_true = target_flat
            delta_pred = pred_flat
        else:
            # Standard mode: MSE on normalized embeddings
            use_projector = os.environ.get("WM_USE_PROJECTOR", "1") == "1"
            if use_projector:
                loss_pred = F.mse_loss(
                    F.normalize(self.pred_projector(pred_flat), dim=-1),
                    F.normalize(self.target_projector(target_flat), dim=-1),
                )
            else:
                loss_pred = F.mse_loss(
                    F.normalize(pred_flat, dim=-1),
                    F.normalize(target_flat, dim=-1),
                )
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

        # ─── Predictor-output regularizer (Phase 9.5) ───
        # Phase 9 audit: eff_rank_pred collapses to ~6/128 across all checkpoints
        # even when eff_rank_online stays at 53-63. Apply a variance hinge directly
        # on the predictor output to prevent that collapse.
        pred_reg_mode = os.environ.get("WM_PRED_REG_MODE", "none")
        lambda_pred_reg = float(os.environ.get("WM_LAMBDA_PRED_REG", "0.0"))
        pred_reg_loss = torch.tensor(0.0, device=z_all.device)
        if pred_reg_mode == "vicreg" and lambda_pred_reg > 0:
            pred_std = pred_flat.std(dim=0)
            pred_reg_loss = F.relu(1.0 - pred_std).mean()

        # ─── Combined loss (base) ───
        loss = (self.lambda_pred * loss_pred
                + self.lambda_dir * loss_dir
                + self.lambda_mag * loss_mag
                + self.lambda_cov * loss_cov
                + self.sigreg_weight * static_reg_loss
                + lambda_pred_reg * pred_reg_loss)

        # Diagnostic: mean delta cosine sim and norm ratio
        delta_cos_sim = cos_sim.mean()
        delta_norm_ratio = (dp_norm / dt_norm).mean()

        # ─── Composition losses (opt-in, only during training) ───
        loss_rollout = torch.tensor(0.0, device=states.device)
        loss_path_consistency = torch.tensor(0.0, device=states.device)

        if self.training and num_transitions > 1:
            # Multi-step rollout loss: roll out predictor autoregressively,
            # penalize divergence from ground-truth targets at steps 2+.
            if self.lambda_rollout > 0 and self.rollout_steps > 0:
                max_steps = min(self.rollout_steps, num_transitions)
                z_rolled = pred_list[0]  # step-1 prediction (already computed)
                decay = self.rollout_loss_decay
                rollout_total = torch.tensor(0.0, device=states.device)
                for k in range(1, max_steps):
                    z_rolled = self.predictor(z_rolled, z_action[:, k])
                    step_loss = F.mse_loss(
                        F.normalize(z_rolled, dim=-1),
                        F.normalize(z_target[:, k], dim=-1),
                    )
                    rollout_total = rollout_total + (decay ** k) * step_loss
                loss_rollout = rollout_total / max(max_steps - 1, 1)
                loss = loss + self.lambda_rollout * loss_rollout

            # Path consistency loss: for consecutive transition pairs (t, t+1),
            # 2-step composition should reach the same place as ground truth.
            # predict(predict(z_t, a_t), a_{t+1}) ≈ target_{t+1}
            if self.lambda_path_consistency > 0:
                pc_total = torch.tensor(0.0, device=states.device)
                n_pairs = num_transitions - 1
                for t in range(n_pairs):
                    z_step1 = self.predictor(
                        z_all[:, t].detach(), z_action[:, t],
                    )
                    z_step2 = self.predictor(z_step1, z_action[:, t + 1])
                    pc_total = pc_total + F.mse_loss(
                        F.normalize(z_step2, dim=-1),
                        F.normalize(z_target[:, t + 1], dim=-1),
                    )
                loss_path_consistency = pc_total / max(n_pairs, 1)
                loss = loss + self.lambda_path_consistency * loss_path_consistency

        # ─── Contrastive retrieval loss (Phase 4, opt-in) ───
        # Supervised contrastive on the actual edit deltas, supervised by
        # action vector cosine similarity. Pulls action-similar deltas
        # together, pushes action-different deltas apart in latent space.
        # This directly attacks the retrieval objective without changing
        # the prediction objective.
        #
        # Stability notes (prevents NaN blowup at high lambda):
        #   1. Normalize with explicit clamp_min on the delta norm (not
        #      F.normalize's tiny default eps) so near-zero deltas don't
        #      produce large-magnitude unit vectors.
        #   2. Filter out rows with degenerate deltas (norm < eps) from the
        #      contrastive loss entirely.
        #   3. Check final loss is finite; skip this step if not.
        loss_contrast = torch.tensor(0.0, device=states.device)
        if self.training and self.lambda_contrast > 0:
            # delta_true is [N, D] where N = B * num_transitions
            delta_norm_vec = delta_true.norm(dim=-1, keepdim=True)
            norm_eps = 1e-4
            # Mask out degenerate rows (near-zero deltas)
            nondegenerate = (delta_norm_vec.squeeze(-1) > norm_eps)
            n_valid_rows = int(nondegenerate.sum().item())

            if n_valid_rows >= 2:
                # Select only non-degenerate rows
                delta_keep = delta_true[nondegenerate]
                actions_keep = actions.reshape(-1, actions.shape[-1])[nondegenerate]

                # Normalize with a larger clamp to prevent gradient explosion
                delta_n = delta_keep / delta_keep.norm(
                    dim=-1, keepdim=True
                ).clamp_min(norm_eps)
                N = delta_n.shape[0]

                # Pairwise delta cosine similarity (the latent representation)
                # Clamp temperature to a reasonable floor
                temp = max(float(self.contrast_temperature), 0.01)
                delta_sim = (delta_n @ delta_n.T) / temp

                # Supervision: pairwise action cosine
                act_n = actions_keep / actions_keep.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-6)
                action_sim = act_n @ act_n.T  # [N, N]

                # Positive pairs: high action cosine, exclude self
                eye = torch.eye(N, device=delta_n.device, dtype=torch.bool)
                pos_mask = (action_sim > self.contrast_pos_threshold) & ~eye

                pos_count = pos_mask.float().sum(dim=-1)
                valid_anchors = (pos_count > 0)

                if valid_anchors.any():
                    # Standard supervised contrastive loss (NT-Xent)
                    delta_sim_masked = delta_sim.masked_fill(eye, -float("inf"))
                    log_prob = F.log_softmax(delta_sim_masked, dim=-1)

                    mean_log_prob_pos = (
                        (pos_mask.float() * log_prob).sum(dim=-1)
                        / pos_count.clamp_min(1)
                    )
                    loss_contrast = -(
                        mean_log_prob_pos * valid_anchors.float()
                    ).sum() / valid_anchors.float().sum().clamp_min(1)

                    # Safety: only add to loss if finite
                    if torch.isfinite(loss_contrast):
                        loss = loss + self.lambda_contrast * loss_contrast
                    else:
                        loss_contrast = torch.tensor(0.0, device=states.device)

        # ─── Auxiliary loop losses (Phase 9, deep supervision) ───
        # Apply contrastive or predictive loss at intermediate encoder loops
        # to prevent gradient attenuation from destroying delta signal.
        loss_aux = torch.tensor(0.0, device=states.device)
        if self.training and self.aux_loops and self._last_intermediates is not None:
            intermediates = self._last_intermediates  # [B*T, D] per loop
            for idx, loop_idx in enumerate(self.aux_loops):
                if loop_idx - 1 >= len(intermediates):
                    continue
                lam = self.lambda_aux[idx] if idx < len(self.lambda_aux) else self.lambda_aux[-1]
                # Reshape from [B*T, D] to [B, T, D], take current states [:, :-1]
                z_loop_all = intermediates[loop_idx - 1].reshape(B, T, -1)
                z_loop_cur = z_loop_all[:, :-1].reshape(B * num_transitions, -1)  # [B*(T-1), D]
                z_tgt_flat = z_target.reshape(B * num_transitions, -1)  # [B*(T-1), D]

                if self.aux_type == "cos_align":
                    # Plain cosine alignment between loop output and target
                    # (no action conditioning — likely impossible task, kept for ablation)
                    z_n = F.normalize(z_loop_cur, dim=-1)
                    z_n_t = F.normalize(z_tgt_flat, dim=-1)
                    aux_cos = (z_n * z_n_t).sum(dim=-1).mean()
                    loop_loss = 1.0 - aux_cos
                elif self.aux_type == "contrast":
                    # NT-Xent supervised contrastive loss on per-loop deltas
                    # Mirrors the final-layer lambda_contrast loss but applied
                    # at intermediate loop l. Deltas = target - loop_cur.
                    delta_loop = z_tgt_flat - z_loop_cur
                    delta_norm_vec = delta_loop.norm(dim=-1, keepdim=True)
                    norm_eps = 1e-4
                    nondegenerate = (delta_norm_vec.squeeze(-1) > norm_eps)
                    n_valid = int(nondegenerate.sum().item())
                    if n_valid >= 2:
                        delta_keep = delta_loop[nondegenerate]
                        actions_keep = actions.reshape(-1, actions.shape[-1])[nondegenerate]
                        delta_n = delta_keep / delta_keep.norm(
                            dim=-1, keepdim=True
                        ).clamp_min(norm_eps)
                        N = delta_n.shape[0]
                        temp = max(float(self.contrast_temperature), 0.01)
                        delta_sim = (delta_n @ delta_n.T) / temp
                        act_n = actions_keep / actions_keep.norm(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-6)
                        action_sim = act_n @ act_n.T
                        eye = torch.eye(N, device=delta_n.device, dtype=torch.bool)
                        pos_mask = (action_sim > self.contrast_pos_threshold) & ~eye
                        pos_count = pos_mask.float().sum(dim=-1)
                        valid_anchors = (pos_count > 0)
                        if valid_anchors.any():
                            delta_sim_masked = delta_sim.masked_fill(eye, -float("inf"))
                            log_prob = F.log_softmax(delta_sim_masked, dim=-1)
                            mean_log_prob_pos = (
                                (pos_mask.float() * log_prob).sum(dim=-1)
                                / pos_count.clamp_min(1)
                            )
                            loop_loss = -(
                                mean_log_prob_pos * valid_anchors.float()
                            ).sum() / valid_anchors.float().sum().clamp_min(1)
                        else:
                            loop_loss = torch.tensor(0.0, device=states.device)
                    else:
                        loop_loss = torch.tensor(0.0, device=states.device)
                else:
                    # "pred" (default): predict target from this loop's output
                    # via the predictor (ACTION-CONDITIONED).
                    z_act_flat = z_action.reshape(B * num_transitions, -1)
                    z_loop_pred = self.predictor(z_loop_cur, z_act_flat)
                    loop_loss = F.mse_loss(
                        F.normalize(z_loop_pred, dim=-1),
                        F.normalize(z_tgt_flat, dim=-1),
                    )

                if torch.isfinite(loop_loss):
                    loss_aux = loss_aux + lam * loop_loss

            loss = loss + loss_aux

        # ─── Trajectory straightness regularizer (Phase 9.5, per arxiv 2603.12231) ───
        # Penalize curvature of the encoder's per-loop latent trajectory:
        #   L_straight = mean(1 - cos(z_{l+1} - z_l, z_l - z_{l-1}))
        # Lower curvature = straighter latent path = JEPA-style trajectory smoothing.
        loss_straight = torch.tensor(0.0, device=states.device)
        lambda_straight = float(os.environ.get("WM_LAMBDA_STRAIGHTNESS", "0.0"))
        if (self.training and lambda_straight > 0
                and self._last_intermediates is not None
                and len(self._last_intermediates) >= 3):
            inters = self._last_intermediates
            triples = []
            for l in range(1, len(inters) - 1):
                d_prev = inters[l] - inters[l - 1]
                d_next = inters[l + 1] - inters[l]
                cos_l = F.cosine_similarity(d_prev, d_next, dim=-1)
                triples.append((1.0 - cos_l).mean())
            if triples:
                loss_straight = torch.stack(triples).mean()
                loss = loss + lambda_straight * loss_straight

        # Update EMA target encoder and step counter
        if self.training:
            self._update_ema()
            self._train_step += 1

        return {
            "loss": loss,
            "loss_pred": loss_pred,
            "loss_dir": loss_dir,
            "loss_mag": loss_mag,
            "loss_cov": loss_cov,
            "loss_static_reg": static_reg_loss,
            "loss_rollout": loss_rollout,
            "loss_path_consistency": loss_path_consistency,
            "loss_contrast": loss_contrast,
            "loss_aux": loss_aux,
            "loss_straight": loss_straight,
            "delta_cos_sim": delta_cos_sim,
            "delta_norm_ratio": delta_norm_ratio,
            "pred_embeddings": pred_embeddings,
            "target_embeddings": z_target,
            "z_std": z_std,
            # Phase 6 diagnostics
            "noise_sigma": torch.tensor(current_sigma if use_noise else 0.0),
            "rollout_prob": torch.tensor(rollout_prob if use_sched else 0.0),
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
        "ema_decay": float(_get("ema_decay", "WM_EMA_DECAY", "0.99999")),
        "sigreg_weight": float(_get("sigreg_weight", "WM_SIGREG_WEIGHT", "0.1")),
        "mlp_ratio": float(_get("mlp_ratio", "WM_MLP_RATIO", "4.0")),
        "dropout": float(_get("dropout", "WM_DROPOUT", "0.1")),
        "action_dim": int(_get("action_dim", "ACTION_DIM", "4")),
        "lambda_pred": float(_get("lambda_pred", "WM_LAMBDA_PRED", "1.0")),
        "lambda_dir": float(_get("lambda_dir", "WM_LAMBDA_DIR", "0.5")),
        "lambda_mag": float(_get("lambda_mag", "WM_LAMBDA_MAG", "0.1")),
        "lambda_cov": float(_get("lambda_cov", "WM_LAMBDA_COV", "0.01")),
        # Composition losses (opt-in)
        "rollout_steps": int(_get("rollout_steps", "WM_ROLLOUT_STEPS", "0")),
        "lambda_rollout": float(_get("lambda_rollout", "WM_LAMBDA_ROLLOUT", "0.0")),
        "rollout_loss_decay": float(_get("rollout_loss_decay", "WM_ROLLOUT_LOSS_DECAY", "0.5")),
        "lambda_path_consistency": float(_get("lambda_path_consistency", "WM_LAMBDA_PATH_CONSISTENCY", "0.0")),
        # Contrastive retrieval loss (Phase 4, opt-in)
        "lambda_contrast": float(_get("lambda_contrast", "WM_LAMBDA_CONTRAST", "0.0")),
        "contrast_temperature": float(_get("contrast_temperature", "WM_CONTRAST_TEMPERATURE", "0.07")),
        "contrast_pos_threshold": float(_get("contrast_pos_threshold", "WM_CONTRAST_POS_THRESHOLD", "0.9")),
        # Rollout robustness (Phase 6, opt-in)
        "noise_sigma": float(_get("noise_sigma", "WM_NOISE_SIGMA", "0.0")),
        "noise_anneal_steps": int(_get("noise_anneal_steps", "WM_NOISE_ANNEAL_STEPS", "0")),
        "scheduled_rollout": _get("scheduled_rollout", "WM_SCHEDULED_ROLLOUT", "0") == "1",
        "rollout_warmup_steps": int(_get("rollout_warmup_steps", "WM_ROLLOUT_WARMUP", "2000")),
        "rollout_max_prob": float(_get("rollout_max_prob", "WM_ROLLOUT_MAX_PROB", "0.5")),
        "norm_project": _get("norm_project", "WM_NORM_PROJECT", "0") == "1",
        # Delta-native mode (Phase 7, DeltaTok-inspired)
        "delta_mode": _get("delta_mode", "WM_DELTA_MODE", "0") == "1",
        "diff_vocab_size": int(_get("diff_vocab_size", "WM_DIFF_VOCAB_SIZE", "700")),
        "diff_encoder_loops": int(_get("diff_encoder_loops", "WM_DIFF_ENCODER_LOOPS", "4")),
        "freeze_diff_encoder": _get("freeze_diff_encoder", "WM_FREEZE_DIFF_ENCODER", "0") == "1",
        "predictor_dropout": float(_pd) if (_pd := _get("predictor_dropout", "WM_PREDICTOR_DROPOUT", "")) else None,
        # Early-layer readout (Phase 8, geometry-informed)
        "early_readout": _get("early_readout", "WM_EARLY_READOUT", "0") == "1",
        "early_readout_loops": tuple(
            int(x) for x in _get("early_readout_loops", "WM_EARLY_READOUT_LOOPS", "1,2,3").split(",")
        ),
        # Auxiliary loop losses (Phase 9, deep supervision)
        "aux_loops": tuple(
            int(x) for x in _get("aux_loops", "WM_AUX_LOOPS", "").split(",")
        ) if _get("aux_loops", "WM_AUX_LOOPS", "") else (),
        "lambda_aux": tuple(
            float(x) for x in _get("lambda_aux", "WM_LAMBDA_AUX", "0.3").split(",")
        ) if "," in _get("lambda_aux", "WM_LAMBDA_AUX", "0.3") else float(
            _get("lambda_aux", "WM_LAMBDA_AUX", "0.3")
        ),
        "aux_type": _get("aux_type", "WM_AUX_TYPE", "contrast"),
        # Dense skip connections (Phase 9)
        "dense_skip": _get("dense_skip", "WM_DENSE_SKIP", "0") == "1",
        # State-space delta prediction (Phase 8)
        "delta_statespace": _get("delta_statespace", "WM_DELTA_STATESPACE", "0") == "1",
        "delta_normalize_loss": _get("delta_normalize_loss", "WM_DELTA_NORMALIZE_LOSS", "0") == "1",
        "delta_residual": _get("delta_residual", "WM_DELTA_RESIDUAL", "0") == "1",
        # Koopman predictor (Phase 11)
        "predictor_type": _get("predictor_type", "WM_PREDICTOR_TYPE", "looped"),
        "koopman_rank": int(_get("koopman_rank", "WM_KOOPMAN_RANK", "16")),
        # Frozen encoder mode (Phase 11, DeltaTok-inspired)
        "freeze_encoder": _get("freeze_encoder", "WM_FREEZE_ENCODER", "0") == "1",
    }
