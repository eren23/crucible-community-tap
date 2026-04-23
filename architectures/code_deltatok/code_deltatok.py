"""CodeDeltaTok — DeltaTok-inspired code change tokenizer.

Compresses the difference between frozen backbone features of consecutive
code states into 1-K dense delta tokens. Inspired by DeltaTok (CVPR 2026)
which does the same for video frames using frozen DINOv3 features.

Key differences from our earlier DiffEncoder (Phase 7, collapsed):
  - Backbone is FROZEN — no representation collapse possible
  - Operates in frozen feature space (768-dim), not tiny JEPA space (128-dim)
  - Uses DeltaTok's proven stabilization: layer scale, gradient clipping
  - LogCosh loss on frozen features, not MSE on learned latent space

Architecture:
    prev_code → frozen_encoder → features_prev [B, D]
    next_code → frozen_encoder → features_next [B, D]

    Encoder:
        z_init (K learnable bottleneck tokens) + features_prev + features_next
        → N transformer blocks
        → extract z positions → delta_tokens [B, K, D]

    Decoder:
        delta_tokens + features_prev
        → N transformer blocks
        → reconstructed features_next_hat [B, D]

    Loss: LogCosh(features_next_hat, features_next)

Env vars:
    CDT_FEATURE_DIM:   Backbone feature dimension (default: 768)
    CDT_NUM_BLOCKS:    Transformer blocks per encoder/decoder (default: 6)
    CDT_NUM_HEADS:     Attention heads (default: 12)
    CDT_NUM_TOKENS:    Number of delta tokens K (default: 1)
    CDT_DROPOUT:       Dropout rate (default: 0.0)
    CDT_LAYER_SCALE:   Layer scale init value (default: 1e-5)
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Transformer block with DeltaTok stabilization
# ---------------------------------------------------------------------------

class DeltaTokBlock(nn.Module):
    """Pre-norm transformer block with layer scale + SwiGLU MLP.

    Uses DeltaTok's proven stabilization techniques:
      - Layer scale initialization (default 1e-5)
      - Pre-norm (LayerNorm before attention/MLP)
      - SwiGLU activation in MLP (replaces GELU)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)

        # SwiGLU MLP: hidden = dim * mlp_ratio, but split for gate
        hidden = int(dim * mlp_ratio)
        self.mlp_gate = nn.Linear(dim, hidden)
        self.mlp_up = nn.Linear(dim, hidden)
        self.mlp_down = nn.Linear(hidden, dim)
        self.mlp_drop = nn.Dropout(dropout)

        # Layer scale (DeltaTok stabilization)
        self.scale1 = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.scale2 = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        # Self-attention with layer scale
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.scale1 * h

        # SwiGLU MLP with layer scale
        h = self.norm2(x)
        gate = F.silu(self.mlp_gate(h))
        h = gate * self.mlp_up(h)
        h = self.mlp_drop(self.mlp_down(h))
        x = x + self.scale2 * h

        return x


# ---------------------------------------------------------------------------
# LogCosh loss (same as wm_base.py)
# ---------------------------------------------------------------------------

def logcosh_loss(pred: Tensor, target: Tensor) -> Tensor:
    """LogCosh loss: smooth near zero, robust to outliers."""
    diff = pred - target
    abs_diff = diff.abs()
    return (abs_diff + F.softplus(-2.0 * abs_diff) - math.log(2.0)).mean()


# ---------------------------------------------------------------------------
# CodeDeltaTok
# ---------------------------------------------------------------------------

class CodeDeltaTok(nn.Module):
    """DeltaTok-inspired delta tokenizer for code changes.

    Compresses the difference between frozen backbone features of
    consecutive code states into K dense delta tokens (default K=1).
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_blocks: int = 6,
        num_heads: int = 12,
        num_delta_tokens: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_delta_tokens = num_delta_tokens

        # Learnable bottleneck token(s) — the delta representation
        self.z_embed = nn.Parameter(
            torch.randn(1, num_delta_tokens, feature_dim) * 0.02,
        )

        # Positional embeddings to distinguish prev/next/z
        self.pos_prev = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.pos_next = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.pos_z = nn.Parameter(torch.randn(1, num_delta_tokens, feature_dim) * 0.02)

        # Encoder: compress (prev, next) → z
        self.encoder = nn.ModuleList([
            DeltaTokBlock(feature_dim, num_heads, mlp_ratio, dropout, layer_scale_init)
            for _ in range(num_blocks)
        ])
        self.encoder_norm = nn.LayerNorm(feature_dim)

        # Decoder: reconstruct next from (z, prev)
        self.decoder = nn.ModuleList([
            DeltaTokBlock(feature_dim, num_heads, mlp_ratio, dropout, layer_scale_init)
            for _ in range(num_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[CodeDeltaTok] {n_params:,} params "
              f"(blocks={num_blocks}, heads={num_heads}, "
              f"dim={feature_dim}, K={num_delta_tokens})")

    def encode(self, prev_feat: Tensor, next_feat: Tensor) -> Tensor:
        """Compress code change into K delta tokens.

        Args:
            prev_feat: [B, D] — frozen backbone features of previous code state
            next_feat: [B, D] — frozen backbone features of next code state

        Returns:
            [B, K, D] — K delta tokens capturing the change
        """
        B = prev_feat.shape[0]

        # Build input sequence: [z_1..z_K, prev, next]
        z = self.z_embed.expand(B, -1, -1) + self.pos_z
        prev = prev_feat.unsqueeze(1) + self.pos_prev  # [B, 1, D]
        nxt = next_feat.unsqueeze(1) + self.pos_next    # [B, 1, D]
        h = torch.cat([z, prev, nxt], dim=1)  # [B, K+2, D]

        for blk in self.encoder:
            h = blk(h)
        h = self.encoder_norm(h)

        # Extract delta tokens (first K positions)
        return h[:, :self.num_delta_tokens]  # [B, K, D]

    def decode(self, delta_tokens: Tensor, prev_feat: Tensor) -> Tensor:
        """Reconstruct next-state features from delta tokens + prev.

        Args:
            delta_tokens: [B, K, D] — compressed delta tokens
            prev_feat: [B, D] — frozen features of previous state

        Returns:
            [B, D] — reconstructed next-state features
        """
        prev = prev_feat.unsqueeze(1) + self.pos_prev  # [B, 1, D]
        h = torch.cat([delta_tokens, prev], dim=1)  # [B, K+1, D]

        for blk in self.decoder:
            h = blk(h)
        h = self.decoder_norm(h)

        # Reconstruction from the prev position (after attending to delta)
        recon = h[:, self.num_delta_tokens]  # [B, D] — the prev position
        return self.out_proj(recon)

    def forward(
        self, prev_feat: Tensor, next_feat: Tensor,
    ) -> dict[str, Tensor]:
        """Full forward: encode delta, decode, compute loss.

        Args:
            prev_feat: [B, D] — frozen features of before-state
            next_feat: [B, D] — frozen features of after-state

        Returns:
            dict with loss, delta_tokens, recon, and diagnostics
        """
        # Encode → delta tokens
        delta_tokens = self.encode(prev_feat, next_feat)

        # Decode → reconstructed next features
        recon = self.decode(delta_tokens, prev_feat)

        # Loss: LogCosh in frozen feature space
        loss = logcosh_loss(recon, next_feat)

        # Diagnostics
        with torch.no_grad():
            recon_cos = F.cosine_similarity(recon, next_feat, dim=-1).mean()
            # Delta token effective rank
            dt_flat = delta_tokens.reshape(-1, self.feature_dim)
            if dt_flat.shape[0] > 1:
                svd = torch.linalg.svdvals(dt_flat - dt_flat.mean(dim=0))
                p = svd / svd.sum()
                eff_rank = torch.exp(-torch.sum(p * torch.log(p + 1e-8)))
            else:
                eff_rank = torch.tensor(0.0)
            # Raw delta baseline (how similar are before/after?)
            raw_cos = F.cosine_similarity(prev_feat, next_feat, dim=-1).mean()

        return {
            "loss": loss,
            "delta_tokens": delta_tokens,
            "recon": recon,
            "recon_cos": recon_cos,
            "delta_eff_rank": eff_rank,
            "raw_before_after_cos": raw_cos,
        }


def codedeltatik_from_env() -> CodeDeltaTok:
    """Build CodeDeltaTok from environment variables."""
    return CodeDeltaTok(
        feature_dim=int(os.environ.get("CDT_FEATURE_DIM", "768")),
        num_blocks=int(os.environ.get("CDT_NUM_BLOCKS", "6")),
        num_heads=int(os.environ.get("CDT_NUM_HEADS", "12")),
        num_delta_tokens=int(os.environ.get("CDT_NUM_TOKENS", "1")),
        mlp_ratio=float(os.environ.get("CDT_MLP_RATIO", "4.0")),
        dropout=float(os.environ.get("CDT_DROPOUT", "0.0")),
        layer_scale_init=float(os.environ.get("CDT_LAYER_SCALE", "1e-5")),
    )
