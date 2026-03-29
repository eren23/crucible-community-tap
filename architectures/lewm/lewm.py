"""LE-WM: Latent Emergent World Model architecture plugin.

ViT-Tiny encoder producing a single CLS token (192-dim) + DiT predictor
with Adaptive Layer Normalization (AdaLN) for action conditioning.

Key LE-WM insight: NO EMA target encoder. The encoder's outputs serve
directly as targets (with stop-gradient on the target side). SIGReg
regularization prevents collapse instead of EMA.

Env vars (defaults match HF quentinll/lewm-pusht config):
    MODEL_DIM:           Embedding dimension (default: 192)
    PATCH_SIZE:          ViT patch size (default: 14)
    IMAGE_SIZE:          Input image size (default: 224)
    ENCODER_DEPTH:       Number of ViT transformer blocks (default: 6)
    ENCODER_HEADS:       Number of attention heads in encoder (default: 3)
    PREDICTOR_DEPTH:     Number of DiT blocks (default: 6)
    PREDICTOR_HEADS:     Number of attention heads in predictor (default: 16)
    PREDICTOR_DIM_HEAD:  Dimension per head in predictor (default: 64, inner_dim=1024)
    PREDICTOR_MLP_DIM:   MLP hidden dim in predictor (default: 2048)
    ACTION_DIM:          Action vector dimensionality (default: 10)
    NUM_FRAMES:          Sub-trajectory length for training (default: 3)
    SIGREG_WEIGHT:       Weight for SIGReg regularizer (default: 1.0)
    SIGREG_PROJECTIONS:  Number of random projections (default: 128)
    DROPOUT:             Predictor dropout rate (default: 0.1)
    PRETRAINED_WEIGHTS:  Path to pretrained weights.pt for fine-tuning (default: "")
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.base import CrucibleModel
from crucible.models.registry import register_model


# ---------------------------------------------------------------------------
# ViT Encoder
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    """Standard ViT transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder producing a single CLS token embedding.

    Architecture follows ViT-Tiny:
        - Patch embedding with learnable position embeddings
        - CLS token prepended
        - N transformer blocks
        - Extract CLS token as the frame representation
        - BatchNorm projection (critical for SIGReg to work)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Post-encoder projection with BatchNorm (critical for SIGReg)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Encode image to CLS token embedding.

        Args:
            x: [B, C, H, W]

        Returns:
            [B, embed_dim]
        """
        B = x.shape[0]

        # Patch embed + CLS token + position
        patches = self.patch_embed(x)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)  # [B, N+1, D]
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Extract CLS token
        x = self.norm(x[:, 0])  # [B, D]

        # BatchNorm projection
        x = self.proj(x)  # [B, D]

        return x


# ---------------------------------------------------------------------------
# DiT Predictor with AdaLN
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive Layer Normalization — modulates norm output with action-conditioned scale/shift."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, dim * 2)  # scale + shift
        # Zero-initialize for training stability (LE-WM paper)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT transformer block with AdaLN action conditioning.

    Attention uses inner_dim = dim_head * num_heads (can differ from dim).
    MLP uses mlp_dim as intermediate size.
    """

    def __init__(self, dim: int, num_heads: int, cond_dim: int, dim_head: int = 64, mlp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.adaln1 = AdaLN(dim, cond_dim)
        # Custom attention with separate inner_dim
        self.q_proj = nn.Linear(dim, inner_dim, bias=False)
        self.k_proj = nn.Linear(dim, inner_dim, bias=False)
        self.v_proj = nn.Linear(dim, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.attn_dropout = nn.Dropout(dropout)

        self.adaln2 = AdaLN(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.adaln1(x, cond)
        B, N, _ = h.shape
        q = self.q_proj(h).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.k_proj(h).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.v_proj(h).reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)
        attn_out = self.attn_dropout(self.out_proj(attn_out))
        x = x + attn_out
        x = x + self.mlp(self.adaln2(x, cond))
        return x


class DiTPredictor(nn.Module):
    """DiT-based dynamics predictor with AdaLN action conditioning.

    Predicts z_{t+1} from (z_t, action_t) using transformer blocks
    with adaptive layer normalization modulated by the action embedding.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        action_dim: int = 10,
        depth: int = 6,
        num_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Action encoder (matches HF: input_dim -> emb_dim with MLP scale 4)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Transformer operates at hidden_dim = embed_dim (192)
        # Attention projects to inner_dim = num_heads * dim_head (1024) internally
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, embed_dim, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict next-step embedding.

        Args:
            z: [B, embed_dim] — current latent state
            action: [B, action_dim] — action taken

        Returns:
            [B, embed_dim] — predicted next latent state
        """
        # Encode action as conditioning signal
        cond = self.action_encoder(action)  # [B, embed_dim]

        # Transformer operates at embed_dim
        h = z.unsqueeze(1)  # [B, 1, embed_dim]

        # DiT blocks with action conditioning
        for block in self.blocks:
            h = block(h, cond.unsqueeze(1))

        h = self.norm(h.squeeze(1))  # [B, embed_dim]
        return h


# ---------------------------------------------------------------------------
# LE-WM World Model
# ---------------------------------------------------------------------------

class LeWMModel(CrucibleModel):
    """LE-WM: Latent Emergent World Model.

    JEPA world model that trains end-to-end from pixels using SIGReg
    regularization instead of EMA. The encoder produces a single 192-dim
    CLS token (784x spatial compression) and the predictor uses DiT
    blocks with AdaLN for action conditioning.

    Training flow:
        1. Encode all frames: z_t = Encoder(o_t)
        2. For each transition: z_hat_{t+1} = Predictor(z_t.detach(), a_t)
        3. Loss = MSE(z_hat_{t+1}, z_{t+1}.detach()) + lambda * SIGReg(Z)

    Note: SIGReg is computed externally via the SIGReg objective plugin.
    This model returns embeddings for both prediction loss and regularization.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 192,
        encoder_depth: int = 6,
        encoder_heads: int = 3,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        action_dim: int = 10,
        sigreg_weight: float = 1.0,
        sigreg_projections: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sigreg_weight = sigreg_weight
        self.sigreg_projections = sigreg_projections

        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        self.predictor = DiTPredictor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            dropout=dropout,
        )

        # Projectors matching HF config (192 -> 2048 -> 192)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
        )
        self.pred_proj = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim),
        )

    def forward(
        self,
        frames: Tensor,
        actions: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Forward pass over a frame sequence.

        Args:
            frames: [B, T, C, H, W] — sequence of observations
            actions: [B, T-1, A] — actions between consecutive frames

        Returns:
            dict with loss, pred_loss, sigreg, pred_embeddings,
            target_embeddings, embeddings, z_std
        """
        B, T, C, H, W = frames.shape

        # Encode all frames
        flat_frames = frames.reshape(B * T, C, H, W)
        z_all = self.encoder(flat_frames).reshape(B, T, -1)  # [B, T, D]

        # Predict next-frame embeddings for each transition
        # LE-WM key: encoder outputs ARE the targets (stop-gradient)
        total_pred_loss = torch.tensor(0.0, device=frames.device)
        num_transitions = T - 1

        pred_embeddings_list = []
        for t in range(num_transitions):
            # Predictor gets detached current embedding + action
            z_pred = self.predictor(z_all[:, t].detach(), actions[:, t])
            # Target is next-step embedding (detached — no gradient through target)
            target = z_all[:, t + 1].detach()
            pred_loss = F.mse_loss(z_pred, target)
            total_pred_loss = total_pred_loss + pred_loss
            pred_embeddings_list.append(z_pred)

        total_pred_loss = total_pred_loss / max(num_transitions, 1)

        # SIGReg: compute inline (simple version) or let external objective handle it
        # We compute a simple variance-based regularization here as fallback;
        # the full SIGReg objective plugin provides the proper Cramér-Wold version
        z_flat = z_all.reshape(-1, self.embed_dim)  # [B*T, D]
        z_std = z_flat.std(dim=0)  # [D]

        # Simple inline SIGReg approximation: encourage unit variance per dimension
        # The proper SIGReg objective (if composed via CompositeObjective) will override
        sigreg_approx = (1.0 - z_std).pow(2).mean()

        loss = total_pred_loss + self.sigreg_weight * sigreg_approx

        return {
            "loss": loss,
            "pred_loss": total_pred_loss,
            "sigreg": sigreg_approx,
            # Keys for external objectives (CompositeObjective with SIGReg + MSE)
            "pred_embeddings": torch.stack(pred_embeddings_list, dim=1) if pred_embeddings_list else z_all[:, 1:],
            "target_embeddings": z_all[:, 1:].detach(),
            "embeddings": z_flat,  # For SIGReg objective
            "z_std": z_std,  # For JEPAObjective compatibility
        }

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        return self.forward(**batch)

    def encode(self, frames: Tensor) -> Tensor:
        """Encode frames to CLS token embeddings (for inference/planning)."""
        if frames.dim() == 4:
            return self.encoder(frames)
        B, T, C, H, W = frames.shape
        return self.encoder(frames.reshape(B * T, C, H, W)).reshape(B, T, -1)

    def predict_next(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict next embedding from current embedding + action (for planning)."""
        return self.predictor(z, action)

    def param_groups(self) -> list[dict[str, Any]]:
        """Separate encoder and predictor for different learning rates."""
        return [
            {"params": list(self.encoder.parameters()), "name": "encoder"},
            {"params": list(self.predictor.parameters()), "name": "predictor"},
        ]

    @classmethod
    def modality(cls) -> str:
        return "world_model"


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------

def _build_lewm(args: Any) -> LeWMModel:
    """Build LE-WM from Crucible args namespace or env vars."""
    model = LeWMModel(
        image_size=int(getattr(args, "image_size", os.environ.get("IMAGE_SIZE", "224"))),
        patch_size=int(getattr(args, "patch_size", os.environ.get("PATCH_SIZE", "14"))),
        in_channels=int(getattr(args, "image_channels", 3)),
        embed_dim=int(getattr(args, "model_dim", os.environ.get("MODEL_DIM", "192"))),
        encoder_depth=int(getattr(args, "encoder_depth", os.environ.get("ENCODER_DEPTH", "6"))),
        encoder_heads=int(getattr(args, "encoder_heads", os.environ.get("ENCODER_HEADS", "3"))),
        predictor_depth=int(getattr(args, "predictor_depth", os.environ.get("PREDICTOR_DEPTH", "6"))),
        predictor_heads=int(getattr(args, "predictor_heads", os.environ.get("PREDICTOR_HEADS", "16"))),
        predictor_dim_head=int(getattr(args, "predictor_dim_head", os.environ.get("PREDICTOR_DIM_HEAD", "64"))),
        predictor_mlp_dim=int(getattr(args, "predictor_mlp_dim", os.environ.get("PREDICTOR_MLP_DIM", "2048"))),
        action_dim=int(getattr(args, "action_dim", os.environ.get("ACTION_DIM", "10"))),
        sigreg_weight=float(getattr(args, "sigreg_weight", os.environ.get("SIGREG_WEIGHT", "1.0"))),
        sigreg_projections=int(getattr(args, "sigreg_projections", os.environ.get("SIGREG_PROJECTIONS", "128"))),
        dropout=float(getattr(args, "dropout", os.environ.get("DROPOUT", "0.1"))),
    )

    # Load pretrained weights if specified
    pretrained = os.environ.get("PRETRAINED_WEIGHTS", "")
    if pretrained and os.path.exists(pretrained):
        import sys
        weights = torch.load(pretrained, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            print(f"lewm: {len(missing)} missing keys (fine-tuning from partial checkpoint)", file=sys.stderr)
        if unexpected:
            print(f"lewm: {len(unexpected)} unexpected keys ignored", file=sys.stderr)

    return model


register_model("lewm", _build_lewm, source="local")
