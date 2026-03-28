"""MoE Baseline: standard transformer with Mixture of Experts replacing MLP.

A baseline-like architecture where every block uses a top-k routed MoE layer
instead of a dense MLP. Retains U-Net skip connections and the standard
attention stack. The forward pass adds MoE auxiliary (load-balancing) losses
to the cross-entropy loss.

Usage in an experiment design:
    MODEL_FAMILY=moe_baseline
    NUM_LAYERS=9
    MODEL_DIM=512
    NUM_HEADS=8
    MOE_NUM_EXPERTS=4
    MOE_TOP_K=2
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.attention import CausalSelfAttention
from crucible.models.components.moe import MoELayer
from crucible.models.components.norm import RMSNorm


class MoEBlock(nn.Module):
    """Transformer block with MoE replacing the dense MLP.

    Structure: RMSNorm -> CausalSelfAttention -> residual,
               RMSNorm -> MoELayer -> residual.
    Supports standard and gated residual variants.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        num_experts: int,
        top_k: int,
        mlp_mult: int,
        activation: str,
        aux_loss_coeff: float,
        attention_variant: str = "standard",
        residual_variant: str = "standard",
    ):
        super().__init__()
        if residual_variant not in {"standard", "gated"}:
            raise ValueError(f"Unsupported RESIDUAL_VARIANT={residual_variant!r}")
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            attention_variant,
        )
        self.moe = MoELayer(
            model_dim,
            num_experts=num_experts,
            top_k=top_k,
            mlp_mult=mlp_mult,
            activation=activation,
            aux_loss_coeff=aux_loss_coeff,
        )
        self.residual_variant = residual_variant
        self.attn_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float()
        )
        if residual_variant == "gated":
            self.delta_gate = nn.Parameter(
                torch.full((model_dim,), 2.0, dtype=torch.float32)
            )

    def forward(self, x: Tensor, v1: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * v1
        x_in = x
        # Attention
        n = self.attn_norm(x)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n)
        # MoE MLP
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.moe(self.mlp_norm(x))
        # Gated residual
        if self.residual_variant == "gated":
            x = x_in + torch.sigmoid(self.delta_gate).to(dtype=x.dtype)[None, None, :] * (x - x_in)
        return x


class MoEBaselineGPT(TiedEmbeddingLM):
    """Baseline transformer with Mixture of Experts in every block.

    Same U-Net skip connection structure as BaselineGPT, but each block
    uses a top-k routed MoE layer instead of a dense MLP.
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        embed_bottleneck_dim: int = 0,
        spectral_embed_init: bool = False,
        attention_variant: str = "standard",
        residual_variant: str = "standard",
        activation: str = "relu_sq",
        num_experts: int = 4,
        top_k: int = 2,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__(
            vocab_size, model_dim, tie_embeddings, tied_embed_init_std,
            logit_softcap, embed_bottleneck_dim, spectral_embed_init,
        )
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(
            torch.ones(
                min(self.num_encoder_layers, self.num_decoder_layers),
                model_dim,
                dtype=torch.float32,
            )
        )
        self.blocks = nn.ModuleList([
            MoEBlock(
                model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                num_experts=num_experts,
                top_k=top_k,
                mlp_mult=mlp_mult,
                activation=activation,
                aux_loss_coeff=aux_loss_coeff,
                attention_variant=attention_variant,
                residual_variant=residual_variant,
            )
            for _ in range(num_layers)
        ])

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            x = self.blocks[bi](x, x0)
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        ce_loss = self.compute_loss(
            self.hidden(input_ids, lora=lora), target_ids, lora=lora,
        )
        # Sum auxiliary load-balancing losses from all MoE layers
        aux = sum(block.moe.aux_loss for block in self.blocks)
        return ce_loss + aux


def _build_moe_baseline(args: Any) -> MoEBaselineGPT:
    return MoEBaselineGPT(
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        embed_bottleneck_dim=args.embed_bottleneck_dim,
        spectral_embed_init=getattr(args, "spectral_embed_init", False),
        attention_variant=args.attention_variant,
        residual_variant=args.residual_variant,
        activation=getattr(args, "activation", "relu_sq"),
        num_experts=getattr(args, "moe_num_experts", 4),
        top_k=getattr(args, "moe_top_k", 2),
    )


register_model("moe_baseline", _build_moe_baseline)
