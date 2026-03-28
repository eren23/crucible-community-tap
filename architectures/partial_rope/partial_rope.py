"""SOTA-inspired architecture with partial RoPE.

Applies rotary positional embeddings to only the first `rope_dims` dimensions
of each attention head, leaving the rest position-independent. This technique
(from top parameter-golf submissions) allows the model to learn both
position-dependent and position-independent features within each head.

Also includes: encoder-decoder skip connections, GQA, gated residuals,
SmearGate, BigramHash, TrigramHash, orthogonal init.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.linear import CastedLinear
from crucible.models.components.norm import RMSNorm
from crucible.models.components.rotary import Rotary, apply_rotary_emb
from crucible.models.components.mlp import MLP
from crucible.models.components.gate import SmearGate
from crucible.models.components.hash_embed import BigramHash, TrigramHash


class PartialRoPEAttention(nn.Module):
    """CausalSelfAttention with RoPE applied to only first `rope_dims` of each head."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 16,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.rope_dims = min(rope_dims, self.head_dim)

        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        # Rotary only for rope_dims, not full head_dim
        self.rotary = Rotary(self.rope_dims, base=rope_base)

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Partial RoPE: only apply to first rope_dims of each head
        rd = self.rope_dims
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q_rope = apply_rotary_emb(q[..., :rd], cos, sin)
        k_rope = apply_rotary_emb(k[..., :rd], cos, sin)
        q = torch.cat([q_rope, q[..., rd:]], dim=-1)
        k = torch.cat([k_rope, k[..., rd:]], dim=-1)

        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        use_gqa = self.num_kv_heads != self.num_heads
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=use_gqa)
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class PartialRoPEBlock(nn.Module):
    """Transformer block using PartialRoPEAttention with gated residuals."""

    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, rope_dims=16, activation="relu_sq"):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = PartialRoPEAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims)
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.delta_gate = nn.Parameter(torch.full((dim,), 2.0, dtype=torch.float32))

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x_in = x
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n, qd, vd)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        # Gated residual
        x = x_in + torch.sigmoid(self.delta_gate).to(dtype=x.dtype)[None, None, :] * (x - x_in)
        return x


class SotaPartialRopeGPT(TiedEmbeddingLM):
    """SOTA architecture: encoder-decoder skip + partial RoPE + all augmentations."""

    def __init__(
        self,
        vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
        mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
        rope_base, qk_gain_init,
        rope_dims=16,
        activation="relu_sq",
        use_smear_gate=False,
        use_bigram_hash=False,
        bigram_hash_buckets=4096,
        bigram_hash_embed_dim=128,
        use_trigram_hash=False,
        trigram_hash_buckets=4096,
        ortho_init=False,
        embed_bottleneck_dim=0,
        spectral_embed_init=False,
    ):
        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std,
                         logit_softcap, embed_bottleneck_dim, spectral_embed_init)
        # Encoder-decoder skip structure
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(
            torch.ones(min(self.num_encoder_layers, self.num_decoder_layers),
                       model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            PartialRoPEBlock(model_dim, num_heads, num_kv_heads, mlp_mult,
                             rope_base, qk_gain_init, rope_dims, activation)
            for _ in range(num_layers)
        ])
        # Augmentations
        self.smear_gate_mod = SmearGate(model_dim) if use_smear_gate else None
        self.bigram_hash_mod = BigramHash(
            vocab_size, num_buckets=bigram_hash_buckets,
            embed_dim=bigram_hash_embed_dim, model_dim=model_dim
        ) if use_bigram_hash else None
        self.trigram_hash_mod = TrigramHash(
            vocab_size, num_buckets=trigram_hash_buckets,
            embed_dim=bigram_hash_embed_dim, model_dim=model_dim
        ) if use_trigram_hash else None
        if ortho_init:
            self._apply_ortho_init(num_layers)

    def _apply_ortho_init(self, num_layers):
        skip_patterns = ('tok_emb', 'embed_low', 'embed_proj', 'lm_head',
                         'bigram_hash', 'trigram_hash', 'smear_gate')
        for name, p in self.named_parameters():
            if p.ndim == 2 and p.numel() > 256 and not any(pat in name for pat in skip_patterns):
                nn.init.orthogonal_(p)
                if 'proj' in name:
                    p.data *= 1.0 / math.sqrt(2 * num_layers)

    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:
        x = self.embed_tokens(input_ids)
        if self.smear_gate_mod is not None:
            x = self.smear_gate_mod(x)
        if self.bigram_hash_mod is not None:
            prev_ids = torch.cat([input_ids[:, :1], input_ids[:, :-1]], dim=1)
            x = x + self.bigram_hash_mod(prev_ids, input_ids)
        if self.trigram_hash_mod is not None:
            x = x + self.trigram_hash_mod(input_ids)
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[i](x, x0, qd, vd)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = self.blocks[bi](x, x0, qd, vd)
        return x


def _build_sota_partial_rope(args: Any) -> SotaPartialRopeGPT:
    return SotaPartialRopeGPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        rope_dims=getattr(args, 'rope_dims', 16),
        activation=getattr(args, 'activation', 'relu_sq'),
        use_smear_gate=getattr(args, 'smear_gate', False),
        use_bigram_hash=getattr(args, 'bigram_hash', False),
        bigram_hash_buckets=getattr(args, 'bigram_hash_buckets', 4096),
        bigram_hash_embed_dim=getattr(args, 'bigram_hash_embed_dim', 128),
        use_trigram_hash=getattr(args, 'trigram_hash', False),
        trigram_hash_buckets=getattr(args, 'trigram_hash_buckets', 4096),
        ortho_init=getattr(args, 'ortho_init', False),
        embed_bottleneck_dim=getattr(args, 'embed_bottleneck_dim', 0),
        spectral_embed_init=getattr(args, 'spectral_embed_init', False),
    )


register_model("sota_partial_rope", _build_sota_partial_rope)
