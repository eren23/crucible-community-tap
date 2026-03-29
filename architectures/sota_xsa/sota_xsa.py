"""SOTA Parameter Golf reproduction: XSA + LeakyReLU(0.5)² + full augmentation stack.

Reproduces the top techniques from the OpenAI Parameter Golf competition:
- Exclusive Self-Attention (XSA) on designated layers (post-attention self-value removal)
- LeakyReLU(0.5)² activation (current SOTA, -0.003 BPB vs ReLU²)
- Encoder-decoder skip connections (U-Net style)
- GQA (8 query heads, 4 KV heads)
- 3x MLP expansion
- Gated residuals
- SmearGate + BigramHash + TrigramHash augmentations
- Orthogonal initialization

Config env vars:
  XSA_LAYERS: comma-separated layer indices for XSA (e.g. "8,9,10")
  ACTIVATION: activation function name (default: leaky05_sq)
  All standard baseline vars: NUM_LAYERS, MODEL_DIM, NUM_HEADS, etc.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any


import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.base import TiedEmbeddingLM
from crucible.models.registry import register_model
from crucible.models.components.linear import CastedLinear
from crucible.models.components.norm import RMSNorm
from crucible.models.components.rotary import Rotary, apply_rotary_emb
from crucible.models.components.mlp import MLP, ACTIVATIONS
from crucible.models.components.gate import SmearGate
from crucible.models.components.hash_embed import BigramHash, TrigramHash

# ---------------------------------------------------------------------------
# Register leaky05_sq activation (SOTA: LeakyReLU(negative_slope=0.5)²)
# Bundled here so it survives pod sync without modifying core mlp.py
# ---------------------------------------------------------------------------
def _leaky05_sq(x: Tensor) -> Tensor:
    return F.leaky_relu(x, 0.5).square()

if "leaky05_sq" not in ACTIVATIONS:
    ACTIVATIONS["leaky05_sq"] = _leaky05_sq


# ---------------------------------------------------------------------------
# XSA Attention: causal mask with self-attention diagonal excluded
# ---------------------------------------------------------------------------
class XSACausalSelfAttention(nn.Module):
    """CausalSelfAttention with optional Exclusive Self-Attention (XSA).

    When use_xsa=True, the attention mask zeros the diagonal so each token
    cannot attend to its own value, forcing reliance on context from other
    tokens. From arXiv:2603.09078, used in all top Parameter Golf submissions.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        use_xsa: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.use_xsa = use_xsa

        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def _xsa_remove_self_value(self, y: Tensor, v: Tensor) -> Tensor:
        """Remove self-value projection from attention output (post-attention XSA).

        Subtracts the component of each output vector that lies along the
        corresponding value vector. GQA-aware via free reshape + broadcast,
        no repeat_interleave needed. All ops are compile-friendly.

        Args:
            y: attention output [B, H, T, D]
            v: value vectors [B, Hkv, T, D]
        """
        B, H, T, D = y.shape
        Hkv = v.size(1)
        group = H // Hkv
        y_g = y.reshape(B, Hkv, group, T, D)
        vn = F.normalize(v, dim=-1).unsqueeze(2)  # [B, Hkv, 1, T, D]
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, H, T, D)

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
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        use_gqa = self.num_kv_heads != self.num_heads

        # Standard Flash Attention — stays on fast path, compile-friendly
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=use_gqa
        )

        # XSA: remove self-value projection AFTER attention (pure tensor ops)
        if self.use_xsa:
            y = self._xsa_remove_self_value(y, v)

        return self.proj(
            y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        )


# ---------------------------------------------------------------------------
# Block with XSA + gated residuals (always gated for SOTA)
# ---------------------------------------------------------------------------
class XSABlock(nn.Module):
    """Transformer block with optional XSA and gated residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        activation: str = "relu_sq",
        use_xsa: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = XSACausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            use_xsa=use_xsa,
        )
        self.mlp = MLP(dim, mlp_mult, activation=activation)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )
        # Gated residual (always on for SOTA reproduction)
        self.delta_gate = nn.Parameter(
            torch.full((dim,), 2.0, dtype=torch.float32)
        )

    def forward(
        self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None,
    ) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x_in = x
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n, qd, vd)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )
        # Gated residual
        x = x_in + torch.sigmoid(self.delta_gate).to(dtype=x.dtype)[None, None, :] * (
            x - x_in
        )
        return x


# ---------------------------------------------------------------------------
# Full SOTA architecture: encoder-decoder skip + XSA + augmentations
# ---------------------------------------------------------------------------
class SotaXSAGPT(TiedEmbeddingLM):
    """SOTA Parameter Golf reproduction architecture.

    Encoder-decoder skip connections (U-Net style) with XSA on designated
    layers, LeakyReLU(0.5)² activation, and the full augmentation stack.
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        xsa_layers: set[int] | None = None,
        activation: str = "relu_sq",
        use_smear_gate: bool = False,
        use_bigram_hash: bool = False,
        bigram_hash_buckets: int = 4096,
        bigram_hash_embed_dim: int = 128,
        use_trigram_hash: bool = False,
        trigram_hash_buckets: int = 4096,
        ortho_init: bool = False,
        embed_bottleneck_dim: int = 0,
        spectral_embed_init: bool = False,
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
                model_dim, dtype=torch.float32,
            )
        )
        xsa_set = xsa_layers or set()
        self.blocks = nn.ModuleList([
            XSABlock(
                model_dim, num_heads, num_kv_heads, mlp_mult,
                rope_base, qk_gain_init, activation,
                use_xsa=(i in xsa_set),
            )
            for i in range(num_layers)
        ])
        # Augmentations
        self.smear_gate_mod = SmearGate(model_dim) if use_smear_gate else None
        self.bigram_hash_mod = (
            BigramHash(
                vocab_size, num_buckets=bigram_hash_buckets,
                embed_dim=bigram_hash_embed_dim, model_dim=model_dim,
            )
            if use_bigram_hash else None
        )
        self.trigram_hash_mod = (
            TrigramHash(
                vocab_size, num_buckets=trigram_hash_buckets,
                embed_dim=bigram_hash_embed_dim, model_dim=model_dim,
            )
            if use_trigram_hash else None
        )
        if ortho_init:
            self._apply_ortho_init(num_layers)

    def _apply_ortho_init(self, num_layers: int) -> None:
        skip_patterns = (
            "tok_emb", "embed_low", "embed_proj", "lm_head",
            "bigram_hash", "trigram_hash", "smear_gate",
        )
        for name, p in self.named_parameters():
            if (
                p.ndim == 2
                and p.numel() > 256
                and not any(pat in name for pat in skip_patterns)
            ):
                nn.init.orthogonal_(p)
                if "proj" in name:
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
        skips: list[Tensor] = []
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


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------
def _parse_xsa_layers(env_val: str) -> set[int]:
    """Parse XSA_LAYERS: '8,9,10' -> {8, 9, 10}"""
    if not env_val:
        return set()
    return {int(x.strip()) for x in env_val.split(",") if x.strip()}


def _build_sota_xsa(args: Any) -> SotaXSAGPT:
    xsa_raw = getattr(args, "xsa_layers", "") or os.environ.get("XSA_LAYERS", "")
    xsa_layers = _parse_xsa_layers(xsa_raw)
    return SotaXSAGPT(
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
        xsa_layers=xsa_layers,
        activation=getattr(args, "activation", "leaky05_sq"),
        use_smear_gate=getattr(args, "smear_gate", False),
        use_bigram_hash=getattr(args, "bigram_hash", False),
        bigram_hash_buckets=getattr(args, "bigram_hash_buckets", 4096),
        bigram_hash_embed_dim=getattr(args, "bigram_hash_embed_dim", 128),
        use_trigram_hash=getattr(args, "trigram_hash", False),
        trigram_hash_buckets=getattr(args, "trigram_hash_buckets", 4096),
        ortho_init=getattr(args, "ortho_init", False),
        embed_bottleneck_dim=getattr(args, "embed_bottleneck_dim", 0),
        spectral_embed_init=getattr(args, "spectral_embed_init", False),
    )


register_model("sota_xsa", _build_sota_xsa)

# ---------------------------------------------------------------------------
# torch.compile(fullgraph=True) fix: the registry loads plugins via
# importlib.util.exec_module() but never registers them in sys.modules.
# torch._dynamo needs to import_module(__name__) during tracing.
# We register a module with our full namespace AFTER all definitions.
# ---------------------------------------------------------------------------
if __name__ not in sys.modules:
    import types as _types
    _m = _types.ModuleType(__name__)
    _m.__file__ = globals().get("__file__", "")
    _m.__dict__.update(globals())
    sys.modules[__name__] = _m
