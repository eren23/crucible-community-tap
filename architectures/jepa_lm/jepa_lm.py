"""JEPA-on-LM architecture for parameter-golf non-record / unlimited-compute track.

A standard parameter-golf BaselineGPT backbone (encoder-decoder skip, GQA,
augmentations, tied embeddings) drives the cross-entropy LM head and val_bpb.
On top of that, two auxiliary JEPA paths share a small predictor MLP:

  Path A — Hidden-state aux JEPA:
      For each non-final position t, predict the model's own final hidden
      state at position t + chunk (stop-grad target). Loss = MSE + VICReg
      variance regularization. No EMA / target-encoder copy — keeps params
      under the 16MB artifact budget.

  Path B — Token-decoder JEPA:
      Project the predicted embedding through the tied LM head and apply CE
      against the actual token at position t + chunk. Provides token-level
      supervision through the JEPA latent (LCM-flavored chunk decoding).

Combined loss returned to the trainer:

      total = ce_main + alpha * (mse_aux + var_weight * vicreg) + beta * ce_jepa

Setting ``JEPA_ALPHA=0`` disables path A. Setting ``JEPA_BETA=0`` disables
path B. Setting both to 0 collapses to pure BaselineGPT (sanity smoke).

Env vars (read in the builder, not via Hyperparameters since this plugin
introduces them):

    JEPA_ALPHA           default 0.1   weight for hidden-state aux loss
    JEPA_BETA            default 0.05  weight for token-decoder loss
    JEPA_VAR_WEIGHT      default 0.1   VICReg variance-reg weight
    JEPA_CHUNK           default 8     positions ahead to predict
    JEPA_PREDICTOR_DIM   default 64    bottleneck dim of predictor MLP

The predictor adds two linear layers (model_dim -> predictor_dim -> model_dim)
zero-initialized on the second layer, so JEPA paths start as a no-op and the
trainer sees pure baseline gradients at step 0.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crucible.models.architectures.baseline import BaselineGPT
from crucible.models.registry import register_model, register_schema


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return default if val is None or val == "" else float(val)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return default if val is None or val == "" else int(val)


class JepaLM(BaselineGPT):
    """BaselineGPT backbone + JEPA aux head + JEPA token-decoder head."""

    def __init__(
        self,
        *,
        jepa_alpha: float = 0.1,
        jepa_beta: float = 0.05,
        jepa_var_weight: float = 0.1,
        jepa_chunk: int = 8,
        jepa_predictor_dim: int = 64,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        if jepa_chunk < 1:
            raise ValueError(f"JEPA_CHUNK must be >= 1, got {jepa_chunk}")
        if jepa_predictor_dim < 1:
            raise ValueError(f"JEPA_PREDICTOR_DIM must be >= 1, got {jepa_predictor_dim}")
        self.jepa_alpha = float(jepa_alpha)
        self.jepa_beta = float(jepa_beta)
        self.jepa_var_weight = float(jepa_var_weight)
        self.jepa_chunk = int(jepa_chunk)
        d = base_kwargs["model_dim"]
        self.jepa_predictor = nn.Sequential(
            nn.Linear(d, jepa_predictor_dim, bias=False),
            nn.GELU(),
            nn.Linear(jepa_predictor_dim, d, bias=False),
        )
        # Zero-init the output projection so JEPA contributes nothing at step 0.
        # This makes the disabled-mode (alpha=beta=0) and enabled-mode losses
        # identical at init, guaranteeing the baseline-equivalence property.
        nn.init.zeros_(self.jepa_predictor[2].weight)
        nn.init.normal_(
            self.jepa_predictor[0].weight,
            std=1.0 / math.sqrt(d),
        )

    def _components(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        lora: Any = None,
    ) -> dict[str, Tensor]:
        """Forward + per-component losses. Returns dict suitable for training_step."""
        h = self.hidden(input_ids, lora=lora)
        ce_main = self.compute_loss(h, target_ids, lora=lora)
        out: dict[str, Tensor] = {"ce_loss": ce_main}

        chunk = self.jepa_chunk
        seq_len = h.size(1)
        do_jepa = (self.jepa_alpha > 0.0 or self.jepa_beta > 0.0) and seq_len > chunk

        if not do_jepa:
            out["loss"] = ce_main
            return out

        h_curr = h[:, :-chunk, :]                    # [B, T-chunk, D]
        h_target = h[:, chunk:, :].detach()          # stop-grad target
        h_pred = self.jepa_predictor(h_curr)         # [B, T-chunk, D]

        total = ce_main

        if self.jepa_alpha > 0.0:
            # Normalize before MSE — un-RMSNormed hidden states grow across
            # layers and produce huge MSE values that drown out CE_main.
            # Match the val_bpb path which applies final_norm before decoding.
            h_pred_n = self.final_norm(h_pred)
            h_target_n = self.final_norm(h_target)
            mse_aux = F.mse_loss(h_pred_n, h_target_n)
            # VICReg-style variance regularization (matches JEPAObjective at
            # crucible.training.objectives.JEPAObjective lines 184-189).
            z_std = torch.sqrt(h_pred_n.float().var(dim=(0, 1)) + 1e-4)
            vicreg = torch.relu(1.0 - z_std).mean()
            total = total + self.jepa_alpha * (mse_aux + self.jepa_var_weight * vicreg)
            out["jepa_mse"] = mse_aux.detach()
            out["jepa_vicreg"] = vicreg.detach()

        if self.jepa_beta > 0.0:
            # Token-decoder JEPA: decode predicted embedding through tied LM head.
            # Target = input_ids[:, chunk:] (the actual token chunk steps ahead).
            target_chunk_ids = input_ids[:, chunk:]              # [B, T-chunk]
            x = self.final_norm(h_pred)
            flat = x.reshape(-1, x.size(-1))
            logits_proj = (
                self.tied_logits(flat) if self.tie_embeddings else self.lm_head(flat)
            )
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            ce_jepa = F.cross_entropy(
                logits.float(),
                target_chunk_ids.reshape(-1),
                reduction="mean",
                ignore_index=-100,
            )
            total = total + self.jepa_beta * ce_jepa
            out["jepa_token_ce"] = ce_jepa.detach()

        out["loss"] = total
        return out

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        lora: Any = None,
    ) -> Tensor:  # type: ignore[override]
        return self._components(input_ids, target_ids, lora=lora)["loss"]

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        return self._components(
            batch["input_ids"],
            batch["target_ids"],
            lora=batch.get("lora"),
        )

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        # Validation reports val_bpb based on ce_loss only — JEPA aux is
        # training-time regularization. Disable JEPA paths here so val
        # numbers stay comparable to the baseline.
        h = self.hidden(batch["input_ids"], lora=batch.get("lora"))
        ce = self.compute_loss(h, batch["target_ids"], lora=batch.get("lora"))
        return {"loss": ce, "ce_loss": ce}


def _build_jepa_lm(args: Any) -> JepaLM:
    base_kwargs = dict(
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
        attention_variant=args.attention_variant,
        residual_variant=args.residual_variant,
        embed_bottleneck_dim=getattr(args, "embed_bottleneck_dim", 0),
        use_smear_gate=getattr(args, "smear_gate", False),
        use_bigram_hash=getattr(args, "bigram_hash", False),
        bigram_hash_buckets=getattr(args, "bigram_hash_buckets", 2048),
        bigram_hash_embed_dim=getattr(args, "bigram_hash_embed_dim", 128),
        ortho_init=getattr(args, "ortho_init", False),
        spectral_embed_init=getattr(args, "spectral_embed_init", False),
        use_conv_block=getattr(args, "conv_block", False),
        conv_kernel=getattr(args, "conv_kernel", 3),
        multiscale_window=getattr(args, "multiscale_window", 0),
        token_merge_layer=getattr(args, "token_merge_layer", 0),
        token_merge_threshold=getattr(args, "token_merge_threshold", 0.9),
        block_pattern=getattr(args, "block_pattern", ""),
        use_trigram_hash=getattr(args, "trigram_hash", False),
        trigram_hash_buckets=getattr(args, "trigram_hash_buckets", 4096),
        activation=getattr(args, "activation", "relu_sq"),
        use_moe=getattr(args, "use_moe", False),
        moe_num_experts=getattr(args, "moe_num_experts", 4),
        moe_top_k=getattr(args, "moe_top_k", 2),
    )
    return JepaLM(
        jepa_alpha=_env_float("JEPA_ALPHA", 0.1),
        jepa_beta=_env_float("JEPA_BETA", 0.05),
        jepa_var_weight=_env_float("JEPA_VAR_WEIGHT", 0.1),
        jepa_chunk=_env_int("JEPA_CHUNK", 8),
        jepa_predictor_dim=_env_int("JEPA_PREDICTOR_DIM", 64),
        **base_kwargs,
    )


register_model("jepa_lm", _build_jepa_lm)
register_schema("jepa_lm", {
    # Inherits all baseline knobs (MODEL_DIM, NUM_LAYERS, ...) — those are
    # honored via the BaselineGPT constructor. Schema below documents the
    # JEPA-specific env vars introduced by this plugin.
    "JEPA_ALPHA": {"type": "float", "default": 0.1, "description": "Weight for hidden-state aux JEPA loss (MSE + VICReg)"},
    "JEPA_BETA": {"type": "float", "default": 0.05, "description": "Weight for token-decoder JEPA cross-entropy loss"},
    "JEPA_VAR_WEIGHT": {"type": "float", "default": 0.1, "description": "VICReg variance-regularization weight inside the aux JEPA loss"},
    "JEPA_CHUNK": {"type": "int", "default": 8, "description": "Lookahead distance (positions) for JEPA prediction"},
    "JEPA_PREDICTOR_DIM": {"type": "int", "default": 64, "description": "Bottleneck dim of the JEPA predictor MLP"},
})
