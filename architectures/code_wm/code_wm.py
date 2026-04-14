"""Code Intelligence World Model -- predicts code state transitions.

Applies the LE-WM (Looped Efficient World Model) approach to code
intelligence.  A byte-level token encoder processes source code through
weight-shared transformer loops, projecting it to a dense latent space.
A small MLP encodes edit actions.  The LoopedPredictor from WorldModelBase
predicts the next code state embedding from the current state + action.

The key insight: a 4-layer encoder with 8 loops gives effective 32-layer
depth at 4-layer parameter cost -- the same principle as LE-WM for video,
but applied to code token sequences.

State encoder pipeline:
    token IDs -> nn.Embedding -> PositionalEncoding -> LoopedTransformerBlock x encoder_loops
    -> MeanPooling -> LayerNorm -> [B, model_dim]

Action encoder:
    [B, action_dim] -> Linear -> GELU -> Linear -> [B, model_dim]

The data adapter (code_state) returns single transitions::

    states:      [B, seq_len]   (long)  -- before tokens
    actions:     [B, action_dim] (float) -- edit action
    next_states: [B, seq_len]   (long)  -- after tokens

This model's ``training_step`` reshapes those into the temporal format
the base class expects: states=[B, 2, seq_len], actions=[B, 1, action_dim].

Model sizes (approximate):
    Smoke:  MODEL_DIM=64,  ENCODER_LOOPS=4, HEADS=4  -> ~500K params
    Small:  MODEL_DIM=128, ENCODER_LOOPS=6, HEADS=4  -> ~2M params
    Medium: MODEL_DIM=256, ENCODER_LOOPS=8, HEADS=8  -> ~8M params

Env vars (in addition to WM_* from WorldModelBase):
    WM_VOCAB_SIZE      Byte-level vocabulary size (default: 260)
    WM_MAX_SEQ_LEN     Maximum sequence length for positional encoding (default: 256)
    WM_ENCODER_LOOPS   Number of loops for the state encoder (default: 4)
"""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from crucible.models.registry import register_model
except ImportError:
    register_model = None  # Standalone mode (no full Crucible install)

# ---------------------------------------------------------------------------
# Import shared components from wm_base.
#
# wm_base lives in ITS OWN plugin directory (architectures/wm_base/) in the
# tap source, but when wm_base is installed via `crucible tap install wm_base
# --type architectures` it lands as a flat file at
# ~/.crucible-hub/plugins/architectures/wm_base.py — no nested subdir. To
# support both the tap-source layout and every crucible tap install layout,
# try a list of candidate paths.
# ---------------------------------------------------------------------------
def _discover_wm_base_path() -> Path:
    here = Path(__file__).parent
    candidates = [
        # 1. Tap source layout: architectures/code_wm/code_wm.py imports
        #    architectures/wm_base/wm_base.py as a "sibling plugin".
        here.parent / "wm_base" / "wm_base.py",
        # 2. Flat-install layout: tap install copies both plugins to
        #    ~/.crucible-hub/plugins/architectures/{code_wm,wm_base}.py.
        here / "wm_base.py",
        # 3. Bundle-install layout: wm_base installed as a directory bundle
        #    alongside code_wm.py.
        here / "wm_base" / "wm_base.py",
        # 4. Same layout as (3) but from bundle-installed code_wm (the
        #    code_wm.py file lives INSIDE its own directory).
        here.parent / "wm_base.py",
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Fall back to the historic location so the error message points
    # somewhere obvious.
    return candidates[0]

_wm_base_path = _discover_wm_base_path()
_spec = importlib.util.spec_from_file_location("wm_base", _wm_base_path)
if _spec is None or _spec.loader is None:
    raise ImportError(
        f"Cannot load wm_base. Tried: "
        + ", ".join(str(p) for p in [
            Path(__file__).parent.parent / "wm_base" / "wm_base.py",
            Path(__file__).parent / "wm_base.py",
            Path(__file__).parent / "wm_base" / "wm_base.py",
        ])
    )
_wm_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wm_base)

WorldModelBase = _wm_base.WorldModelBase
LoopedTransformerBlock = _wm_base.LoopedTransformerBlock
wm_base_kwargs_from_env = _wm_base.wm_base_kwargs_from_env


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding.

    Precomputes a [1, max_len, dim] buffer of sin/cos position embeddings
    and adds them to the input.  No learnable parameters.
    """

    def __init__(self, dim: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to x: [B, S, D] -> [B, S, D]."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# State Encoder -- byte-level code tokens to latent embedding
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learned attention readout: query a learnable vector against sequence.

    A single learnable query vector attends to all positions, producing a
    weighted sum that focuses on the most informative tokens. Much better
    than mean pooling for code edits where ~1% of tokens change.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=1, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, S, D] -> [B, D]"""
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        out, _ = self.attn(q, x, x, need_weights=False)  # [B, 1, D]
        return out.squeeze(1)  # [B, D]


class CodeStateEncoder(nn.Module):
    """Encode AST token IDs to a dense latent vector.

    Pipeline: Embedding -> PositionalEncoding -> CLS token prepend
              -> LoopedTransformerBlock x loops -> Readout -> LayerNorm

    Readout mode (WM_POOL_MODE env var):
      - "cls":  Extract CLS token (position 0) after transformer
      - "attn": Learned attention pooling over all positions (default)
      - "mean": Mean pooling (legacy, loses edit signal)
    """

    def __init__(
        self,
        vocab_size: int = 260,
        model_dim: int = 128,
        max_seq_len: int = 256,
        encoder_loops: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_enc = PositionalEncoding(model_dim, max_len=max_seq_len + 1, dropout=dropout)

        # Learnable CLS token (prepended to sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Single transformer block, iterated encoder_loops times (weight-shared)
        self.block = LoopedTransformerBlock(
            dim=model_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.encoder_loops = encoder_loops

        # Readout options
        self.pool_mode = os.environ.get("WM_POOL_MODE", "attn")
        self.attn_pool = AttentionPooling(model_dim) if self.pool_mode == "attn" else None

        self.norm = nn.LayerNorm(model_dim)

    def _readout(self, h: Tensor) -> Tensor:
        """Apply pooling + norm to sequence tensor. h: [B, S+1, D] -> [B, D]."""
        if self.pool_mode == "cls":
            h = h[:, 0]
        elif self.pool_mode == "attn":
            h = self.attn_pool(h)
        else:
            h = h.mean(dim=1)
        return self.norm(h)

    def forward(self, x: Tensor, return_intermediates: bool = False) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Encode token IDs to latent vector.

        Args:
            x: [B, seq_len] long tensor of token IDs
            return_intermediates: if True, also return pooled output
                after each loop iteration (for multi-layer readout).

        Returns:
            [B, model_dim] latent embedding (standard mode), or
            tuple of (final_embedding, [loop_1_emb, ..., loop_N_emb])
        """
        B = x.shape[0]

        # Token embedding + CLS prepend
        h = self.embedding(x)                              # [B, S, D]
        cls = self.cls_token.expand(B, -1, -1)              # [B, 1, D]
        h = torch.cat([cls, h], dim=1)                      # [B, S+1, D]
        h = self.pos_enc(h)                                 # [B, S+1, D]

        intermediates = [] if return_intermediates else None

        # Looped transformer passes (weight-shared)
        for _ in range(self.encoder_loops):
            h = self.block(h)                               # [B, S+1, D]
            if return_intermediates:
                intermediates.append(self._readout(h))

        final = self._readout(h)                            # [B, D]

        if return_intermediates:
            return final, intermediates
        return final


# ---------------------------------------------------------------------------
# Action Encoder -- edit action vector to latent embedding
# ---------------------------------------------------------------------------

class CodeActionEncoder(nn.Module):
    """Encode edit action vector to latent space.

    Simple 2-layer MLP: Linear -> GELU -> Linear.
    Input: [B, action_dim] (edit_type_onehot[3] + line_offset[1])
    Output: [B, model_dim]
    """

    def __init__(self, action_dim: int = 4, model_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode action vector: [B, action_dim] -> [B, model_dim]."""
        return self.net(x)


# ---------------------------------------------------------------------------
# CodeWorldModel
# ---------------------------------------------------------------------------

class CodeWorldModel(WorldModelBase):
    """Code Intelligence World Model.

    Predicts code state transitions in latent space using JEPA-style
    self-supervised learning.  The state encoder converts byte-level
    token sequences to dense embeddings via a weight-shared (looped)
    transformer.  The action encoder maps edit actions (type + offset)
    to the same latent space.  The looped predictor (from WorldModelBase)
    predicts the next state embedding.

    This model overrides ``training_step`` to bridge the single-transition
    format from the code_state data adapter into the temporal sequence
    format expected by WorldModelBase.forward().
    """

    def __init__(
        self,
        vocab_size: int = 260,
        max_seq_len: int = 256,
        encoder_loops: int = 4,
        # WorldModelBase params forwarded via **kwargs
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.encoder_loops = encoder_loops

        # Build domain-specific encoders
        state_encoder = self.build_state_encoder(None)
        action_encoder = self.build_action_encoder(None)

        # Initialize shared components (predictor, EMA target)
        self._build_common(state_encoder, action_encoder)

    def build_state_encoder(self, args: Any) -> nn.Module:
        """Build byte-level code state encoder with looped transformer."""
        return CodeStateEncoder(
            vocab_size=self.vocab_size,
            model_dim=self.model_dim,
            max_seq_len=self.max_seq_len,
            encoder_loops=self.encoder_loops,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout_rate,
        )

    def build_action_encoder(self, args: Any) -> nn.Module:
        """Build edit action encoder (2-layer MLP)."""
        return CodeActionEncoder(
            action_dim=self.action_dim,
            model_dim=self.model_dim,
        )

    def training_step(self, **batch: Any) -> dict[str, Tensor]:
        """Bridge single-transition batches to temporal sequence format.

        The code_state data adapter returns::

            states:      [B, seq_len]    -- before tokens
            actions:     [B, action_dim] -- edit action
            next_states: [B, seq_len]    -- after tokens

        WorldModelBase.forward() expects::

            states:  [B, T, seq_len]    -- T timesteps
            actions: [B, T-1, action_dim] -- transitions between timesteps

        We stack (states, next_states) into T=2 and unsqueeze actions to T-1=1.
        """
        states = batch["states"]           # [B, seq_len]
        next_states = batch["next_states"] # [B, seq_len]
        actions = batch["actions"]         # [B, action_dim]

        # Reshape to temporal format: T=2 states, T-1=1 actions
        combined_states = torch.stack([states, next_states], dim=1)  # [B, 2, seq_len]
        combined_actions = actions.unsqueeze(1)                       # [B, 1, action_dim]

        return self.forward(states=combined_states, actions=combined_actions)

    def validation_step(self, **batch: Any) -> dict[str, Tensor]:
        """Same reshaping as training_step for validation batches."""
        return self.training_step(**batch)

    def metric_names(self) -> list[str]:
        return ["pred_loss", "var_reg"]


# ---------------------------------------------------------------------------
# Factory + Registration
# ---------------------------------------------------------------------------

def _build_code_wm(args: Any) -> CodeWorldModel:
    """Factory function for the model registry.

    Reads configuration from env vars and/or the args namespace.
    """
    kwargs = wm_base_kwargs_from_env(args)

    # Code-specific params
    def _get(attr: str, env_var: str, default: str) -> str:
        if args is not None:
            val = getattr(args, attr, None)
            if val is not None:
                return str(val)
        return os.environ.get(env_var, default)

    kwargs["vocab_size"] = int(_get("vocab_size", "WM_VOCAB_SIZE", "662"))
    kwargs["max_seq_len"] = int(_get("max_seq_len", "WM_MAX_SEQ_LEN", "512"))
    kwargs["encoder_loops"] = int(_get("encoder_loops", "WM_ENCODER_LOOPS", "4"))

    # Override action_dim default from wm_base (4) with code-specific default (7)
    kwargs["action_dim"] = int(_get("action_dim", "ACTION_DIM", "7"))

    return CodeWorldModel(**kwargs)


if register_model is not None:
    register_model("code_wm", _build_code_wm, source="local")
