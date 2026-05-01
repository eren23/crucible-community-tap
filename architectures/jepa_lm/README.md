# jepa_lm

JEPA-on-LM hybrid for the [openai/parameter-golf](https://github.com/openai/parameter-golf)
non-record / unlimited-compute track. JEPA appears in that repo's "Requests for PRs"
list; this plugin is a reference implementation for that wishlist item.

## What it does

A standard Crucible `BaselineGPT` (encoder-decoder skip, GQA, tied embeddings,
all the standard parameter-golf augmentations) drives the cross-entropy LM head
and `val_bpb`. On top of that, two auxiliary JEPA paths share a small predictor
MLP:

- **Path A — hidden-state aux JEPA**: predict the model's own final hidden
  state at position `t + chunk` from position `t` (stop-grad target). Loss is
  MSE + VICReg variance regularization. No EMA / target-encoder copy, so total
  parameter count stays under the 16 MB artifact budget.
- **Path B — token-decoder JEPA**: project the predicted embedding through
  the tied LM head and apply CE against the actual token at `t + chunk`. This
  is an LCM-flavored chunk-decoder signal, no extra parameters.

Total loss returned to the trainer:

```
total = ce_main + alpha * (mse_aux + var_weight * vicreg) + beta * ce_jepa
```

## When to use it

- Submitting to the parameter-golf non-record / unlimited-compute leaderboard.
- Studying JEPA-as-regularizer for token-level LM in general.
- Sanity-check: setting `JEPA_ALPHA=0 JEPA_BETA=0` collapses to pure
  `BaselineGPT` (the predictor's output projection is zero-init, so the
  numerics match at step 0 and stay close throughout).

## Env vars

| Var | Default | Notes |
|---|---|---|
| `MODEL_FAMILY` | `jepa_lm` | Selects this plugin. |
| `JEPA_ALPHA` | `0.1` | Weight for hidden-state aux loss. `0` disables path A. |
| `JEPA_BETA` | `0.05` | Weight for token-decoder loss. `0` disables path B. |
| `JEPA_VAR_WEIGHT` | `0.1` | VICReg variance-reg weight inside path A. |
| `JEPA_CHUNK` | `8` | Lookahead distance (positions) for both paths. |
| `JEPA_PREDICTOR_DIM` | `64` | Bottleneck dim of the predictor MLP. |

All standard `BaselineGPT` env vars (`MODEL_DIM`, `NUM_LAYERS`, `NUM_HEADS`,
`SMEAR_GATE`, `BIGRAM_HASH`, `TRIGRAM_HASH`, `ORTHO_INIT`, etc.) are honored.

## Install + run

```bash
crucible tap sync crucible-community-tap
crucible tap install jepa_lm --type architectures

# Sanity smoke (JEPA disabled — should match baseline):
MODEL_FAMILY=jepa_lm JEPA_ALPHA=0 JEPA_BETA=0 \
  PYTHONPATH=src python -m crucible.cli.main run experiment --preset smoke

# Real smoke (both JEPA paths on):
MODEL_FAMILY=jepa_lm JEPA_ALPHA=0.1 JEPA_BETA=0.05 JEPA_CHUNK=8 \
  PYTHONPATH=src python -m crucible.cli.main run experiment --preset smoke
```

## Notes

- Validation only computes the cross-entropy head (`val_bpb` is unchanged by
  JEPA paths). The aux losses are training-time regularization.
- Training metrics: `loss`, `ce_loss`, `jepa_mse`, `jepa_vicreg`, `jepa_token_ce`.
- The predictor is ~`2 * model_dim * predictor_dim` params (≈33 K at
  `MODEL_DIM=256`, `JEPA_PREDICTOR_DIM=64`) — negligible vs the 16 MB budget.
