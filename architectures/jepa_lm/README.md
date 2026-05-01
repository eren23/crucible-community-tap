# jepa_lm

JEPA-on-LM hybrid for the [openai/parameter-golf](https://github.com/openai/parameter-golf)
non-record / unlimited-compute track. JEPA appears in that repo's "Requests for PRs"
list; this plugin is a reference implementation for that wishlist item.

## What's new in v0.2 (2026-05-01)

Informed by community PRs (#832, #896, #1330, #1556, #1581):

- **Defaults dropped 40x** — `JEPA_ALPHA=0.005, JEPA_BETA=0.005` (was 0.2 / 0.05).
  Successful JEPA submissions in parameter-golf use λ ≈ 0.001-0.005, not 0.1+.
  PR #832: "JEPA contributes ~0.1% of peak gradient signal."
- **Off-diagonal covariance penalty** (V-JEPA style) — opt-in via
  `JEPA_COVAR_WEIGHT > 0`. PR #1581 finding: pure variance regularization
  doesn't fully prevent low-rank predictor collapse.
- **Predictor injection mode** — `JEPA_INJECTION=1` projects the predicted
  latent through a zero-init linear and adds it to the hidden stream at
  chunk-aligned positions. JEPA becomes a feature, not just a side loss.
  PR #832 winner pattern (val_bpb 1.1903, beats baseline 1.2244 by 0.034).

## Architecture

A standard parameter-golf `BaselineGPT` (encoder-decoder skip, GQA, tied
embeddings, augmentations) drives the cross-entropy LM head and `val_bpb`.
On top, JEPA paths share a small predictor MLP:

- **Path A — hidden-state aux JEPA**: predict the model's own final hidden
  state at position `t + chunk` from position `t` (stop-grad target). Loss =
  `MSE(pred_n, target_n) + var_w * VICReg + covar_w * cov_off_diag`. Predictor
  output and target both pass through `final_norm` so MSE stays sane.
- **Path B — token-decoder JEPA**: project predicted embedding through the
  tied LM head, CE against the actual token at `t + chunk`.
- **Injection (optional)**: zero-init linear projects predictor output back
  to model_dim, ADDED to the hidden stream at positions `chunk..T-1`. The
  CE main loss now sees the JEPA contribution directly. Validation honors
  injection so `val_bpb` reflects the live model.

```
total = ce_main + alpha * (mse_aux + var_w * vicreg + covar_w * covar) + beta * ce_jepa
```

## When to use it

- Submitting to the parameter-golf non-record / unlimited-compute leaderboard.
- Studying JEPA-as-regularizer / JEPA-as-feature for token-level LM.
- Sanity-check: `JEPA_ALPHA=0 JEPA_BETA=0 JEPA_INJECTION=0` collapses to pure
  `BaselineGPT` numerics (predictor and inject_proj are zero-init).

## Env vars

| Var | Default | Notes |
|---|---|---|
| `MODEL_FAMILY` | `jepa_lm` | Selects this plugin. |
| `JEPA_ALPHA` | `0.005` | Weight for hidden-state aux loss. `0` disables path A. |
| `JEPA_BETA` | `0.005` | Weight for token-decoder loss. `0` disables path B. |
| `JEPA_VAR_WEIGHT` | `0.1` | VICReg variance-reg weight inside path A. |
| `JEPA_COVAR_WEIGHT` | `0.0` | V-JEPA off-diagonal covariance penalty. `0` = off. |
| `JEPA_CHUNK` | `8` | Lookahead distance (positions) for both paths. |
| `JEPA_PREDICTOR_DIM` | `64` | Bottleneck dim of the predictor MLP. |
| `JEPA_INJECTION` | `0` | `1` = inject predicted latent into hidden stream. |

All standard `BaselineGPT` env vars (`MODEL_DIM`, `NUM_LAYERS`, `NUM_HEADS`,
`SMEAR_GATE`, `BIGRAM_HASH`, `TRIGRAM_HASH`, `ORTHO_INIT`, etc.) are honored.

## Install + run

```bash
crucible tap sync crucible-community-tap
crucible tap install jepa_lm --type architectures

# v2 default config (small alpha, no injection, regularizer mode):
MODEL_FAMILY=jepa_lm \
  PYTHONPATH=src python -m crucible.cli.main run experiment --preset smoke

# v2 with injection (PR #832 winner pattern):
MODEL_FAMILY=jepa_lm JEPA_INJECTION=1 JEPA_ALPHA=0.005 JEPA_BETA=0 \
  PYTHONPATH=src python -m crucible.cli.main run experiment --preset proxy

# v2 with V-JEPA covariance penalty:
MODEL_FAMILY=jepa_lm JEPA_COVAR_WEIGHT=0.05 JEPA_INJECTION=1 \
  PYTHONPATH=src python -m crucible.cli.main run experiment --preset proxy
```

## Notes

- Validation (`val_bpb`) uses CE main only (no JEPA aux/token losses), but
  HONORS injection because the predicted latent is part of the live model's
  forward pass when `JEPA_INJECTION=1`.
- Training metrics: `loss`, `ce_loss`, `jepa_mse`, `jepa_vicreg`,
  `jepa_covar` (if enabled), `jepa_token_ce` (if enabled).
- Predictor adds ~`2 * model_dim * predictor_dim` params (≈65K at
  `MODEL_DIM=512, JEPA_PREDICTOR_DIM=64`). Inject_proj adds another
  `model_dim^2` params if `JEPA_INJECTION=1` (≈260K at `MODEL_DIM=512`).
  Total still ~0.4-2% over baseline — well under the 16MB compressed budget.

## v3 followups (not yet shipped)

- **Span-masking** (PR #1581): replace target tokens with a learned
  `jepa_mask_emb` in the context-encoder pass. Forces non-trivial prediction
  in causal LMs (PR #1330 collapse analysis). Requires double-forward.
- **Phased α ramp**: pure AR (30%) → AR+JEPA ramp (50%) → pure AR cooldown
  (20%). PR #832 schedule. Needs current-step access in forward.
- **EMA target encoder**: BYOL-style EMA copy of the model as the target. PR
  #896 found no gain over CE at this scale, so deprioritized.
