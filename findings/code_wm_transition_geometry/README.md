# Finding: Code world models need TWO geometries

**Claim**: A useful code world model needs both a structured latent **state** space
(for retrieval/classification) and a structured latent **transition** field (for
prediction/rollouts). A single regularizer cannot serve both — but one specific
hybrid recipe (G8: SIGReg + delta-direction) hits both targets.

**Date validated**: 2026-04-05
**Contributors**: eren23 + Claude
**Reproducibility**: see `reproducible_run.md`

---

## TL;DR

| Recipe | Use case | Evidence |
|--------|----------|----------|
| **VICReg / SIGReg alone** | State geometry — great retrieval/classification, **broken predictor** | edit_type 95%, heldout_pred_cos 0.06 |
| **Delta-direction alone** | Transition geometry — working predictor, weaker retrieval | heldout_pred_cos 0.9995, retrieval lift 0.254 |
| **G8 = SIGReg(0.01) + delta-dir(0.5)** | Both — balanced hybrid | edit_type 92.5%, heldout_pred_cos 0.975 |

The predictor-breaking behavior of VICReg-style regularizers is fundamental, not
a hyperparameter artifact: we tested VICReg at w=0.1 and w=0.01, both broke the
predictor. Only SIGReg's sketched/CLT formulation is gentle enough to pair with
a delta objective.

## Why this matters

Most world-model research regularizes the **state** space (VICReg, BYOL, SimCLR).
For **code editing**, that's the wrong target. Editing is fundamentally about
*how states transition*. A delta-space objective (predict the direction and
magnitude of state change) gives the predictor a functional rollout.

But transition-space regularization alone flattens the encoder. Combining:
- **Light state regularization** (SIGReg, w=0.01) to prevent collapse
- **Moderate transition regularization** (delta-direction, w=0.5) for functional predictor

...is what works. This is the **G8 recipe**.

## Evidence

### 11-run ablation table (56K CommitPackFT, 5K steps, 128d)

See `ablation_results.md` for the full matrix. Headline numbers:

| Metric | VICReg-heavy (G1.b) | delta-dir (G1.d) | **G8 (hybrid)** |
|--------|:-------------------:|:----------------:|:---------------:|
| edit_type classification | **95.0%** | 90.3% | 92.5% |
| heldout_pred_cos | 0.062 ❌ | **0.9995** ✓ | **0.975** ✓ |
| heldout_delta_cos | 0.137 | 0.234 | **0.397** |
| retrieval lift over baseline | 0.285 | 0.254 | 0.255 |

G8 wins neither category outright but is the only config that is **functional**
across both: >90% classification AND working predictor.

### Weight sweep (500K CommitPack, 2026-04-05)

Tested alternatives to G8's (reg=SIGReg, w_sig=0.01, w_dir=0.5):

| Config | Verdict | Why |
|--------|---------|-----|
| VICReg(0.01) + dir(0.5) | ❌ Predictor broken (lift=-0.04) | VICReg formulation fundamentally incompatible |
| SIGReg(0.01) + dir(**1.0**) | ⚠️ Works but grad spikes | Heavy transition weight = instability |
| SIGReg(0.01) + dir(**0.25**) | ❌ Magnitude explodes (dratio=21x) | Light weight fails to constrain |
| **G8: SIGReg(0.01) + dir(0.5)** | ✅ Stable, balanced | Sweet spot |

### Data scaling (56K → 500K)

G8 at 500K peaked at **dcos=0.82** vs 0.40 at 56K — data scaling materially helps
the transition geometry. Capacity scaling (128d → 192d) tested next.

## Reproducibility

```bash
# Train G8 at 500K scale (requires crucible-community-tap + WandB)
WM_POOL_MODE=cls WM_REG_MODE=sigreg WM_SIGREG_WEIGHT=0.01 \
WM_LAMBDA_PRED=1.0 WM_LAMBDA_DIR=0.5 WM_LAMBDA_MAG=0.0 WM_LAMBDA_COV=0.0 \
WM_MODEL_DIM=128 WM_NUM_LOOPS=6 WM_NUM_HEADS=4 WM_ENCODER_LOOPS=6 \
WM_STEPS=15000 WM_BATCH_SIZE=64 WM_LR=1e-4 WM_WARMUP_STEPS=500 \
python launchers/code_wm/train_code_wm.py

# Evaluate
python evaluation/semantic_eval.py --checkpoint checkpoints/code_wm_final.pt \
    --data data/commitpack_python_500k.h5 --num-samples 2000
```

Full-run data (HuggingFace Hub): `eren23/code-wm-commitpackft-ast` (56K)
and `eren23/code-wm-commitpack-500k` (500K, WIP).

## Limitations

- **Cross-repo generalization untested**: all evals on CommitPack splits
- **Single-step transitions only**: multi-step rollout stability not yet measured
- **Single seed (42)** for most runs — multi-seed variance not characterized
- **Python-only**: trained on Python AST, no transfer experiments to other languages
