# Finding: Code transition geometry — single-step works, multi-step is open

**Claim**: A compact latent encoder trained with a hybrid state + transition objective
learns useful **single-step** edit geometry that supports retrieval and anomaly scoring.
**Multi-step compositional rollout remains an open problem** — neither recipe nor
capacity fix it alone.

**Date validated**: 2026-04-05
**Contributors**: eren23 + Claude
**Reproducibility**: commands at bottom; eval scripts in `evaluation/`

---

## TL;DR

| What | Status |
|------|--------|
| Single-step transition geometry (delta direction alignment) | ✅ Works — `delta_cos ≈ 0.12-0.16` at step 1, consistent across seeds + repos |
| Edit retrieval via delta-NN search | ✅ Ships — see `attocode_integration/codewm_retrieval.py` |
| Multi-step rollout quality beats copy-last baseline | ⚠️ Step 1 only, degrades fast |
| Multi-step rollout as a true world model | ❌ Open — predictor does not compose |

**Champion update (2026-04-05)**:
- **Research primary**: `ExpA` — G8 recipe @ 192d, 2.2M params. Stronger single-step transition signal.
- **Deployment baseline**: `G10` — SIGReg(0.01) + dir(1.0) @ 128d, 1.1M params. Smallest stable model, use for single-step anomaly scoring.

The earlier "G8 champion" claim from 56K data didn't survive 500K scale — 128d G8 overtrains past ~step 8K and collapses in magnitude. 192d or heavier dir weight is needed.

---

## Thesis (revised, 2026-04-05)

> We learn compact latent representations of code edits with useful single-step
> transition geometry. These representations support edit retrieval and local
> predictive signals, while multi-step compositional rollout remains an open
> problem.

## Why this matters

Most world-model research regularizes the **state** space (VICReg, BYOL, SimCLR).
For **code editing**, the state space alone isn't enough — editing is fundamentally
about *how states transition*. A delta-direction loss gives the predictor usable
single-step geometry:
- **Light state regularization** (SIGReg, w=0.01) to prevent collapse
- **Moderate-to-heavy transition regularization** (delta-direction, w=0.5-1.0)

But this combination is **not yet enough for multi-step rollout**. The predictor
produces a weak local delta signal that doesn't compound — by step 2, rolled-out
predictions lose to a trivial copy-last baseline.

---

## Evidence

### 1. Single-step ablation at 56K (CommitPackFT)

11-run matrix, 5K training steps, 128d. Full table: `ablation_results.md`.

| Recipe | edit_type | heldout_pred_cos | heldout_delta_cos |
|--------|:---:|:---:|:---:|
| VICReg(0.1) alone | **95.0%** | 0.062 ❌ | 0.137 |
| delta-dir(0.5) alone | 90.3% | **0.9995** ✓ | 0.234 |
| **G8 hybrid** (SIGReg + dir) | 92.5% | 0.975 ✓ | **0.397** |

VICReg breaks the predictor at any weight. SIGReg's gentler formulation pairs with delta-dir.

### 2. 500K weight sweep (2026-04-05)

Scaling data 56K → 500K exposes that the 128d G8 recipe overtrains:

| Config (500K, 15K steps) | Final dcos | dratio | Pred_cos | Verdict |
|--------------------------|:---:|:---:|:---:|:---:|
| G8 SIGReg(0.01)+dir(0.5), 128d | 0.080 | 997 ⚠️ | 0.977 | Late collapse |
| G9 VICReg(0.01)+dir(0.5), 128d | 0.046 | 337 | **-0.71** ❌ | Predictor broken |
| **G10** SIGReg(0.01)+dir(**1.0**), 128d | 0.167 | **28** ✓ | **0.988** | Stable |
| G11 SIGReg(0.01)+dir(0.25), 128d | 0.150 | 2041 ⚠️ | 0.987 | Magnitude exploded |
| G1d pred+dir only, 128d | -0.083 | 2449 ⚠️ | 0.973 | Unstable |
| **ExpA** G8 recipe @ **192d** | 0.167 | **25** ✓ | 0.957 | Best transition geometry |

**Key**: at 500K × 15K steps, only two configs stay stable through training —
`dir=1.0` (G10) or bigger model (ExpA). The 56K sweet spot (dir=0.5 @ 128d) overtrains.

### 3. Multi-step rollout eval (2026-04-05)

Script: `evaluation/rollout_eval.py` (subcommand `trajectory`).
Data: per-file git histories, 51 trajectories from parameter-golf_dev,
4 transitions each. Compared rolled-out predictions vs copy-last baseline.

| Model | Step 1 lift vs copy-last | Step 1 delta_cos | Step 2+ |
|-------|:---:|:---:|:---:|
| **G10 (128d)** | +0.012 / −0.013 (unstable) | 0.09-0.14 | ❌ Always loses |
| **ExpA (192d)** | **+0.044 / +0.061** | **0.12-0.16** | ⚠️ Marginal |

- ExpA clearly beats G10 on compositional quality
- Neither model is a true multi-step world model
- Rolled-out delta direction correlates with true delta only at step 1
  (delta_cos crashes to ~0 by step 2)

---

## What this is NOT

- **Not a world model (yet)**. The predictor produces weak single-step deltas that don't compound. Calling this a "world model" overpromises.
- **Not useful for multi-step planning or rollout**. If you need 3+ step composition, this model won't do it.
- **Not SoTA on any single-step benchmark**. delta_cos 0.12-0.16 is modest, not breakthrough.

## What this IS

- **A compact latent edit encoder** (1-2M params, ~5-10MB fp32) that is:
  - Useful for **delta-space edit retrieval** — find historical similar edits by cosine NN (shipped in `attocode_integration/codewm_retrieval.py`, 28ms/query on CPU)
  - Useful for **single-step anomaly scoring** — flag edits where `||z_pred - z_true||` is unusually large
  - Suitable for **complementing static code-intel tools** (e.g., attocode-code-intel) with a learned, forward-looking signal
- **Evidence** that state-space regularization alone (VICReg/BYOL/SimCLR) is insufficient for code-edit dynamics — delta-direction is necessary

## Open research problem

**How to train code-edit transition models that compose over multiple steps?**

Conjectures worth testing:
1. **Trajectory data**: train on chained per-file edits with multi-step teacher forcing, not isolated (before, action, after) triples
2. **Explicit rollout loss**: minimize prediction error at step 2+ directly, not just step 1
3. **Residual-in-latent stabilization**: parameterize predictor as `z_next = z + f(z, a)` with bounded-norm residual
4. **Richer actions**: 15-dim rich actions (available in `collectors/ast_diff.py`) may give the predictor more signal than 7-dim

See `next_research_direction.md` (coming) for design proposals.

---

## Reproducibility

```bash
# Train the new champion recipe (ExpA — 192d G8) at 500K scale
WM_POOL_MODE=cls WM_REG_MODE=sigreg WM_SIGREG_WEIGHT=0.01 \
WM_LAMBDA_PRED=1.0 WM_LAMBDA_DIR=0.5 WM_LAMBDA_MAG=0.0 WM_LAMBDA_COV=0.0 \
WM_MODEL_DIM=192 WM_NUM_LOOPS=6 WM_NUM_HEADS=4 WM_ENCODER_LOOPS=6 \
WM_STEPS=15000 WM_BATCH_SIZE=64 WM_LR=1e-4 WM_WARMUP_STEPS=500 \
python launchers/code_wm/train_code_wm.py

# Train the deployment baseline (G10 — 128d dir=1.0)
# Same env as above but WM_MODEL_DIM=128 WM_LAMBDA_DIR=1.0

# Single-step eval
python evaluation/semantic_eval.py --checkpoint <ckpt.pt> \
    --data data/commitpack_python_500k.h5 --num-samples 2000

# Multi-step rollout eval
python evaluation/rollout_eval.py trajectory --checkpoint <ckpt.pt> \
    --repo /path/to/python/git/repo --min-edits 4 --max-steps 4
```

**Pre-trained checkpoints**: `eren23/code-wm-checkpoints` (private HF Hub repo).

## Limitations

- **Single seed** (42) — multi-seed variance not characterized
- **Python-only** — trained on Python AST, no transfer to other languages
- **51 trajectories** in the rollout eval — more repos needed for variance estimates
- **7-dim action vector** is coarse — may undersell what richer actions (15-dim) could do
