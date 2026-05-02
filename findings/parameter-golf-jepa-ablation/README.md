# JEPA-on-LM ablation for parameter-golf — comprehensive negative result (2026-05-02)

A 14-run ablation establishing that **JEPA auxiliary objectives do NOT
improve val_bpb on parameter-golf at the standard 17M / sp1024 / FineWeb
scale**, but the cleanest recipe (α=0.001, VAR_WEIGHT=0, MSE-only Path A)
ties baseline exactly. Confirms and refines [PR #896 (Manav Pandey)][pr896]
and [PR #1330 (luciobaiocchi)][pr1330].

[pr896]: https://github.com/openai/parameter-golf/pull/896
[pr1330]: https://github.com/openai/parameter-golf/pull/1330

## TL;DR

- Best JEPA configuration (`var-zero`): `val_bpb = 1.2311` at step 50K — **exact tie with baseline** at the same seed.
- Same-seed JEPA-vs-baseline gap: **+0.0007 to +0.0009** across two seeds (1337, 42).
- Cross-seed baseline variance: **0.0022** > JEPA gap → statistically indistinguishable.
- JEPA aux at λ ≥ 0.005 actively hurts; only the whisper regime (λ ≈ 0.001 with VICReg variance reg disabled) is neutral.

## What we tested

All variants share the same 17,059,912-parameter `BaselineGPT` backbone
(9L, MODEL_DIM=512, MLP_MULT=2, GQA 8q/4kv, tied embeddings, sp1024,
relu_sq activation). JEPA variants add a 65,536-parameter predictor MLP
(model_dim → 64 → model_dim, zero-init on output). Total JEPA model:
17,125,448 params (+0.4%). Same parameter budget across the 14-run
ablation — no model-shape changes.

Architecture lives at [`architectures/jepa_lm/`](../../architectures/jepa_lm/) (tap commit `bc93273`).

Training: parameter-golf `promotion` preset (~50K steps, 7200s wallclock,
65,536 train_batch_tokens, sp1024 FineWeb 10B). Trained on RTX 4090
(jepa-01) and 4090 cards via RunPod.

## Result table — final val_bpb at step 50K

(Sorted by final val_bpb. Star = run did not reach step 50K due to
wallclock cap on slower hardware; final_step column shows actual.)

| run | seed | config | step | **val_bpb** | gap vs baseline-1337 |
|---|---|---|---|---|---|
| `baseline-seed42` | 42 | (control) | 50K | **1.2289** | −0.0022 (seed advantage) |
| `tiny-lambda-seed42` | 42 | α=0.001 | 50K | 1.2298 | −0.0013 |
| **`var-zero`** | 1337 | **α=0.001, VAR=0** | 50K | **1.2311** | **0.0000** ✅ TIE |
| `baseline-promo` | 1337 | (control) | 50K | 1.2311 | (reference) |
| `tiny-lambda-v3` | 1337 | α=0.001 | 50K | 1.2318 | +0.0007 |
| `half-lambda` | 1337 | α=0.0005 | 50K | 1.2318 | +0.0007 |
| `chunk16` | 1337 | α=0.001, CHUNK=16 | 50K | 1.2318 | +0.0007 |
| `aux+token-tiny` | 1337 | α=β=0.001 | 50K | 1.2361 | +0.0050 |
| `tenth-lambda*` | 1337 | α=0.0001 | 40K | 1.2362 | tied at step 40K |
| `covar-v3` | 1337 | α=0.005, COVAR=0.05 | 50K | 1.2374 | +0.0063 |
| `token-only-tiny*` | 1337 | β=0.001 | 40K | 1.2408 | (40K) +0.0046 |
| `injection-v2*` | 1337 | α=0.005, INJECT=1 | 40K | 1.2456 | (40K) +0.0094 |
| `aux-v1` | 1337 | α=0.2 | 50K | 1.2492 | +0.0181 |
| `aux-low-v2*` | 1337 | α=0.005 | 30K | 1.2553 | (30K) +0.0060 |

Per-step val_bpb curves: see [`val_bpb_curves.csv`](./val_bpb_curves.csv).

## Component-by-component verdict at the whisper regime (λ=0.001)

| component active | effect on val_bpb @ 50K |
|---|---|
| Path A MSE alone (VAR_WEIGHT=0) | **0.000** ← exact baseline |
| Path A + VICReg variance reg (VAR_WEIGHT=0.1) | +0.0007 (within seed noise) |
| Path A + V-JEPA off-diag covariance (COVAR=0.05) | +0.0063 |
| Path B (token decoder via tied LM head) alone | +0.0046 |
| Path A + Path B both at whisper | +0.0050 |
| Path A + injection (zero-init latent → hidden stream) | +0.0094 |
| Higher λ: 0.005 | +0.005 to +0.010 |
| Higher λ: 0.2 | +0.018 (catastrophic, v1 default) |

## Why everyone keeps getting "JEPA doesn't help" on parameter-golf

The community has 30+ JEPA submissions, most reporting negative or
neutral. PR #1330 (luciobaiocchi) named the failure mode: "Most 'vanilla'
JEPA implementations produce near-identical negative results... The task
collapses to trivially easy within the first step." Our ablation
reinforces this and adds three sub-conclusions:

1. **λ matters most, by orders of magnitude.** PR #832 used λ=0.001 in a
   byte-level submission that beat baseline by 0.034 BPB. We tested the
   same magnitude on sp1024 and got parity. Going to λ=0.005 already
   costs ≥0.005 BPB. λ=0.2 (our v1 default) costs 0.018 BPB. This is the
   single most consequential knob.

2. **VICReg variance reg adds small harm at this λ.** With λ already at
   the noise floor, the variance hinge `relu(1 - z_std)` injects a tiny
   asymmetric force that nudges JEPA slightly away from baseline. Setting
   `VAR_WEIGHT=0` recovers exact parity.

3. **Path B (token-decoder JEPA) hurts even at β=0.001.** The token CE
   competes with main CE for the LM head, so even whisper magnitudes pull
   the head in two directions. Path A (hidden-state aux MSE) is benign at
   small λ because it doesn't touch the LM head.

## Why this is publishable as parameter-golf non-record

- 14 runs at the same N (17.06M / 17.13M with predictor), promotion-tier
  budget each (~2h wallclock, ~50K steps).
- Two-seed paired baselines (seed=1337, seed=42) establish a 0.0022
  noise floor.
- λ sweep across 4 orders of magnitude (0.0001, 0.0005, 0.001, 0.005, 0.2).
- Path ablation (A only / B only / both / injection / covar).
- `chunk16`, `var-zero`, `tenth-lambda` — three previously untested knobs.

The cleanest negative result on parameter-golf JEPA submitted to date.
PR #896 was a single-config failure; this is a saturated grid.

## Reproducibility

- **Architecture**: [`architectures/jepa_lm/jepa_lm.py`](../../architectures/jepa_lm/jepa_lm.py) — tap commit `bc93273`. Same env-var contract across all 14 runs (only configs differ).
- **Crucible**: `eren23/crucible@969cac5`, all bootstrap and torch_backend fixes upstream (`0e6f311` data_download cd-prefix, `7a4b5b3` TORCH_COMPILE opt-out + smoke wallclock bump).
- **Compute**: 4× RunPod RTX 4090 (3 dedicated + 1 shared during overnight chain).
- **Total cost**: ~$15 over ~16 GPU-hours.
- **Wandb**: project `parameter-golf`, entity `eren23`. Run names match table above.

## Files

- [`README.md`](./README.md) — this doc
- [`val_bpb_curves.csv`](./val_bpb_curves.csv) — per-step val_bpb across all 14 runs
- [`../../architectures/jepa_lm/`](../../architectures/jepa_lm/) — installable plugin (`crucible tap install jepa_lm --type architectures`)

## Next directions (not yet tested)

1. **Span-masking** (PR #1581 approach): replace target tokens with a learned mask in the context-encoder pass. Forces non-trivial prediction, addresses PR #1330 collapse. Requires double forward pass — implementation cost is real.
2. **Phased α ramp**: pure AR (30%) → AR+JEPA ramp (50%) → pure AR cooldown (20%). PR #832 schedule.
3. **EMA target encoder**: BYOL-style. PR #896 already showed no-gain at this scale, deprioritized.
4. **Different backbone scale**: PR #832 won at 24M / byte-level. Maybe JEPA helps below 17M but hurts above. Untested here.
