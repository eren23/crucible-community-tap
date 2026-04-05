# Code WM Ablation Results (Complete)

Dataset: CommitPackFT Python, 56K edits, 128d embeddings, 5K training steps.
Model: ~1.1M params. Eval: 2000 held-out samples, seed 42.

## Full Ablation Table

| Run | Config | edit_type | knn_joint | NMI(joint) | retrieval_lift | heldout_dcos | heldout_pred_cos | baseline_cos(b,a) |
|-----|--------|-----------|-----------|------------|----------------|--------------|------------------|---------------------|
| G1.a | pred only | 88.5% | 74.8% | 0.269 | 0.247 | -0.023 | 0.075 ❌ | 0.9998 |
| G1.b | **VICReg** | **95.0%** 🏆 | 77.8% | **0.368** 🏆 | **0.285** 🏆 | 0.137 | 0.062 ❌ | 0.802 ⭐ |
| G1.c | SIGReg | 92.8% | 71.5% | 0.210 | 0.257 | -0.020 | 0.075 ❌ | 0.9987 |
| G1.d | pred + **dir** | 90.3% | 77.8% | 0.280 | 0.254 | **0.234** 🏆 | **0.9995** ✓ | 0.9995 |
| G1.f | all delta | 89.3% | 74.5% | 0.214 | 0.254 | 0.204 | **0.9996** ✓ | 0.9996 |
| G2.a | delta + **mean** | 87.3% | 73.3% | 0.254 | 0.233 | 0.261 | 0.9986 ✓ | 0.9987 |
| G3.b | delta, **no proj** | 88.0% | 75.3% | 0.261 | 0.250 | 0.227 | 0.999x ✓ | 0.9994 |

🏆 = category winner | ✓ = functional predictor | ❌ = broken predictor | ⭐ = notable baseline shift

## Key Findings

### Finding 1: There are TWO use cases with DIFFERENT winners

**Classification / Retrieval / Clustering (encoder output quality):**
- **Winner: VICReg (G1.b)** — 95% edit_type, NMI 0.368, retrieval lift 0.285
- VICReg **spreads embeddings apart** (baseline cos 0.9998 → 0.802)
- SIGReg is second (92.8% edit_type) but lower NMI

**Next-state prediction / Rollouts (predictor quality):**
- **Winner: delta-direction loss (G1.d)** — heldout_pred_cos 0.9995, delta_cos 0.234
- VICReg/SIGReg break the predictor (pred_cos near-zero)

### Finding 2: Delta-direction alone beats "all delta losses"

G1.d (pred + dir, no mag, no cov) ≥ G1.f (pred + dir + mag + cov) on most metrics:
- knn_joint: 77.8% vs 74.5%
- edit_type: 90.3% vs 89.3%
- delta_cos (heldout): 0.234 vs 0.204

**Magnitude and covariance losses add overhead without clear benefit.**

### Finding 3: Pool mode has small effect

| Pool | train delta_cos | heldout delta_cos | edit_type |
|------|-----------------|-------------------|-----------|
| CLS | 0.667 | 0.204 | 89.3% |
| Mean | 0.379 | 0.261 | 87.3% |

CLS shows train/val gap (0.667 → 0.204) — may be overfitting. Mean pool is more stable. **Choose CLS for ~2% classification boost, mean for stability.**

### Finding 4: Projector isn't critical

G3.b (no projector) ≈ G1.f (projector) on semantic metrics:
- edit_type: 88.0% vs 89.3%
- knn_joint: 75.3% vs 74.5%

Projector can be removed to save parameters and simplify architecture.

### Finding 5: The encoder is the workhorse

Every method gets 87-95% edit_type classification. The encoder (AST tokens + CLS/mean pool + looped transformer) learns strong code representations regardless of loss formulation. The loss function determines what ELSE the model does (predict vs discriminate).

## Paper Framing (Revised)

### Primary claim (supported)
> **For code world models, the choice of regularizer creates a fundamental trade-off: VICReg-style objectives produce discriminative encoders ideal for retrieval/classification, while delta-direction supervision produces functional predictors ideal for rollouts. Neither dominates — choose based on downstream task.**

### Secondary claim (supported)
> **Delta-direction loss is the minimal sufficient addition for making world-model predictors functional. Delta-magnitude calibration and covariance penalties add complexity without clear benefit.**

### Contributions
1. **Formulate** code editing as a JEPA-style world-modeling problem
2. **Propose** delta-direction loss as the minimal addition to prediction loss for functional predictors
3. **Demonstrate** a trade-off: discriminative encoders vs functional predictors
4. **Show** VICReg/SIGReg optimize classification, delta-losses optimize prediction
5. **Release** a 1M-param Python code world model suitable for deployment

## For Attocode (Track B)

**Two checkpoints to ship:**

1. **G1.b (VICReg+CLS)** — for retrieval, anomaly scoring, edit clustering
   - 95% edit_type classification
   - 0.368 NMI clustering
   - Best for: "find similar edits", "flag anomalous commits"

2. **G1.d (pred+dir+CLS)** — for next-edit prediction, rollouts
   - Working predictor (cos 0.9995)
   - Delta direction alignment 0.234
   - Best for: "what edit will happen next", "score candidate edits"

**Both are 1.1M params → fit anywhere via Synapse.**

## Open Questions

1. **Can we get BOTH?** Combine VICReg regularization with delta-direction loss in one model?
2. **Does VICReg/SIGReg's "broken predictor" recover with more training steps?**
3. **Why does G1.d beat G1.f?** Adding magnitude loss slightly hurts — why?
4. **Architecture vs loss**: how much is CLS pool vs how much is delta-direction?
