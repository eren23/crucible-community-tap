# Crucible Community Tap

Community plugins, tools, and research findings for the
[Crucible ML research platform](https://github.com/eren23/parameter-golf_dev).

## What's in this tap

| | Count | What |
|---|:---:|---|
| [`architectures/`](architectures/) | 10 | Model architectures — LLM (5) + World Model (5) |
| [`callbacks/`](callbacks/) | 12 | Compression (pruning/QAT), research (TDA, CEM) |
| [`objectives/`](objectives/) | 3 | Custom training objectives |
| [`data_adapters/`](data_adapters/) | 2 | Dataset adapters |
| [`collectors/`](collectors/) | 1 | Git edit → AST token pipelines |
| [`launchers/`](launchers/) | 4 | End-to-end training scripts |
| [`evaluation/`](evaluation/) | — | Reusable eval tools |
| [`findings/`](findings/) | 1 | Documented research findings |
| [`attocode_integration/`](attocode_integration/) | — | Tools for Attocode + Synapse integration |

## Quick Start

```bash
crucible tap add https://github.com/eren23/crucible-community-tap
crucible tap search world-model
crucible tap install code_wm
```

---

## Architectures

### LLM (from Parameter Golf)

| Plugin | What it does |
|--------|-------------|
| [`moe`](architectures/moe/) | Top-k routed Mixture of Experts. U-Net skips + load-balancing aux loss. |
| [`partial_rope`](architectures/partial_rope/) | RoPE on first 16 dims per head only. Full SmearGate + BigramHash + TrigramHash stack. |
| [`looped_aug`](architectures/looped_aug/) | Weight-shared recurrence: 3 blocks × 12 steps. Parameter-efficient depth. YAML spec. |
| [`fullstack`](architectures/fullstack/) | Kitchen sink: U-Net + gated residuals + 3× MLP + all augments. YAML spec. |
| [`sota_xsa`](architectures/sota_xsa/) | Cross-self attention variant from Parameter Golf top-5. |

### World Model (JEPA-style state + transition)

| Plugin | What it does |
|--------|-------------|
| [`wm_base`](architectures/wm_base/) | Abstract JEPA base: shared predictor + delta-space losses (direction, magnitude, covariance). Used by all WM architectures below. |
| [`code_wm`](architectures/code_wm/) | **Code World Model** — AST-tokenized Python editor. 1.1M params, 128d, 6-loop transformer. CLS/attention/mean pooling. |
| [`lewm`](architectures/lewm/) | Latent Energy World Model (LE-WM). Slim variants for compact latent modeling. |
| [`hybrid_lewm`](architectures/hybrid_lewm/) | Autoregressive-latent hybrid LE-WM. |
| [`elastic_lewm`](architectures/elastic_lewm/) | Elastic-capacity LE-WM with dynamic width. |

---

## Callbacks

**Compression** (ship-ready model shrinkage):
`pruning_magnitude`, `pruning_attention_head`, `pruning_layer_removal`, `pruning_wanda`,
`qat_int8`, `qat_int4`, `qat_mixed`, `compression_metrics`, `distillation`, `sensitivity_analysis`

**Research:** `cem_eval` (cross-entropy method evaluation), `tda_monitor` (topological data analysis during training)

---

## Objectives

- **`sigreg`** — Sketched Isotropic Gaussian Regularizer (Cramér-Wold + Epps-Pulley). Gentle state-space regularizer that pairs with delta-geometry losses.
- **`state_prediction`** — JEPA state-prediction loss for world models.
- **`distillation`** — Teacher-student KL with temperature.

---

## Data Adapters

- **`code_state`** — HDF5-backed code state batches for world model training.
- **`trajectory_hdf5`** — Trajectory data adapter for sequential models.

---

## Collectors

- **`collectors/`** — Git history → AST-tokenized edit pairs pipeline. Includes:
  - `ast_tokenizer.py` (662-vocab Python AST tokenizer)
  - `ast_diff.py` (15-dim structural action vector)
  - `commitpack_processor.py` (CommitPack / CommitPackFT streaming)
  - `git_edit.py` (local git history harvester)

---

## Findings

Documented research insights with evidence and reproducibility.

| Finding | Claim |
|---------|-------|
| [`code_wm_transition_geometry`](findings/code_wm_transition_geometry/) | Code world models need both state geometry (SIGReg) and transition geometry (delta-direction) — the **G8 hybrid recipe** works. Includes 11-run ablation table + weight sweep. |

See [`findings/README.md`](findings/README.md) for how to add your own.

---

## Tools

### Evaluation

- **`evaluation/semantic_eval.py`** — Downstream probes for code world models: edit retrieval, k-NN classification, cluster purity, held-out prediction.
- **`evaluation/eval_code_wm.py`** — Standalone checkpoint evaluator.

### Attocode Integration

- **`attocode_integration/codewm_retrieval.py`** — Delta-NN edit retrieval CLI. Index a repo's git history, query with (before, after) hunks, get top-k historically similar edits with SHAs in ~30ms.
- **`attocode_integration/export_onnx.py`** — Export trained Code WM to ONNX for Synapse deployment.
- See [`attocode_integration/README.md`](attocode_integration/README.md) for benchmarks + usage.

---

## Examples

### YOLO Object Detection via MCP
Fine-tuned YOLOv8n on COCO8 using Crucible's MCP tools — 8 tool calls, 7 minutes, ~$0.10.
Full trace: [`examples/yolo/`](examples/yolo/)

### Code World Model
Train a 1.1M-param code editor with the G8 recipe:
[`launchers/code_wm/train_code_wm.py`](launchers/code_wm/train_code_wm.py) + [`projects/code_wm.yaml`](projects/code_wm.yaml)

---

## How Plugins Work

```
Install priority (highest wins):
  1. Local:   .crucible/<type>/
  2. Hub:     ~/.crucible-hub/plugins/<type>/   <-- tap installs here
  3. Builtin: src/crucible/<type>/
```

`crucible tap install <name>` copies to `~/.crucible-hub/plugins/<type>/`. Auto-discovered on next import.

## MCP Integration

```python
model_list_families()                        # all architectures (builtins + installed)
model_fetch_architecture(family="code_wm")   # read source code
hub_search(query="world-model")              # search tap
hub_install(name="code_wm")                  # install from MCP
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for submission guidelines — plugins, findings, examples.

## License

MIT
