# Crucible Community Tap

Community plugins and examples for the [Crucible ML research platform](https://github.com/eren23/parameter-golf_dev).

## Quick Start

```bash
crucible tap add https://github.com/eren23/crucible-community-tap
crucible tap search architecture
crucible tap install moe
```

## Plugins

### Architectures

| Plugin | What it does | Type |
|--------|-------------|------|
| **`moe`** | Top-k routed Mixture of Experts replacing dense MLP. U-Net skips, load-balancing aux loss. | Python |
| **`partial_rope`** | RoPE on first 16 dims per head only. Full augmentation stack (SmearGate + BigramHash + TrigramHash). | Python |
| **`looped_aug`** | Weight-shared recurrence: 3 blocks x 12 steps + SmearGate + BigramHash. Parameter-efficient depth. | YAML spec |
| **`fullstack`** | Kitchen sink: U-Net skips + gated residuals + 3x MLP + all augments. Competition-winning config. | YAML spec |

### `moe` — Mixture of Experts

Replaces every dense MLP with a top-k routed MoE layer (default: 2 of 4 experts active per token). Same U-Net encoder-decoder skip structure as baseline. Forward pass adds load-balancing auxiliary loss to cross-entropy.

```
MODEL_FAMILY=moe  MOE_NUM_EXPERTS=4  MOE_TOP_K=2
MODEL_DIM=512  NUM_LAYERS=9  ACTIVATION=relu_sq
```

### `partial_rope` — Partial Rotary Embeddings

From top-5 Parameter Golf submissions. Applies RoPE to only the first `rope_dims` dimensions of each attention head — the rest learn position-independent features. Includes gated residuals, orthogonal init, and the full SmearGate + BigramHash + TrigramHash stack.

```
MODEL_FAMILY=partial_rope  ROPE_DIMS=16
NUM_LAYERS=11  SMEAR_GATE=true  BIGRAM_HASH=true  TRIGRAM_HASH=true  ORTHO_INIT=true
```

### `looped_aug` — Looped Transformer + Augmentations

3 unique transformer blocks applied 12 times with per-step scaling. Gets 12 layers of depth from 3 layers of parameters. Declarative YAML spec — zero Python.

```
MODEL_FAMILY=looped_aug  RECURRENCE_STEPS=12  SHARE_BLOCKS=3
SMEAR_GATE=true  BIGRAM_HASH=true  ORTHO_INIT=true
```

### `fullstack` — Everything At Once

The competition kitchen sink. 11-layer U-Net with gated residuals, 3x MLP expansion, multiscale attention, SmearGate, BigramHash, TrigramHash, orthogonal init. Declarative YAML spec.

```
MODEL_FAMILY=fullstack  NUM_LAYERS=11  MLP_MULT=3
SMEAR_GATE=true  BIGRAM_HASH=true  TRIGRAM_HASH=true
```

---

## Examples

### YOLO Object Detection via MCP

We fine-tuned YOLOv8n on COCO8 using Crucible's MCP tools — provision GPU, clone Ultralytics, train, collect metrics, tear down. 8 tool calls, 7 minutes, ~$0.10.

```
Results (10 epochs, RTX 4090):
  precision:  0.679
  recall:     0.750
  mAP50:      0.772
  mAP50-95:   0.606
```

Full trace with every MCP call and response: [`examples/yolo/`](examples/yolo/)

**Run it yourself:**
```bash
# Copy project spec to your crucible project
cp examples/yolo/yolo11-demo.yaml .crucible/projects/

# Then via MCP:
provision_project(project_name="yolo11-demo", count=1)
fleet_refresh()
bootstrap_project(project_name="yolo11-demo")
run_project(project_name="yolo11-demo", overrides={"EPOCHS": "10"})
collect_project_results(run_id="...")
destroy_nodes()
```

Swap `yolov8n.pt` for `yolo11n.pt`, `yolo11s.pt`, etc. Swap `coco8.yaml` for `coco128.yaml` or `coco.yaml`.

---

## How Plugins Work

```
Install priority (highest wins):
  1. Local:   .crucible/architectures/
  2. Hub:     ~/.crucible-hub/plugins/architectures/   <-- tap installs here
  3. Builtin: src/crucible/models/architectures/
```

`crucible tap install moe` copies to `~/.crucible-hub/plugins/architectures/moe.py`. Auto-discovered on next import.

## MCP Integration

```python
model_list_families()                        # see all (builtins + installed)
model_fetch_architecture(family="moe")       # read source code
model_get_spec(family="looped_aug")          # read YAML spec
hub_search(query="rope")                     # search tap
hub_install(name="partial_rope")             # install from MCP
```

## Contributing

1. Fork this repo
2. Add `{type}/{name}/` with `{name}.py` + `plugin.yaml`
3. PR

**Plugin types:** architectures, optimizers, schedulers, callbacks, loggers, data_adapters, objectives, block_types, stack_patterns, augmentations, activations, providers

**Names:** `[a-zA-Z0-9][a-zA-Z0-9_-]*` — underscores for multi-word.

## License

MIT
