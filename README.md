# Crucible Community Tap

Community plugin repository for the [Crucible ML research platform](https://github.com/eren23/parameter-golf_dev). Plugins are auto-discovered after installation — no config changes needed.

## Quick Start

```bash
# Add this tap
crucible tap add https://github.com/eren23/crucible-community-tap

# Browse plugins
crucible tap search architecture

# Install one (or all)
crucible tap install moe_baseline
crucible tap install sota_partial_rope
crucible tap install looped_augmented
crucible tap install sota_inspired_v1

# Run an experiment with an installed architecture
crucible run experiment --preset smoke  # with MODEL_FAMILY=moe_baseline in design config
```

## Available Plugins

### Architecture Plugins (4)

#### `moe_baseline` — Mixture of Experts Transformer

**Type:** Python plugin | **Origin:** Parameter Golf competition research

Replaces the dense MLP in every transformer block with a top-k routed Mixture of Experts layer. Retains U-Net encoder-decoder skip connections from the baseline architecture. Includes load-balancing auxiliary loss for expert utilization.

```
Architecture: TiedEmbeddingLM
  Blocks: MoEBlock (RMSNorm -> CausalSelfAttention -> MoELayer)
  Skip: U-Net encoder-decoder with learned per-dim weights
  Routing: Top-k (default k=2 of 4 experts)
  Loss: CE + auxiliary load-balancing loss
```

**Key config:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_FAMILY` | `moe_baseline` | Select this architecture |
| `MOE_NUM_EXPERTS` | `4` | Number of expert FFNs |
| `MOE_TOP_K` | `2` | Experts active per token |
| `MODEL_DIM` | `512` | Hidden dimension |
| `NUM_LAYERS` | `9` | Total transformer layers |
| `ACTIVATION` | `relu_sq` | Activation function |

---

#### `sota_partial_rope` — Partial Rotary Position Embeddings

**Type:** Python plugin | **Origin:** Top-5 Parameter Golf submissions

Applies rotary positional embeddings to only the first `rope_dims` dimensions of each attention head, leaving the rest position-independent. This allows the model to learn both position-dependent and position-independent features within the same head — a technique from the highest-scoring competition entries.

Includes the full augmentation stack: SmearGate, BigramHash, TrigramHash, encoder-decoder skip connections, gated residuals, and orthogonal initialization.

```
Architecture: TiedEmbeddingLM
  Blocks: PartialRoPEBlock (PartialRoPEAttention + MLP + gated residual)
  Skip: U-Net encoder-decoder with learned per-dim weights
  RoPE: First 16 dims per head (configurable)
  Augmentations: SmearGate + BigramHash + TrigramHash
  Init: Orthogonal with projection scaling
```

**Key config:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_FAMILY` | `sota_partial_rope` | Select this architecture |
| `ROPE_DIMS` | `16` | Dims per head with RoPE applied |
| `SMEAR_GATE` | `true` | Enable SmearGate augmentation |
| `BIGRAM_HASH` | `true` | Enable BigramHash context |
| `TRIGRAM_HASH` | `true` | Enable TrigramHash context |
| `ORTHO_INIT` | `true` | Orthogonal weight initialization |
| `NUM_LAYERS` | `11` | Total transformer layers |

---

#### `looped_augmented` — Recurrent Transformer with Augmentations

**Type:** Declarative YAML spec | **Origin:** QLabs research on weight-sharing efficiency

A parameter-efficient architecture that reuses a small set of transformer blocks across multiple recurrence steps. 3 unique blocks are applied 12 times (configurable), with per-step scaling parameters — achieving depth equivalent to 12 layers with roughly 3 layers' worth of parameters.

Combined with SmearGate and BigramHash augmentations from the competition baseline.

```
Architecture: Composed via YAML spec (no custom Python)
  Stack: Looped — 3 unique blocks x 12 steps
  Block: attention_block (standard attention + MLP)
  Augmentations: SmearGate + BigramHash
  Init: Orthogonal
```

**Key config:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_FAMILY` | `looped_augmented` | Select this architecture |
| `RECURRENCE_STEPS` | `12` | Logical forward pass steps |
| `SHARE_BLOCKS` | `3` | Unique blocks (weight sets) |
| `SMEAR_GATE` | `true` | Enable SmearGate |
| `BIGRAM_HASH` | `true` | Enable BigramHash |

---

#### `sota_inspired_v1` — Competition-Winning Full Stack

**Type:** Declarative YAML spec | **Origin:** Parameter Golf leaderboard top entries

The "kitchen sink" architecture combining every proven technique from competition winners. U-Net encoder-decoder skip connections with gated residuals, 3x MLP expansion, multiscale attention windows, and all three augmentations (SmearGate, BigramHash, TrigramHash). 11 layers with orthogonal initialization.

```
Architecture: Composed via YAML spec (no custom Python)
  Stack: encoder_decoder_skip — U-Net style, 11 layers
  Block: attention_block with gated residuals + multiscale windows
  MLP: 3x expansion (hidden = 3 * model_dim)
  Augmentations: SmearGate + BigramHash + TrigramHash
  Init: Orthogonal
```

**Key config:**
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_FAMILY` | `sota_inspired_v1` | Select this architecture |
| `NUM_LAYERS` | `11` | Total transformer layers |
| `MLP_MULT` | `3` | MLP expansion factor |
| `MODEL_DIM` | `512` | Hidden dimension |

---

## How Plugins Work

Crucible uses a 3-tier plugin system with auto-discovery:

```
Priority (highest to lowest):
  1. Local:   .crucible/architectures/*.py and *.yaml
  2. Global:  ~/.crucible-hub/plugins/architectures/*.py  <-- tap installs go here
  3. Builtin: src/crucible/models/architectures/
```

When you `crucible tap install moe_baseline`, the plugin file is copied to `~/.crucible-hub/plugins/architectures/moe_baseline.py`. On next import, Crucible's architecture registry discovers and loads it automatically.

## MCP Integration

All installed architectures are immediately available through Crucible's 114 MCP tools:

```python
# List all families (including installed plugins)
model_list_families()

# Get source code for a Python plugin
model_fetch_architecture(family="moe_baseline")

# Get YAML spec for a declarative plugin
model_get_spec(family="looped_augmented")

# Search the tap for plugins
hub_search(query="moe")

# Install from MCP
hub_install(name="moe_baseline")
```

## Experiment Design Example

```yaml
# .crucible/designs/moe-screen/v1.yaml
name: moe-screen
config:
  MODEL_FAMILY: moe_baseline
  MODEL_DIM: 512
  NUM_LAYERS: 9
  NUM_HEADS: 8
  NUM_KV_HEADS: 4
  MOE_NUM_EXPERTS: 4
  MOE_TOP_K: 2
  ACTIVATION: relu_sq
base_preset: screen
tags:
  - moe
  - community-tap
```

## Contributing

### Add a Plugin

1. Fork this repository
2. Create `{type}/{name}/` directory (e.g., `architectures/my_arch/`)
3. Add your plugin code as `{name}.py`
4. Add a `plugin.yaml` manifest:

```yaml
name: my_arch
type: architectures
version: 0.1.0
description: Short description of what this architecture does
author: your-github-username
tags:
  - architecture
  - relevant-tags
```

5. Open a PR

### Plugin Types

| Type | Directory | Description |
|------|-----------|-------------|
| `architectures` | `architectures/` | Model architectures (nn.Module builders) |
| `optimizers` | `optimizers/` | Custom optimizers |
| `schedulers` | `schedulers/` | LR schedulers |
| `callbacks` | `callbacks/` | Training callbacks |
| `loggers` | `loggers/` | Logging backends |
| `data_adapters` | `data_adapters/` | Data loading adapters |
| `objectives` | `objectives/` | Training objectives/loss functions |
| `block_types` | `block_types/` | Composer block types |
| `stack_patterns` | `stack_patterns/` | Composer stack patterns |
| `augmentations` | `augmentations/` | Composer augmentations |
| `activations` | `activations/` | Activation functions |
| `providers` | `providers/` | Fleet compute providers |

### Naming Rules

- Names must match `[a-zA-Z0-9][a-zA-Z0-9_-]*`
- No spaces, no special characters, no path traversal
- Use underscores for multi-word names

## License

MIT
