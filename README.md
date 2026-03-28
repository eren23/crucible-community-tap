# Crucible Community Tap

Community plugin repository for the [Crucible ML research platform](https://github.com/eren23/parameter-golf_dev).

## Available Plugins

### Architectures

| Plugin | Description | Type |
|--------|-------------|------|
| **moe_baseline** | MoE transformer with top-k routed experts and U-Net skips | Python |
| **sota_partial_rope** | Partial RoPE + encoder-decoder skips + all augmentations | Python |
| **looped_augmented** | Weight-shared recurrence (12 steps, 3 blocks) + SmearGate + BigramHash | YAML spec |
| **sota_inspired_v1** | Competition-winning U-Net + gated residuals + 3x MLP + all augments | YAML spec |

## Installation

```bash
# Add this tap
crucible tap add https://github.com/eren23/crucible-community-tap

# Search available plugins
crucible tap search moe

# Install a plugin
crucible tap install moe_baseline

# Verify
crucible models list
```

## Usage

After installing, use architectures via the `MODEL_FAMILY` environment variable:

```bash
MODEL_FAMILY=moe_baseline crucible run experiment --preset smoke
MODEL_FAMILY=sota_partial_rope crucible run experiment --preset smoke
```

Or in experiment designs:

```yaml
config:
  MODEL_FAMILY: moe_baseline
  MOE_NUM_EXPERTS: 4
  MOE_TOP_K: 2
```

## Contributing

1. Fork this repository
2. Add your plugin under `{type}/{name}/` with a `plugin.yaml` manifest
3. Open a PR
