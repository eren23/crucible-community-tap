# Contributing to crucible-community-tap

This tap is a shared registry of Crucible plugins, evaluation tools, and
research findings. Contributions are welcome.

## Structure

```
crucible-community-tap/
├── architectures/     # Model architecture plugins (world models, LE-WM, etc.)
├── callbacks/         # Training callbacks (TDA monitor, etc.)
├── collectors/        # Data collectors (git, CommitPack, AST tokenizer)
├── data_adapters/     # Dataset adapters (code_state, etc.)
├── objectives/        # Training objectives (state_prediction, etc.)
├── launchers/         # End-to-end training scripts
├── evaluation/        # Reusable eval tools (semantic_eval.py, etc.)
├── findings/          # Community research findings (see findings/README.md)
├── attocode_integration/  # Tools for Attocode + Synapse integration
└── examples/          # Example projects using tap plugins
```

## Adding a plugin

1. Pick the right subdirectory (`architectures/`, `callbacks/`, etc.)
2. Create a `plugin.yaml` manifest:
   ```yaml
   name: my_plugin
   type: architectures  # or callbacks, collectors, etc.
   version: 0.1.0
   description: One-line description
   author: your_handle
   tags: [world_model, code, etc.]
   ```
3. Add your Python module(s) alongside `plugin.yaml`
4. Optional: add a `README.md` with usage details
5. Run `crucible tap push <tap-name>` and open a PR

## Adding a finding

Findings are **research insights**, not experiment logs. Each documents a
claim with evidence that someone else could reproduce.

1. Create `findings/<topic>/README.md` following the template in
   [findings/README.md](findings/README.md)
2. Include concrete numbers and comparison tables
3. State limitations honestly
4. Link to reproducibility commands + any dataset/checkpoint pointers
5. Open a PR

## Adding an example project

Example projects show how to use tap plugins end-to-end. They live in
`examples/<project-name>/` and should include:

- `README.md` — what the project does
- `experiment_spec.yaml` — plugin composition
- Commands to reproduce

## What NOT to commit

- `.h5` / `.npy` / `.pt` data files larger than a few MB — use HuggingFace Hub
- Personal `.env` files with API keys
- Raw experiment logs or W&B artifacts
- Hardcoded local paths (`/tmp/...`, `/Users/...`)

## Style

- Python: match existing module style (type hints where useful, docstrings at
  module + function level, line length ~100)
- YAML: 2-space indent, lowercase keys
- Markdown: use tables for comparisons, code blocks for commands
- Keep READMEs scannable — table of contents for anything >200 lines
