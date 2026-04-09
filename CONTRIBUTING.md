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
2. Create a `plugin.yaml` manifest following the schema below
3. Add your Python module(s) alongside `plugin.yaml`
4. Optional: add a `README.md` with usage details
5. Run `crucible tap validate .` locally — it walks every plugin.yaml
   and reports errors/warnings. Fix errors before committing.
6. Run `crucible tap push <tap-name>` and open a PR

### plugin.yaml schema

The full schema is defined in `crucible.core.plugin_schema` and
enforced by `crucible tap validate`. Required fields:

```yaml
name: my_plugin                # [a-zA-Z_][a-zA-Z0-9_-]*, unique
type: architectures            # see KNOWN_PLUGIN_TYPES
version: "0.1.0"               # semver-ish (M.m.p, -suffix ok)
description: One-line what-it-does
```

Strongly recommended (missing = warning, not error):

```yaml
author: your_handle
tags: [world-model, jepa, lewm]
crucible_compat: ">=0.2,<0.3"  # version range against crucible-ml
dependencies:                  # Python deps your plugin needs
  - "torch>=2.0"
  - "einops>=0.7"
  - {name: "h5py", version: ">=3"}   # dict form also accepted
```

Free-form (not validated, documentation only):

```yaml
config:                        # default env vars the plugin sets
  MODEL_FAMILY: looped_augmented
parameters:                    # runtime knobs your plugin reads
  RECURRENCE_STEPS: "Number of logical recurrence steps (default: 12)"
```

### Validating your plugin

Before committing:

```bash
# Validate one plugin (fastest iteration)
python3 -c "
from crucible.core.plugin_schema import validate_manifest_file
from pathlib import Path
r = validate_manifest_file(Path('my_category/my_plugin/plugin.yaml'))
for i in r.issues:
    print(f'[{i.severity.upper()}] {i.field}: {i.message}')
print('OK' if r.ok else 'FAIL')
"

# Validate the whole tap
crucible tap validate .

# Fail on warnings too (CI-strict mode)
crucible tap validate . --warnings-as-errors
```

### Versioning policy

- Every plugin starts at `0.1.0`.
- Bump the **patch** (`0.1.0` → `0.1.1`) for bug fixes or doc-only changes.
- Bump the **minor** (`0.1.0` → `0.2.0`) when adding new features that
  don't break existing configs.
- Bump the **major** (`0.1.0` → `1.0.0`) on breaking changes to env
  vars, config keys, or module API.
- Add a `CHANGELOG.md` next to `plugin.yaml` for non-trivial updates.

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
