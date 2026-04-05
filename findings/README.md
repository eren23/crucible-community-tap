# Findings

Community research findings documented in this tap. Each subdirectory is one
finding with:

- `README.md` — the claim, evidence, and reproducibility
- Supporting analysis files (optional)

Findings are **reusable insights** — not experiment logs. They should be
interpretable by someone who didn't run the experiments.

## Current findings

| Topic | Claim |
|-------|-------|
| [code_wm_transition_geometry](code_wm_transition_geometry/README.md) | Code world models need both state geometry (SIGReg) and transition geometry (delta-direction) — the G8 hybrid recipe is what works. |

## How to add a finding

See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-finding).

**Minimum template** for `findings/<topic>/README.md`:

```markdown
# Finding: <one-sentence claim>

**Claim**: <one sentence>
**Date validated**: YYYY-MM-DD
**Contributor**: <your handle>
**Reproducibility**: <pointer to commands/dataset>

## Evidence
<table/figure/numbers>

## Limitations
<what we didn't test>
```
