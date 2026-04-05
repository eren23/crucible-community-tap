# Attocode Integration: Code WM for Edit Retrieval

Tools to use a trained Code World Model as an **edit retrieval index** on top of any
Python git repository. Complements attocode-code-intel's static/symbol-based search
with a learned, forward-looking, delta-space similarity metric.

## What this gives you

Given a (before, after) code hunk you're working on, find the **top-k historically
similar edits** from the repo's git history — retrieved by transition geometry, not
by file/symbol overlap.

"How did we handle this kind of edit before?" — answered in ~30ms on CPU.

## Contents

```
attocode_integration/
├── codewm_retrieval.py    # Index + query + benchmark CLI
├── export_onnx.py         # Export checkpoint to ONNX for Synapse deployment
└── README.md              # This file
```

## Quick Start

### 1. Grab a Code WM checkpoint

Any G8-style checkpoint works. Two sources:

- Trained locally: `/tmp/synapse_codewm_package/g8_sigreg_dir.pt`
- Retrained at 500K scale: see `launchers/code_wm/train_code_wm.py`

### 2. Index a repo's git history

```bash
cd crucible-community-tap

WM_POOL_MODE=cls python attocode_integration/codewm_retrieval.py index \
    --repo /path/to/any/python/git/repo \
    --checkpoint /tmp/synapse_codewm_package/g8_sigreg_dir.pt \
    --out ./my_repo_idx \
    --max-commits 500 \
    --max-pairs 2000
```

Walks up to 500 commits, extracts (before, after) for each Python file touched,
encodes the delta, writes a compact index (deltas + metadata JSON).

**Size**: ~450KB index per 1K indexed edits (128d delta × float32 × L2-normalized).

### 3. Query with a new edit

```bash
WM_POOL_MODE=cls python attocode_integration/codewm_retrieval.py query \
    --index ./my_repo_idx \
    --checkpoint /tmp/synapse_codewm_package/g8_sigreg_dir.pt \
    --before before.py --after after.py \
    --top-k 5
```

The `--before` and `--after` arguments accept either a file path or raw code string.

Output:
```
Top-5 similar edits (query in 27.87 ms, over 200 indexed edits):

  1. [+0.987] 15ee6ffa  src/crucible/data_sources/__init__.py
       "export all three built-in plugins"
  2. [+0.541] 0d470dca  src/crucible/data_sources/__init__.py
       "add HuggingFaceDataSource plugin"
  3. [+0.531] 5659cd02  tests/test_data_sources_wandb.py
       "add WandBArtifactSource plugin"
  4. [+0.528] 349d185f  tests/test_runpod_provider.py
       "bootstrap process"
  5. [+0.497] 0d470dca  tests/test_data_sources_hf.py
       "add HuggingFaceDataSource plugin"
```

All 5 results are "add imports / new exports" type edits — a semantic cluster, not
just textual matches. Including across test files the query never touched.

### 4. Measure latency

```bash
WM_POOL_MODE=cls python attocode_integration/codewm_retrieval.py benchmark \
    --index ./my_repo_idx \
    --checkpoint /tmp/synapse_codewm_package/g8_sigreg_dir.pt \
    --num-queries 100
```

## How it works

```
input: (before_code, after_code)

  ├─ ast_tokenize(before_code)   ─┐
  │                                ├─ CodeWM encoder ─ z_before (128d)
  └─ ast_tokenize(after_code)    ─┤
                                   └─ CodeWM encoder ─ z_after  (128d)

  delta = z_after - z_before
  delta_unit = delta / ||delta||

  cosine search vs indexed deltas → top-k SHAs + file paths + commit messages
```

The encoder is 1.1M params (or 2.2M for 192d), ~4.8MB checkpoint fp32. Index
lookup is pure numpy dot-product over a tiny matrix.

**Measured latency on CPU (Apple M-series, 200-entry index):**

| Operation | Time |
|-----------|------|
| Pure NN search (dot-product + argsort) | 0.08 ms/query |
| Tokenize + encode small hunk (~10 lines) | ~28 ms |
| Tokenize + encode large file (512 tokens saturated) | ~89 ms |
| **End-to-end query (small edit)** | **~30 ms** |

## Comparison vs attocode's native search

| What | Attocode's tools | CodeWM retrieval |
|------|------------------|------------------|
| `find_related(file)` | Files with overlapping symbols | — |
| `semantic_search(query)` | Code text similarity | — |
| `change_coupling(file)` | Files historically changed together | — |
| `frecency_search` | Recent/frequent files | — |
| **Edit retrieval (before, after)** | — | **Delta-space cosine NN** |

CodeWM **complements** attocode — it indexes *edits*, not *files* or *symbols*.
The gap filled: "what past edits had similar STRUCTURE to this one?"

## Combining with attocode MCP tools

Example integration sketch (not yet wired up):

```python
# In an agent loop:
recent = mcp_attocode.recent_changes(file_path="src/foo.py")
similar = codewm_retrieval.query(before=hunk_before, after=hunk_after, top_k=3)

# Fuse: attocode's recency signal + CodeWM's semantic edit signal
combined = rank_by(recent, similar, weights=[0.3, 0.7])
```

## Synapse deployment

The Rust implementation of the same pipeline is in the Synapse repo:
`crates/synapse-code-tokenizer/` (byte-exact AST tokenizer) +
`crates/synapse-code-wm/` (ONNX/Rust inference).

With Synapse, the same retrieval works in WASM, ESP32, and any Rust target —
zero Python runtime needed at query time.

For ONNX export: `export_onnx.py` (in this directory).

## Limitations & Honest Caveats

- **Trained on CommitPack Python** (56K or 500K commits). Cross-repo generalization
  untested — a repo with very unusual patterns may get weak clusters.
- **Top-1 is often a self-match** (the query edit itself) — that's expected behavior
  when querying against an indexed edit; lift is in the top-2 through top-k.
- **Large files skipped**: edits touching files >40KB are excluded from the index
  (AST parsing gets slow, 512-token budget saturates).
- **Merge commits skipped** by construction.
