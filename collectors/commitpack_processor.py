"""CommitPack processor -- downloads Python commits from HuggingFace and builds flat HDF5.

Downloads Python commit diffs from CommitPack or CommitPackFT (bigcode),
AST-tokenizes both before/after source code, computes a 7-dimensional action
vector capturing edit semantics, and writes everything to a flat HDF5 file
optimized for random-access batch sampling.

7-dim action vector layout::

    [0] ADD      -- one-hot: new file added
    [1] DELETE   -- one-hot: file deleted
    [2] MODIFY   -- one-hot: file modified
    [3] SCOPE_FUNC    -- one-hot: function-level change
    [4] SCOPE_CLASS   -- one-hot: class-level change
    [5] SCOPE_MODULE  -- one-hot: module-level change
    [6] LOCATION      -- normalized position of first diff (0.0-1.0)

HDF5 output schema (flat, chunked for streaming)::

    /metadata          (group with attrs: vocab_size, context_window, action_dim,
                         num_edits, tokenizer, source)
    /before_tokens     (num_edits, context_window)  uint16
    /after_tokens      (num_edits, context_window)  uint16
    /edit_actions      (num_edits, action_dim)       float32

Usage::

    processor = CommitPackProcessor()
    stats = processor.collect(
        Path("."), Path("commitpack_python.h5"),
        max_samples=50000, use_ft=True, context_window=512,
    )
"""
from __future__ import annotations

import ast as ast_mod
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .ast_tokenizer import ast_tokenize, get_vocab_size
from .base import BaseCollector, CollectionStats

logger = logging.getLogger(__name__)

ACTION_DIM = 7


# ---------------------------------------------------------------------------
# 7-dim action vector
# ---------------------------------------------------------------------------

def compute_action(old_source: str, new_source: str) -> np.ndarray:
    """Compute 7-dim action vector from before/after source code.

    Dimensions:
        0-2: Edit type one-hot (ADD / DELETE / MODIFY)
        3-5: Scope one-hot (function / class / module)
        6:   Normalized position of first difference (0.0-1.0)
    """
    action = np.zeros(ACTION_DIM, dtype=np.float32)

    # Edit type (one-hot, dim 0-2)
    if not old_source or not old_source.strip():
        action[0] = 1.0  # ADD
    elif not new_source or not new_source.strip():
        action[1] = 1.0  # DELETE
    else:
        action[2] = 1.0  # MODIFY

    # Scope detection (one-hot, dim 3-5)
    scope = _detect_scope(old_source, new_source)
    action[3 + scope] = 1.0  # 0=function, 1=class, 2=module

    # Location (dim 6) -- normalized position of first difference
    action[6] = _estimate_location(old_source, new_source)

    return action


def _detect_scope(old_source: str, new_source: str) -> int:
    """Detect structural scope of the change.

    Returns:
        0 -- function-level (function added/removed/body changed)
        1 -- class-level (class added/removed)
        2 -- module-level (imports, top-level statements, fallback)
    """
    try:
        old_tree = ast_mod.parse(old_source) if old_source and old_source.strip() else ast_mod.parse("")
        new_tree = ast_mod.parse(new_source) if new_source and new_source.strip() else ast_mod.parse("")
    except SyntaxError:
        return 2  # module-level fallback

    # Collect function and class names from both versions
    old_funcs = {
        n.name for n in ast_mod.walk(old_tree)
        if isinstance(n, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef))
    }
    new_funcs = {
        n.name for n in ast_mod.walk(new_tree)
        if isinstance(n, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef))
    }
    old_classes = {n.name for n in ast_mod.walk(old_tree) if isinstance(n, ast_mod.ClassDef)}
    new_classes = {n.name for n in ast_mod.walk(new_tree) if isinstance(n, ast_mod.ClassDef)}

    # If function set changed, it's function-level
    if old_funcs != new_funcs:
        return 0

    # If class set changed, it's class-level
    if old_classes != new_classes:
        return 1

    # Check if function bodies changed (same names but different content)
    old_func_sizes = _func_body_sizes(old_tree)
    new_func_sizes = _func_body_sizes(new_tree)
    if old_func_sizes != new_func_sizes:
        return 0  # function-level body change

    return 2  # module-level (imports, top-level statements)


def _func_body_sizes(tree: ast_mod.AST) -> dict[str, int]:
    """Map function name to approximate body size (AST node count)."""
    sizes: dict[str, int] = {}
    for node in ast_mod.walk(tree):
        if isinstance(node, (ast_mod.FunctionDef, ast_mod.AsyncFunctionDef)):
            sizes[node.name] = sum(1 for _ in ast_mod.walk(node))
    return sizes


def _estimate_location(old_source: str, new_source: str) -> float:
    """Estimate normalized position (0.0-1.0) of first change.

    Compares line-by-line and returns the fraction of the file where the
    first difference appears. Returns 1.0 for identical or empty inputs.
    """
    old_lines = old_source.splitlines() if old_source else []
    new_lines = new_source.splitlines() if new_source else []

    total = max(len(old_lines), len(new_lines), 1)

    for i, (a, b) in enumerate(zip(old_lines, new_lines)):
        if a != b:
            return i / total

    # Lines match up to the shorter file -- difference is at the end
    return min(len(old_lines), len(new_lines)) / total


# ---------------------------------------------------------------------------
# Flat HDF5 writer
# ---------------------------------------------------------------------------

def _write_flat_hdf5(
    output: Path,
    before_list: list[np.ndarray],
    after_list: list[np.ndarray],
    action_list: list[np.ndarray],
    context_window: int,
    action_dim: int,
    source_name: str,
) -> None:
    """Write collected data to a flat, chunked HDF5 file.

    All arrays are stacked into contiguous datasets for efficient random
    batch access. Chunks are sized for streaming reads.
    """
    import h5py

    n = len(before_list)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output), "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["vocab_size"] = get_vocab_size()
        meta.attrs["context_window"] = context_window
        meta.attrs["action_dim"] = action_dim
        meta.attrs["num_edits"] = n
        meta.attrs["tokenizer"] = "python_ast"
        meta.attrs["source"] = source_name

        # Flat concatenated arrays -- efficient for random batch sampling
        chunk_rows = min(1024, n)
        f.create_dataset(
            "before_tokens",
            data=np.stack(before_list),
            dtype="uint16",
            chunks=(chunk_rows, context_window),
        )
        f.create_dataset(
            "after_tokens",
            data=np.stack(after_list),
            dtype="uint16",
            chunks=(chunk_rows, context_window),
        )
        f.create_dataset(
            "edit_actions",
            data=np.stack(action_list),
            dtype="float32",
            chunks=(chunk_rows, action_dim),
        )

    logger.info(
        "Written flat HDF5: %d edits, context_window=%d, action_dim=%d -> %s",
        n, context_window, action_dim, output,
    )


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class CommitPackProcessor(BaseCollector):
    """Download and preprocess CommitPack Python data into flat HDF5.

    Uses HuggingFace ``datasets`` with streaming to avoid downloading the
    entire dataset upfront. Each sample yields a (before_tokens, action,
    after_tokens) triple where tokens come from the Python AST tokenizer
    and actions are 7-dim semantic edit descriptors.
    """

    def collect(
        self,
        source: Path,
        output: Path,
        *,
        max_samples: int | None = None,
        use_ft: bool = True,
        context_window: int = 512,
        max_file_size: int = 50_000,
        **kwargs: Any,
    ) -> CollectionStats:
        """Download and preprocess CommitPack Python data.

        Parameters
        ----------
        source:
            Ignored (data comes from HuggingFace). Pass ``Path(".")``.
        output:
            Path to output HDF5 file.
        max_samples:
            Maximum number of samples to process. ``None`` processes all
            available data (can be very large for full CommitPack).
        use_ft:
            If ``True`` (default), use CommitPackFT (filtered/cleaned).
            If ``False``, use the full CommitPack dataset.
        context_window:
            AST token sequence length for before/after arrays. Default 512.
        max_file_size:
            Skip samples where either old or new content exceeds this
            character count. Default 50000.
        """
        from datasets import load_dataset

        dataset_name = "bigcode/commitpackft" if use_ft else "bigcode/commitpack"
        logger.info(
            "Loading %s (python split, streaming), max_samples=%s",
            dataset_name, max_samples,
        )

        # CommitPack datasets store data as JSONL files at
        # data/{language}/data.jsonl. Newer versions of the `datasets` library
        # no longer support the custom loading scripts these repos ship.
        # Load directly from the JSONL file via HuggingFace URLs.
        ds = None
        jsonl_url = f"hf://datasets/{dataset_name}/data/python/data.jsonl"
        strategies = [
            # Strategy 1: load JSONL directly (most reliable)
            lambda: load_dataset(
                "json", data_files=jsonl_url, split="train", streaming=True,
            ),
            # Strategy 2: standard load (works with datasets < 4.5)
            lambda: load_dataset(
                dataset_name, "python", split="train", streaming=True,
            ),
        ]

        last_error = None
        for i, strategy in enumerate(strategies):
            try:
                ds = strategy()
                # Verify we can iterate and the schema is right
                peek = next(iter(ds))
                assert "old_contents" in peek or "new_contents" in peek, \
                    f"Unexpected schema: {list(peek.keys())}"
                logger.info("Loaded dataset via strategy %d", i + 1)
                break
            except Exception as exc:
                last_error = exc
                logger.debug("Strategy %d failed: %s", i + 1, exc)
                ds = None

        if ds is None:
            logger.error("All loading strategies failed. Last error: %s", last_error)
            raise RuntimeError(
                f"Cannot load {dataset_name}. Last error: {last_error}"
            )

        # Accumulate in lists, write all at once
        before_list: list[np.ndarray] = []
        after_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []

        skipped_empty = 0
        skipped_size = 0
        processed = 0

        for sample in ds:
            if max_samples is not None and processed >= max_samples:
                break

            old_contents: str = sample.get("old_contents", "") or ""
            new_contents: str = sample.get("new_contents", "") or ""

            # Skip if both empty
            if not old_contents.strip() and not new_contents.strip():
                skipped_empty += 1
                continue

            # Skip oversized files
            if len(old_contents) > max_file_size or len(new_contents) > max_file_size:
                skipped_size += 1
                continue

            # AST-tokenize both versions
            before_tokens = ast_tokenize(old_contents, context_window)
            after_tokens = ast_tokenize(new_contents, context_window)

            # Compute 7-dim action vector
            action = compute_action(old_contents, new_contents)

            before_list.append(before_tokens)
            after_list.append(after_tokens)
            action_list.append(action)

            processed += 1

            if processed % 1000 == 0:
                logger.info(
                    "Processed %d samples (skipped: %d empty, %d oversized)",
                    processed, skipped_empty, skipped_size,
                )

        if not before_list:
            logger.warning("No valid samples found -- nothing to write")
            return CollectionStats(
                source=dataset_name,
                output_path=str(output),
                metadata={"error": "no valid samples"},
            )

        logger.info(
            "Processing complete: %d samples kept, %d empty skipped, %d oversized skipped",
            processed, skipped_empty, skipped_size,
        )

        # Write flat HDF5
        _write_flat_hdf5(
            output=output,
            before_list=before_list,
            after_list=after_list,
            action_list=action_list,
            context_window=context_window,
            action_dim=ACTION_DIM,
            source_name=dataset_name,
        )

        return CollectionStats(
            num_transitions=processed,
            num_sequences=processed,
            source=dataset_name,
            output_path=str(output),
            metadata={
                "dataset": dataset_name,
                "vocab_size": get_vocab_size(),
                "context_window": context_window,
                "action_dim": ACTION_DIM,
                "max_samples": max_samples,
                "max_file_size": max_file_size,
                "skipped_empty": skipped_empty,
                "skipped_size": skipped_size,
            },
        )

    def validate(self, output: Path) -> bool:
        """Verify flat HDF5 integrity and expected schema.

        Checks that all three datasets (before_tokens, after_tokens,
        edit_actions) exist with shapes consistent with metadata attributes.
        """
        import h5py

        try:
            with h5py.File(str(output), "r") as f:
                if "metadata" not in f:
                    logger.error("Missing /metadata group")
                    return False

                meta = f["metadata"]
                for key in ("num_edits", "context_window", "action_dim", "vocab_size"):
                    if key not in meta.attrs:
                        logger.error("Missing metadata attribute: %s", key)
                        return False

                n = int(meta.attrs["num_edits"])
                cw = int(meta.attrs["context_window"])
                ad = int(meta.attrs["action_dim"])

                for ds_name, expected_shape in [
                    ("before_tokens", (n, cw)),
                    ("after_tokens", (n, cw)),
                    ("edit_actions", (n, ad)),
                ]:
                    if ds_name not in f:
                        logger.error("Missing dataset: %s", ds_name)
                        return False
                    if f[ds_name].shape != expected_shape:
                        logger.error(
                            "Shape mismatch for %s: expected %s, got %s",
                            ds_name, expected_shape, f[ds_name].shape,
                        )
                        return False

            return True
        except Exception:
            logger.exception("Validation failed for %s", output)
            return False
