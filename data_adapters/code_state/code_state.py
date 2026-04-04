"""Code state data adapter -- loads git edit HDF5 for world model training.

Loads HDF5 files produced by the ``git_edit`` or ``commitpack`` collectors
and serves (state, action, next_state) tuples for world model training.
Each transition is a localized code change:

- **states**: AST token IDs of file content *before* the edit
- **actions**: action vector (edit features)
- **next_states**: AST token IDs of file content *after* the edit

Supports two HDF5 schemas, auto-detected at load time:

- **flat** (new): top-level datasets ``/before_tokens``, ``/edit_actions``,
  ``/after_tokens`` with shape ``[N, ...]``.  Fast random access via fancy
  indexing -- no per-sample group traversal.
- **grouped** (legacy): ``/edits/0/before_tokens``, etc.  One group per
  sample.  Slower for large datasets but still supported.

The actual embedding (tokens -> vectors) happens in the architecture's
encoder, not here. This adapter returns raw token IDs.

Env vars::

    WM_HDF5_PATH   -- path to HDF5 file (required; falls back to synthetic data)
    WM_BATCH_SIZE   -- default batch size (default: 64)
    WM_DATA_FORMAT  -- data format hint (default: "git_edit", future: "trace")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from crucible.core.errors import DataError
from crucible.training.data_adapters import DataAdapter, register_data_adapter

logger = logging.getLogger(__name__)


class CodeStateAdapter(DataAdapter):
    """Load code edit transitions from HDF5 for world model training.

    Lazily loads the HDF5 on first ``next_batch()`` call. Falls back to
    synthetic random data when ``WM_HDF5_PATH`` is unset, allowing
    pipeline testing without real data.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.hdf5_path = os.environ.get("WM_HDF5_PATH", "")
        self.default_batch_size = int(os.environ.get("WM_BATCH_SIZE", "64"))
        self.data_format = os.environ.get("WM_DATA_FORMAT", "git_edit")
        self.device = kwargs.get("device", "cpu")

        # Populated by _ensure_loaded
        self._loaded = False
        self._use_synthetic = False
        self._num_edits = 0
        self._context_window = 256
        self._vocab_size = 260
        self._h5file: Any = None  # h5py.File kept open for random access
        self._h5_format: str = ""  # "flat" or "grouped"
        self._edits_group: Any = None

        # Pre-loaded arrays for synthetic mode or preloaded mode
        self._before_tokens: Any = None  # np.ndarray [N, context_window]
        self._edit_actions: Any = None   # np.ndarray [N, 4]
        self._after_tokens: Any = None   # np.ndarray [N, context_window]

    def _ensure_loaded(self) -> None:
        """Lazy-load on first batch request."""
        if self._loaded:
            return
        self._loaded = True

        if not self.hdf5_path:
            logger.info("WM_HDF5_PATH not set -- using synthetic code state data")
            self._use_synthetic = True
            self._generate_synthetic_pool()
            return

        self._use_synthetic = False
        self._load_hdf5()

    def _detect_format(self) -> str:
        """Auto-detect HDF5 format: 'flat' or 'grouped'."""
        if "before_tokens" in self._h5file:
            return "flat"
        elif "edits" in self._h5file:
            return "grouped"
        else:
            raise DataError(f"Unrecognized HDF5 format in {self.hdf5_path}")

    def _load_hdf5(self) -> None:
        """Open HDF5 and read metadata. Keeps file open for random access."""
        import h5py

        path = Path(self.hdf5_path)
        if not path.exists():
            logger.warning("HDF5 file not found: %s -- falling back to synthetic", path)
            self._use_synthetic = True
            self._generate_synthetic_pool()
            return

        self._h5file = h5py.File(str(path), "r")
        self._h5_format = self._detect_format()

        meta = self._h5file["metadata"]
        self._num_edits = int(meta.attrs["num_edits"])
        self._context_window = int(meta.attrs["context_window"])
        self._vocab_size = int(meta.attrs["vocab_size"])

        if self._h5_format == "grouped":
            self._edits_group = self._h5file["edits"]

        if self._num_edits == 0:
            logger.warning("HDF5 has zero edits -- falling back to synthetic")
            self._h5file.close()
            self._h5file = None
            self._use_synthetic = True
            self._generate_synthetic_pool()
            return

        logger.info(
            "Loaded code state HDF5 (%s): %d edits, context_window=%d, vocab=%d",
            self._h5_format, self._num_edits, self._context_window, self._vocab_size,
        )

    def _generate_synthetic_pool(self, pool_size: int = 512) -> None:
        """Generate a pool of synthetic transitions for testing."""
        import numpy as np

        self._context_window = 256
        self._vocab_size = 260

        # Random token sequences (byte range 0-255, plus occasional special tokens)
        self._before_tokens = np.random.randint(0, 256, size=(pool_size, self._context_window), dtype=np.uint16)
        self._after_tokens = np.random.randint(0, 256, size=(pool_size, self._context_window), dtype=np.uint16)

        # Random edit actions: one-hot edit type + normalized line offset
        actions = np.zeros((pool_size, 4), dtype=np.float32)
        edit_types = np.random.randint(0, 3, size=pool_size)
        for i in range(pool_size):
            actions[i, edit_types[i]] = 1.0
        actions[:, 3] = np.random.uniform(0.0, 1.0, size=pool_size).astype(np.float32)
        self._edit_actions = actions

        self._num_edits = pool_size
        logger.info("Generated synthetic code state pool: %d transitions", pool_size)

    def next_batch(self, batch_size: int = 0, **kwargs: Any) -> dict[str, Any]:
        """Return a batch of code edit transitions.

        Returns
        -------
        dict with:
            - ``states``: [B, seq_len] long tensor -- before tokens
            - ``actions``: [B, action_dim] float tensor -- edit action
            - ``next_states``: [B, seq_len] long tensor -- after tokens
        """
        import numpy as np
        import torch

        self._ensure_loaded()

        if batch_size <= 0:
            batch_size = self.default_batch_size

        # Sample random indices
        indices = np.random.randint(0, self._num_edits, size=batch_size)

        if self._use_synthetic:
            before = self._before_tokens[indices]
            actions = self._edit_actions[indices]
            after = self._after_tokens[indices]
        elif self._h5_format == "flat":
            # Flat schema: top-level datasets with shape [N, ...].
            # h5py fancy indexing requires unique sorted indices.
            unique_sorted = np.unique(indices)
            before_uniq = self._h5file["before_tokens"][unique_sorted.tolist()]
            actions_uniq = self._h5file["edit_actions"][unique_sorted.tolist()]
            after_uniq = self._h5file["after_tokens"][unique_sorted.tolist()]
            # Map back to original order (handles duplicates)
            remap = np.searchsorted(unique_sorted, indices)
            before = before_uniq[remap]
            actions = actions_uniq[remap]
            after = after_uniq[remap]
        else:
            # Grouped schema: one HDF5 group per sample
            before_list = []
            action_list = []
            after_list = []
            for idx in indices:
                g = self._edits_group[str(idx)]
                before_list.append(np.array(g["before_tokens"]))
                action_list.append(np.array(g["edit_action"]))
                after_list.append(np.array(g["after_tokens"]))
            before = np.stack(before_list)
            actions = np.stack(action_list)
            after = np.stack(after_list)

        states = torch.from_numpy(before.astype(np.int64)).to(self.device)
        action_tensor = torch.from_numpy(actions).to(self.device)
        next_states = torch.from_numpy(after.astype(np.int64)).to(self.device)

        return {
            "states": states,
            "actions": action_tensor,
            "next_states": next_states,
        }

    @classmethod
    def modality(cls) -> str:
        return "world_model"

    def __del__(self) -> None:
        """Close HDF5 file handle on garbage collection."""
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass


register_data_adapter("code_state", CodeStateAdapter, source="local")
