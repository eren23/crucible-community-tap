"""Trajectory-aware data adapter for multi-step world model training.

Loads trajectory-structured HDF5 (produced by TrajectoryCollector) and serves
multi-step (state, action, next_state) chains.  Falls back to single-step
random sampling when trajectory metadata is absent.

Each batch returns a window of ``window_len`` consecutive transitions from a
randomly sampled trajectory at a random starting position::

    {
        "states":      [B, window_len+1, seq_len]  long   -- token IDs
        "actions":     [B, window_len, action_dim]  float  -- action vectors
        "traj_ids":    [B]                          long   -- trajectory ID per sample
        "start_steps": [B]                          long   -- starting step per sample
    }

The first state in each window is the "initial state" for rollout.
Subsequent states are ground truth for multi-step prediction targets.

Env vars::

    WM_HDF5_PATH     -- path to trajectory HDF5 (required)
    WM_BATCH_SIZE     -- batch size (default: 64)
    WM_WINDOW_LEN     -- number of transitions per sample (default: 4)
    WM_SINGLE_STEP    -- if "1", fall back to single-step sampling (default: "0")
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryStateAdapter:
    """Multi-step trajectory data adapter for code world model training.

    Reads trajectory-structured HDF5 and samples windows of consecutive
    transitions for multi-step training.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.hdf5_path = os.environ.get("WM_HDF5_PATH", "")
        self.default_batch_size = int(os.environ.get("WM_BATCH_SIZE", "64"))
        self.window_len = int(os.environ.get("WM_WINDOW_LEN", "4"))
        self.force_single_step = os.environ.get("WM_SINGLE_STEP", "0") == "1"
        self.device = kwargs.get("device", "cpu")

        self._loaded = False
        self._h5file: Any = None
        self._has_trajectories = False

        # Trajectory metadata
        self._traj_offsets: np.ndarray | None = None
        self._traj_lengths: np.ndarray | None = None
        self._num_trajectories = 0
        self._num_edits = 0
        self._context_window = 512
        self._action_dim = 7
        self._vocab_size = 662

        # For single-step fallback
        self._eligible_trajs: np.ndarray | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self.hdf5_path:
            raise ValueError("WM_HDF5_PATH must be set for TrajectoryStateAdapter")

        import h5py
        from pathlib import Path

        path = Path(self.hdf5_path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 not found: {path}")

        self._h5file = h5py.File(str(path), "r")
        meta = self._h5file["metadata"]

        self._num_edits = int(meta.attrs["num_edits"])
        self._context_window = int(meta.attrs["context_window"])
        self._action_dim = int(meta.attrs["action_dim"])
        self._vocab_size = int(meta.attrs["vocab_size"])
        self._has_trajectories = bool(meta.attrs.get("has_trajectories", False))

        if self._has_trajectories and not self.force_single_step:
            traj = self._h5file["trajectory"]
            self._traj_offsets = traj["traj_offsets"][:]
            self._traj_lengths = traj["traj_lengths"][:]
            self._num_trajectories = len(self._traj_lengths)

            # Find trajectories long enough for our window
            self._eligible_trajs = np.where(
                self._traj_lengths >= self.window_len
            )[0]

            if len(self._eligible_trajs) == 0:
                logger.warning(
                    "No trajectories with length >= %d (max is %d). "
                    "Falling back to single-step.",
                    self.window_len, int(self._traj_lengths.max()),
                )
                self._has_trajectories = False
            else:
                logger.info(
                    "Loaded trajectory HDF5: %d trajectories (%d eligible for window=%d), "
                    "%d total transitions",
                    self._num_trajectories, len(self._eligible_trajs),
                    self.window_len, self._num_edits,
                )
        else:
            logger.info(
                "Loaded HDF5 in single-step mode: %d transitions",
                self._num_edits,
            )

    def next_batch(self, batch_size: int = 0, **kwargs: Any) -> dict[str, Any]:
        """Sample a batch of multi-step trajectory windows.

        Returns dict with:
            - states: [B, window_len+1, seq_len] long -- all states in window
            - actions: [B, window_len, action_dim] float -- actions between states
            - traj_ids: [B] long -- which trajectory each sample comes from
            - start_steps: [B] long -- starting step within trajectory
        """
        import torch

        self._ensure_loaded()

        if batch_size <= 0:
            batch_size = self.default_batch_size

        if self._has_trajectories and not self.force_single_step:
            return self._sample_trajectory_windows(batch_size)
        else:
            return self._sample_single_step(batch_size)

    def _sample_trajectory_windows(self, batch_size: int) -> dict[str, Any]:
        """Sample windows of consecutive transitions from trajectories."""
        import torch

        W = self.window_len
        CW = self._context_window
        AD = self._action_dim

        # Sample random eligible trajectories
        traj_indices = self._eligible_trajs[
            np.random.randint(0, len(self._eligible_trajs), size=batch_size)
        ]

        all_states = np.zeros((batch_size, W + 1, CW), dtype=np.int64)
        all_actions = np.zeros((batch_size, W, AD), dtype=np.float32)
        all_traj_ids = np.zeros(batch_size, dtype=np.int64)
        all_start_steps = np.zeros(batch_size, dtype=np.int64)

        before_ds = self._h5file["before_tokens"]
        after_ds = self._h5file["after_tokens"]
        action_ds = self._h5file["edit_actions"]

        for b, traj_idx in enumerate(traj_indices):
            offset = int(self._traj_offsets[traj_idx])
            length = int(self._traj_lengths[traj_idx])

            # Random start within trajectory (must have W transitions ahead)
            max_start = length - W
            start = np.random.randint(0, max_start + 1)

            # Global indices in flat arrays
            global_start = offset + start
            global_end = global_start + W

            # States: before_tokens[start:start+W] gives W states,
            # plus after_tokens[start+W-1] for the final state
            idx_range = list(range(global_start, global_end))

            befores = before_ds[idx_range].astype(np.int64)
            actions = action_ds[idx_range].astype(np.float32)
            last_after = after_ds[global_end - 1].astype(np.int64)

            # states[0..W-1] = before_tokens of each transition
            # states[W] = after_tokens of last transition
            all_states[b, :W, :] = befores
            all_states[b, W, :] = last_after
            all_actions[b] = actions
            all_traj_ids[b] = traj_idx
            all_start_steps[b] = start

        states_t = torch.from_numpy(all_states).to(self.device)
        actions_t = torch.from_numpy(all_actions).to(self.device)
        traj_ids_t = torch.from_numpy(all_traj_ids).to(self.device)
        start_steps_t = torch.from_numpy(all_start_steps).to(self.device)

        return {
            "states": states_t,
            "actions": actions_t,
            "traj_ids": traj_ids_t,
            "start_steps": start_steps_t,
        }

    def _sample_single_step(self, batch_size: int) -> dict[str, Any]:
        """Fall back to single-step sampling (same as CodeStateAdapter)."""
        import torch

        indices = np.random.randint(0, self._num_edits, size=batch_size)
        unique_sorted = np.unique(indices)

        before_uniq = self._h5file["before_tokens"][unique_sorted.tolist()]
        actions_uniq = self._h5file["edit_actions"][unique_sorted.tolist()]
        after_uniq = self._h5file["after_tokens"][unique_sorted.tolist()]

        remap = np.searchsorted(unique_sorted, indices)
        before = before_uniq[remap].astype(np.int64)
        actions = actions_uniq[remap].astype(np.float32)
        after = after_uniq[remap].astype(np.int64)

        # Reshape to match trajectory format: window_len=1
        states = np.stack([before, after], axis=1)  # [B, 2, seq_len]
        actions = actions[:, np.newaxis, :]  # [B, 1, action_dim]

        return {
            "states": torch.from_numpy(states).to(self.device),
            "actions": torch.from_numpy(actions).to(self.device),
            "traj_ids": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            "start_steps": torch.zeros(batch_size, dtype=torch.long, device=self.device),
        }

    @classmethod
    def modality(cls) -> str:
        return "world_model"

    def __del__(self) -> None:
        if self._h5file is not None:
            try:
                self._h5file.close()
            except Exception:
                pass
