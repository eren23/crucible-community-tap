"""HDF5 trajectory data adapter for world model training.

Loads trajectory datasets with frames, actions, and optional goals from
HDF5 files. Returns sub-trajectory slices for JEPA-style training.

Expected HDF5 layout::

    /trajectories/
        /0/
            frames: [T, H, W, C] uint8 or [T, C, H, W] float32
            actions: [T-1, action_dim] float32
            goals: [H, W, C] uint8 (optional)
        /1/
            ...

Env vars:
    HDF5_PATH:     Path to HDF5 file or directory of HDF5 files (required)
    ACTION_DIM:    Action dimensionality (default: 2)
    NUM_FRAMES:    Sub-trajectory length (default: 4)
    IMAGE_SIZE:    Resize frames to this size (default: 224)
    HDF5_PRELOAD:  "1" to preload entire dataset to memory (default: "0")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from crucible.training.data_adapters import DataAdapter, register_data_adapter


class TrajectoryHDF5Adapter(DataAdapter):
    """Load trajectory sub-sequences from HDF5 for world model training."""

    def __init__(self, **kwargs: Any) -> None:
        self.hdf5_path = os.environ.get("HDF5_PATH", "")
        self.action_dim = int(os.environ.get("ACTION_DIM", "2"))
        self.num_frames = int(os.environ.get("NUM_FRAMES", "4"))
        self.image_size = int(os.environ.get("IMAGE_SIZE", "224"))
        self.preload = os.environ.get("HDF5_PRELOAD", "0") == "1"
        self.device = kwargs.get("device", "cpu")

        self._trajectories: list[dict[str, Any]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load trajectory data on first batch request."""
        if self._loaded:
            return
        self._loaded = True

        if not self.hdf5_path:
            # Fall back to synthetic bouncing balls
            self._use_synthetic = True
            return

        self._use_synthetic = False
        self._load_hdf5()

    def _load_hdf5(self) -> None:
        """Load trajectories from HDF5 file(s)."""
        import h5py
        import numpy as np

        path = Path(self.hdf5_path)
        files = list(path.glob("*.hdf5")) + list(path.glob("*.h5")) if path.is_dir() else [path]

        for fpath in files:
            with h5py.File(str(fpath), "r") as f:
                traj_group = f.get("trajectories", f)
                for traj_key in traj_group:
                    traj = traj_group[traj_key]
                    frames = np.array(traj["frames"])  # [T, H, W, C] or [T, C, H, W]
                    actions = np.array(traj["actions"])  # [T-1, A]

                    # Normalize to [T, C, H, W] float32 in [0, 1]
                    if frames.dtype == np.uint8:
                        frames = frames.astype(np.float32) / 255.0
                    if frames.ndim == 4 and frames.shape[-1] in (1, 3):
                        # [T, H, W, C] -> [T, C, H, W]
                        frames = np.transpose(frames, (0, 3, 1, 2))

                    entry: dict[str, Any] = {
                        "frames": frames,
                        "actions": actions.astype(np.float32),
                    }

                    # Optional goals
                    if "goals" in traj:
                        goals = np.array(traj["goals"])
                        if goals.dtype == np.uint8:
                            goals = goals.astype(np.float32) / 255.0
                        if goals.ndim == 3 and goals.shape[-1] in (1, 3):
                            goals = np.transpose(goals, (2, 0, 1))
                        entry["goals"] = goals

                    if self.preload:
                        self._trajectories.append(entry)
                    else:
                        self._trajectories.append(entry)

    def _generate_synthetic_batch(self, batch_size: int) -> dict[str, Any]:
        """Generate synthetic bouncing ball trajectories as fallback."""
        import torch

        T = self.num_frames
        H = W = self.image_size
        C = 3

        frames = torch.zeros(batch_size, T, C, H, W)
        actions = torch.randn(batch_size, T - 1, self.action_dim)

        # Simple bouncing balls
        for b in range(batch_size):
            # Ball position and velocity
            px, py = H * 0.5, W * 0.5
            vx = (torch.randn(1).item()) * 3
            vy = (torch.randn(1).item()) * 3
            radius = max(3, int(H * 0.05))

            for t in range(T):
                # Draw ball
                yy, xx = torch.meshgrid(
                    torch.arange(H, dtype=torch.float32),
                    torch.arange(W, dtype=torch.float32),
                    indexing="ij",
                )
                dist = ((xx - px) ** 2 + (yy - py) ** 2).sqrt()
                ball = (dist < radius).float()
                frames[b, t, 0] = ball  # Red channel
                frames[b, t, 1] = ball * 0.5  # Green
                frames[b, t, 2] = 0.0  # Blue

                # Update position with bouncing
                px += vx
                py += vy
                if px < radius or px > W - radius:
                    vx = -vx
                if py < radius or py > H - radius:
                    vy = -vy
                px = max(radius, min(W - radius, px))
                py = max(radius, min(H - radius, py))

                if t < T - 1:
                    actions[b, t] = torch.tensor([vx, vy])[: self.action_dim]

        return {"frames": frames, "actions": actions}

    def next_batch(self, batch_size: int = 8, **kwargs: Any) -> dict[str, Any]:
        """Return a batch of trajectory sub-sequences.

        Returns:
            dict with:
                - ``frames``: [B, T, C, H, W] float32 in [0, 1]
                - ``actions``: [B, T-1, action_dim] float32
                - ``goals``: [B, C, H, W] float32 (if available)
        """
        import torch

        self._ensure_loaded()

        if self._use_synthetic:
            batch = self._generate_synthetic_batch(batch_size)
            return {k: v.to(self.device) for k, v in batch.items()}

        import numpy as np

        T = self.num_frames
        frames_batch = []
        actions_batch = []
        goals_batch = []

        for _ in range(batch_size):
            # Random trajectory
            idx = np.random.randint(0, len(self._trajectories))
            traj = self._trajectories[idx]

            traj_frames = traj["frames"]  # [T_total, C, H, W]
            traj_actions = traj["actions"]  # [T_total-1, A]
            T_total = traj_frames.shape[0]

            # Random sub-sequence start
            max_start = max(0, T_total - T)
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0

            sub_frames = traj_frames[start : start + T]  # [T, C, H, W]
            sub_actions = traj_actions[start : start + T - 1]  # [T-1, A]

            # Resize if needed
            sub_frames_t = torch.from_numpy(sub_frames)
            if sub_frames_t.shape[-2:] != (self.image_size, self.image_size):
                import torch.nn.functional as F

                sub_frames_t = F.interpolate(
                    sub_frames_t,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )

            frames_batch.append(sub_frames_t)
            actions_batch.append(torch.from_numpy(sub_actions))

            if "goals" in traj:
                goal = torch.from_numpy(traj["goals"])
                if goal.shape[-2:] != (self.image_size, self.image_size):
                    import torch.nn.functional as F

                    goal = F.interpolate(
                        goal.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                goals_batch.append(goal)

        result: dict[str, Any] = {
            "frames": torch.stack(frames_batch).to(self.device),
            "actions": torch.stack(actions_batch).to(self.device),
        }
        if goals_batch:
            result["goals"] = torch.stack(goals_batch).to(self.device)

        return result

    @classmethod
    def modality(cls) -> str:
        return "world_model"


register_data_adapter("trajectory_hdf5", TrajectoryHDF5Adapter, source="local")
