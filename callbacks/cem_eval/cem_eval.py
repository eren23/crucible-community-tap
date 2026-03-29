"""CEM planner evaluation callback for world models.

Runs Cross-Entropy Method planning at validation time to measure
goal-reaching success rate. This is the meaningful validation metric
for world models (as opposed to just prediction loss).

Env vars:
    CEM_EVAL_EVERY:     Steps between evals (0 = only at end, default: 0)
    CEM_EPISODES:       Number of episodes (default: 10)
    CEM_SAMPLES:        Candidate action sequences per CEM iteration (default: 300)
    CEM_ITERATIONS:     CEM refinement iterations (default: 30)
    CEM_TOPK:           Elite samples to keep (default: 30)
    CEM_HORIZON:        Planning horizon in steps (default: 5)
    CEM_ACTION_DIM:     Action dimensionality (default: 2)
    CEM_SUCCESS_THRESH: Distance threshold for success (default: 5.0)
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class CEMPlanner:
    """Cross-Entropy Method planner for goal-conditioned world models."""

    def __init__(
        self,
        action_dim: int = 2,
        horizon: int = 5,
        num_samples: int = 300,
        num_iterations: int = 30,
        topk: int = 30,
    ):
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.topk = topk

    def plan(self, model: Any, z_current: "torch.Tensor", z_goal: "torch.Tensor") -> "torch.Tensor":
        """Plan an action sequence to reach z_goal from z_current.

        Returns:
            [horizon, action_dim] optimal action sequence
        """
        import torch

        device = z_current.device
        if z_current.dim() == 1:
            z_current = z_current.unsqueeze(0)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(0)

        H, A, N, K = self.horizon, self.action_dim, self.num_samples, self.topk

        mean = torch.zeros(H, A, device=device)
        std = torch.ones(H, A, device=device)

        for _ in range(self.num_iterations):
            noise = torch.randn(N, H, A, device=device)
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1.0, 1.0)

            # Rollout each sequence
            costs = self._rollout_costs(model, z_current, z_goal, actions)

            # Select elites
            _, elite_idx = costs.topk(K, largest=False)
            elite_actions = actions[elite_idx]

            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp_min(0.01)

        return mean

    def _rollout_costs(self, model, z_current, z_goal, actions):
        """Roll out N action sequences and return L2 cost to goal."""
        import torch

        N = actions.shape[0]
        z = z_current.expand(N, -1)

        with torch.no_grad():
            for t in range(self.horizon):
                z = model.predict_next(z, actions[:, t])

        return (z - z_goal.expand(N, -1)).pow(2).sum(dim=-1)


class CEMEvalCallback(TrainingCallback):
    """Run CEM planning at validation time to measure world model quality."""

    priority = 80

    def __init__(self, **kwargs: Any) -> None:
        self.freq = int(os.environ.get("CEM_EVAL_EVERY", "0"))
        self.num_episodes = int(os.environ.get("CEM_EPISODES", "10"))
        self.num_samples = int(os.environ.get("CEM_SAMPLES", "300"))
        self.num_iterations = int(os.environ.get("CEM_ITERATIONS", "30"))
        self.topk = int(os.environ.get("CEM_TOPK", "30"))
        self.horizon = int(os.environ.get("CEM_HORIZON", "5"))
        self.action_dim = int(os.environ.get("CEM_ACTION_DIM", "2"))
        self.success_thresh = float(os.environ.get("CEM_SUCCESS_THRESH", "5.0"))

    def on_validation_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        if self.freq <= 0 or step % self.freq != 0:
            return
        rate = self._run(state)
        if rate is not None:
            metrics["cem_success_rate"] = rate

    def on_train_end(self, state: dict[str, Any]) -> None:
        rate = self._run(state)
        if rate is not None:
            print(f"metric:cem_success_rate={rate:.6f}", flush=True)

    def _run(self, state: dict[str, Any]) -> float | None:
        import torch

        model = state.get("model")
        if model is None or not hasattr(model, "predict_next"):
            return None

        device = next(model.parameters()).device
        planner = CEMPlanner(self.action_dim, self.horizon, self.num_samples, self.num_iterations, self.topk)

        embed_dim = getattr(model, "embed_dim", 192)
        successes = 0
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for _ in range(self.num_episodes):
                z_start = torch.randn(1, embed_dim, device=device) * 0.5
                z_goal = torch.randn(1, embed_dim, device=device) * 0.5

                actions = planner.plan(model, z_start, z_goal)

                z = z_start
                for t in range(actions.shape[0]):
                    z = model.predict_next(z, actions[t].unsqueeze(0))

                dist = (z - z_goal).pow(2).sum().sqrt().item()
                if dist < self.success_thresh:
                    successes += 1

        if was_training:
            model.train()

        return successes / max(self.num_episodes, 1)


register_callback("cem_eval", CEMEvalCallback, source="local")
