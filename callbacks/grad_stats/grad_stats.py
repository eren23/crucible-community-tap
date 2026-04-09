"""Gradient statistics callback — logs per-layer gradient norms.

Fires on_after_backward every N steps and stores stats in state["grad_stats"].
Zero overhead when not on a logging step.
"""
from crucible.training.callbacks import TrainingCallback, register_callback


class GradStatsCallback(TrainingCallback):
    """Log per-layer gradient norms for dead-layer and exploding-grad diagnosis."""

    priority = 15  # After grad_clip (10), before nan_detector (20)

    def __init__(self, *, log_every: int = 200, **kwargs):
        self.log_every = log_every

    def on_after_backward(self, step, state):
        if step % self.log_every != 0:
            return
        model = state.get("model")
        if model is None:
            return
        stats = {}
        total_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                norm = p.grad.norm().item()
                stats[name] = norm
                total_norm += norm ** 2
        stats["_total_norm"] = total_norm ** 0.5
        state.setdefault("grad_stats", {})[step] = stats


register_callback("grad_stats", GradStatsCallback, source="local")
