"""Late-stage Stochastic Weight Averaging callback.

Averages model weights over the final fraction of training for better
generalization. Activates at start_frac * total_steps and accumulates
a running mean, applied on_train_end.
"""
import copy
from crucible.training.callbacks import TrainingCallback, register_callback


class SWALateCallback(TrainingCallback):
    """SWA over the final portion of training."""

    priority = 80  # Late — after most other callbacks

    def __init__(self, *, start_frac: float = 0.8, **kwargs):
        self.start_frac = start_frac
        self._swa_state = None
        self._swa_count = 0

    def on_step_end(self, step, metrics, state):
        total = state.get("total_steps", 0)
        if total <= 0 or step < int(total * self.start_frac):
            return

        model = state.get("model")
        if model is None:
            return

        if self._swa_state is None:
            self._swa_state = {k: v.float().clone() for k, v in model.state_dict().items()}
            self._swa_count = 1
        else:
            for k, v in model.state_dict().items():
                self._swa_state[k] += v.float()
            self._swa_count += 1

    def on_train_end(self, state):
        model = state.get("model")
        if model is None or self._swa_state is None or self._swa_count == 0:
            return

        # Compute mean and load back
        avg_state = {}
        for k, v in self._swa_state.items():
            avg = v / self._swa_count
            # Cast back to the model's dtype
            orig_dtype = model.state_dict()[k].dtype
            avg_state[k] = avg.to(dtype=orig_dtype)

        model.load_state_dict(avg_state)
        state["swa_applied"] = True
        state["swa_count"] = self._swa_count


register_callback("swa_late", SWALateCallback, source="local")
