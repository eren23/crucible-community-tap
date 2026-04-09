"""Lion optimizer — EvoLved Sign Momentum (Chen et al., 2023).

Sign-based update with decoupled weight decay. Memory-efficient
alternative to Adam: only stores momentum (no second moment),
halving optimizer state memory.

Usage:
    EMBED_OPTIMIZER=lion MATRIX_OPTIMIZER=lion ...
"""
import torch
from crucible.training.optimizers import register_optimizer, OPTIMIZER_REGISTRY


class Lion(torch.optim.Optimizer):
    """Lion: sign(interpolation) update with decoupled weight decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Update: sign of interpolation between momentum and gradient
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign_(), alpha=-group["lr"])

                # Momentum update (separate from the sign step above)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def _lion_factory(params, **kwargs):
    # Filter to Lion-accepted params only (ignore Muon-specific kwargs like momentum, backend_steps)
    accepted = {"lr", "betas", "weight_decay"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return Lion(params, **filtered)


register_optimizer("lion", _lion_factory, source="local")

OPTIMIZER_REGISTRY.register_schema("lion", {
    "lr": {"type": "float", "default": 1e-4, "description": "Learning rate (typically 3-10x lower than Adam)"},
    "betas": {"type": "tuple", "default": "(0.9, 0.99)", "description": "Momentum interpolation coefficients"},
    "weight_decay": {"type": "float", "default": 0.0, "description": "Decoupled weight decay"},
})
