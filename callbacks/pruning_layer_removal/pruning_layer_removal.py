"""ShortGPT-style layer removal callback.

Identifies redundant transformer layers by measuring angular distance
between each layer's input and output activations.  Layers with small
angular distance behave close to identity and can be removed with
minimal quality loss.

After removal the optimizer parameter groups are rebuilt so training
continues cleanly on the reduced model.

Env vars:
    LAYER_REMOVE_COUNT: Number of layers to remove (default 2).
    LAYER_REMOVE_METRIC: Redundancy metric, currently "angular_distance"
        (default "angular_distance").
    LAYER_REMOVE_AT_STEP: Step at which removal happens; 0 means before
        training (default 0).
    LAYER_REMOVE_INDICES: Optional comma-separated layer indices to remove
        directly, bypassing metric-based selection (e.g. "2,5").
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class LayerRemovalCallback(TrainingCallback):
    """Remove redundant layers identified by angular distance or explicit indices."""

    priority = 8

    def __init__(self, **kwargs: Any) -> None:
        self.remove_count = int(os.environ.get("LAYER_REMOVE_COUNT", "2"))
        self.metric = os.environ.get("LAYER_REMOVE_METRIC", "angular_distance")
        self.remove_at_step = int(os.environ.get("LAYER_REMOVE_AT_STEP", "0"))
        indices_raw = os.environ.get("LAYER_REMOVE_INDICES", "")
        self.explicit_indices: list[int] | None = None
        if indices_raw.strip():
            self.explicit_indices = [int(i.strip()) for i in indices_raw.split(",") if i.strip()]
        self._applied = False

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_block_list(model: Any) -> tuple[str, Any] | None:
        """Find the nn.ModuleList that holds repeated transformer blocks.

        Checks common attribute names first, then falls back to finding
        the first ModuleList whose children share a type.
        """
        import torch.nn as nn

        for attr_name in ("blocks", "layers", "transformer_blocks"):
            mod = getattr(model, attr_name, None)
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                return attr_name, mod

        # Fallback: first ModuleList with >1 child of the same type
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 1:
                types = [type(c) for c in module]
                if len(set(types)) == 1:
                    return name, module

        return None

    # ------------------------------------------------------------------
    # Redundancy scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _angular_distance(input_act: Any, output_act: Any) -> float:
        """Angular distance between layer input and output activations."""
        import torch
        import torch.nn.functional as F

        cos_sim = F.cosine_similarity(
            input_act.flatten(1), output_act.flatten(1), dim=1,
        )
        return torch.acos(cos_sim.clamp(-1, 1)).mean().item()

    def _score_layers(self, model: Any, block_list: Any) -> list[tuple[int, float]]:
        """Run a dummy forward to measure per-layer angular distance.

        Returns list of (layer_index, score) sorted ascending (most
        redundant first).
        """
        import torch

        scores: list[tuple[int, float]] = []
        hooks: list[Any] = []
        layer_io: dict[int, dict[str, Any]] = {}

        for idx, block in enumerate(block_list):
            layer_io[idx] = {}

            def _make_hooks(i: int):
                def _pre(mod, inputs):
                    x = inputs[0] if isinstance(inputs, tuple) else inputs
                    layer_io[i]["input"] = x.detach()

                def _post(mod, inputs, output):
                    out = output[0] if isinstance(output, tuple) else output
                    layer_io[i]["output"] = out.detach()

                return _pre, _post

            pre_fn, post_fn = _make_hooks(idx)
            hooks.append(block.register_forward_pre_hook(pre_fn))
            hooks.append(block.register_forward_hook(post_fn))

        # Tiny dummy forward — use model's device
        device = next(model.parameters()).device
        try:
            dummy = torch.randint(0, 100, (1, 16), device=device)
            with torch.no_grad():
                model(dummy)
        except Exception:
            # If the model signature is different, try keyword
            try:
                with torch.no_grad():
                    model(input_ids=dummy)
            except Exception:
                pass

        for h in hooks:
            h.remove()

        for idx in range(len(block_list)):
            io = layer_io.get(idx, {})
            inp = io.get("input")
            out = io.get("output")
            if inp is not None and out is not None:
                dist = self._angular_distance(inp, out)
                scores.append((idx, dist))

        # Sort ascending — smallest angular distance = most redundant
        scores.sort(key=lambda t: t[1])
        return scores

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def _remove_layers(self, model: Any, state: dict[str, Any]) -> None:
        import torch.nn as nn

        result = self._find_block_list(model)
        if result is None:
            return
        attr_name, block_list = result

        num_layers = len(block_list)
        if num_layers == 0:
            return

        # Determine which indices to remove
        if self.explicit_indices is not None:
            to_remove = sorted(
                [i for i in self.explicit_indices if 0 <= i < num_layers],
                reverse=True,
            )
        else:
            scores = self._score_layers(model, block_list)
            to_remove = sorted(
                [idx for idx, _ in scores[: self.remove_count]],
                reverse=True,
            )

        if not to_remove:
            return

        # Remove layers from the ModuleList (reverse order preserves indices)
        for idx in to_remove:
            del block_list[idx]

        # If the block list is a direct attribute, reassign to update num_layers
        if hasattr(model, attr_name):
            setattr(model, attr_name, block_list)

        # Update num_layers if the model tracks it
        if hasattr(model, "num_layers"):
            model.num_layers = len(block_list)

        # Rebuild optimizer parameter groups
        optimizers = state.get("optimizers")
        if optimizers is not None:
            import torch
            if isinstance(optimizers, dict):
                for key, opt in optimizers.items():
                    new_params = [p for p in model.parameters() if p.requires_grad]
                    optimizers[key] = type(opt)(new_params, **opt.defaults)
            elif hasattr(optimizers, "defaults"):
                new_params = [p for p in model.parameters() if p.requires_grad]
                state["optimizers"] = type(optimizers)(new_params, **optimizers.defaults)

        self._applied = True

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, state: dict[str, Any]) -> None:
        if self._applied:
            return
        if self.remove_at_step == 0:
            model = state.get("model")
            if model is not None:
                self._remove_layers(model, state)

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        if self._applied:
            return
        if step != self.remove_at_step:
            return
        model = state.get("model")
        if model is not None:
            self._remove_layers(model, state)
            metrics["layer_removal/removed"] = (
                len(self.explicit_indices) if self.explicit_indices else self.remove_count
            )


register_callback("pruning_layer_removal", LayerRemovalCallback, source="local")
