"""Attention head pruning callback.

Scores attention heads by Taylor importance (gradient * activation) and
zeros out Q/K/V projection slices for the least important heads at a
target training step.

During calibration, forward hooks on attention modules accumulate
per-head activation magnitudes.  At the prune step, a backward pass
captures gradients, Taylor scores are computed, and the bottom heads
are permanently zeroed.

Env vars:
    HEAD_PRUNE_RATIO: Fraction of heads to prune (default 0.25).
    HEAD_PRUNE_METRIC: Importance metric, currently "taylor"
        (default "taylor").
    HEAD_PRUNE_AT_STEP: Step at which pruning fires (default 500).
    HEAD_PRUNE_EXCLUDE_PATTERNS: Comma-separated module name patterns to
        skip (default "").
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class AttentionHeadPruningCallback(TrainingCallback):
    """Prune attention heads by Taylor importance scoring."""

    priority = 8

    def __init__(self, **kwargs: Any) -> None:
        self.prune_ratio = float(os.environ.get("HEAD_PRUNE_RATIO", "0.25"))
        self.metric = os.environ.get("HEAD_PRUNE_METRIC", "taylor")
        self.prune_at_step = int(os.environ.get("HEAD_PRUNE_AT_STEP", "500"))
        exclude_raw = os.environ.get("HEAD_PRUNE_EXCLUDE_PATTERNS", "")
        self.exclude_patterns = tuple(p.strip() for p in exclude_raw.split(",") if p.strip())

        # Populated during setup
        self._attn_modules: list[tuple[str, Any, int]] = []  # (name, module, num_heads)
        self._act_hooks: list[Any] = []
        self._head_activations: dict[str, Any] = {}   # name -> accumulated per-head magnitudes
        self._head_act_counts: dict[str, int] = {}
        self._applied = False

    # ------------------------------------------------------------------
    # Attention module discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_attention_modules(model: Any, exclude_patterns: tuple[str, ...]) -> list[tuple[str, Any, int]]:
        """Find attention modules and infer their head count.

        Looks for modules with Q/K/V weight attributes whose feature
        dimension is divisible by num_heads.
        """
        import torch.nn as nn

        results: list[tuple[str, Any, int]] = []

        for name, module in model.named_modules():
            if any(p in name for p in exclude_patterns):
                continue

            # Strategy 1: module has explicit num_heads / n_head attribute
            num_heads = getattr(module, "num_heads", None) or getattr(module, "n_head", None)

            # Strategy 2: look for w_q or qkv weight
            q_weight = None
            if hasattr(module, "w_q") and hasattr(module.w_q, "weight"):
                q_weight = module.w_q.weight
            elif hasattr(module, "qkv") and hasattr(module.qkv, "weight"):
                q_weight = module.qkv.weight
            elif hasattr(module, "q_proj") and hasattr(module.q_proj, "weight"):
                q_weight = module.q_proj.weight

            if q_weight is None:
                # Also check if "attn" in name and has a weight
                if "attn" not in name.lower():
                    continue
                # Try any child named w_q, q_proj, qkv
                for child_name, child in module.named_children():
                    if child_name in ("w_q", "q_proj", "qkv") and hasattr(child, "weight"):
                        q_weight = child.weight
                        break
                if q_weight is None:
                    continue

            if num_heads is None:
                # Cannot determine head count without it
                continue

            out_features = q_weight.shape[0]
            if out_features % num_heads != 0:
                continue

            results.append((name, module, num_heads))

        return results

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_activation_hooks(self, model: Any) -> None:
        """Register forward hooks to accumulate per-head activation magnitudes."""
        import torch

        self._attn_modules = self._find_attention_modules(model, self.exclude_patterns)

        for name, module, num_heads in self._attn_modules:
            self._head_activations[name] = None
            self._head_act_counts[name] = 0

            def _make_hook(mod_name: str, n_heads: int):
                def _hook(mod, inputs, output):
                    if self._applied:
                        return
                    out = output[0] if isinstance(output, tuple) else output
                    if out is None:
                        return
                    # out shape: (batch, seq, model_dim) or (batch, seq, num_heads, head_dim)
                    if out.ndim == 4:
                        # (batch, seq, num_heads, head_dim) -> per-head magnitude
                        per_head = out.abs().mean(dim=(0, 1, 3))  # (num_heads,)
                    elif out.ndim == 3:
                        # (batch, seq, model_dim) -> reshape to heads
                        b, s, d = out.shape
                        if d % n_heads != 0:
                            return
                        head_dim = d // n_heads
                        reshaped = out.view(b, s, n_heads, head_dim)
                        per_head = reshaped.abs().mean(dim=(0, 1, 3))
                    else:
                        return

                    if self._head_activations[mod_name] is None:
                        self._head_activations[mod_name] = torch.zeros_like(per_head)
                    self._head_activations[mod_name] += per_head.detach()
                    self._head_act_counts[mod_name] += 1
                return _hook

            handle = module.register_forward_hook(_make_hook(name, num_heads))
            self._act_hooks.append(handle)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _compute_taylor_scores(self, model: Any) -> list[tuple[str, int, float]]:
        """Compute Taylor importance = |activation * gradient| per head.

        Returns list of (module_name, head_index, importance_score).
        """
        import torch

        scores: list[tuple[str, int, float]] = []

        for name, module, num_heads in self._attn_modules:
            act = self._head_activations.get(name)
            count = max(self._head_act_counts.get(name, 1), 1)
            if act is None:
                # No activations collected — assign zero importance
                for h in range(num_heads):
                    scores.append((name, h, 0.0))
                continue

            mean_act = act / count

            # Collect gradient information from Q projection weights
            q_weight = None
            for child_name in ("w_q", "q_proj"):
                child = getattr(module, child_name, None)
                if child is not None and hasattr(child, "weight"):
                    q_weight = child.weight
                    break
            if q_weight is None and hasattr(module, "qkv"):
                q_weight = module.qkv.weight

            if q_weight is not None and q_weight.grad is not None:
                out_features = q_weight.shape[0]
                head_dim = out_features // num_heads
                grad_reshaped = q_weight.grad.view(num_heads, head_dim, -1)
                weight_reshaped = q_weight.data.view(num_heads, head_dim, -1)
                # Taylor: |grad * weight| summed per head
                taylor_per_head = (grad_reshaped * weight_reshaped).abs().sum(dim=(1, 2))
                # Combine with activation magnitude
                combined = taylor_per_head * mean_act
                for h in range(num_heads):
                    scores.append((name, h, combined[h].item()))
            else:
                # Fallback: use activation magnitude only
                for h in range(num_heads):
                    scores.append((name, h, mean_act[h].item()))

        return scores

    def _zero_head_projections(self, model: Any, heads_to_prune: list[tuple[str, int]]) -> None:
        """Zero out Q/K/V projection weight slices for pruned heads."""
        import torch

        # Group by module name
        prune_map: dict[str, list[int]] = {}
        for mod_name, head_idx in heads_to_prune:
            prune_map.setdefault(mod_name, []).append(head_idx)

        for name, module, num_heads in self._attn_modules:
            head_indices = prune_map.get(name, [])
            if not head_indices:
                continue

            # Zero Q, K, V projections for each pruned head
            for proj_name in ("w_q", "w_k", "w_v", "q_proj", "k_proj", "v_proj"):
                proj = getattr(module, proj_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue
                out_features = proj.weight.shape[0]
                head_dim = out_features // num_heads
                with torch.no_grad():
                    for h in head_indices:
                        start = h * head_dim
                        end = (h + 1) * head_dim
                        proj.weight.data[start:end] = 0.0
                        if proj.bias is not None:
                            proj.bias.data[start:end] = 0.0

            # Handle fused QKV projection
            if hasattr(module, "qkv") and hasattr(module.qkv, "weight"):
                qkv = module.qkv
                total_out = qkv.weight.shape[0]
                per_proj = total_out // 3
                head_dim = per_proj // num_heads
                with torch.no_grad():
                    for h in head_indices:
                        for proj_offset in range(3):
                            start = proj_offset * per_proj + h * head_dim
                            end = start + head_dim
                            qkv.weight.data[start:end] = 0.0
                            if qkv.bias is not None:
                                qkv.bias.data[start:end] = 0.0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, state: dict[str, Any]) -> None:
        model = state.get("model")
        if model is None:
            return
        self._register_activation_hooks(model)

    def on_step_end(self, step: int, metrics: dict[str, Any], state: dict[str, Any]) -> None:
        if self._applied:
            return
        if step != self.prune_at_step:
            return

        model = state.get("model")
        if model is None:
            return

        # Compute Taylor importance scores
        scores = self._compute_taylor_scores(model)
        if not scores:
            return

        # Sort by importance ascending — prune the least important
        scores.sort(key=lambda t: t[2])
        total_heads = len(scores)
        num_to_prune = max(1, int(total_heads * self.prune_ratio))
        heads_to_prune = [(name, head_idx) for name, head_idx, _ in scores[:num_to_prune]]

        # Zero out projections
        self._zero_head_projections(model, heads_to_prune)

        # Remove activation hooks
        for handle in self._act_hooks:
            handle.remove()
        self._act_hooks.clear()

        self._applied = True
        metrics["head_prune/pruned_count"] = num_to_prune
        metrics["head_prune/total_heads"] = total_heads
        metrics["head_prune/ratio_actual"] = num_to_prune / total_heads if total_heads > 0 else 0.0

    def on_train_end(self, state: dict[str, Any]) -> None:
        # Clean up any remaining hooks
        for handle in self._act_hooks:
            handle.remove()
        self._act_hooks.clear()


register_callback("pruning_attention_head", AttentionHeadPruningCallback, source="local")
