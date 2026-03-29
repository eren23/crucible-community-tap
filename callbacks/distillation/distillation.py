"""Knowledge distillation callback for torch_backend.

Loads a frozen teacher model and injects distillation loss into the
student's training via a forward hook. The combined loss is:

    L = alpha * T^2 * KL(student_soft || teacher_soft) + (1 - alpha) * task_loss

where soft distributions use temperature T.

Env vars:
    DISTILL_TEACHER_PATH:   Path to teacher state dict (required)
    DISTILL_TEMPERATURE:    Softmax temperature (default: 4.0)
    DISTILL_ALPHA:          Weight for distillation loss (default: 0.5)
    DISTILL_TEACHER_DEVICE: Device for teacher model (default: same as student)
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.callbacks import TrainingCallback, register_callback


class DistillationCallback(TrainingCallback):
    """Teacher-student knowledge distillation via forward hook."""

    priority = 15  # After pruning (8), before metrics (95)

    def __init__(self, **kwargs: Any) -> None:
        self.teacher_path = os.environ.get("DISTILL_TEACHER_PATH", "")
        self.temperature = float(os.environ.get("DISTILL_TEMPERATURE", "4.0"))
        self.alpha = float(os.environ.get("DISTILL_ALPHA", "0.5"))
        self.teacher_device = os.environ.get("DISTILL_TEACHER_DEVICE", "")
        self._teacher = None
        self._hook_handles: list = []

    def on_train_begin(self, state: dict[str, Any]) -> None:
        if not self.teacher_path:
            return

        model = state.get("model")
        if model is None:
            return

        self._teacher = self._load_teacher(model)
        if self._teacher is None:
            return

        state["teacher_model"] = self._teacher
        self._setup_logit_hooks(model, self._teacher, self.temperature, self.alpha)

    def _setup_logit_hooks(self, student, teacher, temperature, alpha):
        """Set up hooks on the output projection to capture logits for KD."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # Find the output projection (lm_head, output, head, etc.)
        lm_head = None
        lm_head_name = None
        for name, module in student.named_modules():
            if isinstance(module, nn.Linear) and any(
                kw in name for kw in ("lm_head", "output", "head", "out_proj")
            ):
                lm_head = module
                lm_head_name = name
                break

        if lm_head is None:
            # Fall back to last Linear layer
            for name, module in student.named_modules():
                if isinstance(module, nn.Linear):
                    lm_head_name, lm_head = name, module
            if lm_head is None:
                return

        # Find corresponding teacher layer
        teacher_lm_head = None
        for name, module in teacher.named_modules():
            if name == lm_head_name:
                teacher_lm_head = module
                break
        if teacher_lm_head is None:
            for name, module in teacher.named_modules():
                if isinstance(module, nn.Linear) and any(
                    kw in name for kw in ("lm_head", "output", "head", "out_proj")
                ):
                    teacher_lm_head = module
                    break
        if teacher_lm_head is None:
            return

        # Storage for captured hidden states
        _student_input: dict[str, torch.Tensor] = {}
        _teacher_input: dict[str, torch.Tensor] = {}

        def _capture_student_input(module, input):
            if input and isinstance(input[0], torch.Tensor):
                _student_input["hidden"] = input[0].detach()

        def _capture_teacher_input(module, input):
            if input and isinstance(input[0], torch.Tensor):
                _teacher_input["hidden"] = input[0].detach()

        h1 = lm_head.register_forward_pre_hook(_capture_student_input)
        h2 = teacher_lm_head.register_forward_pre_hook(_capture_teacher_input)

        # After-forward hook on the student model to add KD loss
        def _add_kd_loss(module, input, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 0:
                return output
            if "hidden" not in _student_input or "hidden" not in _teacher_input:
                return output

            s_hidden = _student_input["hidden"]
            t_hidden = _teacher_input["hidden"]

            # Compute logits from hidden states via the head weights
            with torch.no_grad():
                teacher_logits = F.linear(t_hidden, teacher_lm_head.weight)
            student_logits = F.linear(s_hidden, lm_head.weight)

            # KD loss: KL divergence on softened distributions
            soft_student = F.log_softmax(student_logits / temperature, dim=-1)
            soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
            kd_loss = F.kl_div(
                soft_student.reshape(-1, soft_student.size(-1)),
                soft_teacher.reshape(-1, soft_teacher.size(-1)),
                reduction="batchmean",
            ) * (temperature ** 2)

            # Combined loss
            return alpha * kd_loss + (1.0 - alpha) * output

        h3 = student.register_forward_hook(_add_kd_loss)
        self._hook_handles = [h1, h2, h3]

    def _load_teacher(self, student_model):
        """Build teacher by cloning student architecture and loading saved weights."""
        import copy
        import sys

        import torch

        try:
            teacher_state = torch.load(
                self.teacher_path, map_location="cpu", weights_only=True
            )
            teacher = copy.deepcopy(student_model)
            teacher.load_state_dict(teacher_state, strict=False)

            # Freeze
            teacher.requires_grad_(False)
            teacher.eval()

            # Device placement
            if self.teacher_device:
                teacher = teacher.to(self.teacher_device)
            else:
                device = next(student_model.parameters()).device
                teacher = teacher.to(device)

            return teacher
        except Exception as e:
            print(f"WARNING: Failed to load teacher model: {e}", file=sys.stderr)
            return None

    def on_train_end(self, state: dict[str, Any]) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        if self._teacher is not None:
            del self._teacher
            self._teacher = None
            state.pop("teacher_model", None)


register_callback("distillation", DistillationCallback, source="local")
