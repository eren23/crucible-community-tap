"""Knowledge distillation objective for generic_backend.

Wraps a teacher model and computes the combined loss:
    L = alpha * T^2 * KL(student_soft || teacher_soft) + (1 - alpha) * task_loss

Env vars:
    DISTILL_TEACHER_PATH:   Path to teacher state dict (required)
    DISTILL_TEACHER_FAMILY: Model family for teacher architecture (default: same as student)
    DISTILL_TEMPERATURE:    Softmax temperature (default: 4.0)
    DISTILL_ALPHA:          Weight for distillation loss (default: 0.5)
    DISTILL_LOSS_TYPE:      Loss type: kl | mse | cosine (default: kl)
"""
from __future__ import annotations

import os
from typing import Any

from crucible.training.objectives import TrainingObjective, register_objective


class DistillationObjective(TrainingObjective):
    """Knowledge distillation with configurable loss type."""

    name = "distillation"

    def __init__(self, **kwargs: Any) -> None:
        self.teacher_path = os.environ.get("DISTILL_TEACHER_PATH", "")
        self.temperature = float(os.environ.get("DISTILL_TEMPERATURE", "4.0"))
        self.alpha = float(os.environ.get("DISTILL_ALPHA", "0.5"))
        self.loss_type = os.environ.get("DISTILL_LOSS_TYPE", "kl")
        self._teacher = None
        self._teacher_loaded = False

    def _ensure_teacher(self, predictions: dict[str, Any]) -> None:
        """Lazy-load teacher on first compute call."""
        if self._teacher_loaded:
            return
        self._teacher_loaded = True

        if not self.teacher_path:
            return

        import torch

        # If predictions contain teacher_logits (pre-computed), skip loading
        if "teacher_logits" in predictions:
            return

        try:
            self._teacher = torch.load(
                self.teacher_path, map_location="cpu", weights_only=True
            )
        except Exception:
            self._teacher = None

    def compute(
        self, predictions: dict[str, Any], targets: dict[str, Any]
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        self._ensure_teacher(predictions)

        student_logits = predictions.get("logits")
        if student_logits is None:
            # Fallback: if model returns embeddings instead of logits
            # compute MSE on embeddings
            pred_emb = predictions.get("pred_embeddings")
            target_emb = targets.get("target_embeddings")
            if pred_emb is not None and target_emb is not None:
                loss = F.mse_loss(pred_emb, target_emb)
                return {"loss": loss, "pred_loss": loss}
            raise KeyError("DistillationObjective requires 'logits' or 'pred_embeddings' in predictions")

        # Task loss (standard cross-entropy)
        labels = targets.get("labels", targets.get("target_ids"))
        if labels is not None:
            if student_logits.dim() > 2:
                task_loss = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            else:
                task_loss = F.cross_entropy(student_logits, labels, ignore_index=-100)
        else:
            task_loss = torch.tensor(0.0, device=student_logits.device)

        # Teacher logits (pre-computed or from stored teacher)
        teacher_logits = predictions.get("teacher_logits")
        if teacher_logits is None:
            # No teacher available — return task loss only
            return {"loss": task_loss, "task_loss": task_loss, "distill_loss": torch.tensor(0.0)}

        # Distillation loss
        T = self.temperature
        if self.loss_type == "kl":
            soft_student = F.log_softmax(student_logits / T, dim=-1)
            soft_teacher = F.softmax(teacher_logits / T, dim=-1)
            distill_loss = F.kl_div(
                soft_student.reshape(-1, soft_student.size(-1)),
                soft_teacher.reshape(-1, soft_teacher.size(-1)),
                reduction="batchmean",
            ) * (T ** 2)
        elif self.loss_type == "mse":
            distill_loss = F.mse_loss(student_logits, teacher_logits)
        elif self.loss_type == "cosine":
            distill_loss = 1.0 - F.cosine_similarity(
                student_logits.reshape(-1, student_logits.size(-1)),
                teacher_logits.reshape(-1, teacher_logits.size(-1)),
                dim=-1,
            ).mean()
        else:
            distill_loss = torch.tensor(0.0, device=student_logits.device)

        combined = self.alpha * distill_loss + (1.0 - self.alpha) * task_loss
        return {
            "loss": combined,
            "task_loss": task_loss,
            "distill_loss": distill_loss,
        }

    def metric_names(self) -> list[str]:
        return ["task_loss", "distill_loss"]


register_objective("distillation", DistillationObjective, source="local")
