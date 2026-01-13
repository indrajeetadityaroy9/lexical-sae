"""
Knowledge Distillation for SPLADE models.

Enables training smaller, faster student models (DistilBERT)
from larger, more powerful teacher models (Mistral-SPLADE).

Key techniques:
1. Temperature-scaled soft targets
2. Feature-level distillation (sparse vector matching)
3. Hard negative mining during distillation

Reference:
    From Distillation to Hard Negative Sampling
    https://arxiv.org/abs/2205.04733
"""

from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DistillationConfig


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.

    Components:
    1. KL divergence between soft targets (temperature-scaled)
    2. Task loss (BCE for classification)
    3. Feature matching loss (MSE between sparse vectors)
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        feature_weight: float = 0.1,
        feature_distill: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.feature_distill = feature_distill

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.

        Args:
            student_logits: Student classification logits
            teacher_logits: Teacher classification logits
            labels: Ground truth labels
            student_features: Student sparse vectors (optional)
            teacher_features: Teacher sparse vectors (optional)

        Returns:
            Dictionary with individual losses and total
        """
        # 1. KL Divergence Loss (soft targets)
        # For binary classification, use sigmoid; for multi-class use softmax
        soft_teacher = torch.sigmoid(teacher_logits / self.temperature)
        soft_student = torch.sigmoid(student_logits / self.temperature)

        # Binary KL divergence
        kl_loss = F.binary_cross_entropy(
            soft_student,
            soft_teacher.detach(),
            reduction="mean"
        )
        # Scale by temperature squared (Hinton et al.)
        kl_loss = kl_loss * (self.temperature ** 2)

        # 2. Task Loss (hard targets)
        task_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            labels,
            reduction="mean"
        )

        # 3. Feature Matching Loss (optional)
        if self.feature_distill and student_features is not None and teacher_features is not None:
            # Normalize features for stable matching
            student_norm = F.normalize(student_features, p=2, dim=-1)
            teacher_norm = F.normalize(teacher_features, p=2, dim=-1)

            feature_loss = F.mse_loss(student_norm, teacher_norm.detach())
        else:
            feature_loss = torch.tensor(0.0, device=student_logits.device)

        # Combined loss
        total_loss = (
            self.alpha * kl_loss +
            (1 - self.alpha) * task_loss +
            self.feature_weight * feature_loss
        )

        return {
            "total": total_loss,
            "kl_loss": kl_loss,
            "task_loss": task_loss,
            "feature_loss": feature_loss
        }


class DistillationTrainer:
    """
    Trainer for knowledge distillation.

    Distills knowledge from a large teacher model (e.g., Mistral-SPLADE)
    to a smaller student model (e.g., DistilBERT-SPLADE).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
        device: Optional[torch.device] = None
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.teacher.to(self.device)
        self.student.to(self.device)

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Loss function
        self.criterion = DistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha,
            feature_weight=config.feature_weight if config.feature_distill else 0.0,
            feature_distill=config.feature_distill
        )

        self.history: List[Dict[str, float]] = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()

        total_loss = 0.0
        total_kl = 0.0
        total_task = 0.0
        total_feature = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].unsqueeze(1).to(self.device)

            # Get teacher predictions (no grad)
            with torch.no_grad():
                teacher_logits, teacher_features = self.teacher(
                    input_ids, attention_mask
                )

            # Get student predictions
            optimizer.zero_grad()
            student_logits, student_features = self.student(
                input_ids, attention_mask
            )

            # Compute loss
            losses = self.criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_features=student_features,
                teacher_features=teacher_features
            )

            # Backward
            losses["total"].backward()
            optimizer.step()

            # Track metrics
            total_loss += losses["total"].item()
            total_kl += losses["kl_loss"].item()
            total_task += losses["task_loss"].item()
            total_feature += losses["feature_loss"].item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "kl": f"{losses['kl_loss'].item():.4f}",
                "task": f"{losses['task_loss'].item():.4f}"
            })

        metrics = {
            "loss": total_loss / num_batches,
            "kl_loss": total_kl / num_batches,
            "task_loss": total_task / num_batches,
            "feature_loss": total_feature / num_batches
        }

        self.history.append(metrics)
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 2e-5,
        weight_decay: float = 0.01
    ) -> List[Dict[str, float]]:
        """
        Full distillation training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: AdamW weight decay

        Returns:
            Training history
        """
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, optimizer, epoch)

            log_str = (
                f"Epoch {epoch}: "
                f"loss={metrics['loss']:.4f}, "
                f"kl={metrics['kl_loss']:.4f}, "
                f"task={metrics['task_loss']:.4f}"
            )

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                log_str += f", val_acc={val_metrics['accuracy']:.4f}"

            print(log_str)

        return self.history

    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate student model."""
        self.student.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits, _ = self.student(input_ids, attention_mask)
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {"accuracy": correct / total if total > 0 else 0.0}

    def save_student(self, path: str):
        """Save distilled student model."""
        torch.save({
            "model_state_dict": self.student.state_dict(),
            "config": self.config.__dict__,
            "history": self.history
        }, path)

    @classmethod
    def load_student(
        cls,
        path: str,
        student_model: nn.Module,
        device: Optional[torch.device] = None
    ) -> nn.Module:
        """Load distilled student model."""
        checkpoint = torch.load(path, map_location=device)
        student_model.load_state_dict(checkpoint["model_state_dict"])
        return student_model


def create_teacher_labels(
    teacher: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> torch.Tensor:
    """
    Pre-compute teacher soft labels for offline distillation.

    Useful when teacher is too large to run during training.

    Args:
        teacher: Teacher model
        dataloader: Data loader
        device: Device

    Returns:
        Tensor of teacher logits [num_samples, num_classes]
    """
    teacher.eval()
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing teacher labels"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits, _ = teacher(input_ids, attention_mask)
            all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)
