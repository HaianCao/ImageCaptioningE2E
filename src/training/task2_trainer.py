"""
Task 2 Trainer: Relationship Classification.

Trains relationship classifier using union features and spatial information.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List

from .trainer import BaseTrainer
from ..models.task2 import RelationClassifier
from ..evaluation import compute_classification_metrics
from ..utils import CheckpointManager
from ..utils.memory import cleanup_cuda_memory


class Task2Trainer(BaseTrainer):
    """
    Trainer for Task 2: Relationship classification.

    Trains a single model that predicts relationships between object pairs.
    """

    def __init__(
        self,
        model: RelationClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        label_smoothing: float = 0.0,
        checkpoint_manager: Optional[CheckpointManager] = None,
        **kwargs
    ):
        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir="checkpoints",
                experiment_name="task2",
            )

        super().__init__(
            model,
            train_loader,
            val_loader,
            optimizer,
            checkpoint_manager=checkpoint_manager,
            **kwargs,
        )

        self.label_smoothing = label_smoothing

    def train(self) -> Dict[str, Any]:
        try:
            return super().train()
        finally:
            cleanup_cuda_memory(note="Task 2 training finished")

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for relationship classification."""
        features = batch['feature'] if 'feature' in batch else batch['image']
        spatial = batch['spatial']
        targets = batch['relation_label']

        # Forward pass
        logits = self.model(features, spatial)

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)

        return loss

    def _get_predictions_and_targets(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Get predictions and targets for metrics computation."""
        features = batch['feature'] if 'feature' in batch else batch['image']
        spatial = batch['spatial']

        with torch.no_grad():
            logits = self.model(features, spatial)

        preds = logits.argmax(dim=1)
        targets = batch['relation_label']

        return preds, targets

    def _get_metrics(self, all_preds: List[torch.Tensor], all_targets: List[torch.Tensor]) -> Dict[str, float]:
        """Compute validation metrics for Task 2."""
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        metrics = compute_classification_metrics(
            preds, targets, num_classes=self.model.num_relations
        )

        return metrics

    def _log_batch(self, batch: Dict[str, torch.Tensor], loss: float, batch_idx: int):
        """Log batch-level information for Task 2."""
        self.logger.log_metrics({
            'batch_loss': loss,
            'batch_idx': batch_idx
        }, self.global_step, prefix="batch")

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint for Task 2 model."""
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=metrics.get('train_loss', 0),
                metric=metrics.get('f1', 0),
                task="task2",
                filename=f"task2_epoch_{epoch}.pth"
            )