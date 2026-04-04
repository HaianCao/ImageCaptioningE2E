"""
Task 1 Trainer: Object & Attribute Classification.

Trains object classifier and attribute classifier jointly.
Loss = object_loss + attribute_loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List

from .trainer import BaseTrainer
from ..models.task1 import ObjectClassifier, AttributeClassifier
from ..evaluation import compute_classification_metrics, compute_multilabel_metrics


class Task1Trainer(BaseTrainer):
    """
    Trainer for Task 1: Object and attribute classification.

    Trains two models jointly:
    - Object classifier (single-label)
    - Attribute classifier (multi-label)
    """

    def __init__(
        self,
        object_model: ObjectClassifier,
        attribute_model: AttributeClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        object_optimizer: torch.optim.Optimizer,
        attribute_optimizer: torch.optim.Optimizer,
        object_weight: float = 1.0,
        attribute_weight: float = 1.0,
        attribute_pos_weight: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # Initialize with object model as primary model
        super().__init__(object_model, train_loader, val_loader, object_optimizer, **kwargs)

        self.object_model = object_model
        self.attribute_model = attribute_model
        self.attribute_optimizer = attribute_optimizer

        # Loss weights
        self.object_weight = object_weight
        self.attribute_weight = attribute_weight

        # For imbalanced attributes
        self.attribute_pos_weight = attribute_pos_weight
        if self.attribute_pos_weight is not None:
            self.attribute_pos_weight = self.attribute_pos_weight.to(self.device)

        # Move attribute model to device
        self.attribute_model.to(self.device)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute combined loss for object + attribute classification."""
        features = batch['feature'] if 'feature' in batch else batch['image']

        # Object classification loss
        object_logits = self.object_model(features)
        object_targets = batch['object_label']
        object_loss = nn.functional.cross_entropy(object_logits, object_targets)

        # Attribute classification loss (multi-label)
        attribute_logits = self.attribute_model(features)
        attribute_targets = batch['attribute_labels']

        if self.attribute_pos_weight is not None:
            # Weighted BCE for imbalanced classes
            attribute_loss = nn.functional.binary_cross_entropy_with_logits(
                attribute_logits, attribute_targets.float(),
                pos_weight=self.attribute_pos_weight
            )
        else:
            attribute_loss = nn.functional.binary_cross_entropy_with_logits(
                attribute_logits, attribute_targets.float()
            )

        # Combined loss
        total_loss = self.object_weight * object_loss + self.attribute_weight * attribute_loss

        return total_loss

    def _get_predictions_and_targets(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Get predictions and targets for metrics computation."""
        features = batch['feature'] if 'feature' in batch else batch['image']

        # Object predictions
        with torch.no_grad():
            object_logits = self.object_model(features)
            attribute_logits = self.attribute_model(features)

        object_preds = object_logits.argmax(dim=1)
        attribute_preds = (attribute_logits.sigmoid() > 0.5).long()

        object_targets = batch['object_label']
        attribute_targets = batch['attribute_labels']

        return {
            'object_preds': object_preds,
            'object_targets': object_targets,
            'attribute_preds': attribute_preds,
            'attribute_targets': attribute_targets
        }, None  # targets already included in preds dict

    def _get_metrics(self, all_preds: List[Dict], all_targets: List) -> Dict[str, float]:
        """Compute validation metrics for Task 1."""
        # Aggregate predictions
        object_preds = torch.cat([p['object_preds'] for p in all_preds])
        object_targets = torch.cat([p['object_targets'] for p in all_preds])
        attribute_preds = torch.cat([p['attribute_preds'] for p in all_preds])
        attribute_targets = torch.cat([p['attribute_targets'] for p in all_preds])

        # Object metrics
        object_metrics = compute_classification_metrics(
            object_preds, object_targets, num_classes=self.object_model.num_classes
        )

        # Attribute metrics (multi-label)
        attribute_metrics = compute_multilabel_metrics(
            attribute_preds.float(), attribute_targets.float()
        )

        # Combine metrics with prefixes
        metrics = {}
        for key, value in object_metrics.items():
            metrics[f'object_{key}'] = value
        for key, value in attribute_metrics.items():
            metrics[f'attribute_{key}'] = value

        return metrics

    def _log_batch(self, batch: Dict[str, torch.Tensor], loss: float, batch_idx: int):
        """Log batch-level information for Task 1."""
        # Compute individual losses for logging
        features = batch['feature'] if 'feature' in batch else batch['image']

        with torch.no_grad():
            object_logits = self.object_model(features)
            attribute_logits = self.attribute_model(features)

        object_loss = nn.functional.cross_entropy(
            object_logits, batch['object_label']
        ).item()

        attribute_loss = nn.functional.binary_cross_entropy_with_logits(
            attribute_logits, batch['attribute_labels'].float()
        ).item()

        self.logger.log_metrics({
            'batch_loss': loss,
            'batch_object_loss': object_loss,
            'batch_attribute_loss': attribute_loss,
            'batch_idx': batch_idx
        }, self.global_step, prefix="batch")

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoints for both models."""
        if self.checkpoint_manager:
            # Save object model
            self.checkpoint_manager.save_checkpoint(
                model=self.object_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=metrics.get('train_loss', 0),
                metric=metrics.get('object_f1', 0),
                task="task1_object",
                filename=f"task1_object_epoch_{epoch}.pth"
            )

            # Save attribute model
            self.checkpoint_manager.save_checkpoint(
                model=self.attribute_model,
                optimizer=self.attribute_optimizer,
                epoch=epoch,
                loss=metrics.get('train_loss', 0),
                metric=metrics.get('attribute_f1_macro', 0),
                task="task1_attribute",
                filename=f"task1_attribute_epoch_{epoch}.pth"
            )

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Resume training from checkpoint for both models."""
        metadata = {}

        if self.checkpoint_manager:
            # Load object model
            obj_metadata = self.checkpoint_manager.load_checkpoint(
                self.object_model, self.optimizer, self.scheduler,
                checkpoint_path or "task1_object_best.pth"
            )

            # Load attribute model
            attr_metadata = self.checkpoint_manager.load_checkpoint(
                self.attribute_model, self.attribute_optimizer, None,
                checkpoint_path or "task1_attribute_best.pth"
            )

            metadata = {**obj_metadata, **attr_metadata}
            self.current_epoch = metadata.get('epoch', 0)

        return metadata