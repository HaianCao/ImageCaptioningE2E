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
from ..utils import CheckpointManager
from ..utils.class_balance import compute_single_label_class_weights, compute_multilabel_pos_weight
from ..utils.memory import cleanup_cuda_memory


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
        checkpoint_manager: Optional[CheckpointManager] = None,
        attribute_checkpoint_manager: Optional[CheckpointManager] = None,
        use_auto_class_weights: bool = True,
        *,
        attribute_threshold_mode: str = "adaptive_mean_std",
        attribute_threshold: float = 0.5,
        attribute_threshold_scale: float = 0.5,
        attribute_threshold_min: float = 0.05,
        attribute_threshold_max: float = 0.95,
        **kwargs
    ):
        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir="checkpoints",
                experiment_name="task1_object",
            )

        # Initialize with object model as primary model
        super().__init__(
            object_model,
            train_loader,
            val_loader,
            object_optimizer,
            checkpoint_manager=checkpoint_manager,
            **kwargs,
        )

        self.object_model = object_model
        self.attribute_model = attribute_model
        self.attribute_optimizer = attribute_optimizer

        if attribute_checkpoint_manager is None:
            attribute_checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(self.checkpoint_manager.checkpoint_dir.parent),
                experiment_name="task1_attribute",
                max_checkpoints=self.checkpoint_manager.max_checkpoints,
                save_best_only=self.checkpoint_manager.save_best_only,
            )

        self.attribute_checkpoint_manager = attribute_checkpoint_manager

        # Loss weights
        self.object_weight = object_weight
        self.attribute_weight = attribute_weight

        # Automatic class balancing derived from the training split.
        self.use_auto_class_weights = use_auto_class_weights
        self.object_class_weight = None
        self.attribute_pos_weight = None

        if self.use_auto_class_weights:
            self.object_class_weight = compute_single_label_class_weights(
                self.train_loader.dataset,
                num_classes=self.object_model.num_classes,
                label_key="object_label",
                power=0.5,
                clip_min=0.25,
                clip_max=10.0,
                device=self.device,
            )
            self.attribute_pos_weight = compute_multilabel_pos_weight(
                self.train_loader.dataset,
                num_labels=self.attribute_model.num_attributes,
                label_key="attribute_labels",
                power=0.5,
                clip_min=1.0,
                clip_max=20.0,
                device=self.device,
            )
        else:
            if attribute_pos_weight is not None:
                self.attribute_pos_weight = attribute_pos_weight.to(self.device)

        if self.object_class_weight is not None:
            self.logger.info(
                f"Task 1 auto class balancing enabled | object weight mean={self.object_class_weight.mean().item():.3f}"
            )
        if self.attribute_pos_weight is not None:
            self.logger.info(
                f"Task 1 auto attribute balancing enabled | pos_weight mean={self.attribute_pos_weight.mean().item():.3f}"
            )
            if attribute_pos_weight is not None and self.use_auto_class_weights:
                self.logger.info("Task 1 manual attribute_pos_weight input is ignored because auto balancing is enabled.")

        # Attribute evaluation thresholds
        self.attribute_threshold_mode = attribute_threshold_mode
        self.attribute_threshold = attribute_threshold
        self.attribute_threshold_scale = attribute_threshold_scale
        self.attribute_threshold_min = attribute_threshold_min
        self.attribute_threshold_max = attribute_threshold_max

        # Move attribute model to device
        self.attribute_model.to(self.device)

    def train(self) -> Dict[str, Any]:
        try:
            return super().train()
        finally:
            cleanup_cuda_memory(note="Task 1 training finished")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with separate optimizers for object and attribute heads."""
        self.object_model.train()
        self.attribute_model.train()

        epoch_loss = 0.0
        num_batches = 0
        combined_params = list(self.object_model.parameters()) + list(self.attribute_model.parameters())

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()
            self.attribute_optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    loss = self._compute_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.unscale_(self.attribute_optimizer)
            else:
                loss = self._compute_loss(batch)
                loss.backward()

            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(combined_params, self.gradient_clip_val)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.step(self.attribute_optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                self.attribute_optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.log_every_n_steps > 0 and batch_idx % self.log_every_n_steps == 0:
                self._log_batch(batch, loss.item(), batch_idx)

        return {
            'train_loss': epoch_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute combined loss for object + attribute classification."""
        features = batch['feature'] if 'feature' in batch else batch['image']

        # Object classification loss
        object_logits = self.object_model(features)
        object_targets = batch['object_label']
        object_loss = nn.functional.cross_entropy(
            object_logits,
            object_targets,
            weight=self.object_class_weight,
        )

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

        object_targets = batch['object_label']
        attribute_targets = batch['attribute_labels']

        return {
            'object_preds': object_preds,
            'object_targets': object_targets,
            'attribute_logits': attribute_logits,
            'attribute_targets': attribute_targets
        }, None  # targets already included in preds dict

    def _get_metrics(self, all_preds: List[Dict], all_targets: List) -> Dict[str, float]:
        """Compute validation metrics for Task 1."""
        # Aggregate predictions
        object_preds = torch.cat([p['object_preds'] for p in all_preds])
        object_targets = torch.cat([p['object_targets'] for p in all_preds])
        attribute_logits = torch.cat([p['attribute_logits'] for p in all_preds])
        attribute_targets = torch.cat([p['attribute_targets'] for p in all_preds])

        # Object metrics
        object_metrics = compute_classification_metrics(
            object_preds, object_targets, num_classes=self.object_model.num_classes
        )

        # Attribute metrics (multi-label)
        attribute_metrics = compute_multilabel_metrics(
            attribute_logits,
            attribute_targets,
            threshold=self.attribute_threshold,
            threshold_mode=self.attribute_threshold_mode,
            threshold_scale=self.attribute_threshold_scale,
            threshold_min=self.attribute_threshold_min,
            threshold_max=self.attribute_threshold_max,
        )

        # Combine metrics with prefixes
        metrics = {}
        for key, value in object_metrics.items():
            metrics[f'object_{key}'] = value

        if 'exact_match_accuracy' in attribute_metrics:
            metrics['attribute_exact_match_accuracy'] = attribute_metrics['exact_match_accuracy']

        if 'sample_accuracy_mean' in attribute_metrics:
            metrics['attribute_sample_accuracy_mean'] = attribute_metrics['sample_accuracy_mean']

        if 'sample_accuracy_variance' in attribute_metrics:
            metrics['attribute_sample_accuracy_variance'] = attribute_metrics['sample_accuracy_variance']

        if 'sample_f1' in attribute_metrics:
            metrics['attribute_sample_f1'] = attribute_metrics['sample_f1']

        if 'jaccard_index' in attribute_metrics:
            metrics['attribute_jaccard_index'] = attribute_metrics['jaccard_index']

        if 'hamming_loss' in attribute_metrics:
            metrics['attribute_hamming_loss'] = attribute_metrics['hamming_loss']

        return metrics

    def _log_batch(self, batch: Dict[str, torch.Tensor], loss: float, batch_idx: int):
        """Log batch-level information for Task 1."""
        # Compute individual losses for logging
        features = batch['feature'] if 'feature' in batch else batch['image']

        with torch.no_grad():
            object_logits = self.object_model(features)
            attribute_logits = self.attribute_model(features)

        object_loss = nn.functional.cross_entropy(
            object_logits,
            batch['object_label'],
            weight=self.object_class_weight,
        ).item()

        attribute_loss = nn.functional.binary_cross_entropy_with_logits(
            attribute_logits,
            batch['attribute_labels'].float(),
            pos_weight=self.attribute_pos_weight,
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
            self.attribute_checkpoint_manager.save_checkpoint(
                model=self.attribute_model,
                optimizer=self.attribute_optimizer,
                epoch=epoch,
                loss=metrics.get('train_loss', 0),
                metric=metrics.get('attribute_sample_f1', 0),
                task="task1_attribute",
                filename=f"task1_attribute_epoch_{epoch}.pth"
            )

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Resume training from checkpoint for both models."""
        metadata = {}

        if self.checkpoint_manager:
            # Load object model
            object_checkpoint_path = checkpoint_path or self.checkpoint_manager.get_best_checkpoint_path()
            obj_metadata = {}
            if object_checkpoint_path:
                obj_metadata = self.checkpoint_manager.load_checkpoint(
                    self.object_model,
                    self.optimizer,
                    self.scheduler,
                    object_checkpoint_path,
                    load_best=False,
                )

            # Load attribute model
            attr_checkpoint_path = self.attribute_checkpoint_manager.get_best_checkpoint_path()
            attr_metadata = {}
            if attr_checkpoint_path:
                attr_metadata = self.attribute_checkpoint_manager.load_checkpoint(
                    self.attribute_model,
                    self.attribute_optimizer,
                    None,
                    attr_checkpoint_path,
                    load_best=False,
                )

            metadata = {**obj_metadata, **{f"attribute_{k}": v for k, v in attr_metadata.items()}}
            self.current_epoch = metadata.get('epoch', 0)

        return metadata