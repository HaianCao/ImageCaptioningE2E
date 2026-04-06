"""
Trainer for object classification.

This trainer is intentionally single-purpose:
- one object classifier
- one loss
- one validation metric stream

That separation keeps the notebook pipeline readable and makes it easier to
load the resulting checkpoint later into an E2E wrapper.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import BaseTrainer
from ..evaluation import compute_classification_metrics
from ..models.object import ObjectClassifier
from ..utils import CheckpointManager
from ..utils.class_balance import compute_single_label_class_weights
from ..utils.memory import cleanup_cuda_memory


class ObjectTrainer(BaseTrainer):
    """Train an object classifier either on ROI features or on ROI images."""

    def __init__(
        self,
        model: ObjectClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_manager: Optional[CheckpointManager] = None,
        use_auto_class_weights: bool = True,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0,
        **kwargs,
    ):
        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir="checkpoints",
                experiment_name="object",
            )

        super().__init__(
            model,
            train_loader,
            val_loader,
            optimizer,
            checkpoint_manager=checkpoint_manager,
            **kwargs,
        )

        self.object_model = model
        self.use_auto_class_weights = use_auto_class_weights
        self.class_weights: Optional[torch.Tensor] = None
        self.freeze_backbone = bool(freeze_backbone)
        self.freeze_epochs = max(0, int(freeze_epochs))
        self._backbone_frozen = False

        if self.use_auto_class_weights:
            self.class_weights = compute_single_label_class_weights(
                self.train_loader.dataset,
                num_classes=self.object_model.num_classes,
                label_key="object_label",
                power=0.5,
                clip_min=0.25,
                clip_max=10.0,
                device=self.device,
            )

        if self.class_weights is not None:
            self.logger.info(
                f"ObjectTrainer auto class balancing enabled | weight mean={self.class_weights.mean().item():.3f}"
            )

        if hasattr(self.object_model, "freeze_backbone") and (self.freeze_backbone or self.freeze_epochs > 0):
            self.object_model.freeze_backbone()
            self._backbone_frozen = True
            self.logger.info(
                f"Object backbone frozen at start | freeze_backbone={self.freeze_backbone}, freeze_epochs={self.freeze_epochs}"
            )

    def train(self) -> Dict[str, Any]:
        try:
            return super().train()
        finally:
            cleanup_cuda_memory(note="Object training finished")

    def _get_inputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch.get("feature")
        if features is None:
            features = batch.get("object_feature")
        if features is None:
            features = batch.get("image")
        if features is None:
            features = batch.get("object_image")
        if features is None:
            raise KeyError("ObjectTrainer requires a 'feature' or 'image' tensor in the batch")
        return features

    def _update_backbone_state(self) -> None:
        if not hasattr(self.object_model, "freeze_backbone"):
            return

        if self._backbone_frozen:
            if self.freeze_epochs > 0 and self.current_epoch >= self.freeze_epochs:
                self.object_model.unfreeze_backbone()
                self._backbone_frozen = False
                self.logger.info(f"Object backbone unfrozen at epoch {self.current_epoch}")
            else:
                self.object_model.freeze_backbone()

    def _train_epoch(self) -> Dict[str, float]:
        self.object_model.train()
        self._update_backbone_state()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    loss = self._compute_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss = self._compute_loss(batch)
                loss.backward()

            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.object_model.parameters(), self.gradient_clip_val)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if self.log_every_n_steps > 0 and batch_idx % self.log_every_n_steps == 0:
                self._log_batch(batch, loss.item(), batch_idx)

        return {
            "train_loss": epoch_loss / max(num_batches, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self._get_inputs(batch)
        logits = self.object_model(features)
        targets = batch["object_label"]
        return nn.functional.cross_entropy(logits, targets, weight=self.class_weights)

    def _get_predictions_and_targets(self, batch: Dict[str, torch.Tensor]) -> tuple:
        features = self._get_inputs(batch)

        with torch.no_grad():
            logits = self.object_model(features)

        preds = logits.argmax(dim=1)
        targets = batch["object_label"]
        return preds, targets

    def _get_metrics(self, all_preds: List[torch.Tensor], all_targets: List[torch.Tensor]) -> Dict[str, float]:
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        return compute_classification_metrics(preds, targets, num_classes=self.object_model.num_classes)

    def _log_batch(self, batch: Dict[str, torch.Tensor], loss: float, batch_idx: int):
        self.logger.log_metrics(
            {
                "batch_loss": loss,
                "batch_idx": batch_idx,
            },
            self.global_step,
            prefix="batch",
        )

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                model=self.object_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=metrics.get("train_loss", 0),
                metric=metrics.get("f1", 0),
                task="object",
                filename=f"object_epoch_{epoch}.pth",
            )