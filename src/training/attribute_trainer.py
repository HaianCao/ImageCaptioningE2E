"""
Trainer for attribute classification.

This trainer mirrors the object trainer but uses multi-label BCE and the
attribute metrics that are meaningful for sparse label vectors.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .trainer import BaseTrainer
from ..evaluation import compute_multilabel_metrics
from ..models.attribute import AttributeClassifier
from ..utils import CheckpointManager
from ..utils.class_balance import compute_multilabel_pos_weight
from ..utils.memory import cleanup_cuda_memory


class AttributeTrainer(BaseTrainer):
    """Train an attribute classifier either on ROI features or on ROI images."""

    def __init__(
        self,
        model: AttributeClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoint_manager: Optional[CheckpointManager] = None,
        use_auto_class_weights: bool = True,
        attribute_threshold_mode: str = "fixed",
        attribute_threshold: float = 0.5,
        attribute_threshold_scale: float = 0.5,
        attribute_threshold_min: float = 0.05,
        attribute_threshold_max: float = 0.95,
        freeze_backbone: bool = False,
        freeze_epochs: int = 0,
        **kwargs,
    ):
        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir="checkpoints",
                experiment_name="attribute",
            )

        super().__init__(
            model,
            train_loader,
            val_loader,
            optimizer,
            checkpoint_manager=checkpoint_manager,
            **kwargs,
        )

        self.attribute_model = model
        self.use_auto_class_weights = use_auto_class_weights
        self.pos_weight: Optional[torch.Tensor] = None
        self.attribute_threshold_mode = attribute_threshold_mode
        self.attribute_threshold = attribute_threshold
        self.attribute_threshold_scale = attribute_threshold_scale
        self.attribute_threshold_min = attribute_threshold_min
        self.attribute_threshold_max = attribute_threshold_max
        self.freeze_backbone = bool(freeze_backbone)
        self.freeze_epochs = max(0, int(freeze_epochs))
        self._backbone_frozen = False

        if self.use_auto_class_weights:
            self.pos_weight = compute_multilabel_pos_weight(
                self.train_loader.dataset,
                num_labels=self.attribute_model.num_attributes,
                label_key="attribute_labels",
                power=0.5,
                clip_min=1.0,
                clip_max=20.0,
                device=self.device,
            )

        if self.pos_weight is not None:
            self.logger.info(
                f"AttributeTrainer auto class balancing enabled | pos_weight mean={self.pos_weight.mean().item():.3f}"
            )

        if hasattr(self.attribute_model, "freeze_backbone") and (self.freeze_backbone or self.freeze_epochs > 0):
            self.attribute_model.freeze_backbone()
            self._backbone_frozen = True
            self.logger.info(
                f"Attribute backbone frozen at start | freeze_backbone={self.freeze_backbone}, freeze_epochs={self.freeze_epochs}"
            )

    def train(self) -> Dict[str, Any]:
        try:
            return super().train()
        finally:
            cleanup_cuda_memory(note="Attribute training finished")

    def _get_inputs(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = batch.get("feature")
        if features is None:
            features = batch.get("attribute_feature")
        if features is None:
            features = batch.get("image")
        if features is None:
            features = batch.get("attribute_image")
        if features is None:
            raise KeyError("AttributeTrainer requires a 'feature' or 'image' tensor in the batch")
        return features

    def _update_backbone_state(self) -> None:
        if not hasattr(self.attribute_model, "freeze_backbone"):
            return

        if self._backbone_frozen:
            if self.freeze_epochs > 0 and self.current_epoch >= self.freeze_epochs:
                self.attribute_model.unfreeze_backbone()
                self._backbone_frozen = False
                self.logger.info(f"Attribute backbone unfrozen at epoch {self.current_epoch}")
            else:
                self.attribute_model.freeze_backbone()

    def _train_epoch(self) -> Dict[str, float]:
        self.attribute_model.train()
        self._update_backbone_state()

        epoch_loss = 0.0
        num_batches = 0

        train_loader = self._progress(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs} [train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(train_loader):
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
                torch.nn.utils.clip_grad_norm_(self.attribute_model.parameters(), self.gradient_clip_val)

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
        logits = self.attribute_model(features)
        targets = batch["attribute_labels"]

        if self.pos_weight is not None:
            return nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets.float(),
                pos_weight=self.pos_weight,
            )

        return nn.functional.binary_cross_entropy_with_logits(logits, targets.float())

    def _get_predictions_and_targets(self, batch: Dict[str, torch.Tensor]) -> tuple:
        features = self._get_inputs(batch)

        with torch.no_grad():
            logits = self.attribute_model(features)

        return {
            "attribute_logits": logits,
            "attribute_targets": batch["attribute_labels"],
        }, None

    def _get_metrics(self, all_preds: List[Dict], all_targets: List) -> Dict[str, float]:
        attribute_logits = torch.cat([p["attribute_logits"] for p in all_preds])
        attribute_targets = torch.cat([p["attribute_targets"] for p in all_preds])

        metrics = compute_multilabel_metrics(
            attribute_logits,
            attribute_targets,
            threshold=self.attribute_threshold,
            threshold_mode=self.attribute_threshold_mode,
            threshold_scale=self.attribute_threshold_scale,
            threshold_min=self.attribute_threshold_min,
            threshold_max=self.attribute_threshold_max,
        )

        return metrics

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
                model=self.attribute_model,
                optimizer=self.optimizer,
                epoch=epoch,
                loss=metrics.get("train_loss", 0),
                metric=metrics.get("micro_f1", 0),
                task="attribute",
                filename=f"attribute_epoch_{epoch}.pth",
            )