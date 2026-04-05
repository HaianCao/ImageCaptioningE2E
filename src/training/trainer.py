"""
Base trainer class for Visual Genome project.

Provides common training functionality:
- Training/validation loops
- Early stopping
- Checkpoint saving
- Gradient clipping
- Mixed precision training
- Logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod

from ..utils import Logger, CheckpointManager
from ..evaluation import compute_classification_metrics, compute_multilabel_metrics


class BaseTrainer(ABC):
    """
    Abstract base trainer with common training functionality.

    Subclasses should implement:
    - _compute_loss(): Compute loss for a batch
    - _get_metrics(): Compute validation metrics
    - _log_batch(): Log batch-level information
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        logger: Optional[Logger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        # Training config
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        gradient_clip_val: float = 1.0,
        use_amp: bool = False,  # Mixed precision
        log_every_n_steps: int = 50,
        save_every_n_epochs: int = 1,
        # Validation
        validate_every_n_epochs: int = 1,
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",  # "min" or "max"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Logging and checkpointing
        self.logger = logger or Logger()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()

        # Training config
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val
        self.use_amp = use_amp
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs

        # Validation config
        self.validate_every_n_epochs = validate_every_n_epochs
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric_value = float('inf') if monitor_mode == 'min' else float('-inf')
        self.early_stopping_counter = 0
        self.should_stop = False

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Move model to device
        self.model.to(device)

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Dict with training statistics
        """
        self.logger.info(f"Starting training for {self.max_epochs} epochs")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch

                # Train epoch
                train_metrics = self._train_epoch()

                # Validate
                val_metrics = {}
                if epoch % self.validate_every_n_epochs == 0:
                    val_metrics = self._validate_epoch()

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}

                # Log epoch metrics
                self.logger.log_metrics(epoch_metrics, epoch, prefix="epoch")

                # Check early stopping
                self._check_early_stopping(val_metrics)

                # Save checkpoint
                if epoch % self.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch, epoch_metrics)

                # Update scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(epoch_metrics.get(self.monitor_metric, 0))
                    else:
                        self.scheduler.step()

                if self.should_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        # Load best checkpoint
        if self.checkpoint_manager:
            try:
                best_path = self.checkpoint_manager.get_best_checkpoint_path()
                if best_path:
                    self.checkpoint_manager.load_checkpoint(
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        checkpoint_path=best_path,
                        load_best=False,
                    )
                    self.logger.info("Loaded best checkpoint")
            except Exception as e:
                self.logger.warning(f"Could not load best checkpoint: {e}")

        final_metrics = self._validate_epoch()
        self.logger.info("Training completed")
        return final_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Forward pass with optional mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log batch
            if self.log_every_n_steps > 0 and batch_idx % self.log_every_n_steps == 0:
                self._log_batch(batch, loss.item(), batch_idx)

        return {
            'train_loss': epoch_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    @abstractmethod
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch. Must be implemented by subclasses."""
        pass

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch_to_device(batch)

                # Compute loss
                loss = self._compute_loss(batch)
                val_loss += loss.item()
                num_batches += 1

                # Collect predictions and targets for metrics
                preds, targets = self._get_predictions_and_targets(batch)
                all_preds.append(preds)
                all_targets.append(targets)

        # Compute metrics
        val_metrics = self._get_metrics(all_preds, all_targets)
        val_metrics['val_loss'] = val_loss / num_batches

        return val_metrics

    @abstractmethod
    def _get_predictions_and_targets(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Get predictions and targets for metrics computation."""
        pass

    @abstractmethod
    def _get_metrics(self, all_preds: list, all_targets: list) -> Dict[str, float]:
        """Compute validation metrics."""
        pass

    def _log_batch(self, batch: Dict[str, torch.Tensor], loss: float, batch_idx: int):
        """Log batch-level information."""
        self.logger.log_metrics({
            'batch_loss': loss,
            'batch_idx': batch_idx
        }, self.global_step, prefix="batch")

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _check_early_stopping(self, val_metrics: Dict[str, float]):
        """Check if training should stop early."""
        if not val_metrics or self.monitor_metric not in val_metrics:
            return

        current_value = val_metrics[self.monitor_metric]

        if self.monitor_mode == 'min':
            is_better = current_value < self.best_metric_value
        else:
            is_better = current_value > self.best_metric_value

        if is_better:
            self.best_metric_value = current_value
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.early_stopping_patience:
            self.should_stop = True

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if self.checkpoint_manager:
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=metrics.get('train_loss', 0),
                metric=metrics.get(self.monitor_metric, 0),
                task=self.__class__.__name__
            )

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume training from checkpoint.

        Returns:
            Checkpoint metadata
        """
        if self.checkpoint_manager:
            metadata = self.checkpoint_manager.load_checkpoint(
                self.model, self.optimizer, self.scheduler, checkpoint_path
            )
            self.current_epoch = metadata.get('epoch', 0)
            self.best_metric_value = metadata.get('best_metric', self.best_metric_value)
            return metadata
        return {}