"""
Checkpoint management utilities.

Handles saving/loading model checkpoints with metadata.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json


class CheckpointManager:
    """
    Manages model checkpoints with automatic saving and loading.

    Features:
    - Save checkpoints with metadata
    - Load latest or best checkpoints
    - Automatic cleanup of old checkpoints
    - Resume training from checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "vg_caption",
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only

        self.best_metric = float('-inf')
        self.checkpoints = []

        # Load existing checkpoints
        self._load_checkpoint_list()

    def _load_checkpoint_list(self):
        """Load list of existing checkpoints."""
        self.checkpoints = []
        if self.checkpoint_dir.exists():
            for ckpt_file in self.checkpoint_dir.glob("*.pth"):
                try:
                    # Load metadata from checkpoint
                    checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                    metadata = checkpoint.get('metadata', {})

                    self.checkpoints.append({
                        'path': ckpt_file,
                        'epoch': metadata.get('epoch', 0),
                        'loss': metadata.get('loss', float('inf')),
                        'metric': metadata.get('best_metric', float('-inf')),
                        'timestamp': ckpt_file.stat().st_mtime
                    })
                except Exception as e:
                    print(f"Warning: Could not load checkpoint {ckpt_file}: {e}")

        # Sort by timestamp (newest first)
        self.checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        loss: float = 0.0,
        metric: Optional[float] = None,
        filename: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            loss: Current loss
            metric: Current metric value
            filename: Custom filename
            **metadata: Additional metadata

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            if metric is not None and metric > self.best_metric:
                filename = "best_model.pth"
                self.best_metric = metric
            else:
                filename = f"checkpoint_epoch_{epoch}.pth"

        checkpoint_path = self.checkpoint_dir / filename

        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': {
                'epoch': epoch,
                'loss': loss,
                'best_metric': self.best_metric,
                **metadata
            }
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update checkpoint list
        self._load_checkpoint_list()

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optimizer to load state
            scheduler: Scheduler to load state
            checkpoint_path: Specific checkpoint path
            load_best: Load best checkpoint if path not specified

        Returns:
            Checkpoint metadata
        """
        if checkpoint_path is None:
            if load_best and self.checkpoints:
                # Load best checkpoint (highest metric)
                best_ckpt = max(self.checkpoints, key=lambda x: x['metric'])
                checkpoint_path = best_ckpt['path']
            elif self.checkpoints:
                # Load latest checkpoint
                checkpoint_path = self.checkpoints[0]['path']
            else:
                raise FileNotFoundError("No checkpoints found")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer/scheduler
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        metadata = checkpoint.get('metadata', {})
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {metadata.get('epoch', 'unknown')}, Loss: {metadata.get('loss', 'unknown')}")

        return metadata

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if self.max_checkpoints <= 0:
            return

        # Keep best checkpoint + recent ones
        keep_files = set()

        # Always keep best model
        best_files = [ckpt for ckpt in self.checkpoints if ckpt['path'].name == "best_model.pth"]
        keep_files.update(str(ckpt['path']) for ckpt in best_files)

        # Keep recent checkpoints
        recent_files = self.checkpoints[:self.max_checkpoints]
        keep_files.update(str(ckpt['path']) for ckpt in recent_files)

        # Remove old ones
        for ckpt in self.checkpoints:
            if str(ckpt['path']) not in keep_files:
                ckpt['path'].unlink()
                print(f"Removed old checkpoint: {ckpt['path']}")

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return [{
            'filename': ckpt['path'].name,
            'epoch': ckpt['epoch'],
            'loss': ckpt['loss'],
            'metric': ckpt['metric'],
            'timestamp': ckpt['timestamp']
        } for ckpt in self.checkpoints]

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint."""
        if not self.checkpoints:
            return None

        best_ckpt = max(self.checkpoints, key=lambda x: x['metric'])
        return str(best_ckpt['path'])

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if not self.checkpoints:
            return None

        return str(self.checkpoints[0]['path'])