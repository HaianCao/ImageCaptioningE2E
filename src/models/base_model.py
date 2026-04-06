"""
Base model class for Visual Genome project.

Provides common functionality for all models:
- Forward pass
- Prediction
- Save/load checkpoints
- Device handling
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in the Visual Genome project.

    Provides:
    - Device management
    - Checkpoint saving/loading
    - Common prediction interface
    - Training mode utilities
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        self.to(self.device)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass - must be implemented by subclasses."""
        pass

    def predict(self, inputs: Any) -> Any:
        """
        Prediction interface.

        Args:
            inputs: Model inputs (tensor, dict, etc.)

        Returns:
            Model predictions
        """
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}

            outputs = self.forward(inputs)
            return self._postprocess_predictions(outputs)

    def _postprocess_predictions(self, outputs: Any) -> Any:
        """
        Post-process model outputs for predictions.

        Default: return outputs as-is. Subclasses can override.
        """
        return outputs

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            optimizer: Optimizer state (optional)
            scheduler: Scheduler state (optional)
            epoch: Current epoch
            loss: Current loss
            **kwargs: Additional metadata
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'epoch': epoch,
            'loss': loss,
            **kwargs
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint
            optimizer: Optimizer to load state (optional)
            scheduler: Scheduler to load state (optional)
            strict: Strict loading for model weights

        Returns:
            Checkpoint metadata
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model weights
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        # Load optimizer/scheduler if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded: {filepath}")
        return {k: v for k, v in checkpoint.items()
                if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']}

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Model summary string."""
        return (
            f"{self.__class__.__name__} | "
            f"Parameters: {self.get_num_parameters():,} | "
            f"Device: {self.device}"
        )