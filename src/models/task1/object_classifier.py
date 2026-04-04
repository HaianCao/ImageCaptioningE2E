"""
Object classifier for Task 1.

Classifies objects from ROI features.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..base_model import BaseModel


class ObjectClassifier(BaseModel):
    """
    Object classification model.

    Takes ROI features as input and predicts object class.
    """

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_layers: int = 2,
        device: str = "cuda"
    ):
        super().__init__(device)

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        if num_layers < 1:
            raise ValueError("num_layers phải >= 1")

        layers = []
        current_dim = feature_dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: ROI features of shape (batch_size, feature_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.classifier(features)

    def _postprocess_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert logits to class predictions."""
        return torch.argmax(outputs, dim=1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features.to(self.device))
            return torch.softmax(logits, dim=1)