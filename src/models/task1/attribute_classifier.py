"""
Attribute classifier for Task 1.

Multi-label classification of object attributes.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..base_model import BaseModel


class AttributeClassifier(BaseModel):
    """
    Attribute classification model.

    Takes ROI features as input and predicts multiple attributes (multi-label).
    """

    def __init__(
        self,
        num_attributes: int,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_layers: int = 2,
        device: str = "cuda"
    ):
        super().__init__(device)

        self.num_attributes = num_attributes
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
        layers.append(nn.Linear(current_dim, num_attributes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: ROI features of shape (batch_size, feature_dim)

        Returns:
            Logits of shape (batch_size, num_attributes)
        """
        return self.classifier(features)

    def _postprocess_predictions(self, outputs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert logits to binary predictions using threshold."""
        return (torch.sigmoid(outputs) > threshold).long()

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get attribute probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features.to(self.device))
            return torch.sigmoid(logits)