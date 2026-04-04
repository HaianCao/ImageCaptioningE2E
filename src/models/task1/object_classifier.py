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
        device: str = "cuda"
    ):
        super().__init__(device)

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

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