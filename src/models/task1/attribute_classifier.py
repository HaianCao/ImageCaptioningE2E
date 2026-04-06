"""
Attribute classifier for Task 1.

Supports two input modes:
- feature mode: consumes pre-extracted ROI features
- image mode: attaches a learnable visual backbone and consumes ROI images
"""

import torch
import torch.nn as nn
from typing import Optional

from ..base_model import BaseModel
from ...features.visual_encoder import VisualEncoder


class AttributeClassifier(BaseModel):
    """
    Attribute classification model.

    Takes ROI features or ROI images as input and predicts multiple attributes.
    """

    def __init__(
        self,
        num_attributes: int,
        feature_dim: Optional[int] = None,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_layers: int = 2,
        backbone_name: Optional[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device: str = "cuda"
    ):
        super().__init__(device)

        self.num_attributes = num_attributes
        self.feature_dim = feature_dim
        self.learnable_backbone = backbone_name is not None
        self.backbone_name = backbone_name
        self.encoder: Optional[VisualEncoder] = None
        self.feature_projection: Optional[nn.Module] = None

        if num_layers < 1:
            raise ValueError("num_layers phải >= 1")

        if self.learnable_backbone:
            if backbone_name is None:
                raise ValueError("backbone_name phải được cung cấp khi learnable_backbone=True")

            self.encoder = VisualEncoder(
                backbone_name=backbone_name,
                pretrained=pretrained,
                frozen=freeze_backbone,
            )

            input_dim = self.encoder.output_dim
            if feature_dim is not None and feature_dim != input_dim:
                self.feature_projection = nn.Linear(input_dim, feature_dim)
                input_dim = feature_dim
            self.feature_dim = input_dim
        else:
            input_dim = feature_dim if feature_dim is not None else 2048
            self.feature_dim = input_dim

        layers = []
        current_dim = input_dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_attributes))

        self.classifier = nn.Sequential(*layers)

    def freeze_backbone(self) -> None:
        """Freeze the visual backbone when the model is used in image mode."""
        if self.encoder is not None:
            self.encoder.freeze()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the visual backbone when fine-tuning should start."""
        if self.encoder is not None:
            self.encoder.unfreeze()

    def _encode_inputs(self, features: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return features

        if features.dim() != 4:
            raise ValueError(
                "learnable_backbone=True requires image tensors with shape (batch, channels, height, width)"
            )

        encoded = self.encoder(features)
        if self.feature_projection is not None:
            encoded = self.feature_projection(encoded)
        return encoded

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: ROI features of shape (batch_size, feature_dim)
                or ROI images of shape (batch_size, channels, height, width)

        Returns:
            Logits of shape (batch_size, num_attributes)
        """
        encoded = self._encode_inputs(features)
        return self.classifier(encoded)

    def _postprocess_predictions(self, outputs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert logits to binary predictions using threshold."""
        return (torch.sigmoid(outputs) > threshold).long()

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get attribute probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features.to(self.device))
            return torch.sigmoid(logits)