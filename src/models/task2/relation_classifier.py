"""
Relation classifier for Task 2.

Classifies relationships between object pairs using union features and spatial info.
"""

import torch
import torch.nn as nn
from typing import Optional

from ..base_model import BaseModel


class RelationClassifier(BaseModel):
    """
    Relationship classification model.

    Takes union ROI features + spatial features and predicts relationship type.
    """

    def __init__(
        self,
        num_relations: int,
        feature_dim: int = 2048,
        spatial_dim: int = 9,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        fusion_method: str = "concat",  # "concat", "attention", "gated"
        device: str = "cuda"
    ):
        super().__init__(device)

        self.num_relations = num_relations
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.fusion_method = fusion_method

        # Feature fusion
        if fusion_method == "concat":
            input_dim = feature_dim + spatial_dim
        elif fusion_method == "attention":
            input_dim = feature_dim
            self.spatial_encoder = nn.Linear(spatial_dim, feature_dim)
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        elif fusion_method == "gated":
            input_dim = feature_dim
            self.spatial_gate = nn.Sequential(
                nn.Linear(spatial_dim, feature_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations)
        )

    def forward(self, features: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Union ROI features of shape (batch_size, feature_dim)
            spatial: Spatial features of shape (batch_size, spatial_dim)

        Returns:
            Logits of shape (batch_size, num_relations)
        """
        if self.fusion_method == "concat":
            # Concatenate features and spatial
            combined = torch.cat([features, spatial], dim=1)

        elif self.fusion_method == "attention":
            # Use attention to fuse spatial into features
            spatial_encoded = self.spatial_encoder(spatial).unsqueeze(1)  # (batch, 1, dim)
            features_unsqueezed = features.unsqueeze(1)  # (batch, 1, dim)

            # Self-attention with spatial as query
            attn_output, _ = self.attention(
                spatial_encoded, features_unsqueezed, features_unsqueezed
            )
            combined = attn_output.squeeze(1)

        elif self.fusion_method == "gated":
            # Gated fusion
            gate = self.spatial_gate(spatial)
            combined = features * gate

        return self.classifier(combined)

    def _postprocess_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert logits to class predictions."""
        return torch.argmax(outputs, dim=1)

    def predict_proba(self, features: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """Get relation probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features.to(self.device), spatial.to(self.device))
            return torch.softmax(logits, dim=1)