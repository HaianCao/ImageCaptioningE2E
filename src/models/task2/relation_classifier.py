"""
Relation classifier for Task 2.

Supports two input modes:
- feature mode: consumes pre-extracted union ROI features
- image mode: attaches a learnable visual backbone and consumes union ROI images
"""

import torch
import torch.nn as nn
from typing import Optional

from ..base_model import BaseModel
from ...features.visual_encoder import VisualEncoder


class RelationClassifier(BaseModel):
    """
    Relationship classification model.

    Takes union ROI features or union ROI images + spatial features and predicts relationship type.
    """

    def __init__(
        self,
        num_relations: int,
        feature_dim: int = 2048,
        spatial_dim: int = 9,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_layers: int = 2,
        attention_heads: int = 8,
        fusion_method: str = "concat",  # "concat", "attention", "gated"
        backbone_name: Optional[str] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        learnable_backbone: bool = False,
        device: str = "cuda"
    ):
        super().__init__(device)

        self.num_relations = num_relations
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.fusion_method = fusion_method
        self.learnable_backbone = bool(learnable_backbone or backbone_name)
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

            encoded_dim = self.encoder.output_dim
            if feature_dim is not None and feature_dim != encoded_dim:
                self.feature_projection = nn.Linear(encoded_dim, feature_dim)
                encoded_dim = feature_dim
            self.feature_dim = encoded_dim

        # Feature fusion
        if fusion_method == "concat":
            input_dim = self.feature_dim + spatial_dim
        elif fusion_method == "attention":
            input_dim = self.feature_dim
            self.spatial_encoder = nn.Linear(spatial_dim, self.feature_dim)
            if self.feature_dim % attention_heads != 0:
                raise ValueError(
                    f"feature_dim ({self.feature_dim}) phải chia hết cho attention_heads ({attention_heads})"
                )
            self.attention = nn.MultiheadAttention(self.feature_dim, num_heads=attention_heads, batch_first=True)
        elif fusion_method == "gated":
            input_dim = self.feature_dim
            self.spatial_gate = nn.Sequential(
                nn.Linear(spatial_dim, self.feature_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Classification head
        layers = []
        current_dim = input_dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_relations))
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

    def forward(self, features: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Union ROI features of shape (batch_size, feature_dim)
                or Union ROI images of shape (batch_size, channels, height, width)
            spatial: Spatial features of shape (batch_size, spatial_dim)

        Returns:
            Logits of shape (batch_size, num_relations)
        """
        features = self._encode_inputs(features)

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