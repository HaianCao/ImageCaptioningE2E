"""
Visual Encoder: Configurable pretrained backbone cho feature extraction.

Hỗ trợ các backbone:
- ResNet-50, ResNet-101 (torchvision)
- EfficientNet-B3, EfficientNet-B5 (timm)
- ViT-Base/16, ViT-Large/16 (timm)

Tất cả đều trả về feature vector 1D (global average pooled).
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
from torchvision import models

BACKBONE_CONFIGS: Dict[str, Dict] = {
    "resnet50": {
        "feature_dim": 2048,
        "source": "torchvision",
        "timm_name": None,
    },
    "resnet101": {
        "feature_dim": 2048,
        "source": "torchvision",
        "timm_name": None,
    },
    "efficientnet_b3": {
        "feature_dim": 1536,
        "source": "timm",
        "timm_name": "efficientnet_b3",
    },
    "efficientnet_b5": {
        "feature_dim": 2048,
        "source": "timm",
        "timm_name": "efficientnet_b5",
    },
    "vit_base_16": {
        "feature_dim": 768,
        "source": "timm",
        "timm_name": "vit_base_patch16_224",
    },
    "vit_large_16": {
        "feature_dim": 1024,
        "source": "timm",
        "timm_name": "vit_large_patch16_224",
    },
}

class VisualEncoder(nn.Module):
    """
    Wrapper quanh pretrained backbone cho visual feature extraction.
    """
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, frozen: bool = False):
        super().__init__()
        if backbone_name not in BACKBONE_CONFIGS:
            raise ValueError(f"Backbone '{backbone_name}' không được hỗ trợ.")

        self.backbone_name = backbone_name
        config = BACKBONE_CONFIGS[backbone_name]
        self.feature_dim = config["feature_dim"]

        if config["source"] == "torchvision":
            self.backbone = self._build_torchvision(backbone_name, pretrained)
        else:
            self.backbone = self._build_timm(config["timm_name"], pretrained)

        if frozen:
            self.freeze()

    def _build_torchvision(self, name: str, pretrained: bool) -> nn.Module:
        weights_map = {
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
            "resnet101": models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None,
        }
        model_fn = getattr(models, name)
        model = model_fn(weights=weights_map[name])
        model.fc = nn.Identity()
        return model

    def _build_timm(self, timm_name: str, pretrained: bool) -> nn.Module:
        import timm
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return model

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def output_dim(self) -> int:
        return self.feature_dim


def build_encoder(backbone_name: str, pretrained: bool = True, frozen: bool = False) -> VisualEncoder:
    return VisualEncoder(backbone_name=backbone_name, pretrained=pretrained, frozen=frozen)
