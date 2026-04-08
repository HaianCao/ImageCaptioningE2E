"""
Visual Encoder: Configurable pretrained backbone cho feature extraction.

Hỗ trợ:
- ResNet-50, ResNet-101 (torchvision)
- Bất kỳ backbone timm hợp lệ nào, gồm ConvNeXt, ConvNeXt V2, Swin,
  EfficientNetV2, và ViT variants.

Tất cả backbone đều trả về feature vector 1D sau pooling.
"""

from typing import Tuple
import warnings
import torch
import torch.nn as nn
from torchvision import models

TORCHVISION_BACKBONES = {
    "resnet50": "resnet50",
    "resnet101": "resnet101",
}

TIMM_BACKBONE_ALIASES = {
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b5": "efficientnet_b5",
    "efficientnetv2_s": "tf_efficientnetv2_s",
    "efficientnetv2_m": "tf_efficientnetv2_m",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnextv2_tiny": "convnextv2_tiny",
    "convnextv2_base": "convnextv2_base",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224": "swin_small_patch4_window7_224",
    "vit_base_16": "vit_base_patch16_224",
    "vit_large_16": "vit_large_patch16_224",
}

class VisualEncoder(nn.Module):
    """
    Wrapper quanh pretrained backbone cho visual feature extraction.
    """
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True, frozen: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name in TORCHVISION_BACKBONES:
            self.backbone, self.feature_dim = self._build_torchvision(backbone_name, pretrained)
        else:
            self.backbone, self.feature_dim = self._build_timm(backbone_name, pretrained)

        if self.feature_dim is None:
            raise ValueError(f"Không thể suy ra feature dimension cho backbone '{backbone_name}'")

        if frozen:
            self.freeze()

    def _build_torchvision(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        weights_map = {
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
            "resnet101": models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None,
        }
        model_fn = getattr(models, name)
        model = model_fn(weights=weights_map[name])
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, feature_dim

    def _build_timm(self, backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        import timm

        timm_name = TIMM_BACKBONE_ALIASES.get(backbone_name, backbone_name)
        try:
            model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        except RuntimeError as exc:
            if pretrained and "No pretrained weights exist for" in str(exc):
                warnings.warn(
                    f"Pretrained weights are unavailable for '{timm_name}'. Falling back to random init.",
                    RuntimeWarning,
                )
                model = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="avg")
            else:
                raise

        feature_dim = getattr(model, "num_features", None)
        if feature_dim is None and hasattr(model, "feature_info") and hasattr(model.feature_info, "channels"):
            channels = model.feature_info.channels()
            feature_dim = channels[-1] if channels else None

        if feature_dim is None:
            raise ValueError(f"Không thể suy ra feature dimension cho backbone '{backbone_name}'")

        return model, int(feature_dim)

    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def output_dim(self) -> int:
        return self.feature_dim


def build_encoder(backbone_name: str, pretrained: bool = True, frozen: bool = False) -> VisualEncoder:
    return VisualEncoder(backbone_name=backbone_name, pretrained=pretrained, frozen=frozen)
