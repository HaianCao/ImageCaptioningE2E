"""Registry of attribute-model strategies.

Each strategy maps a config key to a concrete AttributeClassifier preset.
This keeps the public API simple while making new attribute architectures easy
to register.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Type

from .attribute_classifier import AttributeClassifier


ATTRIBUTE_STRATEGY_REGISTRY: Dict[str, Type["AttributeModelStrategy"]] = {}


def register_attribute_strategy(*names: str):
    """Register one strategy class under one or more config keys."""

    def decorator(strategy_cls: Type["AttributeModelStrategy"]):
        for name in names:
            ATTRIBUTE_STRATEGY_REGISTRY[name] = strategy_cls
        return strategy_cls

    return decorator


@dataclass(frozen=True)
class AttributeStrategyDefaults:
    backbone_name: str
    hidden_dim: int
    dropout: float
    num_layers: int


class AttributeModelStrategy(ABC):
    """Base class for attribute model presets."""

    defaults = AttributeStrategyDefaults(
        backbone_name="resnet50",
        hidden_dim=512,
        dropout=0.3,
        num_layers=2,
    )

    def build(
        self,
        num_attributes: int,
        *,
        feature_dim: Optional[int] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device: str = "cuda",
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        num_layers: Optional[int] = None,
    ) -> AttributeClassifier:
        return AttributeClassifier(
            num_attributes=num_attributes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim if hidden_dim is not None else self.defaults.hidden_dim,
            dropout=dropout if dropout is not None else self.defaults.dropout,
            num_layers=num_layers if num_layers is not None else self.defaults.num_layers,
            backbone_name=self.defaults.backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            device=device,
        )


@register_attribute_strategy("resnet50", "baseline_cnn")
class ResNet50AttributeStrategy(AttributeModelStrategy):
    defaults = AttributeStrategyDefaults(
        backbone_name="resnet50",
        hidden_dim=512,
        dropout=0.3,
        num_layers=2,
    )


@register_attribute_strategy("convnextv2_tiny", "modern_cnn")
class ConvNeXtV2TinyAttributeStrategy(AttributeModelStrategy):
    defaults = AttributeStrategyDefaults(
        backbone_name="convnextv2_tiny",
        hidden_dim=768,
        dropout=0.2,
        num_layers=2,
    )


@register_attribute_strategy("efficientnetv2_s")
class EfficientNetV2SAttributeStrategy(AttributeModelStrategy):
    defaults = AttributeStrategyDefaults(
        backbone_name="efficientnetv2_s",
        hidden_dim=640,
        dropout=0.2,
        num_layers=2,
    )


@register_attribute_strategy("swin_tiny_patch4_window7_224", "transformer")
class SwinTinyAttributeStrategy(AttributeModelStrategy):
    defaults = AttributeStrategyDefaults(
        backbone_name="swin_tiny_patch4_window7_224",
        hidden_dim=768,
        dropout=0.2,
        num_layers=2,
    )


def get_attribute_strategy(strategy_name: str) -> AttributeModelStrategy:
    """Resolve a strategy name to a strategy instance."""
    strategy_cls = ATTRIBUTE_STRATEGY_REGISTRY.get(strategy_name)
    if strategy_cls is None:
        available = ", ".join(sorted(ATTRIBUTE_STRATEGY_REGISTRY))
        raise ValueError(f"Unknown attribute strategy '{strategy_name}'. Available: {available}")
    return strategy_cls()


def build_attribute_classifier(
    strategy_name: str,
    *,
    num_attributes: int,
    feature_dim: Optional[int] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cuda",
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    num_layers: Optional[int] = None,
) -> AttributeClassifier:
    """Convenience helper that builds an attribute classifier from a registered strategy."""
    strategy = get_attribute_strategy(strategy_name)
    return strategy.build(
        num_attributes,
        feature_dim=feature_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
    )


__all__ = [
    "AttributeModelStrategy",
    "AttributeStrategyDefaults",
    "ATTRIBUTE_STRATEGY_REGISTRY",
    "register_attribute_strategy",
    "get_attribute_strategy",
    "build_attribute_classifier",
]