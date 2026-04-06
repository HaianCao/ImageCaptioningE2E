"""Registry of object-model strategies.

Each strategy is a small preset that maps a config key to a concrete
ObjectClassifier configuration. This keeps the public API simple while making
new object architectures easy to register.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Type

from .object_classifier import ObjectClassifier


OBJECT_STRATEGY_REGISTRY: Dict[str, Type["ObjectModelStrategy"]] = {}


def register_object_strategy(*names: str):
    """Register one strategy class under one or more config keys."""

    def decorator(strategy_cls: Type["ObjectModelStrategy"]):
        for name in names:
            OBJECT_STRATEGY_REGISTRY[name] = strategy_cls
        return strategy_cls

    return decorator


@dataclass(frozen=True)
class ObjectStrategyDefaults:
    backbone_name: str
    hidden_dim: int
    dropout: float
    num_layers: int


class ObjectModelStrategy(ABC):
    """Base class for object model presets."""

    defaults = ObjectStrategyDefaults(
        backbone_name="resnet50",
        hidden_dim=512,
        dropout=0.3,
        num_layers=2,
    )

    def build(
        self,
        num_classes: int,
        *,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device: str = "cuda",
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        num_layers: Optional[int] = None,
    ) -> ObjectClassifier:
        return ObjectClassifier(
            num_classes=num_classes,
            hidden_dim=hidden_dim if hidden_dim is not None else self.defaults.hidden_dim,
            dropout=dropout if dropout is not None else self.defaults.dropout,
            num_layers=num_layers if num_layers is not None else self.defaults.num_layers,
            backbone_name=self.defaults.backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            device=device,
        )


@register_object_strategy("resnet50", "baseline_cnn")
class ResNet50ObjectStrategy(ObjectModelStrategy):
    defaults = ObjectStrategyDefaults(
        backbone_name="resnet50",
        hidden_dim=512,
        dropout=0.3,
        num_layers=2,
    )


@register_object_strategy("convnextv2_tiny", "modern_cnn")
class ConvNeXtV2TinyObjectStrategy(ObjectModelStrategy):
    defaults = ObjectStrategyDefaults(
        backbone_name="convnextv2_tiny",
        hidden_dim=768,
        dropout=0.2,
        num_layers=2,
    )


@register_object_strategy("efficientnetv2_s")
class EfficientNetV2SObjectStrategy(ObjectModelStrategy):
    defaults = ObjectStrategyDefaults(
        backbone_name="efficientnetv2_s",
        hidden_dim=640,
        dropout=0.2,
        num_layers=2,
    )


@register_object_strategy("swin_tiny_patch4_window7_224", "transformer")
class SwinTinyObjectStrategy(ObjectModelStrategy):
    defaults = ObjectStrategyDefaults(
        backbone_name="swin_tiny_patch4_window7_224",
        hidden_dim=768,
        dropout=0.2,
        num_layers=2,
    )


def get_object_strategy(strategy_name: str) -> ObjectModelStrategy:
    """Resolve a strategy name to a strategy instance."""
    strategy_cls = OBJECT_STRATEGY_REGISTRY.get(strategy_name)
    if strategy_cls is None:
        available = ", ".join(sorted(OBJECT_STRATEGY_REGISTRY))
        raise ValueError(f"Unknown object strategy '{strategy_name}'. Available: {available}")
    return strategy_cls()


def build_object_classifier(
    strategy_name: str,
    *,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cuda",
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    num_layers: Optional[int] = None,
) -> ObjectClassifier:
    """Convenience helper that builds an object classifier from a registered strategy."""
    strategy = get_object_strategy(strategy_name)
    return strategy.build(
        num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
    )


__all__ = [
    "ObjectModelStrategy",
    "ObjectStrategyDefaults",
    "OBJECT_STRATEGY_REGISTRY",
    "register_object_strategy",
    "get_object_strategy",
    "build_object_classifier",
]