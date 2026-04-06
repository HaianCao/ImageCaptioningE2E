"""Registry of relation-model strategies.

Each strategy maps a config key to a concrete RelationClassifier preset.
This keeps the public API simple while making new relation architectures easy
to register.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Optional, Type

from .relation_classifier import RelationClassifier


RELATION_STRATEGY_REGISTRY: Dict[str, Type["RelationModelStrategy"]] = {}


def register_relation_strategy(*names: str):
    """Register one strategy class under one or more config keys."""

    def decorator(strategy_cls: Type["RelationModelStrategy"]):
        for name in names:
            RELATION_STRATEGY_REGISTRY[name] = strategy_cls
        return strategy_cls

    return decorator


@dataclass(frozen=True)
class RelationStrategyDefaults:
    backbone_name: str
    fusion_method: str
    hidden_dim: int
    dropout: float
    num_layers: int
    attention_heads: int


class RelationModelStrategy(ABC):
    """Base class for relation model presets."""

    defaults = RelationStrategyDefaults(
        backbone_name="resnet50",
        fusion_method="concat",
        hidden_dim=1024,
        dropout=0.4,
        num_layers=2,
        attention_heads=8,
    )

    def build(
        self,
        num_relations: int,
        *,
        spatial_dim: int = 9,
        feature_dim: Optional[int] = None,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        device: str = "cuda",
        hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        num_layers: Optional[int] = None,
        attention_heads: Optional[int] = None,
        fusion_method: Optional[str] = None,
    ) -> RelationClassifier:
        return RelationClassifier(
            num_relations=num_relations,
            feature_dim=feature_dim,
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim if hidden_dim is not None else self.defaults.hidden_dim,
            dropout=dropout if dropout is not None else self.defaults.dropout,
            num_layers=num_layers if num_layers is not None else self.defaults.num_layers,
            attention_heads=attention_heads if attention_heads is not None else self.defaults.attention_heads,
            fusion_method=fusion_method if fusion_method is not None else self.defaults.fusion_method,
            backbone_name=self.defaults.backbone_name,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            device=device,
        )


@register_relation_strategy("resnet50", "baseline_cnn")
class ResNet50RelationStrategy(RelationModelStrategy):
    defaults = RelationStrategyDefaults(
        backbone_name="resnet50",
        fusion_method="concat",
        hidden_dim=1024,
        dropout=0.4,
        num_layers=2,
        attention_heads=8,
    )


@register_relation_strategy("convnextv2_tiny", "modern_cnn")
class ConvNeXtV2TinyRelationStrategy(RelationModelStrategy):
    defaults = RelationStrategyDefaults(
        backbone_name="convnextv2_tiny",
        fusion_method="concat",
        hidden_dim=1024,
        dropout=0.3,
        num_layers=2,
        attention_heads=8,
    )


@register_relation_strategy("efficientnetv2_s")
class EfficientNetV2SRelationStrategy(RelationModelStrategy):
    defaults = RelationStrategyDefaults(
        backbone_name="efficientnetv2_s",
        fusion_method="concat",
        hidden_dim=896,
        dropout=0.3,
        num_layers=2,
        attention_heads=8,
    )


@register_relation_strategy("swin_tiny_patch4_window7_224", "transformer")
class SwinTinyRelationStrategy(RelationModelStrategy):
    defaults = RelationStrategyDefaults(
        backbone_name="swin_tiny_patch4_window7_224",
        fusion_method="attention",
        hidden_dim=768,
        dropout=0.2,
        num_layers=2,
        attention_heads=8,
    )


def get_relation_strategy(strategy_name: str) -> RelationModelStrategy:
    """Resolve a strategy name to a strategy instance."""
    strategy_cls = RELATION_STRATEGY_REGISTRY.get(strategy_name)
    if strategy_cls is None:
        available = ", ".join(sorted(RELATION_STRATEGY_REGISTRY))
        raise ValueError(f"Unknown relation strategy '{strategy_name}'. Available: {available}")
    return strategy_cls()


def build_relation_classifier(
    strategy_name: str,
    *,
    num_relations: int,
    spatial_dim: int = 9,
    feature_dim: Optional[int] = None,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cuda",
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    num_layers: Optional[int] = None,
    attention_heads: Optional[int] = None,
    fusion_method: Optional[str] = None,
) -> RelationClassifier:
    """Convenience helper that builds a relation classifier from a registered strategy."""
    strategy = get_relation_strategy(strategy_name)
    return strategy.build(
        num_relations,
        spatial_dim=spatial_dim,
        feature_dim=feature_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        attention_heads=attention_heads,
        fusion_method=fusion_method,
    )


__all__ = [
    "RelationModelStrategy",
    "RelationStrategyDefaults",
    "RELATION_STRATEGY_REGISTRY",
    "register_relation_strategy",
    "get_relation_strategy",
    "build_relation_classifier",
]