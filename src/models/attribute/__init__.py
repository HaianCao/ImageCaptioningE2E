"""Attribute model wrappers."""

from ..task1.attribute_classifier import AttributeClassifier
from ..task1.attribute_strategies import (
	ATTRIBUTE_STRATEGY_REGISTRY,
	AttributeModelStrategy,
	AttributeStrategyDefaults,
	build_attribute_classifier,
	get_attribute_strategy,
	register_attribute_strategy,
)

__all__ = [
	"AttributeClassifier",
	"AttributeModelStrategy",
	"AttributeStrategyDefaults",
	"ATTRIBUTE_STRATEGY_REGISTRY",
	"register_attribute_strategy",
	"get_attribute_strategy",
	"build_attribute_classifier",
]