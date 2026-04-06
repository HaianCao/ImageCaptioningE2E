"""Task 1 models: Object and attribute classification."""

from .attribute_classifier import AttributeClassifier
from .attribute_strategies import (
	ATTRIBUTE_STRATEGY_REGISTRY,
	AttributeModelStrategy,
	AttributeStrategyDefaults,
	build_attribute_classifier,
	get_attribute_strategy,
	register_attribute_strategy,
)
from .object_classifier import ObjectClassifier
from .object_strategies import (
	OBJECT_STRATEGY_REGISTRY,
	ObjectModelStrategy,
	ObjectStrategyDefaults,
	build_object_classifier,
	get_object_strategy,
	register_object_strategy,
)

__all__ = [
	"ObjectClassifier",
	"AttributeClassifier",
	"ObjectModelStrategy",
	"ObjectStrategyDefaults",
	"OBJECT_STRATEGY_REGISTRY",
	"register_object_strategy",
	"get_object_strategy",
	"build_object_classifier",
	"AttributeModelStrategy",
	"AttributeStrategyDefaults",
	"ATTRIBUTE_STRATEGY_REGISTRY",
	"register_attribute_strategy",
	"get_attribute_strategy",
	"build_attribute_classifier",
]