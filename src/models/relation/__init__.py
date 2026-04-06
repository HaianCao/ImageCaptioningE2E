"""Relation model wrappers."""

from ..task2.relation_classifier import RelationClassifier
from ..task2.relation_strategies import (
	RELATION_STRATEGY_REGISTRY,
	RelationModelStrategy,
	RelationStrategyDefaults,
	build_relation_classifier,
	get_relation_strategy,
	register_relation_strategy,
)

__all__ = [
	"RelationClassifier",
	"RelationModelStrategy",
	"RelationStrategyDefaults",
	"RELATION_STRATEGY_REGISTRY",
	"register_relation_strategy",
	"get_relation_strategy",
	"build_relation_classifier",
]