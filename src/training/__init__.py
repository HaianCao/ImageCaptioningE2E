"""
Training module for Visual Genome Caption Generation.

Contains training loops and utilities for:
- Base trainer with common functionality
- Object classification training
- Attribute classification training
- Relation classification training
"""

from .trainer import BaseTrainer
from .object_trainer import ObjectTrainer
from .attribute_trainer import AttributeTrainer
from .relation_trainer import RelationTrainer

__all__ = ["BaseTrainer", "ObjectTrainer", "AttributeTrainer", "RelationTrainer"]