"""
Models module for Visual Genome Caption Generation.

Contains model definitions for:
- Task 1: Object and attribute classification
- Task 2: Relationship classification
- Caption generation
"""

from .base_model import BaseModel
from . import task1, task2, caption

__all__ = ["BaseModel", "task1", "task2", "caption"]