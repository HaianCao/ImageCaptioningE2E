"""
Training module for Visual Genome Caption Generation.

Contains training loops and utilities for:
- Base trainer with common functionality
- Task 1: Object & attribute classification training
- Task 2: Relationship classification training
"""

from .trainer import BaseTrainer
from .task1_trainer import Task1Trainer
from .task2_trainer import Task2Trainer

__all__ = ["BaseTrainer", "Task1Trainer", "Task2Trainer"]