"""
Evaluation module for Visual Genome Caption Generation.

Contains metrics and evaluation functions for:
- Task 1: Object/attribute classification metrics
- Task 2: Relationship classification metrics
- Caption generation: BLEU, CIDEr, METEOR scores
"""

from .metrics import *

__all__ = [
    'compute_classification_metrics',
    'compute_multilabel_metrics',
    'compute_caption_metrics'
]