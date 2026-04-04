"""
Evaluation metrics for Visual Genome tasks.

The training pipeline currently treats Task 1 and Task 2 as classification problems,
so the primary metrics here are accuracy and F1.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        preds: Predicted labels (N,) or logits (N, C)
        targets: Ground truth labels (N,)
        num_classes: Kept for backward compatibility, ignored.
        average: Averaging method for F1 ('macro', 'micro', 'weighted')

    Returns:
        Dict with accuracy and F1
    """
    # Convert to numpy
    if isinstance(preds, torch.Tensor):
        if preds.dim() == 2:  # logits
            preds = preds.argmax(dim=1)
        preds = preds.cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    return {
        'accuracy': float(accuracy_score(targets, preds)),
        'f1': float(f1_score(targets, preds, average=average, zero_division=0)),
    }


def compute_mAP(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> float:
    """
    Compute mean Average Precision.

    Args:
        preds: Predicted labels (N,)
        targets: Ground truth labels (N,)
        num_classes: Number of classes

    Returns:
        mAP score
    """
    aps = []
    for class_idx in range(num_classes):
        # Binary classification for this class
        y_true = (targets == class_idx).astype(int)
        y_pred = (preds == class_idx).astype(int)

        if y_true.sum() == 0:
            continue  # Skip classes with no true positives

        ap = compute_average_precision(y_true, y_pred)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def compute_average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Average Precision for binary classification."""
    if not np.any(y_true):
        return 0.0

    # Sort by prediction confidence (assume higher = more confident)
    # For now, use prediction as confidence
    indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    precision = tp / (tp + fp)
    recall = tp / np.sum(y_true)

    # Add (0, 1) point
    precision = np.concatenate([[1], precision])
    recall = np.concatenate([[0], recall])

    # Compute AP
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])

    return ap


def compute_multilabel_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.

    Args:
        preds: Predicted logits (N, num_attributes)
        targets: Ground truth binary labels (N, num_attributes)
        threshold: Threshold for positive predictions
        average: F1 averaging method ('macro', 'micro', 'weighted')

    Returns:
        Dict with accuracy and F1
    """
    # Convert to binary predictions
    pred_binary = (preds.sigmoid() > threshold).int()

    # Convert to numpy
    pred_np = pred_binary.cpu().numpy()
    target_np = targets.cpu().numpy().astype(int)

    return {
        'accuracy': float(accuracy_score(target_np, pred_np)),
        'f1': float(f1_score(target_np, pred_np, average=average, zero_division=0)),
    }


def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
    metrics: List[str] = ['bleu', 'meteor', 'rouge', 'cider']
) -> Dict[str, float]:
    """
    Compute caption evaluation metrics.

    Args:
        predictions: List of generated captions
        references: List of lists of reference captions
        metrics: Which metrics to compute

    Returns:
        Dict with metric scores
    """
    results = {}

    try:
        if 'bleu' in metrics:
            results.update(compute_bleu(predictions, references))
        if 'meteor' in metrics:
            results['meteor'] = compute_meteor(predictions, references)
        if 'rouge' in metrics:
            results.update(compute_rouge(predictions, references))
        if 'cider' in metrics:
            results['cider'] = compute_cider(predictions, references)
    except ImportError as e:
        print(f"Warning: Caption metrics not available: {e}")
        print("Install pycocoevalcap for full caption evaluation")

    return results


def compute_bleu(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """Compute BLEU scores."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        scorer = Bleu(n=4)
        score, _ = scorer.compute_score(references, predictions)
        return {f'bleu_{i+1}': s for i, s in enumerate(score)}
    except ImportError:
        return {}


def compute_meteor(predictions: List[str], references: List[List[str]]) -> float:
    """Compute METEOR score."""
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        scorer = Meteor()
        score, _ = scorer.compute_score(references, predictions)
        return score
    except ImportError:
        return 0.0


def compute_rouge(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    try:
        from pycocoevalcap.rouge.rouge import Rouge
        scorer = Rouge()
        score, _ = scorer.compute_score(references, predictions)
        return {'rouge_l': score}
    except ImportError:
        return {}


def compute_cider(predictions: List[str], references: List[List[str]]) -> float:
    """Compute CIDEr score."""
    try:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()
        score, _ = scorer.compute_score(references, predictions)
        return score
    except ImportError:
        return 0.0


# Update __all__ in __init__.py
__all__ = [
    'compute_classification_metrics',
    'compute_multilabel_metrics',
    'compute_caption_metrics'
]