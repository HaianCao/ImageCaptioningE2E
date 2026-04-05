"""
Evaluation metrics for Visual Genome tasks.

The training pipeline currently treats Task 1 and Task 2 as classification problems,
so the primary metrics here are accuracy and F1.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, hamming_loss


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
    average: str = 'macro',
    *,
    threshold_mode: str = 'adaptive_mean_std',
    threshold_scale: float = 0.5,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.

    Args:
        preds: Predicted logits (N, num_attributes)
        targets: Ground truth binary labels (N, num_attributes)
        threshold: Base threshold for fixed mode
        threshold_mode: "adaptive_mean_std" for per-sample thresholding or "fixed"
        threshold_scale: How strongly to adjust the threshold with per-sample std
        threshold_min: Lower bound for adaptive thresholds
        threshold_max: Upper bound for adaptive thresholds
        average: Kept for backward compatibility; not used by the sample metrics

    Returns:
        Dict with exact-match accuracy, sample-wise accuracy summary,
        sample F1, Jaccard, and Hamming loss.
    """
    if isinstance(preds, torch.Tensor):
        pred_tensor = preds.detach().cpu()
    else:
        pred_tensor = torch.as_tensor(preds)

    if isinstance(targets, torch.Tensor):
        target_tensor = targets.detach().cpu()
    else:
        target_tensor = torch.as_tensor(targets)

    if pred_tensor.ndim == 1:
        pred_tensor = pred_tensor.unsqueeze(0)
    if target_tensor.ndim == 1:
        target_tensor = target_tensor.unsqueeze(0)

    if pred_tensor.numel() == 0 or target_tensor.numel() == 0:
        return {
            'exact_match_accuracy': 0.0,
            'sample_accuracy_mean': 0.0,
            'sample_accuracy_variance': 0.0,
            'sample_f1': 0.0,
            'jaccard_index': 0.0,
            'hamming_loss': 0.0,
        }

    if pred_tensor.shape != target_tensor.shape:
        raise ValueError(
            f"Shape mismatch for multilabel metrics: preds={tuple(pred_tensor.shape)} vs targets={tuple(target_tensor.shape)}"
        )

    # Accept either logits/probabilities or already-binarized predictions.
    if pred_tensor.dtype.is_floating_point and torch.all((pred_tensor == 0) | (pred_tensor == 1)).item():
        pred_binary = pred_tensor.int()
    elif pred_tensor.dtype.is_floating_point:
        pred_scores = pred_tensor.sigmoid() if (pred_tensor.min().item() < 0.0 or pred_tensor.max().item() > 1.0) else pred_tensor.float()
        mode = (threshold_mode or 'fixed').lower()

        if mode in {'fixed', 'static'}:
            pred_binary = (pred_scores >= threshold).int()
        elif mode in {'adaptive_mean_std', 'dynamic', 'adaptive'}:
            sample_mean = pred_scores.mean(dim=1, keepdim=True)
            sample_std = pred_scores.std(dim=1, unbiased=False, keepdim=True)
            adaptive_threshold = sample_mean - (threshold_scale * sample_std)
            adaptive_threshold = torch.clamp(adaptive_threshold, min=threshold_min, max=threshold_max)
            pred_binary = (pred_scores >= adaptive_threshold).int()
        elif mode in {'adaptive_mean', 'mean'}:
            adaptive_threshold = torch.clamp(
                pred_scores.mean(dim=1, keepdim=True),
                min=threshold_min,
                max=threshold_max,
            )
            pred_binary = (pred_scores >= adaptive_threshold).int()
        else:
            raise ValueError(
                f"Unsupported threshold_mode='{threshold_mode}'. Use 'fixed' or 'adaptive_mean_std'."
            )
    else:
        pred_binary = pred_tensor.int()

    pred_np = pred_binary.cpu().numpy().astype(int)
    target_np = target_tensor.cpu().numpy().astype(int)

    exact_match_accuracy = float(accuracy_score(target_np, pred_np))
    sample_accuracy_per_row = (pred_np == target_np).mean(axis=1) if pred_np.size else np.array([])
    sample_accuracy_mean = float(sample_accuracy_per_row.mean()) if sample_accuracy_per_row.size else 0.0
    sample_accuracy_variance = float(sample_accuracy_per_row.var(ddof=0)) if sample_accuracy_per_row.size else 0.0

    intersection = np.logical_and(pred_np, target_np).sum(axis=1) if pred_np.size else np.array([])
    pred_count = pred_np.sum(axis=1) if pred_np.size else np.array([])
    target_count = target_np.sum(axis=1) if target_np.size else np.array([])

    f1_denominator = pred_count + target_count
    sample_f1 = np.where(f1_denominator == 0, 1.0, (2.0 * intersection) / f1_denominator)

    union = np.logical_or(pred_np, target_np).sum(axis=1) if pred_np.size else np.array([])
    jaccard = np.where(union == 0, 1.0, intersection / union)

    return {
        'exact_match_accuracy': exact_match_accuracy,
        'sample_accuracy_mean': sample_accuracy_mean,
        'sample_accuracy_variance': sample_accuracy_variance,
        'sample_f1': float(sample_f1.mean()) if sample_f1.size else 0.0,
        'jaccard_index': float(jaccard.mean()) if jaccard.size else 0.0,
        'hamming_loss': float(hamming_loss(target_np, pred_np)),
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