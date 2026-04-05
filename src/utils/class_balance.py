"""
Class-balance helpers for long-tail classification tasks.

These utilities derive loss weights from the training split so the trainers can
use automatic balancing without manual tuning in the notebook.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch


def _extract_samples(dataset: Any) -> List[dict]:
    """Best-effort extraction of raw sample dictionaries from a dataset-like object."""
    if dataset is None:
        return []

    if hasattr(dataset, "samples") and isinstance(getattr(dataset, "samples"), list):
        return list(getattr(dataset, "samples"))

    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_samples = _extract_samples(getattr(dataset, "dataset"))
        if base_samples:
            return [base_samples[int(index)] for index in getattr(dataset, "indices")]

    if hasattr(dataset, "dataset"):
        return _extract_samples(getattr(dataset, "dataset"))

    if hasattr(dataset, "datasets"):
        samples: List[dict] = []
        for sub_dataset in getattr(dataset, "datasets"):
            samples.extend(_extract_samples(sub_dataset))
        return samples

    return []


def _normalize_weights(weights: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    """Normalize observed weights to have mean 1.0 while keeping unseen classes neutral."""
    if observed_mask.any():
        observed_weights = weights[observed_mask]
        mean_value = observed_weights.mean().clamp(min=1e-6)
        weights = weights.clone()
        weights[observed_mask] = observed_weights / mean_value
    return weights


def compute_single_label_class_weights(
    dataset: Any,
    num_classes: int,
    label_key: str,
    *,
    power: float = 0.5,
    clip_min: float = 0.25,
    clip_max: float = 10.0,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Compute inverse-frequency class weights for single-label classification."""
    samples = _extract_samples(dataset)
    if not samples:
        return None

    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample in samples:
        label = sample.get(label_key)
        if label is None:
            continue
        label_index = int(label)
        if 0 <= label_index < num_classes:
            counts[label_index] += 1.0

    weights = torch.ones(num_classes, dtype=torch.float32)
    observed_mask = counts > 0
    if observed_mask.any():
        total_observed = counts[observed_mask].sum().clamp(min=1.0)
        inverse_frequency = (total_observed / counts[observed_mask]).pow(power)
        inverse_frequency = inverse_frequency.clamp(min=clip_min, max=clip_max)
        weights[observed_mask] = inverse_frequency
        weights = _normalize_weights(weights, observed_mask)

    if device is not None:
        weights = weights.to(device)
    return weights


def compute_multilabel_pos_weight(
    dataset: Any,
    num_labels: int,
    label_key: str,
    *,
    power: float = 0.5,
    clip_min: float = 1.0,
    clip_max: float = 20.0,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Compute positive-class weights for multi-label BCE from positive label counts."""
    samples = _extract_samples(dataset)
    if not samples:
        return None

    counts = torch.zeros(num_labels, dtype=torch.float32)
    total_samples = 0
    for sample in samples:
        labels = sample.get(label_key, []) or []
        total_samples += 1
        for label in labels:
            label_index = int(label)
            if 0 <= label_index < num_labels:
                counts[label_index] += 1.0

    pos_weight = torch.ones(num_labels, dtype=torch.float32)
    observed_mask = counts > 0
    if observed_mask.any():
        negatives = (total_samples - counts[observed_mask]).clamp(min=1.0)
        positives = counts[observed_mask].clamp(min=1.0)
        ratio = (negatives / positives).pow(power)
        ratio = ratio.clamp(min=clip_min, max=clip_max)
        pos_weight[observed_mask] = ratio

    if device is not None:
        pos_weight = pos_weight.to(device)
    return pos_weight