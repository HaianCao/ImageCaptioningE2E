"""Explicit feature extraction wrapper for the relation pipeline."""

from .feature_extractor import extract_task2_features


def extract_relation_features(*args, **kwargs):
    return extract_task2_features(*args, **kwargs)


__all__ = ["extract_relation_features"]