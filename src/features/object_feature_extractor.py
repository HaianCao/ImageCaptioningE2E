"""Explicit feature extraction wrapper for the object pipeline."""

from .feature_extractor import extract_task1_features


def extract_object_features(*args, **kwargs):
    return extract_task1_features(*args, **kwargs)


__all__ = ["extract_object_features"]