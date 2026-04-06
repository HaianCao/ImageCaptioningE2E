"""Explicit dataset wrapper for the attribute pipeline."""

from .task1_dataset import ObjectAttributeDataset, build_task1_datasets


def build_attribute_datasets(*args, **kwargs):
    return build_task1_datasets(*args, **kwargs)


__all__ = ["ObjectAttributeDataset", "build_attribute_datasets"]