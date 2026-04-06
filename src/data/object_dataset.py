"""Explicit dataset wrapper for the object pipeline."""

from .task1_dataset import ObjectAttributeDataset, build_task1_datasets


def build_object_datasets(*args, **kwargs):
    return build_task1_datasets(*args, **kwargs)


__all__ = ["ObjectAttributeDataset", "build_object_datasets"]