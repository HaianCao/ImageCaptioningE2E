"""Explicit dataset wrapper for the relation pipeline."""

from .task2_dataset import RelationshipDataset, build_task2_datasets


def build_relation_datasets(*args, **kwargs):
    return build_task2_datasets(*args, **kwargs)


__all__ = ["RelationshipDataset", "build_relation_datasets"]