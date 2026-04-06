"""Data subpackage: loading, preprocessing, and dataset utilities."""

from .download import download_and_extract_metadata, download_vg_images, verify_download
from .dataset import BaseVGDataset
from .preprocessing_strategies import (
	PREPROCESSING_STRATEGY_REGISTRY,
	PreprocessingPreset,
	PreprocessingStrategy,
	Task1PreprocessingStrategy,
	Task2PreprocessingStrategy,
	build_preprocessing_preset,
	get_preprocessing_strategy,
	register_preprocessing_strategy,
)
from .task2_dataset import RelationshipDataset
from .object_dataset import build_object_datasets
from .attribute_dataset import build_attribute_datasets
from .relation_dataset import build_relation_datasets
from .transforms import get_train_transforms, get_val_transforms
from .preprocessing import (
	build_vocab_and_splits,
	preprocess_task1,
	preprocess_task2,
	preprocess_object_attribute,
	preprocess_relation,
	build_object_attribute_vocab_and_splits,
	build_relation_vocab_and_splits,
)
