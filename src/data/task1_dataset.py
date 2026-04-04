"""
Dataset cho Task 1: Object & Attribute Classification.

Mỗi sample là một ROI crop tương ứng với 1 object trong Visual Genome:
- Input: Ảnh ROI đã crop từ bounding box
- Output 1: Object class index (single-label)
- Output 2: Attribute label vector (multi-label binary)

Hỗ trợ 2 chế độ:
- Image mode: Load ảnh và crop ROI on-the-fly
- Feature cache mode: Load pre-extracted feature vectors từ .pt file
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from .dataset import BaseVGDataset, load_vocab, load_json
from .transforms import get_train_transforms, get_val_transforms


class ObjectAttributeDataset(BaseVGDataset):
    """
    Dataset cho Task 1: nhận diện đối tượng + phân loại thuộc tính.

    Args:
        annotation_file: File JSON chứa processed annotations Task 1
            Format mỗi record: {
                "image_id": int,
                "object_id": int,
                "bbox": [x, y, w, h],
                "object_label": int,       # index trong object_vocab
                "attribute_labels": [int]  # list indices trong attribute_vocab
            }
        image_dir: Thư mục chứa ảnh gốc
        object_vocab: Dict {object_name: index}
        attribute_vocab: Dict {attribute_name: index}
        transform: torchvision transform pipeline
        feature_cache_file: Nếu set, load features từ .pt thay vì ảnh
        split: "train", "val", hoặc "test"
        max_samples: Giới hạn số samples (debug)

    Returns (khi không dùng cache):
        dict với keys: "image", "object_label", "attribute_labels", "meta"

    Returns (khi dùng feature cache):
        dict với keys: "feature", "object_label", "attribute_labels", "meta"
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        object_vocab: Dict[str, int],
        attribute_vocab: Dict[str, int],
        transform=None,
        feature_cache_file: Optional[str] = None,
        split: str = "train",
        max_samples: Optional[int] = None,
        cache_images: bool = False,
    ):
        super().__init__(
            image_dir=image_dir,
            transform=transform,
            cache_images=cache_images,
            max_samples=max_samples,
        )

        self.object_vocab = object_vocab
        self.attribute_vocab = attribute_vocab
        self.num_objects = len(object_vocab)
        self.num_attributes = len(attribute_vocab)
        self.split = split
        self.feature_cache_file = feature_cache_file

        # Load annotation
        self.annotation_file = Path(annotation_file)
        self._load_annotations()

        # Load feature cache nếu có
        self._feature_cache: Optional[Dict] = None
        if feature_cache_file and Path(feature_cache_file).exists():
            print(f"[Task1Dataset] Loading feature cache: {feature_cache_file}")
            self._feature_cache = torch.load(feature_cache_file, map_location="cpu")
            print(f"[Task1Dataset] Loaded {len(self._feature_cache)} cached features")

        # Giới hạn samples nếu cần
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            print(f"[Task1Dataset] Giới hạn {max_samples} samples (debug mode)")

        print(self.summary())

    def _load_annotations(self) -> None:
        """Load annotation từ processed JSON file."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy annotation: {self.annotation_file}\n"
                f"Hãy chạy notebook 02_data_processing.ipynb trước."
            )

        raw = load_json(str(self.annotation_file))
        self.samples = raw if isinstance(raw, list) else raw.get("samples", [])
        print(f"[Task1Dataset] Loaded {len(self.samples)} annotations từ {self.annotation_file}")

    def _make_attribute_vector(self, attribute_indices: List[int]) -> torch.Tensor:
        """
        Tạo binary multi-label vector cho attributes.

        Args:
            attribute_indices: List các attribute index

        Returns:
            Tensor shape (num_attributes,) với 0/1 values
        """
        vec = torch.zeros(self.num_attributes, dtype=torch.float32)
        for idx in attribute_indices:
            if 0 <= idx < self.num_attributes:
                vec[idx] = 1.0
        return vec

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict:
                - "feature" hoặc "image": tensor đặc trưng hoặc ảnh ROI
                - "object_label": tensor scalar (long) - class index
                - "attribute_labels": tensor (num_attributes,) - binary vector
                - "meta": dict chứa image_id, object_id, bbox
        """
        sample = self.samples[idx]
        object_label = torch.tensor(sample["object_label"], dtype=torch.long)
        attribute_labels = self._make_attribute_vector(sample.get("attribute_labels", []))

        meta = {
            "image_id": sample["image_id"],
            "object_id": sample.get("object_id", -1),
            "bbox": sample["bbox"],
        }

        # Chế độ feature cache
        if self._feature_cache is not None:
            key = str(sample.get("object_id", idx))
            if key not in self._feature_cache:
                key = str(idx)
            feature = self._feature_cache.get(key, torch.zeros(2048))
            return {
                "feature": feature,
                "object_label": object_label,
                "attribute_labels": attribute_labels,
                "meta": meta,
            }

        # Chế độ image on-the-fly
        image = self._load_image(sample["image_id"])
        x, y, w, h = sample["bbox"]
        roi = self._crop_roi(image, x, y, w, h, padding=5)

        if self.transform:
            roi = self.transform(roi)

        return {
            "image": roi,
            "object_label": object_label,
            "attribute_labels": attribute_labels,
            "meta": meta,
        }

    def summary(self) -> str:
        n_images = len(self.get_image_ids())
        has_cache = self._feature_cache is not None
        return (
            f"ObjectAttributeDataset [{self.split}]: "
            f"{len(self.samples)} ROIs | {n_images} images | "
            f"{self.num_objects} objects | {self.num_attributes} attributes | "
            f"cache={'✅' if has_cache else '❌'}"
        )


def build_task1_datasets(
    processed_dir: str,
    image_dir: str,
    roi_size: int = 224,
    use_feature_cache: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple["ObjectAttributeDataset", "ObjectAttributeDataset", "ObjectAttributeDataset"]:
    """
    Tạo train/val/test datasets cho Task 1 từ processed directory.

    Args:
        processed_dir: Thư mục chứa annotation JSON + vocab files
        image_dir: Thư mục ảnh gốc
        roi_size: Kích thước ROI crop
        use_feature_cache: Dùng cached features nếu có
        max_samples: Giới hạn samples (debug)

    Returns:
        Tuple (train_dataset, val_dataset, test_dataset)
    """
    proc_path = Path(processed_dir)

    object_vocab = load_vocab(str(proc_path / "object_vocab.json"))
    attribute_vocab = load_vocab(str(proc_path / "attribute_vocab.json"))

    train_transform = get_train_transforms(roi_size=roi_size)
    val_transform = get_val_transforms(roi_size=roi_size)

    def _make_ds(split, transform):
        cache_file = None
        if use_feature_cache:
            cache_path = proc_path / "features" / f"{split}_features.pt"
            if cache_path.exists():
                cache_file = str(cache_path)

        return ObjectAttributeDataset(
            annotation_file=str(proc_path / split / "annotations.json"),
            image_dir=image_dir,
            object_vocab=object_vocab,
            attribute_vocab=attribute_vocab,
            transform=transform,
            feature_cache_file=cache_file,
            split=split,
            max_samples=max_samples,
        )

    train_ds = _make_ds("train", train_transform)
    val_ds = _make_ds("val", val_transform)
    test_ds = _make_ds("test", val_transform)

    return train_ds, val_ds, test_ds
