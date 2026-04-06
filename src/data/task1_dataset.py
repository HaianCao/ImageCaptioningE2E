"""
Dataset cho object/attribute classification.

Mỗi sample là một ROI crop tương ứng với 1 object trong Visual Genome:
- Input: Ảnh ROI đã crop từ bounding box
- Output 1: Object class index (single-label)
- Output 2: Attribute label vector (multi-label binary)

Hiện chỉ hỗ trợ image mode on-the-fly.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from PIL import Image

from .dataset import BaseVGDataset, load_vocab, load_json
from .preprocessing_strategies import build_preprocessing_preset


def _normalize_object_input_mode(input_mode: str) -> str:
    mode = str(input_mode).lower().strip()
    aliases = {
        "grayscale": "gray",
        "grey": "gray",
        "edge": "contour",
        "edges": "contour",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"rgb", "gray", "contour"}:
        raise ValueError("object_input_mode phải là 'rgb', 'gray', hoặc 'contour'")
    return mode


def _prepare_input_image(image: Image.Image, input_mode: str) -> Image.Image:
    """Convert a cropped ROI into the requested input mode before augmentation."""
    mode = _normalize_object_input_mode(input_mode)

    if mode == "rgb":
        return image.convert("RGB")

    if mode == "gray":
        return image.convert("L").convert("RGB")

    if mode == "contour":
        import cv2

        gray_image = np.array(image.convert("L"))
        edges = cv2.Canny(gray_image, 100, 200)
        return Image.fromarray(edges).convert("RGB")

    raise ValueError(f"Unsupported input_mode: {input_mode}")


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
        split: "train", "val", hoặc "test"
        max_samples: Giới hạn số samples (debug)

    Returns:
        dict với keys: "image", "object_label", "attribute_labels", "meta"
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        object_vocab: Dict[str, int],
        attribute_vocab: Dict[str, int],
        transform=None,
        input_mode: str = "rgb",
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
        self.input_mode = _normalize_object_input_mode(input_mode)

        # Load annotation
        self.annotation_file = Path(annotation_file)
        self._load_annotations()

        # Giới hạn samples nếu cần
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            print(f"[ObjectAttributeDataset] Giới hạn {max_samples} samples (debug mode)")

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
        print(f"[ObjectAttributeDataset] Loaded {len(self.samples)} annotations từ {self.annotation_file}")

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
                - "image": tensor ảnh ROI
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

        image = self._load_image(sample["image_id"])
        x, y, w, h = sample["bbox"]
        roi = self._crop_roi(image, x, y, w, h, padding=0)
        roi = _prepare_input_image(roi, self.input_mode)

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
        return (
            f"ObjectAttributeDataset [{self.split}]: "
            f"{len(self.samples)} ROIs | {n_images} images | "
            f"{self.num_objects} objects | {self.num_attributes} attributes"
        )


def build_task1_datasets(
    processed_dir: str,
    image_dir: str,
    roi_size: int = 224,
    input_mode: str = "rgb",
    object_input_mode: Optional[str] = None,
    max_samples: Optional[int] = None,
    train_horizontal_flip_p: float = 0.5,
    train_color_jitter: bool = True,
    train_brightness: float = 0.2,
    train_contrast: float = 0.2,
    train_saturation: float = 0.2,
    train_hue: float = 0.1,
    train_random_erasing_p: float = 0.0,
    train_resize_delta: int = 32,
    mean: List[float] = None,
    std: List[float] = None,
    preprocessing_strategy: str = "baseline_task1",
) -> Tuple["ObjectAttributeDataset", "ObjectAttributeDataset", "ObjectAttributeDataset"]:
    """
    Tạo train/val/test datasets cho Task 1 từ processed directory.

    Args:
        processed_dir: Thư mục chứa annotation JSON + vocab files
        image_dir: Thư mục ảnh gốc
        roi_size: Kích thước ROI crop
        max_samples: Giới hạn samples (debug)

    Returns:
        Tuple (train_dataset, val_dataset, test_dataset)
    """
    proc_path = Path(processed_dir)

    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    if object_input_mode is not None:
        input_mode = object_input_mode
    input_mode = _normalize_object_input_mode(input_mode)

    object_vocab = load_vocab(str(proc_path / "object_vocab.json"))
    attribute_vocab = load_vocab(str(proc_path / "attribute_vocab.json"))

    preprocessing = build_preprocessing_preset(
        preprocessing_strategy,
        roi_size=roi_size,
        mean=mean,
        std=std,
        train_horizontal_flip_p=train_horizontal_flip_p,
        train_color_jitter=train_color_jitter,
        train_brightness=train_brightness,
        train_contrast=train_contrast,
        train_saturation=train_saturation,
        train_hue=train_hue,
        train_random_erasing_p=train_random_erasing_p,
        train_resize_delta=train_resize_delta,
    )
    train_transform = preprocessing.train
    val_transform = preprocessing.val

    def _make_ds(split, transform):
        return ObjectAttributeDataset(
            annotation_file=str(proc_path / split / "annotations.json"),
            image_dir=image_dir,
            object_vocab=object_vocab,
            attribute_vocab=attribute_vocab,
            transform=transform,
            input_mode=input_mode,
            split=split,
            max_samples=max_samples,
        )

    train_ds = _make_ds("train", train_transform)
    val_ds = _make_ds("val", val_transform)
    test_ds = _make_ds("test", val_transform)

    return train_ds, val_ds, test_ds
