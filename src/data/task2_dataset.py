"""
Dataset cho relation classification.

Mỗi sample là một cặp (subject, object) với union bounding box:
- Input: Ảnh union ROI crop chứa cả 2 đối tượng
- Output: Relation/predicate class index

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


def _normalize_input_mode(input_mode: str) -> str:
    mode = str(input_mode).lower().strip()
    aliases = {
        "grayscale": "gray",
        "grey": "gray",
        "edge": "contour",
        "edges": "contour",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"rgb", "gray", "contour"}:
        raise ValueError("input_mode phải là 'rgb', 'gray', hoặc 'contour'")
    return mode


def _prepare_input_image(image: Image.Image, input_mode: str) -> Image.Image:
    """Convert a cropped union ROI into the requested input mode before augmentation."""
    mode = _normalize_input_mode(input_mode)

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


class RelationshipDataset(BaseVGDataset):
    """
    Dataset cho Task 2: phân loại mối quan hệ giữa cặp đối tượng.

    Args:
        annotation_file: File JSON chứa processed annotations Task 2
            Format mỗi record: {
                "image_id": int,
                "relationship_id": int,
                "subject_bbox": [x, y, w, h],
                "object_bbox": [x, y, w, h],
                "subject_name": str,
                "object_name": str,
                "predicate": str,
                "relation_label": int   # index trong relation_vocab
            }
        image_dir: Thư mục ảnh gốc
        relation_vocab: Dict {predicate_name: index}
        transform: torchvision transform pipeline
        use_spatial_features: Append spatial geometric features
        split: "train", "val", "test"
        max_samples: Giới hạn samples (debug)

    Returns:
        dict: "image", "spatial", "relation_label", "meta"
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        relation_vocab: Dict[str, int],
        transform=None,
        use_spatial_features: bool = True,
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

        self.relation_vocab = relation_vocab
        self.num_relations = len(relation_vocab)
        self.use_spatial_features = use_spatial_features
        self.input_mode = _normalize_input_mode(input_mode)
        self.split = split
        self.annotation_file = Path(annotation_file)

        self._load_annotations()

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        print(self.summary())

    def _load_annotations(self) -> None:
        """Load annotation từ processed JSON file."""
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy: {self.annotation_file}\n"
                f"Hãy chạy notebook 02_data_processing.ipynb trước."
            )
        raw = load_json(str(self.annotation_file))
        self.samples = raw if isinstance(raw, list) else raw.get("samples", [])
        print(f"[RelationshipDataset] Loaded {len(self.samples)} pairs từ {self.annotation_file}")

    def _compute_spatial_features(
        self,
        subj_bbox: List[int],
        obj_bbox: List[int],
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        """
        Tính spatial features từ 2 bounding boxes.

        Features (9 dims):
            [subj_cx_norm, subj_cy_norm, subj_w_norm, subj_h_norm,
             obj_cx_norm, obj_cy_norm, obj_w_norm, obj_h_norm,
             iou]

        Args:
            subj_bbox, obj_bbox: [x, y, w, h]
            img_w, img_h: Kích thước ảnh gốc

        Returns:
            Tensor shape (9,)
        """
        def normalize_bbox(bbox):
            x, y, w, h = bbox
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            return [cx, cy, nw, nh]

        def compute_iou(b1, b2):
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[0] + b1[2], b2[0] + b2[2])
            y2 = min(b1[1] + b1[3], b2[1] + b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = b1[2] * b1[3]
            area2 = b2[2] * b2[3]
            union = area1 + area2 - inter
            return inter / (union + 1e-6)

        subj_norm = normalize_bbox(subj_bbox)
        obj_norm = normalize_bbox(obj_bbox)
        iou = compute_iou(subj_bbox, obj_bbox)

        feats = subj_norm + obj_norm + [iou]
        return torch.tensor(feats, dtype=torch.float32)

    def _get_image_size(self, sample: Dict, image=None) -> Tuple[int, int]:
        """Get image size from annotation metadata or loaded image."""
        img_info = sample.get("image_info") or {}
        width = img_info.get("width")
        height = img_info.get("height")

        if width is not None and height is not None:
            return int(width), int(height)

        if image is not None:
            return image.size

        return 800, 600

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict:
                - "image": union ROI ảnh
                - "spatial": tensor shape (9,) - spatial geometric features
                - "relation_label": tensor scalar (long)
                - "meta": dict với image_id, relationship_id, subject/object info
        """
        sample = self.samples[idx]
        relation_label = torch.tensor(sample["relation_label"], dtype=torch.long)

        meta = {
            "image_id": sample["image_id"],
            "relationship_id": sample.get("relationship_id", -1),
            "subject_name": sample.get("subject_name", ""),
            "object_name": sample.get("object_name", ""),
            "predicate": sample.get("predicate", ""),
        }

        subj_bbox = sample["subject_bbox"]
        obj_bbox = sample["object_bbox"]

        image = self._load_image(sample["image_id"])
        img_w, img_h = self._get_image_size(sample, image)

        union_roi = self._crop_union_roi(image, subj_bbox, obj_bbox, padding=10)
        union_roi = _prepare_input_image(union_roi, self.input_mode)

        if self.transform:
            union_roi = self.transform(union_roi)

        spatial = torch.zeros(9)
        if self.use_spatial_features:
            spatial = self._compute_spatial_features(subj_bbox, obj_bbox, img_w, img_h)

        return {
            "image": union_roi,
            "spatial": spatial,
            "relation_label": relation_label,
            "meta": meta,
        }

    def summary(self) -> str:
        n_images = len(self.get_image_ids())
        return (
            f"RelationshipDataset [{self.split}]: "
            f"{len(self.samples)} pairs | {n_images} images | "
            f"{self.num_relations} relations | spatial={'✅' if self.use_spatial_features else '❌'}"
        )


def build_task2_datasets(
    processed_dir: str,
    image_dir: str,
    roi_size: int = 224,
    use_spatial_features: bool = True,
    input_mode: str = "rgb",
    max_samples: Optional[int] = None,
    train_horizontal_flip_p: float = 0.5,
    train_color_jitter: bool = True,
    train_brightness: float = 0.2,
    train_contrast: float = 0.2,
    train_saturation: float = 0.1,
    train_hue: float = 0.05,
    train_random_erasing_p: float = 0.0,
    train_resize_delta: int = 32,
    mean: List[float] = None,
    std: List[float] = None,
    preprocessing_strategy: str = "baseline_task2",
) -> Tuple["RelationshipDataset", "RelationshipDataset", "RelationshipDataset"]:
    """
    Tạo train/val/test datasets cho Task 2.

    Args:
        processed_dir: Thư mục processed Task 2
        image_dir: Thư mục ảnh gốc
        roi_size: Kích thước union ROI crop
        use_spatial_features: Sử dụng spatial geometric features
        max_samples: Giới hạn (debug)

    Returns:
        Tuple (train_ds, val_ds, test_ds)
    """
    proc_path = Path(processed_dir)

    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    relation_vocab = load_vocab(str(proc_path / "relation_vocab.json"))
    input_mode = _normalize_input_mode(input_mode)

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
        return RelationshipDataset(
            annotation_file=str(proc_path / split / "annotations.json"),
            image_dir=image_dir,
            relation_vocab=relation_vocab,
            transform=transform,
            use_spatial_features=use_spatial_features,
            input_mode=input_mode,
            split=split,
            max_samples=max_samples,
        )

    train_ds = _make_ds("train", train_transform)
    val_ds = _make_ds("val", val_transform)
    test_ds = _make_ds("test", val_transform)

    return train_ds, val_ds, test_ds
