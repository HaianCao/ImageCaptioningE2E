"""
Base Dataset class cho Visual Genome.

Cung cấp:
- Load ảnh từ disk theo image_id
- Crop ROI từ Bounding Box
- Caching ảnh trong memory (optional)
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BaseVGDataset(Dataset, ABC):
    """
    Abstract base dataset cho Visual Genome.

    Subclass phải implement:
        - __len__()
        - __getitem__()
        - _load_annotations()

    Args:
        image_dir: Thư mục chứa ảnh gốc (.jpg)
        transform: torchvision transforms pipeline
        cache_images: Nếu True, cache toàn bộ ảnh vào RAM (cẩn thận với dataset lớn)
        max_samples: Giới hạn số lượng samples (dùng cho debug)
    """

    def __init__(
        self,
        image_dir: str,
        transform=None,
        cache_images: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.cache_images = cache_images
        self.max_samples = max_samples

        self._image_cache: Dict[int, Image.Image] = {}
        self.samples: List[dict] = []  # Populated by subclass

    @abstractmethod
    def _load_annotations(self) -> None:
        """Load và parse annotation files, populate self.samples."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_id: int) -> Image.Image:
        """
        Load ảnh PIL từ disk hoặc cache.

        Args:
            image_id: ID của ảnh trong Visual Genome

        Returns:
            PIL.Image in RGB mode
        """
        if self.cache_images and image_id in self._image_cache:
            return self._image_cache[image_id]

        img_path = self.image_dir / f"{image_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy ảnh: {img_path}. "
                f"Hãy chạy notebook notebooks/complete_pipeline.ipynb hoặc đảm bảo ảnh đã được tải về."
            )

        image = Image.open(img_path).convert("RGB")

        if self.cache_images:
            self._image_cache[image_id] = image

        return image

    def _crop_roi(
        self,
        image: Image.Image,
        x: int,
        y: int,
        w: int,
        h: int,
        padding: int = 0,
    ) -> Image.Image:
        """
        Crop vùng ROI từ ảnh theo Bounding Box.

        Args:
            image: PIL Image gốc
            x, y: Tọa độ góc trên-trái của bbox
            w, h: Chiều rộng và cao của bbox
            padding: Số pixels padding thêm xung quanh bbox

        Returns:
            PIL.Image đã crop
        """
        img_w, img_h = image.size
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        return image.crop((x1, y1, x2, y2))

    def _crop_union_roi(
        self,
        image: Image.Image,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        padding: int = 10,
    ) -> Image.Image:
        """
        Crop vùng bao phủ cả 2 bounding boxes (union).

        Args:
            image: PIL Image gốc
            bbox1: (x, y, w, h) của đối tượng 1
            bbox2: (x, y, w, h) của đối tượng 2
            padding: Số pixels padding

        Returns:
            PIL.Image chứa cả 2 đối tượng
        """
        img_w, img_h = image.size

        x1_min = min(bbox1[0], bbox2[0])
        y1_min = min(bbox1[1], bbox2[1])
        x2_max = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2_max = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

        x1 = max(0, x1_min - padding)
        y1 = max(0, y1_min - padding)
        x2 = min(img_w, x2_max + padding)
        y2 = min(img_h, y2_max + padding)

        return image.crop((x1, y1, x2, y2))

    def get_image_ids(self) -> List[int]:
        """Trả về danh sách image_id duy nhất trong dataset."""
        return list(set(s["image_id"] for s in self.samples))

    def summary(self) -> str:
        """Thống kê ngắn về dataset."""
        n_images = len(self.get_image_ids())
        return (
            f"{self.__class__.__name__}: "
            f"{len(self.samples)} samples | {n_images} images"
        )


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """
    Load vocabulary JSON: {label_name: label_idx}.

    Args:
        vocab_path: Đường dẫn file JSON vocab

    Returns:
        Dict mapping label_string -> int index
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab


def load_json(json_path: str) -> any:
    """Load JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
