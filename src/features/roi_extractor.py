"""
ROI (Region of Interest) Extractor từ Bounding Box.

Cung cấp các hàm tiện ích để:
- Crop ROI từ ảnh PIL theo bounding box [x, y, w, h]
- Tính union bounding box của 2 đối tượng
- Tính IoU giữa 2 bounding boxes
- Tính spatial feature vector giữa 2 đối tượng
"""

from typing import Tuple, List, Optional
from PIL import Image


BBox = Tuple[int, int, int, int]  # (x, y, w, h)


def extract_roi(
    image: Image.Image,
    bbox: BBox,
    target_size: Optional[Tuple[int, int]] = None,
    padding: int = 0,
) -> Image.Image:
    """
    Crop và resize ROI từ ảnh theo bounding box.

    Args:
        image: PIL Image gốc (RGB)
        bbox: (x, y, w, h) - tọa độ góc trên-trái + kích thước
        target_size: (width, height) để resize. None = không resize
        padding: Số pixels thêm vào 4 cạnh

    Returns:
        PIL.Image ROI đã crop (và resize nếu chỉ định)
    """
    x, y, w, h = bbox
    img_w, img_h = image.size

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    if x2 <= x1 or y2 <= y1:
        roi = image
    else:
        roi = image.crop((x1, y1, x2, y2))

    if target_size is not None:
        roi = roi.resize(target_size, Image.BILINEAR)

    return roi


def extract_union_roi(
    image: Image.Image,
    bbox1: BBox,
    bbox2: BBox,
    target_size: Optional[Tuple[int, int]] = None,
    padding: int = 10,
) -> Image.Image:
    """
    Crop vùng union bao phủ cả 2 bounding boxes (dùng cho Task 2).

    Args:
        image: PIL Image gốc
        bbox1: (x, y, w, h) subject bbox
        bbox2: (x, y, w, h) object bbox
        target_size: Kích thước resize đầu ra
        padding: Pixels padding xung quanh union

    Returns:
        PIL.Image chứa vùng ảnh bao phủ cả 2 đối tượng
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

    if x2 <= x1 or y2 <= y1:
        roi = image
    else:
        roi = image.crop((x1, y1, x2, y2))

    if target_size is not None:
        roi = roi.resize(target_size, Image.BILINEAR)

    return roi


def compute_iou(bbox1: BBox, bbox2: BBox) -> float:
    """Tính Intersection over Union (IoU) giữa 2 bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]
    union_area = area1 + area2 - inter_area

    return inter_area / (union_area + 1e-6)


def compute_spatial_features(
    subj_bbox: BBox,
    obj_bbox: BBox,
    img_w: int,
    img_h: int,
) -> List[float]:
    """
    Tính spatial feature vector (9 dims) từ 2 bboxes.
    [subj_cx, subj_cy, subj_w, subj_h, obj_cx, obj_cy, obj_w, obj_h, iou]
    """
    def _center(bbox, iw, ih):
        x, y, w, h = bbox
        return [(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih]

    return _center(subj_bbox, img_w, img_h) + _center(obj_bbox, img_w, img_h) + [compute_iou(subj_bbox, obj_bbox)]
