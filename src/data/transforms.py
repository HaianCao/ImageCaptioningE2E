"""
Image Transforms / Augmentation Pipeline cho Visual Genome ROI crops.

Cung cấp:
- get_train_transforms(): Augmentation cho training
- get_val_transforms(): Normalization only cho validation/test
- get_feature_transforms(): Chỉ resize + normalize, dùng khi trích xuất features
"""

from typing import Tuple, List, Optional
from torchvision import transforms


# ImageNet statistics (dùng cho pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    resize_delta: int = 32,
    horizontal_flip_p: float = 0.5,
    color_jitter: bool = True,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    random_erasing_p: float = 0.0,
) -> transforms.Compose:
    """
    Tạo augmentation pipeline cho training.

    Args:
        roi_size: Kích thước ảnh đầu vào (chiều dài/rộng)
        mean, std: ImageNet statistics cho normalization
        horizontal_flip_p: Xác suất flip ngang
        color_jitter: Bật/tắt color jitter
        brightness/contrast/saturation/hue: Cường độ color jitter
        random_erasing_p: Xác suất random erasing (0 = tắt)

    Returns:
        transforms.Compose pipeline
    """
    aug_list = [
        transforms.Resize((roi_size + resize_delta, roi_size + resize_delta)),  # Resize lớn hơn rồi crop
        transforms.RandomCrop(roi_size),
        transforms.RandomHorizontalFlip(p=horizontal_flip_p),
    ]

    if color_jitter:
        aug_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    if random_erasing_p > 0:
        aug_list.append(transforms.RandomErasing(p=random_erasing_p))

    return transforms.Compose(aug_list)


def get_val_transforms(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Transform pipeline cho validation và test (không augmentation).

    Args:
        roi_size: Kích thước ảnh đầu ra
        mean, std: ImageNet statistics

    Returns:
        transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.Resize((roi_size, roi_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_feature_transforms(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Transform đơn giản dùng khi pre-extract features (không augmentation).
    Giống val_transforms nhưng tách riêng để ngữ nghĩa rõ ràng.
    """
    return get_val_transforms(roi_size=roi_size, mean=mean, std=std)


def denormalize(
    tensor,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
):
    """
    Đảo ngược normalization để visualize ảnh.

    Args:
        tensor: Tensor ảnh đã normalize (C, H, W)
        mean, std: Giá trị đã dùng khi normalize

    Returns:
        Tensor ảnh gốc (C, H, W), giá trị trong [0, 1]
    """
    import torch
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    return tensor * std_t + mean_t
