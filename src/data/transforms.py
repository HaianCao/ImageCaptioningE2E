"""
Image Transforms / Augmentation Pipeline cho Visual Genome ROI crops.

Mỗi transform pipeline được bọc trong một class facade callable để notebook và
dataset builder chỉ làm việc với một interface thống nhất.
"""

from abc import ABC, abstractmethod
from typing import List
from torchvision import transforms


# ImageNet statistics (dùng cho pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TransformFacade(ABC):
    """Callable facade around a torchvision Compose pipeline."""

    def __init__(self) -> None:
        self._compose = self._build_compose()

    @abstractmethod
    def _build_compose(self) -> transforms.Compose:
        """Build the underlying torchvision pipeline."""

    def __call__(self, image):
        return self._compose(image)

    def __getattr__(self, name):
        return getattr(self._compose, name)

    @property
    def compose(self) -> transforms.Compose:
        return self._compose


def _append_color_jitter(
    aug_list,
    enabled: bool,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> None:
    if enabled:
        aug_list.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        )


def _append_random_erasing(aug_list, random_erasing_p: float) -> None:
    if random_erasing_p > 0:
        aug_list.append(transforms.RandomErasing(p=random_erasing_p))


def _build_generic_train_pipeline(
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
    aug_list = [
        transforms.Resize((roi_size + resize_delta, roi_size + resize_delta)),
        transforms.RandomCrop(roi_size),
        transforms.RandomHorizontalFlip(p=horizontal_flip_p),
    ]

    _append_color_jitter(aug_list, color_jitter, brightness, contrast, saturation, hue)

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    _append_random_erasing(aug_list, random_erasing_p)
    return transforms.Compose(aug_list)


def _build_object_train_pipeline(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    horizontal_flip_p: float = 0.5,
    color_jitter: bool = True,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    random_erasing_p: float = 0.0,
) -> transforms.Compose:
    aug_list = [
        transforms.Resize((roi_size, roi_size)),
        transforms.RandomHorizontalFlip(p=horizontal_flip_p),
    ]

    _append_color_jitter(aug_list, color_jitter, brightness, contrast, saturation, hue)

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    _append_random_erasing(aug_list, random_erasing_p)
    return transforms.Compose(aug_list)


def _build_relation_train_pipeline(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    color_jitter: bool = False,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    random_erasing_p: float = 0.0,
) -> transforms.Compose:
    aug_list = [
        transforms.Resize((roi_size, roi_size)),
    ]

    _append_color_jitter(aug_list, color_jitter, brightness, contrast, saturation, hue)

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    _append_random_erasing(aug_list, random_erasing_p)
    return transforms.Compose(aug_list)


def _build_eval_pipeline(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((roi_size, roi_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


class TrainTransformFacade(TransformFacade):
    def __init__(
        self,
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
    ) -> None:
        self.roi_size = roi_size
        self.mean = mean
        self.std = std
        self.resize_delta = resize_delta
        self.horizontal_flip_p = horizontal_flip_p
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_erasing_p = random_erasing_p
        super().__init__()

    def _build_compose(self) -> transforms.Compose:
        return _build_generic_train_pipeline(
            roi_size=self.roi_size,
            mean=self.mean,
            std=self.std,
            resize_delta=self.resize_delta,
            horizontal_flip_p=self.horizontal_flip_p,
            color_jitter=self.color_jitter,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
            random_erasing_p=self.random_erasing_p,
        )


class ObjectTrainTransformFacade(TransformFacade):
    def __init__(
        self,
        roi_size: int = 224,
        mean: List[float] = IMAGENET_MEAN,
        std: List[float] = IMAGENET_STD,
        horizontal_flip_p: float = 0.5,
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        random_erasing_p: float = 0.0,
    ) -> None:
        self.roi_size = roi_size
        self.mean = mean
        self.std = std
        self.horizontal_flip_p = horizontal_flip_p
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_erasing_p = random_erasing_p
        super().__init__()

    def _build_compose(self) -> transforms.Compose:
        return _build_object_train_pipeline(
            roi_size=self.roi_size,
            mean=self.mean,
            std=self.std,
            horizontal_flip_p=self.horizontal_flip_p,
            color_jitter=self.color_jitter,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
            random_erasing_p=self.random_erasing_p,
        )


class RelationTrainTransformFacade(TransformFacade):
    def __init__(
        self,
        roi_size: int = 224,
        mean: List[float] = IMAGENET_MEAN,
        std: List[float] = IMAGENET_STD,
        color_jitter: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        random_erasing_p: float = 0.0,
    ) -> None:
        self.roi_size = roi_size
        self.mean = mean
        self.std = std
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.random_erasing_p = random_erasing_p
        super().__init__()

    def _build_compose(self) -> transforms.Compose:
        return _build_relation_train_pipeline(
            roi_size=self.roi_size,
            mean=self.mean,
            std=self.std,
            color_jitter=self.color_jitter,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
            random_erasing_p=self.random_erasing_p,
        )


class ValidationTransformFacade(TransformFacade):
    def __init__(
        self,
        roi_size: int = 224,
        mean: List[float] = IMAGENET_MEAN,
        std: List[float] = IMAGENET_STD,
    ) -> None:
        self.roi_size = roi_size
        self.mean = mean
        self.std = std
        super().__init__()

    def _build_compose(self) -> transforms.Compose:
        return _build_eval_pipeline(
            roi_size=self.roi_size,
            mean=self.mean,
            std=self.std,
        )


class FeatureTransformFacade(ValidationTransformFacade):
    """Feature extraction uses the same pipeline as validation/test."""



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
) -> TrainTransformFacade:
    return TrainTransformFacade(
        roi_size=roi_size,
        mean=mean,
        std=std,
        resize_delta=resize_delta,
        horizontal_flip_p=horizontal_flip_p,
        color_jitter=color_jitter,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        random_erasing_p=random_erasing_p,
    )


def get_object_train_transforms(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    horizontal_flip_p: float = 0.5,
    color_jitter: bool = True,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    random_erasing_p: float = 0.0,
) -> ObjectTrainTransformFacade:
    return ObjectTrainTransformFacade(
        roi_size=roi_size,
        mean=mean,
        std=std,
        horizontal_flip_p=horizontal_flip_p,
        color_jitter=color_jitter,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        random_erasing_p=random_erasing_p,
    )


def get_relation_train_transforms(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
    color_jitter: bool = False,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
    random_erasing_p: float = 0.0,
) -> RelationTrainTransformFacade:
    return RelationTrainTransformFacade(
        roi_size=roi_size,
        mean=mean,
        std=std,
        color_jitter=color_jitter,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        random_erasing_p=random_erasing_p,
    )


def get_val_transforms(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> ValidationTransformFacade:
    return ValidationTransformFacade(
        roi_size=roi_size,
        mean=mean,
        std=std,
    )


def get_feature_transforms(
    roi_size: int = 224,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> FeatureTransformFacade:
    return FeatureTransformFacade(
        roi_size=roi_size,
        mean=mean,
        std=std,
    )


__all__ = [
    "TransformFacade",
    "TrainTransformFacade",
    "ObjectTrainTransformFacade",
    "RelationTrainTransformFacade",
    "ValidationTransformFacade",
    "FeatureTransformFacade",
    "get_train_transforms",
    "get_object_train_transforms",
    "get_relation_train_transforms",
    "get_val_transforms",
    "get_feature_transforms",
    "denormalize",
]


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
