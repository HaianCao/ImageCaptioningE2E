"""Public preprocessing strategy registry.

Each strategy returns a consistent pair of train/validation image transform
facades. The pipeline decides the task-specific preset, while the strategy
encapsulates the exact preprocessing recipe.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ObjectTrainTransformFacade,
    RelationTrainTransformFacade,
    TransformFacade,
    ValidationTransformFacade,
    get_object_train_transforms,
    get_relation_train_transforms,
    get_val_transforms,
)


PREPROCESSING_STRATEGY_REGISTRY: Dict[str, Type["PreprocessingStrategy"]] = {}


def register_preprocessing_strategy(*names: str):
    """Register one preprocessing strategy class under one or more config keys."""

    def decorator(strategy_cls: Type["PreprocessingStrategy"]):
        for name in names:
            PREPROCESSING_STRATEGY_REGISTRY[name] = strategy_cls
        return strategy_cls

    return decorator


@dataclass(frozen=True)
class PreprocessingPreset:
    """Resolved train/validation transform pair returned by a strategy."""

    train: TransformFacade
    val: ValidationTransformFacade


class PreprocessingStrategy(ABC):
    """Base class for image preprocessing presets."""

    def build(
        self,
        *,
        roi_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        train_horizontal_flip_p: float = 0.5,
        train_color_jitter: bool = True,
        train_brightness: float = 0.2,
        train_contrast: float = 0.2,
        train_saturation: float = 0.2,
        train_hue: float = 0.1,
        train_random_erasing_p: float = 0.0,
        train_resize_delta: int = 32,
    ) -> PreprocessingPreset:
        return PreprocessingPreset(
            train=self.build_train(
                roi_size=roi_size,
                mean=mean or IMAGENET_MEAN,
                std=std or IMAGENET_STD,
                train_horizontal_flip_p=train_horizontal_flip_p,
                train_color_jitter=train_color_jitter,
                train_brightness=train_brightness,
                train_contrast=train_contrast,
                train_saturation=train_saturation,
                train_hue=train_hue,
                train_random_erasing_p=train_random_erasing_p,
                train_resize_delta=train_resize_delta,
            ),
            val=self.build_val(
                roi_size=roi_size,
                mean=mean or IMAGENET_MEAN,
                std=std or IMAGENET_STD,
            ),
        )

    def build_val(
        self,
        *,
        roi_size: int,
        mean: List[float],
        std: List[float],
    ) -> ValidationTransformFacade:
        """Build the validation/test pipeline shared by all strategies."""
        return get_val_transforms(roi_size=roi_size, mean=mean, std=std)

    def _build_task1_train(
        self,
        *,
        roi_size: int,
        mean: List[float],
        std: List[float],
        train_horizontal_flip_p: float,
        train_color_jitter: bool,
        train_brightness: float,
        train_contrast: float,
        train_saturation: float,
        train_hue: float,
        train_random_erasing_p: float,
    ) -> ObjectTrainTransformFacade:
        return get_object_train_transforms(
            roi_size=roi_size,
            mean=mean,
            std=std,
            horizontal_flip_p=train_horizontal_flip_p,
            color_jitter=train_color_jitter,
            brightness=train_brightness,
            contrast=train_contrast,
            saturation=train_saturation,
            hue=train_hue,
            random_erasing_p=train_random_erasing_p,
        )

    def _build_task2_train(
        self,
        *,
        roi_size: int,
        mean: List[float],
        std: List[float],
        train_horizontal_flip_p: float,
        train_color_jitter: bool,
        train_brightness: float,
        train_contrast: float,
        train_saturation: float,
        train_hue: float,
        train_random_erasing_p: float,
    ) -> RelationTrainTransformFacade:
        return get_relation_train_transforms(
            roi_size=roi_size,
            mean=mean,
            std=std,
            color_jitter=train_color_jitter,
            brightness=train_brightness,
            contrast=train_contrast,
            saturation=train_saturation,
            hue=train_hue,
            random_erasing_p=train_random_erasing_p,
        )

    def build_train(
        self,
        *,
        roi_size: int,
        mean: List[float],
        std: List[float],
        train_horizontal_flip_p: float,
        train_color_jitter: bool,
        train_brightness: float,
        train_contrast: float,
        train_saturation: float,
        train_hue: float,
        train_random_erasing_p: float,
        train_resize_delta: int,
    ) -> TransformFacade:
        raise NotImplementedError


@register_preprocessing_strategy("baseline_task1", "baseline_object", "baseline_attribute")
class Task1PreprocessingStrategy(PreprocessingStrategy):
    """Default preprocessing preset for object and attribute tasks."""

    def build_train(
        self,
        *,
        roi_size: int,
        mean: List[float],
        std: List[float],
        train_horizontal_flip_p: float,
        train_color_jitter: bool,
        train_brightness: float,
        train_contrast: float,
        train_saturation: float,
        train_hue: float,
        train_random_erasing_p: float,
        train_resize_delta: int,
    ) -> ObjectTrainTransformFacade:
        _ = train_resize_delta
        return self._build_task1_train(
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
        )


@register_preprocessing_strategy("baseline_task2", "baseline_relation")
class Task2PreprocessingStrategy(PreprocessingStrategy):
    """Default preprocessing preset for relation tasks."""

    def build_train(
        self,
        *,
        roi_size: int,
        mean: List[float],
        std: List[float],
        train_horizontal_flip_p: float,
        train_color_jitter: bool,
        train_brightness: float,
        train_contrast: float,
        train_saturation: float,
        train_hue: float,
        train_random_erasing_p: float,
        train_resize_delta: int,
    ) -> RelationTrainTransformFacade:
        _ = train_horizontal_flip_p
        _ = train_resize_delta
        return self._build_task2_train(
            roi_size=roi_size,
            mean=mean,
            std=std,
            train_horizontal_flip_p=0.0,
            train_color_jitter=train_color_jitter,
            train_brightness=train_brightness,
            train_contrast=train_contrast,
            train_saturation=train_saturation,
            train_hue=train_hue,
            train_random_erasing_p=train_random_erasing_p,
        )


def get_preprocessing_strategy(strategy_name: str) -> PreprocessingStrategy:
    """Resolve a preprocessing strategy name to a strategy instance."""
    strategy_cls = PREPROCESSING_STRATEGY_REGISTRY.get(strategy_name)
    if strategy_cls is None:
        available = ", ".join(sorted(PREPROCESSING_STRATEGY_REGISTRY))
        raise ValueError(f"Unknown preprocessing strategy '{strategy_name}'. Available: {available}")
    return strategy_cls()


def build_preprocessing_preset(
    strategy_name: str,
    *,
    roi_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    train_horizontal_flip_p: float = 0.5,
    train_color_jitter: bool = True,
    train_brightness: float = 0.2,
    train_contrast: float = 0.2,
    train_saturation: float = 0.2,
    train_hue: float = 0.1,
    train_random_erasing_p: float = 0.0,
    train_resize_delta: int = 32,
) -> PreprocessingPreset:
    """Convenience helper that builds a registered preprocessing preset."""
    strategy = get_preprocessing_strategy(strategy_name)
    return strategy.build(
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


__all__ = [
    "PreprocessingPreset",
    "PreprocessingStrategy",
    "Task1PreprocessingStrategy",
    "Task2PreprocessingStrategy",
    "PREPROCESSING_STRATEGY_REGISTRY",
    "register_preprocessing_strategy",
    "get_preprocessing_strategy",
    "build_preprocessing_preset",
]