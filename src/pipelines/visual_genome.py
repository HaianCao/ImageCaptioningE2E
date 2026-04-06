"""Notebook-friendly runners for the Visual Genome pipelines."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from ..data.attribute_dataset import build_attribute_datasets
from ..data.download import download_and_extract_metadata, download_vg_images, verify_download
from ..data.object_dataset import build_object_datasets
from ..data.preprocessing import build_object_attribute_vocab_and_splits, build_relation_vocab_and_splits
from ..data.relation_dataset import build_relation_datasets
from ..evaluation import compute_classification_metrics, compute_multilabel_metrics
from ..features.attribute_feature_extractor import extract_attribute_features
from ..features.object_feature_extractor import extract_object_features
from ..features.relation_feature_extractor import extract_relation_features
from ..models.attribute import AttributeClassifier
from ..models.object import ObjectClassifier
from ..models.relation import RelationClassifier
from ..training.attribute_trainer import AttributeTrainer
from ..training.object_trainer import ObjectTrainer
from ..training.relation_trainer import RelationTrainer
from ..utils.config_loader import load_task_configs, print_config
from ..utils.memory import cleanup_cuda_memory
from .common import (
    cache_output,
    collect_split_image_ids,
    configure_notebook_environment,
    feature_cache_ready,
    get_device,
    missing_local_image_ids,
    normalize_input_mode,
    require_files,
    resolve_project_root,
    seed_everything,
)


@dataclass(frozen=True)
class PipelineRuntime:
    """Resolved runtime state shared by the notebook wrappers."""

    project_root: Path
    base_config: Any
    task_config: Any
    raw_dir: Path
    image_dir: Path
    processed_dir: Path
    feature_dir: Path
    checkpoint_dir: Path
    device: str
    download_data: bool
    strict_sample_mode: bool
    sample_size: int
    sample_seed: int
    image_download_mode: str
    pre_extract_features: bool
    feature_batch_size: int
    feature_resize_size: int
    feature_crop_size: int
    feature_mean: List[float]
    feature_std: List[float]
    split_ratios: Tuple[float, float, float]
    max_samples: Optional[int]
    roi_size: int


@dataclass(frozen=True)
class PipelineResult:
    """Compact summary returned by each pipeline runner."""

    name: str
    best_checkpoint: Optional[str]
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    processed_dir: str
    feature_dir: str


def _build_runtime(
    base_config_path: str,
    task_config_path: str,
    project_root: Optional[Path] = None,
) -> PipelineRuntime:
    project_root = resolve_project_root(project_root or Path.cwd())
    base_config = load_task_configs(base_config_path)
    task_config = load_task_configs(base_config_path, task_config_path)
    device = get_device(str(base_config.device))

    seed_everything(int(base_config.seed))

    raw_dir = project_root / Path(base_config.paths.raw_dir)
    image_dir = project_root / Path(base_config.dataset.image_dir)
    processed_dir = project_root / Path(task_config.dataset.processed_dir)
    feature_dir = processed_dir / Path(task_config.dataset.feature_cache_dir)
    checkpoint_dir = project_root / Path(base_config.paths.checkpoint_dir)

    strict_sample_mode = bool(base_config.sampling.strict_mode)
    image_download_mode = "none" if strict_sample_mode else str(base_config.sampling.image_download_mode)

    return PipelineRuntime(
        project_root=project_root,
        base_config=base_config,
        task_config=task_config,
        raw_dir=raw_dir,
        image_dir=image_dir,
        processed_dir=processed_dir,
        feature_dir=feature_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
        download_data=bool(base_config.pipeline.download_data),
        strict_sample_mode=strict_sample_mode,
        sample_size=int(base_config.sampling.sample_size),
        sample_seed=int(base_config.sampling.seed),
        image_download_mode=image_download_mode,
        pre_extract_features=bool(base_config.pipeline.pre_extract_features),
        feature_batch_size=int(base_config.feature_extraction.batch_size),
        feature_resize_size=int(base_config.feature_extraction.resize_size),
        feature_crop_size=int(base_config.feature_extraction.crop_size),
        feature_mean=[float(x) for x in base_config.image.mean],
        feature_std=[float(x) for x in base_config.image.std],
        split_ratios=(
            float(base_config.split.train),
            float(base_config.split.val),
            float(base_config.split.test),
        ),
        max_samples=None if base_config.sampling.max_samples is None else int(base_config.sampling.max_samples),
        roi_size=int(base_config.image.roi_size),
    )


def _print_runtime_summary(runtime: PipelineRuntime, task_name: str, show_full_config: bool) -> None:
    print(f"========== PIPELINE: {task_name.upper()} ==========")
    print(f"Project root: {runtime.project_root}")
    print(f"Using device: {runtime.device}")
    print(f"Processed root: {runtime.processed_dir}")
    print(f"Feature cache root: {runtime.feature_dir}")
    print(f"Strict sample mode: {runtime.strict_sample_mode}")
    print(f"Sample size: {runtime.sample_size} | split ratios: {runtime.split_ratios} | seed: {runtime.sample_seed}")
    print(f"Download data: {runtime.download_data} | image mode: {runtime.image_download_mode} | pre-extract features: {runtime.pre_extract_features}")
    if show_full_config:
        print_config(runtime.base_config)
        print_config(runtime.task_config)


def _load_image_data(raw_dir: Path) -> List[Dict[str, Any]]:
    image_data_file = raw_dir / "image_data.json"
    require_files([image_data_file], "image_data.json")

    with open(image_data_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return raw.get("images", raw.get("samples", []))
    raise TypeError(f"Unsupported image_data.json format: {type(raw)!r}")


def _prepare_raw_data(runtime: PipelineRuntime, task_name: str) -> List[Dict[str, Any]]:
    if runtime.download_data:
        print(f"Downloading metadata for the {task_name} pipeline...")
        download_and_extract_metadata(raw_dir=str(runtime.raw_dir), keep_zip=False)

    raw_status = verify_download(raw_dir=str(runtime.raw_dir))
    missing_metadata = [name for name, ok in raw_status.items() if not ok]
    if missing_metadata:
        raise RuntimeError(
            "Thiếu dữ liệu RAW cần thiết: "
            + ", ".join(missing_metadata)
            + ". Hãy bật DOWNLOAD_DATA = True hoặc đặt đúng thư mục data/raw."
        )

    image_data = _load_image_data(runtime.raw_dir)

    if runtime.download_data:
        if runtime.image_download_mode == "sample":
            sample_ids = [img["image_id"] for img in image_data[: runtime.sample_size]]
            print(f"[{task_name}] Đang tải bộ sample {len(sample_ids)} ảnh...")
            downloaded_images = download_vg_images(sample_ids, image_dir=str(runtime.image_dir))
            if not downloaded_images:
                raise RuntimeError(f"Không tải được ảnh sample nào cho {task_name}.")
        elif runtime.image_download_mode == "all":
            all_ids = [img["image_id"] for img in image_data]
            print(f"[{task_name}] Đang tải toàn bộ {len(all_ids)} ảnh...")
            downloaded_images = download_vg_images(all_ids, image_dir=str(runtime.image_dir))
            if not downloaded_images:
                raise RuntimeError(f"Không tải được ảnh nào cho {task_name}.")
        elif runtime.image_download_mode == "none":
            print(f"[{task_name}] Bỏ qua tải ảnh theo cấu hình.")
        else:
            raise ValueError("IMAGE_DOWNLOAD_MODE phải là 'none', 'sample', hoặc 'all'.")

    return image_data


def _sample_image_ids(image_data: List[Dict[str, Any]], runtime: PipelineRuntime) -> Optional[List[int]]:
    if not runtime.strict_sample_mode:
        print("Bỏ qua strict sample; sẽ dùng toàn bộ dữ liệu theo split mặc định.")
        return None

    all_image_ids = [img["image_id"] for img in image_data]
    sample_count = min(runtime.sample_size, len(all_image_ids))
    sample_image_ids = random.Random(runtime.sample_seed).sample(all_image_ids, sample_count)
    print(f"Đã chọn trước {len(sample_image_ids)} image_id cho sample strict.")
    return sample_image_ids


def _download_missing_images_for_splits(runtime: PipelineRuntime, task_name: str, split_names: Sequence[str]) -> None:
    image_ids = collect_split_image_ids(runtime.processed_dir, split_names)
    missing_ids = missing_local_image_ids(runtime.image_dir, image_ids)

    print(f"[{task_name}] Tổng ảnh tham chiếu: {len(image_ids)} | thiếu local: {len(missing_ids)}")
    if missing_ids:
        print(f"[{task_name}] Đang tải bổ sung ảnh còn thiếu...")
        downloaded = download_vg_images(missing_ids, image_dir=str(runtime.image_dir))
        if len(downloaded) < len(missing_ids):
            print(f"[Warning] {task_name}: còn {len(missing_ids) - len(downloaded)} ảnh chưa tải được.")


def _move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _dataloader_kwargs(runtime: PipelineRuntime) -> Dict[str, Any]:
    return {
        "num_workers": int(runtime.task_config.training.num_workers),
        "pin_memory": bool(runtime.task_config.training.pin_memory and runtime.device == "cuda"),
    }


def _evaluate_object_model(model: ObjectClassifier, loader: DataLoader, device: str, num_classes: int) -> Dict[str, float]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            features = batch.get("feature")
            if features is None:
                features = batch.get("object_feature")
            if features is None:
                features = batch.get("image")
            if features is None:
                features = batch.get("object_image")
            if features is None:
                raise KeyError("Object evaluation requires a feature or image tensor in the batch")

            logits = model(features)
            logits_list.append(logits.cpu())
            targets_list.append(batch["object_label"].cpu())

    if not logits_list:
        return {"f1": 0.0, "accuracy": 0.0}

    return compute_classification_metrics(
        torch.cat(logits_list),
        torch.cat(targets_list),
        num_classes=num_classes,
    )


def _evaluate_attribute_model(
    model: AttributeClassifier,
    loader: DataLoader,
    device: str,
    task_config: Any,
) -> Dict[str, float]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            features = batch.get("feature")
            if features is None:
                features = batch.get("attribute_feature")
            if features is None:
                features = batch.get("image")
            if features is None:
                features = batch.get("attribute_image")
            if features is None:
                raise KeyError("Attribute evaluation requires a feature or image tensor in the batch")

            logits = model(features)
            logits_list.append(logits.cpu())
            targets_list.append(batch["attribute_labels"].cpu())

    if not logits_list:
        return {"micro_f1": 0.0}

    return compute_multilabel_metrics(
        torch.cat(logits_list),
        torch.cat(targets_list),
        threshold=float(task_config.eval.attribute_threshold),
        threshold_mode=str(task_config.eval.attribute_threshold_mode),
        threshold_scale=float(task_config.eval.attribute_threshold_scale),
        threshold_min=float(task_config.eval.attribute_threshold_min),
        threshold_max=float(task_config.eval.attribute_threshold_max),
    )


def _evaluate_relation_model(model: RelationClassifier, loader: DataLoader, device: str, num_classes: int) -> Dict[str, float]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            features = batch.get("feature")
            if features is None:
                features = batch.get("image")
            if features is None:
                features = batch.get("union_image")
            if features is None:
                raise KeyError("Relation evaluation requires a feature or image tensor in the batch")

            logits = model(features, batch["spatial"])
            logits_list.append(logits.cpu())
            targets_list.append(batch["relation_label"].cpu())

    if not logits_list:
        return {"f1": 0.0, "accuracy": 0.0}

    return compute_classification_metrics(
        torch.cat(logits_list),
        torch.cat(targets_list),
        num_classes=num_classes,
    )


def _prepare_object_attribute_split_files(runtime: PipelineRuntime, image_data: List[Dict[str, Any]]) -> Optional[List[int]]:
    sample_image_ids = _sample_image_ids(image_data, runtime)

    build_object_attribute_vocab_and_splits(
        raw_dir=str(runtime.raw_dir),
        processed_dir=str(runtime.processed_dir),
        max_objects=int(runtime.task_config.labels.max_objects),
        max_attributes=int(runtime.task_config.labels.max_attributes),
        sample_image_ids=sample_image_ids,
        split_by_image_id=runtime.strict_sample_mode,
        split_ratios=runtime.split_ratios,
        seed=runtime.sample_seed,
    )

    require_files(
        [
            runtime.processed_dir / "object_vocab.json",
            runtime.processed_dir / "attribute_vocab.json",
            runtime.processed_dir / "train" / "annotations.json",
            runtime.processed_dir / "val" / "annotations.json",
            runtime.processed_dir / "test" / "annotations.json",
        ],
        "Object/attribute processed files",
    )

    return sample_image_ids


def _prepare_relation_split_files(runtime: PipelineRuntime, image_data: List[Dict[str, Any]]) -> Optional[List[int]]:
    sample_image_ids = _sample_image_ids(image_data, runtime)

    build_relation_vocab_and_splits(
        raw_dir=str(runtime.raw_dir),
        processed_dir=str(runtime.processed_dir),
        max_relations=int(runtime.task_config.labels.max_relations),
        sample_image_ids=sample_image_ids,
        split_by_image_id=runtime.strict_sample_mode,
        split_ratios=runtime.split_ratios,
        seed=runtime.sample_seed,
    )

    require_files(
        [
            runtime.processed_dir / "relation_vocab.json",
            runtime.processed_dir / "train" / "annotations.json",
            runtime.processed_dir / "val" / "annotations.json",
            runtime.processed_dir / "test" / "annotations.json",
        ],
        "Relation processed files",
    )

    return sample_image_ids


def _ensure_task1_feature_cache(runtime: PipelineRuntime, input_mode: str) -> bool:
    split_names = ["train", "val", "test"]
    cache_ready = feature_cache_ready(runtime.feature_dir, split_names, input_mode)

    if runtime.pre_extract_features and bool(runtime.task_config.dataset.use_feature_cache):
        runtime.feature_dir.mkdir(parents=True, exist_ok=True)
        _download_missing_images_for_splits(runtime, "Task 1", split_names)
        print("Extracting task 1 features...")
        for split_name in split_names:
            output_path = cache_output(runtime.feature_dir, split_name, input_mode)
            extract_object_features(
                annotation_file=str(runtime.processed_dir / split_name / "annotations.json"),
                image_dir=str(runtime.image_dir),
                output_file=str(output_path),
                backbone=str(runtime.task_config.backbone.name),
                pretrained=bool(runtime.task_config.backbone.pretrained),
                batch_size=runtime.feature_batch_size,
                device=runtime.device,
                resize_size=runtime.feature_resize_size,
                crop_size=runtime.feature_crop_size,
                mean=runtime.feature_mean,
                std=runtime.feature_std,
                input_mode=input_mode,
            )
        cache_ready = feature_cache_ready(runtime.feature_dir, split_names, input_mode)
        if not cache_ready:
            raise RuntimeError("Task 1 feature cache is empty after extraction.")
        return True

    if not cache_ready:
        _download_missing_images_for_splits(runtime, "Task 1", split_names)
    return cache_ready


def _ensure_attribute_feature_cache(runtime: PipelineRuntime, input_mode: str) -> bool:
    split_names = ["train", "val", "test"]
    cache_ready = feature_cache_ready(runtime.feature_dir, split_names, input_mode)

    if runtime.pre_extract_features and bool(runtime.task_config.dataset.use_feature_cache):
        runtime.feature_dir.mkdir(parents=True, exist_ok=True)
        _download_missing_images_for_splits(runtime, "Task 1", split_names)
        print("Extracting attribute features...")
        for split_name in split_names:
            output_path = cache_output(runtime.feature_dir, split_name, input_mode)
            extract_attribute_features(
                annotation_file=str(runtime.processed_dir / split_name / "annotations.json"),
                image_dir=str(runtime.image_dir),
                output_file=str(output_path),
                backbone=str(runtime.task_config.backbone.name),
                pretrained=bool(runtime.task_config.backbone.pretrained),
                batch_size=runtime.feature_batch_size,
                device=runtime.device,
                resize_size=runtime.feature_resize_size,
                crop_size=runtime.feature_crop_size,
                mean=runtime.feature_mean,
                std=runtime.feature_std,
                input_mode=input_mode,
            )
        cache_ready = feature_cache_ready(runtime.feature_dir, split_names, input_mode)
        if not cache_ready:
            raise RuntimeError("Task 1 feature cache is empty after extraction.")
        return True

    if not cache_ready:
        _download_missing_images_for_splits(runtime, "Task 1", split_names)
    return cache_ready


def _ensure_relation_feature_cache(runtime: PipelineRuntime, input_mode: str) -> bool:
    split_names = ["train", "val", "test"]
    cache_ready = feature_cache_ready(runtime.feature_dir, split_names, input_mode)

    if runtime.pre_extract_features and bool(runtime.task_config.dataset.use_feature_cache):
        runtime.feature_dir.mkdir(parents=True, exist_ok=True)
        _download_missing_images_for_splits(runtime, "Task 2", split_names)
        print("Extracting relation features...")
        for split_name in split_names:
            output_path = cache_output(runtime.feature_dir, split_name, input_mode)
            extract_relation_features(
                annotation_file=str(runtime.processed_dir / split_name / "annotations.json"),
                image_dir=str(runtime.image_dir),
                output_file=str(output_path),
                backbone=str(runtime.task_config.backbone.name),
                pretrained=bool(runtime.task_config.backbone.pretrained),
                batch_size=runtime.feature_batch_size,
                device=runtime.device,
                resize_size=runtime.feature_resize_size,
                crop_size=runtime.feature_crop_size,
                mean=runtime.feature_mean,
                std=runtime.feature_std,
                input_mode=input_mode,
            )
        cache_ready = feature_cache_ready(runtime.feature_dir, split_names, input_mode)
        if not cache_ready:
            raise RuntimeError("Task 2 feature cache is empty after extraction.")
        return True

    if not cache_ready:
        _download_missing_images_for_splits(runtime, "Task 2", split_names)
    return cache_ready


def run_object_pipeline(
    base_config_path: str = "configs/config.yaml",
    task_config_path: str = "configs/object_config.yaml",
    project_root: Optional[Path] = None,
    *,
    show_full_config: bool = False,
) -> PipelineResult:
    """Run the object classification pipeline end to end."""
    runtime = _build_runtime(base_config_path, task_config_path, project_root)
    _print_runtime_summary(runtime, "object", show_full_config)

    image_data = _prepare_raw_data(runtime, "Task 1")
    _prepare_object_attribute_split_files(runtime, image_data)

    object_input_mode = normalize_input_mode(str(runtime.task_config.dataset.object_input_mode))
    object_learnable_backbone = bool(runtime.task_config.backbone.learnable_backbone)
    object_use_cache = bool(runtime.task_config.dataset.use_feature_cache) and not object_learnable_backbone

    cache_ready = _ensure_task1_feature_cache(runtime, object_input_mode) if object_use_cache else False
    if not object_use_cache or not cache_ready:
        _download_missing_images_for_splits(runtime, "Task 1", ["train", "val", "test"])

    object_train_ds, object_val_ds, object_test_ds = build_object_datasets(
        processed_dir=str(runtime.processed_dir),
        image_dir=str(runtime.image_dir),
        roi_size=runtime.roi_size,
        use_feature_cache=object_use_cache,
        feature_cache_dir=str(runtime.feature_dir),
        input_mode=object_input_mode,
        max_samples=runtime.max_samples,
        train_horizontal_flip_p=float(runtime.task_config.augmentation.random_horizontal_flip),
        train_color_jitter=bool(runtime.task_config.augmentation.color_jitter.enabled),
        train_brightness=float(runtime.task_config.augmentation.color_jitter.brightness),
        train_contrast=float(runtime.task_config.augmentation.color_jitter.contrast),
        train_saturation=float(runtime.task_config.augmentation.color_jitter.saturation),
        train_hue=float(runtime.task_config.augmentation.color_jitter.hue),
        train_random_erasing_p=float(runtime.task_config.augmentation.random_erasing_p),
        train_resize_delta=int(runtime.task_config.augmentation.resize_delta),
        mean=runtime.feature_mean,
        std=runtime.feature_std,
    )

    object_model = ObjectClassifier(
        num_classes=object_train_ds.num_objects,
        feature_dim=int(runtime.task_config.backbone.feature_dim),
        hidden_dim=int(runtime.task_config.model.object_hidden_dim),
        dropout=float(runtime.task_config.model.object_dropout),
        num_layers=int(runtime.task_config.model.object_num_layers),
        backbone_name=str(runtime.task_config.backbone.name) if object_learnable_backbone else None,
        pretrained=bool(runtime.task_config.backbone.pretrained),
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        learnable_backbone=object_learnable_backbone,
        device=runtime.device,
    )

    object_optimizer = torch.optim.AdamW(
        object_model.parameters(),
        lr=float(runtime.task_config.training.lr),
        weight_decay=float(runtime.task_config.training.weight_decay),
    )

    loader_kwargs = _dataloader_kwargs(runtime)
    object_train_loader = DataLoader(object_train_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=True, **loader_kwargs)
    object_val_loader = DataLoader(object_val_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=False, **loader_kwargs)
    object_test_loader = DataLoader(object_test_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=False, **loader_kwargs)

    object_trainer = ObjectTrainer(
        model=object_model,
        train_loader=object_train_loader,
        val_loader=object_val_loader,
        optimizer=object_optimizer,
        use_auto_class_weights=True,
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        freeze_epochs=int(runtime.task_config.backbone.freeze_epochs),
        max_epochs=int(runtime.task_config.training.max_epochs),
        early_stopping_patience=int(runtime.task_config.training.early_stopping_patience),
        gradient_clip_val=float(runtime.task_config.training.gradient_clip_val),
        log_every_n_steps=int(runtime.task_config.training.log_every_n_steps),
        device=runtime.device,
        use_amp=(runtime.device == "cuda"),
    )

    try:
        train_metrics = object_trainer.train()
        best_checkpoint = object_trainer.checkpoint_manager.get_best_checkpoint_path()
        test_metrics = _evaluate_object_model(object_trainer.object_model, object_test_loader, runtime.device, object_train_ds.num_objects)
        print("Object pipeline completed.")
        print(f"Best object checkpoint: {best_checkpoint}")
        print("Object test metrics:")
        print(test_metrics)
        return PipelineResult(
            name="object",
            best_checkpoint=str(best_checkpoint) if best_checkpoint is not None else None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            processed_dir=str(runtime.processed_dir),
            feature_dir=str(runtime.feature_dir),
        )
    finally:
        cleanup_cuda_memory(note="Object pipeline finished")


def run_attribute_pipeline(
    base_config_path: str = "configs/config.yaml",
    task_config_path: str = "configs/attribute_config.yaml",
    project_root: Optional[Path] = None,
    *,
    show_full_config: bool = False,
) -> PipelineResult:
    """Run the attribute classification pipeline end to end."""
    runtime = _build_runtime(base_config_path, task_config_path, project_root)
    _print_runtime_summary(runtime, "attribute", show_full_config)

    image_data = _prepare_raw_data(runtime, "Task 1")
    _prepare_object_attribute_split_files(runtime, image_data)

    attribute_input_mode = normalize_input_mode(str(runtime.task_config.dataset.attribute_input_mode))
    attribute_learnable_backbone = bool(runtime.task_config.backbone.learnable_backbone)
    attribute_use_cache = bool(runtime.task_config.dataset.use_feature_cache) and not attribute_learnable_backbone

    cache_ready = _ensure_attribute_feature_cache(runtime, attribute_input_mode) if attribute_use_cache else False
    if not attribute_use_cache or not cache_ready:
        _download_missing_images_for_splits(runtime, "Task 1", ["train", "val", "test"])

    attribute_train_ds, attribute_val_ds, attribute_test_ds = build_attribute_datasets(
        processed_dir=str(runtime.processed_dir),
        image_dir=str(runtime.image_dir),
        roi_size=runtime.roi_size,
        use_feature_cache=attribute_use_cache,
        feature_cache_dir=str(runtime.feature_dir),
        input_mode=attribute_input_mode,
        max_samples=runtime.max_samples,
        train_horizontal_flip_p=float(runtime.task_config.augmentation.random_horizontal_flip),
        train_color_jitter=bool(runtime.task_config.augmentation.color_jitter.enabled),
        train_brightness=float(runtime.task_config.augmentation.color_jitter.brightness),
        train_contrast=float(runtime.task_config.augmentation.color_jitter.contrast),
        train_saturation=float(runtime.task_config.augmentation.color_jitter.saturation),
        train_hue=float(runtime.task_config.augmentation.color_jitter.hue),
        train_random_erasing_p=float(runtime.task_config.augmentation.random_erasing_p),
        train_resize_delta=int(runtime.task_config.augmentation.resize_delta),
        mean=runtime.feature_mean,
        std=runtime.feature_std,
    )

    attribute_model = AttributeClassifier(
        num_attributes=attribute_train_ds.num_attributes,
        feature_dim=int(runtime.task_config.backbone.feature_dim),
        hidden_dim=int(runtime.task_config.model.attribute_hidden_dim),
        dropout=float(runtime.task_config.model.attribute_dropout),
        num_layers=int(runtime.task_config.model.attribute_num_layers),
        backbone_name=str(runtime.task_config.backbone.name) if attribute_learnable_backbone else None,
        pretrained=bool(runtime.task_config.backbone.pretrained),
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        learnable_backbone=attribute_learnable_backbone,
        device=runtime.device,
    )

    attribute_optimizer = torch.optim.AdamW(
        attribute_model.parameters(),
        lr=float(runtime.task_config.training.lr),
        weight_decay=float(runtime.task_config.training.weight_decay),
    )

    loader_kwargs = _dataloader_kwargs(runtime)
    attribute_train_loader = DataLoader(attribute_train_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=True, **loader_kwargs)
    attribute_val_loader = DataLoader(attribute_val_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=False, **loader_kwargs)
    attribute_test_loader = DataLoader(attribute_test_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=False, **loader_kwargs)

    attribute_trainer = AttributeTrainer(
        model=attribute_model,
        train_loader=attribute_train_loader,
        val_loader=attribute_val_loader,
        optimizer=attribute_optimizer,
        use_auto_class_weights=True,
        attribute_threshold_mode=str(runtime.task_config.eval.attribute_threshold_mode),
        attribute_threshold=float(runtime.task_config.eval.attribute_threshold),
        attribute_threshold_scale=float(runtime.task_config.eval.attribute_threshold_scale),
        attribute_threshold_min=float(runtime.task_config.eval.attribute_threshold_min),
        attribute_threshold_max=float(runtime.task_config.eval.attribute_threshold_max),
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        freeze_epochs=int(runtime.task_config.backbone.freeze_epochs),
        max_epochs=int(runtime.task_config.training.max_epochs),
        early_stopping_patience=int(runtime.task_config.training.early_stopping_patience),
        gradient_clip_val=float(runtime.task_config.training.gradient_clip_val),
        log_every_n_steps=int(runtime.task_config.training.log_every_n_steps),
        device=runtime.device,
        use_amp=(runtime.device == "cuda"),
    )

    try:
        train_metrics = attribute_trainer.train()
        best_checkpoint = attribute_trainer.checkpoint_manager.get_best_checkpoint_path()
        test_metrics = _evaluate_attribute_model(attribute_trainer.attribute_model, attribute_test_loader, runtime.device, runtime.task_config)
        print("Attribute pipeline completed.")
        print(f"Best attribute checkpoint: {best_checkpoint}")
        print("Attribute test metrics:")
        print(test_metrics)
        return PipelineResult(
            name="attribute",
            best_checkpoint=str(best_checkpoint) if best_checkpoint is not None else None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            processed_dir=str(runtime.processed_dir),
            feature_dir=str(runtime.feature_dir),
        )
    finally:
        cleanup_cuda_memory(note="Attribute pipeline finished")


def run_relation_pipeline(
    base_config_path: str = "configs/config.yaml",
    task_config_path: str = "configs/relation_config.yaml",
    project_root: Optional[Path] = None,
    *,
    show_full_config: bool = False,
) -> PipelineResult:
    """Run the relation classification pipeline end to end."""
    runtime = _build_runtime(base_config_path, task_config_path, project_root)
    _print_runtime_summary(runtime, "relation", show_full_config)

    image_data = _prepare_raw_data(runtime, "Task 2")
    _prepare_relation_split_files(runtime, image_data)

    relation_input_mode = normalize_input_mode(str(runtime.task_config.dataset.input_mode))
    relation_learnable_backbone = bool(runtime.task_config.backbone.learnable_backbone)
    relation_use_cache = bool(runtime.task_config.dataset.use_feature_cache) and not relation_learnable_backbone

    cache_ready = _ensure_relation_feature_cache(runtime, relation_input_mode) if relation_use_cache else False
    if not relation_use_cache or not cache_ready:
        _download_missing_images_for_splits(runtime, "Task 2", ["train", "val", "test"])

    relation_train_ds, relation_val_ds, relation_test_ds = build_relation_datasets(
        processed_dir=str(runtime.processed_dir),
        image_dir=str(runtime.image_dir),
        roi_size=runtime.roi_size,
        use_feature_cache=relation_use_cache,
        use_spatial_features=bool(runtime.task_config.spatial.use_spatial_features),
        feature_cache_dir=str(runtime.feature_dir),
        input_mode=relation_input_mode,
        max_samples=runtime.max_samples,
        train_horizontal_flip_p=float(runtime.task_config.augmentation.random_horizontal_flip),
        train_color_jitter=bool(runtime.task_config.augmentation.color_jitter.enabled),
        train_brightness=float(runtime.task_config.augmentation.color_jitter.brightness),
        train_contrast=float(runtime.task_config.augmentation.color_jitter.contrast),
        train_saturation=float(runtime.task_config.augmentation.color_jitter.saturation),
        train_hue=float(runtime.task_config.augmentation.color_jitter.hue),
        train_random_erasing_p=float(runtime.task_config.augmentation.random_erasing_p),
        train_resize_delta=int(runtime.task_config.augmentation.resize_delta),
        mean=runtime.feature_mean,
        std=runtime.feature_std,
    )

    relation_model = RelationClassifier(
        num_relations=relation_train_ds.num_relations,
        feature_dim=int(runtime.task_config.backbone.feature_dim),
        spatial_dim=int(runtime.task_config.spatial.spatial_dim),
        hidden_dim=int(runtime.task_config.model.hidden_dim),
        dropout=float(runtime.task_config.model.dropout),
        num_layers=int(runtime.task_config.model.num_layers),
        attention_heads=int(runtime.task_config.model.attention_heads),
        fusion_method=str(runtime.task_config.model.fusion_method),
        backbone_name=str(runtime.task_config.backbone.name) if relation_learnable_backbone else None,
        pretrained=bool(runtime.task_config.backbone.pretrained),
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        learnable_backbone=relation_learnable_backbone,
        device=runtime.device,
    )

    relation_optimizer = torch.optim.AdamW(
        relation_model.parameters(),
        lr=float(runtime.task_config.training.lr),
        weight_decay=float(runtime.task_config.training.weight_decay),
    )

    loader_kwargs = _dataloader_kwargs(runtime)
    relation_train_loader = DataLoader(relation_train_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=True, **loader_kwargs)
    relation_val_loader = DataLoader(relation_val_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=False, **loader_kwargs)
    relation_test_loader = DataLoader(relation_test_ds, batch_size=int(runtime.task_config.training.batch_size), shuffle=False, **loader_kwargs)

    relation_trainer = RelationTrainer(
        model=relation_model,
        train_loader=relation_train_loader,
        val_loader=relation_val_loader,
        optimizer=relation_optimizer,
        label_smoothing=float(runtime.task_config.loss.label_smoothing),
        use_auto_class_weights=True,
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        freeze_epochs=int(runtime.task_config.backbone.freeze_epochs),
        max_epochs=int(runtime.task_config.training.max_epochs),
        early_stopping_patience=int(runtime.task_config.training.early_stopping_patience),
        gradient_clip_val=float(runtime.task_config.training.gradient_clip_val),
        log_every_n_steps=int(runtime.task_config.training.log_every_n_steps),
        use_amp=(runtime.device == "cuda"),
    )

    try:
        train_metrics = relation_trainer.train()
        best_checkpoint = relation_trainer.checkpoint_manager.get_best_checkpoint_path()
        test_metrics = _evaluate_relation_model(relation_trainer.model, relation_test_loader, runtime.device, relation_train_ds.num_relations)
        print("Relation pipeline completed.")
        print(f"Best relation checkpoint: {best_checkpoint}")
        print("Relation test metrics:")
        print(test_metrics)
        return PipelineResult(
            name="relation",
            best_checkpoint=str(best_checkpoint) if best_checkpoint is not None else None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            processed_dir=str(runtime.processed_dir),
            feature_dir=str(runtime.feature_dir),
        )
    finally:
        cleanup_cuda_memory(note="Relation pipeline finished")


def run_all_pipelines(
    base_config_path: str = "configs/config.yaml",
    project_root: Optional[Path] = None,
    *,
    show_full_config: bool = False,
) -> Dict[str, PipelineResult]:
    """Run the three task pipelines sequentially."""
    project_root = resolve_project_root(project_root or Path.cwd())
    results = {
        "object": run_object_pipeline(base_config_path=base_config_path, project_root=project_root, show_full_config=show_full_config),
        "attribute": run_attribute_pipeline(base_config_path=base_config_path, project_root=project_root, show_full_config=show_full_config),
        "relation": run_relation_pipeline(base_config_path=base_config_path, project_root=project_root, show_full_config=show_full_config),
    }
    return results
