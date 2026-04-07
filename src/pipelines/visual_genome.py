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
from ..models.attribute import AttributeClassifier, build_attribute_classifier
from ..models.object import ObjectClassifier
from ..models.task1.object_strategies import build_object_classifier
from ..models.relation import RelationClassifier, build_relation_classifier
from ..training.attribute_trainer import AttributeTrainer
from ..training.object_trainer import ObjectTrainer
from ..training.relation_trainer import RelationTrainer
from ..utils.config_loader import load_task_configs, print_config
from ..utils.memory import cleanup_cuda_memory
from .common import (
    collect_split_image_ids,
    get_device,
    missing_local_image_ids,
    normalize_input_mode,
    require_files,
    resolve_project_root,
    seed_everything,
)


def _get_preprocessing_config(task_config: Any) -> Any:
    preprocessing_config = getattr(task_config, "preprocessing", None)
    if preprocessing_config is None:
        raise AttributeError("task_config.preprocessing is required")
    return preprocessing_config


@dataclass(frozen=True)
class PipelineRuntime:
    """Resolved runtime state shared by the notebook wrappers."""

    project_root: Path
    base_config: Any
    task_config: Any
    raw_dir: Path
    image_dir: Path
    processed_dir: Path
    checkpoint_dir: Path
    device: str
    download_data: bool
    strict_sample_mode: bool
    sample_size: int
    sample_seed: int
    sample_strategy: str
    sample_focus: str
    image_download_mode: str
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
    checkpoint_dir = project_root / Path(base_config.paths.checkpoint_dir)

    base_sampling = getattr(base_config, "sampling", None)
    task_sampling = getattr(task_config, "sampling", None)

    def _sampling_value(name: str, default: Any) -> Any:
        if task_sampling is not None and hasattr(task_sampling, name):
            return getattr(task_sampling, name)
        if base_sampling is not None and hasattr(base_sampling, name):
            return getattr(base_sampling, name)
        return default

    strict_sample_mode = bool(_sampling_value("strict_mode", base_config.sampling.strict_mode))
    sample_strategy = str(_sampling_value("strategy", "random"))
    sample_focus = str(_sampling_value("focus", "combined"))
    image_download_mode = "none" if strict_sample_mode else str(_sampling_value("image_download_mode", base_config.sampling.image_download_mode))

    return PipelineRuntime(
        project_root=project_root,
        base_config=base_config,
        task_config=task_config,
        raw_dir=raw_dir,
        image_dir=image_dir,
        processed_dir=processed_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
        download_data=bool(base_config.pipeline.download_data),
        strict_sample_mode=strict_sample_mode,
        sample_size=int(_sampling_value("sample_size", base_config.sampling.sample_size)),
        sample_seed=int(_sampling_value("seed", base_config.sampling.seed)),
        sample_strategy=sample_strategy,
        sample_focus=sample_focus,
        image_download_mode=image_download_mode,
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
    print(f"Strict sample mode: {runtime.strict_sample_mode}")
    print(
        f"Sample size: {runtime.sample_size} | sample strategy: {runtime.sample_strategy} | sample focus: {runtime.sample_focus} | split ratios: {runtime.split_ratios} | seed: {runtime.sample_seed}"
    )
    print(f"Download data: {runtime.download_data} | image mode: {runtime.image_download_mode}")
    strategy_name = str(getattr(runtime.task_config.model, "strategy", "baseline_cnn"))
    backbone_name = getattr(runtime.task_config.backbone, "name", None)
    summary_bits = [f"Strategy: {strategy_name}"]
    if backbone_name:
        summary_bits.append(f"backbone: {backbone_name}")
    summary_bits.append(f"pretrained: {bool(runtime.task_config.backbone.pretrained)}")
    summary_bits.append(f"freeze_backbone: {bool(runtime.task_config.backbone.freeze_backbone)}")
    print(" | ".join(summary_bits))
    preprocessing_config = _get_preprocessing_config(runtime.task_config)
    default_preprocessing_strategy = "baseline_task2" if task_name == "relation" else "baseline_task1"
    preprocessing_strategy = str(getattr(preprocessing_config, "strategy", default_preprocessing_strategy))
    print(f"Preprocessing strategy: {preprocessing_strategy}")
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


def _prepare_object_attribute_split_files(runtime: PipelineRuntime) -> Optional[List[int]]:
    sample_image_ids = None

    build_object_attribute_vocab_and_splits(
        raw_dir=str(runtime.raw_dir),
        processed_dir=str(runtime.processed_dir),
        max_objects=int(runtime.task_config.labels.max_objects),
        max_attributes=int(runtime.task_config.labels.max_attributes),
        sample_image_ids=sample_image_ids,
        sample_size=int(runtime.sample_size),
        sample_strategy=runtime.sample_strategy,
        sample_focus=runtime.sample_focus,
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


def _prepare_relation_split_files(runtime: PipelineRuntime) -> Optional[List[int]]:
    sample_image_ids = None

    build_relation_vocab_and_splits(
        raw_dir=str(runtime.raw_dir),
        processed_dir=str(runtime.processed_dir),
        max_relations=int(runtime.task_config.labels.max_relations),
        sample_image_ids=sample_image_ids,
        sample_size=int(runtime.sample_size),
        sample_strategy=runtime.sample_strategy,
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

    _prepare_raw_data(runtime, "Task 1")
    _prepare_object_attribute_split_files(runtime)

    preprocessing_cfg = _get_preprocessing_config(runtime.task_config)
    preprocessing_color_jitter_cfg = getattr(preprocessing_cfg, "color_jitter", None)
    object_input_mode = normalize_input_mode(str(runtime.task_config.dataset.object_input_mode))
    object_strategy = str(getattr(runtime.task_config.model, "strategy", "modern_cnn"))
    object_preprocessing_strategy = str(getattr(preprocessing_cfg, "strategy", "baseline_task1"))
    _download_missing_images_for_splits(runtime, "Task 1", ["train", "val", "test"])

    object_train_ds, object_val_ds, object_test_ds = build_object_datasets(
        processed_dir=str(runtime.processed_dir),
        image_dir=str(runtime.image_dir),
        roi_size=runtime.roi_size,
        input_mode=object_input_mode,
        max_samples=runtime.max_samples,
        train_horizontal_flip_p=float(getattr(preprocessing_cfg, "random_horizontal_flip", 0.5)),
        train_color_jitter=bool(getattr(preprocessing_color_jitter_cfg, "enabled", False)),
        train_brightness=float(getattr(preprocessing_color_jitter_cfg, "brightness", 0.2)),
        train_contrast=float(getattr(preprocessing_color_jitter_cfg, "contrast", 0.2)),
        train_saturation=float(getattr(preprocessing_color_jitter_cfg, "saturation", 0.2)),
        train_hue=float(getattr(preprocessing_color_jitter_cfg, "hue", 0.1)),
        train_random_erasing_p=float(getattr(preprocessing_cfg, "random_erasing_p", 0.0)),
        train_resize_delta=int(getattr(preprocessing_cfg, "resize_delta", 32)),
        preprocessing_strategy=object_preprocessing_strategy,
        mean=runtime.feature_mean,
        std=runtime.feature_std,
    )

    object_model = build_object_classifier(
        object_strategy,
        num_classes=object_train_ds.num_objects,
        pretrained=bool(runtime.task_config.backbone.pretrained),
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        device=runtime.device,
        hidden_dim=int(runtime.task_config.model.object_hidden_dim),
        dropout=float(runtime.task_config.model.object_dropout),
        num_layers=int(runtime.task_config.model.object_num_layers),
    )

    object_lr = float(runtime.task_config.training.lr)
    object_weight_decay = float(runtime.task_config.training.weight_decay)
    backbone_params = []
    head_params = []
    for name, parameter in object_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("encoder."):
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": object_lr, "weight_decay": object_weight_decay})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": object_lr * 0.1, "weight_decay": object_weight_decay})

    object_optimizer = torch.optim.AdamW(param_groups)

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

    _prepare_raw_data(runtime, "Task 1")
    _prepare_object_attribute_split_files(runtime)

    preprocessing_cfg = _get_preprocessing_config(runtime.task_config)
    preprocessing_color_jitter_cfg = getattr(preprocessing_cfg, "color_jitter", None)
    attribute_input_mode = normalize_input_mode(str(runtime.task_config.dataset.attribute_input_mode))
    attribute_preprocessing_strategy = str(getattr(preprocessing_cfg, "strategy", "baseline_task1"))
    attribute_strategy = str(getattr(runtime.task_config.model, "strategy", "baseline_cnn"))
    _download_missing_images_for_splits(runtime, "Task 1", ["train", "val", "test"])

    attribute_train_ds, attribute_val_ds, attribute_test_ds = build_attribute_datasets(
        processed_dir=str(runtime.processed_dir),
        image_dir=str(runtime.image_dir),
        roi_size=runtime.roi_size,
        input_mode=attribute_input_mode,
        max_samples=runtime.max_samples,
        train_horizontal_flip_p=float(getattr(preprocessing_cfg, "random_horizontal_flip", 0.5)),
        train_color_jitter=bool(getattr(preprocessing_color_jitter_cfg, "enabled", False)),
        train_brightness=float(getattr(preprocessing_color_jitter_cfg, "brightness", 0.2)),
        train_contrast=float(getattr(preprocessing_color_jitter_cfg, "contrast", 0.2)),
        train_saturation=float(getattr(preprocessing_color_jitter_cfg, "saturation", 0.2)),
        train_hue=float(getattr(preprocessing_color_jitter_cfg, "hue", 0.1)),
        train_random_erasing_p=float(getattr(preprocessing_cfg, "random_erasing_p", 0.0)),
        train_resize_delta=int(getattr(preprocessing_cfg, "resize_delta", 32)),
        preprocessing_strategy=attribute_preprocessing_strategy,
        mean=runtime.feature_mean,
        std=runtime.feature_std,
    )

    attribute_model = build_attribute_classifier(
        attribute_strategy,
        num_attributes=attribute_train_ds.num_attributes,
        pretrained=bool(runtime.task_config.backbone.pretrained),
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        device=runtime.device,
        hidden_dim=int(runtime.task_config.model.attribute_hidden_dim),
        dropout=float(runtime.task_config.model.attribute_dropout),
        num_layers=int(runtime.task_config.model.attribute_num_layers),
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

    _prepare_raw_data(runtime, "Task 2")
    _prepare_relation_split_files(runtime)

    preprocessing_cfg = _get_preprocessing_config(runtime.task_config)
    preprocessing_color_jitter_cfg = getattr(preprocessing_cfg, "color_jitter", None)
    relation_input_mode = normalize_input_mode(str(runtime.task_config.dataset.input_mode))
    relation_strategy = str(getattr(runtime.task_config.model, "strategy", "baseline_cnn"))
    relation_preprocessing_strategy = str(getattr(preprocessing_cfg, "strategy", "baseline_task2"))
    _download_missing_images_for_splits(runtime, "Task 2", ["train", "val", "test"])

    relation_train_ds, relation_val_ds, relation_test_ds = build_relation_datasets(
        processed_dir=str(runtime.processed_dir),
        image_dir=str(runtime.image_dir),
        roi_size=runtime.roi_size,
        use_spatial_features=bool(runtime.task_config.spatial.use_spatial_features),
        input_mode=relation_input_mode,
        max_samples=runtime.max_samples,
        train_horizontal_flip_p=float(getattr(preprocessing_cfg, "random_horizontal_flip", 0.0)),
        train_color_jitter=bool(getattr(preprocessing_color_jitter_cfg, "enabled", False)),
        train_brightness=float(getattr(preprocessing_color_jitter_cfg, "brightness", 0.2)),
        train_contrast=float(getattr(preprocessing_color_jitter_cfg, "contrast", 0.2)),
        train_saturation=float(getattr(preprocessing_color_jitter_cfg, "saturation", 0.1)),
        train_hue=float(getattr(preprocessing_color_jitter_cfg, "hue", 0.05)),
        train_random_erasing_p=float(getattr(preprocessing_cfg, "random_erasing_p", 0.0)),
        train_resize_delta=int(getattr(preprocessing_cfg, "resize_delta", 0)),
        preprocessing_strategy=relation_preprocessing_strategy,
        mean=runtime.feature_mean,
        std=runtime.feature_std,
    )

    relation_model = build_relation_classifier(
        relation_strategy,
        num_relations=relation_train_ds.num_relations,
        spatial_dim=int(runtime.task_config.spatial.spatial_dim),
        pretrained=bool(runtime.task_config.backbone.pretrained),
        freeze_backbone=bool(runtime.task_config.backbone.freeze_backbone),
        device=runtime.device,
        hidden_dim=int(runtime.task_config.model.hidden_dim),
        dropout=float(runtime.task_config.model.dropout),
        num_layers=int(runtime.task_config.model.num_layers),
        attention_heads=int(runtime.task_config.model.attention_heads),
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
