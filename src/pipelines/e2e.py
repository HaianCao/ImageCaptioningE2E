"""Reusable end-to-end caption demo helpers for the Visual Genome project."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from ..data.attribute_dataset import build_attribute_datasets
from ..data.object_dataset import build_object_datasets
from ..data.relation_dataset import build_relation_datasets
from ..models.caption import CaptionGenerator
from ..models.e2e import VisualGenomeE2EModel
from ..models.task1.attribute_strategies import build_attribute_classifier
from ..models.task1.object_strategies import build_object_classifier
from ..models.task2.relation_strategies import build_relation_classifier
from ..utils.config_loader import load_config, load_task_configs
from ..utils.memory import cleanup_cuda_memory
from .common import get_device, normalize_input_mode, require_files, resolve_project_root, seed_everything


DEFAULT_CHECKPOINT_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "object": ("object_", "task1_object_"),
    "attribute": ("attribute_", "task1_attribute_"),
    "relation": ("relation_", "task2_"),
}


@dataclass(frozen=True)
class E2ECaptionDemoConfig:
    """Configuration for the lightweight caption demo."""

    base_config_path: str = "configs/config.yaml"
    object_config_path: str = "configs/object_config.yaml"
    attribute_config_path: str = "configs/attribute_config.yaml"
    relation_config_path: str = "configs/relation_config.yaml"
    sample_index: int = 0
    dataset_max_samples: Optional[int] = 1
    prediction_top_k: int = 5
    caption_top_k: int = 3
    templates_file: Optional[str] = None


@dataclass(frozen=True)
class E2ECaptionDemoResult:
    """Structured output returned by the demo runner."""

    project_root: Path
    device: str
    sample_index: int
    checkpoint_paths: Dict[str, Optional[str]]
    checkpoint_metadata: Dict[str, Dict[str, Any]]
    model_summary: Dict[str, Any]
    sample_metadata: Dict[str, Dict[str, Any]]
    object_predictions: List[Tuple[str, float]]
    attribute_predictions: List[Tuple[str, float]]
    relation_predictions: List[Tuple[str, float]]
    captions: List[str]

    def describe(self) -> str:
        """Render a concise text summary for notebook output."""
        lines: List[str] = ["========== E2E CAPTION DEMO =========="]
        lines.append(f"Project root: {self.project_root}")
        lines.append(f"Device: {self.device}")
        lines.append(f"Sample index: {self.sample_index}")
        lines.append("Checkpoint paths:")
        for name, path in self.checkpoint_paths.items():
            lines.append(f"  {name}: {path}")
        lines.append("Checkpoint metadata:")
        for name, metadata in self.checkpoint_metadata.items():
            lines.append(f"  {name}: {metadata}")
        lines.append("Model summary:")
        for name, summary in self.model_summary.items():
            lines.append(f"  {name}: {summary}")
        lines.append("Sample metadata:")
        for name, metadata in self.sample_metadata.items():
            lines.append(f"  {name}: {metadata}")
        lines.append("Top predictions:")
        lines.extend(_format_prediction_block("object", self.object_predictions))
        lines.extend(_format_prediction_block("attribute", self.attribute_predictions))
        lines.extend(_format_prediction_block("relation", self.relation_predictions))
        lines.append("Generated captions:")
        if self.captions:
            lines.extend(f"  - {caption}" for caption in self.captions)
        else:
            lines.append("  - <none>")
        return "\n".join(lines)


def _resolve_path(project_root: Path, configured_path: str) -> Path:
    return project_root / Path(configured_path)


def _load_vocab_file(vocab_path: Path) -> Dict[str, int]:
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _inverse_vocab(vocab: Mapping[str, int]) -> Dict[int, str]:
    return {int(index): label for label, index in vocab.items()}


def _find_checkpoint_file(checkpoint_root: Path, prefixes: Sequence[str]) -> Path:
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {checkpoint_root}")

    candidates = [
        path
        for path in checkpoint_root.rglob("*.pth")
        if any(path.name.startswith(prefix) for prefix in prefixes)
    ]
    if not candidates:
        prefixes_text = ", ".join(prefixes)
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_root} for prefixes: {prefixes_text}")

    def score(path: Path) -> Tuple[float, float]:
        try:
            checkpoint = torch.load(path, map_location="cpu")
            if isinstance(checkpoint, dict):
                metadata = checkpoint.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                metric = metadata.get("metric", checkpoint.get("metric", float("-inf")))
                if metric is None:
                    metric = float("-inf")
            else:
                metric = float("-inf")
        except Exception:
            metric = float("-inf")
        return float(metric), float(path.stat().st_mtime)

    return max(candidates, key=score)


def _checkpoint_uses_backbone(checkpoint_path: Path) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        return False
    return any(key.startswith(("encoder.", "feature_projection.")) for key in state_dict.keys())


def _augmentation_kwargs(preprocessing_config: Any) -> Dict[str, Any]:
    color_jitter_config = getattr(preprocessing_config, "color_jitter", None)
    return {
        "train_horizontal_flip_p": float(getattr(preprocessing_config, "random_horizontal_flip", 0.0)),
        "train_color_jitter": bool(getattr(color_jitter_config, "enabled", False)),
        "train_brightness": float(getattr(color_jitter_config, "brightness", 0.2)),
        "train_contrast": float(getattr(color_jitter_config, "contrast", 0.2)),
        "train_saturation": float(getattr(color_jitter_config, "saturation", 0.2)),
        "train_hue": float(getattr(color_jitter_config, "hue", 0.1)),
        "train_random_erasing_p": float(getattr(preprocessing_config, "random_erasing_p", 0.0)),
        "train_resize_delta": int(getattr(preprocessing_config, "resize_delta", 0)),
    }


def _task1_dataset_kwargs(task_config: Any, base_config: Any, input_mode: str, max_samples: Optional[int]) -> Dict[str, Any]:
    preprocessing_config = task_config.preprocessing
    kwargs = _augmentation_kwargs(preprocessing_config)
    kwargs.update(
        {
            "roi_size": int(base_config.image.roi_size),
            "input_mode": normalize_input_mode(str(input_mode)),
            "max_samples": max_samples,
            "preprocessing_strategy": str(getattr(preprocessing_config, "strategy", "baseline_task1")),
            "mean": [float(value) for value in base_config.image.mean],
            "std": [float(value) for value in base_config.image.std],
        }
    )
    return kwargs


def _task2_dataset_kwargs(task_config: Any, base_config: Any, input_mode: str, max_samples: Optional[int]) -> Dict[str, Any]:
    preprocessing_config = task_config.preprocessing
    kwargs = _augmentation_kwargs(preprocessing_config)
    kwargs.update(
        {
            "roi_size": int(base_config.image.roi_size),
            "use_spatial_features": bool(task_config.spatial.use_spatial_features),
            "input_mode": normalize_input_mode(str(input_mode)),
            "max_samples": max_samples,
            "preprocessing_strategy": str(getattr(preprocessing_config, "strategy", "baseline_task2")),
            "mean": [float(value) for value in base_config.image.mean],
            "std": [float(value) for value in base_config.image.std],
        }
    )
    return kwargs


def _ensure_processed_files(processed_dir: Path, task_name: str) -> None:
    if task_name == "task1":
        require_files(
            [
                processed_dir / "object_vocab.json",
                processed_dir / "attribute_vocab.json",
                processed_dir / "train" / "annotations.json",
                processed_dir / "val" / "annotations.json",
                processed_dir / "test" / "annotations.json",
            ],
            f"task1 processed files at {processed_dir}",
        )
        return

    if task_name == "task2":
        require_files(
            [
                processed_dir / "relation_vocab.json",
                processed_dir / "train" / "annotations.json",
                processed_dir / "val" / "annotations.json",
                processed_dir / "test" / "annotations.json",
            ],
            f"task2 processed files at {processed_dir}",
        )
        return

    raise ValueError(f"Unknown task name: {task_name}")


def _batch_tensor(sample: Dict[str, Any], keys: Sequence[str], device: str) -> torch.Tensor:
    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        else:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device)
    keys_text = ", ".join(keys)
    raise KeyError(f"No tensor found in sample for keys: {keys_text}")


def _top_class_predictions(logits: torch.Tensor, idx_to_label: Dict[int, str], top_k: int) -> List[Tuple[str, float]]:
    probabilities = torch.softmax(logits, dim=1)[0]
    top_k = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=top_k)
    return [
        (idx_to_label.get(int(index), str(int(index))), float(value))
        for value, index in zip(values, indices)
    ]


def _top_attribute_predictions(
    logits: torch.Tensor,
    idx_to_label: Dict[int, str],
    top_k: int,
    threshold: float = 0.5,
) -> List[Tuple[str, float]]:
    probabilities = torch.sigmoid(logits)[0]
    top_k = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=top_k)
    decoded = [
        (idx_to_label.get(int(index), str(int(index))), float(value))
        for value, index in zip(values, indices)
    ]
    active = [item for item in decoded if item[1] >= threshold]
    return active if active else decoded[:1]


def _format_prediction_block(label: str, predictions: Sequence[Tuple[str, float]]) -> List[str]:
    if not predictions:
        return [f"  {label}: <none>"]
    return [f"  {label}: {name} ({score:.4f})" for name, score in predictions]


def run_e2e_caption_demo(
    config: Optional[E2ECaptionDemoConfig] = None,
) -> E2ECaptionDemoResult:
    """Run the end-to-end caption demo and return a structured summary."""
    config = config or E2ECaptionDemoConfig()
    resolved_project_root = resolve_project_root()

    base_config_path = _resolve_path(resolved_project_root, config.base_config_path)
    object_config_path = _resolve_path(resolved_project_root, config.object_config_path)
    attribute_config_path = _resolve_path(resolved_project_root, config.attribute_config_path)
    relation_config_path = _resolve_path(resolved_project_root, config.relation_config_path)

    base_config = load_config(base_config_path)
    object_config = load_task_configs(base_config_path, object_config_path)
    attribute_config = load_task_configs(base_config_path, attribute_config_path)
    relation_config = load_task_configs(base_config_path, relation_config_path)

    seed_everything(int(base_config.seed))
    device = get_device(str(base_config.device))

    image_dir = _resolve_path(resolved_project_root, str(base_config.dataset.image_dir))
    checkpoint_root = _resolve_path(resolved_project_root, str(base_config.paths.checkpoint_dir))

    object_processed_dir = _resolve_path(resolved_project_root, str(object_config.dataset.processed_dir))
    attribute_processed_dir = _resolve_path(resolved_project_root, str(attribute_config.dataset.processed_dir))
    relation_processed_dir = _resolve_path(resolved_project_root, str(relation_config.dataset.processed_dir))

    _ensure_processed_files(object_processed_dir, "task1")
    _ensure_processed_files(attribute_processed_dir, "task1")
    _ensure_processed_files(relation_processed_dir, "task2")

    object_vocab = _load_vocab_file(object_processed_dir / "object_vocab.json")
    attribute_vocab = _load_vocab_file(attribute_processed_dir / "attribute_vocab.json")
    relation_vocab = _load_vocab_file(relation_processed_dir / "relation_vocab.json")

    idx_to_object = _inverse_vocab(object_vocab)
    idx_to_attribute = _inverse_vocab(attribute_vocab)
    idx_to_relation = _inverse_vocab(relation_vocab)

    checkpoint_paths = {
        name: _find_checkpoint_file(checkpoint_root, prefixes)
        for name, prefixes in DEFAULT_CHECKPOINT_PREFIXES.items()
    }
    checkpoint_compatibility = {
        name: _checkpoint_uses_backbone(path)
        for name, path in checkpoint_paths.items()
    }
    incompatible = [name for name, uses_backbone in checkpoint_compatibility.items() if not uses_backbone]
    if incompatible:
        joined_names = ", ".join(incompatible)
        raise RuntimeError(
            "The E2E caption demo expects image-mode checkpoints. "
            f"Feature-mode checkpoints are not supported for: {joined_names}."
        )

    object_strategy_name = str(object_config.model.strategy)
    attribute_strategy_name = str(attribute_config.model.strategy)
    relation_strategy_name = str(relation_config.model.strategy)

    object_model = build_object_classifier(
        object_strategy_name,
        num_classes=len(object_vocab),
        pretrained=bool(object_config.backbone.pretrained),
        freeze_backbone=bool(object_config.backbone.freeze_backbone),
        device=device,
        hidden_dim=int(object_config.model.object_hidden_dim),
        dropout=float(object_config.model.object_dropout),
        num_layers=int(object_config.model.object_num_layers),
    )
    attribute_model = build_attribute_classifier(
        attribute_strategy_name,
        num_attributes=len(attribute_vocab),
        pretrained=bool(attribute_config.backbone.pretrained),
        freeze_backbone=bool(attribute_config.backbone.freeze_backbone),
        device=device,
        hidden_dim=int(attribute_config.model.attribute_hidden_dim),
        dropout=float(attribute_config.model.attribute_dropout),
        num_layers=int(attribute_config.model.attribute_num_layers),
    )
    relation_model = build_relation_classifier(
        relation_strategy_name,
        num_relations=len(relation_vocab),
        spatial_dim=int(relation_config.spatial.spatial_dim),
        pretrained=bool(relation_config.backbone.pretrained),
        freeze_backbone=bool(relation_config.backbone.freeze_backbone),
        device=device,
        hidden_dim=int(relation_config.model.hidden_dim),
        dropout=float(relation_config.model.dropout),
        num_layers=int(relation_config.model.num_layers),
        attention_heads=int(relation_config.model.attention_heads),
    )

    e2e_model = VisualGenomeE2EModel.from_models(
        object_model=object_model,
        attribute_model=attribute_model,
        relation_model=relation_model,
    )
    checkpoint_metadata = e2e_model.load_checkpoints(
        object_checkpoint=str(checkpoint_paths["object"]),
        attribute_checkpoint=str(checkpoint_paths["attribute"]),
        relation_checkpoint=str(checkpoint_paths["relation"]),
        strict=True,
    )
    e2e_model.eval()

    caption_templates_file = None
    if config.templates_file is not None:
        caption_templates_file = _resolve_path(resolved_project_root, config.templates_file)
        if not caption_templates_file.exists():
            raise FileNotFoundError(f"Caption templates file not found: {caption_templates_file}")

    caption_generator = CaptionGenerator(
        object_vocab=object_vocab,
        attribute_vocab=attribute_vocab,
        relation_vocab=relation_vocab,
        templates_file=str(caption_templates_file) if caption_templates_file is not None else None,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        object_train_ds, object_val_ds, object_test_ds = build_object_datasets(
            processed_dir=str(object_processed_dir),
            image_dir=str(image_dir),
            **_task1_dataset_kwargs(
                object_config,
                base_config,
                object_config.dataset.object_input_mode,
                config.dataset_max_samples,
            ),
        )
        attribute_train_ds, attribute_val_ds, attribute_test_ds = build_attribute_datasets(
            processed_dir=str(attribute_processed_dir),
            image_dir=str(image_dir),
            **_task1_dataset_kwargs(
                attribute_config,
                base_config,
                attribute_config.dataset.attribute_input_mode,
                config.dataset_max_samples,
            ),
        )
        relation_train_ds, relation_val_ds, relation_test_ds = build_relation_datasets(
            processed_dir=str(relation_processed_dir),
            image_dir=str(image_dir),
            **_task2_dataset_kwargs(
                relation_config,
                base_config,
                relation_config.dataset.input_mode,
                config.dataset_max_samples,
            ),
        )

    if config.sample_index < 0:
        raise IndexError("sample_index must be non-negative")
    if config.sample_index >= len(object_test_ds):
        raise IndexError(
            f"sample_index {config.sample_index} is out of range for object dataset of length {len(object_test_ds)}"
        )
    if config.sample_index >= len(attribute_test_ds):
        raise IndexError(
            f"sample_index {config.sample_index} is out of range for attribute dataset of length {len(attribute_test_ds)}"
        )
    if config.sample_index >= len(relation_test_ds):
        raise IndexError(
            f"sample_index {config.sample_index} is out of range for relation dataset of length {len(relation_test_ds)}"
        )

    object_sample = object_test_ds[config.sample_index]
    attribute_sample = attribute_test_ds[config.sample_index]
    relation_sample = relation_test_ds[config.sample_index]

    object_inputs = _batch_tensor(object_sample, ("image", "object_image", "feature", "object_feature"), device)
    attribute_inputs = _batch_tensor(attribute_sample, ("image", "attribute_image", "feature", "attribute_feature"), device)
    relation_inputs = _batch_tensor(relation_sample, ("image", "union_image", "feature", "union_feature"), device)
    relation_spatial = _batch_tensor(relation_sample, ("spatial",), device)

    with torch.no_grad():
        logits = e2e_model(
            object_inputs=object_inputs,
            attribute_inputs=attribute_inputs,
            relation_inputs=relation_inputs,
            spatial=relation_spatial,
        )

    object_logits = logits["object_logits"]
    attribute_logits = logits["attribute_logits"]
    relation_logits = logits["relation_logits"]

    object_predictions = _top_class_predictions(object_logits, idx_to_object, config.prediction_top_k)
    attribute_predictions = _top_attribute_predictions(attribute_logits, idx_to_attribute, config.prediction_top_k)
    relation_predictions = _top_class_predictions(relation_logits, idx_to_relation, config.prediction_top_k)

    captions = caption_generator.generate_from_predictions(
        object_attribute_results={
            "object_logits": object_logits.cpu(),
            "attribute_logits": attribute_logits.cpu(),
        },
        relation_results={
            "relation_logits": relation_logits.cpu(),
            "subject_name": relation_sample.get("meta", {}).get("subject_name", ""),
            "object_name": relation_sample.get("meta", {}).get("object_name", ""),
        },
        top_k=config.caption_top_k,
    )

    result = E2ECaptionDemoResult(
        project_root=resolved_project_root,
        device=device,
        sample_index=config.sample_index,
        checkpoint_paths={name: str(path) for name, path in checkpoint_paths.items()},
        checkpoint_metadata=checkpoint_metadata,
        model_summary=e2e_model.save_summary(),
        sample_metadata={
            "object": object_sample.get("meta", {}),
            "attribute": attribute_sample.get("meta", {}),
            "relation": relation_sample.get("meta", {}),
        },
        object_predictions=object_predictions,
        attribute_predictions=attribute_predictions,
        relation_predictions=relation_predictions,
        captions=captions,
    )

    try:
        return result
    finally:
        cleanup_cuda_memory(note="E2E caption demo finished")


__all__ = [
    "E2ECaptionDemoConfig",
    "E2ECaptionDemoResult",
    "run_e2e_caption_demo",
]
