# Visual Genome Captioning E2E

Config-driven Visual Genome pipelines for object, attribute, and relation classification, plus a lightweight caption demo wrapper.

## Overview

The repository is organized around three independent public pipelines:

- object classification
- attribute classification
- relation classification

Each pipeline has its own notebook, task-specific YAML config, trainer, and checkpoint flow. Object and attribute are separate tasks. Shared preprocessing and feature code still uses `task1`/`task2` names in a compatibility layer, but new entry points should use `object`, `attribute`, and `relation`.

The recommended entry points are [notebooks/object_pipeline.ipynb](notebooks/object_pipeline.ipynb), [notebooks/attribute_pipeline.ipynb](notebooks/attribute_pipeline.ipynb), [notebooks/relation_pipeline.ipynb](notebooks/relation_pipeline.ipynb), and [notebooks/e2e_wrapper.ipynb](notebooks/e2e_wrapper.ipynb). [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb) remains available as the legacy combined workflow.

Configuration is split across:

- [configs/config.yaml](configs/config.yaml)
- [configs/object_config.yaml](configs/object_config.yaml)
- [configs/attribute_config.yaml](configs/attribute_config.yaml)
- [configs/relation_config.yaml](configs/relation_config.yaml)

## Quick Start

1. Clone the repository.

```bash
git clone https://github.com/HaianCao/ImageCaptioningE2E.git
cd ImageCaptioningE2E
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run one of the split notebooks for the pipeline you want to work on, or [notebooks/e2e_wrapper.ipynb](notebooks/e2e_wrapper.ipynb) to load checkpoints and run the caption demo.

4. Open [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb) only if you want the legacy combined workflow.

5. Edit the YAML configs, not the notebook, for the main runtime controls.

Key settings:

- `pipeline.training_mode`: `object`, `attribute`, `relation`, `all`, or `e2e`.
- Legacy aliases `task1`, `task2`, and `both` remain available for backward compatibility.
- `pipeline.download_data`: download missing Visual Genome metadata and images.
- `pipeline.pre_extract_features`: precompute and cache features before training.
- `sampling.strict_mode`: sample image IDs first, then split by image ID.
- `sampling.sample_size`: number of image IDs used in strict sample mode.
- `sampling.image_download_mode`: `none`, `sample`, or `all`.
- `sampling.max_samples`: optional debug cap per split.
- `feature_extraction.batch_size`: batch size used for feature extraction.
- `feature_extraction.resize_size` / `feature_extraction.crop_size`: image preprocessing sizes.

First-run guidance:

- Keep `pipeline.download_data` enabled if raw files are missing.
- Use `sampling.max_samples` for small debug runs.
- In strict sample mode, only the selected sample image IDs are split and downloaded.

## Data Sources

Visual Genome raw data is downloaded directly from Stanford-hosted files.

- Objects: `objects_v1_2.json.zip`
- Attributes: `attributes.json.zip`
- Relationships: `relationships_v1_2.json.zip`
- Image metadata: `image_data.json.zip`
- Images: downloaded on demand from the VG image hosts

The pipeline no longer depends on the deprecated HuggingFace loading script.

## Project Structure

```text
prj/
├── configs/
│   ├── config.yaml
│   ├── object_config.yaml
│   ├── attribute_config.yaml
│   └── relation_config.yaml
├── checkpoints/
├── data/
├── logs/
├── notebooks/
│   ├── object_pipeline.ipynb
│   ├── attribute_pipeline.ipynb
│   ├── relation_pipeline.ipynb
│   ├── e2e_wrapper.ipynb
│   └── complete_pipeline.ipynb
└── src/
    ├── data/
    ├── evaluation/
    ├── features/
    ├── models/
    ├── training/
    └── utils/
```

## Supported Backbones

| Backbone | Config key |
|---|---|
| ResNet-50 | `resnet50` |
| ResNet-101 | `resnet101` |
| EfficientNet-B3 | `efficientnet_b3` |
| EfficientNet-B5 | `efficientnet_b5` |
| ViT-Base/16 | `vit_base_16` |
| ViT-Large/16 | `vit_large_16` |

## Model Behavior

- `ObjectClassifier` is a single-label classifier: each ROI predicts exactly one object class with a softmax / cross-entropy head.
- `AttributeClassifier` is a multi-label classifier: each ROI can activate multiple attributes at once with a sigmoid / BCE head.
- The attribute output dimension is fixed by the attribute vocabulary size, but the number of active labels can vary per object.
- `RelationClassifier` is also single-label over the relation vocabulary.
- `VisualGenomeE2EModel` is a thin wrapper around the three trained checkpoints; it does not create a new joint multi-task model.

## Notes

- Object and attribute use separate configs, notebooks, trainers, and checkpoints even though they share the same preprocessing family.
- `task1`/`task2` naming is kept only in shared compatibility layers and legacy notebooks.
- The object pipeline uses a configurable MLP head for single-label object prediction.
- The attribute pipeline uses a configurable multi-label MLP head for attribute prediction.
- Object evaluation reports top-k accuracy, and attribute evaluation reports exact-match accuracy plus micro metrics.
- The relation pipeline uses a configurable relation classifier with `concat`, `attention`, or `gated` fusion.
- Feature caches are stored under each pipeline's processed directory in a relative `features/` folder.
- See [requirements.txt](requirements.txt) for the full dependency list.
