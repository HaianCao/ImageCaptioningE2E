# Visual Genome Captioning E2E

Config-driven Visual Genome pipelines for object, attribute, and relation classification, plus a lightweight caption demo wrapper.

## Overview

The repository is organized around three independent public pipelines:

- object classification
- attribute classification
- relation classification

Each pipeline has its own notebook, task-specific YAML config, trainer, and checkpoint flow. Object, attribute, and relation are separate tasks. Shared preprocessing and feature code still uses `task1`/`task2` names in a compatibility layer, but new entry points should use `object`, `attribute`, and `relation`.

The recommended entry points are [notebooks/object_pipeline.ipynb](notebooks/object_pipeline.ipynb), [notebooks/attribute_pipeline.ipynb](notebooks/attribute_pipeline.ipynb), [notebooks/relation_pipeline.ipynb](notebooks/relation_pipeline.ipynb), and [notebooks/e2e_wrapper.ipynb](notebooks/e2e_wrapper.ipynb). [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb) remains available as a thin legacy wrapper around the shared pipeline runner.

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

4. Open [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb) only if you want the legacy combined wrapper over the shared pipeline runner.

5. Edit the YAML configs, not the notebook, for the main runtime controls.

Key settings:

- `pipeline.training_mode`: `object`, `attribute`, `relation`, `all`, or `e2e`.
- Legacy aliases `task1`, `task2`, and `both` remain available for backward compatibility.
- `pipeline.download_data`: download missing Visual Genome metadata and images.
- `object.model.strategy`: registered object model preset such as `modern_cnn`, `baseline_cnn`, or `transformer`.
- `attribute.model.strategy`: registered attribute model preset such as `baseline_cnn`, `modern_cnn`, or `transformer`.
- `relation.model.strategy`: registered relation model preset such as `baseline_cnn`, `modern_cnn`, or `transformer`.
- `preprocessing.strategy`: registered preprocessing preset such as `baseline_task1` or `baseline_task2`.
- `sampling.strict_mode`: sample image IDs first, then split by image ID.
- `sampling.sample_size`: number of image IDs used in strict sample mode.
- `sampling.image_download_mode`: `none`, `sample`, or `all`.
- `sampling.max_samples`: optional debug cap per split.
- `backbone.freeze_backbone`: freeze the visual backbone or train it end to end.

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
    ├── pipelines/
    ├── data/
    ├── evaluation/
    ├── features/
    ├── models/
    ├── training/
    └── utils/
```

## Strategy Presets

| Strategy key | Backbone | Notes |
|---|---|---|
| `baseline_cnn` | `resnet50` | Default public preset for attribute and relation |
| `modern_cnn` | `convnextv2_tiny` | Stronger CNN preset used by the object pipeline |
| `efficientnetv2_s` | `efficientnetv2_s` | Lightweight modern CNN preset |
| `transformer` | `swin_tiny_patch4_window7_224` | Attention-based preset |

The registries also accept raw backbone aliases such as `resnet50` and `convnextv2_tiny` for compatibility, but the public configs default to the strategy keys above.

## Preprocess Presets

| Strategy key | Applies to | Notes |
|---|---|---|
| `baseline_task1` | object, attribute | Shared ROI preprocessing preset with flip, optional jitter, and normalization |
| `baseline_task2` | relation | Geometry-safe preset with no horizontal flip by default |

The preprocessing registry keeps the contract consistent: each strategy returns the same kind of image transform pair, so the pipeline can swap presets without changing dataset code.

## Model Behavior

- `ObjectClassifier` is a single-label classifier: each ROI predicts exactly one object class with a softmax / cross-entropy head.
- `AttributeClassifier` is a multi-label classifier: each ROI can activate multiple attributes at once with a sigmoid / BCE head.
- The attribute output dimension is fixed by the attribute vocabulary size, but the number of active labels can vary per object.
- `RelationClassifier` is also single-label over the relation vocabulary.
- `VisualGenomeE2EModel` is a thin wrapper around the three trained checkpoints; it does not create a new joint multi-task model.

## Notes

- Object, attribute, and relation use separate configs, notebooks, trainers, and checkpoints even though they share the same preprocessing family.
- `task1`/`task2` naming is kept only in shared compatibility layers and legacy notebooks.
- The object, attribute, and relation pipelines each expose a `model.strategy` registry, so adding a new preset means registering one new strategy class and selecting it from YAML.
- The object, attribute, and relation pipelines expose a `preprocessing.strategy` registry, and the preprocessing contract stays image-to-image so decorators can be shared when their output shape is the same.
- The object pipeline uses a configurable MLP head for single-label object prediction.
- The attribute pipeline uses a configurable multi-label MLP head for attribute prediction.
- Object evaluation reports top-k accuracy, and attribute evaluation reports exact-match accuracy plus micro metrics.
- The relation pipeline uses a configurable relation classifier with `concat`, `attention`, or `gated` fusion, and the selected preset can switch the backbone and fusion behavior together.
- The public training pipelines run directly on image ROIs; there is no pre-extract feature stage in the public notebooks.
- See [requirements.txt](requirements.txt) for the full dependency list.
