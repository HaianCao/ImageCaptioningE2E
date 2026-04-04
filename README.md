# Visual Genome Captioning E2E

End-to-end pipeline for Visual Genome object/attribute classification, relationship classification, and caption demo generation.

## Overview

The recommended entry point is [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb). It reads runtime settings from YAML instead of notebook constants.

Configuration is split across:

- [configs/config.yaml](configs/config.yaml)
- [configs/task1_config.yaml](configs/task1_config.yaml)
- [configs/task2_config.yaml](configs/task2_config.yaml)

## Quick Start

1. Clone the repository.

```bash
git clone https://github.com/HaianCao/ImageCaptioningE2E.git
cd ImageCaptioningE2E
```

2. Open and run [notebooks/complete_pipeline.ipynb](notebooks/complete_pipeline.ipynb).

3. Edit the YAML configs, not the notebook, for the main runtime controls.

Key settings:

- `pipeline.training_mode`: `task1`, `task2`, or `both`
- `pipeline.download_data`: download missing Visual Genome metadata and images
- `pipeline.pre_extract_features`: precompute and cache features before training
- `sampling.strict_mode`: sample image IDs first, then split by image ID
- `sampling.sample_size`: number of image IDs used in strict sample mode
- `sampling.image_download_mode`: `none`, `sample`, or `all`
- `sampling.max_samples`: optional debug cap per split
- `feature_extraction.batch_size`: batch size used for feature extraction
- `feature_extraction.resize_size` / `feature_extraction.crop_size`: image preprocessing sizes

4. Run the notebook end to end.

- First run: keep `pipeline.download_data` enabled if raw files are missing.
- Debug runs: set `sampling.max_samples` to a small number.
- Strict sample mode: only the selected sample image IDs are split and downloaded.

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
│   ├── task1_config.yaml
│   └── task2_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
├── logs/
├── notebooks/
│   ├── complete_pipeline.ipynb
│   ├── 00_setup_and_download.ipynb
│   ├── task1_train.ipynb
│   └── task2_train.ipynb
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

## Notes

- Task 1 uses configurable MLP heads for object and attribute classification.
- Task 2 uses a configurable relation classifier with `concat`, `attention`, or `gated` fusion.
- Feature caches are stored under each task's processed directory in a relative `features/` folder.
- See [requirements.txt](requirements.txt) for the full dependency list.
