"""
Feature extraction utilities for pre-computing visual features.

Supports various backbones: ResNet, EfficientNet, ViT.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm

import timm
from PIL import Image
import torchvision.transforms as T

from .roi_extractor import extract_roi, extract_union_roi


class FeatureExtractor(nn.Module):
    """
    Wrapper around pretrained vision backbones for feature extraction.

    Supports:
    - ResNet (50, 101, 152)
    - EfficientNet (B0-B7)
    - Vision Transformer (ViT-Base, ViT-Large)
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        feature_dim: Optional[int] = None,
        device: str = "cuda",
        resize_size: int = 256,
        crop_size: int = 224,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.device = device

        # Create backbone
        if backbone.startswith("resnet"):
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        elif backbone.startswith("efficientnet"):
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        elif backbone.startswith("vit"):
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if feature_dim and feature_dim != self.feature_dim:
            self.backbone.head = nn.Linear(self.feature_dim, feature_dim)
            self.feature_dim = feature_dim

        self.backbone.to(device)
        self.backbone.eval()

        # Image preprocessing
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    @torch.no_grad()
    def extract_features(self, images: Union[Image.Image, torch.Tensor, List]) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: Single PIL Image, tensor, or list of images

        Returns:
            Features tensor of shape (N, feature_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, torch.Tensor) and images.dim() == 3:
            images = [images]

        if isinstance(images, list):
            # Convert PIL images to tensors
            tensors = []
            for img in images:
                if isinstance(img, Image.Image):
                    tensors.append(self.transform(img))
                else:
                    tensors.append(img)
            batch = torch.stack(tensors).to(self.device)
        else:
            batch = images.to(self.device)

        features = self.backbone(batch)
        return features.cpu()

    def extract_roi_features(
        self,
        image: Image.Image,
        bboxes: List[List[int]],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Extract features from multiple ROIs in an image.

        Args:
            image: PIL Image
            bboxes: List of [x, y, w, h] bounding boxes
            batch_size: Batch size for processing

        Returns:
            Features tensor of shape (N, feature_dim)
        """
        if not bboxes:
            return torch.empty(0, self.feature_dim)

        # Crop ROIs
        roi_images = [extract_roi(image, tuple(bbox)) for bbox in bboxes]

        # Process in batches
        all_features = []
        for i in range(0, len(roi_images), batch_size):
            batch_rois = roi_images[i:i + batch_size]
            batch_features = self.extract_features(batch_rois)
            all_features.append(batch_features)

        return torch.cat(all_features, dim=0)

    def extract_union_features(
        self,
        image: Image.Image,
        bbox_pairs: List[Tuple[List[int], List[int]]],
        batch_size: int = 32,
        padding: int = 10
    ) -> torch.Tensor:
        """
        Extract features from union bounding boxes of object pairs.

        Args:
            image: PIL Image
            bbox_pairs: List of (subj_bbox, obj_bbox) tuples
            batch_size: Batch size for processing
            padding: Padding around union box

        Returns:
            Features tensor of shape (N, feature_dim)
        """
        if not bbox_pairs:
            return torch.empty(0, self.feature_dim)

        union_images = [extract_union_roi(image, tuple(subj), tuple(obj), padding=padding) 
                        for subj, obj in bbox_pairs]

        # Process in batches
        all_features = []
        for i in range(0, len(union_images), batch_size):
            batch_unions = union_images[i:i + batch_size]
            batch_features = self.extract_features(batch_unions)
            all_features.append(batch_features)

        return torch.cat(all_features, dim=0)


def extract_task1_features(
    annotation_file: str,
    image_dir: str,
    output_file: str,
    backbone: str = "resnet50",
    pretrained: bool = True,
    batch_size: int = 32,
    device: str = "cuda",
    resize_size: int = 256,
    crop_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> None:
    """
    Pre-extract features for Task 1 dataset.

    Args:
        annotation_file: Path to task1 annotations JSON
        image_dir: Directory containing images
        output_file: Output .pt file for features
        backbone: Backbone model name
        batch_size: Processing batch size
        device: Device for extraction
    """
    import json
    from collections import defaultdict

    print(f"Extracting Task 1 features using {backbone}...")

    # Load annotations
    with open(annotation_file, 'r') as f:
        raw = json.load(f)
        samples = raw if isinstance(raw, list) else raw.get("samples", [])

    # Group by image_id
    image_groups = defaultdict(list)
    for sample in samples:
        image_groups[sample['image_id']].append(sample)

    # Initialize extractor
    extractor = FeatureExtractor(
        backbone=backbone,
        pretrained=pretrained,
        device=device,
        resize_size=resize_size,
        crop_size=crop_size,
        mean=mean,
        std=std,
    )

    # Process each image
    all_features = {}
    for image_id, img_samples in tqdm(image_groups.items(), desc="Processing images"):
        image_path = Path(image_dir) / f"{image_id}.jpg"
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert('RGB')
        bboxes = [s['bbox'] for s in img_samples]

        features = extractor.extract_roi_features(image, bboxes, batch_size=batch_size)

        # Store features by object_id so the dataset can look them up directly.
        for i, sample in enumerate(img_samples):
            feature_key = sample.get('object_id', i)
            all_features[str(feature_key)] = features[i]

    # Save features
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_features, output_path)
    print(f"Saved {len(all_features)} features to {output_path}")


def extract_task2_features(
    annotation_file: str,
    image_dir: str,
    output_file: str,
    backbone: str = "resnet50",
    pretrained: bool = True,
    batch_size: int = 32,
    device: str = "cuda",
    resize_size: int = 256,
    crop_size: int = 224,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> None:
    """
    Pre-extract features for Task 2 dataset.

    Args:
        annotation_file: Path to task2 annotations JSON
        image_dir: Directory containing images
        output_file: Output .pt file for features
        backbone: Backbone model name
        batch_size: Processing batch size
        device: Device for extraction
    """
    import json
    from collections import defaultdict

    print(f"Extracting Task 2 features using {backbone}...")

    # Load annotations
    with open(annotation_file, 'r') as f:
        raw = json.load(f)
        samples = raw if isinstance(raw, list) else raw.get("samples", [])

    # Group by image_id
    image_groups = defaultdict(list)
    for sample in samples:
        image_groups[sample['image_id']].append(sample)

    # Initialize extractor
    extractor = FeatureExtractor(
        backbone=backbone,
        pretrained=pretrained,
        device=device,
        resize_size=resize_size,
        crop_size=crop_size,
        mean=mean,
        std=std,
    )

    # Process each image
    all_features = {}
    for image_id, img_samples in tqdm(image_groups.items(), desc="Processing images"):
        image_path = Path(image_dir) / f"{image_id}.jpg"
        if not image_path.exists():
            continue

        image = Image.open(image_path).convert('RGB')
        bbox_pairs = [(s['subject_bbox'], s['object_bbox']) for s in img_samples]

        features = extractor.extract_union_features(image, bbox_pairs, batch_size=batch_size)

        # Store features by relationship_id
        for i, sample in enumerate(img_samples):
            rel_id = sample.get('relationship_id', i)
            all_features[str(rel_id)] = features[i]

    # Save features
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_features, output_path)
    print(f"Saved {len(all_features)} features to {output_path}")


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor("resnet50")
    print(f"Feature dim: {extractor.feature_dim}")

    # Test with dummy image
    dummy_img = Image.new('RGB', (224, 224), color='red')
    features = extractor.extract_features(dummy_img)
    print(f"Features shape: {features.shape}")