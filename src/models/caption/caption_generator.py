"""
Caption generator for combining Task 1 + Task 2 results.

Uses template-based generation for simplicity and interpretability.
"""

import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch


class CaptionGenerator:
    """
    Template-based caption generator.

    Combines object detections, attributes, and relationships into natural language descriptions.
    """

    def __init__(
        self,
        object_vocab: Optional[Dict[str, int]] = None,
        attribute_vocab: Optional[Dict[str, int]] = None,
        relation_vocab: Optional[Dict[str, int]] = None,
        templates_file: Optional[str] = None
    ):
        # Reverse vocabularies for decoding
        self.idx_to_object = {v: k for k, v in (object_vocab or {}).items()}
        self.idx_to_attribute = {v: k for k, v in (attribute_vocab or {}).items()}
        self.idx_to_relation = {v: k for k, v in (relation_vocab or {}).items()}

        # Default templates
        self.templates = [
            "{subject} that is {attributes} {relation} {object}",
            "A {subject} {attributes} {relation} a {object}",
            "The {subject} is {attributes} and {relation} the {object}",
            "{attributes} {subject} {relation} {object}",
        ]

        # Load custom templates if provided
        if templates_file and Path(templates_file).exists():
            with open(templates_file, 'r') as f:
                self.templates = [line.strip() for line in f if line.strip()]

    def generate_caption(
        self,
        subject_name: str,
        object_name: str,
        attributes: List[str],
        relation: str,
        template_idx: Optional[int] = None
    ) -> str:
        """
        Generate caption from components.

        Args:
            subject_name: Subject object name
            object_name: Object object name
            attributes: List of attribute names
            relation: Relationship name
            template_idx: Template index (random if None)

        Returns:
            Generated caption
        """
        # Format attributes
        if attributes:
            attr_str = ", ".join(attributes)
        else:
            attr_str = ""

        # Select template
        if template_idx is None:
            template_idx = random.randint(0, len(self.templates) - 1)

        template = self.templates[template_idx]

        # Fill template
        caption = template.format(
            subject=subject_name,
            object=object_name,
            attributes=attr_str,
            relation=relation
        )

        # Clean up
        caption = self._clean_caption(caption)

        return caption

    def _decode_class_prediction(self, value: Any) -> Optional[int]:
        """Decode a single class prediction from logits, tensors, or scalars."""
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim == 0:
                return int(tensor.item())
            if tensor.ndim > 1:
                tensor = tensor[0]
            if tensor.dtype.is_floating_point:
                return int(torch.argmax(tensor).item())
            if tensor.numel() == 1:
                return int(tensor.item())
            return int(tensor.flatten()[0].item())

        if isinstance(value, (list, tuple)):
            if not value:
                return None
            first = value[0]
            if isinstance(first, (list, tuple, torch.Tensor)):
                return self._decode_class_prediction(first)
            try:
                return int(first)
            except (TypeError, ValueError):
                return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _decode_attributes(self, value: Any, top_k: int = 3) -> List[str]:
        """Decode active attribute names from logits or binary predictions."""
        if value is None:
            return []

        indices: List[int] = []

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim > 1:
                tensor = tensor[0]
            if tensor.dtype.is_floating_point:
                active = torch.nonzero(torch.sigmoid(tensor) > 0.5, as_tuple=True)[0]
            else:
                active = torch.nonzero(tensor > 0, as_tuple=True)[0]
            indices = active.tolist()
        elif isinstance(value, (list, tuple)):
            if not value:
                return []
            first = value[0]
            if isinstance(first, (list, tuple, torch.Tensor)):
                return self._decode_attributes(first, top_k=top_k)
            try:
                indices = [int(item) for item in value]
            except (TypeError, ValueError):
                return []
        else:
            return []

        attributes = []
        for idx in indices[:top_k]:
            name = self.idx_to_attribute.get(int(idx), "")
            if name:
                attributes.append(name)
        return attributes

    def generate_from_predictions(
        self,
        task1_results: Dict[str, Any],
        task2_results: Dict[str, Any],
        top_k: int = 3
    ) -> List[str]:
        """
        Generate captions from model predictions.

        Args:
            task1_results: Results from Task 1 (objects + attributes)
            task2_results: Results from Task 2 (relationships)
            top_k: Number of captions to generate

        Returns:
            List of generated captions
        """
        captions = []

        object_source = task1_results.get("object_preds")
        if object_source is None:
            object_source = task1_results.get("object_logits")
        if object_source is None:
            object_source = task1_results.get("object_pred")

        relation_source = task2_results.get("relation_preds")
        if relation_source is None:
            relation_source = task2_results.get("relation_logits")
        if relation_source is None:
            relation_source = task2_results.get("relation_pred")

        attribute_source = task1_results.get("attribute_preds")
        if attribute_source is None:
            attribute_source = task1_results.get("attribute_logits")

        obj_pred = self._decode_class_prediction(object_source)
        rel_pred = self._decode_class_prediction(relation_source)

        subject_name = task2_results.get("subject_name") or (self.idx_to_object.get(obj_pred, "object") if obj_pred is not None else None)
        object_name = task2_results.get("object_name") or (self.idx_to_object.get(obj_pred, "object") if obj_pred is not None else None)
        relation = task2_results.get("predicate")
        if relation is None and rel_pred is not None:
            relation = self.idx_to_relation.get(rel_pred, "relates to")
        if relation is None:
            relation = "relates to"

        if subject_name and object_name:
            attributes = self._decode_attributes(attribute_source)

            # Generate multiple captions
            for i in range(min(top_k, len(self.templates))):
                caption = self.generate_caption(
                    subject_name,
                    object_name,
                    attributes,
                    relation,
                    template_idx=i,
                )
                captions.append(caption)

        return captions if captions else ["A scene with objects and relationships."]

    def _clean_caption(self, caption: str) -> str:
        """Clean up generated caption."""
        # Remove empty attribute placeholders
        caption = caption.replace(" that is ", " ")
        caption = caption.replace(" is  and ", " ")

        # Remove double spaces
        while "  " in caption:
            caption = caption.replace("  ", " ")

        # Capitalize first letter
        if caption:
            caption = caption[0].upper() + caption[1:]

        # Add period if missing
        if caption and not caption.endswith('.'):
            caption += '.'

        return caption.strip()

    def add_template(self, template: str) -> None:
        """Add a new caption template."""
        self.templates.append(template)

    def get_templates(self) -> List[str]:
        """Get all available templates."""
        return self.templates.copy()