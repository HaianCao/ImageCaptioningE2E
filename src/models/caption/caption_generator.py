"""
Caption generator for combining Task 1 + Task 2 results.

Uses template-based generation for simplicity and interpretability.
"""

import random
from typing import List, Dict, Any, Optional
from pathlib import Path


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

        # Extract top predictions
        if 'object_preds' in task1_results and 'relation_preds' in task2_results:
            # Get top object predictions
            obj_logits = task1_results.get('object_logits', [])
            attr_logits = task1_results.get('attribute_logits', [])

            # Get top relation predictions
            rel_logits = task2_results.get('relation_logits', [])

            # For simplicity, take top-1 for each
            if obj_logits and rel_logits:
                obj_pred = obj_logits.argmax(dim=1).item()
                rel_pred = rel_logits.argmax(dim=1).item()

                subject_name = self.idx_to_object.get(obj_pred, "object")
                object_name = self.idx_to_object.get(obj_pred, "object")  # Same for demo
                relation = self.idx_to_relation.get(rel_pred, "relates to")

                # Get attributes above threshold
                if attr_logits:
                    attr_probs = (attr_logits.sigmoid() > 0.5).nonzero(as_tuple=True)[1]
                    attributes = [self.idx_to_attribute.get(idx.item(), "")
                                for idx in attr_probs[:3]]  # Top 3 attributes
                else:
                    attributes = []

                # Generate multiple captions
                for i in range(min(top_k, len(self.templates))):
                    caption = self.generate_caption(
                        subject_name, object_name, attributes, relation, template_idx=i
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