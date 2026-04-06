"""
End-to-end wrapper for the three pipeline outputs.

This wrapper does not invent a new joint architecture. Instead, it provides a
clean inference-time container that can load the separately trained object,
attribute, and relation checkpoints and expose them through one API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..attribute import AttributeClassifier
from ..object import ObjectClassifier
from ..relation import RelationClassifier


def _load_state_dict(module: nn.Module, checkpoint_path: str, device: str, strict: bool = True) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=strict)
    return checkpoint


class VisualGenomeE2EModel(nn.Module):
    """A thin inference wrapper over the three task-specific checkpoints."""

    def __init__(self, object_model: ObjectClassifier, attribute_model: AttributeClassifier, relation_model: RelationClassifier):
        super().__init__()
        self.object_model = object_model
        self.attribute_model = attribute_model
        self.relation_model = relation_model

    @classmethod
    def from_models(
        cls,
        object_model: ObjectClassifier,
        attribute_model: AttributeClassifier,
        relation_model: RelationClassifier,
    ) -> "VisualGenomeE2EModel":
        return cls(object_model=object_model, attribute_model=attribute_model, relation_model=relation_model)

    def load_checkpoints(
        self,
        object_checkpoint: str,
        attribute_checkpoint: str,
        relation_checkpoint: str,
        *,
        strict: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Load task-specific checkpoints into the wrapped submodels."""
        self.object_model.eval()
        self.attribute_model.eval()
        self.relation_model.eval()

        loaded = {
            "object": _load_state_dict(self.object_model, object_checkpoint, self.object_model.device, strict=strict),
            "attribute": _load_state_dict(self.attribute_model, attribute_checkpoint, self.attribute_model.device, strict=strict),
            "relation": _load_state_dict(self.relation_model, relation_checkpoint, self.relation_model.device, strict=strict),
        }

        return loaded

    def forward(
        self,
        object_inputs: Optional[torch.Tensor] = None,
        attribute_inputs: Optional[torch.Tensor] = None,
        relation_inputs: Optional[torch.Tensor] = None,
        spatial: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run whichever branches receive inputs and return their logits."""
        outputs: Dict[str, torch.Tensor] = {}

        if object_inputs is not None:
            outputs["object_logits"] = self.object_model(object_inputs)

        if attribute_inputs is not None:
            outputs["attribute_logits"] = self.attribute_model(attribute_inputs)

        if relation_inputs is not None:
            if spatial is None:
                raise ValueError("relation_inputs requires spatial features")
            outputs["relation_logits"] = self.relation_model(relation_inputs, spatial)

        return outputs

    @torch.no_grad()
    def predict_object(self, object_inputs: torch.Tensor) -> torch.Tensor:
        return self.object_model.predict_proba(object_inputs)

    @torch.no_grad()
    def predict_attribute(self, attribute_inputs: torch.Tensor) -> torch.Tensor:
        return self.attribute_model.predict_proba(attribute_inputs)

    @torch.no_grad()
    def predict_relation(self, relation_inputs: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        return self.relation_model.predict_proba(relation_inputs, spatial)

    def save_summary(self) -> Dict[str, Any]:
        """Return a small summary that is useful for notebook logging."""
        return {
            "object_model": self.object_model.summary() if hasattr(self.object_model, "summary") else self.object_model.__class__.__name__,
            "attribute_model": self.attribute_model.summary() if hasattr(self.attribute_model, "summary") else self.attribute_model.__class__.__name__,
            "relation_model": self.relation_model.summary() if hasattr(self.relation_model, "summary") else self.relation_model.__class__.__name__,
        }