"""
Parametric Timeline — ordered feature history with rollback, edit, and rebuild.

The Timeline holds an ordered list of Features. Each Feature wraps either a
Sketch or a 3D Operation. When any feature's parameters change, the timeline
rebuilds from that point forward, recomputing all downstream geometry.

This is the core data structure that makes the modeler "parametric."
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any

from ..kernel.sketch import Sketch, primitive_from_dict
from ..kernel.operations import (
    BaseOperation,
    ExtrudeOperation,
    OperationContext,
    operation_from_dict,
)


class FeatureType(Enum):
    SKETCH = auto()
    OPERATION = auto()


@dataclass
class Feature:
    """
    A single entry in the parametric timeline.

    A feature is either a Sketch or an Operation (extrude, fillet, etc.).
    Features form an ordered chain — each operation may depend on the
    sketch/solid from previous features.
    """
    name: str
    feature_type: FeatureType
    sketch: Optional[Sketch] = None
    operation: Optional[BaseOperation] = None
    suppressed: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def display_name(self) -> str:
        if self.feature_type == FeatureType.SKETCH:
            return f"Sketch: {self.name}"
        elif self.operation:
            return f"{self.operation.op_type.name.title()}: {self.name}"
        return self.name

    @property
    def is_sketch(self) -> bool:
        return self.feature_type == FeatureType.SKETCH

    @property
    def is_operation(self) -> bool:
        return self.feature_type == FeatureType.OPERATION

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "feature_type": self.feature_type.name,
            "suppressed": self.suppressed,
            "created_at": self.created_at,
        }
        if self.sketch:
            d["sketch"] = self.sketch.to_dict()
        if self.operation:
            d["operation"] = self.operation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Feature":
        feat_type = FeatureType[d["feature_type"]]
        feature = cls(
            name=d["name"],
            feature_type=feat_type,
            suppressed=d.get("suppressed", False),
            created_at=d.get("created_at", time.time()),
        )
        if "sketch" in d:
            feature.sketch = Sketch.from_dict(d["sketch"])
        if "operation" in d:
            feature.operation = operation_from_dict(d["operation"])
        return feature


class Timeline:
    """
    The parametric timeline — an ordered list of features with rebuild logic.

    The timeline manages:
    - Adding/removing features
    - Reordering features (with dependency checking)
    - Rebuilding from a given point when parameters change
    - Scrubbing (viewing the model at any historical point)
    - Suppressing/unsuppressing features
    - Serialization to/from dicts (for USD attribute storage)

    Callbacks:
        on_rebuild: Called after a rebuild with the resulting solid.
        on_feature_changed: Called when any feature is added/removed/modified.
    """

    def __init__(self):
        self._features: List[Feature] = []
        self._scrub_index: Optional[int] = None  # None = show latest
        self._current_solid = None
        self._current_sketch_face = None
        self._current_sketch_all_faces: list = []

        # Callbacks
        self.on_rebuild: Optional[Callable] = None
        self.on_feature_changed: Optional[Callable] = None

    # -- Properties ----------------------------------------------------------

    @property
    def features(self) -> List[Feature]:
        return list(self._features)

    @property
    def feature_count(self) -> int:
        return len(self._features)

    @property
    def current_solid(self):
        """The solid at the current scrub position (or latest)."""
        return self._current_solid

    @property
    def scrub_index(self) -> int:
        """
        Current scrub position (0-based).

        Returns:
            -1 means "before all features" (nothing active).
            None / not set defaults to the last feature index (latest).
        """
        if self._scrub_index is None:
            return len(self._features) - 1
        return self._scrub_index

    @scrub_index.setter
    def scrub_index(self, value: Optional[int]):
        if value is not None:
            value = max(-1, min(value, len(self._features) - 1))
        self._scrub_index = value

    # -- Feature management --------------------------------------------------

    def add_sketch(self, sketch: Sketch, name: Optional[str] = None,
                   insert_after: Optional[int] = None) -> Feature:
        """
        Add a sketch feature to the timeline.

        Args:
            sketch: The Sketch object.
            name: Display name override.
            insert_after: If given, insert after this feature index
                          (-1 = before all features).  None = append at end.
        """
        feature = Feature(
            name=name or sketch.name,
            feature_type=FeatureType.SKETCH,
            sketch=sketch,
        )
        idx = self._insert_feature(feature, insert_after)
        self._scrub_index = None
        self._notify_changed()
        self.rebuild_from(idx)
        return feature

    def add_operation(
        self, operation: BaseOperation, name: Optional[str] = None,
        insert_after: Optional[int] = None,
    ) -> Feature:
        """
        Add an operation feature to the timeline.

        Args:
            operation: The operation object.
            name: Display name override.
            insert_after: If given, insert after this feature index
                          (-1 = before all features).  None = append at end.
        """
        feature = Feature(
            name=name or operation.name or operation.op_type.name.title(),
            feature_type=FeatureType.OPERATION,
            operation=operation,
        )
        idx = self._insert_feature(feature, insert_after)
        self._scrub_index = None
        self._notify_changed()
        self.rebuild_from(idx)
        return feature

    def _insert_feature(self, feature: Feature, insert_after: Optional[int]) -> int:
        """
        Insert a feature at the right position.

        Returns the index where the feature was inserted.
        """
        if insert_after is None or insert_after >= len(self._features) - 1:
            # Append at end
            self._features.append(feature)
            return len(self._features) - 1
        else:
            # Insert after the given index (insert_after=-1 means before all)
            pos = insert_after + 1
            pos = max(0, pos)
            self._features.insert(pos, feature)
            return pos

    def remove_feature(self, index: int):
        """Remove a feature and rebuild downstream."""
        if 0 <= index < len(self._features):
            self._features.pop(index)
            rebuild_from = max(0, index - 1) if self._features else 0
            self._notify_changed()
            if self._features:
                self.rebuild_from(rebuild_from)
            else:
                self._current_solid = None
                self._current_sketch_face = None
                self._notify_rebuild()

    def move_feature(self, from_index: int, to_index: int):
        """Move a feature to a new position and rebuild."""
        if from_index == to_index:
            return
        if not (0 <= from_index < len(self._features)):
            return
        if not (0 <= to_index < len(self._features)):
            return

        feature = self._features.pop(from_index)
        self._features.insert(to_index, feature)
        rebuild_from = min(from_index, to_index)
        self._notify_changed()
        self.rebuild_from(rebuild_from)

    def suppress_feature(self, index: int, suppressed: bool = True):
        """Suppress or unsuppress a feature and rebuild."""
        if 0 <= index < len(self._features):
            self._features[index].suppressed = suppressed
            if self._features[index].operation:
                self._features[index].operation.suppressed = suppressed
            self._notify_changed()
            self.rebuild_from(index)

    def update_feature_params(self, index: int, params: Dict[str, Any]):
        """
        Update parameters on a feature and rebuild from that point.

        For sketch features, params can include primitive modifications.
        For operations, params are set directly on the operation dataclass.
        """
        if not (0 <= index < len(self._features)):
            return

        feature = self._features[index]

        if feature.is_operation and feature.operation:
            for key, value in params.items():
                if hasattr(feature.operation, key):
                    setattr(feature.operation, key, value)
        elif feature.is_sketch and feature.sketch:
            # For sketches, params might update primitive properties
            for key, value in params.items():
                if hasattr(feature.sketch, key):
                    setattr(feature.sketch, key, value)

        self._notify_changed()
        self.rebuild_from(index)

    # -- Rebuild engine ------------------------------------------------------

    def rebuild_from(self, start_index: int = 0):
        """
        Rebuild the model from start_index forward.

        This is the core parametric rebuild: we replay all features from
        start_index to the current scrub position, accumulating geometry.
        """
        # Determine end point (scrub position or end of list)
        end_index = self.scrub_index + 1 if self._scrub_index is not None else len(self._features)
        end_index = min(end_index, len(self._features))

        # Start fresh if rebuilding from the beginning
        if start_index == 0:
            self._current_solid = None
            self._current_sketch_face = None
            self._current_sketch_all_faces = []
        else:
            # We'd need cached state at start_index - not implemented yet in Phase 1
            # For now, always rebuild from scratch
            self._current_solid = None
            self._current_sketch_face = None
            self._current_sketch_all_faces = []
            start_index = 0

        context = OperationContext()

        for i in range(start_index, end_index):
            feature = self._features[i]

            if feature.suppressed:
                continue

            if feature.is_sketch and feature.sketch:
                # Build the sketch face — this becomes the profile for operations.
                # Also cache all faces so operations can select a specific profile.
                face = feature.sketch.build_face()
                self._current_sketch_face = face
                self._current_sketch_all_faces = feature.sketch.build_all_faces()
                context.sketch_face = face

            elif feature.is_operation and feature.operation:
                op = feature.operation
                n_faces = len(self._current_sketch_all_faces) if self._current_sketch_all_faces else 0

                # Determine which profile(s) to extrude
                pis = getattr(op, "profile_indices", [])
                if not pis:
                    # Backward compat: fall back to single profile_index
                    pis = [getattr(op, "profile_index", 0)]

                if self._current_sketch_all_faces:
                    selected = [
                        self._current_sketch_all_faces[pi]
                        for pi in pis
                        if 0 <= pi < n_faces
                    ]
                    if selected:
                        # Pass faces as a list — the operation will extrude
                        # each one separately and fuse the resulting solids
                        context.sketch_faces = selected
                        context.sketch_face = selected[0]
                        print(f"[SparkWorks] Using {len(selected)} profile(s) {pis} for '{feature.name}'")
                    else:
                        context.sketch_faces = []
                        context.sketch_face = self._current_sketch_face
                        print(f"[SparkWorks] Fallback sketch face for '{feature.name}' (pis={pis}, n={n_faces})")
                else:
                    context.sketch_faces = []
                    context.sketch_face = self._current_sketch_face
                    print(f"[SparkWorks] Using default sketch face for '{feature.name}' (no all_faces)")

                context.current_solid = self._current_solid
                # Pass the existing solid as join target for face-on-body extrudes
                context.join_solid = self._current_solid
                op.execute(context)
                self._current_solid = context.current_solid

        self._notify_rebuild()

    def rebuild_all(self):
        """Full rebuild from the beginning."""
        self.rebuild_from(0)

    # -- Scrubbing -----------------------------------------------------------

    def scrub_to(self, index: int):
        """
        Scrub the timeline to show the model at a specific feature.

        This rebuilds up to (and including) the given index.
        """
        self._scrub_index = index
        self.rebuild_all()

    def scrub_to_end(self):
        """Show the latest state of the model."""
        self._scrub_index = None
        self.rebuild_all()

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "features": [f.to_dict() for f in self._features],
            "scrub_index": self._scrub_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        timeline = cls()
        for feat_dict in d.get("features", []):
            feature = Feature.from_dict(feat_dict)
            timeline._features.append(feature)
        timeline._scrub_index = d.get("scrub_index")
        return timeline

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Timeline":
        return cls.from_dict(json.loads(json_str))

    # -- Notifications -------------------------------------------------------

    def _notify_rebuild(self):
        if self.on_rebuild:
            self.on_rebuild(self._current_solid)

    def _notify_changed(self):
        if self.on_feature_changed:
            self.on_feature_changed(self._features)
