"""
Parametric Timeline — ordered feature history with rollback, edit, and rebuild.

The Timeline holds an ordered list of Features. Each Feature wraps either a
*reference* to a Sketch (by ID) or a 3D Operation. Sketch objects live in a
separate **SketchRegistry** (like Fusion 360's Sketches folder).

When any feature's parameters change, the timeline rebuilds from that point
forward, recomputing all downstream geometry.

This is the core data structure that makes the modeler "parametric."
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any, TYPE_CHECKING

from ..kernel.sketch import Sketch, primitive_from_dict
from ..kernel.construction_plane import get_planar_face
from ..kernel.operations import (
    BaseOperation,
    BooleanOperation,
    ChamferOperation,
    ExtrudeOperation,
    FilletOperation,
    OperationContext,
    operation_from_dict,
)

if TYPE_CHECKING:
    from .sketch_registry import SketchRegistry


class FeatureType(Enum):
    SKETCH = auto()
    OPERATION = auto()


@dataclass
class Feature:
    """
    A single entry in the parametric timeline.

    A feature is either:
    - **SKETCH** — holds a *sketch_id* that references a Sketch in the
      SketchRegistry (the Sketch data itself lives in the registry, not
      here — just like Fusion 360's Sketches folder).
    - **OPERATION** — holds an operation object (extrude, fillet, …).

    Features form an ordered chain — each operation may depend on the
    sketch/solid from previous features.
    """
    name: str
    feature_type: FeatureType
    sketch_id: Optional[str] = None  # references SketchRegistry
    operation: Optional[BaseOperation] = None
    suppressed: bool = False
    created_at: float = field(default_factory=time.time)

    # ── Convenience property ------------------------------------------------
    # The ``sketch`` property lets existing call-sites that read
    # ``feature.sketch`` keep working transparently.  It requires a
    # SketchRegistry to be attached via ``bind_registry()``.
    _registry: Optional["SketchRegistry"] = field(
        default=None, repr=False, compare=False,
    )

    def bind_registry(self, registry: "SketchRegistry"):
        object.__setattr__(self, "_registry", registry)

    @property
    def sketch(self) -> Optional[Sketch]:
        """Look up the Sketch from the registry.  Returns None if unbound."""
        if self.sketch_id and self._registry:
            return self._registry.get(self.sketch_id)
        return None

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
        if self.sketch_id:
            d["sketch_id"] = self.sketch_id
        if self.operation:
            d["operation"] = self.operation.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Feature":
        feat_type = FeatureType[d["feature_type"]]
        feature = cls(
            name=d["name"],
            feature_type=feat_type,
            sketch_id=d.get("sketch_id"),
            suppressed=d.get("suppressed", False),
            created_at=d.get("created_at", time.time()),
        )
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
        on_rebuild: Called after a rebuild with the bodies dict.
        on_feature_changed: Called when any feature is added/removed/modified.
    """

    def __init__(self, sketch_registry: Optional["SketchRegistry"] = None):
        self._features: List[Feature] = []
        self._scrub_index: Optional[int] = None  # None = show latest
        self._current_sketch_face = None
        self._current_sketch_all_faces: list = []

        # Dict of body_name -> solid for multi-body tracking.
        # This is the single source of truth for all body geometry.
        # Each independent extrude creates a new entry; join extrudes
        # update an existing entry.
        self._bodies: Dict[str, Any] = {}

        # ── Snapshot cache for fast scrubbing ───────────────────────
        # Maps feature index → frozen state *after* that feature was
        # replayed.  OCC/build123d shapes are immutable so shallow
        # copies of the dicts/lists are safe.
        self._snapshots: Dict[int, Dict[str, Any]] = {}

        # The SketchRegistry that owns all Sketch objects.
        # Features store only sketch_id; the registry resolves them.
        self._sketch_registry: Optional["SketchRegistry"] = sketch_registry

        # Callbacks
        self.on_rebuild: Optional[Callable] = None
        self.on_feature_changed: Optional[Callable] = None

    @property
    def sketch_registry(self) -> Optional["SketchRegistry"]:
        return self._sketch_registry

    @sketch_registry.setter
    def sketch_registry(self, registry: "SketchRegistry"):
        self._sketch_registry = registry
        # Bind existing features to the new registry
        for f in self._features:
            f.bind_registry(registry)

    # -- Properties ----------------------------------------------------------

    @property
    def features(self) -> List[Feature]:
        return list(self._features)

    @property
    def feature_count(self) -> int:
        return len(self._features)

    @property
    def bodies(self) -> Dict[str, Any]:
        """Dict of body_name -> solid for all bodies at the current rebuild state."""
        return dict(self._bodies)

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
                   insert_after: Optional[int] = None,
                   sketch_id: Optional[str] = None) -> Feature:
        """
        Add a sketch feature to the timeline.

        The Sketch object is registered in the SketchRegistry; the Feature
        stores only a ``sketch_id`` reference — just like Fusion 360.

        Args:
            sketch: The Sketch object.
            name: Display name override.
            insert_after: If given, insert after this feature index
                          (-1 = before all features).  None = append at end.
            sketch_id: Explicit registry ID.  Auto-generated if None.
        """
        # Register the sketch in the central registry
        if self._sketch_registry is not None:
            sid = self._sketch_registry.register(sketch, sketch_id)
        else:
            sid = sketch_id or sketch.name

        feature = Feature(
            name=name or sid,
            feature_type=FeatureType.SKETCH,
            sketch_id=sid,
        )
        feature.bind_registry(self._sketch_registry)
        idx = self._insert_feature(feature, insert_after)
        # Place the marker right after the newly inserted feature
        self._scrub_index = idx
        self._invalidate_snapshots(idx)
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
        # Place the marker right after the newly inserted feature
        self._scrub_index = idx
        self._invalidate_snapshots(idx)
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
            removed = self._features.pop(index)
            # If we just removed a sketch feature, also remove it from the
            # registry — unless another feature still references it.
            if removed.sketch_id and self._sketch_registry:
                still_used = any(
                    f.sketch_id == removed.sketch_id
                    for f in self._features
                )
                if not still_used:
                    self._sketch_registry.remove(removed.sketch_id)
            rebuild_from = max(0, index - 1) if self._features else 0
            self._invalidate_snapshots(rebuild_from)
            self._notify_changed()
            if self._features:
                self.rebuild_from(rebuild_from)
            else:
                self._current_sketch_face = None
                self._bodies = {}
                self._snapshots.clear()
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
        self._invalidate_snapshots(rebuild_from)
        self._notify_changed()
        self.rebuild_from(rebuild_from)

    def suppress_feature(self, index: int, suppressed: bool = True):
        """Suppress or unsuppress a feature and rebuild."""
        if 0 <= index < len(self._features):
            self._features[index].suppressed = suppressed
            if self._features[index].operation:
                self._features[index].operation.suppressed = suppressed
            self._invalidate_snapshots(index)
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

        self._invalidate_snapshots(index)
        self._notify_changed()
        self.rebuild_from(index)

    # -- Rebuild engine ------------------------------------------------------

    def rebuild_from(self, start_index: int = 0):
        """
        Rebuild the model from *start_index* forward.

        Uses a **snapshot cache** to avoid replaying the entire history
        every time.  The cache stores the state (bodies, current solid,
        sketch faces) after each feature, keyed by feature index.

        * Moving the marker **backward** is O(1) — just restore the
          snapshot at the target position.
        * Moving **forward** replays only the delta from the nearest
          upstream snapshot.
        * Editing a feature invalidates all snapshots at and after that
          index, so the next rebuild replays from there.
        """
        # Determine end point (scrub position or end of list)
        end_index = self.scrub_index + 1 if self._scrub_index is not None else len(self._features)
        end_index = min(end_index, len(self._features))

        # ── Try to restore from a snapshot ──────────────────────────
        # Find the best cached snapshot strictly *before* end_index
        # that is at or after start_index.
        restored = False
        if self._snapshots:
            # Check if we have an exact snapshot for end_index - 1
            # (the last feature we need).  If so, just restore it
            # without replaying anything.
            target = end_index - 1
            if target in self._snapshots:
                self._restore_snapshot(target)
                self._notify_rebuild()
                return

            # Otherwise find the nearest snapshot before end_index
            # that lets us skip some replay.
            best = -1
            for idx in self._snapshots:
                if start_index <= idx < end_index and idx > best:
                    best = idx
            if best >= 0:
                self._restore_snapshot(best)
                start_index = best + 1
                restored = True

        # ── Fresh start (no usable snapshot) ────────────────────────
        if not restored:
            self._current_sketch_face = None
            self._current_sketch_all_faces = []
            self._bodies = {}
            start_index = 0

        # ── Replay features from start_index to end_index ───────────
        context = OperationContext()

        for i in range(start_index, end_index):
            feature = self._features[i]

            if feature.suppressed:
                # Still save a snapshot so we can skip past suppressed
                # features on future scrubs.
                self._save_snapshot(i)
                continue

            if feature.is_sketch and feature.sketch_id:
                sketch = feature.sketch  # property resolves from registry
                if sketch is None:
                    print(f"[SparkWorks] WARNING: sketch '{feature.sketch_id}' not found in registry")
                    self._save_snapshot(i)
                    continue
                face = sketch.build_face()
                self._current_sketch_face = face
                self._current_sketch_all_faces = sketch.build_all_faces()
                context.sketch_face = face

            elif feature.is_operation and feature.operation:
                op = feature.operation

                # ── Boolean body merge (union / cut / intersect) ─────
                if isinstance(op, BooleanOperation):
                    target_solid = self._bodies.get(op.target_body)
                    tool_solid = self._bodies.get(op.tool_body)

                    if target_solid is None:
                        print(f"[SparkWorks] Boolean: target body '{op.target_body}' not found")
                    elif tool_solid is None:
                        print(f"[SparkWorks] Boolean: tool body '{op.tool_body}' not found")
                    else:
                        context.current_solid = target_solid
                        context.join_solid = tool_solid
                        op.execute(context)

                        if context.current_solid is not None:
                            self._bodies[op.target_body] = context.current_solid
                            # Remove tool body (consumed) unless keep_tool
                            if not op.keep_tool and op.tool_body in self._bodies:
                                del self._bodies[op.tool_body]

                # ── Fillet / Chamfer (modify a named body in-place) ───
                elif isinstance(op, (FilletOperation, ChamferOperation)):
                    body_name = op.body_name
                    target = self._bodies.get(body_name) if body_name else None
                    if target is None:
                        print(f"[SparkWorks] {op.op_type.name}: body '{body_name}' not found")
                    else:
                        context.current_solid = target
                        op.execute(context)
                        if context.current_solid is not None:
                            self._bodies[body_name] = context.current_solid

                # ── Extrude / Revolve ─────────────────────────────────
                else:
                    # ── Face-based extrude (Press/Pull) ──────────────────
                    face_body = getattr(op, "face_body_name", "")
                    face_idx = getattr(op, "face_index", -1)

                    if face_body and face_idx >= 0:
                        src_solid = self._bodies.get(face_body)
                        face = get_planar_face(src_solid, face_idx) if src_solid else None
                        if face is not None:
                            context.sketch_face = face
                            context.sketch_faces = [face]
                        else:
                            print(f"[SparkWorks] WARNING: could not resolve face {face_idx} on '{face_body}'")
                            context.sketch_face = None
                            context.sketch_faces = []
                    else:
                        # ── Sketch-profile-based extrude ─────────────────
                        n_faces = len(self._current_sketch_all_faces) if self._current_sketch_all_faces else 0

                        pis = getattr(op, "profile_indices", [])
                        if not pis:
                            pis = [getattr(op, "profile_index", 0)]

                        if self._current_sketch_all_faces:
                            selected = [
                                self._current_sketch_all_faces[pi]
                                for pi in pis
                                if 0 <= pi < n_faces
                            ]
                            if selected:
                                context.sketch_faces = selected
                                context.sketch_face = selected[0]
                            else:
                                context.sketch_faces = []
                                context.sketch_face = self._current_sketch_face
                        else:
                            context.sketch_faces = []
                            context.sketch_face = self._current_sketch_face

                    # Resolve join target from the bodies dict
                    body_name = getattr(op, "body_name", "")
                    join_body = getattr(op, "join_body_name", "")
                    is_join = getattr(op, "join", False) and join_body

                    if is_join:
                        context.join_solid = self._bodies.get(join_body)
                        context.current_solid = self._bodies.get(join_body)
                    else:
                        context.join_solid = None
                        context.current_solid = None

                    op.execute(context)

                    if context.current_solid is not None and body_name:
                        self._bodies[body_name] = context.current_solid

            # Cache this position for future scrubs
            self._save_snapshot(i)

        self._notify_rebuild()

    def rebuild_all(self):
        """Full rebuild from the beginning."""
        self.rebuild_from(0)

    # -- Snapshot cache ------------------------------------------------------

    def _save_snapshot(self, index: int):
        """Store the current state keyed by feature *index*."""
        self._snapshots[index] = {
            "sketch_face": self._current_sketch_face,
            "sketch_all_faces": list(self._current_sketch_all_faces),
            "bodies": dict(self._bodies),
        }

    def _restore_snapshot(self, index: int):
        """Restore state from a cached snapshot at *index*."""
        snap = self._snapshots[index]
        self._current_sketch_face = snap["sketch_face"]
        self._current_sketch_all_faces = list(snap["sketch_all_faces"])
        self._bodies = dict(snap["bodies"])

    def _invalidate_snapshots(self, from_index: int = 0):
        """
        Remove all cached snapshots at or after *from_index*.

        Called whenever a feature is added, removed, moved, suppressed,
        or edited — anything that makes downstream cached state stale.
        """
        self._snapshots = {
            k: v for k, v in self._snapshots.items() if k < from_index
        }

    def clear_snapshots(self):
        """Remove all cached snapshots."""
        self._snapshots.clear()

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
    def from_dict(cls, d: dict,
                  sketch_registry: Optional["SketchRegistry"] = None) -> "Timeline":
        timeline = cls(sketch_registry=sketch_registry)
        for feat_dict in d.get("features", []):
            feature = Feature.from_dict(feat_dict)
            feature.bind_registry(sketch_registry)
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
            self.on_rebuild(self._bodies)

    def _notify_changed(self):
        if self.on_feature_changed:
            self.on_feature_changed(self._features)
