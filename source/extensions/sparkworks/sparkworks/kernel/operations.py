"""
3D CAD Operations — wraps build123d/Open Cascade operations.

Each operation takes input geometry (a Face profile and/or existing solid)
and produces a new solid. Operations are designed to be stored in the
parametric timeline and replayed with different parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple

from build123d import (
    Axis,
    BuildPart,
    Compound,
    Part,
    Plane,
    Vector,
    extrude,
    revolve,
    fillet,
    chamfer,
)


class OperationType(Enum):
    EXTRUDE = auto()
    REVOLVE = auto()
    FILLET = auto()
    CHAMFER = auto()
    BOOLEAN_JOIN = auto()
    BOOLEAN_CUT = auto()
    BOOLEAN_INTERSECT = auto()


# ---------------------------------------------------------------------------
# Base operation
# ---------------------------------------------------------------------------

@dataclass
class BaseOperation:
    """Base class for all parametric operations."""
    name: str = ""
    op_type: OperationType = OperationType.EXTRUDE
    suppressed: bool = False

    def execute(self, context: "OperationContext"):
        """Execute this operation. Override in subclasses."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict) -> "BaseOperation":
        """Deserialize from dict."""
        raise NotImplementedError


@dataclass
class OperationContext:
    """
    Mutable context passed through the operation chain.

    Holds the current solid being built and any intermediate state.
    """
    current_solid: Optional[Part] = None
    sketch_face = None  # The Face from the active sketch
    sketch_faces: List = field(default_factory=list)  # Multiple faces for multi-profile extrude
    join_solid: Optional[Part] = None  # Existing solid to fuse into (sketch-on-face)


# ---------------------------------------------------------------------------
# Extrude
# ---------------------------------------------------------------------------

@dataclass
class ExtrudeOperation(BaseOperation):
    """
    Extrude a sketch profile into a 3D solid.

    Parameters:
        distance: Extrusion distance (positive = along normal, negative = opposite)
        symmetric: If True, extrude distance/2 in both directions
        both: If True, also extrude in the negative direction by neg_distance
        neg_distance: Distance for the negative direction (used when both=True)
        join: If True, fuse the extruded solid with ``context.join_solid``
              (used when sketching on a body face).
        body_name: The name of the body this extrusion creates or modifies
              (e.g. "Body1").  Used during rebuild to track multiple bodies.
        join_body_name: When ``join=True``, the body to fuse into.
        profile_index: Which profile (from ``Sketch.build_all_faces()``) to
              extrude when a sketch contains multiple extrudable regions.
              Kept for backward compatibility — prefer ``profile_indices``.
        profile_indices: A list of profile indices to extrude. When multiple
              are given their faces are fused before extrusion.
    """
    distance: float = 10.0
    symmetric: bool = False
    both: bool = False
    neg_distance: float = 0.0
    join: bool = False
    body_name: str = ""
    join_body_name: str = ""
    profile_index: int = 0
    profile_indices: List[int] = field(default_factory=list)
    op_type: OperationType = field(default=OperationType.EXTRUDE, init=False)

    def _extrude_single(self, face):
        """Extrude a single face and return the resulting solid."""
        if self.symmetric:
            half = self.distance / 2.0
            with BuildPart() as part:
                extrude(face, amount=half, both=True)
            return part.part
        else:
            with BuildPart() as part:
                extrude(face, amount=self.distance)
            return part.part

    def execute(self, context: OperationContext):
        if self.suppressed:
            return

        # Collect the faces to extrude — prefer the multi-face list
        faces = context.sketch_faces if context.sketch_faces else []
        if not faces and context.sketch_face is not None:
            faces = [context.sketch_face]
        if not faces:
            return

        # Extrude each face separately, then fuse the results
        new_solid = None
        for face in faces:
            try:
                solid = self._extrude_single(face)
                if solid is None:
                    continue
                if new_solid is None:
                    new_solid = solid
                else:
                    new_solid = new_solid.fuse(solid).clean()
            except Exception as e:
                print(f"[SparkWorks] Extrude face failed: {e}")

        if new_solid is None:
            return

        # If join mode is active and there's an existing solid, fuse them
        if self.join and context.join_solid is not None:
            try:
                context.current_solid = context.join_solid.fuse(new_solid).clean()
                print("[SparkWorks] Fused new extrusion with parent body")
            except Exception as e:
                print(f"[SparkWorks] Boolean fuse failed, keeping new solid: {e}")
                context.current_solid = new_solid
        else:
            context.current_solid = new_solid

    def to_dict(self) -> dict:
        d = {
            "type": "extrude",
            "name": self.name,
            "distance": self.distance,
            "symmetric": self.symmetric,
            "both": self.both,
            "neg_distance": self.neg_distance,
            "join": self.join,
            "body_name": self.body_name,
            "join_body_name": self.join_body_name,
            "profile_index": self.profile_index,
            "suppressed": self.suppressed,
        }
        if self.profile_indices:
            d["profile_indices"] = list(self.profile_indices)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExtrudeOperation":
        op = cls(
            distance=d.get("distance", 10.0),
            symmetric=d.get("symmetric", False),
            both=d.get("both", False),
            neg_distance=d.get("neg_distance", 0.0),
            join=d.get("join", False),
            body_name=d.get("body_name", ""),
            join_body_name=d.get("join_body_name", ""),
            profile_index=d.get("profile_index", 0),
            profile_indices=d.get("profile_indices", []),
        )
        op.name = d.get("name", "Extrude")
        op.suppressed = d.get("suppressed", False)
        return op


# ---------------------------------------------------------------------------
# Revolve
# ---------------------------------------------------------------------------

@dataclass
class RevolveOperation(BaseOperation):
    """
    Revolve a sketch profile around an axis.

    Parameters:
        angle: Revolution angle in degrees (360 = full revolution)
        axis: Which axis to revolve around ("X", "Y", or "Z")
    """
    angle: float = 360.0
    axis_name: str = "Z"
    op_type: OperationType = field(default=OperationType.REVOLVE, init=False)

    @property
    def axis(self) -> Axis:
        mapping = {"X": Axis.X, "Y": Axis.Y, "Z": Axis.Z}
        return mapping.get(self.axis_name.upper(), Axis.Z)

    def execute(self, context: OperationContext):
        if self.suppressed or context.sketch_face is None:
            return

        with BuildPart() as part:
            revolve(context.sketch_face, axis=self.axis, revolution_arc=self.angle)
        context.current_solid = part.part

    def to_dict(self) -> dict:
        return {
            "type": "revolve",
            "name": self.name,
            "angle": self.angle,
            "axis": self.axis_name,
            "suppressed": self.suppressed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RevolveOperation":
        op = cls(angle=d.get("angle", 360.0), axis_name=d.get("axis", "Z"))
        op.name = d.get("name", "Revolve")
        op.suppressed = d.get("suppressed", False)
        return op


# ---------------------------------------------------------------------------
# Fillet
# ---------------------------------------------------------------------------

@dataclass
class FilletOperation(BaseOperation):
    """
    Apply a fillet (rounding) to edges of the current solid.

    Parameters:
        radius: Fillet radius
        edge_indices: Which edges to fillet (None = all edges)
    """
    radius: float = 1.0
    edge_indices: Optional[List[int]] = None
    op_type: OperationType = field(default=OperationType.FILLET, init=False)

    def execute(self, context: OperationContext):
        if self.suppressed or context.current_solid is None:
            return

        solid = context.current_solid
        edges = solid.edges()

        if self.edge_indices is not None:
            selected_edges = [edges[i] for i in self.edge_indices if i < len(edges)]
        else:
            selected_edges = edges

        if selected_edges:
            context.current_solid = fillet(selected_edges, radius=self.radius)

    def to_dict(self) -> dict:
        return {
            "type": "fillet",
            "name": self.name,
            "radius": self.radius,
            "edge_indices": self.edge_indices,
            "suppressed": self.suppressed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FilletOperation":
        op = cls(
            radius=d.get("radius", 1.0),
            edge_indices=d.get("edge_indices"),
        )
        op.name = d.get("name", "Fillet")
        op.suppressed = d.get("suppressed", False)
        return op


# ---------------------------------------------------------------------------
# Chamfer
# ---------------------------------------------------------------------------

@dataclass
class ChamferOperation(BaseOperation):
    """
    Apply a chamfer (bevel) to edges of the current solid.

    Parameters:
        length: Chamfer length
        edge_indices: Which edges to chamfer (None = all edges)
    """
    length: float = 1.0
    edge_indices: Optional[List[int]] = None
    op_type: OperationType = field(default=OperationType.CHAMFER, init=False)

    def execute(self, context: OperationContext):
        if self.suppressed or context.current_solid is None:
            return

        solid = context.current_solid
        edges = solid.edges()

        if self.edge_indices is not None:
            selected_edges = [edges[i] for i in self.edge_indices if i < len(edges)]
        else:
            selected_edges = edges

        if selected_edges:
            context.current_solid = chamfer(selected_edges, length=self.length)

    def to_dict(self) -> dict:
        return {
            "type": "chamfer",
            "name": self.name,
            "length": self.length,
            "edge_indices": self.edge_indices,
            "suppressed": self.suppressed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChamferOperation":
        op = cls(
            length=d.get("length", 1.0),
            edge_indices=d.get("edge_indices"),
        )
        op.name = d.get("name", "Chamfer")
        op.suppressed = d.get("suppressed", False)
        return op


# ---------------------------------------------------------------------------
# Boolean Operations
# ---------------------------------------------------------------------------

@dataclass
class BooleanOperation(BaseOperation):
    """
    Boolean operation between current solid and a tool solid.

    The tool_solid should be set before execution (typically from another
    feature's output stored in the timeline).

    Parameters:
        mode: "join", "cut", or "intersect"
    """
    mode: str = "join"
    tool_feature_index: Optional[int] = None  # Index into timeline of the tool body
    _tool_solid: Optional[Part] = field(default=None, repr=False)

    def __post_init__(self):
        mode_map = {
            "join": OperationType.BOOLEAN_JOIN,
            "cut": OperationType.BOOLEAN_CUT,
            "intersect": OperationType.BOOLEAN_INTERSECT,
        }
        self.op_type = mode_map.get(self.mode, OperationType.BOOLEAN_JOIN)

    def execute(self, context: OperationContext):
        if self.suppressed or context.current_solid is None or self._tool_solid is None:
            return

        solid = context.current_solid
        tool = self._tool_solid

        if self.mode == "join":
            context.current_solid = solid.fuse(tool)
        elif self.mode == "cut":
            context.current_solid = solid.cut(tool)
        elif self.mode == "intersect":
            context.current_solid = solid.intersect(tool)

    def to_dict(self) -> dict:
        return {
            "type": "boolean",
            "name": self.name,
            "mode": self.mode,
            "tool_feature_index": self.tool_feature_index,
            "suppressed": self.suppressed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BooleanOperation":
        op = cls(
            mode=d.get("mode", "join"),
            tool_feature_index=d.get("tool_feature_index"),
        )
        op.name = d.get("name", "Boolean")
        op.suppressed = d.get("suppressed", False)
        return op


# ---------------------------------------------------------------------------
# Deserialization registry
# ---------------------------------------------------------------------------

_OPERATION_DESERIALIZERS = {
    "extrude": ExtrudeOperation.from_dict,
    "revolve": RevolveOperation.from_dict,
    "fillet": FilletOperation.from_dict,
    "chamfer": ChamferOperation.from_dict,
    "boolean": BooleanOperation.from_dict,
}


def operation_from_dict(d: dict) -> BaseOperation:
    """Deserialize any operation from a dict."""
    op_type = d.get("type", "extrude")
    deserializer = _OPERATION_DESERIALIZERS.get(op_type)
    if deserializer is None:
        raise ValueError(f"Unknown operation type: {op_type}")
    return deserializer(d)
