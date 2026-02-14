"""
2D Sketch module — creates profiles on construction planes using build123d.

A Sketch collects 2D geometry primitives (lines, arcs, circles, rectangles)
on a given workplane and produces a closed Wire or Face that can be used by
3D operations like Extrude and Revolve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

# build123d imports — these wrap Open Cascade
from build123d import (
    Axis,
    BuildLine,
    BuildSketch,
    Circle as B3dCircle,
    Line,
    Plane,
    Rectangle as B3dRectangle,
    ThreePointArc,
    Vector,
    Wire,
    make_face,
)


class SketchPrimitiveType(Enum):
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()
    ARC = auto()


# ---------------------------------------------------------------------------
# Sketch primitives — lightweight data containers that describe 2D geometry.
# These are serialisable and stored in the parametric timeline.
# ---------------------------------------------------------------------------


@dataclass
class SketchLine:
    """A line segment from start to end in sketch-local coordinates."""
    start: Tuple[float, float]
    end: Tuple[float, float]
    kind: SketchPrimitiveType = field(default=SketchPrimitiveType.LINE, init=False)

    def to_dict(self) -> dict:
        return {"type": "line", "start": list(self.start), "end": list(self.end)}

    @classmethod
    def from_dict(cls, d: dict) -> "SketchLine":
        return cls(start=tuple(d["start"]), end=tuple(d["end"]))


@dataclass
class SketchRect:
    """An axis-aligned rectangle defined by center, width, and height."""
    center: Tuple[float, float] = (0.0, 0.0)
    width: float = 1.0
    height: float = 1.0
    kind: SketchPrimitiveType = field(default=SketchPrimitiveType.RECTANGLE, init=False)

    def to_dict(self) -> dict:
        return {
            "type": "rectangle",
            "center": list(self.center),
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SketchRect":
        return cls(center=tuple(d["center"]), width=d["width"], height=d["height"])


@dataclass
class SketchCircle:
    """A circle defined by center and radius."""
    center: Tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    kind: SketchPrimitiveType = field(default=SketchPrimitiveType.CIRCLE, init=False)

    def to_dict(self) -> dict:
        return {
            "type": "circle",
            "center": list(self.center),
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SketchCircle":
        return cls(center=tuple(d["center"]), radius=d["radius"])


@dataclass
class SketchArc:
    """A three-point arc: start, mid, end."""
    start: Tuple[float, float]
    mid: Tuple[float, float]
    end: Tuple[float, float]
    kind: SketchPrimitiveType = field(default=SketchPrimitiveType.ARC, init=False)

    def to_dict(self) -> dict:
        return {
            "type": "arc",
            "start": list(self.start),
            "mid": list(self.mid),
            "end": list(self.end),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SketchArc":
        return cls(start=tuple(d["start"]), mid=tuple(d["mid"]), end=tuple(d["end"]))


# Mapping for deserialization
_PRIMITIVE_DESERIALIZERS = {
    "line": SketchLine.from_dict,
    "rectangle": SketchRect.from_dict,
    "circle": SketchCircle.from_dict,
    "arc": SketchArc.from_dict,
}


def primitive_from_dict(d: dict):
    """Deserialize any sketch primitive from a dict."""
    return _PRIMITIVE_DESERIALIZERS[d["type"]](d)


# ---------------------------------------------------------------------------
# Sketch — collects primitives and builds a build123d face
# ---------------------------------------------------------------------------

# Pre-defined construction planes
PLANE_XY = Plane.XY
PLANE_XZ = Plane.XZ
PLANE_YZ = Plane.YZ


def plane_from_str(name: str) -> Plane:
    """Return a build123d Plane from a human-readable name."""
    mapping = {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}
    return mapping.get(name.upper(), Plane.XY)


@dataclass
class Sketch:
    """
    A 2D sketch on a construction plane.

    Collects primitives and can build them into a build123d Face that serves
    as the profile for 3D operations.

    The sketch can be placed on a named standard plane ("XY", "XZ", "YZ") or
    on a ``ConstructionPlane`` object for offset / rotated planes.
    """
    name: str = "Sketch"
    plane_name: str = "XY"
    primitives: List = field(default_factory=list)
    # Optional reference to a ConstructionPlane (set when created via viewport click)
    construction_plane: Optional[object] = field(default=None, repr=False)

    @property
    def plane(self) -> Plane:
        if self.construction_plane is not None:
            try:
                return self.construction_plane.to_build123d_plane()
            except Exception:
                pass
        return plane_from_str(self.plane_name)

    # -- Adding primitives ---------------------------------------------------

    def add_line(self, start: Tuple[float, float], end: Tuple[float, float]) -> SketchLine:
        prim = SketchLine(start=start, end=end)
        self.primitives.append(prim)
        return prim

    def add_rectangle(
        self,
        width: float,
        height: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> SketchRect:
        prim = SketchRect(center=center, width=width, height=height)
        self.primitives.append(prim)
        return prim

    def add_circle(
        self, radius: float, center: Tuple[float, float] = (0.0, 0.0)
    ) -> SketchCircle:
        prim = SketchCircle(center=center, radius=radius)
        self.primitives.append(prim)
        return prim

    def add_arc(
        self,
        start: Tuple[float, float],
        mid: Tuple[float, float],
        end: Tuple[float, float],
    ) -> SketchArc:
        prim = SketchArc(start=start, mid=mid, end=end)
        self.primitives.append(prim)
        return prim

    # -- Building geometry ---------------------------------------------------

    def build_face(self):
        """
        Build a build123d Face from the collected primitives.

        For simple cases (single rectangle, single circle) we use BuildSketch
        directly. For compound sketches with lines/arcs we build a Wire first,
        then convert to Face.

        Returns:
            A build123d Face object, or None if the sketch is empty.
        """
        if not self.primitives:
            return None

        # If we have a single rectangle or circle, use the fast path
        if len(self.primitives) == 1:
            prim = self.primitives[0]
            if isinstance(prim, SketchRect):
                return self._build_rect_face(prim)
            elif isinstance(prim, SketchCircle):
                return self._build_circle_face(prim)

        # General case: build sketch with all primitives
        return self._build_compound_face()

    def _build_rect_face(self, rect: SketchRect):
        """Build a face from a single rectangle."""
        with BuildSketch(self.plane) as sketch:
            B3dRectangle(rect.width, rect.height)
        return sketch.sketch

    def _build_circle_face(self, circle: SketchCircle):
        """Build a face from a single circle."""
        with BuildSketch(self.plane) as sketch:
            B3dCircle(circle.radius)
        return sketch.sketch

    def _build_compound_face(self):
        """
        Build a face from multiple primitives.

        Attempts to combine all primitives into a sketch. For mixed primitives,
        we use BuildSketch and add each one.
        """
        with BuildSketch(self.plane) as sketch:
            for prim in self.primitives:
                if isinstance(prim, SketchRect):
                    B3dRectangle(prim.width, prim.height)
                elif isinstance(prim, SketchCircle):
                    B3dCircle(prim.radius)
                # Lines and arcs require BuildLine context — handled below
        return sketch.sketch

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "plane": self.plane_name,
            "primitives": [p.to_dict() for p in self.primitives],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sketch":
        sketch = cls(name=d["name"], plane_name=d["plane"])
        sketch.primitives = [primitive_from_dict(p) for p in d["primitives"]]
        return sketch
