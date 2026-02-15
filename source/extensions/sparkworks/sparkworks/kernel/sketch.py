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
    # Optional imported face profile — when set, build_face/build_all_faces
    # return this face directly instead of building from primitives.  Used
    # for "extrude face" workflows where the user selects an existing body
    # face and extrudes it without drawing primitives.
    face_profile: Optional[object] = field(default=None, repr=False)
    # Metadata for recreating the face_profile on reload: body name + face index
    face_profile_body: Optional[str] = field(default=None, repr=False)
    face_profile_index: Optional[int] = field(default=None, repr=False)

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

    # -- Closed-loop detection -----------------------------------------------

    def detect_closed_loops(self, tolerance: float = 0.01) -> List[List["SketchLine"]]:
        """
        Find closed loops formed by connected SketchLine segments.

        Walks through the line primitives and groups them into chains where
        each segment's start matches the previous segment's end (within
        *tolerance*).  A chain is "closed" if its last endpoint matches its
        first start point.

        Returns:
            A list of closed loops.  Each loop is an ordered list of
            SketchLine objects whose endpoints connect head-to-tail.
        """
        lines = [p for p in self.primitives if isinstance(p, SketchLine)]
        if not lines:
            return []

        def _close(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
            return math.hypot(a[0] - b[0], a[1] - b[1]) < tolerance

        # Build chains of connected lines
        used = [False] * len(lines)
        loops: List[List[SketchLine]] = []

        for start_idx, line in enumerate(lines):
            if used[start_idx]:
                continue

            chain = [line]
            used[start_idx] = True
            chain_end = line.end

            # Greedily extend the chain
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(lines):
                    if used[j]:
                        continue
                    if _close(chain_end, other.start):
                        chain.append(other)
                        used[j] = True
                        chain_end = other.end
                        changed = True
                        break
                    # Also try reversed segment
                    if _close(chain_end, other.end):
                        # Reverse the segment direction
                        reversed_line = SketchLine(start=other.end, end=other.start)
                        chain.append(reversed_line)
                        used[j] = True
                        chain_end = reversed_line.end
                        changed = True
                        break

            # Check if chain is closed
            if len(chain) >= 3 and _close(chain_end, chain[0].start):
                loops.append(chain)

        return loops

    # -- Building geometry ---------------------------------------------------

    def build_face(self):
        """
        Build a build123d Face from the collected primitives.

        Supports:
        - Single rectangle or circle (fast path)
        - Closed loops formed by connected SketchLine segments
        - Compound sketches with rects + circles

        Returns:
            A build123d Face object, or None if the sketch is empty or
            contains no closed/extrudable profiles.
        """
        if not self.primitives:
            return None

        # Fast path: single rectangle
        if len(self.primitives) == 1:
            prim = self.primitives[0]
            if isinstance(prim, SketchRect):
                return self._build_rect_face(prim)
            elif isinstance(prim, SketchCircle):
                return self._build_circle_face(prim)

        # Check for closed loops from lines
        loops = self.detect_closed_loops()
        if loops:
            face = self._build_wire_face(loops[0])
            if face is not None:
                return face

        # Fallback: build from rectangles / circles
        buildable = [p for p in self.primitives
                     if isinstance(p, (SketchRect, SketchCircle))]
        if buildable:
            return self._build_compound_face()

        print(f"[SparkWorks] Sketch '{self.name}' has no closed profiles to build")
        return None

    def build_all_faces(self) -> list:
        """
        Build ALL extrudable faces from the sketch (not just the first).

        Lines that cross through rectangles/circles split them into
        sub-regions.  When shapes overlap, boolean operations further split
        them into distinct non-overlapping regions (like a Venn diagram).
        Each region is returned as a separate Face that can be individually
        selected and extruded.

        Returns a list of build123d Face objects, one per closed profile.
        """
        raw_faces = []

        # Closed wire loops from lines
        for loop in self.detect_closed_loops():
            face = self._build_wire_face(loop)
            if face is not None:
                raw_faces.append(face)

        # Rectangles and circles
        for prim in self.primitives:
            if isinstance(prim, SketchRect):
                f = self._build_rect_face(prim)
                if f is not None:
                    raw_faces.append(f)
            elif isinstance(prim, SketchCircle):
                f = self._build_circle_face(prim)
                if f is not None:
                    raw_faces.append(f)

        # --- Split faces with crossing line edges ---
        # Collect 3D edges from ALL SketchLine primitives and use them to
        # split any rect/circle/loop faces they cross through.
        line_edges = self._build_line_edges()
        if line_edges and raw_faces:
            split_faces = []
            for face in raw_faces:
                sub = self._split_face_with_edges(face, line_edges)
                split_faces.extend(sub)
            if len(split_faces) != len(raw_faces):
                print(
                    f"[SparkWorks] Line-edge split: {len(raw_faces)} faces "
                    f"-> {len(split_faces)} sub-faces"
                )
            raw_faces = split_faces

        # If multiple faces, split overlapping regions
        if len(raw_faces) > 1:
            return self._split_overlapping_faces(raw_faces)

        return raw_faces

    # -- Line-edge splitting (lines crossing through rects/circles) ------------

    def _build_line_edges(self) -> list:
        """
        Convert all SketchLine primitives to build123d Edge objects on the
        sketch plane.

        These 3D edges can be used with ``BRepFeat_SplitShape`` to split
        faces that they cross through.
        """
        plane = self.plane
        edges = []
        for prim in self.primitives:
            if not isinstance(prim, SketchLine):
                continue
            p1 = plane.from_local_coords(prim.start)
            p2 = plane.from_local_coords(prim.end)
            try:
                with BuildLine() as bl:
                    Line(p1, p2)
                for edge in bl.line.edges():
                    edges.append(edge)
            except Exception:
                pass
        return edges

    @staticmethod
    def _split_face_with_edges(face, edges) -> list:
        """
        Iteratively split a face with each edge using ``BRepFeat_SplitShape``.

        An edge must cross the face from boundary to boundary to produce a
        split.  Edges that miss or lie entirely inside/outside are silently
        skipped, leaving the face unchanged for that iteration.

        Returns a list of sub-faces (may be the original face if no split
        occurred).
        """
        from OCP.BRepFeat import BRepFeat_SplitShape
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopoDS import TopoDS
        from build123d import Face as B3dFace

        current_faces = [face]
        for edge in edges:
            next_faces = []
            for cf in current_faces:
                try:
                    sp = BRepFeat_SplitShape(cf.wrapped)
                    sp.Add(edge.wrapped, TopoDS.Face_s(cf.wrapped))
                    sp.Build()
                    exp = TopExp_Explorer(sp.Shape(), TopAbs_FACE)
                    sub = []
                    while exp.More():
                        sub.append(B3dFace(TopoDS.Face_s(exp.Current())))
                        exp.Next()
                    if len(sub) > 1:
                        next_faces.extend(sub)
                    else:
                        next_faces.append(cf)  # edge didn't split this face
                except Exception:
                    next_faces.append(cf)
            current_faces = next_faces
        return current_faces

    # -- Boolean splitting of overlapping faces --------------------------------

    @staticmethod
    def _faces_from_shape(shape) -> list:
        """Extract non-degenerate Face objects from a boolean result."""
        if shape is None:
            return []
        try:
            result_faces = shape.faces()
            return [f for f in result_faces if f.area > 1e-6]
        except Exception:
            pass
        try:
            if shape.area > 1e-6:
                return [shape]
        except Exception:
            pass
        return []

    @staticmethod
    def _split_overlapping_faces(faces: list) -> list:
        """
        Split a list of faces into distinct non-overlapping regions.

        For each pair of overlapping faces A and B this produces up to
        three regions: A-only, A∩B, and B-only.  Generalises to N faces
        by iteratively cutting each face against every other.
        """
        n = len(faces)
        if n <= 1:
            return list(faces)

        MIN_AREA = 1e-6
        all_regions: list = []

        # 1) For each face, compute the part that doesn't overlap any other
        for i in range(n):
            remaining = faces[i]
            for j in range(n):
                if i == j:
                    continue
                try:
                    remaining = remaining.cut(faces[j])
                except Exception:
                    pass  # cut failed — keep what we have

            for f in Sketch._faces_from_shape(remaining):
                if f.area > MIN_AREA:
                    all_regions.append(f)

        # 2) Compute pairwise intersections (i < j to avoid duplicates)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    common = faces[i].intersect(faces[j])
                    for f in Sketch._faces_from_shape(common):
                        if f.area > MIN_AREA:
                            all_regions.append(f)
                except Exception:
                    pass

        if all_regions:
            print(f"[SparkWorks] Split {n} faces into {len(all_regions)} non-overlapping regions")
            return all_regions

        # Fallback: return originals if splitting failed completely
        return list(faces)

    def _build_wire_face(self, loop: List["SketchLine"]):
        """
        Build a face from a closed loop of SketchLine segments.

        Converts 2D sketch-local coordinates to 3D points on ``self.plane``,
        constructs a Wire via ``BuildLine``, then fills it with ``make_face``.
        """
        plane = self.plane
        try:
            # Convert 2D sketch coords to 3D world points on the plane
            pts_3d = []
            for seg in loop:
                pt = plane.from_local_coords((seg.start[0], seg.start[1]))
                pts_3d.append(pt)
            # Close the loop — add the first point again
            pts_3d.append(pts_3d[0])

            # Build a wire from successive line segments
            with BuildLine() as bl:
                for i in range(len(pts_3d) - 1):
                    Line(pts_3d[i], pts_3d[i + 1])

            wire = bl.line
            if wire is None:
                print("[SparkWorks] BuildLine produced no wire")
                return None

            result = make_face(wire)
            face = self._extract_face(result) if hasattr(result, 'faces') else result
            print(f"[SparkWorks] Built face from {len(loop)}-segment closed loop")
            return face

        except Exception as e:
            print(f"[SparkWorks] Failed to build wire face: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_rect_face(self, rect: SketchRect):
        """Build a face from a single rectangle at its sketch-local center."""
        plane = self.plane
        cx, cy = rect.center
        hw, hh = rect.width / 2.0, rect.height / 2.0

        # Four corners in sketch-local 2D, converted to 3D on the plane
        corners_2d = [
            (cx - hw, cy - hh),
            (cx + hw, cy - hh),
            (cx + hw, cy + hh),
            (cx - hw, cy + hh),
        ]
        try:
            pts_3d = [plane.from_local_coords(c) for c in corners_2d]
            pts_3d.append(pts_3d[0])  # close the loop

            with BuildLine() as bl:
                for i in range(len(pts_3d) - 1):
                    Line(pts_3d[i], pts_3d[i + 1])

            wire = bl.line
            if wire is None:
                return None
            result = make_face(wire)
            return self._extract_face(result) if hasattr(result, 'faces') else result
        except Exception as e:
            print(f"[SparkWorks] Failed to build rect face at {rect.center}: {e}")
            return None

    def _build_circle_face(self, circle: SketchCircle):
        """Build a face from a single circle at its sketch-local center."""
        plane = self.plane
        cx, cy = circle.center
        r = circle.radius

        try:
            # Create an offset plane centered on the circle's 2D position
            center_3d = plane.from_local_coords((cx, cy))
            offset_plane = Plane(
                origin=center_3d,
                x_dir=plane.x_dir,
                z_dir=plane.z_dir,
            )
            with BuildSketch(offset_plane) as sketch:
                B3dCircle(r)
            return self._extract_face(sketch.sketch)
        except Exception as e:
            print(f"[SparkWorks] Failed to build circle face at {circle.center}: {e}")
            return None

    @staticmethod
    def _extract_face(sketch_result):
        """
        Extract a single Face from a BuildSketch result.

        BuildSketch returns a Sketch (Compound) object, but profile
        visualization and normal computation need an actual Face.
        """
        try:
            face_list = sketch_result.faces()
            if face_list:
                return face_list[0]
        except Exception:
            pass
        # Fallback: return the compound itself
        return sketch_result

    def _build_compound_face(self):
        """
        Build a face from the first buildable primitive (rect or circle),
        respecting its center offset.
        """
        for prim in self.primitives:
            if isinstance(prim, SketchRect):
                f = self._build_rect_face(prim)
                if f is not None:
                    return f
            elif isinstance(prim, SketchCircle):
                f = self._build_circle_face(prim)
                if f is not None:
                    return f
        return None

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "plane": self.plane_name,
            "primitives": [p.to_dict() for p in self.primitives],
        }
        # Persist custom plane geometry so it survives save/load
        if self.construction_plane is not None:
            cp = self.construction_plane
            d["plane_type"] = getattr(cp, "plane_type", "XY")
            d["plane_origin"] = list(getattr(cp, "origin", (0, 0, 0)))
            d["plane_normal"] = list(getattr(cp, "normal", (0, 0, 1)))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Sketch":
        sketch = cls(name=d["name"], plane_name=d["plane"])
        sketch.primitives = [primitive_from_dict(p) for p in d["primitives"]]

        # Reconstruct custom plane if origin/normal were saved
        if "plane_origin" in d and "plane_normal" in d:
            from .construction_plane import ConstructionPlane
            sketch.construction_plane = ConstructionPlane(
                name=d.get("plane", "Custom"),
                plane_type=d.get("plane_type", "CUSTOM"),
                origin=tuple(d["plane_origin"]),
                normal=tuple(d["plane_normal"]),
            )
        return sketch
