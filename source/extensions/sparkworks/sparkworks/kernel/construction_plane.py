"""
Construction Plane — defines planes for sketch placement.

Provides a data model for construction planes (origin planes like XY, XZ, YZ
and user-created offset/rotated planes) that can be visualised in the 3D
viewport and used as the basis for 2D sketches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# build123d imports (available inside the container)
from build123d import Plane, Vector


@dataclass
class ConstructionPlane:
    """
    A named construction plane that can be displayed in the viewport and
    used as a sketch surface.

    Attributes:
        name:       Display name (e.g. "OriginXY", "Offset +10 XY").
        plane_type: Base plane identifier ("XY", "XZ", or "YZ").
        origin:     Centre point of the plane in world space.
        normal:     Unit normal vector of the plane.
        offset:     Offset distance along the normal from the base origin.
        rotation:   Rotation angle in degrees around the normal.
        size:       Visual half-extent of the plane rectangle.
        color:      RGB display colour (0-1 per channel).
        opacity:    Display opacity (0 = invisible, 1 = opaque).
        prim_path:  USD prim path once the plane is created on the stage.
    """

    name: str = "Plane"
    plane_type: str = "XY"
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset: float = 0.0
    rotation: float = 0.0
    size: float = 1.0
    size_v: Optional[float] = None  # If set, quad uses size x size_v (rectangular)
    color: Tuple[float, float, float] = (0.2, 0.4, 0.9)
    opacity: float = 0.15
    prim_path: str = ""
    # Optional tessellated face mesh for exact-shape overlays.
    # When set, the bridge renders this mesh instead of a quad.
    # Format: (points: List[(x,y,z)], face_vertex_counts: List[int],
    #          face_vertex_indices: List[int])
    face_mesh: Optional[tuple] = field(default=None, repr=False)

    # -- Derived geometry -----------------------------------------------------

    @property
    def world_origin(self) -> Tuple[float, float, float]:
        """Origin shifted along the normal by *offset*."""
        nx, ny, nz = self.normal
        ox, oy, oz = self.origin
        return (
            ox + nx * self.offset,
            oy + ny * self.offset,
            oz + nz * self.offset,
        )

    def quad_vertices(self) -> List[Tuple[float, float, float]]:
        """
        Return the four corners of the plane rectangle in world space.

        The rectangle is centred on ``world_origin`` and extends
        ``size`` units in each in-plane direction.
        """
        nx, ny, nz = self.normal

        # Build two tangent vectors perpendicular to the normal.
        # Pick a reference vector that isn't parallel to the normal.
        if abs(nx) < 0.9:
            ref = (1.0, 0.0, 0.0)
        else:
            ref = (0.0, 1.0, 0.0)

        # tangent U = normal x ref  (normalised)
        ux = ny * ref[2] - nz * ref[1]
        uy = nz * ref[0] - nx * ref[2]
        uz = nx * ref[1] - ny * ref[0]
        mag = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / mag, uy / mag, uz / mag

        # tangent V = normal x U
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux

        # Apply rotation around the normal
        if self.rotation != 0.0:
            rad = math.radians(self.rotation)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            ux2 = cos_a * ux + sin_a * vx
            uy2 = cos_a * uy + sin_a * vy
            uz2 = cos_a * uz + sin_a * vz
            vx2 = -sin_a * ux + cos_a * vx
            vy2 = -sin_a * uy + cos_a * vy
            vz2 = -sin_a * uz + cos_a * vz
            ux, uy, uz = ux2, uy2, uz2
            vx, vy, vz = vx2, vy2, vz2

        su = self.size                            # half-extent along U
        sv = self.size_v if self.size_v is not None else su  # half-extent along V
        cx, cy, cz = self.world_origin

        return [
            (cx - su * ux - sv * vx, cy - su * uy - sv * vy, cz - su * uz - sv * vz),
            (cx + su * ux - sv * vx, cy + su * uy - sv * vy, cz + su * uz - sv * vz),
            (cx + su * ux + sv * vx, cy + su * uy + sv * vy, cz + su * uz + sv * vz),
            (cx - su * ux + sv * vx, cy - su * uy + sv * vy, cz - su * uz + sv * vz),
        ]

    def to_build123d_plane(self) -> Plane:
        """
        Convert to a ``build123d.Plane`` suitable for sketch construction.

        Handles base plane type, offset, rotation, and CUSTOM planes defined
        by an arbitrary origin + normal.
        """
        if self.plane_type == "CUSTOM":
            # Arbitrary plane from origin + normal
            return Plane(
                origin=Vector(*self.world_origin),
                z_dir=Vector(*self.normal),
            )

        base_planes = {
            "XY": Plane.XY,
            "XZ": Plane.XZ,
            "YZ": Plane.YZ,
        }
        base = base_planes.get(self.plane_type.upper(), Plane.XY)

        if self.offset != 0.0:
            base = base.offset(self.offset)

        if self.rotation != 0.0:
            base = base.rotated((0, 0, self.rotation))

        return base


# ---------------------------------------------------------------------------
# Face-plane extraction from build123d solids
# ---------------------------------------------------------------------------

# Colours assigned to face planes in rotation
_FACE_COLORS = [
    (0.9, 0.7, 0.2),   # gold
    (0.2, 0.9, 0.7),   # teal
    (0.8, 0.3, 0.8),   # magenta
    (0.3, 0.7, 0.9),   # sky blue
    (0.9, 0.5, 0.3),   # orange
    (0.5, 0.9, 0.3),   # lime
]


def _face_half_extents(face, normal: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Compute two half-extents (U, V) for a face by projecting its bounding box
    onto the same tangent vectors that ``quad_vertices`` will use.

    This ensures the U half-extent matches the quad's U direction and
    the V half-extent matches the quad's V direction.

    Returns:
        ``(half_u, half_v)`` — half-extents for the quad, with 5 % padding.
    """
    try:
        bb = face.bounding_box()
        dx = bb.max.X - bb.min.X
        dy = bb.max.Y - bb.min.Y
        dz = bb.max.Z - bb.min.Z

        nx, ny, nz = normal

        # ---- Replicate the tangent-vector logic from quad_vertices ----
        if abs(nx) < 0.9:
            ref = (1.0, 0.0, 0.0)
        else:
            ref = (0.0, 1.0, 0.0)

        # U = normal × ref
        ux = ny * ref[2] - nz * ref[1]
        uy = nz * ref[0] - nx * ref[2]
        uz = nx * ref[1] - ny * ref[0]
        mag = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
        ux, uy, uz = ux / mag, uy / mag, uz / mag

        # V = normal × U
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux

        # Project bbox extents onto U and V
        extent_u = abs(dx * ux) + abs(dy * uy) + abs(dz * uz)
        extent_v = abs(dx * vx) + abs(dy * vy) + abs(dz * vz)

        half_u = extent_u / 2.0 * 1.005
        half_v = extent_v / 2.0 * 1.005

        # Ensure minimum visible size
        half_u = max(half_u, 0.1)
        half_v = max(half_v, 0.1)
        return (half_u, half_v)
    except Exception:
        return (3.0, 3.0)


def _tessellate_face(
    face,
    normal: Tuple[float, float, float],
    offset: float,
) -> Optional[tuple]:
    """
    Tessellate a build123d Face into a triangle mesh, nudged along the normal.

    Returns:
        ``(points, face_vertex_counts, face_vertex_indices)`` or *None* on
        failure.  *points* is a list of ``(x, y, z)`` tuples.
    """
    try:
        tess = face.tessellate(0.001, 0.5)
        if tess is None:
            return None
        raw_verts, raw_tris = tess
        if not raw_verts or not raw_tris:
            return None

        nx, ny, nz = normal
        # Offset each vertex slightly along the face normal
        points = [
            (float(v.X) + nx * offset,
             float(v.Y) + ny * offset,
             float(v.Z) + nz * offset)
            for v in raw_verts
        ]
        face_vertex_counts = [3] * len(raw_tris)
        face_vertex_indices = []
        for tri in raw_tris:
            face_vertex_indices.extend([int(tri[0]), int(tri[1]), int(tri[2])])

        return (points, face_vertex_counts, face_vertex_indices)
    except Exception as e:
        print(f"[SparkWorks] Face tessellation failed: {e}")
        return None


def extract_face_planes(
    solid,
    body_name: str = "Body",
    opacity: float = 0.25,
) -> List[ConstructionPlane]:
    """
    Extract planar faces from a build123d solid and return them as
    ``ConstructionPlane`` objects suitable for "sketch on face" workflows.

    Only truly planar faces are returned (cylinders, splines etc. are skipped).
    Each face plane is sized to match the actual face dimensions.

    Args:
        solid:     A build123d Part/Solid.
        body_name: Name of the parent body (used in plane naming).
        opacity:   Display opacity of the face planes.

    Returns:
        A list of ``ConstructionPlane`` objects, one per planar face.
    """
    if solid is None:
        return []

    try:
        faces = solid.faces()
    except Exception as e:
        print(f"[SparkWorks] Could not enumerate faces: {e}")
        return []

    planes: List[ConstructionPlane] = []
    face_idx = 0

    for face in faces:
        try:
            # Only keep planar faces
            # build123d 0.8: geom_type is a property returning a GeomType enum
            # build123d 0.10+: geom_type() is a method returning a string
            geom = face.geom_type
            if callable(geom):
                geom = geom()
            geom_str = str(geom).upper()
            if "PLANE" not in geom_str:
                continue

            # Get face centre and outward normal
            centre = face.center()
            normal_vec = face.normal_at(centre)

            nx, ny, nz = float(normal_vec.X), float(normal_vec.Y), float(normal_vec.Z)
            # Nudge origin a tiny bit along the normal so the plane
            # sits just in front of the body face and wins raycasts.
            FACE_OFFSET = 0.002
            origin = (
                float(centre.X) + nx * FACE_OFFSET,
                float(centre.Y) + ny * FACE_OFFSET,
                float(centre.Z) + nz * FACE_OFFSET,
            )
            normal = (nx, ny, nz)

            # Tessellate the actual face to get an exact-shape overlay
            face_mesh = _tessellate_face(face, normal, FACE_OFFSET)

            # Fallback quad size (used if tessellation fails)
            half_u, half_v = _face_half_extents(face, normal)

            color = _FACE_COLORS[face_idx % len(_FACE_COLORS)]

            plane = ConstructionPlane(
                name=f"{body_name}_Face{face_idx}",
                plane_type="CUSTOM",
                origin=origin,
                normal=normal,
                size=half_u,
                size_v=half_v,
                color=color,
                opacity=opacity,
                face_mesh=face_mesh,
            )
            planes.append(plane)
            face_idx += 1

        except Exception as e:
            print(f"[SparkWorks] Skipping face: {e}")
            continue

    return planes


# ---------------------------------------------------------------------------
# Planar-face lookup by index (for extrude-from-face rebuild)
# ---------------------------------------------------------------------------

def get_planar_face(solid, face_index: int):
    """
    Return the *face_index*-th **planar** face from *solid*.

    Uses the same enumeration order and planarity check as
    ``extract_face_planes``, so ``face_index`` values stored in
    ``ExtrudeOperation`` stay consistent across rebuilds.

    Returns:
        The ``build123d.Face`` or ``None`` if the index is out of range.
    """
    if solid is None:
        return None
    try:
        faces = solid.faces()
    except Exception:
        return None

    planar_idx = 0
    for face in faces:
        try:
            geom = face.geom_type
            if callable(geom):
                geom = geom()
            if "PLANE" not in str(geom).upper():
                continue
            if planar_idx == face_index:
                return face
            planar_idx += 1
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_origin_planes(size: float = 1.0, opacity: float = 0.15) -> List[ConstructionPlane]:
    """
    Create the three standard origin planes (XY, XZ, YZ).

    Colours follow the Fusion 360 convention:
    - XY (blue)  — perpendicular to Z
    - XZ (green) — perpendicular to Y
    - YZ (red)   — perpendicular to X
    """
    return [
        ConstructionPlane(
            name="OriginXY",
            plane_type="XY",
            normal=(0.0, 0.0, 1.0),
            color=(0.2, 0.4, 0.9),
            size=size,
            opacity=opacity,
        ),
        ConstructionPlane(
            name="OriginXZ",
            plane_type="XZ",
            normal=(0.0, 1.0, 0.0),
            color=(0.2, 0.8, 0.3),
            size=size,
            opacity=opacity,
        ),
        ConstructionPlane(
            name="OriginYZ",
            plane_type="YZ",
            normal=(1.0, 0.0, 0.0),
            color=(0.9, 0.3, 0.2),
            size=size,
            opacity=opacity,
        ),
    ]
