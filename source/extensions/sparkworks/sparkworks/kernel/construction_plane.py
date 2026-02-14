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
    color: Tuple[float, float, float] = (0.2, 0.4, 0.9)
    opacity: float = 0.15
    prim_path: str = ""

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

        s = self.size
        cx, cy, cz = self.world_origin

        return [
            (cx - s * ux - s * vx, cy - s * uy - s * vy, cz - s * uz - s * vz),
            (cx + s * ux - s * vx, cy + s * uy - s * vy, cz + s * uz - s * vz),
            (cx + s * ux + s * vx, cy + s * uy + s * vy, cz + s * uz + s * vz),
            (cx - s * ux + s * vx, cy - s * uy + s * vy, cz - s * uz + s * vz),
        ]

    def to_build123d_plane(self) -> Plane:
        """
        Convert to a ``build123d.Plane`` suitable for sketch construction.

        Handles base plane type, offset, and rotation.
        """
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
