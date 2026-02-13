"""
Tessellator — converts Open Cascade B-Rep solids into triangle meshes.

Uses OCC's built-in BRepMesh algorithm via build123d to produce vertices,
face indices, and normals suitable for USD mesh prims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class TessellatedMesh:
    """
    Triangle mesh result from tessellation.

    Attributes:
        vertices: Nx3 array of vertex positions
        face_indices: Mx3 array of triangle face vertex indices
        normals: Nx3 array of per-vertex normals
        face_vertex_counts: list of vertex count per face (all 3s for triangles)
        face_vertex_indices: flat list of vertex indices for USD
    """
    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    face_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    @property
    def face_vertex_counts(self) -> List[int]:
        """All faces are triangles."""
        return [3] * len(self.face_indices)

    @property
    def face_vertex_indices(self) -> List[int]:
        """Flat list of vertex indices for USD faceVertexIndices."""
        return self.face_indices.flatten().tolist()

    @property
    def points(self) -> List[Tuple[float, float, float]]:
        """Vertex positions as list of tuples for USD."""
        return [tuple(v) for v in self.vertices]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.face_indices)

    @property
    def is_valid(self) -> bool:
        return self.vertex_count > 0 and self.face_count > 0


class Tessellator:
    """
    Converts build123d/OCC solids into triangle meshes.

    Parameters:
        linear_tolerance: Maximum distance between the mesh and the real surface.
            Smaller = more triangles, more accurate.
        angular_tolerance: Maximum angular deflection in radians.
    """

    def __init__(
        self,
        linear_tolerance: float = 0.001,
        angular_tolerance: float = 0.5,
    ):
        self.linear_tolerance = linear_tolerance
        self.angular_tolerance = angular_tolerance

    def tessellate(self, solid) -> TessellatedMesh:
        """
        Tessellate a build123d Part/Solid into a TessellatedMesh.

        Args:
            solid: A build123d Part, Solid, or Shape object.

        Returns:
            TessellatedMesh with vertices, face indices, and normals.
        """
        if solid is None:
            return TessellatedMesh()

        try:
            return self._tessellate_via_build123d(solid)
        except Exception as e:
            print(f"[SparkWorks] Tessellation error: {e}")
            return TessellatedMesh()

    def _tessellate_via_build123d(self, solid) -> TessellatedMesh:
        """
        Use build123d's tessellate method to get mesh data.

        build123d shapes expose a .tessellate(tolerance, angular_tolerance)
        method that returns (vertices, triangles).
        """
        # build123d tessellate returns (vertices, triangles)
        # vertices: list of Vector, triangles: list of (i, j, k) tuples
        tess_result = solid.tessellate(
            self.linear_tolerance, self.angular_tolerance
        )

        if tess_result is None:
            return TessellatedMesh()

        raw_vertices, raw_triangles = tess_result

        if not raw_vertices or not raw_triangles:
            return TessellatedMesh()

        # Convert vertices to numpy array
        vertices = np.array(
            [[v.X, v.Y, v.Z] for v in raw_vertices], dtype=np.float64
        )

        # Convert triangles to numpy array
        face_indices = np.array(raw_triangles, dtype=np.int32)

        # Compute per-vertex normals from face normals
        normals = self._compute_vertex_normals(vertices, face_indices)

        return TessellatedMesh(
            vertices=vertices,
            face_indices=face_indices,
            normals=normals,
        )

    @staticmethod
    def _compute_vertex_normals(
        vertices: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """
        Compute smooth per-vertex normals by averaging face normals.
        """
        normals = np.zeros_like(vertices)

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(face_normal)
            if norm > 1e-10:
                face_normal /= norm
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normals /= norms

        return normals
