"""
USD Bridge — writes tessellated OCC meshes as USD prims on the Isaac Sim stage.

This module handles the conversion from TessellatedMesh data to USD Mesh prims,
including:
- Creating/updating mesh prims with vertex data
- Setting normals and display colors
- Managing the prim hierarchy under a "/World/ParametricCAD" scope
- Optionally generating collision meshes for physics simulation
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..kernel.tessellator import Tessellator, TessellatedMesh


# USD / Omniverse imports — these are available in the Kit Python environment
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics, Vt
    import omni.usd

    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


# Default root path for all CAD-generated prims
DEFAULT_ROOT_PATH = "/World/SparkWorks"
DEFAULT_DISPLAY_COLOR = (0.7, 0.7, 0.8)  # Light steel blue


class UsdBridge:
    """
    Bridges OCC tessellated geometry to USD mesh prims on the Isaac Sim stage.

    Usage:
        bridge = UsdBridge()
        bridge.update_mesh(solid, prim_name="MyPart")
        bridge.clear()
    """

    def __init__(
        self,
        root_path: str = DEFAULT_ROOT_PATH,
        tessellator: Optional[Tessellator] = None,
        display_color: Tuple[float, float, float] = DEFAULT_DISPLAY_COLOR,
    ):
        self.root_path = root_path
        self.tessellator = tessellator or Tessellator()
        self.display_color = display_color
        self._active_prims: dict[str, str] = {}  # name -> prim path

    # -- Public API ----------------------------------------------------------

    def update_mesh(
        self,
        solid,
        prim_name: str = "Part",
        display_color: Optional[Tuple[float, float, float]] = None,
        add_collision: bool = False,
    ) -> Optional[str]:
        """
        Tessellate a build123d solid and write/update it as a USD mesh prim.

        Args:
            solid: A build123d Part/Solid/Shape, or None to clear the prim.
            prim_name: Name for the mesh prim under the root scope.
            display_color: RGB color tuple (0-1 range). Uses default if None.
            add_collision: If True, also creates a collision mesh.

        Returns:
            The prim path of the created/updated mesh, or None on failure.
        """
        if not USD_AVAILABLE:
            print("[SparkWorks] USD API not available")
            return None

        stage = self._get_stage()
        if stage is None:
            return None

        # Ensure root scope exists
        self._ensure_root_scope(stage)

        if solid is None:
            # Clear the mesh if solid is None
            self._remove_prim(stage, prim_name)
            return None

        # Tessellate the solid
        mesh_data = self.tessellator.tessellate(solid)
        if not mesh_data.is_valid:
            print(f"[SparkWorks] Tessellation produced no geometry for {prim_name}")
            return None

        # Write mesh prim
        prim_path = f"{self.root_path}/{prim_name}"
        self._write_mesh_prim(stage, prim_path, mesh_data, display_color or self.display_color)

        # Optionally add collision
        if add_collision:
            self._add_collision_mesh(stage, prim_path)

        self._active_prims[prim_name] = prim_path
        return prim_path

    def clear(self):
        """Remove all CAD-generated prims from the stage."""
        if not USD_AVAILABLE:
            return

        stage = self._get_stage()
        if stage is None:
            return

        # Remove the entire root scope
        root_prim = stage.GetPrimAtPath(self.root_path)
        if root_prim.IsValid():
            stage.RemovePrim(self.root_path)

        self._active_prims.clear()

    def remove_prim(self, prim_name: str):
        """Remove a specific CAD prim by name."""
        if not USD_AVAILABLE:
            return

        stage = self._get_stage()
        if stage is not None:
            self._remove_prim(stage, prim_name)

    def get_prim_path(self, prim_name: str) -> Optional[str]:
        """Get the USD prim path for a named CAD part."""
        return self._active_prims.get(prim_name)

    # -- Internal methods ----------------------------------------------------

    @staticmethod
    def _get_stage():
        """Get the current USD stage from Omniverse."""
        try:
            context = omni.usd.get_context()
            return context.get_stage()
        except Exception as e:
            print(f"[SparkWorks] Could not get USD stage: {e}")
            return None

    def _ensure_root_scope(self, stage):
        """Create the root Scope prim if it doesn't exist."""
        root_prim = stage.GetPrimAtPath(self.root_path)
        if not root_prim.IsValid():
            # Create parent paths as needed
            parts = self.root_path.strip("/").split("/")
            current_path = ""
            for part in parts:
                current_path += f"/{part}"
                prim = stage.GetPrimAtPath(current_path)
                if not prim.IsValid():
                    if current_path == self.root_path:
                        UsdGeom.Scope.Define(stage, current_path)
                    else:
                        UsdGeom.Xform.Define(stage, current_path)

    def _write_mesh_prim(
        self,
        stage,
        prim_path: str,
        mesh_data: TessellatedMesh,
        display_color: Tuple[float, float, float],
    ):
        """Write or update a UsdGeom.Mesh prim with tessellation data."""
        # Define the mesh prim (creates or retrieves existing)
        mesh_prim = UsdGeom.Mesh.Define(stage, prim_path)

        # Set vertex positions
        points = Vt.Vec3fArray([Gf.Vec3f(*p) for p in mesh_data.points])
        mesh_prim.GetPointsAttr().Set(points)

        # Set face topology
        face_counts = Vt.IntArray(mesh_data.face_vertex_counts)
        mesh_prim.GetFaceVertexCountsAttr().Set(face_counts)

        face_indices = Vt.IntArray(mesh_data.face_vertex_indices)
        mesh_prim.GetFaceVertexIndicesAttr().Set(face_indices)

        # Set normals
        normals = Vt.Vec3fArray(
            [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in mesh_data.normals]
        )
        mesh_prim.GetNormalsAttr().Set(normals)
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set subdivision scheme to none (we want exact triangles)
        mesh_prim.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        # Set display color
        color = Vt.Vec3fArray([Gf.Vec3f(*display_color)])
        mesh_prim.GetDisplayColorAttr().Set(color)

        # Set double-sided rendering
        mesh_prim.GetDoubleSidedAttr().Set(True)

    def _add_collision_mesh(self, stage, prim_path: str):
        """Add a physics collision mesh API to an existing mesh prim."""
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        # Apply mesh collision API
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)

        # Apply collision API
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

    def _remove_prim(self, stage, prim_name: str):
        """Remove a prim by name."""
        prim_path = f"{self.root_path}/{prim_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            stage.RemovePrim(prim_path)
        self._active_prims.pop(prim_name, None)
