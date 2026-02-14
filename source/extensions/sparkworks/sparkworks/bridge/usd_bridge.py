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

from typing import Dict, List, Optional, Tuple

from ..kernel.tessellator import Tessellator, TessellatedMesh
from ..kernel.construction_plane import ConstructionPlane


# USD / Omniverse imports — these are available in the Kit Python environment
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics, Vt
    import omni.usd

    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


# Default root path for all CAD-generated prims
DEFAULT_ROOT_PATH = "/World/SparkWorks"
CONSTRUCTION_SCOPE = "Construction"
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

    # -- Construction planes -------------------------------------------------

    @property
    def construction_root(self) -> str:
        """USD path for the Construction scope."""
        return f"{self.root_path}/{CONSTRUCTION_SCOPE}"

    def create_construction_planes(
        self, planes: List[ConstructionPlane]
    ) -> Dict[str, str]:
        """
        Create semi-transparent plane meshes in the viewport.

        Each plane becomes a quad mesh under
        ``/World/SparkWorks/Construction/<name>``.

        Returns:
            Mapping of plane name -> prim path.
        """
        if not USD_AVAILABLE:
            print("[SparkWorks] USD API not available")
            return {}

        stage = self._get_stage()
        if stage is None:
            return {}

        self._ensure_root_scope(stage)

        # Ensure Construction Xform exists
        constr_path = self.construction_root
        constr_prim = stage.GetPrimAtPath(constr_path)
        if not constr_prim.IsValid():
            UsdGeom.Xform.Define(stage, constr_path)

        result: Dict[str, str] = {}
        for plane in planes:
            prim_path = f"{constr_path}/{plane.name}"
            plane.prim_path = prim_path
            self._write_plane_mesh(stage, prim_path, plane)
            result[plane.name] = prim_path
            print(f"[SparkWorks] Created construction plane '{plane.name}' at {prim_path}")

        return result

    def remove_construction_planes(self):
        """Remove the entire Construction scope from the stage."""
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        constr_prim = stage.GetPrimAtPath(self.construction_root)
        if constr_prim.IsValid():
            stage.RemovePrim(self.construction_root)

    def is_construction_plane(self, prim_path: str) -> Optional[str]:
        """
        Check whether a USD prim path belongs to a construction plane.

        Returns:
            The plane name (e.g. "OriginXY") if *prim_path* is under
            the Construction scope, otherwise ``None``.
        """
        prefix = self.construction_root + "/"
        if prim_path.startswith(prefix):
            # The plane name is the first path component after Construction/
            remainder = prim_path[len(prefix):]
            return remainder.split("/")[0]
        return None

    def _write_plane_mesh(
        self,
        stage,
        prim_path: str,
        plane: ConstructionPlane,
    ):
        """
        Write a single construction plane as a semi-transparent quad mesh.
        """
        verts = plane.quad_vertices()

        mesh = UsdGeom.Mesh.Define(stage, prim_path)

        # 4 vertices
        points = Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts])
        mesh.GetPointsAttr().Set(points)

        # 1 quad face (4 vertices)
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))

        # Normal (same for all vertices — flat quad)
        n = Gf.Vec3f(*plane.normal)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray([n, n, n, n]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # No subdivision
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

        # Display colour
        mesh.GetDisplayColorAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(*plane.color)])
        )

        # Semi-transparent via displayOpacity primvar
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        opacity_pv = primvars_api.CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray
        )
        opacity_pv.Set(Vt.FloatArray([plane.opacity]))

        # Double-sided so it's visible from both directions
        mesh.GetDoubleSidedAttr().Set(True)

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
