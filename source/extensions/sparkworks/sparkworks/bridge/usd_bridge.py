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
TIMELINE_SCOPE = "Timeline"
BODIES_SCOPE = "Bodies"
SETTINGS_SCOPE = "Settings"
SKETCHES_SCOPE = "Sketches"
DEFAULT_DISPLAY_COLOR = (0.7, 0.7, 0.8)  # Light steel blue

# Default settings values
DEFAULT_SETTINGS = {
    "units": "mm",              # mm | cm | m | in | ft
    "meters_per_unit": 0.001,   # corresponds to "mm"
}

# Custom attribute namespace
NS = "sparkworks"


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
        Check whether a USD prim path belongs to an origin / user construction plane
        under ``/World/SparkWorks/Construction/<name>``.

        Returns:
            The plane name (e.g. ``"XY"``, ``"XZ"``, ``"YZ"``), or ``None``
            if the path is not a construction plane.
        """
        prefix = self.construction_root + "/"
        if prim_path.startswith(prefix):
            remainder = prim_path[len(prefix):]
            return remainder.split("/")[0]
        return None

    def is_body_face(self, prim_path: str) -> Optional[str]:
        """
        Check whether a USD prim path belongs to a body face plane
        under ``/World/SparkWorks/Bodies/<body>/Construction/<face>``.

        Returns:
            The face key (e.g. ``"Body1_Face0"``), or ``None``
            if the path is not a body face plane.
        """
        bodies_prefix = self.bodies_root + "/"
        if prim_path.startswith(bodies_prefix):
            remainder = prim_path[len(bodies_prefix):]
            parts = remainder.split("/")
            # Expect: ["Body1", "Construction", "Face0"]
            if len(parts) >= 3 and parts[1] == "Construction":
                body_name = parts[0]
                face_label = parts[2]
                return f"{body_name}_{face_label}"
        return None

    def is_timeline_prim(self, prim_path: str) -> Optional[str]:
        """
        Check whether a USD prim path belongs to a timeline feature.

        Returns:
            The feature prim name (e.g. "F00_Sketch1") or None.
        """
        prefix = self.timeline_root + "/"
        if prim_path.startswith(prefix):
            remainder = prim_path[len(prefix):]
            return remainder.split("/")[0]
        return None

    def get_hidden_primitive_indices(self, feature_index: int, sketch_name: str) -> set:
        """
        Return a set of primitive indices whose USD prims have visibility
        set to ``"invisible"`` (i.e. the eye icon is toggled off in the
        stage hierarchy).

        Args:
            feature_index: The 0-based timeline feature index.
            sketch_name:   The sketch's human name (e.g. "Sketch1").

        Returns:
            A set of ``int`` indices (matching the primitive list order).
        """
        hidden: set = set()
        if not USD_AVAILABLE:
            return hidden
        stage = self._get_stage()
        if stage is None:
            return hidden

        prim_name = f"F{feature_index:02d}_{sketch_name}"
        prims_path = f"{self.timeline_root}/{prim_name}/Primitives"
        prims_prim = stage.GetPrimAtPath(prims_path)
        if not prims_prim or not prims_prim.IsValid():
            return hidden

        children = list(prims_prim.GetChildren())
        children.sort(key=lambda p: p.GetName())

        for i, child in enumerate(children):
            imageable = UsdGeom.Imageable(child)
            if imageable:
                vis = imageable.ComputeVisibility()
                if vis == UsdGeom.Tokens.invisible:
                    hidden.add(i)
        return hidden

    def is_body_prim(self, prim_path: str) -> Optional[str]:
        """
        Check whether a USD prim path belongs to a body mesh (not a face plane).

        Returns:
            The body name (e.g. "Body1") if *prim_path* is the body mesh
            directly under the Bodies scope, otherwise ``None``.
            Paths under ``Bodies/<body>/Construction/`` are NOT body prims.
        """
        prefix = self.bodies_root + "/"
        if prim_path.startswith(prefix):
            remainder = prim_path[len(prefix):]
            parts = remainder.split("/")
            # Only match the body mesh itself (e.g. "Body1"), not children
            # like "Body1/Construction/Face0"
            if len(parts) == 1:
                return parts[0]
        return None

    def create_body_face_planes(
        self, body_name: str, planes: List[ConstructionPlane]
    ) -> Dict[str, str]:
        """
        Create face-plane quads under a body's own Construction Xform.

        Structure::

            /World/SparkWorks/Bodies/<body_name>/Construction/
                Face0   (quad mesh)
                Face1   (quad mesh)

        Returns:
            Mapping of plane name -> prim path.
        """
        if not USD_AVAILABLE:
            print("[SparkWorks] USD API not available")
            return {}

        stage = self._get_stage()
        if stage is None:
            return {}

        body_path = f"{self.bodies_root}/{body_name}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            print(f"[SparkWorks] Body '{body_name}' not found, cannot create face planes")
            return {}

        constr_path = f"{body_path}/Construction"
        constr_prim = stage.GetPrimAtPath(constr_path)
        if not constr_prim.IsValid():
            UsdGeom.Xform.Define(stage, constr_path)

        result: Dict[str, str] = {}
        for plane in planes:
            # Use a short face name (e.g. "Face0") for the prim, but keep
            # the full plane.name (e.g. "Body1_Face0") for lookup.
            face_label = plane.name.replace(f"{body_name}_", "")
            prim_path = f"{constr_path}/{face_label}"
            plane.prim_path = prim_path
            self._write_plane_mesh(stage, prim_path, plane)
            result[plane.name] = prim_path
            print(f"[SparkWorks] Created face plane '{plane.name}' at {prim_path}")

        return result

    def remove_body_face_planes(self, body_name: str):
        """
        Remove the Construction Xform (and all face planes) under a body.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        constr_path = f"{self.bodies_root}/{body_name}/Construction"
        constr_prim = stage.GetPrimAtPath(constr_path)
        if constr_prim.IsValid():
            stage.RemovePrim(constr_path)
            print(f"[SparkWorks] Removed face planes at {constr_path}")

    def set_body_face_planes_visible(self, body_name: str, visible: bool):
        """
        Show or hide all face planes under a body's Construction Xform.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        constr_path = f"{self.bodies_root}/{body_name}/Construction"
        constr_prim = stage.GetPrimAtPath(constr_path)
        if not constr_prim.IsValid():
            return
        imageable = UsdGeom.Imageable(constr_prim)
        if imageable:
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def set_all_face_planes_visible(self, visible: bool):
        """
        Show or hide face planes for ALL bodies on the stage.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        bodies_prim = stage.GetPrimAtPath(self.bodies_root)
        if not bodies_prim.IsValid():
            return
        for body_prim in bodies_prim.GetChildren():
            constr_path = f"{body_prim.GetPath()}/Construction"
            constr_prim = stage.GetPrimAtPath(constr_path)
            if constr_prim.IsValid():
                imageable = UsdGeom.Imageable(constr_prim)
                if imageable:
                    if visible:
                        imageable.MakeVisible()
                    else:
                        imageable.MakeInvisible()

    def remove_face_planes(self):
        """Deprecated — face planes now live under each body's Construction Xform.

        Use ``remove_body_face_planes(body_name)`` instead.
        Kept as a no-op for backward compatibility.
        """
        pass

    def _write_plane_mesh(
        self,
        stage,
        prim_path: str,
        plane: ConstructionPlane,
    ):
        """
        Write a construction plane as a semi-transparent mesh.

        If the plane carries a ``face_mesh`` (tessellated from the actual
        B-Rep face), that exact shape is used.  Otherwise falls back to
        a rectangular quad.
        """
        mesh = UsdGeom.Mesh.Define(stage, prim_path)

        if plane.face_mesh is not None:
            # Exact-shape overlay from tessellated face
            pts, fvc, fvi = plane.face_mesh
            points = Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts])
            mesh.GetPointsAttr().Set(points)
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(fvc))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(fvi))

            # Flat normal for every vertex
            n = Gf.Vec3f(*plane.normal)
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray([n] * len(pts)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        else:
            # Fallback: rectangular quad
            verts = plane.quad_vertices()
            points = Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts])
            mesh.GetPointsAttr().Set(points)
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([4]))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))

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

    # -- Sketch profiles (shaded regions for detected closed loops) ----------

    PROFILES_CHILD = "Profiles"

    def _sketch_profiles_path(self, sketch_name: str) -> str:
        """
        USD path for a sketch's Profiles child Xform.

        Profiles now live under Sketches (not Timeline), e.g.::

            /World/SparkWorks/Sketches/Sketch1/Profiles
        """
        safe_name = sketch_name.replace(" ", "_")
        return f"{self.sketches_root}/{safe_name}/{self.PROFILES_CHILD}"

    def get_profile_paths(self, sketch_name: str) -> List[str]:
        """Return the USD paths of all profile prims under the given sketch."""
        if not USD_AVAILABLE:
            return []
        stage = self._get_stage()
        if stage is None:
            return []
        profiles_root = self._sketch_profiles_path(sketch_name)
        profiles_prim = stage.GetPrimAtPath(profiles_root)
        if not profiles_prim.IsValid():
            return []
        return [str(child.GetPath()) for child in profiles_prim.GetChildren()]

    def create_profile_overlays(
        self,
        faces: list,
        sketch_name: str,
        plane_normal: Optional[Tuple[float, float, float]] = None,
        color: Tuple[float, float, float] = (0.2, 0.8, 0.4),
        opacity: float = 0.15,
    ) -> List[str]:
        """
        Write semi-transparent profile meshes under the sketch in the
        Sketches hierarchy.

        Structure::

            /World/SparkWorks/Sketches/Sketch1/Profiles/
                Profile0   (mesh)
                Profile1   (mesh)

        Args:
            faces:         List of build123d Face objects.
            sketch_name:   Name of the parent sketch.
            plane_normal:  Fallback normal from the sketch's plane.
            color:         RGB display colour.
            opacity:       Display opacity.

        Returns:
            List of prim paths created.
        """
        if not USD_AVAILABLE or not faces:
            return []

        stage = self._get_stage()
        if stage is None:
            return []

        profiles_path = self._sketch_profiles_path(sketch_name)

        # Ensure parent Xform exists
        profiles_prim = stage.GetPrimAtPath(profiles_path)
        if not profiles_prim.IsValid():
            UsdGeom.Xform.Define(stage, profiles_path)

        result = []
        for i, face in enumerate(faces):
            profile_name = f"Profile{i}"
            prim_path = f"{profiles_path}/{profile_name}"

            try:
                # Nudge slightly along face normal so it's visible above the plane
                nx, ny, nz = 0.0, 0.0, 0.0
                try:
                    centre = face.center()
                    nv = face.normal_at(centre)
                    nx, ny, nz = float(nv.X), float(nv.Y), float(nv.Z)
                except Exception:
                    pass

                # If normal extraction failed, use the sketch plane normal
                if abs(nx) + abs(ny) + abs(nz) < 1e-6 and plane_normal:
                    nx, ny, nz = plane_normal

                OFFSET = 0.003
                INSET_DIST = 0.15  # uniform border width in scene units

                # Inset the face boundary by a fixed distance using OCC
                # wire offset so the gap is truly uniform on all sides.
                display_face = face
                try:
                    from build123d import make_face as _mk_face
                    outer = face.outer_wire()
                    inset_wire = outer.offset_2d(-INSET_DIST)
                    display_face = _mk_face(inset_wire)
                except Exception:
                    pass  # fall back to the original face

                tess = display_face.tessellate(0.001, 0.5)
                if tess is None:
                    continue
                raw_verts, raw_tris = tess
                if not raw_verts or not raw_tris:
                    continue

                points = Vt.Vec3fArray([
                    Gf.Vec3f(
                        float(v.X) + nx * OFFSET,
                        float(v.Y) + ny * OFFSET,
                        float(v.Z) + nz * OFFSET,
                    )
                    for v in raw_verts
                ])
                fvc = Vt.IntArray([3] * len(raw_tris))
                fvi = Vt.IntArray([int(idx) for tri in raw_tris for idx in tri])

                mesh = UsdGeom.Mesh.Define(stage, prim_path)
                mesh.GetPointsAttr().Set(points)
                mesh.GetFaceVertexCountsAttr().Set(fvc)
                mesh.GetFaceVertexIndicesAttr().Set(fvi)
                mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

                # Flat normal
                n = Gf.Vec3f(nx, ny, nz)
                mesh.GetNormalsAttr().Set(Vt.Vec3fArray([n] * len(raw_verts)))
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

                mesh.GetDisplayColorAttr().Set(
                    Vt.Vec3fArray([Gf.Vec3f(*color)])
                )

                primvars_api = UsdGeom.PrimvarsAPI(mesh)
                op_pv = primvars_api.CreatePrimvar(
                    "displayOpacity", Sdf.ValueTypeNames.FloatArray
                )
                op_pv.Set(Vt.FloatArray([opacity]))
                mesh.GetDoubleSidedAttr().Set(True)

                result.append(prim_path)
                print(f"[SparkWorks] Created profile '{profile_name}' at {prim_path}")

            except Exception as e:
                print(f"[SparkWorks] Failed to create profile '{profile_name}': {e}")
                continue

        return result

    def remove_profile_overlays(self, profile_paths: Optional[List[str]] = None):
        """
        Remove profile overlay meshes.

        If *profile_paths* is given, remove those specific prims and their
        parent ``Profiles`` Xform if empty.  Otherwise does nothing (the old
        global ``Profiles`` scope is no longer used).
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return

        if not profile_paths:
            return

        # Collect unique parent Profiles Xforms
        parents = set()
        for pp in profile_paths:
            prim = stage.GetPrimAtPath(pp)
            if prim.IsValid():
                parents.add(str(prim.GetParent().GetPath()))
                stage.RemovePrim(pp)

        # Remove the Profiles Xform itself if now empty
        for parent_path in parents:
            parent_prim = stage.GetPrimAtPath(parent_path)
            if parent_prim.IsValid() and not list(parent_prim.GetChildren()):
                stage.RemovePrim(parent_path)

    def remove_all_profile_overlays(self):
        """
        Remove every ``Profiles`` child under every sketch in the
        Sketches hierarchy.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        sk_prim = stage.GetPrimAtPath(self.sketches_root)
        if not sk_prim.IsValid():
            return
        for child in sk_prim.GetChildren():
            profiles_path = f"{child.GetPath()}/{self.PROFILES_CHILD}"
            profiles_prim = stage.GetPrimAtPath(profiles_path)
            if profiles_prim.IsValid():
                stage.RemovePrim(profiles_path)

    def set_sketch_profiles_visible(self, sketch_name: str, visible: bool):
        """
        Show or hide the Profiles child of a specific sketch.

        This is used during timeline scrubbing to hide profiles for
        sketches that are after the current marker position.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        profiles_path = self._sketch_profiles_path(sketch_name)
        prim = stage.GetPrimAtPath(profiles_path)
        if prim.IsValid():
            img = UsdGeom.Imageable(prim)
            if visible:
                img.MakeVisible()
            else:
                img.MakeInvisible()

    def set_sketch_visible(self, sketch_name: str, visible: bool):
        """
        Show or hide an entire sketch Xform (primitives **and** profiles).

        In Fusion 360 hiding a sketch hides everything — lines, constraints,
        and profile regions alike.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        safe_name = sketch_name.replace(" ", "_")
        sk_path = f"{self.sketches_root}/{safe_name}"
        prim = stage.GetPrimAtPath(sk_path)
        if prim.IsValid():
            # Ensure the prim is typed so Imageable API works
            if not prim.GetTypeName():
                UsdGeom.Xform.Define(stage, sk_path)
                prim = stage.GetPrimAtPath(sk_path)
            img = UsdGeom.Imageable(prim)
            if visible:
                img.MakeVisible()
            else:
                img.MakeInvisible()

    def hide_all_sketches(self, except_sketch: str = ""):
        """
        Hide every sketch in the Sketches scope (primitives + profiles),
        except for *except_sketch* which is made visible.

        Mimics Fusion 360: when a new sketch is created or an existing one
        is edited, all other sketches disappear entirely.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        sk_prim = stage.GetPrimAtPath(self.sketches_root)
        if not sk_prim.IsValid():
            return
        safe_except = except_sketch.replace(" ", "_")
        for child in sk_prim.GetChildren():
            child_name = child.GetName()
            # Ensure the child is typed so Imageable API works
            if not child.GetTypeName():
                UsdGeom.Xform.Define(stage, str(child.GetPath()))
                child = stage.GetPrimAtPath(str(child.GetPath()))
            img = UsdGeom.Imageable(child)
            if child_name == safe_except:
                img.MakeVisible()
            else:
                img.MakeInvisible()

    def is_profile_prim(self, prim_path: str) -> Optional[str]:
        """
        Check if a prim path belongs to a profile overlay.

        Profile prims live under
        ``/World/SparkWorks/Sketches/Sketch1/Profiles/ProfileN``.

        Returns:
            The profile name (e.g. "Profile0") or ``None``.
        """
        sk_prefix = self.sketches_root + "/"
        if not prim_path.startswith(sk_prefix):
            return None
        remainder = prim_path[len(sk_prefix):]
        # Expected: "Sketch1/Profiles/Profile0"
        parts = remainder.split("/")
        if len(parts) >= 3 and parts[1] == self.PROFILES_CHILD:
            return parts[2]
        return None

    def highlight_profile(
        self,
        profile_paths: List[str],
        selected_idx: int,
        selected_color: Tuple[float, float, float] = (0.1, 0.6, 1.0),
        selected_opacity: float = 0.25,
        default_color: Tuple[float, float, float] = (0.2, 0.8, 0.4),
        default_opacity: float = 0.15,
    ):
        """
        Highlight one profile and dim the others.

        Args:
            profile_paths: List of prim paths for profile overlays.
            selected_idx:  Which index to highlight.
            selected_color / selected_opacity: Visual style for the selected profile.
            default_color / default_opacity: Visual style for non-selected profiles.
        """
        if not USD_AVAILABLE or not profile_paths:
            return
        stage = self._get_stage()
        if stage is None:
            return

        for i, prim_path in enumerate(profile_paths):
            mesh = UsdGeom.Mesh.Get(stage, prim_path)
            if not mesh:
                continue
            if i == selected_idx:
                color, opacity = selected_color, selected_opacity
            else:
                color, opacity = default_color, default_opacity

            mesh.GetDisplayColorAttr().Set(
                Vt.Vec3fArray([Gf.Vec3f(*color)])
            )
            primvars_api = UsdGeom.PrimvarsAPI(mesh)
            op_pv = primvars_api.GetPrimvar("displayOpacity")
            if op_pv:
                op_pv.Set(Vt.FloatArray([opacity]))

    # -- Bodies ---------------------------------------------------------------

    @property
    def bodies_root(self) -> str:
        """USD path for the Bodies scope."""
        return f"{self.root_path}/{BODIES_SCOPE}"

    def update_body_mesh(
        self,
        solid,
        body_name: str = "Body1",
        display_color: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[str]:
        """
        Write or update a body mesh under ``/World/SparkWorks/Bodies/<name>``.

        Args:
            solid: A build123d Part/Solid, or None to clear the body.
            body_name: Name for the body prim.
            display_color: RGB color tuple. Uses default if None.

        Returns:
            The prim path of the body mesh, or None on failure.
        """
        if not USD_AVAILABLE:
            print("[SparkWorks] USD API not available")
            return None

        stage = self._get_stage()
        if stage is None:
            return None

        self._ensure_root_scope(stage)

        # Ensure Bodies Xform exists
        bodies_path = self.bodies_root
        bodies_prim = stage.GetPrimAtPath(bodies_path)
        if not bodies_prim.IsValid():
            UsdGeom.Xform.Define(stage, bodies_path)

        body_path = f"{bodies_path}/{body_name}"

        if solid is None:
            # Remove body if solid is None
            prim = stage.GetPrimAtPath(body_path)
            if prim.IsValid():
                stage.RemovePrim(body_path)
            return None

        # Remove stale face-plane Construction child (geometry is changing)
        stale_constr = f"{body_path}/Construction"
        stale_prim = stage.GetPrimAtPath(stale_constr)
        if stale_prim.IsValid():
            stage.RemovePrim(stale_constr)

        # Tessellate and write
        mesh_data = self.tessellator.tessellate(solid)
        if not mesh_data.is_valid:
            print(f"[SparkWorks] Tessellation produced no geometry for body '{body_name}'")
            return None

        self._write_mesh_prim(stage, body_path, mesh_data, display_color or self.display_color)
        self._active_prims[body_name] = body_path
        print(f"[SparkWorks] Body '{body_name}' updated at {body_path}")
        return body_path

    def get_body_names(self) -> List[str]:
        """List all body names currently on the stage."""
        if not USD_AVAILABLE:
            return []
        stage = self._get_stage()
        if stage is None:
            return []
        bodies_prim = stage.GetPrimAtPath(self.bodies_root)
        if not bodies_prim.IsValid():
            return []
        return [child.GetName() for child in bodies_prim.GetChildren()]

    # -- Settings persistence -------------------------------------------------

    @property
    def settings_root(self) -> str:
        """USD path for the Settings Xform."""
        return f"{self.root_path}/{SETTINGS_SCOPE}"

    def save_settings(self, settings: Optional[Dict] = None) -> bool:
        """
        Write settings to a ``/World/SparkWorks/Settings`` Xform.

        Stored attributes (``sparkworks:`` namespace)::

            sparkworks:units            = "mm"
            sparkworks:metersPerUnit    = 0.001

        Also calls ``UsdGeom.SetStageMetersPerUnit`` so the viewport
        interprets geometry in the chosen unit system.

        Args:
            settings: Dict with keys like ``units``, ``meters_per_unit``.
                      Falls back to DEFAULT_SETTINGS if *None*.
        Returns:
            True on success.
        """
        if not USD_AVAILABLE:
            return False

        stage = self._get_stage()
        if stage is None:
            return False

        self._ensure_root_scope(stage)

        cfg = {**DEFAULT_SETTINGS, **(settings or {})}
        path = self.settings_root
        xf = UsdGeom.Xform.Define(stage, path)
        prim = xf.GetPrim()

        prim.GetAttribute(f"{NS}:units").Set(cfg["units"]) if prim.HasAttribute(f"{NS}:units") \
            else prim.CreateAttribute(f"{NS}:units", Sdf.ValueTypeNames.String).Set(cfg["units"])

        prim.GetAttribute(f"{NS}:metersPerUnit").Set(cfg["meters_per_unit"]) if prim.HasAttribute(f"{NS}:metersPerUnit") \
            else prim.CreateAttribute(f"{NS}:metersPerUnit", Sdf.ValueTypeNames.Double).Set(cfg["meters_per_unit"])

        # Apply to stage metadata so the viewport respects it
        UsdGeom.SetStageMetersPerUnit(stage, cfg["meters_per_unit"])

        print(f"[SparkWorks] Settings saved: units={cfg['units']} metersPerUnit={cfg['meters_per_unit']}")
        return True

    def load_settings(self) -> Dict:
        """
        Read settings from the ``Settings`` Xform on stage.

        Returns:
            A dict with ``units`` and ``meters_per_unit`` keys.
            Falls back to DEFAULT_SETTINGS if the prim doesn't exist.
        """
        result = dict(DEFAULT_SETTINGS)

        if not USD_AVAILABLE:
            return result

        stage = self._get_stage()
        if stage is None:
            return result

        prim = stage.GetPrimAtPath(self.settings_root)
        if not prim.IsValid():
            return result

        units_attr = prim.GetAttribute(f"{NS}:units")
        if units_attr and units_attr.HasValue():
            result["units"] = units_attr.Get()

        mpu_attr = prim.GetAttribute(f"{NS}:metersPerUnit")
        if mpu_attr and mpu_attr.HasValue():
            result["meters_per_unit"] = float(mpu_attr.Get())

        return result

    def apply_settings(self, settings: Optional[Dict] = None):
        """
        Apply settings to the current stage (sets ``metersPerUnit``).

        If *settings* is None the values are loaded from the USD prim first.
        """
        if not USD_AVAILABLE:
            return

        stage = self._get_stage()
        if stage is None:
            return

        cfg = settings if settings is not None else self.load_settings()
        UsdGeom.SetStageMetersPerUnit(stage, cfg["meters_per_unit"])
        print(f"[SparkWorks] Applied stage metersPerUnit={cfg['meters_per_unit']} ({cfg['units']})")

    # -- Sketches persistence (Fusion 360-style separate store) ---------------

    @property
    def sketches_root(self) -> str:
        """USD path for the Sketches scope."""
        return f"{self.root_path}/{SKETCHES_SCOPE}"

    def save_sketches(self, sketch_registry) -> bool:
        """
        Write all sketches from the registry to USD under::

            /World/SparkWorks/Sketches/
                Sketch1/
                    sparkworks:planeName = "XY"
                    sparkworks:sketchName = "Sketch1"
                    Primitives/
                        Line_000/  ...
                    Profiles/          <-- preserved across saves
                        Profile0
                Sketch2/
                    ...

        The sketch data is independent of the timeline — just like
        Fusion 360's Sketches folder.

        **Important**: Profiles children are preserved across saves.
        Only sketch attributes and Primitives are rewritten.
        """
        if not USD_AVAILABLE:
            return False

        stage = self._get_stage()
        if stage is None:
            return False

        self._ensure_root_scope(stage)

        sk_path = self.sketches_root
        sk_prim = stage.GetPrimAtPath(sk_path)

        # Instead of wiping the entire Sketches scope (which would
        # destroy Profile prims), selectively clear each sketch's
        # attributes and Primitives child while preserving Profiles.
        if sk_prim.IsValid():
            for child in sk_prim.GetChildren():
                # Remove Primitives sub-prim (will be rewritten)
                prims_path = f"{child.GetPath()}/Primitives"
                prims_prim = stage.GetPrimAtPath(prims_path)
                if prims_prim.IsValid():
                    stage.RemovePrim(prims_path)
        else:
            UsdGeom.Xform.Define(stage, sk_path)

        # Save the counter for name generation continuity
        xf = stage.GetPrimAtPath(sk_path)
        self._set_attr(xf, f"{NS}:counter", sketch_registry.counter, Sdf.ValueTypeNames.Int)

        # Remove sketches that no longer exist in the registry
        if sk_prim.IsValid():
            existing_ids = set()
            for child in sk_prim.GetChildren():
                existing_ids.add(child.GetName())
            registry_ids = {sid.replace(" ", "_") for sid in sketch_registry.sketches}
            for stale_id in existing_ids - registry_ids:
                stage.RemovePrim(f"{sk_path}/{stale_id}")

        for sketch_id, sketch in sketch_registry.sketches.items():
            safe_id = sketch_id.replace(" ", "_")
            child_path = f"{sk_path}/{safe_id}"
            # Always define as Xform so the prim has a proper type
            # (it may already exist as an untyped auto-created ancestor).
            UsdGeom.Xform.Define(stage, child_path)
            prim = stage.GetPrimAtPath(child_path)
            # Reuse the existing _write_sketch_attrs to persist sketch data
            self._write_sketch_attrs(stage, child_path, prim, sketch)
            print(f"[SparkWorks] Wrote sketch '{sketch_id}' at {child_path}")

        print(f"[SparkWorks] Sketches saved ({sketch_registry.count} sketches)")
        return True

    def load_sketches(self):
        """
        Read all sketches from USD and return a SketchRegistry.

        Returns:
            A SketchRegistry, or None if no sketches scope exists.
        """
        if not USD_AVAILABLE:
            return None

        stage = self._get_stage()
        if stage is None:
            return None

        sk_prim = stage.GetPrimAtPath(self.sketches_root)
        if not sk_prim.IsValid():
            return None

        from ..timeline.sketch_registry import SketchRegistry

        registry = SketchRegistry()
        counter_val = self._get_attr(sk_prim, f"{NS}:counter")
        if counter_val is not None:
            registry.counter = int(counter_val)

        for child in sk_prim.GetChildren():
            sketch_id = child.GetName()
            sketch = self._read_sketch_from_prim(stage, child)
            if sketch is not None:
                registry._sketches[sketch_id] = sketch
                print(f"[SparkWorks] Loaded sketch '{sketch_id}'")

        print(f"[SparkWorks] Sketches loaded ({registry.count} sketches)")
        return registry

    # -- Timeline persistence ------------------------------------------------

    @property
    def timeline_root(self) -> str:
        """USD path for the Timeline scope."""
        return f"{self.root_path}/{TIMELINE_SCOPE}"

    def save_timeline(self, features: List) -> bool:
        """
        Write the full timeline to USD as a hierarchy of Xform prims.

        Structure::

            /World/SparkWorks/Timeline/
                00_Sketch1/
                    sparkworks:featureType = "SKETCH"
                    sparkworks:name = "Sketch1"
                    sparkworks:suppressed = false
                    sparkworks:createdAt = 1707...
                    sparkworks:planeName = "XY"
                    Primitives/
                        Line_0/   sparkworks:type, startX/Y, endX/Y
                        Rect_1/   sparkworks:type, centerX/Y, width, height
                01_Extrude1/
                    sparkworks:featureType = "OPERATION"
                    sparkworks:opType = "extrude"
                    sparkworks:distance = 10.0
                    ...

        Args:
            features: List of Feature objects from the Timeline.

        Returns:
            True on success.
        """
        if not USD_AVAILABLE:
            print("[SparkWorks] USD API not available")
            return False

        stage = self._get_stage()
        if stage is None:
            return False

        self._ensure_root_scope(stage)

        # Clear existing timeline prims
        tl_path = self.timeline_root
        tl_prim = stage.GetPrimAtPath(tl_path)
        if tl_prim.IsValid():
            stage.RemovePrim(tl_path)
            print(f"[SparkWorks] Cleared existing Timeline at {tl_path}")

        # Create Timeline Xform
        UsdGeom.Xform.Define(stage, tl_path)

        for idx, feature in enumerate(features):
            safe_name = feature.name.replace(" ", "_")
            # USD prim names cannot start with a digit — prefix with 'F'
            feature_path = f"{tl_path}/F{idx:02d}_{safe_name}"
            try:
                self._write_feature_prim(stage, feature_path, feature)
                print(f"[SparkWorks] Wrote feature '{feature.name}' at {feature_path}")
            except Exception as e:
                print(f"[SparkWorks] ERROR writing feature '{feature.name}': {e}")
                import traceback
                traceback.print_exc()

        print(f"[SparkWorks] Timeline saved ({len(features)} features) at {tl_path}")
        return True

    def load_timeline(self) -> Optional[List]:
        """
        Read the timeline from USD prims and return a list of Feature objects.

        Returns:
            List of Feature objects, or None if no timeline exists on the stage.
        """
        if not USD_AVAILABLE:
            return None

        stage = self._get_stage()
        if stage is None:
            return None

        tl_prim = stage.GetPrimAtPath(self.timeline_root)
        if not tl_prim.IsValid():
            return None

        # Import here to avoid circular imports at module level
        from ..timeline.timeline import Feature, FeatureType

        # Collect child prims sorted by name (00_, 01_, ...)
        children = []
        for child in tl_prim.GetChildren():
            children.append(child)
        children.sort(key=lambda p: p.GetName())

        features = []
        for child_prim in children:
            feature = self._read_feature_prim(stage, child_prim)
            if feature is not None:
                features.append(feature)

        print(f"[SparkWorks] Timeline loaded ({len(features)} features)")
        return features

    # -- Feature write helpers -----------------------------------------------

    def _write_feature_prim(self, stage, prim_path: str, feature):
        """Write a single Feature as a USD Xform with custom attributes."""
        from ..timeline.timeline import FeatureType

        xform = UsdGeom.Xform.Define(stage, prim_path)
        prim = xform.GetPrim()

        # Common attributes
        self._set_attr(prim, f"{NS}:name", feature.name, Sdf.ValueTypeNames.String)
        self._set_attr(prim, f"{NS}:featureType", feature.feature_type.name, Sdf.ValueTypeNames.String)
        self._set_attr(prim, f"{NS}:suppressed", feature.suppressed, Sdf.ValueTypeNames.Bool)
        self._set_attr(prim, f"{NS}:createdAt", feature.created_at, Sdf.ValueTypeNames.Double)

        if feature.feature_type == FeatureType.SKETCH:
            # Sketch features now store only a sketch_id reference.
            # The sketch data lives under /World/SparkWorks/Sketches/.
            if feature.sketch_id:
                self._set_attr(prim, f"{NS}:sketchId", feature.sketch_id, Sdf.ValueTypeNames.String)
            # Also write inline sketch attrs for backward compat / quick access
            sketch = feature.sketch  # resolves from registry
            if sketch:
                self._write_sketch_attrs(stage, prim_path, prim, sketch)
        elif feature.feature_type == FeatureType.OPERATION and feature.operation:
            self._write_operation_attrs(prim, feature.operation)

    def _ensure_sketch_xform(self, stage, sketch_path: str):
        """Ensure *sketch_path* is a typed ``Xform`` prim.

        When child prims are created first (e.g. ``Primitives/Line_000``),
        USD auto-creates ancestor prims as *untyped overs*.  This helper
        re-defines the sketch prim as a proper Xform so it has a type
        in the stage panel and supports visibility toggling.
        """
        prim = stage.GetPrimAtPath(sketch_path)
        if not prim.IsValid() or not prim.GetTypeName():
            UsdGeom.Xform.Define(stage, sketch_path)

    # -- 3D curve visual for sketch lines ------------------------------------

    def write_sketch_curves(self, sketch_path: str, sketch):
        """Create a ``BasisCurves`` prim showing sketch lines in 3D.

        The curves are projected onto the sketch's construction plane so
        the sketch geometry is visible in the 3D viewport and responds
        to the prim's visibility toggle.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return

        from ..kernel.sketch import SketchLine, SketchRect, SketchCircle, SketchArc

        cp = sketch.construction_plane
        if cp is None:
            return

        curves_path = f"{sketch_path}/CurvesVisual"

        # Collect 3D points and per-curve vertex counts
        all_points = []
        curve_counts = []
        import math as _m

        for prim in sketch.primitives:
            if isinstance(prim, SketchLine):
                p0 = cp.to_world(prim.start[0], prim.start[1])
                p1 = cp.to_world(prim.end[0], prim.end[1])
                all_points.extend([Gf.Vec3f(*p0), Gf.Vec3f(*p1)])
                curve_counts.append(2)
            elif isinstance(prim, SketchRect):
                corners = prim.corners  # BL, BR, TR, TL
                pts = [Gf.Vec3f(*cp.to_world(c[0], c[1])) for c in corners]
                pts.append(pts[0])  # close the loop
                all_points.extend(pts)
                curve_counts.append(5)
            elif isinstance(prim, SketchCircle):
                cx, cy = prim.center
                r = prim.radius
                segs = 32
                pts = []
                for i in range(segs + 1):
                    a = 2.0 * _m.pi * i / segs
                    sx = cx + r * _m.cos(a)
                    sy = cy + r * _m.sin(a)
                    pts.append(Gf.Vec3f(*cp.to_world(sx, sy)))
                all_points.extend(pts)
                curve_counts.append(segs + 1)
            elif isinstance(prim, SketchArc):
                pts_2d = [prim.start, prim.mid, prim.end]
                pts = [Gf.Vec3f(*cp.to_world(p[0], p[1])) for p in pts_2d]
                all_points.extend(pts)
                curve_counts.append(3)

        if not all_points:
            # Remove stale visual if no primitives
            old = stage.GetPrimAtPath(curves_path)
            if old.IsValid():
                stage.RemovePrim(curves_path)
            return

        curves = UsdGeom.BasisCurves.Define(stage, curves_path)
        curves.GetPointsAttr().Set(Vt.Vec3fArray(all_points))
        curves.GetCurveVertexCountsAttr().Set(Vt.IntArray(curve_counts))
        curves.GetTypeAttr().Set(UsdGeom.Tokens.linear)
        curves.GetWidthsAttr().Set(Vt.FloatArray([0.15] * len(all_points)))
        curves.GetWidthsInterpolation()  # default vertex

        # Sketch-like display colour (light blue)
        primvars = UsdGeom.PrimvarsAPI(curves.GetPrim())
        color_pv = primvars.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray
        )
        color_pv.Set(Vt.Vec3fArray([Gf.Vec3f(0.5, 0.5, 0.5)]))

    def _write_sketch_attrs(self, stage, prim_path: str, prim, sketch):
        """Write sketch-specific attributes and primitive children."""
        self._set_attr(prim, f"{NS}:planeName", sketch.plane_name, Sdf.ValueTypeNames.String)
        self._set_attr(prim, f"{NS}:sketchName", sketch.name, Sdf.ValueTypeNames.String)

        # Persist custom plane geometry (origin + normal) for face-on-body sketches
        cp = sketch.construction_plane
        if cp is not None:
            plane_type = getattr(cp, "plane_type", "XY")
            origin = getattr(cp, "origin", (0.0, 0.0, 0.0))
            normal = getattr(cp, "normal", (0.0, 0.0, 1.0))
            self._set_attr(prim, f"{NS}:planeType", plane_type, Sdf.ValueTypeNames.String)
            self._set_attr(prim, f"{NS}:planeOriginX", float(origin[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:planeOriginY", float(origin[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:planeOriginZ", float(origin[2]), Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:planeNormalX", float(normal[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:planeNormalY", float(normal[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:planeNormalZ", float(normal[2]), Sdf.ValueTypeNames.Double)

        # Create Primitives Xform and write each primitive (with point sub-prims)
        prims_path = f"{prim_path}/Primitives"
        UsdGeom.Xform.Define(stage, prims_path)

        for i, p in enumerate(sketch.primitives):
            self._write_single_prim_to_usd(stage, prims_path, i, p)

        # Write 3D curve visual so the sketch appears in the 3D viewport
        self.write_sketch_curves(prim_path, sketch)

    # -- Single-primitive immediate write ------------------------------------

    def write_sketch_primitive(self, sketch_path: str, prim_index: int, primitive) -> str:
        """
        Write a single sketch primitive to USD immediately.

        Creates the appropriate Xform prim under ``sketch_path/Primitives/``
        and returns its USD path.  For lines, also creates ``StartPt`` and
        ``EndPt`` child prims.  For rectangles, creates ``CornerBL/BR/TR/TL``
        child prims.  Sets ``primitive.usd_path`` and point sub-prim paths.
        """
        if not USD_AVAILABLE:
            return ""
        stage = self._get_stage()
        if stage is None:
            return ""

        from ..kernel.sketch import SketchLine, SketchRect, SketchCircle, SketchArc

        # Ensure the sketch prim itself is typed as Xform (not an
        # auto-created untyped ancestor).
        self._ensure_sketch_xform(stage, sketch_path)

        prims_path = f"{sketch_path}/Primitives"
        UsdGeom.Xform.Define(stage, prims_path)

        child_path = self._write_single_prim_to_usd(stage, prims_path, prim_index, primitive)
        return child_path

    def _write_single_prim_to_usd(self, stage, prims_path: str, i: int, p) -> str:
        """Create a single primitive prim and return its path."""
        from ..kernel.sketch import SketchLine, SketchRect, SketchCircle, SketchArc

        if isinstance(p, SketchLine):
            child_path = f"{prims_path}/Line_{i:03d}"
            child_xf = UsdGeom.Xform.Define(stage, child_path)
            cp = child_xf.GetPrim()
            self._set_attr(cp, f"{NS}:type", "line", Sdf.ValueTypeNames.String)
            self._set_attr(cp, f"{NS}:startX", float(p.start[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:startY", float(p.start[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:endX", float(p.end[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:endY", float(p.end[1]), Sdf.ValueTypeNames.Double)
            # Create point child prims for selection
            start_path = f"{child_path}/StartPt"
            end_path = f"{child_path}/EndPt"
            sp = UsdGeom.Xform.Define(stage, start_path).GetPrim()
            ep = UsdGeom.Xform.Define(stage, end_path).GetPrim()
            self._set_attr(sp, f"{NS}:type", "point", Sdf.ValueTypeNames.String)
            self._set_attr(sp, f"{NS}:x", float(p.start[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(sp, f"{NS}:y", float(p.start[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(ep, f"{NS}:type", "point", Sdf.ValueTypeNames.String)
            self._set_attr(ep, f"{NS}:x", float(p.end[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(ep, f"{NS}:y", float(p.end[1]), Sdf.ValueTypeNames.Double)
            p.usd_path = child_path
            p.start_usd_path = start_path
            p.end_usd_path = end_path

        elif isinstance(p, SketchRect):
            child_path = f"{prims_path}/Rect_{i:03d}"
            child_xf = UsdGeom.Xform.Define(stage, child_path)
            cp = child_xf.GetPrim()
            self._set_attr(cp, f"{NS}:type", "rectangle", Sdf.ValueTypeNames.String)
            self._set_attr(cp, f"{NS}:centerX", float(p.center[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:centerY", float(p.center[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:width", float(p.width), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:height", float(p.height), Sdf.ValueTypeNames.Double)
            p.usd_path = child_path
            # Create corner point sub-prims (BL, BR, TR, TL)
            corner_names = ["CornerBL", "CornerBR", "CornerTR", "CornerTL"]
            corners = p.corners
            p.corner_usd_paths = []
            for ci, (cname, (cx, cy)) in enumerate(zip(corner_names, corners)):
                cpath = f"{child_path}/{cname}"
                cp_pt = UsdGeom.Xform.Define(stage, cpath).GetPrim()
                self._set_attr(cp_pt, f"{NS}:type", "point", Sdf.ValueTypeNames.String)
                self._set_attr(cp_pt, f"{NS}:x", float(cx), Sdf.ValueTypeNames.Double)
                self._set_attr(cp_pt, f"{NS}:y", float(cy), Sdf.ValueTypeNames.Double)
                p.corner_usd_paths.append(cpath)

        elif isinstance(p, SketchCircle):
            child_path = f"{prims_path}/Circle_{i:03d}"
            child_xf = UsdGeom.Xform.Define(stage, child_path)
            cp = child_xf.GetPrim()
            self._set_attr(cp, f"{NS}:type", "circle", Sdf.ValueTypeNames.String)
            self._set_attr(cp, f"{NS}:centerX", float(p.center[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:centerY", float(p.center[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:radius", float(p.radius), Sdf.ValueTypeNames.Double)
            p.usd_path = child_path
            # Create edge (radius handle) sub-prim on the circumference (+X)
            edge_path = f"{child_path}/EdgePt"
            ep = UsdGeom.Xform.Define(stage, edge_path).GetPrim()
            ex, ey = p.edge_point
            self._set_attr(ep, f"{NS}:type", "point", Sdf.ValueTypeNames.String)
            self._set_attr(ep, f"{NS}:x", float(ex), Sdf.ValueTypeNames.Double)
            self._set_attr(ep, f"{NS}:y", float(ey), Sdf.ValueTypeNames.Double)
            p.edge_usd_path = edge_path

        elif isinstance(p, SketchArc):
            child_path = f"{prims_path}/Arc_{i:03d}"
            child_xf = UsdGeom.Xform.Define(stage, child_path)
            cp = child_xf.GetPrim()
            self._set_attr(cp, f"{NS}:type", "arc", Sdf.ValueTypeNames.String)
            self._set_attr(cp, f"{NS}:startX", float(p.start[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:startY", float(p.start[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:midX", float(p.mid[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:midY", float(p.mid[1]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:endX", float(p.end[0]), Sdf.ValueTypeNames.Double)
            self._set_attr(cp, f"{NS}:endY", float(p.end[1]), Sdf.ValueTypeNames.Double)
            p.usd_path = child_path
        else:
            return ""

        return child_path

    # -- Constraint prim CRUD --------------------------------------------------

    def write_constraint_prim(self, sketch_path: str, constraint) -> str:
        """
        Write a single constraint as a USD XForm prim under
        ``sketch_path/Constraints/C_xxx``.  Sets sparkworks: attributes
        for type, entity IDs, value, and selectors.  Updates
        ``constraint.usd_path`` and returns the prim path.
        """
        if not USD_AVAILABLE:
            return ""
        stage = self._get_stage()
        if stage is None:
            return ""

        self._ensure_sketch_xform(stage, sketch_path)

        constr_root = f"{sketch_path}/Constraints"
        UsdGeom.Xform.Define(stage, constr_root)

        cpath = f"{constr_root}/C_{constraint.cid:03d}"
        xf = UsdGeom.Xform.Define(stage, cpath)
        cp = xf.GetPrim()
        self._set_attr(cp, f"{NS}:type", "constraint", Sdf.ValueTypeNames.String)
        self._set_attr(cp, f"{NS}:constraintType", constraint.ctype.name, Sdf.ValueTypeNames.String)
        self._set_attr(cp, f"{NS}:cid", constraint.cid, Sdf.ValueTypeNames.Int)
        self._set_attr(cp, f"{NS}:value", float(constraint.value), Sdf.ValueTypeNames.Double)
        self._set_attr(cp, f"{NS}:entityIds",
                       Vt.IntArray(constraint.entity_ids), Sdf.ValueTypeNames.IntArray)
        self._set_attr(cp, f"{NS}:selectors",
                       Vt.StringArray(constraint.selectors), Sdf.ValueTypeNames.StringArray)
        self._set_attr(cp, f"{NS}:driving", constraint.driving, Sdf.ValueTypeNames.Bool)

        constraint.usd_path = cpath
        return cpath

    def remove_constraint_prim(self, prim_path: str):
        """Remove a constraint prim from the stage."""
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage and prim_path:
            stage.RemovePrim(prim_path)

    def load_constraint_prims(self, sketch_path: str):
        """
        Read constraint prims from ``sketch_path/Constraints/`` and return
        a list of dicts suitable for reconstructing constraints in the solver.

        Each dict has: cid, ctype (str), entity_ids, value, selectors, driving.
        """
        if not USD_AVAILABLE:
            return []
        stage = self._get_stage()
        if stage is None:
            return []

        constr_root = f"{sketch_path}/Constraints"
        root_prim = stage.GetPrimAtPath(constr_root)
        if not root_prim or not root_prim.IsValid():
            return []

        result = []
        for child in root_prim.GetChildren():
            tp = child.GetAttribute(f"{NS}:type")
            if not tp.IsValid() or tp.Get() != "constraint":
                continue
            ctype_attr = child.GetAttribute(f"{NS}:constraintType")
            cid_attr = child.GetAttribute(f"{NS}:cid")
            val_attr = child.GetAttribute(f"{NS}:value")
            eids_attr = child.GetAttribute(f"{NS}:entityIds")
            sels_attr = child.GetAttribute(f"{NS}:selectors")
            drv_attr = child.GetAttribute(f"{NS}:driving")
            if not ctype_attr.IsValid():
                continue
            d = {
                "cid": int(cid_attr.Get()) if cid_attr.IsValid() else 0,
                "ctype": str(ctype_attr.Get()),
                "entity_ids": list(eids_attr.Get()) if eids_attr.IsValid() else [],
                "value": float(val_attr.Get()) if val_attr.IsValid() else 0.0,
                "selectors": list(sels_attr.Get()) if sels_attr.IsValid() else [],
                "driving": bool(drv_attr.Get()) if drv_attr.IsValid() else True,
                "usd_path": str(child.GetPath()),
            }
            result.append(d)
        return result

    def write_all_constraint_prims(self, sketch_path: str, solver):
        """
        Write all constraints from *solver* to USD under
        ``sketch_path/Constraints/``.  Clears any existing constraint prims
        first to stay in sync.
        """
        if not USD_AVAILABLE or solver is None:
            return
        stage = self._get_stage()
        if stage is None:
            return

        constr_root = f"{sketch_path}/Constraints"
        root_prim = stage.GetPrimAtPath(constr_root)
        if root_prim and root_prim.IsValid():
            stage.RemovePrim(constr_root)
        UsdGeom.Xform.Define(stage, constr_root)

        for c in solver.constraints:
            self.write_constraint_prim(sketch_path, c)

    def update_point_position(self, point_prim_path: str, x: float, y: float):
        """
        Update a point prim's sparkworks:x/y attributes.

        Also updates the parent line's startX/Y or endX/Y, or the parent
        rectangle's center/width/height to stay in sync.
        """
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(point_prim_path)
        if not prim.IsValid():
            return
        self._set_attr(prim, f"{NS}:x", float(x), Sdf.ValueTypeNames.Double)
        self._set_attr(prim, f"{NS}:y", float(y), Sdf.ValueTypeNames.Double)
        # Sync to parent line attributes
        parent = prim.GetParent()
        if not parent.IsValid():
            return
        parent_type = self._get_attr(parent, f"{NS}:type")
        if parent_type == "line":
            pt_name = prim.GetName()
            if pt_name == "StartPt":
                self._set_attr(parent, f"{NS}:startX", float(x), Sdf.ValueTypeNames.Double)
                self._set_attr(parent, f"{NS}:startY", float(y), Sdf.ValueTypeNames.Double)
            elif pt_name == "EndPt":
                self._set_attr(parent, f"{NS}:endX", float(x), Sdf.ValueTypeNames.Double)
                self._set_attr(parent, f"{NS}:endY", float(y), Sdf.ValueTypeNames.Double)
        elif parent_type == "rectangle":
            self._sync_rect_from_corners(stage, parent)
        elif parent_type == "circle":
            pt_name = prim.GetName()
            if pt_name == "EdgePt":
                # Compute new radius from edge point position and center
                cx_val = self._get_attr(parent, f"{NS}:centerX")
                cy_val = self._get_attr(parent, f"{NS}:centerY")
                if cx_val is not None and cy_val is not None:
                    import math as _m
                    new_r = max(0.1, _m.hypot(x - float(cx_val), y - float(cy_val)))
                    self._set_attr(parent, f"{NS}:radius", new_r, Sdf.ValueTypeNames.Double)
        # If the prim itself is a circle (center is the circle prim), update centerX/Y
        prim_type = self._get_attr(prim, f"{NS}:type")
        if prim_type == "circle":
            self._set_attr(prim, f"{NS}:centerX", float(x), Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:centerY", float(y), Sdf.ValueTypeNames.Double)

    def update_circle_radius(self, circle_prim_path: str, radius: float):
        """Update a circle prim's sparkworks:radius attribute."""
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(circle_prim_path)
        if prim.IsValid():
            self._set_attr(prim, f"{NS}:radius", float(radius), Sdf.ValueTypeNames.Double)

    def _sync_rect_from_corners(self, stage, rect_prim):
        """Recompute rectangle center/width/height from its corner sub-prims."""
        corner_names = ["CornerBL", "CornerBR", "CornerTR", "CornerTL"]
        xs, ys = [], []
        for cname in corner_names:
            cpath = f"{rect_prim.GetPath()}/{cname}"
            cp = stage.GetPrimAtPath(cpath)
            if cp.IsValid():
                cx = self._get_attr(cp, f"{NS}:x")
                cy = self._get_attr(cp, f"{NS}:y")
                if cx is not None and cy is not None:
                    xs.append(float(cx))
                    ys.append(float(cy))
        if len(xs) == 4:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            self._set_attr(rect_prim, f"{NS}:centerX", (min_x + max_x) / 2.0, Sdf.ValueTypeNames.Double)
            self._set_attr(rect_prim, f"{NS}:centerY", (min_y + max_y) / 2.0, Sdf.ValueTypeNames.Double)
            self._set_attr(rect_prim, f"{NS}:width", max_x - min_x, Sdf.ValueTypeNames.Double)
            self._set_attr(rect_prim, f"{NS}:height", max_y - min_y, Sdf.ValueTypeNames.Double)

    def _write_operation_attrs(self, prim, operation):
        """Write operation-specific custom attributes."""
        from ..kernel.operations import (
            ExtrudeOperation, RevolveOperation,
            FilletOperation, ChamferOperation, BooleanOperation,
        )

        op_dict = operation.to_dict()
        op_type = op_dict.get("type", "unknown")
        self._set_attr(prim, f"{NS}:opType", op_type, Sdf.ValueTypeNames.String)

        if isinstance(operation, ExtrudeOperation):
            self._set_attr(prim, f"{NS}:distance", operation.distance, Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:symmetric", operation.symmetric, Sdf.ValueTypeNames.Bool)
            self._set_attr(prim, f"{NS}:both", operation.both, Sdf.ValueTypeNames.Bool)
            self._set_attr(prim, f"{NS}:negDistance", operation.neg_distance, Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:join", operation.join, Sdf.ValueTypeNames.Bool)
            self._set_attr(prim, f"{NS}:bodyName", operation.body_name, Sdf.ValueTypeNames.String)
            self._set_attr(prim, f"{NS}:joinBodyName", operation.join_body_name, Sdf.ValueTypeNames.String)
            self._set_attr(prim, f"{NS}:profileIndex", operation.profile_index, Sdf.ValueTypeNames.Int)
            if operation.profile_indices:
                self._set_attr(
                    prim, f"{NS}:profileIndices",
                    ",".join(str(i) for i in operation.profile_indices),
                    Sdf.ValueTypeNames.String,
                )

        elif isinstance(operation, RevolveOperation):
            self._set_attr(prim, f"{NS}:angle", operation.angle, Sdf.ValueTypeNames.Double)
            self._set_attr(prim, f"{NS}:axisName", operation.axis_name, Sdf.ValueTypeNames.String)

        elif isinstance(operation, FilletOperation):
            self._set_attr(prim, f"{NS}:radius", operation.radius, Sdf.ValueTypeNames.Double)
            if operation.edge_indices is not None:
                self._set_attr(prim, f"{NS}:edgeIndices",
                               Vt.IntArray(operation.edge_indices),
                               Sdf.ValueTypeNames.IntArray)

        elif isinstance(operation, ChamferOperation):
            self._set_attr(prim, f"{NS}:length", operation.length, Sdf.ValueTypeNames.Double)
            if operation.edge_indices is not None:
                self._set_attr(prim, f"{NS}:edgeIndices",
                               Vt.IntArray(operation.edge_indices),
                               Sdf.ValueTypeNames.IntArray)

        elif isinstance(operation, BooleanOperation):
            self._set_attr(prim, f"{NS}:mode", operation.mode, Sdf.ValueTypeNames.String)
            if operation.tool_feature_index is not None:
                self._set_attr(prim, f"{NS}:toolFeatureIndex",
                               operation.tool_feature_index, Sdf.ValueTypeNames.Int)

    # -- Feature read helpers ------------------------------------------------

    def _read_feature_prim(self, stage, prim) -> Optional[object]:
        """Read a single Feature from a USD Xform prim."""
        from ..timeline.timeline import Feature, FeatureType

        feat_type_str = self._get_attr(prim, f"{NS}:featureType")
        if feat_type_str is None:
            return None

        try:
            feat_type = FeatureType[feat_type_str]
        except KeyError:
            print(f"[SparkWorks] Unknown feature type: {feat_type_str}")
            return None

        name = self._get_attr(prim, f"{NS}:name") or prim.GetName()
        suppressed = self._get_attr(prim, f"{NS}:suppressed") or False
        created_at = self._get_attr(prim, f"{NS}:createdAt") or 0.0

        feature = Feature(
            name=name,
            feature_type=feat_type,
            suppressed=suppressed,
            created_at=created_at,
        )

        if feat_type == FeatureType.SKETCH:
            # Read the sketch_id reference (new architecture)
            sketch_id = self._get_attr(prim, f"{NS}:sketchId")
            if sketch_id:
                feature.sketch_id = sketch_id
            else:
                # Backward compat: infer sketch_id from the sketch name
                sketch_name = self._get_attr(prim, f"{NS}:sketchName") or name
                feature.sketch_id = sketch_name
        elif feat_type == FeatureType.OPERATION:
            feature.operation = self._read_operation_from_prim(prim)

        return feature

    def _read_sketch_from_prim(self, stage, prim):
        """Read a Sketch from a feature prim's attributes and children."""
        from ..kernel.sketch import Sketch, SketchLine, SketchRect, SketchCircle, SketchArc
        from ..kernel.construction_plane import ConstructionPlane

        plane_name = self._get_attr(prim, f"{NS}:planeName") or "XY"
        sketch_name = self._get_attr(prim, f"{NS}:sketchName") or self._get_attr(prim, f"{NS}:name") or "Sketch"

        sketch = Sketch(name=sketch_name, plane_name=plane_name)

        # Restore custom plane geometry if it was saved
        plane_type = self._get_attr(prim, f"{NS}:planeType")
        if plane_type is not None:
            # Use explicit None checks — 0.0 is a valid coordinate value
            # and must NOT be replaced by a default (Python `or` treats 0.0 as falsy)
            ox = self._get_attr(prim, f"{NS}:planeOriginX")
            oy = self._get_attr(prim, f"{NS}:planeOriginY")
            oz = self._get_attr(prim, f"{NS}:planeOriginZ")
            origin = (
                ox if ox is not None else 0.0,
                oy if oy is not None else 0.0,
                oz if oz is not None else 0.0,
            )
            nnx = self._get_attr(prim, f"{NS}:planeNormalX")
            nny = self._get_attr(prim, f"{NS}:planeNormalY")
            nnz = self._get_attr(prim, f"{NS}:planeNormalZ")
            normal = (
                nnx if nnx is not None else 0.0,
                nny if nny is not None else 0.0,
                nnz if nnz is not None else 1.0,
            )
            sketch.construction_plane = ConstructionPlane(
                name=plane_name,
                plane_type=plane_type,
                origin=origin,
                normal=normal,
            )
            print(f"[SparkWorks] Restored custom plane for '{sketch_name}': "
                  f"origin={origin} normal={normal}")

        # Read primitives from the Primitives/ child
        prim_path = prim.GetPath().AppendChild("Primitives")
        prims_prim = stage.GetPrimAtPath(prim_path)
        if not prims_prim.IsValid():
            return sketch

        children = list(prims_prim.GetChildren())
        children.sort(key=lambda p: p.GetName())

        for child in children:
            prim_type = self._get_attr(child, f"{NS}:type")
            if prim_type is None:
                continue

            if prim_type == "line":
                # Prefer reading from StartPt/EndPt children if present
                child_path = child.GetPath()
                start_pt = stage.GetPrimAtPath(f"{child_path}/StartPt")
                end_pt = stage.GetPrimAtPath(f"{child_path}/EndPt")
                if start_pt.IsValid() and end_pt.IsValid():
                    sx = self._get_attr(start_pt, f"{NS}:x")
                    sy = self._get_attr(start_pt, f"{NS}:y")
                    ex = self._get_attr(end_pt, f"{NS}:x")
                    ey = self._get_attr(end_pt, f"{NS}:y")
                    start = (sx if sx is not None else 0.0, sy if sy is not None else 0.0)
                    end = (ex if ex is not None else 0.0, ey if ey is not None else 0.0)
                else:
                    start = (
                        self._get_attr(child, f"{NS}:startX") or 0.0,
                        self._get_attr(child, f"{NS}:startY") or 0.0,
                    )
                    end = (
                        self._get_attr(child, f"{NS}:endX") or 0.0,
                        self._get_attr(child, f"{NS}:endY") or 0.0,
                    )
                line = SketchLine(start=start, end=end)
                line.usd_path = str(child_path)
                if start_pt.IsValid():
                    line.start_usd_path = str(child_path) + "/StartPt"
                if end_pt.IsValid():
                    line.end_usd_path = str(child_path) + "/EndPt"
                sketch.primitives.append(line)
            elif prim_type == "rectangle":
                child_path = child.GetPath()
                rect = SketchRect(
                    center=(
                        self._get_attr(child, f"{NS}:centerX") or 0.0,
                        self._get_attr(child, f"{NS}:centerY") or 0.0,
                    ),
                    width=self._get_attr(child, f"{NS}:width") or 1.0,
                    height=self._get_attr(child, f"{NS}:height") or 1.0,
                )
                rect.usd_path = str(child_path)
                # Restore corner USD paths
                corner_names = ["CornerBL", "CornerBR", "CornerTR", "CornerTL"]
                corner_paths = []
                for cname in corner_names:
                    cpath = f"{child_path}/{cname}"
                    cp = stage.GetPrimAtPath(cpath)
                    if cp.IsValid():
                        corner_paths.append(str(cpath))
                    else:
                        corner_paths.append(None)
                rect.corner_usd_paths = corner_paths
                sketch.primitives.append(rect)
            elif prim_type == "circle":
                circ = SketchCircle(
                    center=(
                        self._get_attr(child, f"{NS}:centerX") or 0.0,
                        self._get_attr(child, f"{NS}:centerY") or 0.0,
                    ),
                    radius=self._get_attr(child, f"{NS}:radius") or 1.0,
                )
                circ.usd_path = str(child.GetPath())
                # Load edge (radius handle) sub-prim if present
                edge_path = f"{child.GetPath()}/EdgePt"
                edge_prim = stage.GetPrimAtPath(edge_path)
                if edge_prim and edge_prim.IsValid():
                    circ.edge_usd_path = edge_path
                sketch.primitives.append(circ)
            elif prim_type == "arc":
                arc = SketchArc(
                    start=(
                        self._get_attr(child, f"{NS}:startX") or 0.0,
                        self._get_attr(child, f"{NS}:startY") or 0.0,
                    ),
                    mid=(
                        self._get_attr(child, f"{NS}:midX") or 0.0,
                        self._get_attr(child, f"{NS}:midY") or 0.0,
                    ),
                    end=(
                        self._get_attr(child, f"{NS}:endX") or 0.0,
                        self._get_attr(child, f"{NS}:endY") or 0.0,
                    ),
                )
                arc.usd_path = str(child.GetPath())
                sketch.primitives.append(arc)

        # Load constraints from USD and restore them into the solver
        sketch_path = str(prim.GetPath())
        constraint_dicts = self.load_constraint_prims(sketch_path)
        if constraint_dicts:
            from ..kernel.constraint_solver import ConstraintType
            solver = sketch.ensure_solver()
            for cd in constraint_dicts:
                try:
                    ctype = ConstraintType[cd["ctype"]]
                except KeyError:
                    continue
                cid = solver.add_constraint(
                    ctype,
                    entity_ids=cd["entity_ids"],
                    value=cd["value"],
                    selectors=cd["selectors"],
                    driving=cd["driving"],
                )
                # Restore the USD path on the constraint object
                for c in solver.constraints:
                    if c.cid == cid:
                        c.usd_path = cd.get("usd_path")
                        break

        return sketch

    def _read_operation_from_prim(self, prim):
        """Read an operation from a feature prim's custom attributes."""
        from ..kernel.operations import (
            ExtrudeOperation, RevolveOperation,
            FilletOperation, ChamferOperation, BooleanOperation,
        )

        def _v(attr_name, default):
            """Read an attribute with a proper None-check (0.0 / False are valid)."""
            val = self._get_attr(prim, attr_name)
            return val if val is not None else default

        op_type = self._get_attr(prim, f"{NS}:opType")
        if op_type is None:
            return None

        name = _v(f"{NS}:name", "")
        suppressed = _v(f"{NS}:suppressed", False)

        if op_type == "extrude":
            # Read profile_indices (comma-separated string, or fall back to single index)
            pi_str = _v(f"{NS}:profileIndices", "")
            if pi_str:
                profile_indices = [int(x) for x in pi_str.split(",") if x.strip()]
            else:
                profile_indices = []
            op = ExtrudeOperation(
                distance=_v(f"{NS}:distance", 10.0),
                symmetric=_v(f"{NS}:symmetric", False),
                both=_v(f"{NS}:both", False),
                neg_distance=_v(f"{NS}:negDistance", 0.0),
                join=_v(f"{NS}:join", False),
                body_name=_v(f"{NS}:bodyName", ""),
                join_body_name=_v(f"{NS}:joinBodyName", ""),
                profile_index=_v(f"{NS}:profileIndex", 0),
                profile_indices=profile_indices,
            )
        elif op_type == "revolve":
            op = RevolveOperation(
                angle=_v(f"{NS}:angle", 360.0),
                axis_name=_v(f"{NS}:axisName", "Z"),
            )
        elif op_type == "fillet":
            edge_indices = self._get_attr(prim, f"{NS}:edgeIndices")
            op = FilletOperation(
                radius=_v(f"{NS}:radius", 1.0),
                edge_indices=list(edge_indices) if edge_indices else None,
            )
        elif op_type == "chamfer":
            edge_indices = self._get_attr(prim, f"{NS}:edgeIndices")
            op = ChamferOperation(
                length=_v(f"{NS}:length", 1.0),
                edge_indices=list(edge_indices) if edge_indices else None,
            )
        elif op_type == "boolean":
            tool_idx = self._get_attr(prim, f"{NS}:toolFeatureIndex")
            op = BooleanOperation(
                mode=_v(f"{NS}:mode", "join"),
                tool_feature_index=tool_idx,
            )
        else:
            print(f"[SparkWorks] Unknown operation type on USD: {op_type}")
            return None

        op.name = name
        op.suppressed = suppressed
        return op

    # -- USD attribute helpers -----------------------------------------------

    @staticmethod
    def _set_attr(prim, attr_name: str, value, value_type):
        """Create (or update) a custom attribute on a prim."""
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, value_type)
        attr.Set(value)

    @staticmethod
    def _get_attr(prim, attr_name: str):
        """Read a custom attribute value from a prim, or None if missing."""
        attr = prim.GetAttribute(attr_name)
        if attr.IsValid():
            return attr.Get()
        return None

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
