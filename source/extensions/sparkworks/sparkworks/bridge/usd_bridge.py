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
DEFAULT_DISPLAY_COLOR = (0.7, 0.7, 0.8)  # Light steel blue

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

    def is_body_prim(self, prim_path: str) -> Optional[str]:
        """
        Check whether a USD prim path belongs to a body mesh.

        Returns:
            The body name (e.g. "Body1") if *prim_path* is under
            the Bodies scope, otherwise ``None``.
        """
        prefix = self.bodies_root + "/"
        if prim_path.startswith(prefix):
            remainder = prim_path[len(prefix):]
            return remainder.split("/")[0]
        return None

    def remove_face_planes(self):
        """Remove face-plane visualizations (planes whose name contains '_Face')."""
        if not USD_AVAILABLE:
            return
        stage = self._get_stage()
        if stage is None:
            return
        constr_prim = stage.GetPrimAtPath(self.construction_root)
        if not constr_prim.IsValid():
            return
        to_remove = []
        for child in constr_prim.GetChildren():
            name = child.GetName()
            if "_Face" in name:
                to_remove.append(child.GetPath().pathString)
        for path in to_remove:
            stage.RemovePrim(path)

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

        if feature.feature_type == FeatureType.SKETCH and feature.sketch:
            self._write_sketch_attrs(stage, prim_path, prim, feature.sketch)
        elif feature.feature_type == FeatureType.OPERATION and feature.operation:
            self._write_operation_attrs(prim, feature.operation)

    def _write_sketch_attrs(self, stage, prim_path: str, prim, sketch):
        """Write sketch-specific attributes and primitive children."""
        self._set_attr(prim, f"{NS}:planeName", sketch.plane_name, Sdf.ValueTypeNames.String)
        self._set_attr(prim, f"{NS}:sketchName", sketch.name, Sdf.ValueTypeNames.String)

        # Create Primitives Xform
        prims_path = f"{prim_path}/Primitives"
        UsdGeom.Xform.Define(stage, prims_path)

        from ..kernel.sketch import SketchLine, SketchRect, SketchCircle, SketchArc

        for i, p in enumerate(sketch.primitives):
            if isinstance(p, SketchLine):
                child_path = f"{prims_path}/Line_{i:03d}"
                child_xf = UsdGeom.Xform.Define(stage, child_path)
                cp = child_xf.GetPrim()
                self._set_attr(cp, f"{NS}:type", "line", Sdf.ValueTypeNames.String)
                self._set_attr(cp, f"{NS}:startX", float(p.start[0]), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:startY", float(p.start[1]), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:endX", float(p.end[0]), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:endY", float(p.end[1]), Sdf.ValueTypeNames.Double)

            elif isinstance(p, SketchRect):
                child_path = f"{prims_path}/Rect_{i:03d}"
                child_xf = UsdGeom.Xform.Define(stage, child_path)
                cp = child_xf.GetPrim()
                self._set_attr(cp, f"{NS}:type", "rectangle", Sdf.ValueTypeNames.String)
                self._set_attr(cp, f"{NS}:centerX", float(p.center[0]), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:centerY", float(p.center[1]), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:width", float(p.width), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:height", float(p.height), Sdf.ValueTypeNames.Double)

            elif isinstance(p, SketchCircle):
                child_path = f"{prims_path}/Circle_{i:03d}"
                child_xf = UsdGeom.Xform.Define(stage, child_path)
                cp = child_xf.GetPrim()
                self._set_attr(cp, f"{NS}:type", "circle", Sdf.ValueTypeNames.String)
                self._set_attr(cp, f"{NS}:centerX", float(p.center[0]), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:centerY", float(p.center[1]), Sdf.ValueTypeNames.Double)
                self._set_attr(cp, f"{NS}:radius", float(p.radius), Sdf.ValueTypeNames.Double)

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
            feature.sketch = self._read_sketch_from_prim(stage, prim)
        elif feat_type == FeatureType.OPERATION:
            feature.operation = self._read_operation_from_prim(prim)

        return feature

    def _read_sketch_from_prim(self, stage, prim):
        """Read a Sketch from a feature prim's attributes and children."""
        from ..kernel.sketch import Sketch, SketchLine, SketchRect, SketchCircle, SketchArc

        plane_name = self._get_attr(prim, f"{NS}:planeName") or "XY"
        sketch_name = self._get_attr(prim, f"{NS}:sketchName") or self._get_attr(prim, f"{NS}:name") or "Sketch"

        sketch = Sketch(name=sketch_name, plane_name=plane_name)

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
                sketch.primitives.append(SketchLine(
                    start=(
                        self._get_attr(child, f"{NS}:startX") or 0.0,
                        self._get_attr(child, f"{NS}:startY") or 0.0,
                    ),
                    end=(
                        self._get_attr(child, f"{NS}:endX") or 0.0,
                        self._get_attr(child, f"{NS}:endY") or 0.0,
                    ),
                ))
            elif prim_type == "rectangle":
                sketch.primitives.append(SketchRect(
                    center=(
                        self._get_attr(child, f"{NS}:centerX") or 0.0,
                        self._get_attr(child, f"{NS}:centerY") or 0.0,
                    ),
                    width=self._get_attr(child, f"{NS}:width") or 1.0,
                    height=self._get_attr(child, f"{NS}:height") or 1.0,
                ))
            elif prim_type == "circle":
                sketch.primitives.append(SketchCircle(
                    center=(
                        self._get_attr(child, f"{NS}:centerX") or 0.0,
                        self._get_attr(child, f"{NS}:centerY") or 0.0,
                    ),
                    radius=self._get_attr(child, f"{NS}:radius") or 1.0,
                ))
            elif prim_type == "arc":
                sketch.primitives.append(SketchArc(
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
                ))

        return sketch

    def _read_operation_from_prim(self, prim):
        """Read an operation from a feature prim's custom attributes."""
        from ..kernel.operations import (
            ExtrudeOperation, RevolveOperation,
            FilletOperation, ChamferOperation, BooleanOperation,
        )

        op_type = self._get_attr(prim, f"{NS}:opType")
        if op_type is None:
            return None

        name = self._get_attr(prim, f"{NS}:name") or ""
        suppressed = self._get_attr(prim, f"{NS}:suppressed") or False

        if op_type == "extrude":
            op = ExtrudeOperation(
                distance=self._get_attr(prim, f"{NS}:distance") or 10.0,
                symmetric=self._get_attr(prim, f"{NS}:symmetric") or False,
                both=self._get_attr(prim, f"{NS}:both") or False,
                neg_distance=self._get_attr(prim, f"{NS}:negDistance") or 0.0,
                join=self._get_attr(prim, f"{NS}:join") or False,
            )
        elif op_type == "revolve":
            op = RevolveOperation(
                angle=self._get_attr(prim, f"{NS}:angle") or 360.0,
                axis_name=self._get_attr(prim, f"{NS}:axisName") or "Z",
            )
        elif op_type == "fillet":
            edge_indices = self._get_attr(prim, f"{NS}:edgeIndices")
            op = FilletOperation(
                radius=self._get_attr(prim, f"{NS}:radius") or 1.0,
                edge_indices=list(edge_indices) if edge_indices else None,
            )
        elif op_type == "chamfer":
            edge_indices = self._get_attr(prim, f"{NS}:edgeIndices")
            op = ChamferOperation(
                length=self._get_attr(prim, f"{NS}:length") or 1.0,
                edge_indices=list(edge_indices) if edge_indices else None,
            )
        elif op_type == "boolean":
            tool_idx = self._get_attr(prim, f"{NS}:toolFeatureIndex")
            op = BooleanOperation(
                mode=self._get_attr(prim, f"{NS}:mode") or "join",
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
