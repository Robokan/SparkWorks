"""
SparkWorks Programmatic API — headless facade for all modelling operations.

This module provides a ``SparkWorksAPI`` class that wraps the Timeline,
SketchRegistry, UsdBridge, and all CAD operations in a single, UI-free
interface.  Use it for:

- **Unit / integration tests** — drive operations and assert on state
  without any UI or viewport dependencies.
- **Scripting / automation** — build parametric models from Python code.
- **Visual regression tests** — combine with :func:`capture_viewport` and
  :func:`set_camera` to take screenshots and compare against golden images.

Example::

    from sparkworks.api import SparkWorksAPI

    api = SparkWorksAPI()
    sk = api.create_sketch("XY")
    api.add_rectangle(sk, 20, 10)
    api.finish_sketch(sk)
    body = api.extrude(distance=5)
    assert body == "Body1"
    assert len(api.bodies) == 1
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .kernel.sketch import Sketch
from .kernel.construction_plane import (
    ConstructionPlane,
    create_origin_planes,
    extract_face_planes,
    get_planar_face,
)
from .kernel.operations import (
    ExtrudeOperation,
    RevolveOperation,
    FilletOperation,
    ChamferOperation,
)
from .timeline.timeline import Timeline, Feature, FeatureType
from .timeline.sketch_registry import SketchRegistry


class SparkWorksAPI:
    """
    Headless programmatic API for all SparkWorks modelling operations.

    Manages a :class:`Timeline`, :class:`SketchRegistry`, and optionally a
    :class:`UsdBridge` (when running inside Omniverse Kit with a live stage).

    Parameters:
        use_bridge: If ``True`` (default), initialise a ``UsdBridge`` to
            read / write USD prims.  Set to ``False`` for pure in-memory
            tests that don't need a USD stage.
    """

    def __init__(self, use_bridge: bool = True):
        self._registry = SketchRegistry()
        self._timeline = Timeline(sketch_registry=self._registry)

        self._bridge = None
        if use_bridge:
            try:
                from .bridge import UsdBridge
                self._bridge = UsdBridge()
            except Exception:
                pass

        # Counters (mirroring extension.py bookkeeping)
        self._feature_counter: int = 0
        self._body_counter: int = 0
        self._marker_position: int = -1  # -1 = end

        # Connect rebuild callback so bodies dict stays in sync
        self._timeline.on_rebuild = self._on_rebuild
        self._last_rebuild_bodies: Dict[str, Any] = {}

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _on_rebuild(self, solid, bodies=None):
        """Callback from Timeline — cache the bodies dict."""
        self._last_rebuild_bodies = dict(bodies) if bodies else {}
        # Write meshes to USD if a bridge is available
        if self._bridge is not None:
            try:
                # Clear stale bodies
                existing = self._bridge.get_body_names()
                for bn in existing:
                    self._bridge.update_body_mesh(None, body_name=bn)
                # Write current bodies
                for bname, bsolid in self._last_rebuild_bodies.items():
                    if bsolid is not None:
                        self._bridge.update_body_mesh(bsolid, body_name=bname)
            except Exception as exc:
                print(f"[SparkWorksAPI] Bridge mesh update error: {exc}")

    def _next_feature_name(self, prefix: str) -> str:
        self._feature_counter += 1
        return f"{prefix}{self._feature_counter}"

    def _effective_marker(self) -> Optional[int]:
        """Return the marker position for insert_after, or None for append."""
        if self._marker_position < 0:
            return None
        return self._marker_position

    # =====================================================================
    # Sketch Operations
    # =====================================================================

    def create_sketch(self, plane_name: str = "XY") -> Sketch:
        """
        Create a new empty sketch on a standard plane.

        Args:
            plane_name: One of ``"XY"``, ``"XZ"``, ``"YZ"``.

        Returns:
            The new :class:`Sketch` object.  Add primitives with
            :meth:`add_line`, :meth:`add_rectangle`, etc., then call
            :meth:`finish_sketch` to commit it to the timeline.
        """
        next_num = self._registry.counter + 1
        name = f"Sketch{next_num}"
        return Sketch(name=name, plane_name=plane_name)

    def create_sketch_on_plane(self, plane: ConstructionPlane) -> Sketch:
        """
        Create a new empty sketch on a custom :class:`ConstructionPlane`.

        Returns:
            The new :class:`Sketch` object.
        """
        next_num = self._registry.counter + 1
        name = f"Sketch{next_num}"
        return Sketch(
            name=name,
            plane_name=plane.plane_type,
            construction_plane=plane,
        )

    # -- Adding primitives to an open sketch --------------------------------

    def add_line(
        self, sketch: Sketch,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ):
        """Add a line segment to *sketch*."""
        sketch.add_line(start=start, end=end)

    def add_rectangle(
        self, sketch: Sketch,
        width: float, height: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ):
        """Add an axis-aligned rectangle to *sketch*."""
        sketch.add_rectangle(width=width, height=height, center=center)

    def add_circle(
        self, sketch: Sketch,
        radius: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ):
        """Add a circle to *sketch*."""
        sketch.add_circle(radius=radius, center=center)

    def add_arc(
        self, sketch: Sketch,
        start: Tuple[float, float],
        mid: Tuple[float, float],
        end: Tuple[float, float],
    ):
        """Add a three-point arc to *sketch*."""
        sketch.add_arc(start=start, mid=mid, end=end)

    def finish_sketch(self, sketch: Sketch) -> int:
        """
        Commit an open sketch to the timeline.

        Registers the sketch in the :class:`SketchRegistry` and inserts a
        SKETCH feature into the timeline at the current marker position.

        Args:
            sketch: The sketch to commit.

        Returns:
            The feature index of the newly inserted sketch.
        """
        marker = self._effective_marker()
        idx = self._timeline.add_sketch(
            sketch,
            insert_after=marker,
            sketch_id=sketch.name,
        )

        # Persist to USD if bridge is available
        self._save_to_usd()

        # Move marker past the new sketch
        new_count = self._timeline.feature_count
        if marker is not None:
            self._marker_position = marker + 1
        else:
            self._marker_position = new_count - 1

        return self._marker_position

    # =====================================================================
    # 3D Operations
    # =====================================================================

    def extrude(
        self,
        distance: float = 10.0,
        profile_indices: Optional[List[int]] = None,
        body_name: Optional[str] = None,
        join_body: Optional[str] = None,
        symmetric: bool = False,
    ) -> str:
        """
        Extrude the current sketch profile(s) into a 3D solid.

        Args:
            distance: Extrusion distance.
            profile_indices: Which profile(s) to extrude (default ``[0]``).
            body_name: Explicit body name.  Auto-generated if ``None``.
            join_body: If set, fuse with this existing body.
            symmetric: Extrude symmetrically (distance/2 each way).

        Returns:
            The body name (e.g. ``"Body1"``).
        """
        pis = profile_indices or [0]
        is_join = join_body is not None

        if body_name is None:
            if is_join:
                body_name = join_body
            else:
                self._body_counter += 1
                body_name = f"Body{self._body_counter}"

        op = ExtrudeOperation(
            distance=distance,
            symmetric=symmetric,
            join=is_join,
            body_name=body_name,
            join_body_name=join_body or "",
            profile_index=pis[0],
            profile_indices=pis,
        )
        op.name = self._next_feature_name("Extrude")

        marker = self._effective_marker()
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._save_to_usd()

        # Advance marker
        if marker is not None:
            self._marker_position = marker + 1
        else:
            self._marker_position = self._timeline.feature_count - 1

        return body_name

    def extrude_face(
        self,
        face_body: str,
        face_index: int,
        distance: float = 10.0,
    ) -> str:
        """
        Extrude a body face directly (Press/Pull), fusing the result
        back onto the source body.

        Args:
            face_body: Name of the body whose face to extrude.
            face_index: Index of the planar face on the body.
            distance: Extrusion distance.

        Returns:
            The body name (always the source body — result is fused).
        """
        op = ExtrudeOperation(
            distance=distance,
            join=True,
            body_name=face_body,
            join_body_name=face_body,
            face_body_name=face_body,
            face_index=face_index,
        )
        op.name = self._next_feature_name("Extrude")

        marker = self._effective_marker()
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._save_to_usd()

        if marker is not None:
            self._marker_position = marker + 1
        else:
            self._marker_position = self._timeline.feature_count - 1

        return face_body

    def revolve(
        self,
        angle: float = 360.0,
        axis: str = "Z",
    ) -> None:
        """
        Revolve the current sketch profile around an axis.

        Args:
            angle: Revolution angle in degrees.
            axis: ``"X"``, ``"Y"``, or ``"Z"``.
        """
        op = RevolveOperation(angle=angle, axis_name=axis)
        op.name = self._next_feature_name("Revolve")

        marker = self._effective_marker()
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._save_to_usd()

        if marker is not None:
            self._marker_position = marker + 1
        else:
            self._marker_position = self._timeline.feature_count - 1

    def fillet(
        self,
        radius: float = 1.0,
        edge_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Apply a fillet to edges of the current solid.

        Args:
            radius: Fillet radius.
            edge_indices: Specific edges, or ``None`` for all.
        """
        op = FilletOperation(radius=radius, edge_indices=edge_indices)
        op.name = self._next_feature_name("Fillet")

        marker = self._effective_marker()
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._save_to_usd()

        if marker is not None:
            self._marker_position = marker + 1
        else:
            self._marker_position = self._timeline.feature_count - 1

    def chamfer(
        self,
        length: float = 1.0,
        edge_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Apply a chamfer to edges of the current solid.

        Args:
            length: Chamfer length.
            edge_indices: Specific edges, or ``None`` for all.
        """
        op = ChamferOperation(length=length, edge_indices=edge_indices)
        op.name = self._next_feature_name("Chamfer")

        marker = self._effective_marker()
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._save_to_usd()

        if marker is not None:
            self._marker_position = marker + 1
        else:
            self._marker_position = self._timeline.feature_count - 1

    # =====================================================================
    # Timeline Control
    # =====================================================================

    def scrub_to(self, index: int):
        """
        Scrub the timeline to show the model state after feature *index*.

        Args:
            index: Feature index (0-based).  Use ``-1`` to scrub before
                all features (empty model).
        """
        self._marker_position = index
        if index < 0:
            self._timeline._scrub_index = -1
            self._timeline._current_solid = None
            self._timeline._bodies = {}
            self._timeline._current_sketch_face = None
            self._timeline._notify_rebuild()
        elif index >= self._timeline.feature_count - 1:
            self._timeline.scrub_to_end()
        else:
            self._timeline.scrub_to(index)

    def scrub_to_end(self):
        """Move the marker to the end of the timeline (show latest state)."""
        self._marker_position = self._timeline.feature_count - 1
        self._timeline.scrub_to_end()

    def scrub_to_start(self):
        """Move the marker before all features (empty model)."""
        self.scrub_to(-1)

    def move_marker(self, position: int):
        """
        Move the insertion marker to *position*.

        New operations will be inserted after this position.
        ``-1`` means before all features (insert at the beginning).
        """
        self._marker_position = position

    def delete_feature(self, index: int):
        """Remove a feature from the timeline by index."""
        if 0 <= index < self._timeline.feature_count:
            self._timeline.remove_feature(index)
            self._save_to_usd()

    def suppress_feature(self, index: int, suppressed: bool = True):
        """Suppress or unsuppress a feature."""
        self._timeline.suppress_feature(index, suppressed)
        self._save_to_usd()

    def update_param(self, feature_index: int, param_name: str, value):
        """Update a parameter on a feature and trigger a rebuild."""
        self._timeline.update_feature_params(feature_index, {param_name: value})
        self._save_to_usd()

    def rebuild_all(self):
        """Force a full rebuild from scratch."""
        self._timeline.rebuild_all()

    def clear_all(self):
        """Reset everything — clear the timeline, sketches, and bodies."""
        self._registry = SketchRegistry()
        self._timeline = Timeline(sketch_registry=self._registry)
        self._timeline.on_rebuild = self._on_rebuild
        self._feature_counter = 0
        self._body_counter = 0
        self._marker_position = -1
        self._last_rebuild_bodies = {}
        if self._bridge is not None:
            self._bridge.clear()
        self._save_to_usd()

    # =====================================================================
    # Queries
    # =====================================================================

    @property
    def features(self) -> List[Feature]:
        """All features in the timeline."""
        return self._timeline.features

    @property
    def feature_count(self) -> int:
        """Number of features in the timeline."""
        return self._timeline.feature_count

    @property
    def bodies(self) -> Dict[str, Any]:
        """Dict of ``body_name -> solid`` at the current rebuild state."""
        return self._timeline.bodies

    @property
    def current_solid(self):
        """The last solid produced by the timeline (or ``None``)."""
        return self._timeline.current_solid

    @property
    def marker_position(self) -> int:
        """Current insertion marker position."""
        return self._marker_position

    @property
    def sketch_registry(self) -> SketchRegistry:
        """The sketch registry."""
        return self._registry

    @property
    def timeline(self) -> Timeline:
        """Direct access to the underlying Timeline (for advanced use)."""
        return self._timeline

    @property
    def bridge(self):
        """The UsdBridge, or ``None`` if running without USD."""
        return self._bridge

    def get_sketch(self, name: str) -> Optional[Sketch]:
        """Look up a sketch by name."""
        return self._registry.get(name)

    def get_body_solid(self, body_name: str):
        """Get the build123d solid for a named body, or ``None``."""
        return self._timeline.bodies.get(body_name)

    def get_body_faces(self, body_name: str) -> List[ConstructionPlane]:
        """
        Extract all planar faces from a body as ConstructionPlane objects.

        Useful for determining available face indices for
        :meth:`extrude_face`.
        """
        solid = self._timeline.bodies.get(body_name)
        if solid is None:
            return []
        return extract_face_planes(solid, body_name=body_name)

    def get_planar_face(self, body_name: str, face_index: int):
        """
        Get a specific planar face from a body.

        Args:
            body_name: The body name.
            face_index: Index among planar faces.

        Returns:
            A ``build123d.Face`` or ``None``.
        """
        solid = self._timeline.bodies.get(body_name)
        return get_planar_face(solid, face_index)

    def get_profiles(self, sketch: Sketch) -> list:
        """
        Build and return all closed-loop profiles from a sketch.

        Returns:
            A list of ``build123d.Face`` objects.
        """
        return sketch.build_all_faces()

    # -- USD prim queries (require bridge) ----------------------------------

    def get_body_prim_path(self, body_name: str) -> Optional[str]:
        """Return the USD prim path for a body, or ``None``."""
        if self._bridge is None:
            return None
        return f"{self._bridge.bodies_root}/{body_name}"

    def get_all_sparkworks_prims(self) -> List[str]:
        """
        List all USD prim paths under ``/World/SparkWorks``.

        Requires a USD stage.
        """
        if self._bridge is None:
            return []
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return []
            root = stage.GetPrimAtPath(self._bridge.root_path)
            if not root.IsValid():
                return []
            paths = []
            for prim in root.GetAllChildren():
                paths.append(str(prim.GetPath()))
            return paths
        except Exception:
            return []

    def get_body_names(self) -> List[str]:
        """
        List all body names currently on the USD stage.

        Falls back to the in-memory bodies dict if no bridge is available.
        """
        if self._bridge is not None:
            try:
                return self._bridge.get_body_names()
            except Exception:
                pass
        return list(self._timeline.bodies.keys())

    # =====================================================================
    # Persistence
    # =====================================================================

    def _save_to_usd(self):
        """Persist sketches + timeline to USD (no-op if no bridge)."""
        if self._bridge is None:
            return
        try:
            self._bridge.save_sketches(self._registry)
        except Exception as exc:
            print(f"[SparkWorksAPI] save_sketches error: {exc}")
        try:
            self._bridge.save_timeline(self._timeline.features)
        except Exception as exc:
            print(f"[SparkWorksAPI] save_timeline error: {exc}")

    def save(self):
        """Explicitly save all SparkWorks data to USD."""
        self._save_to_usd()

    def load(self) -> bool:
        """
        Load SparkWorks data (sketches + timeline) from the current USD stage.

        Returns:
            ``True`` if data was loaded, ``False`` otherwise.
        """
        if self._bridge is None:
            return False
        try:
            loaded_registry = self._bridge.load_sketches()
            if loaded_registry is not None:
                self._registry = loaded_registry
                self._timeline.sketch_registry = loaded_registry

            features = self._bridge.load_timeline()
            if features:
                self._timeline = Timeline(sketch_registry=self._registry)
                self._timeline.on_rebuild = self._on_rebuild
                for feature in features:
                    feature.bind_registry(self._registry)
                    self._timeline._features.append(feature)
                self._timeline.rebuild_all()
                self._feature_counter = len(features)
                # Sync body counter
                for bname in self._timeline.bodies:
                    try:
                        num = int(bname.replace("Body", ""))
                        self._body_counter = max(self._body_counter, num)
                    except ValueError:
                        pass
                self._marker_position = self._timeline.feature_count - 1
                return True
        except Exception as exc:
            print(f"[SparkWorksAPI] load error: {exc}")
        return False


# =========================================================================
# Viewport Utilities (for visual regression tests)
# =========================================================================

def set_camera(
    position: Tuple[float, float, float],
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    camera_path: str = "/OmniverseKit_Persp",
):
    """
    Set the active viewport camera position and look-at target.

    Args:
        position: Camera eye position ``(x, y, z)``.
        target: Look-at target ``(x, y, z)``.
        camera_path: USD prim path of the camera.
    """
    try:
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[SparkWorksAPI] No stage — cannot set camera")
            return

        cam_prim = stage.GetPrimAtPath(camera_path)
        if not cam_prim.IsValid():
            print(f"[SparkWorksAPI] Camera prim '{camera_path}' not found")
            return

        xformable = UsdGeom.Xformable(cam_prim)
        # Clear existing transforms
        xformable.ClearXformOpOrder()

        # Build a look-at matrix
        eye = Gf.Vec3d(*position)
        center = Gf.Vec3d(*target)
        up = Gf.Vec3d(0, 1, 0)
        view_matrix = Gf.Matrix4d()
        view_matrix.SetLookAt(eye, center, up)
        # SetLookAt returns the view matrix — we need the inverse for the
        # camera transform (world-space position of the camera).
        cam_transform = view_matrix.GetInverse()

        xform_op = xformable.AddTransformOp()
        xform_op.Set(cam_transform)

        print(f"[SparkWorksAPI] Camera set: eye={position}, target={target}")
    except Exception as exc:
        print(f"[SparkWorksAPI] set_camera error: {exc}")


async def capture_viewport(
    output_path: str,
    wait_frames: int = 5,
) -> bool:
    """
    Capture the active viewport to an image file.

    Must be called from an async context (e.g. inside an
    ``omni.kit.test.AsyncTestCase``).

    Args:
        output_path: File path to save the screenshot (PNG).
        wait_frames: Number of frames to wait for rendering to settle.

    Returns:
        ``True`` if the capture succeeded.
    """
    try:
        import omni.kit.app

        # Let the renderer settle
        for _ in range(wait_frames):
            await omni.kit.app.get_app().next_update_async()

        # Try the viewport utility API first (Kit 105+)
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None:
                import omni.kit.viewport.utility as vp_utils
                await vp_utils.capture_viewport_to_file(viewport, output_path)
                print(f"[SparkWorksAPI] Viewport captured to {output_path}")
                return True
        except (ImportError, AttributeError):
            pass

        # Fallback: renderer capture interface
        try:
            import omni.renderer.capture
            capture_iface = omni.renderer.capture.acquire_renderer_capture_interface()
            capture_iface.capture_next_frame(output_path)
            # Wait for the capture to complete
            for _ in range(wait_frames):
                await omni.kit.app.get_app().next_update_async()
            print(f"[SparkWorksAPI] Viewport captured (renderer) to {output_path}")
            return True
        except (ImportError, AttributeError):
            pass

        print("[SparkWorksAPI] No capture API available")
        return False

    except Exception as exc:
        print(f"[SparkWorksAPI] capture_viewport error: {exc}")
        return False


def compare_images(
    image_a: str,
    image_b: str,
    threshold: float = 0.98,
) -> bool:
    """
    Compare two images and return ``True`` if they are similar enough.

    Uses a simple per-pixel comparison.  Both images must have the same
    dimensions.

    Args:
        image_a: Path to the first image.
        image_b: Path to the second image.
        threshold: Minimum similarity ratio (0–1). Default 0.98 (98%).

    Returns:
        ``True`` if the images are at least *threshold* similar.
    """
    try:
        import numpy as np
        from PIL import Image

        a = np.array(Image.open(image_a).convert("RGB"), dtype=np.float32)
        b = np.array(Image.open(image_b).convert("RGB"), dtype=np.float32)

        if a.shape != b.shape:
            print(f"[SparkWorksAPI] Image dimensions differ: {a.shape} vs {b.shape}")
            return False

        # Normalised pixel-wise difference
        diff = np.abs(a - b) / 255.0
        similarity = 1.0 - diff.mean()

        print(f"[SparkWorksAPI] Image similarity: {similarity:.4f} (threshold={threshold})")
        return similarity >= threshold

    except ImportError:
        print("[SparkWorksAPI] Pillow / numpy not available for image comparison")
        return False
    except Exception as exc:
        print(f"[SparkWorksAPI] compare_images error: {exc}")
        return False
