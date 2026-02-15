"""
SparkWorks Extension — main entry point.

Orchestrates the geometry kernel, parametric timeline, USD bridge, and UI
into a cohesive CAD modeling experience inside Isaac Sim.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import omni.ext
import omni.kit.app
import omni.kit.commands
import omni.ui as ui
import omni.usd

try:
    import omni.kit.notification_manager as notifications
except ImportError:
    notifications = None

from .kernel import Sketch, Tessellator
from .kernel.construction_plane import (
    ConstructionPlane,
    create_origin_planes,
    extract_face_planes,
)
from .kernel.operations import (
    ExtrudeOperation,
    RevolveOperation,
    FilletOperation,
    ChamferOperation,
)
from .timeline import Timeline, Feature, FeatureType
from .bridge import UsdBridge
from .ui import CadToolbar, TimelinePanel, PropertyPanel, SketchViewport
from .ui.sketch_tool import SketchToolManager, SketchToolMode


EXTENSION_NAME = "sparkworks"

# Key codes from carb.input.KeyboardInput (used by omni.ui set_key_pressed_fn)
KEY_ENTER = 51          # carb.input.KeyboardInput.ENTER
KEY_NUMPAD_ENTER = 95   # carb.input.KeyboardInput.NUMPAD_ENTER
KEY_ESCAPE = 49         # carb.input.KeyboardInput.ESCAPE


def _notify(message: str, status="info"):
    """Send a notification to the user."""
    if notifications is not None:
        try:
            if status == "warning":
                notifications.post_notification(
                    message, status=notifications.NotificationStatus.WARNING
                )
            else:
                notifications.post_notification(
                    message, status=notifications.NotificationStatus.INFO
                )
        except Exception:
            pass
    print(f"[{EXTENSION_NAME}] {message}")


class ParametricCadExtension(omni.ext.IExt):
    """Main SparkWorks extension class."""

    def on_startup(self, ext_id: str):
        print(f"[{EXTENSION_NAME}] Extension startup")
        self._ext_id = ext_id

        # -- Core systems ----
        self._timeline = Timeline()
        self._bridge = UsdBridge()
        self._tessellator = Tessellator()

        # -- Construction planes ----
        self._construction_planes: List[ConstructionPlane] = create_origin_planes()
        self._plane_name_map: Dict[str, ConstructionPlane] = {}
        self._selected_plane: Optional[ConstructionPlane] = None

        # -- Active sketch state ----
        self._active_sketch: Optional[Sketch] = None
        self._sketch_counter = 0
        self._feature_counter = 0
        self._body_counter = 0
        self._current_body_name = "Body1"

        # -- Face-on-body state (for "sketch on face" workflow) ----
        self._face_planes: List[ConstructionPlane] = []   # currently-shown face planes
        self._face_plane_body: Optional[str] = None       # body name whose faces are shown
        self._sketch_parent_body: Optional[str] = None    # body to join into on extrude

        # -- Sketch tool manager (interactive drawing state machine) ----
        self._sketch_tool = SketchToolManager()

        # -- USD selection subscription ----
        self._selection_sub = None

        # -- UI ----
        self._toolbar = CadToolbar()
        self._timeline_panel = TimelinePanel()
        self._property_panel = PropertyPanel()
        self._sketch_viewport = SketchViewport()

        self._toolbar.build()
        self._timeline_panel.build()
        self._property_panel.build()
        self._sketch_viewport.build()

        # -- Wire callbacks ----
        self._connect_toolbar_callbacks()
        self._connect_timeline_panel_callbacks()
        self._connect_property_panel_callbacks()
        self._connect_timeline_callbacks()
        self._connect_sketch_tool_callbacks()
        self._connect_viewport_callbacks()
        self._subscribe_selection_changed()

        # -- Create origin planes in the viewport after stage is ready ----
        asyncio.ensure_future(self._create_initial_planes())

        # -- Dock windows after the UI settles ----
        asyncio.ensure_future(self._dock_windows())

        _notify("SparkWorks ready. Select a plane, then click Create Sketch.")

    async def _dock_windows(self):
        """Dock our windows relative to the Viewport after it exists."""
        viewport_win = None
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
            viewport_win = ui.Workspace.get_window("Viewport")
            if viewport_win:
                break

        if viewport_win is None:
            print(f"[{EXTENSION_NAME}] Viewport window not found, skipping dock")
            return

        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        try:
            if self._toolbar.window:
                self._toolbar.window.dock_in(
                    viewport_win, ui.DockPosition.LEFT, ratio=0.18
                )
            await omni.kit.app.get_app().next_update_async()

            if self._property_panel.window:
                self._property_panel.window.dock_in(
                    viewport_win, ui.DockPosition.RIGHT, ratio=0.18
                )
            await omni.kit.app.get_app().next_update_async()

            if self._timeline_panel.window:
                self._timeline_panel.window.dock_in(
                    viewport_win, ui.DockPosition.BOTTOM, ratio=0.18
                )
            await omni.kit.app.get_app().next_update_async()

            if self._sketch_viewport.window:
                self._sketch_viewport.window.dock_in(
                    viewport_win, ui.DockPosition.SAME
                )

            print(f"[{EXTENSION_NAME}] Windows docked successfully")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Docking error (non-fatal): {e}")

    def on_shutdown(self):
        print(f"[{EXTENSION_NAME}] Extension shutdown")
        self._selection_sub = None
        if self._toolbar:
            self._toolbar.destroy()
        if self._timeline_panel:
            self._timeline_panel.destroy()
        if self._property_panel:
            self._property_panel.destroy()
        if self._sketch_viewport:
            self._sketch_viewport.destroy()
        self._timeline = None
        self._bridge = None
        self._active_sketch = None
        self._sketch_tool = None
        self._construction_planes = []
        self._plane_name_map = {}
        self._selected_plane = None
        self._face_planes = []
        self._face_plane_body = None
        self._sketch_parent_body = None

    # ========================================================================
    # Callback Wiring
    # ========================================================================

    def _connect_toolbar_callbacks(self):
        tb = self._toolbar
        tb.on_create_sketch = self._action_create_sketch
        tb.on_finish_sketch = self._action_finish_sketch
        tb.on_add_plane = self._action_add_plane
        # Tool selection (not primitive creation)
        tb.on_tool_line = self._action_tool_line
        tb.on_tool_rectangle = self._action_tool_rectangle
        tb.on_tool_circle = self._action_tool_circle
        # 3D operations
        tb.on_extrude = self._action_extrude
        tb.on_revolve = self._action_revolve
        tb.on_fillet = self._action_fillet
        tb.on_chamfer = self._action_chamfer
        tb.on_rebuild_all = self._action_rebuild_all
        tb.on_clear_all = self._action_clear_all

    def _connect_timeline_panel_callbacks(self):
        tp = self._timeline_panel
        tp.on_feature_click = self._action_scrub_to
        tp.on_feature_delete = self._action_delete_feature
        tp.on_feature_suppress = self._action_suppress_feature
        tp.on_scrub_to_end = self._action_scrub_to_end

    def _connect_property_panel_callbacks(self):
        self._property_panel.on_param_changed = self._action_update_param
        self._property_panel.on_primitive_edited = self._on_primitive_edited

    def _connect_timeline_callbacks(self):
        self._timeline.on_rebuild = self._on_timeline_rebuild
        self._timeline.on_feature_changed = self._on_features_changed

    def _connect_sketch_tool_callbacks(self):
        """Wire the sketch tool state machine to the extension."""
        st = self._sketch_tool
        st.on_primitive_created = self._on_tool_primitive_created
        st.on_status_changed = self._set_status
        st.on_preview_changed = self._on_tool_preview_changed
        st.on_chain_cancelled = self._on_tool_chain_cancelled

    def _connect_viewport_callbacks(self):
        """Wire the sketch viewport mouse/key events to the tool manager."""
        sv = self._sketch_viewport
        sv.on_viewport_click = self._on_viewport_click
        sv.on_viewport_move = self._on_viewport_move
        sv.on_viewport_key = self._on_viewport_key

    def _set_status(self, msg: str):
        self._toolbar.set_status(msg)

    def _update_sketch_view(self):
        """Push the current active sketch's primitives to the sketch viewport."""
        if self._active_sketch:
            self._sketch_viewport.update_primitives(self._active_sketch.primitives)
            self._sketch_viewport.set_sketch_info(
                self._active_sketch.plane_name,
                self._active_sketch.name,
                len(self._active_sketch.primitives),
            )
        else:
            self._sketch_viewport.update_primitives([])
            self._sketch_viewport.clear_info()

    def _focus_sketch_view(self):
        """Bring the Sketch View tab to the front."""
        try:
            sketch_win = self._sketch_viewport.window
            if sketch_win is not None:
                sketch_win.focus()
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not focus Sketch View: {e}")

    def _focus_viewport(self):
        """Bring the 3D Viewport tab to the front."""
        try:
            viewport_win = ui.Workspace.get_window("Viewport")
            if viewport_win is not None:
                viewport_win.focus()
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not focus Viewport: {e}")

    # ========================================================================
    # Construction Planes & Selection
    # ========================================================================

    async def _create_initial_planes(self):
        """Wait for the stage, then create construction planes and load saved timeline."""
        # Give the stage a few frames to be ready
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

        # Initialise settings — load from stage if present, otherwise write defaults
        try:
            existing = self._bridge.load_settings()
            # If Settings prim doesn't exist yet, write defaults (creates the Xform)
            stage = omni.usd.get_context().get_stage()
            settings_prim = stage.GetPrimAtPath(self._bridge.settings_root) if stage else None
            if settings_prim is None or not settings_prim.IsValid():
                self._bridge.save_settings()  # writes DEFAULT_SETTINGS and sets metersPerUnit
            else:
                self._bridge.apply_settings(existing)  # respect whatever is already stored
            print(f"[{EXTENSION_NAME}] Settings: {self._bridge.load_settings()}")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not initialise settings: {e}")

        try:
            result = self._bridge.create_construction_planes(self._construction_planes)
            self._plane_name_map = {
                p.name: p for p in self._construction_planes
            }
            print(f"[{EXTENSION_NAME}] Origin planes created: {list(result.keys())}")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to create origin planes: {e}")

        # Try to load a previously saved timeline from the USD stage
        self._load_timeline_from_usd()

    def _load_timeline_from_usd(self):
        """Load timeline from USD prims if a saved timeline exists on the stage."""
        try:
            features = self._bridge.load_timeline()
            if features and len(features) > 0:
                from .timeline.timeline import Timeline
                self._timeline = Timeline()
                self._connect_timeline_callbacks()
                for feature in features:
                    self._timeline._features.append(feature)
                self._timeline._notify_changed()
                # Rebuild to restore geometry
                self._timeline.rebuild_all()
                self._set_status(f"Restored {len(features)} features from saved stage")
                _notify(f"Loaded {len(features)} features from stage.")
                print(f"[{EXTENSION_NAME}] Restored {len(features)} features from USD")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] No saved timeline to load (or error): {e}")

    def _subscribe_selection_changed(self):
        """Subscribe to USD selection changes to detect plane clicks."""
        try:
            usd_context = omni.usd.get_context()
            events = usd_context.get_stage_event_stream()
            self._selection_sub = events.create_subscription_to_pop(
                self._on_stage_event,
                name=f"{EXTENSION_NAME}_selection",
            )
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not subscribe to selection events: {e}")

    def _on_stage_event(self, event):
        """Handle USD stage events — selection changes and stage open/reload."""
        # --- Stage opened (File > Open, or reload) ---------------------------
        if event.type == int(omni.usd.StageEventType.OPENED):
            print(f"[{EXTENSION_NAME}] Stage OPENED — reloading SparkWorks state")
            asyncio.ensure_future(self._on_stage_opened())
            return

        # --- Selection changed -----------------------------------------------
        if event.type != int(omni.usd.StageEventType.SELECTION_CHANGED):
            return

        usd_context = omni.usd.get_context()
        selection = usd_context.get_selection()
        paths = selection.get_selected_prim_paths()

        if not paths:
            return

        for path in paths:
            # 1) Check if the selected prim is a construction plane (or face plane)
            plane_name = self._bridge.is_construction_plane(path)
            if plane_name and plane_name in self._plane_name_map:
                plane = self._plane_name_map[plane_name]
                self._on_construction_plane_clicked(plane)
                return

            # 2) Check if the selected prim is a body mesh
            body_name = self._bridge.is_body_prim(path)
            if body_name:
                self._on_body_clicked(body_name)
                return

    async def _on_stage_opened(self):
        """Reload all SparkWorks state after a new stage is opened."""
        # Give the stage a few frames to settle
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        # 1) Apply settings from the file (or write defaults)
        try:
            stage = omni.usd.get_context().get_stage()
            settings_prim = stage.GetPrimAtPath(self._bridge.settings_root) if stage else None
            if settings_prim and settings_prim.IsValid():
                self._bridge.apply_settings()
            else:
                self._bridge.save_settings()
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Settings reload error: {e}")

        # 2) Rebuild the construction-plane map from whatever is on stage
        self._construction_planes = create_origin_planes()
        try:
            stage = omni.usd.get_context().get_stage()
            constr_prim = stage.GetPrimAtPath(self._bridge.construction_root) if stage else None
            if constr_prim and constr_prim.IsValid():
                # Planes already exist in the file — just rebuild the lookup map
                self._plane_name_map = {p.name: p for p in self._construction_planes}
            else:
                # No planes in the file — create them
                result = self._bridge.create_construction_planes(self._construction_planes)
                self._plane_name_map = {p.name: p for p in self._construction_planes}
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Plane reload error: {e}")

        # 3) Reset transient state
        self._active_sketch = None
        self._selected_plane = None
        self._face_planes = []
        self._face_plane_body = None
        self._sketch_parent_body = None
        self._sketch_counter = 0
        self._feature_counter = 0
        self._body_counter = 0
        self._current_body_name = "Body1"

        # 4) Load the timeline from USD
        self._load_timeline_from_usd()

        # 5) Update body counter from what's on stage
        try:
            body_names = self._bridge.get_body_names()
            if body_names:
                # Find highest existing body number
                max_num = 0
                for name in body_names:
                    try:
                        num = int(name.replace("Body", ""))
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
                self._body_counter = max_num
                self._current_body_name = f"Body{max_num}"
        except Exception:
            pass

        # 6) Update feature/sketch counters from timeline
        if self._timeline and self._timeline.features:
            self._feature_counter = len(self._timeline.features)
            sketch_count = sum(
                1 for f in self._timeline.features
                if f.feature_type == FeatureType.SKETCH
            )
            self._sketch_counter = sketch_count

        self._set_status("Stage loaded — SparkWorks state restored")
        _notify("SparkWorks state restored from saved stage.")
        print(f"[{EXTENSION_NAME}] Stage reload complete: "
              f"{self._feature_counter} features, {self._body_counter} bodies")

    def _on_construction_plane_clicked(self, plane: ConstructionPlane):
        """Select a construction plane (does not create a sketch yet)."""
        self._selected_plane = plane
        # Track whether this is a face plane (for boolean join on extrude)
        if "_Face" in plane.name and self._face_plane_body:
            self._sketch_parent_body = self._face_plane_body
        else:
            self._sketch_parent_body = None
        self._toolbar.set_plane_hint(f"Selected: {plane.name} — click Create Sketch")
        self._set_status(f"Plane '{plane.name}' selected — click Create Sketch to begin")
        print(f"[{EXTENSION_NAME}] Plane selected: {plane.name}")

    def _on_body_clicked(self, body_name: str):
        """
        Handle a click on a body mesh — show selectable face planes.

        Extracts all planar faces from the build123d solid, creates
        semi-transparent construction plane quads for each, and registers
        them so the user can click on one to start a sketch.
        """
        solid = self._timeline.current_solid
        if solid is None:
            self._set_status(f"Body '{body_name}' has no solid geometry")
            return

        # Remove any previously shown face planes
        self._clear_face_planes()

        # Extract planar faces from the build123d solid
        face_planes = extract_face_planes(solid, body_name=body_name)
        if not face_planes:
            self._set_status(f"No planar faces found on '{body_name}'")
            _notify(f"No planar faces on '{body_name}'.", "warning")
            return

        # Show face planes as semi-transparent quads under the body's Construction Xform
        try:
            result = self._bridge.create_body_face_planes(body_name, face_planes)
            for fp in face_planes:
                self._plane_name_map[fp.name] = fp
            self._face_planes = face_planes
            self._face_plane_body = body_name
            n = len(face_planes)
            self._set_status(
                f"'{body_name}' — {n} face(s) shown. Click a face to select it."
            )
            self._toolbar.set_plane_hint(
                f"{n} faces on {body_name} — click one, then Create Sketch"
            )
            print(f"[{EXTENSION_NAME}] Showing {n} face planes for {body_name}")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to create face planes: {e}")
            import traceback
            traceback.print_exc()

    def _clear_face_planes(self):
        """Remove face-plane overlays from the viewport and internal maps."""
        if not self._face_planes:
            return
        # Remove from plane_name_map
        for fp in self._face_planes:
            self._plane_name_map.pop(fp.name, None)
        # Remove USD prims from the body's Construction Xform
        if self._face_plane_body:
            self._bridge.remove_body_face_planes(self._face_plane_body)
        self._face_planes = []
        self._face_plane_body = None

    def _action_new_sketch_on_plane(self, plane: ConstructionPlane):
        """Create a new sketch on a specific ConstructionPlane."""
        self._sketch_counter += 1
        name = f"Sketch{self._sketch_counter}"
        self._active_sketch = Sketch(
            name=name,
            plane_name=plane.plane_type,
            construction_plane=plane,
        )
        self._toolbar.set_sketch_mode(True)
        self._toolbar.set_plane_hint(f"Sketching on {plane.name}")
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._update_sketch_view()

        # Clear face-plane overlays now that a face has been chosen
        self._clear_face_planes()

        # Automatically switch to the Sketch View tab
        self._focus_sketch_view()

        self._set_status(f"Sketching '{name}' on {plane.name} — select a drawing tool")
        _notify(f"New sketch '{name}' on {plane.name}.")

    def _action_add_plane(self):
        """Stub for adding a custom offset/rotated plane (future feature)."""
        _notify("Custom plane creation coming soon.", "info")

    # ========================================================================
    # Viewport Events (mouse / keyboard)
    # ========================================================================

    def _on_viewport_click(self, world_x: float, world_y: float, button: int):
        """Handle a click in the sketch viewport."""
        if self._active_sketch is None:
            return

        if button == 0:  # Left click — draw
            self._sketch_tool.on_click(world_x, world_y)
        elif button == 1:  # Right click — finish chain
            self._sketch_tool.on_finish()

    def _on_viewport_move(self, world_x: float, world_y: float):
        """Handle mouse movement in the sketch viewport."""
        if self._active_sketch is None:
            return
        self._sketch_tool.on_mouse_move(world_x, world_y)

    def _on_viewport_key(self, key: int, pressed: bool):
        """Handle a key press in the sketch viewport."""
        if not pressed:
            return
        if key in (KEY_ENTER, KEY_NUMPAD_ENTER):
            self._sketch_tool.on_finish()
        elif key == KEY_ESCAPE:
            self._sketch_tool.cancel()
            self._toolbar.set_active_tool(None)

    # ========================================================================
    # Sketch Tool Events
    # ========================================================================

    def _on_tool_primitive_created(self, primitive):
        """Called when the sketch tool completes a primitive (line, rect, circle)."""
        if self._active_sketch is None:
            return

        from .kernel.sketch import SketchLine, SketchRect, SketchCircle

        if isinstance(primitive, SketchLine):
            self._active_sketch.add_line(
                start=primitive.start, end=primitive.end
            )
        elif isinstance(primitive, SketchRect):
            self._active_sketch.add_rectangle(
                width=primitive.width,
                height=primitive.height,
                center=primitive.center,
            )
        elif isinstance(primitive, SketchCircle):
            self._active_sketch.add_circle(
                radius=primitive.radius,
                center=primitive.center,
            )

        prim_idx = len(self._active_sketch.primitives) - 1
        self._update_sketch_view()

        # Show the new primitive's properties in the panel
        if self._active_sketch.primitives:
            last_prim = self._active_sketch.primitives[-1]
            self._property_panel.show_sketch_primitive(last_prim, prim_idx)

    def _on_tool_preview_changed(self):
        """Called when the rubber-band preview or placed-points change."""
        self._sketch_viewport.set_placed_points(self._sketch_tool.points)

        # Rectangle preview
        rect_preview = self._sketch_tool.preview_rect
        if rect_preview:
            c1, c2 = rect_preview
            self._sketch_viewport.set_preview_rect(c1, c2)
            self._sketch_viewport.set_preview_line(None, None)
            return
        else:
            self._sketch_viewport.set_preview_rect(None, None)

        # Line preview (also used for circle radius preview)
        preview = self._sketch_tool.preview_line
        if preview:
            from_pt, to_pt = preview
            self._sketch_viewport.set_preview_line(from_pt, to_pt)
        else:
            self._sketch_viewport.set_preview_line(None, None)

    def _on_tool_chain_cancelled(self, count: int):
        """
        Called when the user presses Escape — remove the last *count*
        primitives that were created during this chain.
        """
        if self._active_sketch is None or count <= 0:
            return
        prims = self._active_sketch.primitives
        for _ in range(min(count, len(prims))):
            prims.pop()
        self._update_sketch_view()
        self._property_panel._show_no_selection()

    def _on_primitive_edited(self, primitive, index: int):
        """Called when the user edits a primitive's properties in the panel."""
        self._update_sketch_view()

    # ========================================================================
    # Toolbar Actions
    # ========================================================================

    def _action_create_sketch(self):
        """Handle the 'Create Sketch' button — create sketch on the selected plane."""
        if self._selected_plane is None:
            self._set_status("Select a construction plane in the viewport first")
            _notify("Click a plane in the viewport first.", "warning")
            self._focus_viewport()
            return

        self._action_new_sketch_on_plane(self._selected_plane)

    def _action_new_sketch(self, plane_name: str = "XY"):
        """Legacy sketch creation from plane name string (fallback)."""
        self._sketch_counter += 1
        name = f"Sketch{self._sketch_counter}"
        self._active_sketch = Sketch(name=name, plane_name=plane_name)
        self._toolbar.set_sketch_mode(True)
        self._toolbar.set_plane_hint(f"Sketching on {plane_name}")
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._update_sketch_view()
        self._set_status(f"Sketching '{name}' on {plane_name} — select a drawing tool")
        _notify(f"New sketch '{name}' on {plane_name} plane.")

    def _action_finish_sketch(self):
        if self._active_sketch is None:
            self._set_status("No active sketch to finish")
            _notify("No active sketch to finish.", "warning")
            return
        count = len(self._active_sketch.primitives)
        name = self._active_sketch.name

        try:
            self._timeline.add_sketch(self._active_sketch)
            self._set_status(f"'{name}' added ({count} primitives) — ready for operations")
            _notify(f"Sketch '{name}' added with {count} primitives.")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Error adding sketch to timeline: {e}")
            import traceback
            traceback.print_exc()
            self._set_status(f"'{name}' saved ({count} primitives)")

        # Explicit save to USD (safety net if callback chain had issues)
        self._save_timeline_to_usd()

        # Always reset state — button label, viewport, tools
        self._active_sketch = None
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._sketch_viewport.clear_preview()
        self._update_sketch_view()
        self._toolbar.set_sketch_mode(False)
        self._toolbar.set_plane_hint("Click a plane in the viewport to start a sketch")
        self._focus_viewport()

    # -- Tool activation (interactive drawing) --------------------------------

    def _action_tool_line(self):
        if self._active_sketch is None:
            self._set_status("Click a construction plane first to start a sketch")
            _notify("Click a plane first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.LINE)
        self._toolbar.set_active_tool("line")

    def _action_tool_rectangle(self):
        if self._active_sketch is None:
            self._set_status("Click a construction plane first to start a sketch")
            _notify("Click a plane first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.RECTANGLE)
        self._toolbar.set_active_tool("rectangle")

    def _action_tool_circle(self):
        if self._active_sketch is None:
            self._set_status("Click a construction plane first to start a sketch")
            _notify("Click a plane first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.CIRCLE)
        self._toolbar.set_active_tool("circle")

    # -- 3D operations --------------------------------------------------------

    def _action_extrude(self):
        if not self._timeline.features:
            self._set_status("Add and finish a sketch first")
            _notify("Add a sketch first.", "warning")
            return
        dist = self._toolbar.extrude_distance
        self._feature_counter += 1

        # Determine if this extrude should fuse with a parent body
        is_join = self._sketch_parent_body is not None
        op = ExtrudeOperation(distance=dist, join=is_join)
        op.name = f"Extrude{self._feature_counter}"

        if is_join:
            # Sketch-on-face: fuse into the same body, don't increment counter
            self._current_body_name = self._sketch_parent_body
            self._set_status(
                f"Extruding {dist} units (join onto {self._sketch_parent_body})..."
            )
        else:
            # New body
            self._body_counter += 1
            self._current_body_name = f"Body{self._body_counter}"
            self._set_status(f"Extruding {dist} units...")

        self._timeline.add_operation(op, name=op.name)

        if is_join:
            self._set_status(
                f"Extrude (d={dist}) joined to {self._sketch_parent_body}"
            )
            self._sketch_parent_body = None  # reset
        else:
            self._set_status(
                f"Extrude (d={dist}) complete — {self._current_body_name} created"
            )

    def _action_revolve(self):
        if not self._timeline.features:
            self._set_status("Add and finish a sketch first")
            _notify("Add a sketch first.", "warning")
            return
        angle = self._toolbar.revolve_angle
        self._feature_counter += 1
        op = RevolveOperation(angle=angle, axis_name="Z")
        op.name = f"Revolve{self._feature_counter}"
        self._timeline.add_operation(op, name=op.name)
        self._set_status(f"Revolve ({angle}) complete")

    def _action_fillet(self):
        if self._timeline.current_solid is None:
            self._set_status("Create a 3D solid first (sketch + extrude)")
            _notify("Create a 3D solid first.", "warning")
            return
        radius = self._toolbar.fillet_radius
        self._feature_counter += 1
        op = FilletOperation(radius=radius)
        op.name = f"Fillet{self._feature_counter}"
        self._timeline.add_operation(op, name=op.name)
        self._set_status(f"Fillet (r={radius}) applied")

    def _action_chamfer(self):
        if self._timeline.current_solid is None:
            self._set_status("Create a 3D solid first (sketch + extrude)")
            _notify("Create a 3D solid first.", "warning")
            return
        length = self._toolbar.chamfer_length
        self._feature_counter += 1
        op = ChamferOperation(length=length)
        op.name = f"Chamfer{self._feature_counter}"
        self._timeline.add_operation(op, name=op.name)
        self._set_status(f"Chamfer (l={length}) applied")

    def _action_rebuild_all(self):
        self._set_status("Rebuilding...")
        self._timeline.rebuild_all()
        self._set_status("Rebuild complete")

    def _action_clear_all(self):
        self._timeline = Timeline()
        self._connect_timeline_callbacks()
        self._bridge.clear()
        self._active_sketch = None
        self._selected_plane = None
        self._sketch_counter = 0
        self._feature_counter = 0
        self._body_counter = 0
        self._current_body_name = "Body1"
        self._face_planes = []
        self._face_plane_body = None
        self._sketch_parent_body = None
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._toolbar.set_sketch_mode(False)
        self._toolbar.set_plane_hint("Click a plane in the viewport to start a sketch")
        self._timeline_panel.update_features([], -1)
        self._property_panel._show_no_selection()
        self._sketch_viewport.clear_preview()
        self._update_sketch_view()
        self._set_status("Cleared — click a construction plane to begin")

        # Re-create the origin planes
        self._construction_planes = create_origin_planes()
        asyncio.ensure_future(self._create_initial_planes())

    # ========================================================================
    # Timeline Panel Actions
    # ========================================================================

    def _action_scrub_to(self, index: int):
        self._timeline.scrub_to(index)
        features = self._timeline.features
        if 0 <= index < len(features):
            self._property_panel.show_feature(features[index], index)
            self._set_status(
                f"Viewing feature #{index + 1}: {features[index].name}"
            )

    def _action_scrub_to_end(self):
        self._timeline.scrub_to_end()
        self._set_status("Showing latest state")

    def _action_delete_feature(self, index: int):
        features = self._timeline.features
        if 0 <= index < len(features):
            name = features[index].name
            self._timeline.remove_feature(index)
            self._set_status(f"Deleted '{name}'")

    def _action_suppress_feature(self, index: int, suppressed: bool):
        self._timeline.suppress_feature(index, suppressed)
        state = "suppressed" if suppressed else "unsuppressed"
        self._set_status(f"Feature #{index + 1} {state}")

    def _action_update_param(self, feature_index: int, param_name: str, value):
        self._timeline.update_feature_params(feature_index, {param_name: value})
        self._set_status(f"Updated {param_name}={value}, rebuilding...")

    # ========================================================================
    # Timeline Events -> Bridge + UI Updates
    # ========================================================================

    def _on_timeline_rebuild(self, solid):
        body_name = getattr(self, '_current_body_name', 'Body1')
        if solid is not None:
            prim_path = self._bridge.update_body_mesh(solid, body_name=body_name)
            if prim_path:
                print(f"[{EXTENSION_NAME}] Body '{body_name}' updated at {prim_path}")
        else:
            # Scrubbed to before any extrude — remove the body mesh from viewport
            self._bridge.update_body_mesh(None, body_name=body_name)
            print(f"[{EXTENSION_NAME}] Scrubbed before extrude — cleared '{body_name}'")
        self._timeline_panel.update_features(
            self._timeline.features,
            self._timeline.scrub_index,
        )

    def _on_features_changed(self, features):
        try:
            self._timeline_panel.update_features(
                features, self._timeline.scrub_index
            )
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Timeline panel update error: {e}")

        # Always persist timeline to USD, even if panel update failed
        self._save_timeline_to_usd()

    def _save_timeline_to_usd(self):
        """Save the current timeline features to USD prims."""
        try:
            features = self._timeline.features if self._timeline else []
            success = self._bridge.save_timeline(features)
            if not success:
                print(f"[{EXTENSION_NAME}] save_timeline returned False")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to save timeline to USD: {e}")
            import traceback
            traceback.print_exc()
