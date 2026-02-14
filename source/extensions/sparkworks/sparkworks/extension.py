"""
SparkWorks Extension — main entry point.

Orchestrates the geometry kernel, parametric timeline, USD bridge, and UI
into a cohesive CAD modeling experience inside Isaac Sim.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import omni.ext
import omni.kit.app
import omni.kit.commands
import omni.ui as ui

try:
    import omni.kit.notification_manager as notifications
except ImportError:
    notifications = None

from .kernel import Sketch, Tessellator
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

        # -- Active sketch state ----
        self._active_sketch: Optional[Sketch] = None
        self._sketch_counter = 0
        self._feature_counter = 0

        # -- Sketch tool manager (interactive drawing state machine) ----
        self._sketch_tool = SketchToolManager()

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

        # -- Dock windows after the UI settles ----
        asyncio.ensure_future(self._dock_windows())

        _notify("SparkWorks ready. Click 'New Sketch' to begin.")

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

    # ========================================================================
    # Callback Wiring
    # ========================================================================

    def _connect_toolbar_callbacks(self):
        tb = self._toolbar
        tb.on_new_sketch = self._action_new_sketch
        tb.on_finish_sketch = self._action_finish_sketch
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
        """Called when the sketch tool completes a primitive (e.g. a line segment)."""
        if self._active_sketch is None:
            return

        from .kernel.sketch import SketchLine

        if isinstance(primitive, SketchLine):
            self._active_sketch.add_line(
                start=primitive.start, end=primitive.end
            )

        prim_idx = len(self._active_sketch.primitives) - 1
        self._update_sketch_view()

        # Show the new primitive's properties in the panel
        if self._active_sketch.primitives:
            last_prim = self._active_sketch.primitives[-1]
            self._property_panel.show_sketch_primitive(last_prim, prim_idx)

    def _on_tool_preview_changed(self):
        """Called when the rubber-band preview or placed-points change."""
        preview = self._sketch_tool.preview_line
        if preview:
            from_pt, to_pt = preview
            self._sketch_viewport.set_placed_points(self._sketch_tool.points)
            self._sketch_viewport.set_preview_line(from_pt, to_pt)
        else:
            self._sketch_viewport.set_placed_points(self._sketch_tool.points)
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

    def _action_new_sketch(self, plane_name: str = "XY"):
        self._sketch_counter += 1
        name = f"Sketch{self._sketch_counter}"
        self._active_sketch = Sketch(name=name, plane_name=plane_name)
        self._toolbar.set_sketch_mode(True)
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
        if not self._active_sketch.primitives:
            self._set_status("Sketch is empty — draw something first")
            _notify("Sketch is empty.", "warning")
            return

        count = len(self._active_sketch.primitives)
        name = self._active_sketch.name
        self._timeline.add_sketch(self._active_sketch)
        self._set_status(f"'{name}' added ({count} primitives) — ready for operations")
        _notify(f"Sketch '{name}' added with {count} primitives.")
        self._active_sketch = None
        self._toolbar.set_sketch_mode(False)
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._sketch_viewport.clear_preview()
        self._update_sketch_view()

    # -- Tool activation (interactive drawing) --------------------------------

    def _action_tool_line(self):
        if self._active_sketch is None:
            self._set_status("Start a New Sketch first")
            _notify("Start a sketch first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.LINE)
        self._toolbar.set_active_tool("line")

    def _action_tool_rectangle(self):
        if self._active_sketch is None:
            self._set_status("Start a New Sketch first")
            _notify("Start a sketch first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.RECTANGLE)
        self._toolbar.set_active_tool("rectangle")
        self._set_status("Rectangle tool — not yet implemented (coming soon)")

    def _action_tool_circle(self):
        if self._active_sketch is None:
            self._set_status("Start a New Sketch first")
            _notify("Start a sketch first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.CIRCLE)
        self._toolbar.set_active_tool("circle")
        self._set_status("Circle tool — not yet implemented (coming soon)")

    # -- 3D operations --------------------------------------------------------

    def _action_extrude(self):
        if not self._timeline.features:
            self._set_status("Add and finish a sketch first")
            _notify("Add a sketch first.", "warning")
            return
        dist = self._toolbar.extrude_distance
        self._feature_counter += 1
        op = ExtrudeOperation(distance=dist)
        op.name = f"Extrude{self._feature_counter}"
        self._set_status(f"Extruding {dist} units...")
        self._timeline.add_operation(op, name=op.name)
        self._set_status(f"Extrude (d={dist}) complete — mesh in viewport")

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
        self._sketch_counter = 0
        self._feature_counter = 0
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._toolbar.set_sketch_mode(False)
        self._timeline_panel.update_features([], -1)
        self._property_panel._show_no_selection()
        self._sketch_viewport.clear_preview()
        self._update_sketch_view()
        self._set_status("Cleared — click New Sketch to begin")

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
        prim_path = self._bridge.update_mesh(solid, prim_name="ActivePart")
        if prim_path:
            print(f"[{EXTENSION_NAME}] Mesh updated at {prim_path}")
        self._timeline_panel.update_features(
            self._timeline.features,
            self._timeline.scrub_index,
        )

    def _on_features_changed(self, features):
        self._timeline_panel.update_features(
            features, self._timeline.scrub_index
        )
