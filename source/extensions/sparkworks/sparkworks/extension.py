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


EXTENSION_NAME = "sparkworks"


def _notify(message: str, status="info"):
    """Send a notification to the user."""
    if notifications is not None:
        try:
            if status == "warning":
                notifications.post_notification(message, status=notifications.NotificationStatus.WARNING)
            else:
                notifications.post_notification(message, status=notifications.NotificationStatus.INFO)
        except Exception:
            pass
    print(f"[{EXTENSION_NAME}] {message}")


class ParametricCadExtension(omni.ext.IExt):
    """Main Parametric CAD extension class."""

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

        # -- Dock windows after the UI settles ----
        asyncio.ensure_future(self._dock_windows())

        _notify("Parametric CAD ready. Click 'New Sketch' to begin.")

    async def _dock_windows(self):
        """Dock our windows relative to the Viewport after it exists."""
        # Wait for the Viewport window to be ready
        viewport_win = None
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
            viewport_win = ui.Workspace.get_window("Viewport")
            if viewport_win:
                break

        if viewport_win is None:
            print(f"[{EXTENSION_NAME}] Viewport window not found, skipping dock")
            return

        # Wait a few more frames for layout to settle
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        try:
            # Dock toolbar to the LEFT of the Viewport
            if self._toolbar.window:
                self._toolbar.window.dock_in(viewport_win, ui.DockPosition.LEFT, ratio=0.18)

            # Wait a frame between docks to let layout update
            await omni.kit.app.get_app().next_update_async()

            # Dock properties to the RIGHT of the Viewport
            if self._property_panel.window:
                self._property_panel.window.dock_in(viewport_win, ui.DockPosition.RIGHT, ratio=0.18)

            await omni.kit.app.get_app().next_update_async()

            # Dock timeline to the BOTTOM of the Viewport
            if self._timeline_panel.window:
                self._timeline_panel.window.dock_in(viewport_win, ui.DockPosition.BOTTOM, ratio=0.18)

            await omni.kit.app.get_app().next_update_async()

            # Dock sketch view as a tab next to the Viewport (center area)
            if self._sketch_viewport.window:
                self._sketch_viewport.window.dock_in(viewport_win, ui.DockPosition.SAME)

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

    # ========================================================================
    # Callback Wiring
    # ========================================================================

    def _connect_toolbar_callbacks(self):
        tb = self._toolbar
        tb.on_new_sketch = self._action_new_sketch
        tb.on_finish_sketch = self._action_finish_sketch
        tb.on_add_line = self._action_add_line
        tb.on_add_rectangle = self._action_add_rectangle
        tb.on_add_circle = self._action_add_circle
        tb.on_add_arc = self._action_add_arc
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

    def _connect_timeline_callbacks(self):
        self._timeline.on_rebuild = self._on_timeline_rebuild
        self._timeline.on_feature_changed = self._on_features_changed

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
    # Toolbar Actions
    # ========================================================================

    def _action_new_sketch(self, plane_name: str = "XY"):
        self._sketch_counter += 1
        name = f"Sketch{self._sketch_counter}"
        self._active_sketch = Sketch(name=name, plane_name=plane_name)
        self._toolbar.set_sketch_mode(True)
        self._update_sketch_view()
        self._set_status(f"Sketching '{name}' on {plane_name} — add primitives, then Finish")
        _notify(f"New sketch '{name}' on {plane_name} plane.")

    def _action_finish_sketch(self):
        if self._active_sketch is None:
            self._set_status("No active sketch to finish")
            _notify("No active sketch to finish.", "warning")
            return
        if not self._active_sketch.primitives:
            self._set_status("Sketch is empty — add a primitive first")
            _notify("Sketch is empty.", "warning")
            return

        count = len(self._active_sketch.primitives)
        name = self._active_sketch.name
        self._timeline.add_sketch(self._active_sketch)
        self._set_status(f"'{name}' added ({count} primitives) — ready for operations")
        _notify(f"Sketch '{name}' added with {count} primitives.")
        self._active_sketch = None
        self._toolbar.set_sketch_mode(False)
        self._update_sketch_view()

    def _action_add_line(self):
        if self._active_sketch is None:
            self._set_status("Start a New Sketch first")
            _notify("Start a sketch first.", "warning")
            return
        start = self._toolbar.line_start
        end = self._toolbar.line_end
        self._active_sketch.add_line(start=start, end=end)
        n = len(self._active_sketch.primitives)
        self._set_status(f"Line added — {n} primitives in sketch")
        self._update_sketch_view()

    def _action_add_rectangle(self):
        if self._active_sketch is None:
            self._set_status("Start a New Sketch first")
            _notify("Start a sketch first.", "warning")
            return
        w = self._toolbar.rect_width
        h = self._toolbar.rect_height
        self._active_sketch.add_rectangle(width=w, height=h)
        n = len(self._active_sketch.primitives)
        self._set_status(f"Rectangle {w}x{h} added — {n} primitives in sketch")
        self._update_sketch_view()

    def _action_add_circle(self):
        if self._active_sketch is None:
            self._set_status("Start a New Sketch first")
            _notify("Start a sketch first.", "warning")
            return
        r = self._toolbar.circle_radius
        self._active_sketch.add_circle(radius=r)
        n = len(self._active_sketch.primitives)
        self._set_status(f"Circle r={r} added — {n} primitives in sketch")
        self._update_sketch_view()

    def _action_add_arc(self):
        if self._active_sketch is None:
            self._set_status("Start a New Sketch first")
            _notify("Start a sketch first.", "warning")
            return
        self._active_sketch.add_arc(
            start=(0.0, 0.0), mid=(5.0, 5.0), end=(10.0, 0.0)
        )
        n = len(self._active_sketch.primitives)
        self._set_status(f"Arc added — {n} primitives in sketch")
        self._update_sketch_view()

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
        self._set_status(f"Revolve ({angle}°) complete")

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
        self._timeline_panel.update_features([], -1)
        self._property_panel._show_no_selection()
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
            self._set_status(f"Viewing feature #{index + 1}: {features[index].name}")

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
    # Timeline Events → Bridge + UI Updates
    # ========================================================================

    def _on_timeline_rebuild(self, solid):
        prim_path = self._bridge.update_mesh(solid, prim_name="ActivePart")
        if prim_path:
            print(f"[{EXTENSION_NAME}] Mesh updated at {prim_path}")
        self._timeline_panel.update_features(
            self._timeline.features, self._timeline.scrub_index,
        )

    def _on_features_changed(self, features):
        self._timeline_panel.update_features(
            features, self._timeline.scrub_index
        )
