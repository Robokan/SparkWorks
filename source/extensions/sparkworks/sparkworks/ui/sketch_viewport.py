"""
Sketch Viewport — interactive 2D sketch canvas using omni.ui.scene.

Renders the active sketch's primitives (rectangles, circles, lines, arcs)
on a 2D grid.  Captures mouse clicks and movement so the user can draw
directly in the viewport (Fusion 360-style).

Coordinate handling notes
-------------------------
* ``set_mouse_released_fn`` delivers **screen-absolute** pixel coords.
  We subtract ``widget.screen_position_x/y`` to get widget-local coords.
* The SceneView uses ``STRETCH`` with a dynamic projection so the full
  widget maps 1-to-1 to NDC.
* For continuous hover tracking (rubber-band preview with no button held)
  we poll the cursor position every frame via an app-update subscription.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

try:
    import omni.ui as ui
    from omni.ui import scene as sc
    import omni.kit.app
    import carb.input
    import omni.appwindow
except ImportError:
    ui = None
    sc = None


# Colors
GRID_COLOR = ui.color(0.2, 0.2, 0.2) if ui else None
AXIS_X_COLOR = ui.color(0.6, 0.15, 0.15) if ui else None
AXIS_Y_COLOR = ui.color(0.15, 0.6, 0.15) if ui else None
SHAPE_COLOR = ui.color(0.3, 0.6, 1.0) if ui else None
SHAPE_HIGHLIGHT_COLOR = ui.color(1.0, 0.8, 0.2) if ui else None
PREVIEW_COLOR = ui.color(1.0, 0.5, 0.2, 0.8) if ui else None
CURSOR_COLOR = ui.color(1.0, 1.0, 1.0, 0.6) if ui else None
POINT_COLOR = ui.color(0.3, 1.0, 0.4) if ui else None

# Grid settings
GRID_LINES = 20
GRID_STEP = 5.0
GRID_EXTENT = GRID_LINES * GRID_STEP

# The base orthographic extent (used for the shorter widget axis)
VIEW_EXTENT = GRID_EXTENT * 1.2


class SketchViewport:
    """
    Interactive 2D sketch viewport.

    Shows a top-down grid with sketch primitives.  Captures mouse clicks
    and movement to support interactive drawing tools.
    """

    def __init__(self):
        self._window: Optional[ui.Window] = None
        self._scene_view = None
        self._shape_transform = None
        self._primitives: List = []
        self._plane_name: str = "XY"
        self._info_label = None
        self._coord_label = None

        # Projection extents (updated on resize)
        self._proj_ex: float = VIEW_EXTENT
        self._proj_ey: float = VIEW_EXTENT

        # Preview state (rubber-band line)
        self._preview_from: Optional[Tuple[float, float]] = None
        self._preview_to: Optional[Tuple[float, float]] = None
        # Preview state (rectangle)
        self._preview_rect_c1: Optional[Tuple[float, float]] = None
        self._preview_rect_c2: Optional[Tuple[float, float]] = None
        # Placed points (green dots)
        self._placed_points: List[Tuple[float, float]] = []

        # Hover tracking subscription
        self._update_sub = None
        self._mouse_hovering = False

        # Callbacks — set by the extension
        self.on_viewport_click: Optional[Callable] = None   # (world_x, world_y, button)
        self.on_viewport_move: Optional[Callable] = None     # (world_x, world_y)
        self.on_viewport_key: Optional[Callable] = None      # (key, pressed)

    @property
    def window(self):
        return self._window

    def build(self):
        """Build the sketch viewport window."""
        if ui is None or sc is None:
            return

        self._window = ui.Window(
            "Sketch View",
            width=400,
            height=400,
        )

        with self._window.frame:
            with ui.VStack(spacing=0):
                # Info bar
                self._info_label = ui.Label(
                    "No active sketch",
                    height=20,
                    style={"color": 0xFF88BBEE, "font_size": 11},
                )
                # Coordinate readout
                self._coord_label = ui.Label(
                    "",
                    height=16,
                    style={"color": 0xFF77AA77, "font_size": 10},
                )
                # SceneView with STRETCH — the projection is adjusted
                # dynamically to maintain a square grid.
                self._scene_view = sc.SceneView(
                    aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH,
                )
                self._scene_view.scene.clear()
                with self._scene_view.scene:
                    self._draw_grid()
                    self._draw_axes()
                    self._shape_transform = sc.Transform()

                # Initial projection
                self._apply_projection()

                # Click detection — use *released* so it fires after a
                # full press-and-release (i.e. a click, not a drag start).
                self._scene_view.set_mouse_released_fn(self._on_mouse_released)

                # Track enter / leave for hover polling
                self._scene_view.set_mouse_hovered_fn(self._on_mouse_hovered)

                # Re-apply projection on resize
                self._scene_view.set_computed_content_size_changed_fn(
                    self._on_scene_resized
                )

        # Key presses on the window (Enter / Escape)
        if self._window:
            self._window.set_key_pressed_fn(self._on_key_pressed)

        # Subscribe to app update for hover mouse tracking
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="sparkworks_hover")
        )

    def destroy(self):
        self._update_sub = None
        if self._window:
            self._window.destroy()
            self._window = None

    # -- Public API -----------------------------------------------------------

    def set_sketch_info(self, plane_name: str, name: str, count: int):
        self._plane_name = plane_name
        if self._info_label:
            self._info_label.text = (
                f"Sketch: {name}  |  Plane: {plane_name}  |  {count} primitives"
            )

    def clear_info(self):
        if self._info_label:
            self._info_label.text = "No active sketch"

    def update_primitives(self, primitives: list):
        if sc is None or self._scene_view is None:
            return
        self._primitives = primitives
        self._redraw()

    def set_preview_line(
        self,
        from_pt: Optional[Tuple[float, float]],
        to_pt: Optional[Tuple[float, float]],
    ):
        self._preview_from = from_pt
        self._preview_to = to_pt
        self._redraw()

    def set_placed_points(self, points: List[Tuple[float, float]]):
        self._placed_points = list(points)

    def set_preview_rect(
        self,
        corner1: Optional[Tuple[float, float]],
        corner2: Optional[Tuple[float, float]],
    ):
        """Set a rectangle preview (two opposite corners)."""
        self._preview_rect_c1 = corner1
        self._preview_rect_c2 = corner2
        self._redraw()

    def clear_preview(self):
        self._preview_from = None
        self._preview_to = None
        self._preview_rect_c1 = None
        self._preview_rect_c2 = None
        self._placed_points = []
        self._redraw()

    # -- Projection / aspect-ratio management --------------------------------

    def _apply_projection(self):
        """
        Set an orthographic projection that keeps the grid square.

        With STRETCH the scene fills the full widget, so we widen the
        projection along the longer axis to compensate.
        """
        if self._scene_view is None:
            return

        w = self._scene_view.computed_width or 1.0
        h = self._scene_view.computed_height or 1.0
        aspect = w / h if h > 0 else 1.0

        if aspect >= 1.0:
            self._proj_ex = VIEW_EXTENT * aspect
            self._proj_ey = VIEW_EXTENT
        else:
            self._proj_ex = VIEW_EXTENT
            self._proj_ey = VIEW_EXTENT / aspect

        self._scene_view.model.set_floats(
            "projection",
            [1.0 / self._proj_ex, 0, 0, 0,
             0, 1.0 / self._proj_ey, 0, 0,
             0, 0, -0.001, 0,
             0, 0, 0, 1],
        )
        self._scene_view.model.set_floats(
            "view",
            [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1],
        )

    def _on_scene_resized(self, w: float, h: float):
        self._apply_projection()

    # -- Coordinate conversion ------------------------------------------------

    def _screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """
        Convert screen-absolute pixel coordinates to world coordinates.

        ``set_mouse_released_fn`` and ``carb.input`` both give
        screen-absolute coords.  We subtract the SceneView's screen
        position to get widget-local coords, then map through the
        orthographic projection.
        """
        if self._scene_view is None:
            return (0.0, 0.0)

        # Widget-local pixel coordinates
        local_x = screen_x - self._scene_view.screen_position_x
        local_y = screen_y - self._scene_view.screen_position_y

        w = self._scene_view.computed_width
        h = self._scene_view.computed_height
        if w <= 0 or h <= 0:
            return (0.0, 0.0)

        # Pixel → NDC [-1, +1]
        ndc_x = 2.0 * (local_x / w) - 1.0
        ndc_y = 1.0 - 2.0 * (local_y / h)  # Y flipped

        # NDC → world
        world_x = ndc_x * self._proj_ex
        world_y = ndc_y * self._proj_ey
        return (world_x, world_y)

    def _is_screen_over_scene(self, screen_x: float, screen_y: float) -> bool:
        """Check if a screen-absolute position is inside the SceneView."""
        if self._scene_view is None:
            return False
        lx = screen_x - self._scene_view.screen_position_x
        ly = screen_y - self._scene_view.screen_position_y
        w = self._scene_view.computed_width
        h = self._scene_view.computed_height
        return 0 <= lx <= w and 0 <= ly <= h

    # -- Mouse / keyboard / hover handlers ------------------------------------

    def _on_mouse_released(self, x: float, y: float, button: int, modifier: int):
        """
        Handle a mouse button release (= click completed).

        Coordinates are screen-absolute.
        """
        wx, wy = self._screen_to_world(x, y)
        if self._coord_label:
            self._coord_label.text = f"  Click: ({wx:.1f}, {wy:.1f})"
        if self.on_viewport_click:
            self.on_viewport_click(wx, wy, button)

    def _on_mouse_hovered(self, hovered: bool):
        """Track whether the mouse is over the SceneView."""
        self._mouse_hovering = hovered

    def _on_update(self, event):
        """
        Per-frame update: poll the cursor position for rubber-band preview.

        This runs every frame so the rubber-band line follows the cursor
        even when no button is held.
        """
        if not self._mouse_hovering:
            return
        if self.on_viewport_move is None:
            return

        try:
            app_window = omni.appwindow.get_default_app_window()
            iinput = carb.input.acquire_input_interface()
            mouse = app_window.get_mouse()
            # carb.input stores normalised mouse position in
            # MouseInput.MOVE_X / MOVE_Y (range 0-1 of the window).
            mx = iinput.get_mouse_value(mouse, carb.input.MouseInput.MOVE_X)
            my = iinput.get_mouse_value(mouse, carb.input.MouseInput.MOVE_Y)

            # Convert normalised window coords → screen-absolute pixels
            win_width = app_window.get_width()
            win_height = app_window.get_height()
            screen_x = mx * win_width
            screen_y = my * win_height

            if self._is_screen_over_scene(screen_x, screen_y):
                wx, wy = self._screen_to_world(screen_x, screen_y)
                if self._coord_label:
                    self._coord_label.text = f"  Cursor: ({wx:.1f}, {wy:.1f})"
                self.on_viewport_move(wx, wy)
        except Exception:
            pass  # silently ignore if APIs aren't available

    def _on_key_pressed(self, key: int, mod: int, pressed: bool):
        if self.on_viewport_key:
            self.on_viewport_key(key, pressed)

    # -- Drawing helpers ------------------------------------------------------

    def _redraw(self):
        if sc is None or self._scene_view is None:
            return

        self._scene_view.scene.clear()
        with self._scene_view.scene:
            self._draw_grid()
            self._draw_axes()
            self._shape_transform = sc.Transform()
            with self._shape_transform:
                for i, prim in enumerate(self._primitives):
                    is_last = (i == len(self._primitives) - 1)
                    color = SHAPE_HIGHLIGHT_COLOR if is_last else SHAPE_COLOR
                    self._draw_primitive(prim, color)

            # Placed-point markers
            for pt in self._placed_points:
                self._draw_point_marker(pt)

            # Rubber-band preview line
            if self._preview_from is not None and self._preview_to is not None:
                sc.Line(
                    (self._preview_from[0], self._preview_from[1], 0),
                    (self._preview_to[0], self._preview_to[1], 0),
                    color=PREVIEW_COLOR,
                    thickness=2.0,
                )

            # Rectangle preview (4 edges)
            if self._preview_rect_c1 is not None and self._preview_rect_c2 is not None:
                x1, y1 = self._preview_rect_c1
                x2, y2 = self._preview_rect_c2
                corners = [
                    (x1, y1, 0), (x2, y1, 0),
                    (x2, y2, 0), (x1, y2, 0),
                ]
                for i in range(4):
                    sc.Line(
                        corners[i],
                        corners[(i + 1) % 4],
                        color=PREVIEW_COLOR,
                        thickness=2.0,
                    )

    def _draw_point_marker(self, pt: Tuple[float, float]):
        size = VIEW_EXTENT * 0.015
        x, y = pt
        sc.Line((x - size, y, 0), (x + size, y, 0), color=POINT_COLOR, thickness=2)
        sc.Line((x, y - size, 0), (x, y + size, 0), color=POINT_COLOR, thickness=2)

    def _draw_grid(self):
        for i in range(-GRID_LINES, GRID_LINES + 1):
            pos = i * GRID_STEP
            sc.Line(
                (-GRID_EXTENT, pos, 0), (GRID_EXTENT, pos, 0),
                color=GRID_COLOR, thickness=1,
            )
            sc.Line(
                (pos, -GRID_EXTENT, 0), (pos, GRID_EXTENT, 0),
                color=GRID_COLOR, thickness=1,
            )

    def _draw_axes(self):
        sc.Line(
            (-GRID_EXTENT, 0, 0), (GRID_EXTENT, 0, 0),
            color=AXIS_X_COLOR, thickness=2,
        )
        sc.Line(
            (0, -GRID_EXTENT, 0), (0, GRID_EXTENT, 0),
            color=AXIS_Y_COLOR, thickness=2,
        )

    def _draw_primitive(self, prim, color):
        from ..kernel.sketch import SketchRect, SketchCircle, SketchLine, SketchArc
        thickness = 2.5
        if isinstance(prim, SketchRect):
            self._draw_rect(prim, color, thickness)
        elif isinstance(prim, SketchCircle):
            self._draw_circle(prim, color, thickness)
        elif isinstance(prim, SketchLine):
            self._draw_line(prim, color, thickness)
        elif isinstance(prim, SketchArc):
            self._draw_arc_prim(prim, color, thickness)

    def _draw_rect(self, rect, color, thickness):
        cx, cy = rect.center
        hw, hh = rect.width / 2.0, rect.height / 2.0
        corners = [
            (cx - hw, cy - hh, 0), (cx + hw, cy - hh, 0),
            (cx + hw, cy + hh, 0), (cx - hw, cy + hh, 0),
        ]
        for i in range(4):
            sc.Line(corners[i], corners[(i + 1) % 4], color=color, thickness=thickness)

    def _draw_circle(self, circle, color, thickness):
        cx, cy = circle.center
        r = circle.radius
        segments = 64
        pts = []
        for i in range(segments + 1):
            a = 2.0 * math.pi * i / segments
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a), 0))
        for i in range(segments):
            sc.Line(pts[i], pts[i + 1], color=color, thickness=thickness)

    def _draw_line(self, line, color, thickness):
        sc.Line(
            (line.start[0], line.start[1], 0),
            (line.end[0], line.end[1], 0),
            color=color, thickness=thickness,
        )

    def _draw_arc_prim(self, arc, color, thickness):
        sc.Line(
            (arc.start[0], arc.start[1], 0),
            (arc.mid[0], arc.mid[1], 0),
            color=color, thickness=thickness,
        )
        sc.Line(
            (arc.mid[0], arc.mid[1], 0),
            (arc.end[0], arc.end[1], 0),
            color=color, thickness=thickness,
        )
