"""
Sketch Viewport — 2D sketch preview using omni.ui.scene.

Renders the active sketch's primitives (rectangles, circles, lines, arcs)
on a 2D grid so the user can see what they've drawn before extruding.
"""

from __future__ import annotations

import math
from typing import List, Optional

try:
    import omni.ui as ui
    from omni.ui import scene as sc
except ImportError:
    ui = None
    sc = None


# Colors
GRID_COLOR = ui.color(0.2, 0.2, 0.2) if ui else None
AXIS_X_COLOR = ui.color(0.6, 0.15, 0.15) if ui else None
AXIS_Y_COLOR = ui.color(0.15, 0.6, 0.15) if ui else None
SHAPE_COLOR = ui.color(0.3, 0.6, 1.0) if ui else None
SHAPE_HIGHLIGHT_COLOR = ui.color(1.0, 0.8, 0.2) if ui else None

# Grid settings
GRID_LINES = 20
GRID_STEP = 5.0
GRID_EXTENT = GRID_LINES * GRID_STEP


class SketchViewport:
    """
    A 2D sketch preview window using omni.ui.scene.

    Shows a top-down grid with the sketch primitives drawn on it.
    The scene uses an orthographic-like 2D view looking down the Z axis.
    """

    def __init__(self):
        self._window: Optional[ui.Window] = None
        self._scene_view = None
        self._shape_transform = None
        self._primitives: List = []
        self._plane_name: str = "XY"
        self._info_label = None

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
                # Scene view fills the rest
                self._scene_view = sc.SceneView(
                    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
                )
                # Set up an orthographic top-down view
                self._scene_view.scene.clear()
                with self._scene_view.scene:
                    self._draw_grid()
                    self._draw_axes()
                    # Container for sketch shapes — we'll rebuild this
                    self._shape_transform = sc.Transform()

                # Set camera to look straight down at the XY plane
                # View matrix: identity (looking down -Z)
                # Projection: orthographic
                extent = GRID_EXTENT * 1.2
                self._scene_view.model.set_floats(
                    "projection",
                    [1.0 / extent, 0, 0, 0,
                     0, 1.0 / extent, 0, 0,
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

    def destroy(self):
        if self._window:
            self._window.destroy()
            self._window = None

    def set_sketch_info(self, plane_name: str, name: str, count: int):
        """Update the info bar."""
        self._plane_name = plane_name
        if self._info_label:
            self._info_label.text = f"Sketch: {name}  |  Plane: {plane_name}  |  {count} primitives"

    def clear_info(self):
        if self._info_label:
            self._info_label.text = "No active sketch"

    def update_primitives(self, primitives: list):
        """
        Redraw all sketch primitives.

        Args:
            primitives: List of sketch primitive objects (SketchRect, SketchCircle, etc.)
        """
        if sc is None or self._scene_view is None:
            return

        self._primitives = primitives

        # Clear and rebuild the shapes transform
        self._scene_view.scene.clear()
        with self._scene_view.scene:
            self._draw_grid()
            self._draw_axes()
            self._shape_transform = sc.Transform()
            with self._shape_transform:
                for i, prim in enumerate(primitives):
                    is_last = (i == len(primitives) - 1)
                    color = SHAPE_HIGHLIGHT_COLOR if is_last else SHAPE_COLOR
                    self._draw_primitive(prim, color)

    def _draw_grid(self):
        """Draw a 2D grid."""
        for i in range(-GRID_LINES, GRID_LINES + 1):
            pos = i * GRID_STEP
            # Horizontal lines
            sc.Line(
                (-GRID_EXTENT, pos, 0),
                (GRID_EXTENT, pos, 0),
                color=GRID_COLOR,
                thickness=1,
            )
            # Vertical lines
            sc.Line(
                (pos, -GRID_EXTENT, 0),
                (pos, GRID_EXTENT, 0),
                color=GRID_COLOR,
                thickness=1,
            )

    def _draw_axes(self):
        """Draw X and Y axis lines through the origin."""
        # X axis (red)
        sc.Line(
            (-GRID_EXTENT, 0, 0),
            (GRID_EXTENT, 0, 0),
            color=AXIS_X_COLOR,
            thickness=2,
        )
        # Y axis (green)
        sc.Line(
            (0, -GRID_EXTENT, 0),
            (0, GRID_EXTENT, 0),
            color=AXIS_Y_COLOR,
            thickness=2,
        )

    def _draw_primitive(self, prim, color):
        """Draw a single sketch primitive."""
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
        """Draw a rectangle as four lines."""
        cx, cy = rect.center
        hw = rect.width / 2.0
        hh = rect.height / 2.0

        corners = [
            (cx - hw, cy - hh, 0),
            (cx + hw, cy - hh, 0),
            (cx + hw, cy + hh, 0),
            (cx - hw, cy + hh, 0),
        ]
        for i in range(4):
            sc.Line(corners[i], corners[(i + 1) % 4], color=color, thickness=thickness)

    def _draw_circle(self, circle, color, thickness):
        """Draw a circle as a series of line segments."""
        cx, cy = circle.center
        r = circle.radius
        segments = 64
        points = []
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y, 0))
        for i in range(segments):
            sc.Line(points[i], points[i + 1], color=color, thickness=thickness)

    def _draw_line(self, line, color, thickness):
        """Draw a line segment."""
        sc.Line(
            (line.start[0], line.start[1], 0),
            (line.end[0], line.end[1], 0),
            color=color,
            thickness=thickness,
        )

    def _draw_arc_prim(self, arc, color, thickness):
        """Draw a three-point arc as line segments."""
        # Approximate the arc with segments through start, mid, end
        # For a proper arc we'd compute the circle through 3 points,
        # but for Phase 1 just draw two line segments through the points
        sc.Line(
            (arc.start[0], arc.start[1], 0),
            (arc.mid[0], arc.mid[1], 0),
            color=color,
            thickness=thickness,
        )
        sc.Line(
            (arc.mid[0], arc.mid[1], 0),
            (arc.end[0], arc.end[1], 0),
            color=color,
            thickness=thickness,
        )
