"""
Sketch Tool Manager — state machine for interactive drawing tools.

Manages the active drawing tool (Line, Rectangle, Circle) and accumulated
click points.  When a tool action completes (e.g. two clicks for a line
segment) it fires the ``on_primitive_created`` callback with the new
sketch primitive so the extension can add it to the active sketch.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

from ..kernel.sketch import SketchLine, SketchRect, SketchCircle


class SketchToolMode(Enum):
    """Which drawing tool is active."""
    NONE = auto()       # No tool — Select mode (drag points, click to select)
    SELECT = auto()     # Explicit select mode (same as NONE but clearer intent)
    LINE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()


class SketchToolManager:
    """
    Manages the active drawing tool and accumulated click points.

    Lifecycle for the **Line** tool:

    1. ``activate_tool(LINE)`` — enters line-drawing mode.
    2. First click → stores the start point.
    3. Each subsequent click → completes a segment (fires
       ``on_primitive_created``), then starts the next segment from that
       endpoint (chained drawing).
    4. ``on_finish()`` (Enter / right-click) — ends the chain, tool stays
       active for the next chain.
    5. ``cancel()`` (Escape) — discards any in-progress points and
       deactivates the tool.
    """

    def __init__(self):
        self._active_tool: SketchToolMode = SketchToolMode.NONE
        self._points: List[Tuple[float, float]] = []
        self._cursor_pos: Optional[Tuple[float, float]] = None
        # How many primitives the current chain has created (for undo)
        self._chain_prim_count: int = 0

        # -- Callbacks (set by extension.py) ----------------------------------
        self.on_primitive_created: Optional[Callable] = None
        self.on_status_changed: Optional[Callable[[str], None]] = None
        self.on_preview_changed: Optional[Callable] = None
        # Fired on Escape with the number of primitives to remove
        self.on_chain_cancelled: Optional[Callable[[int], None]] = None

    # -- Properties -----------------------------------------------------------

    @property
    def active_tool(self) -> SketchToolMode:
        return self._active_tool

    @property
    def points(self) -> List[Tuple[float, float]]:
        return list(self._points)

    @property
    def preview_line(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return (from_pt, to_pt) for rubber-band rendering, or None."""
        if self._cursor_pos is None or not self._points:
            return None
        if self._active_tool == SketchToolMode.LINE:
            return (self._points[-1], self._cursor_pos)
        if self._active_tool == SketchToolMode.RECTANGLE:
            # Preview the diagonal of the rectangle
            return (self._points[0], self._cursor_pos)
        return None

    @property
    def preview_rect(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return (corner1, corner2) for rectangle preview, or None."""
        if (
            self._active_tool == SketchToolMode.RECTANGLE
            and self._points
            and self._cursor_pos is not None
        ):
            return (self._points[0], self._cursor_pos)
        return None

    @property
    def is_drawing(self) -> bool:
        """True when a drawing tool is active and at least one point has been placed."""
        return (
            self._active_tool not in (SketchToolMode.NONE, SketchToolMode.SELECT)
            and len(self._points) > 0
        )

    # -- Public API -----------------------------------------------------------

    def activate_tool(self, mode: SketchToolMode):
        """Activate a drawing tool, clearing any in-progress state."""
        self._active_tool = mode
        self._points.clear()
        self._cursor_pos = None
        self._chain_prim_count = 0
        if mode == SketchToolMode.LINE:
            self._emit_status("Line — click to place points, Enter to finish")
        elif mode == SketchToolMode.RECTANGLE:
            self._emit_status("Rectangle — click first corner")
        elif mode == SketchToolMode.CIRCLE:
            self._emit_status("Circle — click to place center")
        elif mode in (SketchToolMode.NONE, SketchToolMode.SELECT):
            self._emit_status("Select — click and drag points to move")
        else:
            self._emit_status(f"{mode.name.title()} tool active")

    def activate_select(self):
        """Switch to select mode (the default)."""
        self.activate_tool(SketchToolMode.SELECT)

    def deactivate(self):
        """Deactivate whatever tool is active, returning to select."""
        self.activate_tool(SketchToolMode.SELECT)

    @property
    def is_select_mode(self) -> bool:
        """True when in select/drag mode (no drawing tool active)."""
        return self._active_tool in (SketchToolMode.NONE, SketchToolMode.SELECT)

    def on_click(self, world_x: float, world_y: float):
        """
        Handle a left-click at the given world coordinates.

        Returns the completed primitive if an action was finished,
        or ``None`` if we're still accumulating points.
        """
        if self._active_tool in (SketchToolMode.NONE, SketchToolMode.SELECT):
            return None

        if self._active_tool == SketchToolMode.LINE:
            return self._line_click(world_x, world_y)
        elif self._active_tool == SketchToolMode.RECTANGLE:
            return self._rect_click(world_x, world_y)
        elif self._active_tool == SketchToolMode.CIRCLE:
            return self._circle_click(world_x, world_y)

        return None

    def on_mouse_move(self, world_x: float, world_y: float):
        """Update the cursor position for rubber-band previews."""
        self._cursor_pos = (world_x, world_y)
        if self.on_preview_changed:
            self.on_preview_changed()

    def on_finish(self):
        """
        Finish the current drawing chain (Enter / right-click).

        Switches back to Select mode so the user can drag points.
        """
        had_tool = self._active_tool not in (SketchToolMode.NONE, SketchToolMode.SELECT)
        self._points.clear()
        self._chain_prim_count = 0
        if had_tool:
            self._active_tool = SketchToolMode.SELECT
            self._emit_status("Select — click and drag points to move")
        if self.on_preview_changed:
            self.on_preview_changed()

    def cancel(self):
        """
        Escape behaviour (Fusion 360 style):

        * **Line tool with segments drawn**: undo the last segment only.
          The tool stays in LINE mode so the user can continue or press
          Escape again to undo further.
        * **Line tool with no segments (only the first point)**: discard
          the first-click marker and return to Select.
        * **Any other tool / no tool**: return to Select.
        """
        if self._active_tool == SketchToolMode.LINE:
            if self._chain_prim_count > 0:
                # Undo the last segment: remove the last point and tell
                # the extension to pop one primitive.
                self._chain_prim_count -= 1
                if len(self._points) > 1:
                    self._points.pop()
                self._emit_status(
                    "Undid last segment — click to continue, Esc again to undo more"
                )
                if self.on_chain_cancelled:
                    self.on_chain_cancelled(1)
                if self.on_preview_changed:
                    self.on_preview_changed()
                return
            # No segments drawn yet — just discard the first-click marker
            # and exit to Select.

        # Fall-through for all other cases: exit to Select
        self._active_tool = SketchToolMode.SELECT
        self._points.clear()
        self._cursor_pos = None
        self._chain_prim_count = 0
        self._emit_status("Select — click and drag points to move")
        if self.on_preview_changed:
            self.on_preview_changed()

    # -- Internal line tool ---------------------------------------------------

    def _line_click(self, wx: float, wy: float) -> Optional[SketchLine]:
        """Handle a click while the LINE tool is active."""
        pt = (wx, wy)

        if not self._points:
            # First point — just store it
            self._points.append(pt)
            self._emit_status(
                f"First point ({wx:.1f}, {wy:.1f}) — click next point, Enter to finish"
            )
            if self.on_preview_changed:
                self.on_preview_changed()
            return None

        # Subsequent click — complete a segment
        start = self._points[-1]
        line = SketchLine(start=start, end=pt)
        self._points.append(pt)
        self._chain_prim_count += 1
        self._emit_status(
            f"Segment to ({wx:.1f}, {wy:.1f}) — click next point, Enter to finish"
        )

        if self.on_primitive_created:
            self.on_primitive_created(line)
        if self.on_preview_changed:
            self.on_preview_changed()
        return line

    # -- Internal rectangle tool -----------------------------------------------

    def _rect_click(self, wx: float, wy: float):
        """Handle a click while the RECTANGLE tool is active."""
        pt = (wx, wy)

        if not self._points:
            # First corner
            self._points.append(pt)
            self._emit_status(
                f"Corner 1 ({wx:.1f}, {wy:.1f}) — click opposite corner"
            )
            if self.on_preview_changed:
                self.on_preview_changed()
            return None

        # Second corner — complete the rectangle
        c1 = self._points[0]
        c2 = pt
        cx = (c1[0] + c2[0]) / 2.0
        cy = (c1[1] + c2[1]) / 2.0
        w = abs(c2[0] - c1[0])
        h = abs(c2[1] - c1[1])

        if w < 0.001 or h < 0.001:
            self._emit_status("Rectangle too small — click a different point")
            return None

        rect = SketchRect(center=(cx, cy), width=w, height=h)
        self._points.clear()
        self._chain_prim_count += 1
        self._emit_status(
            f"Rectangle {w:.1f} x {h:.1f} at ({cx:.1f}, {cy:.1f}) — click for another"
        )

        if self.on_primitive_created:
            self.on_primitive_created(rect)
        if self.on_preview_changed:
            self.on_preview_changed()
        return rect

    # -- Internal circle tool --------------------------------------------------

    def _circle_click(self, wx: float, wy: float):
        """Handle a click while the CIRCLE tool is active."""
        pt = (wx, wy)

        if not self._points:
            # Center point
            self._points.append(pt)
            self._emit_status(
                f"Center ({wx:.1f}, {wy:.1f}) — click to set radius"
            )
            if self.on_preview_changed:
                self.on_preview_changed()
            return None

        # Second click — radius
        center = self._points[0]
        dx = pt[0] - center[0]
        dy = pt[1] - center[1]
        radius = (dx * dx + dy * dy) ** 0.5

        if radius < 0.001:
            self._emit_status("Radius too small — click farther away")
            return None

        circle = SketchCircle(center=center, radius=radius)
        self._points.clear()
        self._chain_prim_count += 1
        self._emit_status(
            f"Circle r={radius:.1f} at ({center[0]:.1f}, {center[1]:.1f}) — click for another"
        )

        if self.on_primitive_created:
            self.on_primitive_created(circle)
        if self.on_preview_changed:
            self.on_preview_changed()
        return circle

    # -- Helpers --------------------------------------------------------------

    def _emit_status(self, msg: str):
        if self.on_status_changed:
            self.on_status_changed(msg)
