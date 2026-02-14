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

from ..kernel.sketch import SketchLine


class SketchToolMode(Enum):
    """Which drawing tool is active."""
    NONE = auto()
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
        if (
            self._active_tool == SketchToolMode.LINE
            and self._points
            and self._cursor_pos is not None
        ):
            return (self._points[-1], self._cursor_pos)
        return None

    @property
    def is_drawing(self) -> bool:
        """True when a tool is active and at least one point has been placed."""
        return self._active_tool != SketchToolMode.NONE and len(self._points) > 0

    # -- Public API -----------------------------------------------------------

    def activate_tool(self, mode: SketchToolMode):
        """Activate a drawing tool, clearing any in-progress state."""
        self._active_tool = mode
        self._points.clear()
        self._cursor_pos = None
        self._chain_prim_count = 0
        if mode == SketchToolMode.LINE:
            self._emit_status("Line tool active — click to place first point")
        elif mode == SketchToolMode.NONE:
            self._emit_status("Ready")
        else:
            self._emit_status(f"{mode.name.title()} tool active")

    def deactivate(self):
        """Deactivate whatever tool is active."""
        self.activate_tool(SketchToolMode.NONE)

    def on_click(self, world_x: float, world_y: float) -> Optional[SketchLine]:
        """
        Handle a left-click at the given world coordinates.

        Returns the completed ``SketchLine`` if a segment was finished,
        or ``None`` if we're still accumulating points.
        """
        if self._active_tool == SketchToolMode.NONE:
            return None

        if self._active_tool == SketchToolMode.LINE:
            return self._line_click(world_x, world_y)

        # Future: RECTANGLE, CIRCLE
        return None

    def on_mouse_move(self, world_x: float, world_y: float):
        """Update the cursor position for rubber-band previews."""
        self._cursor_pos = (world_x, world_y)
        if self.on_preview_changed:
            self.on_preview_changed()

    def on_finish(self):
        """
        Finish the current drawing chain (Enter / right-click).

        The tool stays active so the user can start another chain
        immediately.
        """
        if self._active_tool == SketchToolMode.LINE:
            if self._points:
                self._points.clear()
                self._chain_prim_count = 0
                self._emit_status("Line chain finished — click to start a new line")
            # Tool remains LINE
        if self.on_preview_changed:
            self.on_preview_changed()

    def cancel(self):
        """
        Cancel the current tool (Escape).

        Removes ALL primitives created during the current chain and
        deactivates the tool.
        """
        prims_to_remove = self._chain_prim_count
        had_tool = self._active_tool != SketchToolMode.NONE
        self._active_tool = SketchToolMode.NONE
        self._points.clear()
        self._cursor_pos = None
        self._chain_prim_count = 0
        if had_tool:
            self._emit_status("Tool cancelled — chain undone")
        if prims_to_remove > 0 and self.on_chain_cancelled:
            self.on_chain_cancelled(prims_to_remove)
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

    # -- Helpers --------------------------------------------------------------

    def _emit_status(self, msg: str):
        if self.on_status_changed:
            self.on_status_changed(msg)
