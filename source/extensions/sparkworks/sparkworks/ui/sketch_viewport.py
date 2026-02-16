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
* Selection is driven by the USD stage — clicking in the viewport
  hit-tests against known primitive/point positions and sets the stage
  selection, which then drives highlighting via ``set_selected_paths()``.
* Rubber-band preview works during click-hold via ``set_mouse_moved_fn``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Set, Tuple

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
SHAPE_COLOR = ui.color(1.0, 1.0, 1.0) if ui else None           # white lines
SHAPE_HIGHLIGHT_COLOR = ui.color(1.0, 0.8, 0.2) if ui else None
PREVIEW_COLOR = ui.color(1.0, 0.5, 0.2, 0.8) if ui else None
POINT_COLOR = ui.color(0.3, 1.0, 0.4) if ui else None
SELECTED_COLOR = ui.color(0.4, 0.7, 1.0, 1.0) if ui else None  # blue for selected
POINT_CONNECTED_COLOR = ui.color(0.1, 0.9, 0.1, 1.0) if ui else None   # green — connected
POINT_DISCONNECTED_COLOR = ui.color(1.0, 0.2, 0.2, 1.0) if ui else None  # red — disconnected
SNAP_HIGHLIGHT_COLOR = ui.color(0.3, 0.5, 1.0, 1.0) if ui else None     # blue — snap candidate
CONSTRAINT_LINE_COLOR = ui.color(0.6, 0.9, 1.0, 1.0) if ui else None   # light cyan — constrained line
CONSTRAINT_LABEL_COLOR = ui.color(1.0, 0.6, 0.2, 1.0) if ui else None  # orange — constraint label
CONSTRAINT_ICON_COLOR = ui.color(1.0, 0.4, 0.4, 0.8) if ui else None
COINCIDENT_COLOR = ui.color(0.9, 0.3, 0.9, 0.9) if ui else None
SNAP_COLOR = ui.color(1.0, 0.3, 1.0, 1.0) if ui else None

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
    and button-held movement to support drawing tools and drag-to-move.
    Selection is driven externally by the USD stage selection.
    """

    def __init__(self):
        self._window: Optional[ui.Window] = None
        self._scene_view = None
        self._shape_transform = None
        self._primitives: List = []
        self._hidden_indices: set = set()
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

        # Constraint data for rendering icons
        self._constraint_data: List[dict] = []

        # Selection-driven highlighting.
        # Maps USD paths to their visual data for hit-testing and highlighting.
        # _point_positions: maps usd_path -> (x, y) for every selectable point
        # _prim_paths: maps prim index -> usd_path for whole primitives
        self._point_positions: Dict[str, Tuple[float, float]] = {}
        self._prim_paths: Dict[int, str] = {}
        self._selected_paths: Set[str] = set()
        self._connected_paths: Set[str] = set()  # points with coincident constraints
        self._prim_constraint_labels: Dict[int, List[str]] = {}  # prim_idx → ["H", "∥"]

        # Batch redraw — prevents redundant scene rebuilds when multiple
        # data updates happen in quick succession (e.g. _update_sketch_view
        # pushes primitives + constraints + connected paths).
        self._batch_depth: int = 0
        self._batch_dirty: bool = False

        # Drag state — driven by selecting a point in the stage
        self._drag_path: Optional[str] = None  # USD path of point being dragged
        self._dragging: bool = False
        self._snap_target_path: Optional[str] = None  # nearby point for snap-to-join
        self._select_mode: bool = True  # True when no drawing tool active

        # Line-body drag state (translate whole line by delta)
        self._drag_line_path: Optional[str] = None
        self._drag_line_anchor: Optional[Tuple[float, float]] = None

        # Constraint hit-map: maps constraint USD path -> (x, y) position
        self._constraint_positions: Dict[str, Tuple[float, float]] = {}

        # Callbacks — set by the extension
        self.on_viewport_click: Optional[Callable] = None   # (world_x, world_y, button)
        self.on_viewport_move: Optional[Callable] = None     # (world_x, world_y)
        self.on_viewport_key: Optional[Callable] = None      # (key, pressed)
        self.on_select_prim: Optional[Callable] = None       # (usd_path, add_to_selection) -> None
        self.on_drag_point_path: Optional[Callable] = None   # (usd_path, wx, wy) -> None
        self.on_drag_line_path: Optional[Callable] = None    # (usd_path, dx, dy) -> None
        self.on_snap_join: Optional[Callable] = None         # (drag_path, target_path) -> None
        self.on_context_menu: Optional[Callable] = None      # (usd_path, screen_x, screen_y) -> None
        self.on_background_click: Optional[Callable] = None  # () -> None  (click on empty area)

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

                # Click / drag / release
                self._scene_view.set_mouse_pressed_fn(self._on_mouse_pressed)
                self._scene_view.set_mouse_released_fn(self._on_mouse_released)
                self._scene_view.set_mouse_moved_fn(self._on_mouse_moved)

                # Re-apply projection on resize
                self._scene_view.set_computed_content_size_changed_fn(
                    self._on_scene_resized
                )

        # Key presses on the window (Enter / Escape)
        if self._window:
            self._window.set_key_pressed_fn(self._on_key_pressed)

    def destroy(self):
        if self._window:
            self._window.destroy()
            self._window = None

    # -- Batch redraw ---------------------------------------------------------

    def begin_batch(self):
        """Defer ``_redraw`` calls until the matching ``end_batch``."""
        self._batch_depth += 1

    def end_batch(self):
        """Complete a batch; redraw once if any update marked dirty."""
        self._batch_depth = max(0, self._batch_depth - 1)
        if self._batch_depth == 0 and self._batch_dirty:
            self._batch_dirty = False
            self._redraw()

    def _request_redraw(self):
        """Schedule a redraw — deferred while inside a batch."""
        if self._batch_depth > 0:
            self._batch_dirty = True
        else:
            self._redraw()

    # -- Public API -----------------------------------------------------------

    def set_sketch_info(self, plane_name: str, name: str, count: int,
                         dof: int = -1, constraint_count: int = 0):
        self._plane_name = plane_name
        if self._info_label:
            base = f"Sketch: {name}  |  Plane: {plane_name}  |  {count} primitives"
            if constraint_count > 0 or dof >= 0:
                dof_str = str(dof) if dof >= 0 else "?"
                base += f"  |  {constraint_count} constraints  |  DOF: {dof_str}"
            self._info_label.text = base

    def clear_info(self):
        if self._info_label:
            self._info_label.text = "No active sketch"

    def update_primitives(self, primitives: list, hidden_indices: set = None):
        """Update the primitive list and rebuild the point/path maps."""
        if sc is None or self._scene_view is None:
            return
        self._primitives = primitives
        self._hidden_indices = hidden_indices or set()
        # Clear stale selection when primitive set changes (e.g. sketch
        # closed and reopened) so highlighting doesn't carry over.
        if not primitives:
            self._selected_paths.clear()
        self._rebuild_hit_maps()
        self._request_redraw()

    def set_preview_line(
        self,
        from_pt: Optional[Tuple[float, float]],
        to_pt: Optional[Tuple[float, float]],
    ):
        self._preview_from = from_pt
        self._preview_to = to_pt
        self._request_redraw()

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
        self._request_redraw()

    def set_select_mode(self, active: bool):
        """Enable/disable select mode (point dragging)."""
        self._select_mode = active

    def set_selected_paths(self, paths: Set[str]):
        """
        Set the USD paths that should be highlighted.

        Called by the extension when the stage selection changes.
        Paths are normalised to plain strings so comparisons with
        ``_prim_paths`` (which are Python str) always work, even if
        the stage selection returns ``Sdf.Path`` objects.
        """
        self._selected_paths = {str(p) for p in paths}
        self._request_redraw()

    def set_connected_paths(self, paths: Set[str]):
        """
        Set point USD paths that have coincident constraints (connected).

        Disconnected points are drawn red; connected ones green.
        """
        if paths != self._connected_paths:
            self._connected_paths = set(paths)
            self._request_redraw()

    def set_prim_constraint_labels(self, labels: Dict[int, List[str]]):
        """
        Set per-primitive constraint labels for visual indicators.

        *labels* maps primitive index → list of short labels like
        ``["H"]``, ``["V"]``, ``["∥"]``, ``["⊥"]``, ``["="]``.
        These are drawn as small text/icons at the midpoint of each
        constrained line.
        """
        self._prim_constraint_labels = dict(labels)

    def update_constraint_data(self, data: List[dict]):
        """
        Update the constraint visualisation data.

        Each dict should have ``type`` (str), ``x`` (float), ``y`` (float),
        and optionally ``usd_path`` (str) for selectable constraint prims.
        """
        self._constraint_data = list(data)
        # Rebuild constraint hit-map from data that carries a usd_path
        self._constraint_positions.clear()
        for d in data:
            cp = d.get("usd_path")
            if cp:
                self._constraint_positions[cp] = (d["x"], d["y"])
        self._request_redraw()

    def clear_preview(self):
        self._preview_from = None
        self._preview_to = None
        self._preview_rect_c1 = None
        self._preview_rect_c2 = None
        self._placed_points = []
        self._request_redraw()

    # -- Hit-map management ---------------------------------------------------

    def _rebuild_hit_maps(self):
        """
        Rebuild the dictionaries that map USD paths to positions
        for hit-testing and highlighting.
        """
        from ..kernel.sketch import SketchLine, SketchRect, SketchCircle, SketchArc

        self._point_positions.clear()
        self._prim_paths.clear()

        for i, prim in enumerate(self._primitives):
            usd_path = getattr(prim, "usd_path", None)
            if usd_path:
                self._prim_paths[i] = usd_path

            if isinstance(prim, SketchLine):
                sp = getattr(prim, "start_usd_path", None)
                ep = getattr(prim, "end_usd_path", None)
                if sp:
                    self._point_positions[sp] = prim.start
                if ep:
                    self._point_positions[ep] = prim.end
            elif isinstance(prim, SketchRect):
                # Center point (for whole-rectangle dragging)
                if usd_path:
                    self._point_positions[usd_path] = prim.center
                # Corner points
                corner_paths = getattr(prim, "corner_usd_paths", None)
                if corner_paths:
                    corners = prim.corners
                    for cp_path, corner_pos in zip(corner_paths, corners):
                        if cp_path:
                            self._point_positions[cp_path] = corner_pos
            elif isinstance(prim, SketchCircle):
                # Center point is the primitive path itself
                if usd_path:
                    self._point_positions[usd_path] = prim.center
                # Edge (radius handle) point on circumference
                edge_path = getattr(prim, "edge_usd_path", None)
                if edge_path:
                    self._point_positions[edge_path] = prim.edge_point

    def refresh_positions(self):
        """
        Lightweight position-only refresh for drag operations.

        Updates point handle positions from the current primitive data
        *without* rebuilding the full hit maps or prim-path dictionaries
        (topology is unchanged during a drag).
        """
        from ..kernel.sketch import SketchLine, SketchRect, SketchCircle
        for i, prim in enumerate(self._primitives):
            if isinstance(prim, SketchLine):
                sp = getattr(prim, "start_usd_path", None)
                ep = getattr(prim, "end_usd_path", None)
                if sp:
                    self._point_positions[sp] = prim.start
                if ep:
                    self._point_positions[ep] = prim.end
            elif isinstance(prim, SketchRect):
                usd_path = getattr(prim, "usd_path", None)
                if usd_path:
                    self._point_positions[usd_path] = prim.center
                corner_paths = getattr(prim, "corner_usd_paths", None)
                if corner_paths:
                    corners = prim.corners
                    for cp_path, corner_pos in zip(corner_paths, corners):
                        if cp_path:
                            self._point_positions[cp_path] = corner_pos
            elif isinstance(prim, SketchCircle):
                usd_path = getattr(prim, "usd_path", None)
                if usd_path:
                    self._point_positions[usd_path] = prim.center
                edge_path = getattr(prim, "edge_usd_path", None)
                if edge_path:
                    self._point_positions[edge_path] = prim.edge_point
        self._request_redraw()

    # -- Projection / aspect-ratio management --------------------------------

    def _apply_projection(self, w: float = 0.0, h: float = 0.0):
        """
        Set an orthographic projection that keeps the grid square.

        With STRETCH the scene fills the full widget, so we widen the
        projection along the longer axis to compensate.

        Args:
            w, h: Widget dimensions.  When zero (e.g. initial call during
                  ``build()``), the values are read from the SceneView.
                  If valid dimensions are unavailable, the projection is
                  left unchanged to avoid corrupting coordinates.
        """
        if self._scene_view is None:
            return

        if w <= 0 or h <= 0:
            w = self._scene_view.computed_width or 0.0
            h = self._scene_view.computed_height or 0.0

        # Don't update projection if we can't get valid dimensions —
        # keep the last known good values from _on_scene_resized.
        if w <= 0 or h <= 0:
            return

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
        self._apply_projection(w, h)

    # -- Coordinate conversion ------------------------------------------------

    def _screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """
        Convert screen-absolute pixel coordinates to world coordinates.

        ``set_mouse_released_fn`` gives screen-absolute coords.
        We subtract the SceneView's screen position to get widget-local
        coords, then map through the orthographic projection.
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

        # Pixel -> NDC [-1, +1]
        ndc_x = 2.0 * (local_x / w) - 1.0
        ndc_y = 1.0 - 2.0 * (local_y / h)  # Y flipped

        # NDC -> world
        world_x = ndc_x * self._proj_ex
        world_y = ndc_y * self._proj_ey
        return (world_x, world_y)

    # -- Hit-testing ----------------------------------------------------------

    def _hit_test(self, wx: float, wy: float) -> Optional[str]:
        """
        Find the nearest selectable USD path at world position (wx, wy).

        Returns the USD path of the closest point/prim within hit radius,
        or None if nothing is close.  Points are checked first (they are
        smaller targets and more useful to select individually).
        """
        hit_radius = VIEW_EXTENT * 0.03
        best_path = None
        best_dist = float("inf")

        # Check point positions first
        for path, (px, py) in self._point_positions.items():
            d = math.hypot(wx - px, wy - py)
            if d < hit_radius and d < best_dist:
                best_dist = d
                best_path = path

        # Check constraint icon positions (slightly larger hit area for small icons)
        constr_hit_radius = hit_radius * 1.5
        for cpath, (cx, cy) in self._constraint_positions.items():
            d = math.hypot(wx - cx, wy - cy)
            if d < constr_hit_radius and d < best_dist:
                best_dist = d
                best_path = cpath

        # If no point or constraint hit, check line segments and circle edges
        if best_path is None:
            from ..kernel.sketch import SketchLine, SketchCircle
            for i, prim in enumerate(self._primitives):
                if i in self._hidden_indices:
                    continue
                usd_path = self._prim_paths.get(i)
                if usd_path is None:
                    continue
                if isinstance(prim, SketchLine):
                    d = self._point_to_segment_dist(
                        wx, wy,
                        prim.start[0], prim.start[1],
                        prim.end[0], prim.end[1],
                    )
                    if d < hit_radius and d < best_dist:
                        best_dist = d
                        best_path = usd_path
                elif isinstance(prim, SketchCircle):
                    # Distance from click to the circumference
                    cx, cy = prim.center
                    dist_to_center = math.hypot(wx - cx, wy - cy)
                    d = abs(dist_to_center - prim.radius)
                    if d < hit_radius and d < best_dist:
                        best_dist = d
                        best_path = usd_path

        return best_path

    def _apply_local_selection(self, usd_path: str, add_to_selection: bool):
        """
        Update ``_selected_paths`` immediately so the next redraw reflects
        the selection without waiting for the async USD stage event.
        """
        if add_to_selection:
            if usd_path in self._selected_paths:
                self._selected_paths.discard(usd_path)
            else:
                self._selected_paths.add(usd_path)
        else:
            self._selected_paths = {usd_path}

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
        """Distance from point (px,py) to the line segment (x1,y1)-(x2,y2)."""
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def _update_snap_target(self, wx: float, wy: float):
        """
        During a drag, find the nearest OTHER point that the dragged point
        could snap to.  Updates ``_snap_target_path`` and redraws to show
        visual feedback (blue highlight).

        Points co-located with the drag position (i.e. connected siblings
        that the solver moved together) are excluded so they don't mask
        the real snap target.
        """
        snap_radius = VIEW_EXTENT * 0.04
        # Threshold to detect points that are "co-moving" with the drag
        # (already connected via solver constraints).  Must be much smaller
        # than snap_radius so it only catches truly coincident points.
        colocate_eps = VIEW_EXTENT * 0.003
        best_path: Optional[str] = None
        best_dist = float("inf")
        for path, (px, py) in self._point_positions.items():
            if path == self._drag_path:
                continue  # skip the point we're dragging
            d = math.hypot(wx - px, wy - py)
            if d < colocate_eps:
                continue  # skip points co-located with drag (connected siblings)
            if d < snap_radius and d < best_dist:
                best_dist = d
                best_path = path
        old = self._snap_target_path
        self._snap_target_path = best_path
        if old != best_path:
            self._request_redraw()

    # -- Mouse / keyboard handlers -------------------------------------------

    def _on_mouse_released(self, x: float, y: float, button: int, modifier: int):
        """Handle a mouse button release (= click completed)."""
        wx, wy = self._screen_to_world(x, y)
        if self._coord_label:
            self._coord_label.text = f"  Click: ({wx:.1f}, {wy:.1f})"

        # End a drag if one was active
        if self._dragging:
            drag_path = self._drag_path
            snap_path = self._snap_target_path
            self._dragging = False
            self._drag_path = None
            self._drag_line_path = None
            self._drag_line_anchor = None
            self._snap_target_path = None
            # If released on a snap target, fire the join callback
            if snap_path is not None and drag_path is not None and self.on_snap_join:
                self.on_snap_join(drag_path, snap_path)
            self._request_redraw()
            return

        # Right-click context menu
        if self._select_mode and button == 1:
            hit_path = self._hit_test(wx, wy)
            if hit_path and self.on_context_menu:
                self.on_context_menu(hit_path, x, y)
                return

        # In select mode, try to select a prim by clicking
        if self._select_mode and button == 0:
            hit_path = self._hit_test(wx, wy)
            if hit_path and self.on_select_prim:
                # Ctrl = modifier bit 2 (value 2) in omni.ui
                add_to_sel = bool(modifier & 2)
                self._apply_local_selection(hit_path, add_to_sel)
                self.on_select_prim(hit_path, add_to_sel)
                return
            # No hit — clicked the background
            if hit_path is None and self.on_background_click:
                self.on_background_click()
                return

        # Otherwise forward to the drawing tool
        if self.on_viewport_click:
            self.on_viewport_click(wx, wy, button)

    def _on_mouse_pressed(self, x: float, y: float, button: int, modifier: int):
        """Start dragging the nearest point or line body if in Select mode."""
        if button != 0:
            return
        if not self._select_mode:
            return
        wx, wy = self._screen_to_world(x, y)
        hit_radius = VIEW_EXTENT * 0.03
        add_to_sel = bool(modifier & 2)  # Ctrl held → multi-select

        # Check if we're near a point that can be dragged (points win over lines)
        best_path = None
        best_dist = float("inf")
        for path, (px, py) in self._point_positions.items():
            d = math.hypot(wx - px, wy - py)
            if d < hit_radius and d < best_dist:
                best_dist = d
                best_path = path

        if best_path is not None:
            self._drag_path = best_path
            self._dragging = True
            self._apply_local_selection(best_path, add_to_sel)
            if self.on_select_prim:
                self.on_select_prim(best_path, add_to_sel)
            return

        # Check constraint icon positions (select-only, no drag)
        constr_best = None
        constr_dist = float("inf")
        constr_hit_radius = hit_radius * 1.5
        for cpath, (cx, cy) in self._constraint_positions.items():
            d = math.hypot(wx - cx, wy - cy)
            if d < constr_hit_radius and d < constr_dist:
                constr_dist = d
                constr_best = cpath

        # No point hit — check line body / circle body proximity for whole-prim dragging
        line_best = None
        line_dist = float("inf")
        from ..kernel.sketch import SketchLine, SketchCircle
        for i, prim in enumerate(self._primitives):
            if i in self._hidden_indices:
                continue
            usd_path = self._prim_paths.get(i)
            if usd_path is None:
                continue
            if isinstance(prim, SketchLine):
                d = self._point_to_segment_dist(
                    wx, wy,
                    prim.start[0], prim.start[1],
                    prim.end[0], prim.end[1],
                )
                if d < hit_radius and d < line_dist:
                    line_dist = d
                    line_best = (usd_path, prim)
            elif isinstance(prim, SketchCircle):
                cx, cy = prim.center
                dist_to_center = math.hypot(wx - cx, wy - cy)
                d = abs(dist_to_center - prim.radius)
                if d < hit_radius and d < line_dist:
                    line_dist = d
                    line_best = (usd_path, prim)

        # Constraint icons take priority over line bodies when closer
        if constr_best is not None and constr_dist <= line_dist:
            self._apply_local_selection(constr_best, add_to_sel)
            if self.on_select_prim:
                self.on_select_prim(constr_best, add_to_sel)
            return

        # Start a line/circle body drag (anchor in screen space to avoid
        # projection-drift issues if _apply_projection runs mid-drag)
        if line_best is not None:
            usd_path = line_best[0]
            self._drag_line_path = usd_path
            self._drag_line_anchor = (x, y)  # screen coords
            self._dragging = True
            self._apply_local_selection(usd_path, add_to_sel)
            if self.on_select_prim:
                self.on_select_prim(usd_path, add_to_sel)
            return

        pass  # No hit at this position

    def _on_mouse_moved(self, x: float, y: float, mod: int, pressed: bool):
        """
        Fires while a mouse button is held and the cursor moves.

        Used for:
        - Drag tracking (select mode, point dragging / line dragging)
        - Rubber-band preview during click-hold (drawing tools)
        """
        wx, wy = self._screen_to_world(x, y)

        # Drag tracking for point prims
        if self._dragging and self._drag_path is not None:
            if self._coord_label:
                self._coord_label.text = f"  Drag: ({wx:.1f}, {wy:.1f})"
            if self.on_drag_point_path:
                self.on_drag_point_path(self._drag_path, wx, wy)
            self._update_snap_target(wx, wy)
            return

        # Drag tracking for whole-line bodies (translate by delta).
        # Anchor is in screen space; compute world delta from screen delta
        # to avoid projection-drift between frames.
        if self._dragging and self._drag_line_path is not None:
            if self._coord_label:
                self._coord_label.text = f"  Drag line: ({wx:.1f}, {wy:.1f})"
            ax_s, ay_s = self._drag_line_anchor
            dx_s, dy_s = x - ax_s, y - ay_s
            self._drag_line_anchor = (x, y)  # update in screen coords
            # Convert screen-pixel delta to world-unit delta
            sv_w = self._scene_view.computed_width if self._scene_view else 0
            sv_h = self._scene_view.computed_height if self._scene_view else 0
            if sv_w > 0 and sv_h > 0:
                dx = dx_s * (2.0 * self._proj_ex / sv_w)
                dy = -dy_s * (2.0 * self._proj_ey / sv_h)
            else:
                dx, dy = 0.0, 0.0
            if self.on_drag_line_path:
                self.on_drag_line_path(self._drag_line_path, dx, dy)
            return

        # Rubber-band preview: forward move to the tool manager so the
        # preview updates while the user holds the button.
        if self._coord_label:
            self._coord_label.text = f"  Move: ({wx:.1f}, {wy:.1f})"
        if self.on_viewport_move:
            self.on_viewport_move(wx, wy)

    def _on_key_pressed(self, key: int, mod: int, pressed: bool):
        if self.on_viewport_key:
            self.on_viewport_key(key, pressed)

    # -- Drawing helpers ------------------------------------------------------

    def _redraw(self):
        if sc is None or self._scene_view is None:
            return

        # Ensure the projection matches the current widget size so that
        # circles remain round even if the resize callback was missed.
        self._apply_projection()

        self._scene_view.scene.clear()
        with self._scene_view.scene:
            self._draw_grid()
            self._draw_axes()
            self._shape_transform = sc.Transform()
            with self._shape_transform:
                from ..kernel.sketch import SketchLine
                for i, prim in enumerate(self._primitives):
                    if i in self._hidden_indices:
                        continue
                    # Highlight if this prim's USD path is selected
                    prim_path = self._prim_paths.get(i)
                    is_selected = prim_path is not None and prim_path in self._selected_paths
                    color = SELECTED_COLOR if is_selected else SHAPE_COLOR
                    self._draw_primitive(prim, color)

                    # Draw constraint label at the midpoint of the line
                    labels = self._prim_constraint_labels.get(i)
                    if labels and isinstance(prim, SketchLine):
                        mx = (prim.start[0] + prim.end[0]) / 2.0
                        my = (prim.start[1] + prim.end[1]) / 2.0
                        self._draw_constraint_label(mx, my + 2.0, labels)

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

            # Point handles (from _point_positions)
            has_snap = self._snap_target_path is not None and self._dragging
            for path, (px, py) in self._point_positions.items():
                is_selected = path in self._selected_paths
                is_drag = (path == self._drag_path and self._dragging)
                is_snap_target = (path == self._snap_target_path)
                # Both the dragged point AND the snap target glow blue
                is_snap = is_snap_target or (is_drag and has_snap)
                is_connected = path in self._connected_paths
                self._draw_point_handle(
                    px, py,
                    highlighted=is_selected or (is_drag and not has_snap),
                    is_snap=is_snap,
                    is_connected=is_connected,
                )

            # Constraint icons (highlight selected ones in blue)
            for cdata in self._constraint_data:
                self._draw_constraint_icon(cdata, self._selected_paths)

        # Force the SceneView widget to repaint immediately so that
        # selection highlights from external Stage-panel clicks appear
        # without needing to move the mouse into the viewport first.
        try:
            self._scene_view.invalidate()
        except Exception:
            pass

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

    def _draw_point_handle(
        self, x: float, y: float, highlighted: bool,
        is_snap: bool = False, is_connected: bool = False,
    ):
        """Draw a small round handle at a point position.

        Colors:
        - Snap target during drag: blue, larger (candidate — not yet bound)
        - Selected/dragging: yellow
        - Connected (has coincident constraint): green
        - Disconnected (free endpoint): red
        """
        if is_snap:
            r = VIEW_EXTENT * 0.018
            color = SNAP_HIGHLIGHT_COLOR
            thickness = 4
        elif highlighted:
            r = VIEW_EXTENT * 0.014
            color = SELECTED_COLOR
            thickness = 3
        elif is_connected:
            r = VIEW_EXTENT * 0.012
            color = POINT_CONNECTED_COLOR
            thickness = 2
        else:
            r = VIEW_EXTENT * 0.012
            color = POINT_DISCONNECTED_COLOR
            thickness = 2
        segments = 12
        pts = []
        for i in range(segments + 1):
            a = 2.0 * math.pi * i / segments
            pts.append((x + r * math.cos(a), y + r * math.sin(a), 0))
        for i in range(segments):
            sc.Line(pts[i], pts[i + 1], color=color, thickness=thickness)

    def _draw_constraint_icon(self, cdata: dict, selected_paths: Set[str] = None):
        """Draw a small icon representing a constraint near the geometry."""
        ctype = cdata.get("type", "")
        x = cdata.get("x", 0.0)
        y = cdata.get("y", 0.0)
        size = VIEW_EXTENT * 0.025

        # If this constraint's USD prim is selected, draw in blue
        cp = cdata.get("usd_path")
        is_selected = cp and selected_paths and cp in selected_paths
        color = SELECTED_COLOR if is_selected else CONSTRAINT_ICON_COLOR

        if ctype == "coincident":
            if not is_selected:
                color = COINCIDENT_COLOR
            segs = 8
            for i in range(segs):
                a1 = 2.0 * math.pi * i / segs
                a2 = 2.0 * math.pi * (i + 1) / segs
                sc.Line(
                    (x + size * 0.5 * math.cos(a1), y + size * 0.5 * math.sin(a1), 0),
                    (x + size * 0.5 * math.cos(a2), y + size * 0.5 * math.sin(a2), 0),
                    color=color, thickness=3,
                )
        elif ctype == "horizontal":
            off = size * 0.5
            sc.Line((x - off, y, 0), (x + off, y, 0), color=color, thickness=2)
        elif ctype == "vertical":
            off = size * 0.5
            sc.Line((x, y - off, 0), (x, y + off, 0), color=color, thickness=2)
        elif ctype == "perpendicular":
            off = size * 0.4
            sc.Line((x, y, 0), (x + off, y, 0), color=color, thickness=2)
            sc.Line((x, y, 0), (x, y + off, 0), color=color, thickness=2)
            sc.Line((x + off, y, 0), (x + off, y + off * 0.3, 0), color=color, thickness=1)
        elif ctype == "parallel":
            off = size * 0.3
            sc.Line((x - off, y - off, 0), (x + off, y + off, 0), color=color, thickness=2)
            sc.Line((x - off + off * 0.3, y - off, 0), (x + off + off * 0.3, y + off, 0), color=color, thickness=2)
        elif ctype == "equal":
            off = size * 0.3
            sc.Line((x - off, y + off * 0.3, 0), (x + off, y + off * 0.3, 0), color=color, thickness=2)
            sc.Line((x - off, y - off * 0.3, 0), (x + off, y - off * 0.3, 0), color=color, thickness=2)
        elif ctype == "distance":
            sc.Line((x - size, y, 0), (x + size, y, 0), color=color, thickness=2)
            sc.Line((x - size, y, 0), (x - size + size * 0.3, y + size * 0.2, 0), color=color, thickness=1)
            sc.Line((x - size, y, 0), (x - size + size * 0.3, y - size * 0.2, 0), color=color, thickness=1)
            sc.Line((x + size, y, 0), (x + size - size * 0.3, y + size * 0.2, 0), color=color, thickness=1)
            sc.Line((x + size, y, 0), (x + size - size * 0.3, y - size * 0.2, 0), color=color, thickness=1)
        elif ctype == "fixed":
            off = size * 0.4
            sc.Line((x - off, y - off, 0), (x + off, y - off, 0), color=color, thickness=2)
            for dx in [-off, 0, off]:
                sc.Line((x + dx, y - off, 0), (x + dx - off * 0.3, y - off * 1.5, 0), color=color, thickness=1)

    def _draw_constraint_label(self, x: float, y: float, labels: List[str]):
        """
        Draw compact constraint symbols at position (x, y).

        Each label is a short string like "H", "V", "∥", "⊥", "=", "d".
        Drawn as small iconic markers so they're visible without text rendering.
        """
        color = CONSTRAINT_LABEL_COLOR
        s = VIEW_EXTENT * 0.015  # symbol half-size
        gap = VIEW_EXTENT * 0.035  # gap between symbols
        # Centre the row of symbols
        total_w = len(labels) * gap
        start_x = x - total_w / 2.0 + gap / 2.0

        for j, label in enumerate(labels):
            cx = start_x + j * gap
            cy = y
            if label == "H":
                # Horizontal: small "H" shape
                sc.Line((cx - s, cy, 0), (cx + s, cy, 0), color=color, thickness=2)
                sc.Line((cx - s, cy - s, 0), (cx - s, cy + s, 0), color=color, thickness=1)
                sc.Line((cx + s, cy - s, 0), (cx + s, cy + s, 0), color=color, thickness=1)
            elif label == "V":
                # Vertical: small "V" shape
                sc.Line((cx - s, cy + s, 0), (cx, cy - s, 0), color=color, thickness=2)
                sc.Line((cx + s, cy + s, 0), (cx, cy - s, 0), color=color, thickness=2)
            elif label == "∥":
                # Parallel: two parallel slashes
                sc.Line((cx - s * 0.5, cy - s, 0), (cx - s * 0.5, cy + s, 0), color=color, thickness=2)
                sc.Line((cx + s * 0.5, cy - s, 0), (cx + s * 0.5, cy + s, 0), color=color, thickness=2)
            elif label == "⊥":
                # Perpendicular: small right-angle
                sc.Line((cx - s, cy - s, 0), (cx - s, cy + s, 0), color=color, thickness=2)
                sc.Line((cx - s, cy - s, 0), (cx + s, cy - s, 0), color=color, thickness=2)
            elif label == "=":
                # Equal: two horizontal bars
                sc.Line((cx - s, cy + s * 0.4, 0), (cx + s, cy + s * 0.4, 0), color=color, thickness=2)
                sc.Line((cx - s, cy - s * 0.4, 0), (cx + s, cy - s * 0.4, 0), color=color, thickness=2)
            elif label == "d":
                # Distance: small arrow
                sc.Line((cx - s, cy, 0), (cx + s, cy, 0), color=color, thickness=2)
                sc.Line((cx + s, cy, 0), (cx + s * 0.4, cy + s * 0.5, 0), color=color, thickness=1)
                sc.Line((cx + s, cy, 0), (cx + s * 0.4, cy - s * 0.5, 0), color=color, thickness=1)
