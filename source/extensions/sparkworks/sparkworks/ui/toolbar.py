"""
CAD Toolbar — native Omniverse Kit toolbar for the SparkWorks extension.

Uses `omni.kit.widget.toolbar.WidgetGroup` to integrate buttons into the
main viewport toolbar, just like the built-in Select / Move / Rotate tools.

Groups (left to right, separated by small spacers):
  1. Sketch:    Create/Finish Sketch
  2. Draw:      Line, Rectangle, Circle
  3. Modeling:  Extrude, Revolve, Fillet, Chamfer
  4. Boolean:   Union, Cut, Intersect
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Set

try:
    import omni.ui as ui
    from omni.kit.widget.toolbar import WidgetGroup, get_instance as get_toolbar
except ImportError:
    ui = None
    WidgetGroup = object  # fallback so class definition still parses

    def get_toolbar():          # type: ignore[misc]
        return None


# ── Icon paths ──────────────────────────────────────────────────────────────

_ICONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")

ICON = {
    "create_sketch":      os.path.join(_ICONS_DIR, "toolbar_create_sketch.svg"),
    "finish_sketch":      os.path.join(_ICONS_DIR, "toolbar_finish_sketch.svg"),
    "select":             os.path.join(_ICONS_DIR, "toolbar_select.svg"),
    "line":               os.path.join(_ICONS_DIR, "toolbar_line.svg"),
    "rectangle":          os.path.join(_ICONS_DIR, "toolbar_rectangle.svg"),
    "circle":             os.path.join(_ICONS_DIR, "toolbar_circle.svg"),
    "coincident":         os.path.join(_ICONS_DIR, "toolbar_coincident.svg"),
    "horizontal":         os.path.join(_ICONS_DIR, "toolbar_horizontal.svg"),
    "vertical":           os.path.join(_ICONS_DIR, "toolbar_vertical.svg"),
    "distance":           os.path.join(_ICONS_DIR, "toolbar_distance.svg"),
    "perpendicular":      os.path.join(_ICONS_DIR, "toolbar_perpendicular.svg"),
    "parallel":           os.path.join(_ICONS_DIR, "toolbar_parallel.svg"),
    "equal":              os.path.join(_ICONS_DIR, "toolbar_equal.svg"),
    "extrude":            os.path.join(_ICONS_DIR, "toolbar_extrude.svg"),
    "revolve":            os.path.join(_ICONS_DIR, "toolbar_revolve.svg"),
    "fillet":             os.path.join(_ICONS_DIR, "toolbar_fillet.svg"),
    "chamfer":            os.path.join(_ICONS_DIR, "toolbar_chamfer.svg"),
    "boolean_union":      os.path.join(_ICONS_DIR, "toolbar_boolean_union.svg"),
    "boolean_cut":        os.path.join(_ICONS_DIR, "toolbar_boolean_cut.svg"),
    "boolean_intersect":  os.path.join(_ICONS_DIR, "toolbar_boolean_intersect.svg"),
}


# ── WidgetGroup implementation ─────────────────────────────────────────────

class SparkWorksToolbarGroup(WidgetGroup):
    """
    A `WidgetGroup` that adds SparkWorks CAD buttons to the native toolbar.
    """

    def __init__(self, owner: "CadToolbar"):
        super().__init__()
        self._owner = owner  # look up callbacks on the facade dynamically

        # State
        self._sketch_mode = False
        self._active_tool_name: Optional[str] = None
        self._updating_programmatically = False  # guard against re-entrancy

        # Widget references (set in create())
        self._btn_sketch: Optional[ui.ToolButton] = None
        self._btn_select: Optional[ui.ToolButton] = None
        self._btn_line: Optional[ui.ToolButton] = None
        self._btn_rect: Optional[ui.ToolButton] = None
        self._btn_circle: Optional[ui.ToolButton] = None
        self._subs: list = []

        # Visibility groups (populated in create(), hidden by default)
        self._draw_tools: list = []        # [gap1, select, line, rect, circle]
        self._constraint_gap: Optional[ui.Spacer] = None
        self._constraint_buttons: Dict[str, ui.ToolButton] = {}  # name → button
        self._op_tools: list = []          # [gap2, extrude, revolve, fillet, chamfer]
        self._bool_tools: list = []        # [gap3, union, cut, intersect]

    # -- WidgetGroup interface ------------------------------------------------

    def get_style(self):
        """Return name-keyed icon styles consumed by the ToolButtons."""
        style = {}
        # Sketch button — toggles between create and finish icons
        style["Button.Image::sw_sketch"] = {
            "image_url": ICON["create_sketch"],
        }
        style["Button.Image::sw_sketch:checked"] = {
            "image_url": ICON["finish_sketch"],
        }
        # Select tool
        style["Button.Image::sw_select"] = {"image_url": ICON["select"]}
        style["Button.Image::sw_select:checked"] = {"image_url": ICON["select"]}
        # Drawing tools
        for name in ("line", "rectangle", "circle"):
            style[f"Button.Image::sw_{name}"] = {"image_url": ICON[name]}
            style[f"Button.Image::sw_{name}:checked"] = {"image_url": ICON[name]}
        # Constraint tools
        for name in ("coincident", "horizontal", "vertical", "distance",
                      "perpendicular", "parallel", "equal"):
            style[f"Button.Image::sw_{name}"] = {"image_url": ICON[name]}
            style[f"Button.Image::sw_{name}:checked"] = {"image_url": ICON[name]}
        # 3D operations (non-toggle, so same icon for both states)
        for name in ("extrude", "revolve", "fillet", "chamfer"):
            style[f"Button.Image::sw_{name}"] = {"image_url": ICON[name]}
            style[f"Button.Image::sw_{name}:checked"] = {"image_url": ICON[name]}
        # Boolean operations
        for name in ("boolean_union", "boolean_cut", "boolean_intersect"):
            style[f"Button.Image::sw_{name}"] = {"image_url": ICON[name]}
            style[f"Button.Image::sw_{name}:checked"] = {"image_url": ICON[name]}
        return style

    def create(self, default_size):
        """
        Build the toolbar widgets.  Returns a dict of name → widget so the
        toolbar can lay them out.
        """
        widgets = {}
        sz = default_size

        # ── Separator before our group ──
        widgets["sw_sep0"] = ui.Line(
            width=1, height=sz * 0.6,
            alignment=ui.Alignment.CENTER,
            style={"color": 0xFF555555},
        )

        # ── Sketch button (toggle: create / finish) ──
        btn = ui.ToolButton(name="sw_sketch", width=sz, height=sz, tooltip="Create Sketch")
        sub = btn.model.subscribe_value_changed_fn(self._on_sketch_clicked)
        self._subs.append(sub)
        self._btn_sketch = btn
        widgets["sw_sketch"] = btn

        # ── Small gap (draw tools group) ──
        gap1 = ui.Spacer(width=6)
        widgets["sw_gap1"] = gap1
        self._draw_tools = [gap1]

        # ── Select (pointer) tool ──
        btn_sel = ui.ToolButton(
            name="sw_select", width=sz, height=sz, tooltip="Select / Drag"
        )
        sub = btn_sel.model.subscribe_value_changed_fn(
            lambda m: self._on_tool_clicked("select", m)
        )
        self._subs.append(sub)
        self._btn_select = btn_sel
        widgets["sw_select"] = btn_sel
        self._draw_tools.append(btn_sel)

        # ── Drawing tools ──
        for name, tooltip, attr in [
            ("line",      "Line",      "_btn_line"),
            ("rectangle", "Rectangle", "_btn_rect"),
            ("circle",    "Circle",    "_btn_circle"),
        ]:
            b = ui.ToolButton(name=f"sw_{name}", width=sz, height=sz, tooltip=tooltip)
            sub = b.model.subscribe_value_changed_fn(
                lambda m, n=name: self._on_tool_clicked(n, m)
            )
            self._subs.append(sub)
            setattr(self, attr, b)
            widgets[f"sw_{name}"] = b
            self._draw_tools.append(b)

        # ── Small gap (constraint tools group) ──
        gap_c = ui.Spacer(width=4)
        widgets["sw_gap_c"] = gap_c
        self._constraint_gap = gap_c

        # ── Constraint tools (fire-and-forget, individually controllable) ──
        self._constraint_buttons = {}
        for name, tooltip in [
            ("coincident",    "Coincident"),
            ("horizontal",    "Horizontal"),
            ("vertical",      "Vertical"),
            ("distance",      "Distance"),
            ("perpendicular", "Perpendicular"),
            ("parallel",      "Parallel"),
            ("equal",         "Equal"),
        ]:
            b = ui.ToolButton(name=f"sw_{name}", width=sz, height=sz, tooltip=tooltip)
            sub = b.model.subscribe_value_changed_fn(
                lambda m, n=name: self._on_constraint_clicked(n, m)
            )
            self._subs.append(sub)
            widgets[f"sw_{name}"] = b
            self._constraint_buttons[name] = b

        # ── Small gap (operation tools group) ──
        gap2 = ui.Spacer(width=6)
        widgets["sw_gap2"] = gap2
        self._op_tools = [gap2]

        # ── 3D operation buttons (fire-and-forget, not toggle) ──
        for name, tooltip in [
            ("extrude", "Extrude"),
            ("revolve", "Revolve"),
            ("fillet",  "Fillet"),
            ("chamfer", "Chamfer"),
        ]:
            b = ui.ToolButton(name=f"sw_{name}", width=sz, height=sz, tooltip=tooltip)
            sub = b.model.subscribe_value_changed_fn(
                lambda m, n=name: self._on_op_clicked(n, m)
            )
            self._subs.append(sub)
            widgets[f"sw_{name}"] = b
            self._op_tools.append(b)

        # ── Small gap (boolean tools group) ──
        gap3 = ui.Spacer(width=6)
        widgets["sw_gap3"] = gap3
        self._bool_tools = [gap3]

        # ── Boolean operation buttons (fire-and-forget) ──
        for name, tooltip in [
            ("boolean_union",     "Boolean Union"),
            ("boolean_cut",       "Boolean Cut"),
            ("boolean_intersect", "Boolean Intersect"),
        ]:
            b = ui.ToolButton(name=f"sw_{name}", width=sz, height=sz, tooltip=tooltip)
            sub = b.model.subscribe_value_changed_fn(
                lambda m, n=name: self._on_bool_clicked(n, m)
            )
            self._subs.append(sub)
            widgets[f"sw_{name}"] = b
            self._bool_tools.append(b)

        # Apply current state.  If create() is called again by the toolbar
        # framework (e.g. on viewport resize), this restores the correct
        # visibility rather than resetting everything to hidden.
        if self._sketch_mode:
            self.set_draw_tools_visible(True)
            self.set_constraint_tools_visible(False)
            self.set_op_tools_visible(False)
            self.set_bool_tools_visible(False)
        else:
            self.set_draw_tools_visible(False)
            self.set_constraint_tools_visible(False)
            self.set_op_tools_visible(False)
            self.set_bool_tools_visible(False)

        return widgets

    def clean(self):
        self._subs.clear()
        self._btn_sketch = None
        self._btn_select = None
        self._btn_line = None
        self._btn_rect = None
        self._btn_circle = None
        self._draw_tools.clear()
        self._constraint_buttons.clear()
        self._constraint_gap = None
        self._op_tools.clear()
        self._bool_tools.clear()
        super().clean()

    # -- Visibility control ---------------------------------------------------

    def set_draw_tools_visible(self, visible: bool):
        """Show or hide the drawing-tool buttons (Select, Line, Rectangle, Circle)."""
        for w in self._draw_tools:
            if w is not None:
                w.visible = visible

    def set_constraint_tools_visible(self, visible: bool):
        """Show or hide ALL constraint buttons (and the gap before them)."""
        if self._constraint_gap is not None:
            self._constraint_gap.visible = visible
        for b in self._constraint_buttons.values():
            if b is not None:
                b.visible = visible

    def set_applicable_constraints(self, names: Set[str]):
        """
        Show only the constraint buttons whose names are in *names*.

        Pass an empty set to hide all constraint buttons.
        Any button whose name is NOT in the set is hidden.
        The gap spacer is hidden when no constraints are applicable.
        """
        any_visible = False
        for name, btn in self._constraint_buttons.items():
            show = name in names
            btn.visible = show
            if show:
                any_visible = True
        if self._constraint_gap is not None:
            self._constraint_gap.visible = any_visible

    # Kept for backward compatibility
    def set_sketch_tools_visible(self, visible: bool):
        """Show or hide ALL sketch-related buttons (draw + constraints)."""
        self.set_draw_tools_visible(visible)
        self.set_constraint_tools_visible(visible)

    def set_op_tools_visible(self, visible: bool):
        """Show or hide the 3D operation buttons (Extrude, Revolve, etc.)."""
        for w in self._op_tools:
            if w is not None:
                w.visible = visible

    def set_bool_tools_visible(self, visible: bool):
        """Show or hide the Boolean operation buttons (Union, Cut, Intersect)."""
        for w in self._bool_tools:
            if w is not None:
                w.visible = visible

    # -- Click handlers -------------------------------------------------------

    def _on_sketch_clicked(self, model):
        """Toggle between create and finish sketch."""
        if self._updating_programmatically:
            return  # ignore programmatic set_value calls
        checked = model.get_value_as_bool()
        o = self._owner
        if checked:
            if o.on_create_sketch:
                o.on_create_sketch()
        else:
            if o.on_finish_sketch:
                o.on_finish_sketch()

    def _on_tool_clicked(self, name: str, model):
        """Toggle drawing / select tool.  Un-check others."""
        if self._updating_programmatically:
            return
        checked = model.get_value_as_bool()
        o = self._owner
        cb_map = {
            "select":    o.on_tool_select,
            "line":      o.on_tool_line,
            "rectangle": o.on_tool_rectangle,
            "circle":    o.on_tool_circle,
        }
        if checked:
            # Uncheck sibling tools
            for tool_name, btn in [
                ("select", self._btn_select),
                ("line", self._btn_line),
                ("rectangle", self._btn_rect),
                ("circle", self._btn_circle),
            ]:
                if tool_name != name and btn is not None:
                    btn.model.set_value(False)
            cb = cb_map.get(name)
            if cb:
                cb()

    def _on_op_clicked(self, name: str, model):
        """Fire-and-forget operation button.  Un-check immediately."""
        if model.get_value_as_bool():
            model.set_value(False)
            o = self._owner
            cb_map = {
                "extrude": o.on_extrude,
                "revolve": o.on_revolve,
                "fillet":  o.on_fillet,
                "chamfer": o.on_chamfer,
            }
            cb = cb_map.get(name)
            if cb:
                cb()

    def _on_constraint_clicked(self, name: str, model):
        """Fire-and-forget constraint button.  Un-check immediately."""
        if model.get_value_as_bool():
            model.set_value(False)
            o = self._owner
            cb = getattr(o, f"on_constraint_{name}", None)
            if cb:
                cb()

    def _on_bool_clicked(self, name: str, model):
        """Fire-and-forget boolean button.  Un-check immediately."""
        if model.get_value_as_bool():
            model.set_value(False)
            o = self._owner
            cb_map = {
                "boolean_union":     o.on_boolean_union,
                "boolean_cut":       o.on_boolean_cut,
                "boolean_intersect": o.on_boolean_intersect,
            }
            cb = cb_map.get(name)
            if cb:
                cb()

    # -- Public state API (called by extension.py) ----------------------------

    def set_sketch_mode(self, active: bool):
        self._sketch_mode = active
        self._updating_programmatically = True
        try:
            if self._btn_sketch is not None:
                self._btn_sketch.model.set_value(active)
                self._btn_sketch.tooltip = (
                    "Finish Sketch" if active else "Create Sketch"
                )
        finally:
            self._updating_programmatically = False
        # Show drawing tools in sketch mode, hide operations + booleans.
        # Constraint buttons start hidden; the extension shows applicable
        # ones based on what the user selects inside the sketch.
        self.set_draw_tools_visible(active)
        if not active:
            self.set_constraint_tools_visible(False)
        if active:
            self.set_op_tools_visible(False)
            self.set_bool_tools_visible(False)

    def set_active_tool(self, tool_name: Optional[str]):
        self._active_tool_name = tool_name
        self._updating_programmatically = True
        try:
            # When tool_name is None, highlight "select" as the default
            effective = tool_name if tool_name else "select"
            for name, btn in [
                ("select", self._btn_select),
                ("line", self._btn_line),
                ("rectangle", self._btn_rect),
                ("circle", self._btn_circle),
            ]:
                if btn is not None:
                    btn.model.set_value(name == effective)
        finally:
            self._updating_programmatically = False


# ── Facade class (preserves extension.py API) ──────────────────────────────

class CadToolbar:
    """
    High-level wrapper that owns a `SparkWorksToolbarGroup` and registers it
    with the native Omniverse toolbar.  Exposes the same public API that
    `extension.py` already uses, so no call-site changes are needed.
    """

    def __init__(self):
        self._group = SparkWorksToolbarGroup(owner=self)
        self._registered = False

        # Proxy callbacks (extension.py sets these)
        self.on_create_sketch: Optional[Callable] = None
        self.on_finish_sketch: Optional[Callable] = None
        self.on_add_plane: Optional[Callable] = None
        self.on_tool_select: Optional[Callable] = None
        self.on_tool_line: Optional[Callable] = None
        self.on_tool_rectangle: Optional[Callable] = None
        self.on_tool_circle: Optional[Callable] = None
        self.on_extrude: Optional[Callable] = None
        self.on_revolve: Optional[Callable] = None
        self.on_fillet: Optional[Callable] = None
        self.on_chamfer: Optional[Callable] = None
        self.on_boolean_union: Optional[Callable] = None
        self.on_boolean_cut: Optional[Callable] = None
        self.on_boolean_intersect: Optional[Callable] = None
        self.on_rebuild_all: Optional[Callable] = None
        self.on_clear_all: Optional[Callable] = None

        # Constraint tool callbacks
        self.on_constraint_coincident: Optional[Callable] = None
        self.on_constraint_horizontal: Optional[Callable] = None
        self.on_constraint_vertical: Optional[Callable] = None
        self.on_constraint_distance: Optional[Callable] = None
        self.on_constraint_perpendicular: Optional[Callable] = None
        self.on_constraint_parallel: Optional[Callable] = None
        self.on_constraint_equal: Optional[Callable] = None

        # Status text (no-op placeholder; the native toolbar doesn't show a
        # status line — we log to the console and/or notification area instead)
        self._hint_text = ""

        # Parameter models (kept so property getters work; not in the toolbar UI)
        self._extrude_dist_model = None
        self._revolve_angle_model = None
        self._fillet_radius_model = None
        self._chamfer_length_model = None

    # -- Compat: window property (returns None — no longer a window) ----------
    @property
    def window(self):
        return None

    # -- Build / Destroy ------------------------------------------------------

    def build(self):
        """Register the widget group with the native toolbar."""
        toolbar = get_toolbar()
        if toolbar is None:
            print("[SparkWorks] omni.kit.widget.toolbar not available — toolbar skipped")
            return
        toolbar.add_widget(self._group, priority=200)
        self._registered = True

    def destroy(self):
        if self._registered:
            try:
                toolbar = get_toolbar()
                if toolbar:
                    toolbar.remove_widget(self._group)
            except Exception:
                pass
            self._group.clean()
            self._registered = False

    # -- State API (proxied to the WidgetGroup) --------------------------------

    def set_sketch_mode(self, active: bool):
        self._group.set_sketch_mode(active)

    def set_active_tool(self, tool_name: Optional[str]):
        self._group.set_active_tool(tool_name)

    def set_draw_tools_visible(self, visible: bool):
        """Show or hide the drawing-tool buttons (Select, Line, Rect, Circle)."""
        self._group.set_draw_tools_visible(visible)

    def set_constraint_tools_visible(self, visible: bool):
        """Show or hide ALL constraint buttons."""
        self._group.set_constraint_tools_visible(visible)

    def set_applicable_constraints(self, names: Set[str]):
        """Show only constraint buttons whose names are in *names*."""
        self._group.set_applicable_constraints(names)

    def set_sketch_tools_visible(self, visible: bool):
        """Show or hide ALL sketch-related buttons (draw + constraints)."""
        self._group.set_sketch_tools_visible(visible)

    def set_op_tools_visible(self, visible: bool):
        """Show or hide the 3D operation buttons (Extrude, Revolve, etc.)."""
        self._group.set_op_tools_visible(visible)

    def set_profiles_selected(self, has_profiles: bool):
        """Show or hide the 3D operation buttons based on profile selection."""
        self._group.set_op_tools_visible(has_profiles)

    def set_bool_tools_visible(self, visible: bool):
        """Show or hide the Boolean operation buttons."""
        self._group.set_bool_tools_visible(visible)

    def set_status(self, message: str):
        # The native toolbar has no status label.  Forward to console.
        self._hint_text = message

    def set_plane_hint(self, text: str):
        self._hint_text = text

    # -- Value getters (3D operations) ----------------------------------------

    @property
    def extrude_distance(self) -> float:
        return self._extrude_dist_model.as_float if self._extrude_dist_model else 10.0

    @property
    def revolve_angle(self) -> float:
        return self._revolve_angle_model.as_float if self._revolve_angle_model else 360.0

    @property
    def fillet_radius(self) -> float:
        return self._fillet_radius_model.as_float if self._fillet_radius_model else 1.0

    @property
    def chamfer_length(self) -> float:
        return self._chamfer_length_model.as_float if self._chamfer_length_model else 1.0

