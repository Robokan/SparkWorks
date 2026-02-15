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
from typing import Callable, Optional

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
    "line":               os.path.join(_ICONS_DIR, "toolbar_line.svg"),
    "rectangle":          os.path.join(_ICONS_DIR, "toolbar_rectangle.svg"),
    "circle":             os.path.join(_ICONS_DIR, "toolbar_circle.svg"),
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

        # Widget references (set in create())
        self._btn_sketch: Optional[ui.ToolButton] = None
        self._btn_line: Optional[ui.ToolButton] = None
        self._btn_rect: Optional[ui.ToolButton] = None
        self._btn_circle: Optional[ui.ToolButton] = None
        self._subs: list = []

        # Visibility groups (populated in create(), hidden by default)
        self._sketch_tools: list = []   # [gap1, line, rect, circle]
        self._op_tools: list = []       # [gap2, extrude, revolve, fillet, chamfer]
        self._bool_tools: list = []     # [gap3, union, cut, intersect]

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
        # Drawing tools
        for name in ("line", "rectangle", "circle"):
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

        # ── Small gap (sketch tools group) ──
        gap1 = ui.Spacer(width=6)
        widgets["sw_gap1"] = gap1
        self._sketch_tools = [gap1]

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
            self._sketch_tools.append(b)

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

        # Start with all groups hidden — only "Create Sketch" is visible
        self.set_sketch_tools_visible(False)
        self.set_op_tools_visible(False)
        self.set_bool_tools_visible(False)

        return widgets

    def clean(self):
        self._subs.clear()
        self._btn_sketch = None
        self._btn_line = None
        self._btn_rect = None
        self._btn_circle = None
        self._sketch_tools.clear()
        self._op_tools.clear()
        self._bool_tools.clear()
        super().clean()

    # -- Visibility control ---------------------------------------------------

    def set_sketch_tools_visible(self, visible: bool):
        """Show or hide the drawing-tool buttons (Line, Rectangle, Circle)."""
        for w in self._sketch_tools:
            if w is not None:
                w.visible = visible

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
        checked = model.get_value_as_bool()
        o = self._owner
        if checked:
            if o.on_create_sketch:
                o.on_create_sketch()
        else:
            if o.on_finish_sketch:
                o.on_finish_sketch()

    def _on_tool_clicked(self, name: str, model):
        """Toggle drawing tool.  Un-check others."""
        checked = model.get_value_as_bool()
        o = self._owner
        cb_map = {
            "line":      o.on_tool_line,
            "rectangle": o.on_tool_rectangle,
            "circle":    o.on_tool_circle,
        }
        if checked:
            # Uncheck sibling tools
            for tool_name, btn in [
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
        if self._btn_sketch is not None:
            self._btn_sketch.model.set_value(active)
            self._btn_sketch.tooltip = (
                "Finish Sketch" if active else "Create Sketch"
            )
        # Show drawing tools in sketch mode, hide operations + booleans
        self.set_sketch_tools_visible(active)
        if active:
            self.set_op_tools_visible(False)
            self.set_bool_tools_visible(False)

    def set_active_tool(self, tool_name: Optional[str]):
        self._active_tool_name = tool_name
        for name, btn in [
            ("line", self._btn_line),
            ("rectangle", self._btn_rect),
            ("circle", self._btn_circle),
        ]:
            if btn is not None:
                btn.model.set_value(name == tool_name)


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

    def set_sketch_tools_visible(self, visible: bool):
        """Show or hide the drawing-tool buttons (Line, Rectangle, Circle)."""
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

