"""
Timeline Panel — displays the parametric feature history.

A dockable panel at the bottom of the viewport that shows each feature as
a clickable node in a horizontal strip.  A **playback marker** (vertical
blue bar) sits between features and controls the scrub position.  The
marker is independent of feature selection:

- **Click a feature** → selects it (shows properties), does NOT move marker.
- **Click between features** → moves the marker there (rebuilds geometry).
- **|< / >| buttons** → jump to beginning / end.
- **Right-click a feature** → context menu (suppress / delete).
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

try:
    import omni.ui as ui
except ImportError:
    ui = None


# ── Icon paths (FreeCAD-sourced SVGs) ───────────────────────────────────────

_ICONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")

FEATURE_ICON_FILES = {
    "SKETCH":             os.path.join(_ICONS_DIR, "toolbar_create_sketch.svg"),
    "EXTRUDE":            os.path.join(_ICONS_DIR, "toolbar_extrude.svg"),
    "REVOLVE":            os.path.join(_ICONS_DIR, "toolbar_revolve.svg"),
    "FILLET":             os.path.join(_ICONS_DIR, "toolbar_fillet.svg"),
    "CHAMFER":            os.path.join(_ICONS_DIR, "toolbar_chamfer.svg"),
    "BOOLEAN_JOIN":       os.path.join(_ICONS_DIR, "toolbar_boolean_union.svg"),
    "BOOLEAN_CUT":        os.path.join(_ICONS_DIR, "toolbar_boolean_cut.svg"),
    "BOOLEAN_INTERSECT":  os.path.join(_ICONS_DIR, "toolbar_boolean_intersect.svg"),
}

# ── Colours ─────────────────────────────────────────────────────────────────

MARKER_COLOR = 0xFF44AAFF          # bright blue marker
SLOT_BG_HOVER = 0xFF444444         # slot on hover
SELECTED_BORDER = 0xFFFF8800       # orange border for selected feature
ACTIVE_BG = 0xFFBBBBBB             # light gray for features before/at marker
INACTIVE_BG = 0xFF888888           # dimmed gray for features after marker
SUPPRESSED_BG = 0xFF555555         # dark gray for suppressed

# ── Dimensions ──────────────────────────────────────────────────────────────

SLOT_W = 6               # width of a clickable marker slot
ICON_SIZE = 20           # icon height/width
FEATURE_W = 28           # box around icon
FEATURE_H = 28

# ── Shared styles (NO Button overrides — those are set per-widget) ──────────

TIMELINE_STYLE = {
    "Label": {"font_size": 13},
}

# Reusable per-widget style dicts ────────────────────────────────────────────

_HEADER_BTN = {
    "Button": {
        "background_color": 0xFF2A2A2A,
        "border_color": 0xFF444444,
        "border_width": 1,
        "border_radius": 3,
        "padding": 2,
        "margin": 1,
    },
    "Button:hovered": {"background_color": 0xFF3A3A3A},
    "Button.Label": {"font_size": 12},
}


def _slot_rect_style(active: bool):
    """Style dict for a marker-slot Rectangle."""
    return {
        "background_color": MARKER_COLOR if active else 0x00000000,
        "border_width": 0,
        "border_radius": 1 if active else 0,
    }


def _feature_style(bg, border_color, border_width):
    return {
        "Button": {
            "background_color": bg,
            "border_color": border_color,
            "border_width": border_width,
            "border_radius": 3,
            "padding": 0,
            "margin": 0,
        },
        "Button:hovered": {
            "background_color": bg,
            "border_color": MARKER_COLOR,
            "border_width": 2,
            "padding": 0,
            "margin": 0,
        },
        "Button.Label": {"font_size": 1, "color": 0x00000000},
        "Button.Image": {"image_url": ""},
    }


class TimelinePanel:
    """Dockable timeline panel with a Fusion 360-style playback marker."""

    def __init__(self):
        self._window: Optional[ui.Window] = None
        self._features: list = []
        self._marker_pos: int = -1
        self._selected_idx: int = -1
        self._ctx_menu = None

        # Callbacks (set by extension)
        self.on_feature_select: Optional[Callable] = None
        self.on_marker_moved: Optional[Callable] = None
        self.on_feature_delete: Optional[Callable] = None
        self.on_feature_suppress: Optional[Callable] = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def window(self):
        return self._window

    @property
    def marker_position(self) -> int:
        return self._marker_pos

    @property
    def selected_index(self) -> int:
        return self._selected_idx

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def build(self):
        if ui is None:
            return
        self._window = ui.Window(
            "CAD Timeline",
            width=400,
            height=130,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.frame.set_style(TIMELINE_STYLE)
        self._build_empty()

    def destroy(self):
        self._ctx_menu = None
        if self._window:
            self._window.destroy()
            self._window = None

    # ── Public API ──────────────────────────────────────────────────────────

    def update_features(self, features: list, marker_pos: int = -1,
                        selected_idx: int = -1):
        if ui is None or self._window is None:
            return
        self._features = list(features)
        if features and marker_pos >= len(features):
            marker_pos = len(features) - 1
        self._marker_pos = marker_pos
        self._selected_idx = selected_idx
        with self._window.frame:
            if not features:
                self._build_empty()
            else:
                self._build_full()

    def set_selected(self, index: int):
        self._selected_idx = index
        if self._features:
            self.update_features(self._features, self._marker_pos, index)

    # ── Build helpers ───────────────────────────────────────────────────────

    def _build_empty(self):
        with self._window.frame:
            with ui.VStack():
                ui.Spacer()
                ui.Label(
                    "No features yet. Create a sketch to begin.",
                    alignment=ui.Alignment.CENTER,
                    style={"color": 0xFF888888, "font_size": 14},
                )
                ui.Spacer()

    def _build_full(self):
        n = len(self._features)
        with ui.VStack(spacing=0):
            # Header row
            with ui.HStack(height=24):
                ui.Label("Feature Timeline",
                         style={"font_size": 14, "color": 0xFFCCCCCC})
                ui.Spacer()
                ui.Button("|<", width=28, height=22,
                          clicked_fn=self._go_to_start,
                          tooltip="Go to beginning",
                          style=_HEADER_BTN)
                ui.Button(">|", width=28, height=22,
                          clicked_fn=self._go_to_end,
                          tooltip="Go to end",
                          style=_HEADER_BTN)

            ui.Spacer(height=4)

            # Scrollable timeline strip
            with ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                height=FEATURE_H + 8,
            ):
                with ui.HStack(spacing=0, height=FEATURE_H):
                    # Layout: [slot -1] [feat 0] [slot 0] [feat 1] [slot 1] …
                    self._build_slot(-1)
                    for i in range(n):
                        self._build_feature_node(i, self._features[i])
                        self._build_slot(i)
                    ui.Spacer(width=4)

    # ── Marker slot ─────────────────────────────────────────────────────────

    def _build_slot(self, position: int):
        """Thin clickable column between features. Active slot = blue bar."""
        is_active = (position == self._marker_pos)
        rect = ui.Rectangle(
            width=SLOT_W, height=FEATURE_H,
            style=_slot_rect_style(is_active),
        )
        rect.set_mouse_released_fn(
            lambda x, y, btn, m, p=position: self._on_slot_clicked(p) if btn == 0 else None
        )

    # ── Feature node ────────────────────────────────────────────────────────

    def _build_feature_node(self, index: int, feature):
        is_selected = (index == self._selected_idx)
        is_before_marker = (index <= self._marker_pos)

        if feature.suppressed:
            bg = SUPPRESSED_BG
        elif is_before_marker:
            bg = ACTIVE_BG
        else:
            bg = INACTIVE_BG

        border_color = SELECTED_BORDER if is_selected else 0xFFAAAAAA
        border_w = 2 if is_selected else 1

        icon_key = "SKETCH"
        if feature.is_operation and feature.operation:
            icon_key = feature.operation.op_type.name
        icon_path = FEATURE_ICON_FILES.get(icon_key)

        style = _feature_style(bg, border_color, border_w)

        # If we have a valid icon, set it via Button.Image style
        if icon_path and os.path.isfile(icon_path):
            style["Button.Image"] = {"image_url": icon_path}

        btn = ui.Button(
            "",
            width=FEATURE_W, height=FEATURE_H,
            image_width=ICON_SIZE, image_height=ICON_SIZE,
            clicked_fn=lambda idx=index: self._on_feature_click(idx),
            tooltip=feature.name,
            style=style,
        )
        btn.set_mouse_pressed_fn(
            lambda x, y, b, m, idx=index, sup=feature.suppressed:
                self._on_feature_right_click(x, y, b, idx, sup)
        )

    # ── Interaction ─────────────────────────────────────────────────────────

    def _on_slot_clicked(self, position: int):
        if position == self._marker_pos:
            return
        self._marker_pos = position
        self.update_features(self._features, self._marker_pos, self._selected_idx)
        if self.on_marker_moved:
            self.on_marker_moved(position)

    def _go_to_start(self):
        if self._marker_pos != -1:
            self._on_slot_clicked(-1)

    def _go_to_end(self):
        end = len(self._features) - 1 if self._features else -1
        if self._marker_pos != end:
            self._on_slot_clicked(end)

    def _on_feature_click(self, index: int):
        self._selected_idx = index
        self.update_features(self._features, self._marker_pos, index)
        if self.on_feature_select:
            self.on_feature_select(index)

    def _on_feature_right_click(self, x, y, button, index, currently_suppressed):
        if button != 1:
            return
        self._ctx_menu = ui.Menu("Feature Actions")
        with self._ctx_menu:
            suppress_label = "Unsuppress" if currently_suppressed else "Suppress"
            ui.MenuItem(
                suppress_label,
                triggered_fn=lambda idx=index, sup=currently_suppressed:
                    self._on_suppress(idx, sup),
            )
            ui.Separator()
            ui.MenuItem(
                "Delete",
                triggered_fn=lambda idx=index: self._on_delete(idx),
            )
        self._ctx_menu.show()

    def _on_delete(self, index: int):
        if self.on_feature_delete:
            self.on_feature_delete(index)

    def _on_suppress(self, index: int, currently_suppressed: bool):
        if self.on_feature_suppress:
            self.on_feature_suppress(index, not currently_suppressed)
