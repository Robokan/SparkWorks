"""
Timeline Panel — displays the parametric feature history.

A dockable panel at the bottom of the viewport that shows each feature as
a clickable node in a horizontal strip.  A **playback marker** (vertical
line with a ball handle on top) sits between features and controls the
scrub position.  The marker is independent of feature selection:

- **Click a feature** → selects it (shows properties), does NOT move marker.
- **Click a marker slot** → jumps the marker there (rebuilds geometry).
- **Drag the marker ball** horizontally to scrub through the timeline.
- **|< / >| buttons** → jump to beginning / end.
- **Right-click a feature** → context menu (suppress / delete).
"""

from __future__ import annotations

from typing import Callable, List, Optional

try:
    import omni.ui as ui
except ImportError:
    ui = None

try:
    import omni.kit.app
    _HAS_KIT = True
except ImportError:
    _HAS_KIT = False


# ── Colours ─────────────────────────────────────────────────────────────────

MARKER_COLOR = 0xFF44AAFF          # bright blue marker line + ball
SLOT_DOT_COLOR = 0x44FFFFFF        # faint dot for idle slots
SELECTED_BORDER = 0xFFFF8800       # orange border for selected feature
ACTIVE_BG = 0xFF1A3355             # feature before/at the marker
INACTIVE_BG = 0xFF2A2A2A           # feature after the marker
SUPPRESSED_BG = 0xFF1A1A1A
SUPPRESSED_FG = 0xFF666666
ACTIVE_FG = 0xFFCCCCCC
INACTIVE_FG = 0xFF888888

# ── Dimensions ──────────────────────────────────────────────────────────────

SLOT_TOTAL_W = 20        # total width of a marker slot column
MARKER_LINE_W = 4        # width of the active marker line
BALL_DIAM = 14           # diameter of the ball handle
FEATURE_W = 96           # width of a feature button
FEATURE_H = 56           # height of a feature button
ROW_H = FEATURE_H + 4   # total height of the timeline strip row

# ── Feature icons ───────────────────────────────────────────────────────────

FEATURE_ICONS = {
    "SKETCH": "[S]",
    "EXTRUDE": "[E]",
    "REVOLVE": "[R]",
    "FILLET": "[F]",
    "CHAMFER": "[C]",
    "BOOLEAN_JOIN": "[+]",
    "BOOLEAN_CUT": "[-]",
    "BOOLEAN_INTERSECT": "[&]",
}

TIMELINE_STYLE = {
    "Label": {"font_size": 13},
    "Button": {
        "background_color": 0xFF2A2A2A,
        "border_color": 0xFF444444,
        "border_width": 1,
        "border_radius": 3,
        "padding": 4,
        "margin": 1,
    },
    "Button:hovered": {
        "background_color": 0xFF3A3A3A,
    },
}


class TimelinePanel:
    """
    Dockable timeline panel with a Fusion 360-style playback marker.
    """

    def __init__(self):
        self._window: Optional[ui.Window] = None
        self._features: list = []
        self._marker_pos: int = -1
        self._selected_idx: int = -1
        self._ctx_menu = None

        # Drag state
        self._dragging: bool = False
        self._scroll_frame = None
        self._update_sub = None

        # Slot layout: list of (slot_position_int, x_left, x_right)
        # so we can map any x coordinate to the nearest slot
        self._slot_ranges: list = []

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
            height=140,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.frame.set_style(TIMELINE_STYLE)
        self._build_empty()

    def destroy(self):
        self._stop_drag()
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
        self._slot_ranges.clear()
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
            # Header
            with ui.HStack(height=24):
                ui.Label("Feature Timeline",
                         style={"font_size": 14, "color": 0xFFCCCCCC})
                ui.Spacer()
                ui.Button("|<", width=28, height=22,
                          clicked_fn=self._go_to_start,
                          tooltip="Go to beginning",
                          style={"Button.Label": {"font_size": 12}})
                ui.Button(">|", width=28, height=22,
                          clicked_fn=self._go_to_end,
                          tooltip="Go to end",
                          style={"Button.Label": {"font_size": 12}})

            ui.Spacer(height=4)

            # Scrollable timeline strip with FIXED height
            self._scroll_frame = ui.ScrollingFrame(
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                height=ROW_H,
            )
            with self._scroll_frame:
                with ui.HStack(spacing=0, height=ROW_H):
                    # Build the slot layout ranges as we go
                    x = 0.0

                    # Leading slot
                    self._slot_ranges.append((-1, x, x + SLOT_TOTAL_W))
                    self._build_slot(-1)
                    x += SLOT_TOTAL_W

                    for i in range(n):
                        self._build_feature_node(i, self._features[i])
                        x += FEATURE_W

                        self._slot_ranges.append((i, x, x + SLOT_TOTAL_W))
                        self._build_slot(i)
                        x += SLOT_TOTAL_W

                    ui.Spacer(width=10)

    # ── Marker slot ─────────────────────────────────────────────────────────

    def _build_slot(self, position: int):
        is_active = (position == self._marker_pos)

        with ui.ZStack(width=SLOT_TOTAL_W, height=ROW_H):
            if is_active:
                self._build_active_marker()
            else:
                self._build_idle_dot()

            # Click catcher
            btn = ui.Button(
                "",
                width=SLOT_TOTAL_W,
                height=ROW_H,
                clicked_fn=lambda pos=position: self._on_slot_clicked(pos),
                style={
                    "Button": {
                        "background_color": 0x00000000,
                        "border_width": 0,
                        "padding": 0,
                        "margin": 0,
                    },
                    "Button.Label": {"color": 0x00000000},
                },
            )
            # Only the active marker ball is draggable
            if is_active:
                btn.set_mouse_pressed_fn(
                    lambda x, y, b, m: self._start_drag(b)
                )

    def _build_active_marker(self):
        """Ball on top, vertical line below — fixed height, centred."""
        with ui.VStack(spacing=0, height=ROW_H):
            # Ball
            with ui.HStack(height=BALL_DIAM + 2):
                ui.Spacer()
                ui.Circle(
                    radius=BALL_DIAM / 2,
                    style={"background_color": MARKER_COLOR},
                )
                ui.Spacer()
            # Line
            line_h = ROW_H - BALL_DIAM - 2
            if line_h > 0:
                with ui.HStack(height=line_h):
                    ui.Spacer()
                    ui.Rectangle(
                        width=MARKER_LINE_W,
                        height=line_h,
                        style={"background_color": MARKER_COLOR,
                               "border_radius": 1},
                    )
                    ui.Spacer()

    def _build_idle_dot(self):
        """A small faint dot vertically centred, fixed height."""
        with ui.VStack(height=ROW_H):
            ui.Spacer()
            with ui.HStack(height=8):
                ui.Spacer()
                ui.Circle(
                    radius=3,
                    style={"background_color": SLOT_DOT_COLOR},
                )
                ui.Spacer()
            ui.Spacer()

    # ── Feature node ────────────────────────────────────────────────────────

    def _build_feature_node(self, index: int, feature):
        is_selected = (index == self._selected_idx)
        is_before_marker = (index <= self._marker_pos)

        if feature.suppressed:
            bg, fg = SUPPRESSED_BG, SUPPRESSED_FG
        elif is_before_marker:
            bg, fg = ACTIVE_BG, ACTIVE_FG
        else:
            bg, fg = INACTIVE_BG, INACTIVE_FG

        border_color = SELECTED_BORDER if is_selected else 0xFF444444
        border_width = 2 if is_selected else 1

        icon_key = "SKETCH"
        if feature.is_operation and feature.operation:
            icon_key = feature.operation.op_type.name
        icon = FEATURE_ICONS.get(icon_key, "[?]")

        btn = ui.Button(
            f"{icon}\n{feature.name}",
            width=FEATURE_W,
            height=FEATURE_H,
            clicked_fn=lambda idx=index: self._on_feature_click(idx),
            style={
                "Button": {
                    "background_color": bg,
                    "border_color": border_color,
                    "border_width": border_width,
                    "border_radius": 4,
                },
                "Button.Label": {"color": fg, "font_size": 11},
            },
        )
        btn.set_mouse_pressed_fn(
            lambda x, y, b, m, idx=index, sup=feature.suppressed:
                self._on_feature_right_click(x, y, b, idx, sup)
        )

    # ── Drag logic (app-update polling) ─────────────────────────────────────

    def _start_drag(self, button):
        """Begin dragging the marker. Uses app update loop to poll mouse."""
        if button != 0:
            return
        self._dragging = True
        if _HAS_KIT and self._update_sub is None:
            self._update_sub = (
                omni.kit.app.get_app()
                .get_update_event_stream()
                .create_subscription_to_pop(self._on_drag_update)
            )

    def _stop_drag(self):
        self._dragging = False
        if self._update_sub is not None:
            self._update_sub = None  # releasing the subscription unsubscribes

    def _on_drag_update(self, _event):
        """Called every frame while dragging — poll the mouse position."""
        if not self._dragging:
            self._stop_drag()
            return

        # Check if left mouse button is still held
        try:
            import carb.input
            input_iface = carb.input.acquire_input_interface()
            mouse = input_iface.get_mouse()
            if mouse is None:
                self._stop_drag()
                return
            left_held = input_iface.get_mouse_button_flags(
                mouse, carb.input.MouseInput.LEFT_BUTTON
            )
            if not left_held:
                self._stop_drag()
                return

            # Get mouse x in app-window coordinates
            mouse_pos = input_iface.get_mouse_coords_pixel(mouse)
            mouse_x = mouse_pos[0] if mouse_pos else 0
        except Exception:
            self._stop_drag()
            return

        # Convert mouse_x from app-window coords to scroll-frame local coords
        # We need the scroll frame's screen position
        if self._scroll_frame is None:
            self._stop_drag()
            return

        try:
            frame_sx = self._scroll_frame.screen_position_x
            scroll_x = self._scroll_frame.scroll_x
            local_x = mouse_x - frame_sx + scroll_x
        except Exception:
            local_x = mouse_x

        # Find which slot the local_x falls into
        best_pos = self._marker_pos
        best_dist = float("inf")
        for pos, x_left, x_right in self._slot_ranges:
            center = (x_left + x_right) / 2
            d = abs(local_x - center)
            if d < best_dist:
                best_dist = d
                best_pos = pos

        if best_pos != self._marker_pos:
            self._marker_pos = best_pos
            self.update_features(
                self._features, self._marker_pos, self._selected_idx
            )
            if self.on_marker_moved:
                self.on_marker_moved(best_pos)

    # ── Slot / nav interaction ──────────────────────────────────────────────

    def _on_slot_clicked(self, position: int):
        if self._dragging:
            return
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

    # ── Feature interaction ─────────────────────────────────────────────────

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
