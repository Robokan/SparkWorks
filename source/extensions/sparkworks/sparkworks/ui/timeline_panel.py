"""
Timeline Panel — displays the parametric feature history.

A dockable panel at the bottom of the viewport that shows each feature as
a clickable row. Supports:
- Clicking a feature to scrub the timeline to that point
- Right-click context menu for suppress/unsuppress, delete, move up/down
- Visual indication of the current scrub position
- Suppressed features shown as dimmed
"""

from __future__ import annotations

from typing import Callable, List, Optional

try:
    import omni.ui as ui
except ImportError:
    ui = None


# Style
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
    "Button:selected": {
        "background_color": 0xFF2266AA,
        "border_color": 0xFF4488CC,
    },
}

# Feature type icons (text-based for Phase 1)
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


class TimelinePanel:
    """
    Dockable timeline panel showing the feature history.

    Callbacks:
        on_feature_click: Called with feature index when a feature is clicked.
        on_feature_delete: Called with feature index to delete.
        on_feature_suppress: Called with (index, suppressed) to toggle.
        on_feature_move: Called with (from_index, to_index) to reorder.
        on_scrub_to_end: Called to show the latest state.
    """

    def __init__(self):
        self._window: Optional[ui.Window] = None
        self._feature_frames: List = []
        self._selected_index: int = -1

        # Callbacks
        self.on_feature_click: Optional[Callable] = None
        self.on_feature_delete: Optional[Callable] = None
        self.on_feature_suppress: Optional[Callable] = None
        self.on_feature_move: Optional[Callable] = None
        self.on_scrub_to_end: Optional[Callable] = None

    @property
    def window(self):
        return self._window

    def build(self):
        """Build and show the timeline panel."""
        if ui is None:
            return

        self._window = ui.Window(
            "CAD Timeline",
            width=400,
            height=200,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.frame.set_style(TIMELINE_STYLE)
        self._build_empty_state()

    def destroy(self):
        """Clean up the timeline panel."""
        if self._window:
            self._window.destroy()
            self._window = None
        self._feature_frames.clear()

    def update_features(self, features: list, scrub_index: int = -1):
        """
        Rebuild the timeline display with the given features.

        Args:
            features: List of Feature objects from the timeline.
            scrub_index: Currently active feature index (-1 = latest).
        """
        if ui is None or self._window is None:
            return

        self._selected_index = scrub_index
        self._feature_frames.clear()

        with self._window.frame:
            if not features:
                self._build_empty_state()
                return

            with ui.VStack(spacing=2):
                # Header
                with ui.HStack(height=24):
                    ui.Label("Feature Timeline", style={"font_size": 14, "color": 0xFFCCCCCC})
                    ui.Spacer()
                    ui.Button(
                        "Show Latest",
                        width=90,
                        height=22,
                        clicked_fn=self._on_scrub_to_end,
                    )

                ui.Spacer(height=4)

                # Feature list (horizontal timeline-like layout)
                with ui.ScrollingFrame(
                    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                    height=ui.Fraction(1),
                ):
                    with ui.HStack(spacing=4):
                        for i, feature in enumerate(features):
                            self._build_feature_node(i, feature, i == scrub_index)

                        # Add arrow indicators between features
                        ui.Spacer(width=20)

    def _build_empty_state(self):
        """Show empty state message."""
        with self._window.frame:
            with ui.VStack():
                ui.Spacer()
                ui.Label(
                    "No features yet. Create a sketch to begin.",
                    alignment=ui.Alignment.CENTER,
                    style={"color": 0xFF888888, "font_size": 14},
                )
                ui.Spacer()

    def _build_feature_node(self, index: int, feature, is_active: bool):
        """Build a single feature node in the timeline."""
        # Determine styling
        if feature.suppressed:
            bg_color = 0xFF1A1A1A
            text_color = 0xFF666666
        elif is_active:
            bg_color = 0xFF2266AA
            text_color = 0xFFFFFFFF
        else:
            bg_color = 0xFF2A2A2A
            text_color = 0xFFCCCCCC

        # Get icon
        icon_key = "SKETCH"
        if feature.is_operation and feature.operation:
            icon_key = feature.operation.op_type.name
        icon = FEATURE_ICONS.get(icon_key, "[?]")

        with ui.VStack(width=100, spacing=2):
            # Feature button
            btn = ui.Button(
                f"{icon}\n{feature.name}",
                width=96,
                height=60,
                clicked_fn=lambda idx=index: self._on_feature_click(idx),
                style={
                    "Button": {"background_color": bg_color},
                    "Button.Label": {"color": text_color, "font_size": 11},
                },
            )

            # Feature index label
            ui.Label(
                f"#{index + 1}",
                alignment=ui.Alignment.CENTER,
                style={"color": 0xFF666666, "font_size": 10},
            )

            # Action buttons row
            with ui.HStack(height=18, spacing=1):
                ui.Button(
                    "S" if not feature.suppressed else "U",
                    width=30,
                    height=18,
                    clicked_fn=lambda idx=index, sup=feature.suppressed: self._on_suppress(idx, sup),
                    tooltip="Suppress/Unsuppress",
                    style={"Button.Label": {"font_size": 9}},
                )
                ui.Button(
                    "X",
                    width=30,
                    height=18,
                    clicked_fn=lambda idx=index: self._on_delete(idx),
                    tooltip="Delete",
                    style={
                        "Button": {"background_color": 0xFF442222},
                        "Button.Label": {"font_size": 9},
                    },
                )

    # -- Callback wrappers ---------------------------------------------------

    def _on_feature_click(self, index: int):
        if self.on_feature_click:
            self.on_feature_click(index)

    def _on_delete(self, index: int):
        if self.on_feature_delete:
            self.on_feature_delete(index)

    def _on_suppress(self, index: int, currently_suppressed: bool):
        if self.on_feature_suppress:
            self.on_feature_suppress(index, not currently_suppressed)

    def _on_scrub_to_end(self):
        if self.on_scrub_to_end:
            self.on_scrub_to_end()
