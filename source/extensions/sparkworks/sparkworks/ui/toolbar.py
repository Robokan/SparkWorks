"""
CAD Toolbar — main action bar for the parametric CAD extension.

Provides buttons for:
- New Sketch (with plane selection)
- Sketch tools with dimension inputs (line, rectangle, circle, arc)
- 3D operations with parameter inputs (extrude, revolve, fillet, chamfer)
- Timeline controls (rebuild, clear)
- Status label showing current state
"""

from __future__ import annotations

from typing import Callable, Optional

try:
    import omni.ui as ui
except ImportError:
    ui = None


# Style constants
TOOLBAR_STYLE = {
    "Button": {
        "background_color": 0xFF333333,
        "border_color": 0xFF555555,
        "border_width": 1,
        "border_radius": 4,
        "margin": 2,
        "padding": 6,
    },
    "Button:hovered": {
        "background_color": 0xFF444444,
    },
    "Button:pressed": {
        "background_color": 0xFF2277CC,
    },
    "Button.Label": {
        "color": 0xFFDDDDDD,
        "font_size": 13,
    },
    "Label": {
        "color": 0xFFAAAAAA,
        "font_size": 11,
    },
    "ComboBox": {
        "background_color": 0xFF333333,
        "color": 0xFFDDDDDD,
        "font_size": 13,
        "border_radius": 4,
    },
    "FloatField": {
        "background_color": 0xFF1A1A1A,
        "border_color": 0xFF555555,
        "border_width": 1,
        "border_radius": 3,
        "color": 0xFFDDDDDD,
        "font_size": 13,
    },
}

BUTTON_HEIGHT = 28
FIELD_HEIGHT = 24


class CadToolbar:
    """
    Dockable toolbar window for CAD operations.

    Includes input fields for primitive dimensions and operation parameters
    so the user can specify values before adding geometry.
    """

    def __init__(self):
        self._window: Optional[ui.Window] = None

        # Callbacks — set these from the extension
        self.on_new_sketch: Optional[Callable] = None
        self.on_finish_sketch: Optional[Callable] = None
        self.on_add_line: Optional[Callable] = None
        self.on_add_rectangle: Optional[Callable] = None
        self.on_add_circle: Optional[Callable] = None
        self.on_add_arc: Optional[Callable] = None
        self.on_extrude: Optional[Callable] = None
        self.on_revolve: Optional[Callable] = None
        self.on_fillet: Optional[Callable] = None
        self.on_chamfer: Optional[Callable] = None
        self.on_boolean: Optional[Callable] = None
        self.on_rebuild_all: Optional[Callable] = None
        self.on_clear_all: Optional[Callable] = None

        # State
        self._sketch_mode = False
        self._plane_model = None
        self._status_label = None

        # -- Dimension input models --
        # Rectangle
        self._rect_width_model = None
        self._rect_height_model = None
        # Circle
        self._circle_radius_model = None
        # Line
        self._line_x1_model = None
        self._line_y1_model = None
        self._line_x2_model = None
        self._line_y2_model = None
        # Extrude
        self._extrude_dist_model = None
        # Revolve
        self._revolve_angle_model = None
        # Fillet
        self._fillet_radius_model = None
        # Chamfer
        self._chamfer_length_model = None

    @property
    def window(self):
        return self._window

    def build(self):
        """Build and show the toolbar window."""
        if ui is None:
            return

        self._window = ui.Window(
            "SparkWorks",
            width=280,
            height=600,
        )
        self._window.frame.set_style(TOOLBAR_STYLE)

        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    # Status bar at top
                    self._build_status_bar()
                    ui.Line(height=2, style={"color": 0xFF444444})
                    self._build_sketch_section()
                    ui.Line(height=2, style={"color": 0xFF444444})
                    self._build_operations_section()
                    ui.Line(height=2, style={"color": 0xFF444444})
                    self._build_modifiers_section()
                    ui.Line(height=2, style={"color": 0xFF444444})
                    self._build_timeline_controls()
                    ui.Spacer()

    def destroy(self):
        """Clean up the toolbar window."""
        if self._window:
            self._window.destroy()
            self._window = None

    def set_sketch_mode(self, active: bool):
        """Update UI to reflect sketch mode state."""
        self._sketch_mode = active

    def set_status(self, message: str):
        """Update the status label text."""
        if self._status_label is not None:
            self._status_label.text = message

    # -- UI Builders ---------------------------------------------------------

    def _build_status_bar(self):
        """Build a status label at the top."""
        with ui.HStack(height=22):
            self._status_label = ui.Label(
                "Ready — click New Sketch to begin",
                style={"color": 0xFF88BBEE, "font_size": 12},
                word_wrap=True,
            )

    def _build_sketch_section(self):
        """Build the sketch tools section with dimension inputs."""
        with ui.CollapsableFrame("  Sketch", height=0):
            with ui.VStack(spacing=4):
                # Plane selection
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("Plane:", width=70)
                    self._plane_model = ui.ComboBox(
                        0, "XY", "XZ", "YZ", width=ui.Fraction(1)
                    )

                # New / Finish sketch
                with ui.HStack(height=BUTTON_HEIGHT):
                    ui.Button("New Sketch", width=ui.Fraction(1), clicked_fn=self._on_new_sketch)
                    ui.Button("Finish Sketch", width=ui.Fraction(1), clicked_fn=self._on_finish_sketch)

                ui.Spacer(height=4)

                # --- Rectangle ---
                ui.Label("Rectangle", style={"font_size": 12, "color": 0xFFCCCCCC})
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("W:", width=20)
                    self._rect_width_model = ui.SimpleFloatModel(20.0)
                    ui.FloatField(model=self._rect_width_model, width=ui.Fraction(1))
                    ui.Spacer(width=8)
                    ui.Label("H:", width=20)
                    self._rect_height_model = ui.SimpleFloatModel(10.0)
                    ui.FloatField(model=self._rect_height_model, width=ui.Fraction(1))
                ui.Button("Add Rectangle", height=BUTTON_HEIGHT, clicked_fn=self._on_add_rectangle)

                ui.Spacer(height=4)

                # --- Circle ---
                ui.Label("Circle", style={"font_size": 12, "color": 0xFFCCCCCC})
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("Radius:", width=70)
                    self._circle_radius_model = ui.SimpleFloatModel(5.0)
                    ui.FloatField(model=self._circle_radius_model, width=ui.Fraction(1))
                ui.Button("Add Circle", height=BUTTON_HEIGHT, clicked_fn=self._on_add_circle)

                ui.Spacer(height=4)

                # --- Line ---
                ui.Label("Line", style={"font_size": 12, "color": 0xFFCCCCCC})
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("X1:", width=24)
                    self._line_x1_model = ui.SimpleFloatModel(0.0)
                    ui.FloatField(model=self._line_x1_model, width=ui.Fraction(1))
                    ui.Label("Y1:", width=24)
                    self._line_y1_model = ui.SimpleFloatModel(0.0)
                    ui.FloatField(model=self._line_y1_model, width=ui.Fraction(1))
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("X2:", width=24)
                    self._line_x2_model = ui.SimpleFloatModel(10.0)
                    ui.FloatField(model=self._line_x2_model, width=ui.Fraction(1))
                    ui.Label("Y2:", width=24)
                    self._line_y2_model = ui.SimpleFloatModel(0.0)
                    ui.FloatField(model=self._line_y2_model, width=ui.Fraction(1))
                ui.Button("Add Line", height=BUTTON_HEIGHT, clicked_fn=self._on_add_line)

    def _build_operations_section(self):
        """Build the 3D operations section with parameter inputs."""
        with ui.CollapsableFrame("  3D Operations", height=0):
            with ui.VStack(spacing=4):
                # --- Extrude ---
                ui.Label("Extrude", style={"font_size": 12, "color": 0xFFCCCCCC})
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("Distance:", width=70)
                    self._extrude_dist_model = ui.SimpleFloatModel(10.0)
                    ui.FloatField(model=self._extrude_dist_model, width=ui.Fraction(1))
                ui.Button("Extrude", height=BUTTON_HEIGHT, clicked_fn=self._on_extrude)

                ui.Spacer(height=4)

                # --- Revolve ---
                ui.Label("Revolve", style={"font_size": 12, "color": 0xFFCCCCCC})
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("Angle:", width=70)
                    self._revolve_angle_model = ui.SimpleFloatModel(360.0)
                    ui.FloatField(model=self._revolve_angle_model, width=ui.Fraction(1))
                ui.Button("Revolve", height=BUTTON_HEIGHT, clicked_fn=self._on_revolve)

    def _build_modifiers_section(self):
        """Build the modifier operations section with parameter inputs."""
        with ui.CollapsableFrame("  Modifiers", height=0):
            with ui.VStack(spacing=4):
                # --- Fillet ---
                ui.Label("Fillet", style={"font_size": 12, "color": 0xFFCCCCCC})
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("Radius:", width=70)
                    self._fillet_radius_model = ui.SimpleFloatModel(1.0)
                    ui.FloatField(model=self._fillet_radius_model, width=ui.Fraction(1))
                ui.Button("Apply Fillet", height=BUTTON_HEIGHT, clicked_fn=self._on_fillet)

                ui.Spacer(height=4)

                # --- Chamfer ---
                ui.Label("Chamfer", style={"font_size": 12, "color": 0xFFCCCCCC})
                with ui.HStack(height=FIELD_HEIGHT):
                    ui.Label("Length:", width=70)
                    self._chamfer_length_model = ui.SimpleFloatModel(1.0)
                    ui.FloatField(model=self._chamfer_length_model, width=ui.Fraction(1))
                ui.Button("Apply Chamfer", height=BUTTON_HEIGHT, clicked_fn=self._on_chamfer)

    def _build_timeline_controls(self):
        """Build timeline control buttons."""
        with ui.CollapsableFrame("  Controls", height=0):
            with ui.VStack(spacing=4):
                with ui.HStack(height=BUTTON_HEIGHT):
                    ui.Button("Rebuild All", width=ui.Fraction(1), clicked_fn=self._on_rebuild_all)
                    ui.Button(
                        "Clear All",
                        width=ui.Fraction(1),
                        clicked_fn=self._on_clear_all,
                        style={"Button": {"background_color": 0xFF552222}},
                    )

    # -- Value getters -------------------------------------------------------

    @property
    def selected_plane(self) -> str:
        if self._plane_model is not None:
            idx = self._plane_model.model.get_item_value_model().as_int
            planes = ["XY", "XZ", "YZ"]
            return planes[idx] if idx < len(planes) else "XY"
        return "XY"

    @property
    def rect_width(self) -> float:
        return self._rect_width_model.as_float if self._rect_width_model else 20.0

    @property
    def rect_height(self) -> float:
        return self._rect_height_model.as_float if self._rect_height_model else 10.0

    @property
    def circle_radius(self) -> float:
        return self._circle_radius_model.as_float if self._circle_radius_model else 5.0

    @property
    def line_start(self):
        x = self._line_x1_model.as_float if self._line_x1_model else 0.0
        y = self._line_y1_model.as_float if self._line_y1_model else 0.0
        return (x, y)

    @property
    def line_end(self):
        x = self._line_x2_model.as_float if self._line_x2_model else 10.0
        y = self._line_y2_model.as_float if self._line_y2_model else 0.0
        return (x, y)

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

    # -- Callback wrappers ---------------------------------------------------

    def _on_new_sketch(self):
        if self.on_new_sketch:
            self.on_new_sketch(self.selected_plane)

    def _on_finish_sketch(self):
        if self.on_finish_sketch:
            self.on_finish_sketch()

    def _on_add_line(self):
        if self.on_add_line:
            self.on_add_line()

    def _on_add_rectangle(self):
        if self.on_add_rectangle:
            self.on_add_rectangle()

    def _on_add_circle(self):
        if self.on_add_circle:
            self.on_add_circle()

    def _on_add_arc(self):
        if self.on_add_arc:
            self.on_add_arc()

    def _on_extrude(self):
        if self.on_extrude:
            self.on_extrude()

    def _on_revolve(self):
        if self.on_revolve:
            self.on_revolve()

    def _on_fillet(self):
        if self.on_fillet:
            self.on_fillet()

    def _on_chamfer(self):
        if self.on_chamfer:
            self.on_chamfer()

    def _on_rebuild_all(self):
        if self.on_rebuild_all:
            self.on_rebuild_all()

    def _on_clear_all(self):
        if self.on_clear_all:
            self.on_clear_all()
