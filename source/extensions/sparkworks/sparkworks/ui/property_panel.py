"""
Property Panel — edits parameters of the selected feature.

When a feature is selected in the timeline, this panel shows its editable
parameters (e.g., extrude distance, fillet radius) and triggers a rebuild
when values change.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Optional

try:
    import omni.ui as ui
except ImportError:
    ui = None


PROPERTY_STYLE = {
    "Label": {"font_size": 13, "color": 0xFFCCCCCC},
    "Field": {
        "background_color": 0xFF333333,
        "border_color": 0xFF555555,
        "border_width": 1,
        "border_radius": 3,
        "padding": 4,
    },
}


class PropertyPanel:
    """
    Dockable property editor for the currently selected CAD feature.

    Callbacks:
        on_param_changed: Called with (feature_index, param_name, new_value)
            when a parameter is edited.
    """

    def __init__(self):
        self._window: Optional[ui.Window] = None
        self._current_feature = None
        self._current_index: int = -1
        self._param_models: Dict[str, Any] = {}

        # Callbacks
        self.on_param_changed: Optional[Callable] = None

    @property
    def window(self):
        return self._window

    def build(self):
        """Build and show the property panel."""
        if ui is None:
            return

        self._window = ui.Window(
            "CAD Properties",
            width=300,
            height=300,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM,
        )
        self._window.frame.set_style(PROPERTY_STYLE)
        self._show_no_selection()

    def destroy(self):
        """Clean up the property panel."""
        if self._window:
            self._window.destroy()
            self._window = None

    def show_feature(self, feature, index: int):
        """
        Display the properties of the given feature for editing.

        Args:
            feature: The Feature object to display.
            index: The feature's index in the timeline.
        """
        if ui is None or self._window is None:
            return

        self._current_feature = feature
        self._current_index = index
        self._param_models.clear()

        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    # Header
                    ui.Label(
                        feature.display_name,
                        style={"font_size": 16, "color": 0xFFFFFFFF},
                        height=24,
                    )
                    ui.Spacer(height=4)

                    if feature.is_sketch and feature.sketch:
                        self._build_sketch_properties(feature.sketch)
                    elif feature.is_operation and feature.operation:
                        self._build_operation_properties(feature.operation)
                    else:
                        ui.Label("No editable properties.")

                    ui.Spacer()

    def _show_no_selection(self):
        """Show empty state when no feature is selected."""
        with self._window.frame:
            with ui.VStack():
                ui.Spacer()
                ui.Label(
                    "Select a feature to edit its properties.",
                    alignment=ui.Alignment.CENTER,
                    style={"color": 0xFF888888},
                )
                ui.Spacer()

    # -- Property builders ---------------------------------------------------

    def _build_sketch_properties(self, sketch):
        """Build property editor for a sketch."""
        ui.Label("Sketch Properties", style={"font_size": 14})
        ui.Spacer(height=4)

        # Plane
        with ui.HStack(height=24):
            ui.Label("Plane:", width=100)
            ui.Label(sketch.plane_name, width=ui.Fraction(1))

        # Primitive count
        with ui.HStack(height=24):
            ui.Label("Primitives:", width=100)
            ui.Label(str(len(sketch.primitives)), width=ui.Fraction(1))

        # List primitives
        ui.Spacer(height=4)
        ui.Label("Elements:", style={"font_size": 12, "color": 0xFFAAAAAA})
        for i, prim in enumerate(sketch.primitives):
            with ui.HStack(height=20):
                ui.Label(f"  {i + 1}. {prim.kind.name.title()}", style={"font_size": 11})

    def _build_operation_properties(self, operation):
        """Build property editor for an operation."""
        from ..kernel.operations import (
            ExtrudeOperation,
            RevolveOperation,
            FilletOperation,
            ChamferOperation,
        )

        if isinstance(operation, ExtrudeOperation):
            self._build_extrude_properties(operation)
        elif isinstance(operation, RevolveOperation):
            self._build_revolve_properties(operation)
        elif isinstance(operation, FilletOperation):
            self._build_fillet_properties(operation)
        elif isinstance(operation, ChamferOperation):
            self._build_chamfer_properties(operation)
        else:
            ui.Label(f"Operation: {operation.op_type.name}")

    def _build_extrude_properties(self, op):
        """Build extrude operation properties."""
        ui.Label("Extrude Properties", style={"font_size": 14})
        ui.Spacer(height=4)

        # Distance
        self._build_float_field("distance", "Distance:", op.distance, 0.1, 10000.0)

        # Symmetric toggle
        self._build_bool_field("symmetric", "Symmetric:", op.symmetric)

    def _build_revolve_properties(self, op):
        """Build revolve operation properties."""
        ui.Label("Revolve Properties", style={"font_size": 14})
        ui.Spacer(height=4)

        self._build_float_field("angle", "Angle (deg):", op.angle, 0.1, 360.0)

        with ui.HStack(height=24):
            ui.Label("Axis:", width=100)
            ui.Label(op.axis_name, width=ui.Fraction(1))

    def _build_fillet_properties(self, op):
        """Build fillet operation properties."""
        ui.Label("Fillet Properties", style={"font_size": 14})
        ui.Spacer(height=4)

        self._build_float_field("radius", "Radius:", op.radius, 0.01, 1000.0)

    def _build_chamfer_properties(self, op):
        """Build chamfer operation properties."""
        ui.Label("Chamfer Properties", style={"font_size": 14})
        ui.Spacer(height=4)

        self._build_float_field("length", "Length:", op.length, 0.01, 1000.0)

    # -- Field builders ------------------------------------------------------

    def _build_float_field(
        self, param_name: str, label: str, value: float, min_val: float, max_val: float
    ):
        """Build a float input field."""
        with ui.HStack(height=24):
            ui.Label(label, width=100)
            model = ui.SimpleFloatModel(value)
            field = ui.FloatField(model=model, width=ui.Fraction(1))

            def on_value_changed(m, name=param_name):
                new_val = m.as_float
                if self.on_param_changed and self._current_index >= 0:
                    self.on_param_changed(self._current_index, name, new_val)

            model.add_value_changed_fn(on_value_changed)
            self._param_models[param_name] = model

    def _build_bool_field(self, param_name: str, label: str, value: bool):
        """Build a boolean checkbox field."""
        with ui.HStack(height=24):
            ui.Label(label, width=100)
            model = ui.SimpleBoolModel(value)
            cb = ui.CheckBox(model=model, width=24)

            def on_value_changed(m, name=param_name):
                new_val = m.as_bool
                if self.on_param_changed and self._current_index >= 0:
                    self.on_param_changed(self._current_index, name, new_val)

            model.add_value_changed_fn(on_value_changed)
            self._param_models[param_name] = model
