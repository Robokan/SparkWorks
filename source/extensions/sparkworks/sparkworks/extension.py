"""
SparkWorks Extension — main entry point.

Orchestrates the geometry kernel, parametric timeline, USD bridge, and UI
into a cohesive CAD modeling experience inside Isaac Sim.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import omni.ext
import omni.kit.app
import omni.kit.commands
import omni.ui as ui
import omni.usd

try:
    import omni.kit.notification_manager as notifications
except ImportError:
    notifications = None

from .kernel import Sketch, Tessellator
from .kernel.construction_plane import (
    ConstructionPlane,
    create_origin_planes,
    extract_face_planes,
)
from .kernel.operations import (
    BooleanOperation,
    ExtrudeOperation,
    RevolveOperation,
    FilletOperation,
    ChamferOperation,
)
from .timeline import Timeline, Feature, FeatureType, SketchRegistry
from .bridge import UsdBridge
from .ui import CadToolbar, TimelinePanel, PropertyPanel, SketchViewport
from .ui.sketch_tool import SketchToolManager, SketchToolMode


EXTENSION_NAME = "sparkworks"

# Key codes from carb.input.KeyboardInput (used by omni.ui set_key_pressed_fn)
KEY_ENTER = 51          # carb.input.KeyboardInput.ENTER
KEY_NUMPAD_ENTER = 95   # carb.input.KeyboardInput.NUMPAD_ENTER
KEY_ESCAPE = 49         # carb.input.KeyboardInput.ESCAPE


def _notify(message: str, status="info"):
    """Send a notification to the user."""
    if notifications is not None:
        try:
            if status == "warning":
                notifications.post_notification(
                    message, status=notifications.NotificationStatus.WARNING
                )
            else:
                notifications.post_notification(
                    message, status=notifications.NotificationStatus.INFO
                )
        except Exception:
            pass
    print(f"[{EXTENSION_NAME}] {message}")


class ParametricCadExtension(omni.ext.IExt):
    """Main SparkWorks extension class."""

    def on_startup(self, ext_id: str):
        print(f"[{EXTENSION_NAME}] Extension startup")
        self._ext_id = ext_id

        # -- Core systems ----
        self._sketch_registry = SketchRegistry()
        self._timeline = Timeline(sketch_registry=self._sketch_registry)
        self._bridge = UsdBridge()
        self._tessellator = Tessellator()

        # -- Construction planes (origin / user-created reference surfaces) ----
        self._construction_planes: List[ConstructionPlane] = create_origin_planes()
        self._construction_plane_map: Dict[str, ConstructionPlane] = {}
        self._picking_plane: bool = False  # True when waiting for user to pick a plane

        # -- Body face map (B-Rep planar faces on bodies) ----
        self._face_map: Dict[str, ConstructionPlane] = {}
        self._selected_face_name: Optional[str] = None  # currently selected body face
        self._last_stage_url: Optional[str] = None  # tracks current file for save-vs-open detection

        # -- Active sketch state ----
        self._active_sketch: Optional[Sketch] = None
        self._editing_feature_index: Optional[int] = None  # set when editing existing sketch
        self._feature_counter = 0
        self._body_counter = 0
        self._current_body_name = "Body1"

        # -- Face-on-body state (for "sketch on face" workflow) ----
        self._face_planes: List[ConstructionPlane] = []   # currently-shown face planes
        self._face_plane_body: Optional[str] = None       # body name whose faces are shown
        self._sketch_parent_body: Optional[str] = None    # body to join into on extrude

        # -- Profile overlay state (for line-based closed-loop profiles) ----
        self._profile_paths: List[str] = []               # USD prim paths of profile overlays
        self._profile_faces: list = []                     # build123d Face objects for each profile
        self._selected_profile_indices: set = set()          # which profiles are selected for extrude
        self._profile_sketch_name: Optional[str] = None    # sketch name that owns current profiles
        self._profile_plane_normal: Optional[tuple] = None # cached plane normal for recreation

        # -- Sketch tool manager (interactive drawing state machine) ----
        self._sketch_tool = SketchToolManager()

        # -- USD selection subscription ----
        self._selection_sub = None
        self._objects_changed_key = None  # Tf.Notice listener for visibility changes

        # -- Dialog state ----
        self._extrude_dialog_window = None
        self._extrude_dist_model = None
        self._boolean_dialog_window = None
        self._boolean_pending_mode: str = "union"

        # -- UI ----
        self._toolbar = CadToolbar()
        self._timeline_panel = TimelinePanel()
        self._property_panel = PropertyPanel()
        self._sketch_viewport = SketchViewport()

        # -- Wire toolbar callbacks BEFORE build (needed for native toolbar) --
        self._connect_toolbar_callbacks()
        self._toolbar.build()

        self._timeline_panel.build()
        self._property_panel.build()
        self._sketch_viewport.build()
        self._connect_timeline_panel_callbacks()
        self._connect_property_panel_callbacks()
        self._connect_timeline_callbacks()
        self._connect_sketch_tool_callbacks()
        self._connect_viewport_callbacks()
        self._subscribe_selection_changed()

        # -- Create origin planes in the viewport after stage is ready ----
        asyncio.ensure_future(self._create_initial_planes())

        # -- Dock windows after the UI settles ----
        asyncio.ensure_future(self._dock_windows())

        _notify("SparkWorks ready. Select a plane, then click Create Sketch.")

    async def _dock_windows(self):
        """Dock our windows relative to the Viewport after it exists."""
        viewport_win = None
        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()
            viewport_win = ui.Workspace.get_window("Viewport")
            if viewport_win:
                break

        if viewport_win is None:
            print(f"[{EXTENSION_NAME}] Viewport window not found, skipping dock")
            return

        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()

        try:
            # Toolbar is now native (omni.kit.widget.toolbar), no docking needed

            if self._property_panel.window:
                self._property_panel.window.dock_in(
                    viewport_win, ui.DockPosition.RIGHT, ratio=0.18
                )
            await omni.kit.app.get_app().next_update_async()

            if self._timeline_panel.window:
                self._timeline_panel.window.dock_in(
                    viewport_win, ui.DockPosition.BOTTOM, ratio=0.18
                )
            await omni.kit.app.get_app().next_update_async()

            if self._sketch_viewport.window:
                self._sketch_viewport.window.dock_in(
                    viewport_win, ui.DockPosition.SAME
                )

            print(f"[{EXTENSION_NAME}] Windows docked successfully")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Docking error (non-fatal): {e}")

    def on_shutdown(self):
        print(f"[{EXTENSION_NAME}] Extension shutdown")
        self._selection_sub = None
        self._objects_changed_key = None
        if self._toolbar:
            self._toolbar.destroy()
        if self._timeline_panel:
            self._timeline_panel.destroy()
        if self._property_panel:
            self._property_panel.destroy()
        if self._sketch_viewport:
            self._sketch_viewport.destroy()
        if self._extrude_dialog_window:
            self._extrude_dialog_window.visible = False
            self._extrude_dialog_window = None
        if self._boolean_dialog_window:
            self._boolean_dialog_window.visible = False
            self._boolean_dialog_window = None
        self._timeline = None
        self._sketch_registry = None
        self._bridge = None
        self._active_sketch = None
        self._sketch_tool = None
        self._construction_planes = []
        self._construction_plane_map = {}
        self._face_map = {}
        self._selected_face_name = None
        self._picking_plane = False
        self._face_planes = []
        self._face_plane_body = None
        self._sketch_parent_body = None
        self._profile_paths = []
        self._profile_faces = []
        self._selected_profile_indices = set()
        self._profile_sketch_name = None
        self._profile_plane_normal = None

    # ========================================================================
    # Callback Wiring
    # ========================================================================

    def _connect_toolbar_callbacks(self):
        tb = self._toolbar
        tb.on_create_sketch = self._action_create_sketch
        tb.on_finish_sketch = self._action_finish_sketch
        tb.on_add_plane = self._action_add_plane
        # Tool selection (not primitive creation)
        tb.on_tool_line = self._action_tool_line
        tb.on_tool_rectangle = self._action_tool_rectangle
        tb.on_tool_circle = self._action_tool_circle
        # 3D operations
        tb.on_extrude = self._action_extrude
        tb.on_revolve = self._action_revolve
        tb.on_fillet = self._action_fillet
        tb.on_chamfer = self._action_chamfer
        # Boolean operations
        tb.on_boolean_union = lambda: self._action_boolean("union")
        tb.on_boolean_cut = lambda: self._action_boolean("cut")
        tb.on_boolean_intersect = lambda: self._action_boolean("intersect")
        tb.on_rebuild_all = self._action_rebuild_all
        tb.on_clear_all = self._action_clear_all

    def _connect_timeline_panel_callbacks(self):
        tp = self._timeline_panel
        tp.on_feature_select = self._action_select_feature
        tp.on_marker_moved = self._action_marker_moved
        tp.on_feature_delete = self._action_delete_feature
        tp.on_feature_suppress = self._action_suppress_feature

    def _connect_property_panel_callbacks(self):
        self._property_panel.on_param_changed = self._action_update_param
        self._property_panel.on_primitive_edited = self._on_primitive_edited

    def _connect_timeline_callbacks(self):
        self._timeline.on_rebuild = self._on_timeline_rebuild
        self._timeline.on_feature_changed = self._on_features_changed

    def _connect_sketch_tool_callbacks(self):
        """Wire the sketch tool state machine to the extension."""
        st = self._sketch_tool
        st.on_primitive_created = self._on_tool_primitive_created
        st.on_status_changed = self._set_status
        st.on_preview_changed = self._on_tool_preview_changed
        st.on_chain_cancelled = self._on_tool_chain_cancelled

    def _connect_viewport_callbacks(self):
        """Wire the sketch viewport mouse/key events to the tool manager."""
        sv = self._sketch_viewport
        sv.on_viewport_click = self._on_viewport_click
        sv.on_viewport_move = self._on_viewport_move
        sv.on_viewport_key = self._on_viewport_key

    def _set_status(self, msg: str):
        self._toolbar.set_status(msg)

    def _set_profile_selection(self, indices: set):
        """Update selected profile indices."""
        self._selected_profile_indices = indices

    def _update_sketch_view(self):
        """Push the current active sketch's primitives to the sketch viewport."""
        if self._active_sketch:
            hidden = self._get_sketch_hidden_indices(self._active_sketch)
            self._sketch_viewport.update_primitives(
                self._active_sketch.primitives, hidden
            )
            self._sketch_viewport.set_sketch_info(
                self._active_sketch.plane_name,
                self._active_sketch.name,
                len(self._active_sketch.primitives),
            )
        else:
            self._sketch_viewport.update_primitives([])
            self._sketch_viewport.clear_info()

    def _get_sketch_hidden_indices(self, sketch: Sketch) -> set:
        """Query USD visibility for each primitive prim under the sketch."""
        try:
            # Find the feature index for this sketch in the timeline
            for idx, feat in enumerate(self._timeline.features):
                if feat.is_sketch and feat.sketch is sketch:
                    return self._bridge.get_hidden_primitive_indices(idx, sketch.name)
            # If editing a new sketch not yet in the timeline, nothing is hidden
        except Exception:
            pass
        return set()

    def _focus_sketch_view(self):
        """Bring the Sketch View tab to the front."""
        try:
            sketch_win = self._sketch_viewport.window
            if sketch_win is not None:
                sketch_win.focus()
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not focus Sketch View: {e}")

    def _focus_viewport(self):
        """Bring the 3D Viewport tab to the front."""
        try:
            viewport_win = ui.Workspace.get_window("Viewport")
            if viewport_win is not None:
                viewport_win.focus()
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not focus Viewport: {e}")

    # ========================================================================
    # Construction Planes & Selection
    # ========================================================================

    async def _create_initial_planes(self):
        """Wait for the stage, then create construction planes and load saved timeline."""
        # Give the stage a few frames to be ready
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()

        # Initialise settings — load from stage if present, otherwise write defaults
        try:
            existing = self._bridge.load_settings()
            # If Settings prim doesn't exist yet, write defaults (creates the Xform)
            stage = omni.usd.get_context().get_stage()
            settings_prim = stage.GetPrimAtPath(self._bridge.settings_root) if stage else None
            if settings_prim is None or not settings_prim.IsValid():
                self._bridge.save_settings()  # writes DEFAULT_SETTINGS and sets metersPerUnit
            else:
                self._bridge.apply_settings(existing)  # respect whatever is already stored
            print(f"[{EXTENSION_NAME}] Settings: {self._bridge.load_settings()}")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not initialise settings: {e}")

        try:
            result = self._bridge.create_construction_planes(self._construction_planes)
            self._construction_plane_map = {
                p.name: p for p in self._construction_planes
            }
            print(f"[{EXTENSION_NAME}] Origin planes created: {list(result.keys())}")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to create origin planes: {e}")

        # Subscribe to ObjectsChanged now that the stage exists
        self._subscribe_objects_changed()

        # Try to load a previously saved timeline from the USD stage
        self._load_timeline_from_usd()

    def _load_timeline_from_usd(self):
        """Load sketches + timeline from USD prims if saved data exists."""
        try:
            # 1) Load sketches into the registry
            loaded_registry = self._bridge.load_sketches()
            if loaded_registry is not None:
                self._sketch_registry = loaded_registry
                self._timeline.sketch_registry = loaded_registry
                print(f"[{EXTENSION_NAME}] Restored {loaded_registry.count} sketches from USD")

            # 2) Load the timeline features (reference sketch IDs in the registry)
            features = self._bridge.load_timeline()
            if features and len(features) > 0:
                from .timeline.timeline import Timeline
                self._timeline = Timeline(sketch_registry=self._sketch_registry)
                self._connect_timeline_callbacks()
                for feature in features:
                    feature.bind_registry(self._sketch_registry)
                    self._timeline._features.append(feature)
                self._timeline._notify_changed()
                self._timeline.rebuild_all()
                self._set_status(f"Restored {len(features)} features from saved stage")
                _notify(f"Loaded {len(features)} features from stage.")
                print(f"[{EXTENSION_NAME}] Restored {len(features)} features from USD")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] No saved timeline to load (or error): {e}")
            import traceback
            traceback.print_exc()

    def _subscribe_selection_changed(self):
        """Subscribe to USD selection changes to detect plane clicks."""
        try:
            usd_context = omni.usd.get_context()
            events = usd_context.get_stage_event_stream()
            self._selection_sub = events.create_subscription_to_pop(
                self._on_stage_event,
                name=f"{EXTENSION_NAME}_selection",
            )
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not subscribe to selection events: {e}")

    def _subscribe_objects_changed(self):
        """
        Subscribe to USD ObjectsChanged notices so we can refresh the
        sketch view when the user toggles a primitive's visibility
        (eye icon) in the stage hierarchy.
        """
        # Revoke previous listener if any (e.g. when stage reloads)
        if self._objects_changed_key is not None:
            try:
                self._objects_changed_key.Revoke()
            except Exception:
                pass
            self._objects_changed_key = None

        try:
            from pxr import Tf, Usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(f"[{EXTENSION_NAME}] ObjectsChanged: stage not ready, skipping")
                return

            def _on_objects_changed(notice, sender):
                if self._active_sketch is None:
                    return
                # Check both GetChangedInfoOnlyPaths and GetResyncedPaths
                changed = list(notice.GetChangedInfoOnlyPaths()) + list(notice.GetResyncedPaths())
                for p in changed:
                    if "/Primitives/" in str(p):
                        self._update_sketch_view()
                        return

            self._objects_changed_key = Tf.Notice.Register(
                Usd.Notice.ObjectsChanged, _on_objects_changed, stage
            )
            print(f"[{EXTENSION_NAME}] ObjectsChanged listener registered")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Could not subscribe to ObjectsChanged: {e}")
            import traceback
            traceback.print_exc()

    def _on_stage_event(self, event):
        """Handle USD stage events — selection changes and stage open/reload."""
        # --- Stage opened (File > Open, or reload) ---------------------------
        # Isaac Sim may fire OPENED after a Save — guard against that by
        # checking whether the stage URL actually changed.
        if event.type == int(omni.usd.StageEventType.OPENED):
            try:
                new_url = omni.usd.get_context().get_stage_url()
            except Exception:
                new_url = ""
            prev_url = getattr(self, "_last_stage_url", None)
            if new_url and new_url == prev_url:
                # Same file — this is a Save, not a reload.  Just
                # re-create profile overlays that may have been lost.
                print(f"[{EXTENSION_NAME}] Stage OPENED (same URL) — restoring profiles")
                self._restore_profiles_after_save()
                return
            self._last_stage_url = new_url
            print(f"[{EXTENSION_NAME}] Stage OPENED (new URL: {new_url}) — reloading SparkWorks state")
            asyncio.ensure_future(self._on_stage_opened())
            return

        # --- Selection changed -----------------------------------------------
        if event.type != int(omni.usd.StageEventType.SELECTION_CHANGED):
            return

        usd_context = omni.usd.get_context()
        selection = usd_context.get_selection()
        paths = selection.get_selected_prim_paths()

        if not paths:
            return

        # ── Classify every selected path ──────────────────────────────
        selected_profiles: List[int] = []
        selected_face: Optional[str] = None
        selected_bodies: List[str] = []
        selected_sketch: Optional[str] = None
        detected_sketch_name: Optional[str] = None
        other_plane: Optional[ConstructionPlane] = None
        tl_name: Optional[str] = None

        sk_prefix = self._bridge.sketches_root + "/"

        for path in paths:
            # Profile overlay
            profile_name = self._bridge.is_profile_prim(path)
            if profile_name:
                try:
                    idx = int(profile_name.rsplit("Profile", 1)[-1])
                    selected_profiles.append(idx)
                    if detected_sketch_name is None and path.startswith(sk_prefix):
                        detected_sketch_name = path[len(sk_prefix):].split("/")[0]
                except (ValueError, IndexError):
                    pass
                continue

            # Body face plane
            face_name = self._bridge.is_body_face(path)
            if face_name and face_name in self._face_map:
                selected_face = face_name
                continue

            # Body mesh
            body_name = self._bridge.is_body_prim(path)
            if body_name:
                selected_bodies.append(body_name)
                continue

            # Sketch prim (the Sketch Xform itself or a child like Primitives)
            if path.startswith(sk_prefix):
                remainder = path[len(sk_prefix):]
                sk_name = remainder.split("/")[0]
                # Ignore Profiles children — those are handled above
                if "/Profiles/" not in path:
                    selected_sketch = sk_name
                continue

            # Timeline feature
            tl = self._bridge.is_timeline_prim(path)
            if tl:
                tl_name = tl
                continue

            # Origin / user construction plane
            plane_name = self._bridge.is_construction_plane(path)
            if plane_name and plane_name in self._construction_plane_map:
                other_plane = self._construction_plane_map[plane_name]
                continue

        # ── Act on classification ─────────────────────────────────────
        # Hide all tool groups first; the matching branch re-shows the right one.
        self._toolbar.set_sketch_tools_visible(False)
        self._toolbar.set_op_tools_visible(False)
        self._toolbar.set_bool_tools_visible(False)

        # 1) Profile or face → extrude tools
        if selected_profiles:
            if not self._profile_faces and detected_sketch_name:
                self._rebuild_profile_faces(detected_sketch_name)
            self._selected_profile_indices = set(selected_profiles)
            self._selected_face_name = None
            self._toolbar.set_op_tools_visible(True)
            sel = sorted(selected_profiles)
            label = ", ".join(str(s) for s in sel)
            self._set_status(f"Profile(s) {label} selected — click Extrude")
            self._toolbar.set_plane_hint(f"Profile(s) {label} selected. Click Extrude.")
            return

        if selected_face:
            self._selected_face_name = selected_face
            self._selected_profile_indices = set()
            self._toolbar.set_op_tools_visible(True)
            plane = self._face_map[selected_face]
            self._on_construction_plane_clicked(plane)
            return

        # 2) Sketch prim → sketch tools
        if selected_sketch:
            self._selected_face_name = None
            self._selected_profile_indices = set()
            sketch_obj = self._sketch_registry.get(selected_sketch) if self._sketch_registry else None
            if sketch_obj:
                self._active_sketch = sketch_obj
                # Find the feature index for this sketch
                for i, f in enumerate(self._timeline.features):
                    if f.sketch_id == selected_sketch:
                        self._editing_feature_index = i
                        break
                self._toolbar.set_sketch_mode(True)
                self._toolbar.set_plane_hint(f"Editing '{selected_sketch}' — select a drawing tool")
                self._bridge.hide_all_sketches(except_sketch=selected_sketch)
                self._show_sketch_in_viewport(sketch_obj)
                self._refresh_profile_overlays_for_active_sketch()
                self._focus_sketch_view()
            return

        # 3) Body selected → boolean tools
        if selected_bodies:
            self._selected_face_name = None
            self._selected_profile_indices = set()
            if len(self._get_all_body_names()) >= 2:
                self._toolbar.set_bool_tools_visible(True)
            self._on_body_clicked(selected_bodies[0])
            return

        # 4) Everything else — timeline prim, construction plane, etc.
        self._selected_face_name = None
        self._selected_profile_indices = set()

        if tl_name:
            self._on_timeline_prim_clicked(tl_name)
            return

        if other_plane:
            self._on_construction_plane_clicked(other_plane)
            return

    async def _on_stage_opened(self):
        """Reload all SparkWorks state after a new stage is opened."""
        # Give the stage a few frames to settle
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()

        # 1) Apply settings from the file (or write defaults)
        try:
            stage = omni.usd.get_context().get_stage()
            settings_prim = stage.GetPrimAtPath(self._bridge.settings_root) if stage else None
            if settings_prim and settings_prim.IsValid():
                self._bridge.apply_settings()
            else:
                self._bridge.save_settings()
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Settings reload error: {e}")

        # 2) Rebuild the construction-plane map from whatever is on stage
        self._construction_planes = create_origin_planes()
        try:
            stage = omni.usd.get_context().get_stage()
            constr_prim = stage.GetPrimAtPath(self._bridge.construction_root) if stage else None
            if constr_prim and constr_prim.IsValid():
                # Planes already exist in the file — just rebuild the lookup map
                self._construction_plane_map = {p.name: p for p in self._construction_planes}
            else:
                # No planes in the file — create them
                result = self._bridge.create_construction_planes(self._construction_planes)
                self._construction_plane_map = {p.name: p for p in self._construction_planes}
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Plane reload error: {e}")

        # 3) Reset everything to a clean state
        self._active_sketch = None
        self._picking_plane = False
        self._face_planes = []
        self._face_plane_body = None
        self._sketch_parent_body = None
        self._profile_paths = []
        self._profile_faces = []
        self._set_profile_selection(set())
        self._profile_sketch_name = None
        self._profile_plane_normal = None
        self._feature_counter = 0
        self._body_counter = 0
        self._current_body_name = "Body1"

        # Reset the sketch registry and timeline; reconnect callbacks
        self._sketch_registry = SketchRegistry()
        self._timeline = Timeline(sketch_registry=self._sketch_registry)
        self._connect_timeline_callbacks()

        # Reset the UI panels
        self._timeline_panel.update_features([], -1)
        self._toolbar.set_sketch_mode(False)
        self._toolbar.set_plane_hint("Click a plane in the viewport to start a sketch")
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)

        # 4) Re-subscribe ObjectsChanged for the new stage
        self._subscribe_objects_changed()

        # 5) Load the timeline from USD (populates if saved data exists)
        self._load_timeline_from_usd()

        # 6) Update body counter from what's on stage
        try:
            body_names = self._bridge.get_body_names()
            if body_names:
                # Find highest existing body number
                max_num = 0
                for name in body_names:
                    try:
                        num = int(name.replace("Body", ""))
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
                self._body_counter = max_num
                self._current_body_name = f"Body{max_num}"
        except Exception:
            pass

        # 7) Update feature counter from timeline
        if self._timeline and self._timeline.features:
            self._feature_counter = len(self._timeline.features)

        # Remember the URL so we can distinguish Save from Open
        try:
            self._last_stage_url = omni.usd.get_context().get_stage_url()
        except Exception:
            pass

        self._set_status("Stage loaded — SparkWorks state restored")
        _notify("SparkWorks state restored from saved stage.")
        print(f"[{EXTENSION_NAME}] Stage reload complete: "
              f"{self._feature_counter} features, {self._body_counter} bodies")

    def _restore_profiles_after_save(self):
        """
        Re-create profile overlays after a save operation.

        When Isaac Sim saves, it may fire an OPENED event that disrupts
        profile prims.  If we had profiles visible before the save, we
        recreate them.
        """
        if self._profile_faces and self._profile_sketch_name:
            saved_faces = list(self._profile_faces)
            saved_indices = set(self._selected_profile_indices)
            self._profile_paths = []  # old paths are stale
            self._recreate_profile_overlays(saved_faces, saved_indices)
            print(f"[{EXTENSION_NAME}] Restored {len(saved_faces)} profiles after save")

    def _on_timeline_prim_clicked(self, tl_prim_name: str):
        """
        Handle a click on a timeline prim in the USD hierarchy.

        The prim name has the form "F00_Sketch1" or "F01_Extrude1".
        We extract the index and scrub to that feature, which also
        shows the sketch in the Sketch View if applicable.
        """
        try:
            # Parse the feature index from the prim name prefix "FNN_"
            idx_str = tl_prim_name.split("_")[0].lstrip("F")
            idx = int(idx_str)
        except (ValueError, IndexError):
            print(f"[{EXTENSION_NAME}] Could not parse timeline index from '{tl_prim_name}'")
            return

        features = self._timeline.features
        if 0 <= idx < len(features):
            self._action_select_feature(idx)
            self._timeline_panel.set_selected(idx)
            print(f"[{EXTENSION_NAME}] Timeline prim '{tl_prim_name}' -> selected index {idx}")

    def _on_construction_plane_clicked(self, plane: ConstructionPlane):
        """Handle a click on a construction plane (origin or face)."""
        # Track whether this is a face plane (for boolean join on extrude)
        if "_Face" in plane.name and self._face_plane_body:
            self._sketch_parent_body = self._face_plane_body
        else:
            self._sketch_parent_body = None

        if self._picking_plane:
            # In picking mode — create the sketch immediately
            self._picking_plane = False
            self._action_new_sketch_on_plane(plane)
        else:
            # Not in picking mode — just acknowledge the selection
            self._toolbar.set_plane_hint(f"Selected: {plane.name} — click Create Sketch")
            self._set_status(f"Plane '{plane.name}' selected")
        print(f"[{EXTENSION_NAME}] Plane clicked: {plane.name}")

    def _on_body_clicked(self, body_name: str):
        """
        Handle a click on a body mesh.

        Face planes are only shown when in picking mode (after clicking
        *Create Sketch*).  Outside of picking mode the click is ignored.
        """
        if not self._picking_plane:
            self._set_status(f"Body '{body_name}' — click Create Sketch first to sketch on a face")
            return

        solid = self._timeline.bodies.get(body_name)
        if solid is None:
            self._set_status(f"Body '{body_name}' has no solid geometry")
            return

        # Clear any previously tracked face planes from another body
        self._clear_face_planes()

        # Extract face planes from this specific body's solid
        face_planes = extract_face_planes(solid, body_name=body_name)
        if not face_planes:
            self._set_status(f"No planar faces found on '{body_name}'")
            _notify(f"No planar faces on '{body_name}'.", "warning")
            return

        try:
            # Ensure USD prims exist (re-create if missing, e.g. after reload)
            self._bridge.remove_body_face_planes(body_name)
            self._bridge.create_body_face_planes(body_name, face_planes)

            for fp in face_planes:
                self._face_map[fp.name] = fp
            self._face_planes = face_planes
            self._face_plane_body = body_name
            n = len(face_planes)
            self._set_status(
                f"'{body_name}' — {n} face(s) shown. Click a face to create a sketch."
            )
            self._toolbar.set_plane_hint(
                f"{n} faces on {body_name} — click one to create a sketch"
            )
            print(f"[{EXTENSION_NAME}] Showing {n} face planes for {body_name}")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to show face planes: {e}")
            import traceback
            traceback.print_exc()

    # Profile selection is handled directly in _on_stage_event to support
    # multi-select (Ctrl+click).  The USD selection is mirrored into
    # self._selected_profile_indices.

    def _rebuild_profile_faces(self, sketch_name: str):
        """Rebuild cached _profile_faces from a sketch in the registry."""
        try:
            sketch = self._sketch_registry.get(sketch_name) if self._sketch_registry else None
            if sketch is None:
                print(f"[{EXTENSION_NAME}] Cannot rebuild profiles: sketch '{sketch_name}' not in registry")
                return
            faces = sketch.build_all_faces()
            if faces:
                self._profile_faces = faces
                self._profile_sketch_name = sketch_name
                # Also rebuild plane normal from the sketch
                try:
                    plane = sketch.plane
                    n = plane.z_dir
                    self._profile_plane_normal = (float(n.X), float(n.Y), float(n.Z))
                except Exception:
                    pass
                # Collect existing profile prim paths from the stage
                self._profile_paths = self._bridge.get_profile_paths(sketch_name)
                print(f"[{EXTENSION_NAME}] Rebuilt {len(faces)} profile faces for '{sketch_name}'")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Error rebuilding profile faces: {e}")
            import traceback
            traceback.print_exc()

    def _recreate_profile_overlays(self, faces: list, selected_indices: set):
        """
        Re-create profile overlays.

        Uses the cached sketch name and plane normal.
        """
        if not faces or not self._profile_sketch_name:
            return
        self._profile_faces = faces
        self._profile_paths = self._bridge.create_profile_overlays(
            faces,
            self._profile_sketch_name,
            plane_normal=self._profile_plane_normal,
        )
        self._set_profile_selection(set(selected_indices) if selected_indices else set())

    def _clear_profile_overlays(self):
        """Clear the currently tracked profile overlays.

        Only removes the profiles for the *currently tracked* sketch
        (``self._profile_sketch_name``).  Other sketches' profiles are
        left intact — their visibility is managed by
        ``_sync_profile_visibility`` during scrubbing.
        """
        if self._profile_paths:
            self._bridge.remove_profile_overlays(self._profile_paths)
        self._profile_paths = []
        self._profile_faces = []
        self._set_profile_selection(set())
        self._profile_sketch_name = None
        self._profile_plane_normal = None

    def _sync_profile_visibility(self, marker_position: int):
        """
        Show profiles for sketches at or before *marker_position*;
        hide profiles for sketches after the marker.

        Profiles live under ``/World/SparkWorks/Sketches/<name>/Profiles``
        and are toggled via USD visibility so they persist across saves.
        """
        features = self._timeline.features if self._timeline else []
        for i, feature in enumerate(features):
            if feature.is_sketch and feature.sketch:
                visible = (i <= marker_position)
                self._bridge.set_sketch_profiles_visible(
                    feature.sketch.name, visible
                )

    def _refresh_profile_overlays_for_active_sketch(self):
        """
        Rebuild profile overlays for the currently active sketch.

        Called whenever primitives change in an existing (timeline) sketch
        so the 3D profile visualisation stays in sync.
        """
        sketch = self._active_sketch
        if sketch is None:
            return

        # Clear old overlays for this sketch
        if self._profile_paths:
            self._bridge.remove_profile_overlays(self._profile_paths)
            self._profile_paths = []
            self._profile_faces = []

        faces = sketch.build_all_faces()
        if not faces:
            self._set_profile_selection(set())
            return

        plane_normal = None
        try:
            plane = sketch.plane
            n = plane.z_dir
            plane_normal = (float(n.X), float(n.Y), float(n.Z))
        except Exception:
            pass

        self._profile_sketch_name = sketch.name
        self._profile_plane_normal = plane_normal
        self._profile_faces = faces
        self._profile_paths = self._bridge.create_profile_overlays(
            faces,
            sketch.name,
            plane_normal=plane_normal,
        )
        count = len(self._profile_paths)
        if count > 0:
            self._set_profile_selection({0})
        else:
            self._set_profile_selection(set())
        print(f"[{EXTENSION_NAME}] Refreshed {count} profile overlays for '{sketch.name}'")

    def _create_profile_overlays(self, sketch: Sketch):
        """
        Detect closed loops in the sketch, build faces, and create
        semi-transparent overlays under the sketch's Sketches prim.
        """
        # Clear any existing profile overlays first
        self._clear_profile_overlays()

        faces = sketch.build_all_faces()
        if not faces:
            print(f"[{EXTENSION_NAME}] No closed profiles detected in '{sketch.name}'")
            return

        # Get the sketch plane's normal as a fallback for profile offset
        plane_normal = None
        try:
            plane = sketch.plane
            n = plane.z_dir
            plane_normal = (float(n.X), float(n.Y), float(n.Z))
        except Exception:
            pass

        # Cache info needed to recreate profiles
        self._profile_sketch_name = sketch.name
        self._profile_plane_normal = plane_normal

        self._profile_faces = faces
        self._profile_paths = self._bridge.create_profile_overlays(
            faces,
            sketch.name,
            plane_normal=plane_normal,
        )
        n = len(self._profile_paths)
        if n > 0:
            # Auto-select the first (and often only) profile
            self._set_profile_selection({0})
            self._set_status(
                f"{n} profile(s) detected in '{sketch.name}' — "
                f"click one then Extrude, or Extrude directly"
            )
            _notify(f"{n} extrudable profile(s) found.")
        print(f"[{EXTENSION_NAME}] Created {n} profile overlays for '{sketch.name}'")

    def _register_all_face_planes(self):
        """
        Ensure all existing body face planes are in ``_face_map``
        so they can be detected when clicked.  Called when entering picking mode.
        """
        bodies = self._timeline.bodies
        if not bodies:
            return
        try:
            for bn, body_solid in bodies.items():
                if body_solid is None:
                    continue
                face_planes = extract_face_planes(body_solid, body_name=bn)
                for fp in face_planes:
                    self._face_map[fp.name] = fp
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Error registering face planes: {e}")

    def _create_hidden_face_planes(self, body_name: str):
        """
        Extract face planes from a body and create them as USD prims.
        Also registers them in ``_face_map`` so clicking them works.
        """
        # Look up the specific body's solid from the bodies dict.
        solid = self._timeline.bodies.get(body_name)
        if solid is None:
            return

        # Remove old face planes for this body first (if any)
        self._bridge.remove_body_face_planes(body_name)

        face_planes = extract_face_planes(solid, body_name=body_name)
        if not face_planes:
            return

        try:
            self._bridge.create_body_face_planes(body_name, face_planes)
            # Visibility is managed by the user via the stage tree — do NOT
            # programmatically hide or show construction planes.
            # Register in the face lookup so they can be selected
            for fp in face_planes:
                self._face_map[fp.name] = fp
            print(f"[{EXTENSION_NAME}] Pre-created {len(face_planes)} hidden face planes for '{body_name}'")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to pre-create face planes: {e}")
            import traceback
            traceback.print_exc()

    def _clear_face_planes(self):
        """Clear internal face-plane tracking state.

        Visibility is managed by the user via the stage tree — we do NOT
        programmatically hide or show construction planes here.
        """
        if self._face_planes:
            for fp in self._face_planes:
                self._face_map.pop(fp.name, None)
        self._face_planes = []
        self._face_plane_body = None
        self._selected_face_name = None

    def _action_new_sketch_on_plane(self, plane: ConstructionPlane):
        """Create a new sketch on a specific ConstructionPlane."""
        self._picking_plane = False  # always exit picking mode

        # The registry auto-generates the next name (Sketch1, Sketch2, …)
        next_num = self._sketch_registry.counter + 1
        name = f"Sketch{next_num}"
        self._active_sketch = Sketch(
            name=name,
            plane_name=plane.plane_type,
            construction_plane=plane,
        )
        self._editing_feature_index = None  # brand-new sketch, not editing
        self._toolbar.set_sketch_mode(True)
        self._toolbar.set_plane_hint(f"Sketching on {plane.name}")
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._update_sketch_view()

        # Hide other sketches' primitives to reduce clutter (Fusion 360 style).
        # The new sketch doesn't have a Primitives prim yet, so pass its name
        # as the exception (no-op for it, hides everything else).
        self._bridge.hide_all_sketches(except_sketch=name)

        # Clear internal tracking now that a face has been chosen
        self._clear_face_planes()

        # Automatically switch to the Sketch View tab
        self._focus_sketch_view()

        self._set_status(f"Sketching '{name}' on {plane.name} — select a drawing tool")
        _notify(f"New sketch '{name}' on {plane.name}.")

    def _action_add_plane(self):
        """Stub for adding a custom offset/rotated plane (future feature)."""
        _notify("Custom plane creation coming soon.", "info")

    # ========================================================================
    # Viewport Events (mouse / keyboard)
    # ========================================================================

    def _on_viewport_click(self, world_x: float, world_y: float, button: int):
        """Handle a click in the sketch viewport."""
        if self._active_sketch is None:
            return

        if button == 0:  # Left click — draw
            self._sketch_tool.on_click(world_x, world_y)
        elif button == 1:  # Right click — finish chain
            self._sketch_tool.on_finish()

    def _on_viewport_move(self, world_x: float, world_y: float):
        """Handle mouse movement in the sketch viewport."""
        if self._active_sketch is None:
            return
        self._sketch_tool.on_mouse_move(world_x, world_y)

    def _on_viewport_key(self, key: int, pressed: bool):
        """Handle a key press in the sketch viewport."""
        if not pressed:
            return
        if key in (KEY_ENTER, KEY_NUMPAD_ENTER):
            self._sketch_tool.on_finish()
        elif key == KEY_ESCAPE:
            self._sketch_tool.cancel()
            self._toolbar.set_active_tool(None)

    # ========================================================================
    # Sketch Tool Events
    # ========================================================================

    def _on_tool_primitive_created(self, primitive):
        """Called when the sketch tool completes a primitive (line, rect, circle)."""
        if self._active_sketch is None:
            return

        from .kernel.sketch import SketchLine, SketchRect, SketchCircle

        if isinstance(primitive, SketchLine):
            self._active_sketch.add_line(
                start=primitive.start, end=primitive.end
            )
        elif isinstance(primitive, SketchRect):
            self._active_sketch.add_rectangle(
                width=primitive.width,
                height=primitive.height,
                center=primitive.center,
            )
        elif isinstance(primitive, SketchCircle):
            self._active_sketch.add_circle(
                radius=primitive.radius,
                center=primitive.center,
            )

        prim_idx = len(self._active_sketch.primitives) - 1
        self._update_sketch_view()

        # Show the new primitive's properties in the panel
        if self._active_sketch.primitives:
            last_prim = self._active_sketch.primitives[-1]
            self._property_panel.show_sketch_primitive(last_prim, prim_idx)

        # Refresh profile overlays so the 3D viewport stays up-to-date
        if self._editing_feature_index is not None:
            self._refresh_profile_overlays_for_active_sketch()

    def _on_tool_preview_changed(self):
        """Called when the rubber-band preview or placed-points change."""
        self._sketch_viewport.set_placed_points(self._sketch_tool.points)

        # Rectangle preview
        rect_preview = self._sketch_tool.preview_rect
        if rect_preview:
            c1, c2 = rect_preview
            self._sketch_viewport.set_preview_rect(c1, c2)
            self._sketch_viewport.set_preview_line(None, None)
            return
        else:
            self._sketch_viewport.set_preview_rect(None, None)

        # Line preview (also used for circle radius preview)
        preview = self._sketch_tool.preview_line
        if preview:
            from_pt, to_pt = preview
            self._sketch_viewport.set_preview_line(from_pt, to_pt)
        else:
            self._sketch_viewport.set_preview_line(None, None)

    def _on_tool_chain_cancelled(self, count: int):
        """
        Called when the user presses Escape — remove the last *count*
        primitives that were created during this chain.
        """
        if self._active_sketch is None or count <= 0:
            return
        prims = self._active_sketch.primitives
        for _ in range(min(count, len(prims))):
            prims.pop()
        self._update_sketch_view()
        self._property_panel._show_no_selection()
        # Refresh profiles after undo
        if self._editing_feature_index is not None:
            self._refresh_profile_overlays_for_active_sketch()

    def _on_primitive_edited(self, primitive, index: int):
        """Called when the user edits a primitive's properties in the panel."""
        self._update_sketch_view()

    # ========================================================================
    # Toolbar Actions
    # ========================================================================

    def _action_create_sketch(self):
        """
        Handle the 'Create Sketch' button.

        Check if a construction plane is *currently* selected in the USD
        viewport.  If so, use it immediately.  Otherwise enter picking
        mode so the user can click one.
        """
        # Clear tracked profile selection (profiles remain in USD)
        self._profile_paths = []
        self._profile_faces = []
        self._set_profile_selection(set())
        self._profile_sketch_name = None
        self._profile_plane_normal = None

        # Check if a construction plane is actively selected right now
        currently_selected = self._get_currently_selected_plane()
        if currently_selected is not None:
            self._action_new_sketch_on_plane(currently_selected)
            return

        # No plane actively selected — enter picking mode
        self._picking_plane = True

        # Ensure all face planes are in the lookup map so clicking works
        self._register_all_face_planes()

        self._set_status("Select a construction plane or a body face")
        self._toolbar.set_plane_hint(
            "Click an origin plane or a body face"
        )
        self._focus_viewport()

    def _get_currently_selected_plane(self) -> Optional[ConstructionPlane]:
        """
        Return the ConstructionPlane if the current USD selection is
        exactly one construction plane or body face prim.  Otherwise return None.

        Checks construction planes first, then body face planes.
        Also sets ``_sketch_parent_body`` when the selected plane is a
        body face plane (needed for join-on-extrude).
        """
        try:
            usd_context = omni.usd.get_context()
            paths = usd_context.get_selection().get_selected_prim_paths()
            if not paths or len(paths) != 1:
                return None
            prim_path = paths[0]

            # Check origin / user construction planes first
            plane_name = self._bridge.is_construction_plane(prim_path)
            if plane_name and plane_name in self._construction_plane_map:
                self._sketch_parent_body = None
                return self._construction_plane_map[plane_name]

            # Then check body face planes
            face_name = self._bridge.is_body_face(prim_path)
            if face_name and face_name in self._face_map:
                plane = self._face_map[face_name]
                # Track body for join extrude
                if self._face_plane_body:
                    self._sketch_parent_body = self._face_plane_body
                else:
                    self._sketch_parent_body = None
                return plane
        except Exception:
            pass
        return None

    def _action_new_sketch(self, plane_name: str = "XY"):
        """Legacy sketch creation from plane name string (fallback)."""
        next_num = self._sketch_registry.counter + 1
        name = f"Sketch{next_num}"
        self._active_sketch = Sketch(name=name, plane_name=plane_name)
        self._toolbar.set_sketch_mode(True)
        self._toolbar.set_plane_hint(f"Sketching on {plane_name}")
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._update_sketch_view()
        self._set_status(f"Sketching '{name}' on {plane_name} — select a drawing tool")
        _notify(f"New sketch '{name}' on {plane_name} plane.")

    def _action_finish_sketch(self):
        if self._active_sketch is None:
            self._set_status("No active sketch to finish")
            _notify("No active sketch to finish.", "warning")
            return
        count = len(self._active_sketch.primitives)
        name = self._active_sketch.name

        # Keep a reference to the sketch before clearing self._active_sketch
        finished_sketch = self._active_sketch
        editing_index = self._editing_feature_index

        if editing_index is not None:
            # Editing an existing sketch — rebuild from that point
            # (the sketch object is already in the timeline, primitives
            # were added directly to it)
            try:
                self._timeline.rebuild_from(editing_index)
                self._set_status(f"'{name}' updated ({count} primitives)")
                _notify(f"Sketch '{name}' updated with {count} primitives.")
            except Exception as e:
                print(f"[{EXTENSION_NAME}] Error rebuilding after sketch edit: {e}")
                import traceback
                traceback.print_exc()
        else:
            # New sketch — insert at marker position.
            # Pass the sketch name as sketch_id so the registry uses it.
            try:
                marker = self._timeline_panel.marker_position
                self._timeline.add_sketch(
                    finished_sketch,
                    insert_after=marker,
                    sketch_id=finished_sketch.name,
                )
                self._set_status(f"'{name}' added ({count} primitives) — ready for operations")
                _notify(f"Sketch '{name}' added with {count} primitives.")
            except Exception as e:
                print(f"[{EXTENSION_NAME}] Error adding sketch to timeline: {e}")
                import traceback
                traceback.print_exc()
                self._set_status(f"'{name}' saved ({count} primitives)")

        # Explicit save to USD (safety net if callback chain had issues)
        self._save_timeline_to_usd()

        # Always reset state — button label, viewport, tools
        self._active_sketch = None
        self._editing_feature_index = None
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._sketch_viewport.clear_preview()
        self._update_sketch_view()
        self._toolbar.set_sketch_mode(False)
        self._toolbar.set_plane_hint("Click a plane in the viewport to start a sketch")
        self._focus_viewport()

        # ---- Profile detection & overlay ----
        # After returning to the viewport, detect and visualize closed
        # profiles from the sketch so they appear on the surface
        if finished_sketch.primitives:
            self._create_profile_overlays(finished_sketch)

    # -- Tool activation (interactive drawing) --------------------------------

    def _action_tool_line(self):
        if self._active_sketch is None:
            self._set_status("Click a construction plane first to start a sketch")
            _notify("Click a plane first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.LINE)
        self._toolbar.set_active_tool("line")

    def _action_tool_rectangle(self):
        if self._active_sketch is None:
            self._set_status("Click a construction plane first to start a sketch")
            _notify("Click a plane first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.RECTANGLE)
        self._toolbar.set_active_tool("rectangle")

    def _action_tool_circle(self):
        if self._active_sketch is None:
            self._set_status("Click a construction plane first to start a sketch")
            _notify("Click a plane first.", "warning")
            return
        self._sketch_tool.activate_tool(SketchToolMode.CIRCLE)
        self._toolbar.set_active_tool("circle")

    # -- 3D operations --------------------------------------------------------

    def _action_extrude(self):
        # Allow extrude if a body face is selected (Press/Pull) OR
        # the timeline has features (sketch-based extrude).
        if not self._selected_face_name and not self._timeline.features:
            self._set_status("Select a body face or add a sketch first")
            _notify("Select a body face or add a sketch first.", "warning")
            return

        # Show a dialog asking for the extrude distance
        self._show_extrude_dialog()

    def _show_extrude_dialog(self):
        """Pop up a small dialog to set the extrude distance and merge target."""
        if self._extrude_dialog_window:
            self._extrude_dialog_window.visible = False
            self._extrude_dialog_window = None

        is_face = bool(self._selected_face_name)
        title = "Extrude Face" if is_face else "Extrude"

        # Determine the source body (the body we're extruding from).
        source_body = None
        if is_face:
            src_body, _ = self._parse_face_name(self._selected_face_name)
            if src_body:
                source_body = src_body
        elif self._sketch_parent_body:
            source_body = self._sketch_parent_body

        # Merge options: "None" (new body) + the source body if applicable
        merge_options = ["None"]
        if source_body:
            merge_options.append(source_body)

        # Default: "None" unless this is a face extrude (merge is natural)
        default_merge_idx = 0
        if is_face and source_body:
            default_merge_idx = merge_options.index(source_body)

        self._extrude_dialog_window = ui.Window(
            title, width=320, height=155,
        )
        self._extrude_dist_model = ui.SimpleFloatModel(10.0)
        self._extrude_merge_options = merge_options

        with self._extrude_dialog_window.frame:
            with ui.VStack(spacing=6):
                ui.Spacer(height=4)
                with ui.HStack(height=24, spacing=4):
                    ui.Spacer(width=8)
                    ui.Label("Distance:", width=80)
                    ui.FloatField(
                        model=self._extrude_dist_model, width=ui.Fraction(1),
                    )
                    ui.Spacer(width=8)
                with ui.HStack(height=24, spacing=4):
                    ui.Spacer(width=8)
                    ui.Label("Merge into:", width=80)
                    self._extrude_merge_combo = ui.ComboBox(
                        default_merge_idx, *merge_options,
                        width=ui.Fraction(1),
                    )
                    ui.Spacer(width=8)
                ui.Spacer(height=4)
                with ui.HStack(height=30, spacing=8):
                    ui.Spacer(width=8)
                    ui.Button("OK", width=ui.Fraction(1),
                              clicked_fn=self._on_extrude_dialog_ok,
                              style={"Button": {"background_color": 0xFF3A7D44}})
                    ui.Button("Cancel", width=ui.Fraction(1),
                              clicked_fn=self._on_extrude_dialog_cancel)
                    ui.Spacer(width=8)
                ui.Spacer(height=4)

    def _on_extrude_dialog_ok(self):
        dist = self._extrude_dist_model.as_float if self._extrude_dist_model else 10.0

        # Read merge target from the combo box widget's own model
        merge_idx = 0
        combo = getattr(self, "_extrude_merge_combo", None)
        if combo is not None:
            merge_idx = combo.model.get_item_value_model().as_int
        merge_options = getattr(self, "_extrude_merge_options", ["None"])
        merge_target = merge_options[merge_idx] if merge_idx < len(merge_options) else "None"

        # Close the dialog
        if self._extrude_dialog_window:
            self._extrude_dialog_window.visible = False
            self._extrude_dialog_window = None

        self._execute_extrude(dist, merge_target=merge_target)

    def _on_extrude_dialog_cancel(self):
        if self._extrude_dialog_window:
            self._extrude_dialog_window.visible = False
            self._extrude_dialog_window = None

    @staticmethod
    def _parse_face_name(face_name: str):
        """
        Parse a face key like ``"Body1_Face3"`` into ``("Body1", 3)``.

        Returns ``(None, -1)`` if the format is unexpected.
        """
        try:
            idx = face_name.rfind("_Face")
            if idx < 0:
                return (None, -1)
            body = face_name[:idx]
            face_idx = int(face_name[idx + 5:])
            return (body, face_idx)
        except (ValueError, IndexError):
            return (None, -1)

    def _execute_extrude(self, dist: float, merge_target: str = "None"):
        """Perform the actual extrude operation with the given distance.

        Args:
            dist: Extrusion distance.
            merge_target: ``"None"`` to create a fresh body, or the name
                of an existing body to fuse into.
        """
        self._feature_counter += 1

        # Resolve merge intent from the dialog choice
        is_merge = merge_target != "None"

        # ── Face-based extrude (Press/Pull) ──────────────────────────
        if self._selected_face_name:
            src_body, face_idx = self._parse_face_name(self._selected_face_name)
            if src_body is None or face_idx < 0:
                self._set_status("Could not resolve selected face")
                _notify("Invalid face selection.", "warning")
                return

            # Use the user's merge choice (default pre-selects the face's body)
            if is_merge:
                target_body = merge_target
                join_body = merge_target
            else:
                self._body_counter += 1
                target_body = f"Body{self._body_counter}"
                join_body = ""

            op = ExtrudeOperation(
                distance=dist,
                join=is_merge,
                body_name=target_body,
                join_body_name=join_body,
                face_body_name=src_body,
                face_index=face_idx,
            )
            op.name = f"Extrude{self._feature_counter}"
            self._current_body_name = target_body
            print(
                f"[{EXTENSION_NAME}] Face extrude: distance={dist}, "
                f"face={self._selected_face_name}, body={target_body}, merge={is_merge}"
            )
            self._set_status(
                f"Extruding face {face_idx} on '{src_body}' by {dist} units..."
            )

            marker = self._timeline_panel.marker_position
            self._timeline.add_operation(op, name=op.name, insert_after=marker)
            if is_merge:
                self._set_status(f"Extrude (d={dist}) fused onto {target_body}")
            else:
                self._set_status(f"Extrude (d={dist}) — {target_body} created")
            self._selected_face_name = None  # consumed

        # ── Sketch-profile-based extrude ─────────────────────────────
        else:
            profile_indices = sorted(self._selected_profile_indices) if self._selected_profile_indices else [0]
            profile_idx = profile_indices[0] if profile_indices else 0

            if is_merge:
                target_body = merge_target
                join_body = merge_target
            else:
                self._body_counter += 1
                target_body = f"Body{self._body_counter}"
                join_body = ""

            op = ExtrudeOperation(
                distance=dist,
                join=is_merge,
                body_name=target_body,
                join_body_name=join_body,
                profile_index=profile_idx,
                profile_indices=profile_indices,
            )
            op.name = f"Extrude{self._feature_counter}"
            self._current_body_name = target_body
            print(
                f"[{EXTENSION_NAME}] Extrude: distance={dist}, body={target_body}, "
                f"profile_indices={profile_indices}, merge={is_merge}"
            )

            if is_merge:
                self._set_status(
                    f"Extruding {dist} units (merge into {target_body})..."
                )
            else:
                self._set_status(f"Extruding {dist} units...")

            marker = self._timeline_panel.marker_position
            self._timeline.add_operation(op, name=op.name, insert_after=marker)

            if is_merge:
                self._set_status(
                    f"Extrude (d={dist}) merged into {target_body}"
                )
                self._sketch_parent_body = None
            else:
                self._set_status(
                    f"Extrude (d={dist}) complete — {target_body} created"
                )

        # Select the newly created/updated body in the viewport
        body_path = f"{self._bridge.bodies_root}/{self._current_body_name}"
        try:
            usd_context = omni.usd.get_context()
            usd_context.get_selection().set_selected_prim_paths([body_path], True)
        except Exception as exc:
            print(f"[{EXTENSION_NAME}] Could not select body after extrude: {exc}")

        # Pre-create face planes for the body (hidden until Create Sketch)
        self._create_hidden_face_planes(self._current_body_name)

    def _action_revolve(self):
        if not self._timeline.features:
            self._set_status("Add and finish a sketch first")
            _notify("Add a sketch first.", "warning")
            return
        angle = self._toolbar.revolve_angle
        self._feature_counter += 1
        self._body_counter += 1
        body_name = f"Body{self._body_counter}"
        op = RevolveOperation(angle=angle, axis_name="Z", body_name=body_name)
        op.name = f"Revolve{self._feature_counter}"
        self._current_body_name = body_name
        marker = self._timeline_panel.marker_position
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._set_status(f"Revolve ({angle}) complete — {body_name} created")

    def _get_selected_body_names(self) -> List[str]:
        """Return the names of all currently selected bodies in the stage."""
        selected = []
        try:
            usd_context = omni.usd.get_context()
            paths = usd_context.get_selection().get_selected_prim_paths()
            all_bodies = self._get_all_body_names()
            for path in (paths or []):
                body_name = self._bridge.is_body_prim(path)
                if body_name and body_name in all_bodies:
                    selected.append(body_name)
        except Exception:
            pass
        return selected

    def _get_selected_body_name(self) -> Optional[str]:
        """Return the name of the first currently selected body, or fallback to last body."""
        selected = self._get_selected_body_names()
        if selected:
            return selected[0]
        bodies = self._get_all_body_names()
        if bodies:
            return bodies[-1]
        return None

    def _get_all_body_names(self) -> List[str]:
        """Authoritative list of body names from USD stage, falling back to timeline."""
        try:
            names = self._bridge.get_body_names()
            if names:
                return names
        except Exception:
            pass
        return list(self._timeline.bodies.keys())

    def _action_fillet(self):
        body_name = self._get_selected_body_name()
        if not body_name:
            self._set_status("Select a body first (or create one with sketch + extrude)")
            _notify("Select a body first.", "warning")
            return
        radius = self._toolbar.fillet_radius
        self._feature_counter += 1
        op = FilletOperation(radius=radius, body_name=body_name)
        op.name = f"Fillet{self._feature_counter}"
        marker = self._timeline_panel.marker_position
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._set_status(f"Fillet (r={radius}) applied to {body_name}")

    def _action_chamfer(self):
        body_name = self._get_selected_body_name()
        if not body_name:
            self._set_status("Select a body first (or create one with sketch + extrude)")
            _notify("Select a body first.", "warning")
            return
        length = self._toolbar.chamfer_length
        self._feature_counter += 1
        op = ChamferOperation(length=length, body_name=body_name)
        op.name = f"Chamfer{self._feature_counter}"
        marker = self._timeline_panel.marker_position
        self._timeline.add_operation(op, name=op.name, insert_after=marker)
        self._set_status(f"Chamfer (l={length}) applied to {body_name}")

    # -- Boolean operations ----------------------------------------------------

    def _action_boolean(self, mode: str):
        """
        Start a boolean merge.  Requires at least 2 bodies.
        Opens a dialog to choose target and tool bodies.
        """
        body_names = self._get_all_body_names()
        if len(body_names) < 2:
            self._set_status("Need at least 2 bodies for a boolean operation")
            _notify("Need at least 2 bodies for a boolean operation.", "warning")
            return

        self._boolean_pending_mode = mode
        self._show_boolean_dialog(mode, body_names)

    def _show_boolean_dialog(self, mode: str, body_names: list):
        """Pop up a dialog with two combo boxes for target/tool body."""
        if self._boolean_dialog_window:
            self._boolean_dialog_window.visible = False
            self._boolean_dialog_window = None

        title_map = {"union": "Boolean Union", "cut": "Boolean Cut", "intersect": "Boolean Intersect"}
        title = title_map.get(mode, "Boolean")

        self._boolean_dialog_window = ui.Window(
            title, width=340, height=180,
        )

        # Add "None" as the first option so no body is pre-selected by default.
        combo_items = ["None"] + body_names
        self._boolean_combo_items = combo_items

        # Pre-select from the current USD stage selection if the user
        # has multi-selected bodies; otherwise both default to "None" (idx 0).
        selected = self._get_selected_body_names()

        target_default = 0
        tool_default = 0
        if len(selected) >= 2:
            if selected[0] in combo_items:
                target_default = combo_items.index(selected[0])
            if selected[1] in combo_items:
                tool_default = combo_items.index(selected[1])
        elif len(selected) == 1:
            if selected[0] in combo_items:
                target_default = combo_items.index(selected[0])

        with self._boolean_dialog_window.frame:
            with ui.VStack(spacing=6):
                ui.Spacer(height=4)
                with ui.HStack(height=24, spacing=4):
                    ui.Spacer(width=8)
                    ui.Label("Target body:", width=90)
                    self._boolean_target_combo = ui.ComboBox(
                        target_default, *combo_items, width=ui.Fraction(1))
                    ui.Spacer(width=8)
                with ui.HStack(height=24, spacing=4):
                    ui.Spacer(width=8)
                    ui.Label("Tool body:", width=90)
                    self._boolean_tool_combo = ui.ComboBox(
                        tool_default, *combo_items, width=ui.Fraction(1))
                    ui.Spacer(width=8)
                with ui.HStack(height=24, spacing=4):
                    ui.Spacer(width=8)
                    ui.Label("Keep tool:", width=90)
                    self._boolean_keep_tool_model = ui.SimpleBoolModel(False)
                    ui.CheckBox(model=self._boolean_keep_tool_model, width=24)
                    ui.Spacer(width=ui.Fraction(1))
                    ui.Spacer(width=8)
                with ui.HStack(height=28, spacing=8):
                    ui.Spacer(width=8)
                    ui.Button("OK", width=ui.Fraction(1),
                              clicked_fn=self._on_boolean_dialog_ok,
                              style={"Button": {"background_color": 0xFF3A7D44}})
                    ui.Button("Cancel", width=ui.Fraction(1),
                              clicked_fn=self._on_boolean_dialog_cancel)
                    ui.Spacer(width=8)
                ui.Spacer(height=4)

    def _on_boolean_dialog_ok(self):
        mode = self._boolean_pending_mode
        combo_items = getattr(self, "_boolean_combo_items", [])
        keep_tool = self._boolean_keep_tool_model.as_bool

        # Read selections from the combo box widgets
        target_idx = 0
        tool_idx = 0
        target_combo = getattr(self, "_boolean_target_combo", None)
        tool_combo = getattr(self, "_boolean_tool_combo", None)
        if target_combo is not None:
            target_idx = target_combo.model.get_item_value_model().as_int
        if tool_combo is not None:
            tool_idx = tool_combo.model.get_item_value_model().as_int

        # Close the dialog
        if self._boolean_dialog_window:
            self._boolean_dialog_window.visible = False
            self._boolean_dialog_window = None

        # Validate selections — "None" (idx 0) is not a valid body
        if target_idx <= 0 or target_idx >= len(combo_items):
            self._set_status("Select a target body")
            _notify("Select a target body.", "warning")
            return
        if tool_idx <= 0 or tool_idx >= len(combo_items):
            self._set_status("Select a tool body")
            _notify("Select a tool body.", "warning")
            return
        if target_idx == tool_idx:
            self._set_status("Target and tool body must be different")
            _notify("Target and tool body must be different.", "warning")
            return

        target_name = combo_items[target_idx]
        tool_name = combo_items[tool_idx]

        self._execute_boolean(mode, target_name, tool_name, keep_tool)

    def _on_boolean_dialog_cancel(self):
        if self._boolean_dialog_window:
            self._boolean_dialog_window.visible = False
            self._boolean_dialog_window = None

    def _execute_boolean(self, mode: str, target_body: str, tool_body: str, keep_tool: bool = False):
        """Perform the actual boolean operation."""
        self._feature_counter += 1
        op = BooleanOperation(
            mode=mode,
            target_body=target_body,
            tool_body=tool_body,
            keep_tool=keep_tool,
        )
        op.name = f"Boolean{self._feature_counter}"

        marker = self._timeline_panel.marker_position
        self._timeline.add_operation(op, name=op.name, insert_after=marker)

        # Select the result body in the stage
        result_path = self._bridge.bodies_root + "/" + target_body
        try:
            import omni.usd
            ctx = omni.usd.get_context()
            if ctx:
                ctx.get_selection().set_selected_prim_paths([result_path], True)
        except Exception:
            pass

        self._set_status(f"Boolean {mode}: {target_body} {mode} {tool_body}")

    def _action_rebuild_all(self):
        self._set_status("Rebuilding...")
        self._timeline.rebuild_all()
        self._set_status("Rebuild complete")

    def _action_clear_all(self):
        self._sketch_registry = SketchRegistry()
        self._timeline = Timeline(sketch_registry=self._sketch_registry)
        self._connect_timeline_callbacks()
        self._clear_profile_overlays()
        self._bridge.clear()
        self._active_sketch = None
        self._feature_counter = 0
        self._body_counter = 0
        self._current_body_name = "Body1"
        self._face_planes = []
        self._face_plane_body = None
        self._sketch_parent_body = None
        self._profile_paths = []
        self._profile_faces = []
        self._set_profile_selection(set())
        self._profile_sketch_name = None
        self._profile_plane_normal = None
        self._sketch_tool.deactivate()
        self._toolbar.set_active_tool(None)
        self._toolbar.set_sketch_mode(False)
        self._toolbar.set_plane_hint("Click a plane in the viewport to start a sketch")
        self._timeline_panel.update_features([], -1)
        self._property_panel._show_no_selection()
        self._sketch_viewport.clear_preview()
        self._update_sketch_view()
        self._set_status("Cleared — click a construction plane to begin")

        # Re-create the origin planes
        self._construction_planes = create_origin_planes()
        asyncio.ensure_future(self._create_initial_planes())

    # ========================================================================
    # Timeline Panel Actions
    # ========================================================================

    def _action_select_feature(self, index: int):
        """
        A feature node was clicked — select it (show properties, allow
        editing if it's a sketch).  Does NOT move the marker / scrub.
        """
        features = self._timeline.features
        if not (0 <= index < len(features)):
            return
        feature = features[index]
        self._property_panel.show_feature(feature, index)
        self._set_status(f"Selected feature #{index + 1}: {feature.name}")

        # If it's a sketch, make it the active sketch so the user
        # can add primitives to it with the drawing tools.
        if feature.is_sketch and feature.sketch:
            self._active_sketch = feature.sketch
            self._editing_feature_index = index
            self._toolbar.set_sketch_mode(True)
            self._toolbar.set_plane_hint(
                f"Editing '{feature.sketch.name}' — select a drawing tool"
            )
            # Hide other sketches, show this one (Fusion 360 style)
            self._bridge.hide_all_sketches(except_sketch=feature.sketch.name)
            self._show_sketch_in_viewport(feature.sketch)
            self._refresh_profile_overlays_for_active_sketch()
        else:
            self._active_sketch = None
            self._editing_feature_index = None
            self._sketch_tool.deactivate()
            self._toolbar.set_active_tool(None)
            self._toolbar.set_sketch_mode(False)
            self._sketch_viewport.update_primitives([])
            self._sketch_viewport.clear_info()
            # Clear tracked selection but leave profile prims in USD —
            # their visibility is managed by _sync_profile_visibility.
            self._profile_paths = []
            self._profile_faces = []
            self._set_profile_selection(set())
            self._profile_sketch_name = None
            self._profile_plane_normal = None

    def _action_marker_moved(self, position: int):
        """
        The playback marker was dragged / clicked to a new position.

        ``position`` is the index of the last *included* feature.
        -1 = before all features (nothing visible).
        """
        # Reset tracked profile selection (we're scrubbing, not selecting)
        self._profile_paths = []
        self._profile_faces = []
        self._set_profile_selection(set())
        self._profile_sketch_name = None
        self._profile_plane_normal = None

        if position < 0:
            # Before all features — clear geometry
            self._timeline._scrub_index = -1
            self._timeline._bodies = {}
            self._timeline._current_sketch_face = None
            self._timeline._notify_rebuild()
            self._set_status("Marker at beginning — no features active")
        elif position >= len(self._timeline.features) - 1:
            # End — show everything
            self._timeline.scrub_to_end()
            self._set_status("Marker at end — showing latest state")
        else:
            self._timeline.scrub_to(position)
            self._set_status(
                f"Marker after feature #{position + 1}"
            )

        # Toggle profile visibility: show for sketches at/before marker,
        # hide for sketches after marker.
        self._sync_profile_visibility(position)

    def _action_scrub_to(self, index: int):
        """Convenience: move marker + select the feature at *index*."""
        self._action_marker_moved(index)
        self._action_select_feature(index)

    def _action_scrub_to_end(self):
        self._action_marker_moved(len(self._timeline.features) - 1)

    def _action_scrub_to_start(self):
        self._action_marker_moved(-1)

    def _show_sketch_in_viewport(self, sketch: Sketch):
        """Display a (read-only) sketch in the Sketch View (without switching tabs)."""
        hidden = self._get_sketch_hidden_indices(sketch)
        self._sketch_viewport.update_primitives(sketch.primitives, hidden)
        self._sketch_viewport.set_sketch_info(
            sketch.plane_name,
            sketch.name,
            len(sketch.primitives),
        )
        # Don't auto-focus — let the user stay in the 3D viewport

    def _action_delete_feature(self, index: int):
        features = self._timeline.features
        if 0 <= index < len(features):
            name = features[index].name
            self._timeline.remove_feature(index)
            self._set_status(f"Deleted '{name}'")

    def _action_suppress_feature(self, index: int, suppressed: bool):
        self._timeline.suppress_feature(index, suppressed)
        state = "suppressed" if suppressed else "unsuppressed"
        self._set_status(f"Feature #{index + 1} {state}")

    def _action_update_param(self, feature_index: int, param_name: str, value):
        self._timeline.update_feature_params(feature_index, {param_name: value})
        self._set_status(f"Updated {param_name}={value}, rebuilding...")

    # ========================================================================
    # Timeline Events -> Bridge + UI Updates
    # ========================================================================

    def _on_timeline_rebuild(self, bodies=None):
        if bodies is None:
            bodies = {}

        # Always clear ALL existing bodies first so stale geometry from
        # previous extrudes doesn't linger after inserts or reorders.
        try:
            all_bodies = self._bridge.get_body_names()
            for bn in all_bodies:
                self._bridge.update_body_mesh(None, body_name=bn)
        except Exception:
            pass

        if bodies:
            # Write each body from the bodies dict
            for bname, bsolid in bodies.items():
                if bsolid is not None:
                    prim_path = self._bridge.update_body_mesh(bsolid, body_name=bname)
                    if prim_path:
                        print(f"[{EXTENSION_NAME}] Body '{bname}' updated at {prim_path}")
                    self._create_hidden_face_planes(bname)
            # Sync counters from the bodies dict
            self._current_body_name = list(bodies.keys())[-1]
            max_num = 0
            for bname in bodies:
                try:
                    num = int(bname.replace("Body", ""))
                    max_num = max(max_num, num)
                except ValueError:
                    pass
            self._body_counter = max(self._body_counter, max_num)
        else:
            print(f"[{EXTENSION_NAME}] Rebuild produced no solid — all bodies cleared")
        self._timeline_panel.update_features(
            self._timeline.features,
            marker_pos=self._timeline.scrub_index,
            selected_idx=self._timeline_panel.selected_index,
        )

        # Toolbar visibility is driven by USD selection — no need to
        # force-show/hide tool groups here.

    def _on_features_changed(self, features):
        try:
            self._timeline_panel.update_features(
                features,
                marker_pos=self._timeline.scrub_index,
                selected_idx=self._timeline_panel.selected_index,
            )
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Timeline panel update error: {e}")

        # Persist timeline to USD.  Profiles now live under Sketches (not
        # Timeline), so they are NOT destroyed by this save.
        self._save_timeline_to_usd()

    def _save_timeline_to_usd(self):
        """Save both the sketch registry and timeline features to USD."""
        try:
            # 1) Persist all sketches under /World/SparkWorks/Sketches/
            self._bridge.save_sketches(self._sketch_registry)
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to save sketches to USD: {e}")
            import traceback
            traceback.print_exc()

        try:
            # 2) Persist the timeline (features with sketch_id references)
            features = self._timeline.features if self._timeline else []
            success = self._bridge.save_timeline(features)
            if not success:
                print(f"[{EXTENSION_NAME}] save_timeline returned False")
        except Exception as e:
            print(f"[{EXTENSION_NAME}] Failed to save timeline to USD: {e}")
            import traceback
            traceback.print_exc()
