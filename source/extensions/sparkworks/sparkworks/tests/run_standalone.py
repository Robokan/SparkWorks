#!/usr/bin/env python3
"""
Standalone SparkWorks test runner — runs via python.sh without Kit.

Usage (inside the container):
    /isaac-sim/python.sh /isaac-sim/exts/sparkworks/sparkworks/tests/run_standalone.py
    /isaac-sim/python.sh /isaac-sim/exts/sparkworks/sparkworks/tests/run_standalone.py -v
    /isaac-sim/python.sh /isaac-sim/exts/sparkworks/sparkworks/tests/run_standalone.py TestExtrude
    /isaac-sim/python.sh /isaac-sim/exts/sparkworks/sparkworks/tests/run_standalone.py TestExtrude.test_basic_extrude

This file bootstraps the package path (bypassing the Kit-dependent __init__.py)
and runs all test cases using standard unittest.
"""

from __future__ import annotations

import os
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Bootstrap: set up the sparkworks package without importing __init__.py
# (which pulls in omni.kit dependencies that aren't available outside Kit)
# ---------------------------------------------------------------------------
_EXT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _EXT_ROOT)

# Create a fake top-level sparkworks package so submodule imports work
_pkg_dir = os.path.join(_EXT_ROOT, "sparkworks")
_pkg = types.ModuleType("sparkworks")
_pkg.__path__ = [_pkg_dir]
_pkg.__package__ = "sparkworks"
_pkg.__file__ = os.path.join(_pkg_dir, "__init__.py")
sys.modules["sparkworks"] = _pkg


# =========================================================================
# Test Cases
# =========================================================================

from sparkworks.api import SparkWorksAPI  # noqa: E402


class TestSketch(unittest.TestCase):
    """Sketch creation and primitives."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)

    def test_create_empty_sketch(self):
        sk = self.api.create_sketch("XY")
        self.assertEqual(sk.name, "Sketch1")
        self.assertEqual(len(sk.primitives), 0)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    def test_sketch_with_rectangle(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 20, 10)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)
        self.assertEqual(len(sk.primitives), 1)

    def test_sketch_with_circle(self):
        sk = self.api.create_sketch("XZ")
        self.api.add_circle(sk, radius=5)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    def test_sketch_with_line(self):
        sk = self.api.create_sketch("YZ")
        self.api.add_line(sk, (0, 0), (10, 0))
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    def test_sketch_with_multiple_primitives(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 20, 10)
        self.api.add_circle(sk, radius=3, center=(5, 5))
        self.api.add_line(sk, (-10, -10), (10, 10))
        self.assertEqual(len(sk.primitives), 3)

    def test_multiple_sketches_naming(self):
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 10, 10)
        self.api.finish_sketch(sk1)
        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=5)
        self.api.finish_sketch(sk2)
        self.assertEqual(sk1.name, "Sketch1")
        self.assertEqual(sk2.name, "Sketch2")
        self.assertEqual(self.api.feature_count, 2)

    def test_sketch_registry(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.assertIn("Sketch1", self.api.sketch_registry.sketches)

    def test_get_sketch(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        found = self.api.get_sketch("Sketch1")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Sketch1")

    def test_get_sketch_not_found(self):
        self.assertIsNone(self.api.get_sketch("NoSuchSketch"))


class TestExtrude(unittest.TestCase):
    """Extrude operations."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)

    def _make_box(self, w=10, h=10, d=5, body_name=None):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, w, h)
        self.api.finish_sketch(sk)
        return self.api.extrude(distance=d, body_name=body_name)

    def test_basic_extrude(self):
        body = self._make_box()
        self.assertEqual(body, "Body1")
        self.assertEqual(len(self.api.bodies), 1)
        self.assertIn("Body1", self.api.bodies)
        self.assertIsNotNone(self.api.get_body_solid("Body1"))

    def test_extrude_produces_solid(self):
        self._make_box()
        self.assertIsNotNone(self.api.get_body_solid("Body1"))

    def test_custom_body_name(self):
        body = self._make_box(body_name="MyBox")
        self.assertEqual(body, "MyBox")
        self.assertIn("MyBox", self.api.bodies)

    def test_two_independent_bodies(self):
        self._make_box(w=10, h=10, d=5)
        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=3)
        self.api.finish_sketch(sk2)
        self.api.extrude(distance=8)
        self.assertEqual(len(self.api.bodies), 2)
        self.assertIn("Body1", self.api.bodies)
        self.assertIn("Body2", self.api.bodies)

    def test_join_extrude(self):
        self._make_box()
        sk2 = self.api.create_sketch("XY")
        self.api.add_circle(sk2, radius=3)
        self.api.finish_sketch(sk2)
        body = self.api.extrude(distance=10, join_body="Body1")
        self.assertEqual(body, "Body1")
        self.assertEqual(len(self.api.bodies), 1)

    def test_feature_count(self):
        self._make_box()
        self.assertEqual(self.api.feature_count, 2)

    def test_extrude_face(self):
        self._make_box()
        faces = self.api.get_body_faces("Body1")
        self.assertGreater(len(faces), 0)
        body = self.api.extrude_face("Body1", face_index=0, distance=3)
        self.assertEqual(body, "Body1")
        self.assertEqual(len(self.api.bodies), 1)
        self.assertEqual(self.api.feature_count, 3)

    def test_symmetric_extrude(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=10, symmetric=True)
        self.assertGreater(len(self.api.bodies), 0)

    def test_three_independent_bodies(self):
        for i in range(3):
            sk = self.api.create_sketch("XY")
            self.api.add_rectangle(sk, 10 + i, 10 + i)
            self.api.finish_sketch(sk)
            self.api.extrude(distance=5 + i)
        self.assertEqual(len(self.api.bodies), 3)
        for i in range(1, 4):
            self.assertIn(f"Body{i}", self.api.bodies)


class TestOtherOps(unittest.TestCase):
    """Revolve, fillet, chamfer."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)

    def test_revolve(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 5, 2, center=(10, 0))
        self.api.finish_sketch(sk)
        self.api.revolve(angle=360, axis="Z")
        self.assertGreater(len(self.api.bodies), 0)

    def test_fillet(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)
        self.api.fillet(radius=0.5)
        self.assertIsNotNone(self.api.get_body_solid("Body1"))
        self.assertEqual(self.api.feature_count, 3)

    def test_chamfer(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)
        self.api.chamfer(length=0.5)
        self.assertIsNotNone(self.api.get_body_solid("Body1"))
        self.assertEqual(self.api.feature_count, 3)


class TestTimeline(unittest.TestCase):
    """Timeline scrubbing, suppress, delete."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)
        # sketch1 -> extrude1 -> sketch2 -> extrude2
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 10, 10)
        self.api.finish_sketch(sk1)
        self.api.extrude(distance=5)
        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=3)
        self.api.finish_sketch(sk2)
        self.api.extrude(distance=8)

    def test_initial_state(self):
        self.assertEqual(self.api.feature_count, 4)
        self.assertEqual(len(self.api.bodies), 2)

    def test_scrub_to_middle(self):
        self.api.scrub_to(1)
        self.assertEqual(len(self.api.bodies), 1)
        self.assertIn("Body1", self.api.bodies)

    def test_scrub_to_start(self):
        self.api.scrub_to_start()
        self.assertEqual(len(self.api.bodies), 0)

    def test_scrub_to_end(self):
        self.api.scrub_to(1)
        self.api.scrub_to_end()
        self.assertEqual(len(self.api.bodies), 2)

    def test_suppress_extrude(self):
        self.api.suppress_feature(1, True)
        self.api.scrub_to_end()
        self.assertNotIn("Body1", self.api.bodies)

    def test_unsuppress_extrude(self):
        self.api.suppress_feature(1, True)
        self.api.suppress_feature(1, False)
        self.api.scrub_to_end()
        self.assertIn("Body1", self.api.bodies)

    def test_delete_feature(self):
        self.api.delete_feature(3)
        self.assertEqual(self.api.feature_count, 3)

    def test_clear_all(self):
        self.api.clear_all()
        self.assertEqual(self.api.feature_count, 0)
        self.assertEqual(len(self.api.bodies), 0)

    def test_rebuild_all(self):
        keys_before = set(self.api.bodies.keys())
        self.api.rebuild_all()
        keys_after = set(self.api.bodies.keys())
        self.assertEqual(keys_before, keys_after)

    def test_marker_position(self):
        self.assertEqual(self.api.marker_position, 3)
        self.api.move_marker(1)
        self.assertEqual(self.api.marker_position, 1)

    def test_scrub_through_multi_body(self):
        """Incremental body visibility when scrubbing."""
        self.api.scrub_to(1)
        self.assertEqual(len(self.api.bodies), 1)
        self.api.scrub_to(3)
        self.assertEqual(len(self.api.bodies), 2)


class TestQueries(unittest.TestCase):
    """Query methods."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)

    def test_features_list(self):
        features = self.api.features
        self.assertEqual(len(features), 2)
        self.assertTrue(features[0].is_sketch)
        self.assertTrue(features[1].is_operation)

    def test_get_body_solid(self):
        self.assertIsNotNone(self.api.get_body_solid("Body1"))

    def test_get_body_solid_missing(self):
        self.assertIsNone(self.api.get_body_solid("NoBody"))

    def test_body_has_faces(self):
        faces = self.api.get_body_faces("Body1")
        self.assertEqual(len(faces), 6)  # box = 6 planar faces

    def test_get_planar_face(self):
        face = self.api.get_planar_face("Body1", 0)
        self.assertIsNotNone(face)

    def test_get_planar_face_out_of_range(self):
        self.assertIsNone(self.api.get_planar_face("Body1", 999))

    def test_get_profiles(self):
        sk = self.api.get_sketch("Sketch1")
        profiles = self.api.get_profiles(sk)
        self.assertGreater(len(profiles), 0)

    def test_body_names_no_bridge(self):
        self.assertEqual(self.api.get_body_names(), ["Body1"])


class TestInsertAtMarker(unittest.TestCase):
    """Operations insert at the marker position."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)

    def test_insert_at_middle(self):
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 10, 10)
        self.api.finish_sketch(sk1)
        self.api.extrude(distance=5)
        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=3)
        self.api.finish_sketch(sk2)
        self.api.extrude(distance=8)
        self.assertEqual(self.api.feature_count, 4)

        self.api.move_marker(1)
        sk3 = self.api.create_sketch("YZ")
        self.api.add_rectangle(sk3, 5, 5)
        self.api.finish_sketch(sk3)
        self.api.extrude(distance=3)
        self.assertEqual(self.api.feature_count, 6)


class TestBoolean(unittest.TestCase):
    """Boolean body merge operations."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)
        # Two overlapping boxes
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 20, 20, center=(0, 0))
        self.api.finish_sketch(sk1)
        self.api.extrude(distance=10)

        sk2 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk2, 10, 10, center=(5, 5))
        self.api.finish_sketch(sk2)
        self.api.extrude(distance=15)

    def test_two_bodies_exist(self):
        self.assertEqual(len(self.api.bodies), 2)

    def test_boolean_union(self):
        result = self.api.boolean_union("Body1", "Body2")
        self.assertEqual(result, "Body1")
        self.assertNotIn("Body2", self.api.bodies)
        self.assertIsNotNone(self.api.bodies["Body1"])

    def test_boolean_cut(self):
        result = self.api.boolean_cut("Body1", "Body2")
        self.assertEqual(result, "Body1")
        self.assertNotIn("Body2", self.api.bodies)

    def test_boolean_intersect(self):
        result = self.api.boolean_intersect("Body1", "Body2")
        self.assertEqual(result, "Body1")
        self.assertNotIn("Body2", self.api.bodies)

    def test_boolean_keep_tool(self):
        self.api.boolean_union("Body1", "Body2", keep_tool=True)
        self.assertIn("Body1", self.api.bodies)
        self.assertIn("Body2", self.api.bodies)

    def test_boolean_same_body_raises(self):
        with self.assertRaises(ValueError):
            self.api.boolean_union("Body1", "Body1")

    def test_boolean_missing_body_raises(self):
        with self.assertRaises(ValueError):
            self.api.boolean_union("Body1", "BodyX")

    def test_boolean_timeline_feature_count(self):
        count_before = self.api.feature_count
        self.api.boolean_union("Body1", "Body2")
        self.assertEqual(self.api.feature_count, count_before + 1)

    def test_boolean_scrub_roundtrip(self):
        self.api.boolean_cut("Body1", "Body2")
        self.api.scrub_to_start()
        self.assertEqual(len(self.api.bodies), 0)
        self.api.scrub_to_end()
        self.assertIn("Body1", self.api.bodies)
        self.assertNotIn("Body2", self.api.bodies)

    def test_boolean_chain(self):
        """Union then cut with a third body."""
        sk3 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk3, 5, 5, center=(-5, -5))
        self.api.finish_sketch(sk3)
        self.api.extrude(distance=8)

        self.api.boolean_union("Body1", "Body2")
        self.assertEqual(len(self.api.bodies), 2)

        self.api.boolean_cut("Body1", "Body3")
        self.assertEqual(len(self.api.bodies), 1)


class TestConstraintSolver(unittest.TestCase):
    """Tests for the geometric constraint solver."""

    def test_solver_basic(self):
        """Solver can constrain a horizontal line distance."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver, ConstraintType
        s = ConstraintSolver()
        p1 = s.add_point(0, 0.1)   # slightly off horizontal
        p2 = s.add_point(10.3, -0.2)
        lid = s.add_line(p1, p2)
        s.constrain_horizontal(lid)
        s.constrain_distance_pp(p1, p2, 10.0)
        s.fix_point(p1)
        ok = s.solve()
        self.assertTrue(ok)
        x1, y1 = s.get_point(p1)
        x2, y2 = s.get_point(p2)
        self.assertAlmostEqual(y1, y2, places=4)
        self.assertAlmostEqual(abs(x2 - x1), 10.0, places=3)

    def test_solver_perpendicular(self):
        """Two lines can be constrained perpendicular."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver, ConstraintType
        s = ConstraintSolver()
        l1, p1, p2 = s.add_line_from_coords(0, 0, 10, 0.5)
        l2, p3, p4 = s.add_line_from_coords(5, 0, 5.3, 10)
        s.fix_point(p1)
        s.constrain_perpendicular(l1, l2)
        ok = s.solve()
        self.assertTrue(ok)
        # Verify perpendicularity: dot product of direction vectors should be ~0
        pa, pb = s.get_line_points(l1)
        pc, pd = s.get_line_points(l2)
        dx1 = pb[0] - pa[0]
        dy1 = pb[1] - pa[1]
        dx2 = pd[0] - pc[0]
        dy2 = pd[1] - pc[1]
        dot = dx1 * dx2 + dy1 * dy2
        self.assertAlmostEqual(dot, 0.0, places=3)

    def test_solver_coincident(self):
        """Coincident constraint snaps two points together."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver
        s = ConstraintSolver()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(5, 5)
        p3 = s.add_point(5.1, 4.9)  # nearly same as p2
        s.fix_point(p1)
        s.constrain_coincident(p2, p3)
        ok = s.solve()
        self.assertTrue(ok)
        x2, y2 = s.get_point(p2)
        x3, y3 = s.get_point(p3)
        self.assertAlmostEqual(x2, x3, places=4)
        self.assertAlmostEqual(y2, y3, places=4)

    def test_solver_parallel(self):
        """Parallel constraint makes two lines have the same direction."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver
        s = ConstraintSolver()
        l1, p1, p2 = s.add_line_from_coords(0, 0, 10, 1)
        l2, p3, p4 = s.add_line_from_coords(0, 5, 10, 6.5)
        s.fix_point(p1)
        s.fix_point(p3)
        s.constrain_parallel(l1, l2)
        ok = s.solve()
        self.assertTrue(ok)
        # Verify parallel: cross product should be ~0
        pa, pb = s.get_line_points(l1)
        pc, pd = s.get_line_points(l2)
        dx1 = pb[0] - pa[0]
        dy1 = pb[1] - pa[1]
        dx2 = pd[0] - pc[0]
        dy2 = pd[1] - pc[1]
        cross = dx1 * dy2 - dy1 * dx2
        self.assertAlmostEqual(cross, 0.0, places=2)

    def test_solver_equal_length(self):
        """Equal length constraint."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver
        import math
        s = ConstraintSolver()
        l1, p1, p2 = s.add_line_from_coords(0, 0, 7, 0)
        l2, p3, p4 = s.add_line_from_coords(0, 5, 12, 5)
        s.fix_point(p1)
        s.fix_point(p3)
        s.constrain_equal_length(l1, l2)
        ok = s.solve()
        self.assertTrue(ok)
        pa, pb = s.get_line_points(l1)
        pc, pd = s.get_line_points(l2)
        len1 = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        len2 = math.hypot(pd[0] - pc[0], pd[1] - pc[1])
        self.assertAlmostEqual(len1, len2, places=3)

    def test_solver_serialise_roundtrip(self):
        """Solver state survives to_dict / from_dict."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver
        s = ConstraintSolver()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(10, 0)
        lid = s.add_line(p1, p2)
        cid = s.constrain_horizontal(lid)
        d = s.to_dict()
        s2 = ConstraintSolver.from_dict(d)
        self.assertEqual(len(s2.entities), len(s.entities))
        self.assertEqual(len(s2.constraints), len(s.constraints))
        self.assertAlmostEqual(s2._params[0], 0.0)
        self.assertAlmostEqual(s2._params[2], 10.0)

    def test_solver_circle_radius(self):
        """Radius constraint on a circle."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver
        s = ConstraintSolver()
        cid, center = s.add_circle(5, 5, 3.0)
        s.fix_point(center)
        s.constrain_radius(cid, 7.0)
        ok = s.solve()
        self.assertTrue(ok)
        cx, cy, r = s.get_circle(cid)
        self.assertAlmostEqual(r, 7.0, places=3)

    def test_solver_drag(self):
        """Drag-solve: move a point while constraints hold."""
        from sparkworks.kernel.constraint_solver import ConstraintSolver
        s = ConstraintSolver()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(10, 0)
        lid = s.add_line(p1, p2)
        s.fix_point(p1)
        s.constrain_horizontal(lid)
        s.constrain_distance_pp(p1, p2, 10.0)
        ok = s.solve_drag(p2, 15, 3)  # drag p2 to (15, 3)
        self.assertTrue(ok)
        x2, y2 = s.get_point(p2)
        # Line is horizontal with distance 10 from fixed p1 at (0,0).
        # The drag target (15, 3) is infeasible, so the solver finds
        # the nearest feasible point: (10, 0).
        self.assertAlmostEqual(y2, 0.0, places=2)
        import math
        dist = math.hypot(x2, y2)
        self.assertAlmostEqual(dist, 10.0, places=2)


class TestSketchConstraints(unittest.TestCase):
    """Tests for constraint integration in the Sketch class."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)

    def test_enable_constraints_on_sketch(self):
        sk = self.api.create_sketch("XY")
        self.api.add_line(sk, (0, 0), (10, 0))
        self.api.enable_constraints(sk)
        self.assertIsNotNone(sk.solver)
        self.assertEqual(len(sk.solver.entities), 3)  # 2 points + 1 line

    def test_horizontal_constraint_on_sketch(self):
        sk = self.api.create_sketch("XY")
        self.api.add_line(sk, (0, 0), (10, 1))  # slightly non-horizontal
        self.api.enable_constraints(sk)
        ents = self.api.get_entity_ids(sk, 0)
        self.api.constrain_horizontal(sk, ents["line"])
        self.api.constrain_fixed(sk, ents["p1"], 0, 0)
        ok = self.api.solve_sketch(sk)
        self.assertTrue(ok)
        # After solving, line should be horizontal
        prim = sk.primitives[0]
        self.assertAlmostEqual(prim.start[1], prim.end[1], places=3)

    def test_distance_constraint_on_sketch(self):
        sk = self.api.create_sketch("XY")
        self.api.add_line(sk, (0, 0), (7, 0))
        self.api.enable_constraints(sk)
        ents = self.api.get_entity_ids(sk, 0)
        self.api.constrain_fixed(sk, ents["p1"], 0, 0)
        self.api.constrain_distance(sk, ents["p1"], ents["p2"], 15.0)
        ok = self.api.solve_sketch(sk)
        self.assertTrue(ok)
        import math
        prim = sk.primitives[0]
        d = math.hypot(prim.end[0] - prim.start[0], prim.end[1] - prim.start[1])
        self.assertAlmostEqual(d, 15.0, places=2)

    def test_coincident_connects_lines(self):
        sk = self.api.create_sketch("XY")
        self.api.add_line(sk, (0, 0), (10, 0))
        self.api.add_line(sk, (10.2, 0.1), (20, 5))  # nearly connected
        self.api.enable_constraints(sk)
        e0 = self.api.get_entity_ids(sk, 0)
        e1 = self.api.get_entity_ids(sk, 1)
        self.api.constrain_fixed(sk, e0["p1"], 0, 0)
        self.api.constrain_coincident(sk, e0["p2"], e1["p1"])
        ok = self.api.solve_sketch(sk)
        self.assertTrue(ok)
        # End of first line should match start of second
        p0_end = sk.primitives[0].end
        p1_start = sk.primitives[1].start
        self.assertAlmostEqual(p0_end[0], p1_start[0], places=3)
        self.assertAlmostEqual(p0_end[1], p1_start[1], places=3)

    def test_dof_reporting(self):
        sk = self.api.create_sketch("XY")
        self.api.add_line(sk, (0, 0), (10, 0))
        self.api.enable_constraints(sk)
        # 4 params (2 points) - 0 constraints = 4 DOF
        self.assertEqual(self.api.get_sketch_dof(sk), 4)
        ents = self.api.get_entity_ids(sk, 0)
        self.api.constrain_horizontal(sk, ents["line"])
        # 4 params - 1 constraint = 3 DOF
        self.assertEqual(self.api.get_sketch_dof(sk), 3)

    def test_constraint_persistence(self):
        """Constraints survive sketch to_dict / from_dict."""
        from sparkworks.kernel.sketch import Sketch
        sk = Sketch(name="Test", plane_name="XY")
        sk.add_line((0, 0), (10, 0))
        sk.ensure_solver()
        ents = sk.get_entity_ids_for_primitive(0)
        sk.solver.constrain_horizontal(ents["line"])
        d = sk.to_dict()
        sk2 = Sketch.from_dict(d)
        self.assertIsNotNone(sk2.solver)
        self.assertEqual(len(sk2.solver.constraints), 1)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def setUp(self):
        self.api = SparkWorksAPI(use_bridge=False)

    def test_extrude_without_sketch(self):
        self.api.extrude(distance=5)
        self.assertEqual(self.api.feature_count, 1)

    def test_delete_out_of_range(self):
        self.api.delete_feature(999)
        self.assertEqual(self.api.feature_count, 0)

    def test_scrub_empty_timeline(self):
        self.api.scrub_to_start()
        self.api.scrub_to_end()

    def test_clear_empty(self):
        self.api.clear_all()
        self.assertEqual(self.api.feature_count, 0)

    def test_get_body_faces_no_body(self):
        self.assertEqual(len(self.api.get_body_faces("NoBody")), 0)

    def test_counter_reset_after_clear(self):
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)
        self.api.clear_all()

        sk2 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk2, 10, 10)
        self.api.finish_sketch(sk2)
        body = self.api.extrude(distance=5)
        self.assertEqual(body, "Body1")


# =========================================================================
# Runner
# =========================================================================

if __name__ == "__main__":
    # Support: python run_standalone.py -v
    #          python run_standalone.py TestExtrude
    #          python run_standalone.py TestExtrude.test_basic_extrude
    unittest.main(verbosity=2)
