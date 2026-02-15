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
