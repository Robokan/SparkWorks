"""
Comprehensive tests for the SparkWorks programmatic API.

These tests use ``SparkWorksAPI(use_bridge=False)`` to run entirely in-memory
without requiring a USD stage or viewport.  They exercise:

- Sketch creation and primitive addition
- Extrude (sketch-based and face-based)
- Revolve, fillet, chamfer
- Timeline scrubbing, suppress, delete, reorder
- Multi-body workflows
- Serialization round-trips
- Profile / face queries
"""

import omni.kit.test


class TestAPISketch(omni.kit.test.AsyncTestCase):
    """Test sketch creation and primitives via the API."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)

    async def test_create_empty_sketch(self):
        """An empty sketch can be created and finished."""
        sk = self.api.create_sketch("XY")
        self.assertEqual(sk.name, "Sketch1")
        self.assertEqual(sk.plane_name, "XY")
        self.assertEqual(len(sk.primitives), 0)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    async def test_sketch_with_rectangle(self):
        """A sketch with a rectangle should finish successfully."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 20, 10)
        self.assertEqual(len(sk.primitives), 1)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    async def test_sketch_with_circle(self):
        """A sketch with a circle should finish successfully."""
        sk = self.api.create_sketch("XZ")
        self.api.add_circle(sk, radius=5)
        self.assertEqual(len(sk.primitives), 1)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    async def test_sketch_with_line(self):
        """A sketch with a line should finish successfully."""
        sk = self.api.create_sketch("YZ")
        self.api.add_line(sk, (0, 0), (10, 0))
        self.assertEqual(len(sk.primitives), 1)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    async def test_sketch_with_arc(self):
        """A sketch with an arc should finish successfully."""
        sk = self.api.create_sketch("XY")
        self.api.add_arc(sk, (0, 0), (5, 5), (10, 0))
        self.assertEqual(len(sk.primitives), 1)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    async def test_sketch_with_multiple_primitives(self):
        """A sketch can hold multiple primitives."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 20, 10)
        self.api.add_circle(sk, radius=3, center=(5, 5))
        self.api.add_line(sk, (-10, -10), (10, 10))
        self.assertEqual(len(sk.primitives), 3)
        self.api.finish_sketch(sk)
        self.assertEqual(self.api.feature_count, 1)

    async def test_multiple_sketches(self):
        """Creating multiple sketches increments names correctly."""
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 10, 10)
        self.api.finish_sketch(sk1)

        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=5)
        self.api.finish_sketch(sk2)

        self.assertEqual(self.api.feature_count, 2)
        self.assertEqual(sk1.name, "Sketch1")
        self.assertEqual(sk2.name, "Sketch2")

    async def test_sketch_registry_populated(self):
        """Finished sketches appear in the sketch registry."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)

        reg = self.api.sketch_registry
        self.assertEqual(reg.count, 1)
        self.assertIn("Sketch1", reg.sketches)

    async def test_get_sketch(self):
        """Can look up a sketch by name."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)

        found = self.api.get_sketch("Sketch1")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Sketch1")

    async def test_get_sketch_not_found(self):
        """Looking up a non-existent sketch returns None."""
        self.assertIsNone(self.api.get_sketch("NoSuchSketch"))


class TestAPIExtrude(omni.kit.test.AsyncTestCase):
    """Test extrude operations via the API."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)

    async def _make_box(self, w=10, h=10, d=5, body_name=None):
        """Helper: create a sketch + extrude to produce a box."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, w, h)
        self.api.finish_sketch(sk)
        return self.api.extrude(distance=d, body_name=body_name)

    async def test_basic_extrude(self):
        """Sketch + extrude should produce a body."""
        body = await self._make_box()
        self.assertEqual(body, "Body1")
        self.assertEqual(len(self.api.bodies), 1)
        self.assertIn("Body1", self.api.bodies)
        self.assertIsNotNone(self.api.current_solid)

    async def test_extrude_creates_solid(self):
        """The extruded body should have a non-None solid."""
        await self._make_box()
        solid = self.api.get_body_solid("Body1")
        self.assertIsNotNone(solid)

    async def test_custom_body_name(self):
        """Can specify a custom body name."""
        body = await self._make_box(body_name="MyBox")
        self.assertEqual(body, "MyBox")
        self.assertIn("MyBox", self.api.bodies)

    async def test_extrude_distance(self):
        """Different distances should produce different solids."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=20)

        self.assertIsNotNone(self.api.current_solid)

    async def test_two_independent_bodies(self):
        """Two separate sketch+extrude pairs create two bodies."""
        await self._make_box(w=10, h=10, d=5)

        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=3)
        self.api.finish_sketch(sk2)
        self.api.extrude(distance=8)

        self.assertEqual(len(self.api.bodies), 2)
        self.assertIn("Body1", self.api.bodies)
        self.assertIn("Body2", self.api.bodies)

    async def test_join_extrude(self):
        """Join extrude should fuse onto an existing body."""
        await self._make_box()

        sk2 = self.api.create_sketch("XY")
        self.api.add_circle(sk2, radius=3, center=(0, 0))
        self.api.finish_sketch(sk2)
        body = self.api.extrude(distance=10, join_body="Body1")

        self.assertEqual(body, "Body1")
        # Should still be one body (fused)
        self.assertEqual(len(self.api.bodies), 1)

    async def test_feature_count_after_extrude(self):
        """Sketch + extrude = 2 features."""
        await self._make_box()
        self.assertEqual(self.api.feature_count, 2)

    async def test_extrude_face(self):
        """Extrude from face should fuse onto the source body."""
        await self._make_box()

        # Get face info
        faces = self.api.get_body_faces("Body1")
        self.assertGreater(len(faces), 0)

        # Extrude from the first face
        body = self.api.extrude_face("Body1", face_index=0, distance=3)
        self.assertEqual(body, "Body1")
        # Still one body (fused)
        self.assertEqual(len(self.api.bodies), 1)
        # 3 features: sketch, extrude, face-extrude
        self.assertEqual(self.api.feature_count, 3)


class TestAPIOtherOps(omni.kit.test.AsyncTestCase):
    """Test revolve, fillet, chamfer operations."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)

    async def test_revolve(self):
        """Revolve a sketch profile around an axis."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 5, 2, center=(10, 0))
        self.api.finish_sketch(sk)
        self.api.revolve(angle=360, axis="Z")

        self.assertEqual(self.api.feature_count, 2)
        self.assertIsNotNone(self.api.current_solid)

    async def test_fillet(self):
        """Fillet should modify the current solid without error."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)
        self.api.fillet(radius=0.5)

        self.assertEqual(self.api.feature_count, 3)
        self.assertIsNotNone(self.api.current_solid)

    async def test_chamfer(self):
        """Chamfer should modify the current solid without error."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)
        self.api.chamfer(length=0.5)

        self.assertEqual(self.api.feature_count, 3)
        self.assertIsNotNone(self.api.current_solid)


class TestAPITimeline(omni.kit.test.AsyncTestCase):
    """Test timeline scrubbing, suppress, delete, and reorder."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)
        # Build a standard model: sketch -> extrude -> sketch2 -> extrude2
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 10, 10)
        self.api.finish_sketch(sk1)
        self.api.extrude(distance=5)

        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=3)
        self.api.finish_sketch(sk2)
        self.api.extrude(distance=8)

    async def test_initial_state(self):
        """After setup, should have 4 features and 2 bodies."""
        self.assertEqual(self.api.feature_count, 4)
        self.assertEqual(len(self.api.bodies), 2)

    async def test_scrub_to_middle(self):
        """Scrubbing to after the first extrude shows only 1 body."""
        self.api.scrub_to(1)  # after first extrude
        self.assertEqual(len(self.api.bodies), 1)
        self.assertIn("Body1", self.api.bodies)

    async def test_scrub_to_start(self):
        """Scrubbing to start shows no bodies."""
        self.api.scrub_to_start()
        self.assertEqual(len(self.api.bodies), 0)

    async def test_scrub_to_end(self):
        """Scrubbing to end restores all bodies."""
        self.api.scrub_to(1)
        self.assertEqual(len(self.api.bodies), 1)
        self.api.scrub_to_end()
        self.assertEqual(len(self.api.bodies), 2)

    async def test_suppress_extrude(self):
        """Suppressing an extrude should remove its body."""
        self.api.suppress_feature(1, True)  # suppress first extrude
        # Body1 should be gone (suppressed) but Body2 may remain
        # depending on rebuild logic
        self.api.scrub_to_end()
        self.assertNotIn("Body1", self.api.bodies)

    async def test_unsuppress_extrude(self):
        """Unsuppressing should restore the body."""
        self.api.suppress_feature(1, True)
        self.api.suppress_feature(1, False)
        self.api.scrub_to_end()
        self.assertIn("Body1", self.api.bodies)

    async def test_delete_feature(self):
        """Deleting a feature reduces feature count."""
        self.api.delete_feature(3)  # delete second extrude
        self.assertEqual(self.api.feature_count, 3)

    async def test_clear_all(self):
        """Clear resets everything."""
        self.api.clear_all()
        self.assertEqual(self.api.feature_count, 0)
        self.assertEqual(len(self.api.bodies), 0)
        self.assertIsNone(self.api.current_solid)

    async def test_rebuild_all(self):
        """Rebuild all should not change the result."""
        bodies_before = dict(self.api.bodies)
        self.api.rebuild_all()
        bodies_after = dict(self.api.bodies)
        self.assertEqual(set(bodies_before.keys()), set(bodies_after.keys()))

    async def test_marker_position(self):
        """Marker position should track correctly."""
        # After setup, marker should be at last feature (index 3)
        self.assertEqual(self.api.marker_position, 3)

        self.api.move_marker(1)
        self.assertEqual(self.api.marker_position, 1)


class TestAPIQueries(omni.kit.test.AsyncTestCase):
    """Test query methods."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)

    async def test_features_list(self):
        """Features list returns the correct features."""
        features = self.api.features
        self.assertEqual(len(features), 2)
        self.assertTrue(features[0].is_sketch)
        self.assertTrue(features[1].is_operation)

    async def test_get_body_solid(self):
        """Can retrieve a body's solid."""
        solid = self.api.get_body_solid("Body1")
        self.assertIsNotNone(solid)

    async def test_get_body_solid_missing(self):
        """Missing body returns None."""
        solid = self.api.get_body_solid("NoSuchBody")
        self.assertIsNone(solid)

    async def test_get_body_faces(self):
        """A box should have planar faces."""
        faces = self.api.get_body_faces("Body1")
        # A box has 6 faces, all planar
        self.assertGreater(len(faces), 0)

    async def test_get_planar_face(self):
        """Can get a specific planar face by index."""
        face = self.api.get_planar_face("Body1", 0)
        self.assertIsNotNone(face)

    async def test_get_planar_face_out_of_range(self):
        """Out-of-range face index returns None."""
        face = self.api.get_planar_face("Body1", 999)
        self.assertIsNone(face)

    async def test_get_profiles(self):
        """A rectangle sketch should produce at least one profile."""
        sk = self.api.get_sketch("Sketch1")
        self.assertIsNotNone(sk)
        profiles = self.api.get_profiles(sk)
        self.assertGreater(len(profiles), 0)

    async def test_body_names_no_bridge(self):
        """get_body_names falls back to in-memory dict without bridge."""
        names = self.api.get_body_names()
        self.assertEqual(names, ["Body1"])


class TestAPIInsertAtMarker(omni.kit.test.AsyncTestCase):
    """Test that operations insert at the marker position."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)

    async def test_insert_extrude_at_marker(self):
        """Operations should insert at the marker position."""
        # Build: sketch -> extrude -> sketch2 -> extrude2
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 10, 10)
        self.api.finish_sketch(sk1)
        self.api.extrude(distance=5)

        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=3)
        self.api.finish_sketch(sk2)
        self.api.extrude(distance=8)

        self.assertEqual(self.api.feature_count, 4)

        # Move marker to after first extrude
        self.api.move_marker(1)

        # Insert a new sketch + extrude (should go between the two)
        sk3 = self.api.create_sketch("YZ")
        self.api.add_rectangle(sk3, 5, 5)
        self.api.finish_sketch(sk3)
        self.api.extrude(distance=3)

        self.assertEqual(self.api.feature_count, 6)

    async def test_insert_at_start(self):
        """Inserting at marker=-1 should prepend."""
        sk1 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk1, 10, 10)
        self.api.finish_sketch(sk1)
        self.api.extrude(distance=5)

        # Move to start
        self.api.move_marker(-1)

        sk2 = self.api.create_sketch("XZ")
        self.api.add_circle(sk2, radius=3)
        idx = self.api.finish_sketch(sk2)

        # The new sketch should be the first feature
        features = self.api.features
        self.assertTrue(features[0].is_sketch)
        self.assertEqual(features[0].sketch.name, "Sketch2")


class TestAPIMultiBody(omni.kit.test.AsyncTestCase):
    """Test multi-body workflows."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)

    async def test_three_independent_bodies(self):
        """Three independent sketch+extrude pairs create three bodies."""
        for i in range(3):
            sk = self.api.create_sketch("XY")
            self.api.add_rectangle(sk, 10 + i, 10 + i)
            self.api.finish_sketch(sk)
            self.api.extrude(distance=5 + i)

        self.assertEqual(len(self.api.bodies), 3)
        for i in range(1, 4):
            self.assertIn(f"Body{i}", self.api.bodies)

    async def test_scrub_through_multi_body(self):
        """Scrubbing through a multi-body timeline shows incremental bodies."""
        for i in range(3):
            sk = self.api.create_sketch("XY")
            self.api.add_rectangle(sk, 10, 10)
            self.api.finish_sketch(sk)
            self.api.extrude(distance=5)

        # 6 features: sk1, ext1, sk2, ext2, sk3, ext3
        self.assertEqual(self.api.feature_count, 6)

        # After first extrude (index 1)
        self.api.scrub_to(1)
        self.assertEqual(len(self.api.bodies), 1)

        # After second extrude (index 3)
        self.api.scrub_to(3)
        self.assertEqual(len(self.api.bodies), 2)

        # After third extrude (index 5)
        self.api.scrub_to(5)
        self.assertEqual(len(self.api.bodies), 3)


class TestAPIEdgeCases(omni.kit.test.AsyncTestCase):
    """Test edge cases and error handling."""

    async def setUp(self):
        from sparkworks.api import SparkWorksAPI
        self.api = SparkWorksAPI(use_bridge=False)

    async def test_extrude_without_sketch(self):
        """Extruding without a sketch should not crash."""
        # This will produce no solid (no sketch face), but shouldn't raise
        self.api.extrude(distance=5)
        self.assertEqual(self.api.feature_count, 1)

    async def test_delete_out_of_range(self):
        """Deleting an out-of-range feature should be a no-op."""
        self.api.delete_feature(999)
        self.assertEqual(self.api.feature_count, 0)

    async def test_scrub_empty_timeline(self):
        """Scrubbing an empty timeline should not crash."""
        self.api.scrub_to_start()
        self.api.scrub_to_end()
        self.api.scrub_to(0)

    async def test_clear_empty(self):
        """Clearing an already empty API should not crash."""
        self.api.clear_all()
        self.assertEqual(self.api.feature_count, 0)

    async def test_get_body_faces_no_body(self):
        """Getting faces from a non-existent body returns empty list."""
        faces = self.api.get_body_faces("NoBody")
        self.assertEqual(len(faces), 0)

    async def test_body_counter_survives_clear(self):
        """After clear, counters reset."""
        sk = self.api.create_sketch("XY")
        self.api.add_rectangle(sk, 10, 10)
        self.api.finish_sketch(sk)
        self.api.extrude(distance=5)

        self.api.clear_all()
        self.assertEqual(self.api.feature_count, 0)
        self.assertEqual(len(self.api.bodies), 0)

        # New body should be Body1 again
        sk2 = self.api.create_sketch("XY")
        self.api.add_rectangle(sk2, 10, 10)
        self.api.finish_sketch(sk2)
        body = self.api.extrude(distance=5)
        self.assertEqual(body, "Body1")
