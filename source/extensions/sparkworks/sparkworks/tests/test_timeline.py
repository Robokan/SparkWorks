"""
Tests for the parametric timeline.
"""

import omni.kit.test


class TestTimeline(omni.kit.test.AsyncTestCase):
    """Test the Timeline class."""

    async def setUp(self):
        from sparkworks.timeline import Timeline
        self.timeline = Timeline()

    async def test_empty_timeline(self):
        """A new timeline should have no features."""
        self.assertEqual(self.timeline.feature_count, 0)
        self.assertEqual(len(self.timeline.bodies), 0)

    async def test_add_sketch(self):
        """Adding a sketch should create a feature."""
        from sparkworks.kernel import Sketch

        sketch = Sketch(name="TestSketch", plane_name="XY")
        sketch.add_rectangle(width=10, height=5)
        self.timeline.add_sketch(sketch)

        self.assertEqual(self.timeline.feature_count, 1)
        self.assertTrue(self.timeline.features[0].is_sketch)

    async def test_add_extrude(self):
        """Adding a sketch + extrude should produce a body."""
        from sparkworks.kernel import Sketch
        from sparkworks.kernel.operations import ExtrudeOperation

        sketch = Sketch(name="TestSketch", plane_name="XY")
        sketch.add_rectangle(width=10, height=5)
        self.timeline.add_sketch(sketch)

        op = ExtrudeOperation(distance=15.0, body_name="Body1")
        self.timeline.add_operation(op)

        self.assertEqual(self.timeline.feature_count, 2)
        self.assertIn("Body1", self.timeline.bodies)
        self.assertIsNotNone(self.timeline.bodies["Body1"])

    async def test_suppress_feature(self):
        """Suppressing a feature should exclude it from the rebuild."""
        from sparkworks.kernel import Sketch
        from sparkworks.kernel.operations import ExtrudeOperation

        sketch = Sketch(name="TestSketch", plane_name="XY")
        sketch.add_rectangle(width=10, height=5)
        self.timeline.add_sketch(sketch)

        op = ExtrudeOperation(distance=15.0, body_name="Body1")
        self.timeline.add_operation(op)

        # Suppress the extrude — body should disappear
        self.timeline.suppress_feature(1, True)
        self.assertNotIn("Body1", self.timeline.bodies)

        # Unsuppress — body should reappear
        self.timeline.suppress_feature(1, False)
        self.assertIn("Body1", self.timeline.bodies)

    async def test_serialization(self):
        """Timeline should round-trip through JSON."""
        from sparkworks.kernel import Sketch
        from sparkworks.kernel.operations import ExtrudeOperation
        from sparkworks.timeline import Timeline

        sketch = Sketch(name="TestSketch", plane_name="XY")
        sketch.add_rectangle(width=10, height=5)
        self.timeline.add_sketch(sketch)

        op = ExtrudeOperation(distance=15.0)
        self.timeline.add_operation(op)

        # Serialize
        json_str = self.timeline.to_json()

        # Deserialize
        restored = Timeline.from_json(json_str)
        self.assertEqual(restored.feature_count, 2)
