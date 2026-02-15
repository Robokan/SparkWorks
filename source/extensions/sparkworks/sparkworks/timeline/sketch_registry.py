"""
Sketch Registry — central store for all Sketch objects.

Like Fusion 360's Sketches folder in the browser tree, the registry owns
all Sketch objects by a stable string ID.  The parametric timeline stores
only *references* (sketch IDs) — it never embeds the Sketch data directly.

This separation means:
- Sketches can be browsed, selected, and edited independently of the timeline.
- Multiple operations can reference profiles from the same sketch.
- Reordering/inserting in the timeline doesn't move sketch data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..kernel.sketch import Sketch


class SketchRegistry:
    """
    Owns all Sketch objects in the current session.

    Each sketch has a unique string ID (e.g. ``"Sketch1"``).
    """

    def __init__(self):
        self._sketches: Dict[str, Sketch] = {}
        self._counter: int = 0

    # ── Queries ─────────────────────────────────────────────────────────────

    @property
    def sketch_ids(self) -> List[str]:
        return list(self._sketches.keys())

    @property
    def sketches(self) -> Dict[str, Sketch]:
        """Read-only view of all sketches."""
        return dict(self._sketches)

    @property
    def count(self) -> int:
        return len(self._sketches)

    @property
    def counter(self) -> int:
        """The next auto-increment number (useful for naming)."""
        return self._counter

    @counter.setter
    def counter(self, value: int):
        self._counter = value

    def get(self, sketch_id: str) -> Optional[Sketch]:
        return self._sketches.get(sketch_id)

    def __contains__(self, sketch_id: str) -> bool:
        return sketch_id in self._sketches

    # ── Mutations ───────────────────────────────────────────────────────────

    def register(self, sketch: Sketch, sketch_id: Optional[str] = None) -> str:
        """
        Add a sketch to the registry.

        Args:
            sketch: The Sketch object.
            sketch_id: Explicit ID.  If None, one is auto-generated.

        Returns:
            The assigned sketch ID.
        """
        if sketch_id is None:
            self._counter += 1
            sketch_id = f"Sketch{self._counter}"
        sketch.name = sketch_id
        self._sketches[sketch_id] = sketch
        # Keep counter in sync with any manually-set IDs
        try:
            num = int(sketch_id.replace("Sketch", ""))
            if num > self._counter:
                self._counter = num
        except (ValueError, AttributeError):
            pass
        return sketch_id

    def remove(self, sketch_id: str) -> Optional[Sketch]:
        """Remove and return a sketch by ID, or None if not found."""
        return self._sketches.pop(sketch_id, None)

    def clear(self):
        """Remove all sketches and reset the counter."""
        self._sketches.clear()
        self._counter = 0

    # ── Serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "counter": self._counter,
            "sketches": {
                sid: sketch.to_dict()
                for sid, sketch in self._sketches.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SketchRegistry":
        registry = cls()
        registry._counter = d.get("counter", 0)
        for sid, sdata in d.get("sketches", {}).items():
            sketch = Sketch.from_dict(sdata)
            registry._sketches[sid] = sketch
        return registry
