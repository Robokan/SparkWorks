"""
OCP compatibility patches for cadquery-ocp >= 7.8.

cadquery-ocp 7.9+ removed ``TopoDS_Shape.HashCode()`` which build123d 0.8
relies on internally. This module patches the missing method so that
build123d can work with the newer OCP versions shipping on aarch64.

**Must be imported before any ``build123d`` import.**
"""
from __future__ import annotations

import importlib
import sys


def apply_ocp_patches():
    """
    Add a ``HashCode`` shim to ``OCP.TopoDS.TopoDS_Shape`` if it is missing.

    The shim delegates to Python's built-in ``hash()`` which OCP 7.9+ supports
    on shape objects.  The ``upper`` parameter that the original OCCT method
    accepted is ignored — build123d only uses it for dict/set operations where
    the absolute hash value doesn't matter.
    """
    try:
        # Force-import the low-level OCP module
        import OCP.TopoDS  # type: ignore[import-untyped]

        shape_cls = OCP.TopoDS.TopoDS_Shape
        if hasattr(shape_cls, "HashCode"):
            # Nothing to patch
            return

        def _hashcode_shim(self, upper: int = 2**31 - 1) -> int:  # noqa: ARG001
            """Compatibility shim: delegates to ``hash()``."""
            return hash(self) % (upper + 1)

        shape_cls.HashCode = _hashcode_shim
        print("[SparkWorks] Patched OCP.TopoDS.TopoDS_Shape.HashCode for OCP >= 7.8 compat")

    except ImportError:
        # OCP not available yet — will be imported later
        print("[SparkWorks] OCP module not found, skipping HashCode patch")
    except Exception as exc:
        print(f"[SparkWorks] Warning: OCP HashCode patch failed: {exc}")
