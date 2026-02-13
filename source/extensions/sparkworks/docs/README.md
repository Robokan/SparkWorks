# SparkWorks for Isaac Sim

A Fusion 360-style parametric CAD modeling environment that runs natively inside NVIDIA Isaac Sim.

## Overview

This extension provides sketch-based design, 3D feature operations, and a parametric timeline
directly within the simulation environment. Parts designed here become USD mesh prims on the
Isaac Sim stage, ready for immediate physics simulation.

## Features (Phase 1)

- **2D Sketching**: Lines, rectangles, circles on construction planes
- **3D Operations**: Extrude (single direction, symmetric, to-distance)
- **Parametric Timeline**: Feature history with rollback and parameter editing
- **USD Bridge**: Automatic tessellation and mesh prim creation
- **Live Viewport**: See your CAD design update in real-time

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Geometry Kernel | Open Cascade via build123d |
| Scene Integration | pxr USD API |
| UI Framework | omni.ui |
| Physics | Isaac Sim (native) |

## Quick Start

1. Enable the extension in the Extension Manager
2. Open the **SparkWorks** toolbar from the menu
3. Click **New Sketch** to start a sketch on the XY plane
4. Draw geometry using the sketch tools
5. Click **Extrude** to create a 3D solid
6. The result appears as a USD mesh prim in the viewport

## License

Open Cascade is LGPL-licensed. This extension uses it via Python bindings without modification.
