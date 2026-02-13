# SparkWorks — Overview

## Architecture

The extension follows a clean layered architecture:

```
┌─────────────────────────────────────────────┐
│                  UI Layer                    │
│  Toolbar │ Timeline Panel │ Property Editor  │
├─────────────────────────────────────────────┤
│              Extension Core                  │
│         (extension.py — orchestrator)        │
├──────────┬──────────────┬───────────────────┤
│  Kernel  │   Timeline   │    USD Bridge     │
│ Sketch   │ Feature list │ Tessellate → Mesh │
│ Ops      │ Rebuild      │ Write USD prims   │
│ Tessellate│ Rollback    │                   │
├──────────┴──────────────┴───────────────────┤
│          Open Cascade (build123d)            │
│         B-Rep Solid Modeling Engine          │
└─────────────────────────────────────────────┘
```

## Data Flow

1. User draws a 2D sketch on a construction plane
2. User applies a 3D operation (extrude, revolve, fillet, etc.)
3. The operation is appended to the parametric timeline
4. Open Cascade computes the resulting 3D solid geometry
5. The solid is tessellated into a triangle mesh
6. The mesh is written as a USD prim, visible immediately in the viewport

## Key Modules

- **`kernel/`** — Geometry operations powered by build123d/Open Cascade
- **`timeline/`** — Parametric feature history with rebuild logic
- **`bridge/`** — Converts OCC solids to USD mesh prims
- **`ui/`** — omni.ui panels, toolbar, and viewport interaction
