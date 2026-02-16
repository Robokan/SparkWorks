"""
2D Geometric Constraint Solver — pure-Python, scipy-based.

Inspired by FreeCAD's PlaneGCS architecture but built from scratch using
``scipy.optimize.least_squares`` so we have zero native dependencies beyond
what's already in the Isaac Sim container.

Architecture
------------
* Each geometric entity (point, line, circle, arc) stores its parameters
  as indices into a flat parameter vector ``params[]``.
* Each constraint produces one or more *linear* residuals — zero means
  satisfied.
* ``solve()`` minimises the sum of squared residuals using
  ``least_squares`` (Trust Region Reflective), which is dramatically
  faster than the previous L-BFGS-B approach because it works with the
  full residual vector and computes the Jacobian efficiently.

Supported constraints
---------------------
Coincident, Horizontal, Vertical, Perpendicular, Parallel, Tangent,
PointOnLine, Distance (point-point, point-line), Angle, EqualLength,
Radius, FixedPoint, HorizontalLine, VerticalLine, Symmetric, Midpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


# =========================================================================
# Entity identifiers
# =========================================================================

class EntityKind(Enum):
    POINT = auto()
    LINE = auto()
    CIRCLE = auto()
    ARC = auto()


@dataclass
class GeoEntity:
    """Reference to a geometric entity inside the solver."""
    eid: int
    kind: EntityKind
    param_indices: List[int] = field(default_factory=list)
    # Metadata — maps back to the SketchPrimitive index in the Sketch
    prim_index: int = -1


# =========================================================================
# Constraint definitions
# =========================================================================

class ConstraintType(Enum):
    COINCIDENT = auto()
    HORIZONTAL = auto()
    VERTICAL = auto()
    PERPENDICULAR = auto()
    PARALLEL = auto()
    TANGENT_LC = auto()       # line tangent to circle
    TANGENT_CC = auto()       # circle tangent to circle
    POINT_ON_LINE = auto()
    DISTANCE_PP = auto()      # point-point
    DISTANCE_PL = auto()      # point-line (signed)
    ANGLE_LL = auto()         # angle between two lines
    EQUAL_LENGTH = auto()
    RADIUS = auto()
    FIXED = auto()
    HORIZONTAL_LINE = auto()
    VERTICAL_LINE = auto()
    SYMMETRIC = auto()
    MIDPOINT = auto()


@dataclass
class Constraint:
    """A single geometric constraint with its residual function."""
    cid: int
    ctype: ConstraintType
    entity_ids: List[int] = field(default_factory=list)
    value: float = 0.0        # e.g. target distance / angle / radius
    # Sub-element selectors ("start", "end", "center", "" = whole)
    selectors: List[str] = field(default_factory=list)
    # Back-reference for serialisation
    driving: bool = True

    def to_dict(self) -> dict:
        return {
            "cid": self.cid,
            "ctype": self.ctype.name,
            "entity_ids": self.entity_ids,
            "value": self.value,
            "selectors": self.selectors,
            "driving": self.driving,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Constraint":
        return cls(
            cid=d["cid"],
            ctype=ConstraintType[d["ctype"]],
            entity_ids=d["entity_ids"],
            value=d.get("value", 0.0),
            selectors=d.get("selectors", []),
            driving=d.get("driving", True),
        )


# =========================================================================
# Solver
# =========================================================================

class ConstraintSolver:
    """
    2D geometric constraint solver.

    Usage::

        solver = ConstraintSolver()
        p1 = solver.add_point(0, 0)
        p2 = solver.add_point(5, 0)
        l1 = solver.add_line(p1, p2)
        solver.add_constraint(ConstraintType.HORIZONTAL_LINE, [l1])
        solver.add_constraint(ConstraintType.DISTANCE_PP, [p1, p2], value=10)
        ok = solver.solve()
        x1, y1 = solver.get_point(p1)
        x2, y2 = solver.get_point(p2)
    """

    def __init__(self):
        self._params: List[float] = []
        self._entities: Dict[int, GeoEntity] = {}
        self._constraints: List[Constraint] = []
        self._fixed: set = set()          # param indices that are locked
        self._next_eid: int = 0
        self._next_cid: int = 0

    # -- State queries -------------------------------------------------------

    @property
    def params(self) -> np.ndarray:
        return np.array(self._params, dtype=np.float64)

    @property
    def entities(self) -> Dict[int, GeoEntity]:
        return dict(self._entities)

    @property
    def constraints(self) -> List[Constraint]:
        return list(self._constraints)

    @property
    def dof(self) -> int:
        """Approximate remaining degrees of freedom."""
        free_params = len(self._params) - len(self._fixed)
        n_constraints = len(self._constraints)
        return max(0, free_params - n_constraints)

    # -- Entity creation ------------------------------------------------------

    def _alloc(self, n: int) -> List[int]:
        """Allocate *n* parameter slots and return their indices."""
        start = len(self._params)
        self._params.extend([0.0] * n)
        return list(range(start, start + n))

    def add_point(self, x: float = 0.0, y: float = 0.0, prim_index: int = -1) -> int:
        """Add a point entity. Returns the entity id."""
        eid = self._next_eid; self._next_eid += 1
        indices = self._alloc(2)
        self._params[indices[0]] = x
        self._params[indices[1]] = y
        self._entities[eid] = GeoEntity(eid, EntityKind.POINT, indices, prim_index)
        return eid

    def add_line(self, p1_eid: int, p2_eid: int, prim_index: int = -1) -> int:
        """
        Add a line entity referencing two existing point entities.

        The line's param_indices are the concatenation of p1's and p2's
        indices: [p1.x, p1.y, p2.x, p2.y].
        """
        eid = self._next_eid; self._next_eid += 1
        e1 = self._entities[p1_eid]
        e2 = self._entities[p2_eid]
        indices = e1.param_indices + e2.param_indices
        self._entities[eid] = GeoEntity(eid, EntityKind.LINE, indices, prim_index)
        return eid

    def add_line_from_coords(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
        prim_index: int = -1,
    ) -> Tuple[int, int, int]:
        """
        Convenience: add two points + a line in one call.

        Returns ``(line_eid, p1_eid, p2_eid)``.
        """
        p1 = self.add_point(x1, y1, prim_index)
        p2 = self.add_point(x2, y2, prim_index)
        lid = self.add_line(p1, p2, prim_index)
        return lid, p1, p2

    def add_circle(
        self,
        cx: float = 0.0, cy: float = 0.0,
        r: float = 1.0,
        prim_index: int = -1,
    ) -> Tuple[int, int]:
        """
        Add a circle entity: centre point + radius parameter.

        Returns ``(circle_eid, center_eid)``.
        """
        center_eid = self.add_point(cx, cy, prim_index)
        eid = self._next_eid; self._next_eid += 1
        center_e = self._entities[center_eid]
        # Allocate one extra param for the radius
        r_idx = self._alloc(1)
        self._params[r_idx[0]] = r
        indices = center_e.param_indices + r_idx
        self._entities[eid] = GeoEntity(eid, EntityKind.CIRCLE, indices, prim_index)
        return eid, center_eid

    def add_arc(
        self,
        cx: float, cy: float,
        r: float,
        start_angle: float, end_angle: float,
        prim_index: int = -1,
    ) -> Tuple[int, int]:
        """
        Add an arc entity: centre, radius, start_angle, end_angle.

        Returns ``(arc_eid, center_eid)``.
        """
        center_eid = self.add_point(cx, cy, prim_index)
        eid = self._next_eid; self._next_eid += 1
        center_e = self._entities[center_eid]
        extra = self._alloc(3)  # radius, start_angle, end_angle
        self._params[extra[0]] = r
        self._params[extra[1]] = start_angle
        self._params[extra[2]] = end_angle
        indices = center_e.param_indices + extra
        self._entities[eid] = GeoEntity(eid, EntityKind.ARC, indices, prim_index)
        return eid, center_eid

    # -- Parameter access helpers --------------------------------------------

    def get_point(self, eid: int) -> Tuple[float, float]:
        """Get the current (x, y) of a point entity."""
        e = self._entities[eid]
        return (self._params[e.param_indices[0]], self._params[e.param_indices[1]])

    def set_point(self, eid: int, x: float, y: float):
        """Set the (x, y) of a point entity."""
        e = self._entities[eid]
        self._params[e.param_indices[0]] = x
        self._params[e.param_indices[1]] = y

    def get_line_points(self, eid: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get (p1, p2) of a line entity."""
        e = self._entities[eid]
        p = self._params
        return ((p[e.param_indices[0]], p[e.param_indices[1]]),
                (p[e.param_indices[2]], p[e.param_indices[3]]))

    def get_circle(self, eid: int) -> Tuple[float, float, float]:
        """Get (cx, cy, radius) of a circle entity."""
        e = self._entities[eid]
        p = self._params
        return (p[e.param_indices[0]], p[e.param_indices[1]], p[e.param_indices[2]])

    def get_arc(self, eid: int) -> Tuple[float, float, float, float, float]:
        """Get (cx, cy, r, start_angle, end_angle) of an arc entity."""
        e = self._entities[eid]
        p = self._params
        return (
            p[e.param_indices[0]], p[e.param_indices[1]],
            p[e.param_indices[2]], p[e.param_indices[3]], p[e.param_indices[4]],
        )

    # -- Fix / unfix parameters -----------------------------------------------

    def fix_point(self, eid: int):
        """Lock a point so the solver won't move it."""
        e = self._entities[eid]
        for idx in e.param_indices:
            self._fixed.add(idx)

    def unfix_point(self, eid: int):
        """Unlock a point so the solver can move it."""
        e = self._entities[eid]
        for idx in e.param_indices:
            self._fixed.discard(idx)

    def fix_param(self, param_index: int):
        self._fixed.add(param_index)

    def unfix_param(self, param_index: int):
        self._fixed.discard(param_index)

    # -- Constraint creation --------------------------------------------------

    def add_constraint(
        self,
        ctype: ConstraintType,
        entity_ids: List[int],
        value: float = 0.0,
        selectors: Optional[List[str]] = None,
        driving: bool = True,
    ) -> int:
        """
        Add a constraint.

        Args:
            ctype: Constraint type.
            entity_ids: Entity ids referenced by this constraint.
            value: Target value (distance, angle, radius, etc.).
            selectors: Sub-element selectors per entity
                       (e.g. ``["start", "end"]`` for coincident on endpoints).
            driving: ``True`` for driving constraints, ``False`` for reference.

        Returns:
            Constraint id.
        """
        cid = self._next_cid; self._next_cid += 1
        c = Constraint(
            cid=cid,
            ctype=ctype,
            entity_ids=list(entity_ids),
            value=value,
            selectors=selectors or [],
            driving=driving,
        )
        self._constraints.append(c)
        return cid

    def remove_constraint(self, cid: int):
        """Remove a constraint by its id."""
        self._constraints = [c for c in self._constraints if c.cid != cid]

    def clear_constraints(self):
        """Remove all constraints."""
        self._constraints.clear()

    # -- Convenience constraint helpers ----------------------------------------

    def constrain_coincident(self, eid_a: int, eid_b: int,
                             sel_a: str = "", sel_b: str = "") -> int:
        """Two points / sub-elements coincide."""
        return self.add_constraint(
            ConstraintType.COINCIDENT, [eid_a, eid_b],
            selectors=[sel_a, sel_b],
        )

    def constrain_horizontal(self, line_eid: int) -> int:
        """Line is horizontal."""
        return self.add_constraint(ConstraintType.HORIZONTAL_LINE, [line_eid])

    def constrain_vertical(self, line_eid: int) -> int:
        """Line is vertical."""
        return self.add_constraint(ConstraintType.VERTICAL_LINE, [line_eid])

    def constrain_distance_pp(self, eid_a: int, eid_b: int,
                               dist: float,
                               sel_a: str = "", sel_b: str = "") -> int:
        """Distance between two points / sub-elements."""
        return self.add_constraint(
            ConstraintType.DISTANCE_PP, [eid_a, eid_b],
            value=dist, selectors=[sel_a, sel_b],
        )

    def constrain_distance_pl(self, point_eid: int, line_eid: int,
                               dist: float) -> int:
        """Distance from a point to a line."""
        return self.add_constraint(
            ConstraintType.DISTANCE_PL, [point_eid, line_eid], value=dist,
        )

    def constrain_angle(self, line_a: int, line_b: int, angle_deg: float) -> int:
        """Angle between two lines in degrees."""
        return self.add_constraint(
            ConstraintType.ANGLE_LL, [line_a, line_b], value=angle_deg,
        )

    def constrain_perpendicular(self, line_a: int, line_b: int) -> int:
        return self.add_constraint(ConstraintType.PERPENDICULAR, [line_a, line_b])

    def constrain_parallel(self, line_a: int, line_b: int) -> int:
        return self.add_constraint(ConstraintType.PARALLEL, [line_a, line_b])

    def constrain_equal_length(self, line_a: int, line_b: int) -> int:
        return self.add_constraint(ConstraintType.EQUAL_LENGTH, [line_a, line_b])

    def constrain_radius(self, circle_eid: int, radius: float) -> int:
        return self.add_constraint(ConstraintType.RADIUS, [circle_eid], value=radius)

    def constrain_tangent_lc(self, line_eid: int, circle_eid: int) -> int:
        """Line tangent to circle."""
        return self.add_constraint(ConstraintType.TANGENT_LC, [line_eid, circle_eid])

    def constrain_tangent_cc(self, circle_a: int, circle_b: int) -> int:
        """Circle tangent to circle."""
        return self.add_constraint(ConstraintType.TANGENT_CC, [circle_a, circle_b])

    def constrain_point_on_line(self, point_eid: int, line_eid: int) -> int:
        return self.add_constraint(ConstraintType.POINT_ON_LINE, [point_eid, line_eid])

    def constrain_fixed(self, eid: int, x: float, y: float) -> int:
        """Pin a point to specific coordinates."""
        return self.add_constraint(ConstraintType.FIXED, [eid], value=0.0,
                                   selectors=[str(x), str(y)])

    def constrain_symmetric(self, eid_a: int, eid_b: int, line_eid: int) -> int:
        """Two points symmetric about a line."""
        return self.add_constraint(ConstraintType.SYMMETRIC, [eid_a, eid_b, line_eid])

    def constrain_midpoint(self, point_eid: int, line_eid: int) -> int:
        """Point at the midpoint of a line."""
        return self.add_constraint(ConstraintType.MIDPOINT, [point_eid, line_eid])

    # -- Sub-element resolution -----------------------------------------------

    def _resolve_point(self, eid: int, selector: str = "") -> Tuple[int, int]:
        """
        Resolve a (entity, selector) pair to the two param indices for a 2D point.

        Selectors:
        - ``""``, ``"center"`` → first two params (point / circle centre)
        - ``"start"`` → line params [0:2]
        - ``"end"``   → line params [2:4]
        """
        e = self._entities[eid]
        pi = e.param_indices

        if e.kind == EntityKind.POINT:
            return (pi[0], pi[1])

        if e.kind == EntityKind.LINE:
            if selector in ("end",):
                return (pi[2], pi[3])
            return (pi[0], pi[1])  # start

        if e.kind in (EntityKind.CIRCLE, EntityKind.ARC):
            return (pi[0], pi[1])  # center

        return (pi[0], pi[1])

    # -- Residual evaluation ---------------------------------------------------

    def _build_residual_vector(self, p: np.ndarray) -> np.ndarray:
        """
        Build the full residual vector for ``least_squares``.

        Each constraint contributes one or more *linear* residuals (not
        squared).  ``least_squares`` internally minimises the sum of their
        squares, so the mathematical problem is identical to before but
        convergence is dramatically faster.
        """
        residuals: List[float] = []
        for c in self._constraints:
            if not c.driving:
                continue
            residuals.extend(self._eval_residuals(c, p))
        return np.array(residuals, dtype=np.float64) if residuals else np.array([0.0])

    def _eval_residuals(self, c: Constraint, p: np.ndarray) -> List[float]:
        """
        Evaluate a single constraint and return a list of *linear*
        residual components.  Each component is zero when the constraint
        is satisfied.
        """
        eids = c.entity_ids
        sels = c.selectors + [""] * (len(eids) - len(c.selectors))

        if c.ctype == ConstraintType.COINCIDENT:
            ix_a, iy_a = self._resolve_point(eids[0], sels[0])
            ix_b, iy_b = self._resolve_point(eids[1], sels[1])
            return [p[ix_a] - p[ix_b], p[iy_a] - p[iy_b]]

        if c.ctype == ConstraintType.HORIZONTAL_LINE:
            e = self._entities[eids[0]]
            pi = e.param_indices
            return [p[pi[1]] - p[pi[3]]]

        if c.ctype == ConstraintType.VERTICAL_LINE:
            e = self._entities[eids[0]]
            pi = e.param_indices
            return [p[pi[0]] - p[pi[2]]]

        if c.ctype == ConstraintType.HORIZONTAL:
            ix_a, iy_a = self._resolve_point(eids[0], sels[0])
            ix_b, iy_b = self._resolve_point(eids[1], sels[1])
            return [p[iy_a] - p[iy_b]]

        if c.ctype == ConstraintType.VERTICAL:
            ix_a, iy_a = self._resolve_point(eids[0], sels[0])
            ix_b, iy_b = self._resolve_point(eids[1], sels[1])
            return [p[ix_a] - p[ix_b]]

        if c.ctype == ConstraintType.DISTANCE_PP:
            ix_a, iy_a = self._resolve_point(eids[0], sels[0])
            ix_b, iy_b = self._resolve_point(eids[1], sels[1])
            dx = p[ix_a] - p[ix_b]
            dy = p[iy_a] - p[iy_b]
            dist = math.sqrt(dx * dx + dy * dy)
            return [dist - c.value]

        if c.ctype == ConstraintType.DISTANCE_PL:
            ix_p, iy_p = self._resolve_point(eids[0], "")
            e_line = self._entities[eids[1]]
            lp = e_line.param_indices
            lx1, ly1, lx2, ly2 = p[lp[0]], p[lp[1]], p[lp[2]], p[lp[3]]
            dx = lx2 - lx1
            dy = ly2 - ly1
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-20:
                return [0.0]
            signed_dist = ((p[ix_p] - lx1) * dy - (p[iy_p] - ly1) * dx) / math.sqrt(length_sq)
            return [signed_dist - c.value]

        if c.ctype == ConstraintType.ANGLE_LL:
            e_a = self._entities[eids[0]]
            e_b = self._entities[eids[1]]
            pa, pb = e_a.param_indices, e_b.param_indices
            dx_a = p[pa[2]] - p[pa[0]]
            dy_a = p[pa[3]] - p[pa[1]]
            dx_b = p[pb[2]] - p[pb[0]]
            dy_b = p[pb[3]] - p[pb[1]]
            angle_a = math.atan2(dy_a, dx_a)
            angle_b = math.atan2(dy_b, dx_b)
            diff = angle_a - angle_b
            target = math.radians(c.value)
            err = math.atan2(math.sin(diff - target), math.cos(diff - target))
            return [err]

        if c.ctype == ConstraintType.PERPENDICULAR:
            e_a = self._entities[eids[0]]
            e_b = self._entities[eids[1]]
            pa, pb = e_a.param_indices, e_b.param_indices
            dx_a = p[pa[2]] - p[pa[0]]
            dy_a = p[pa[3]] - p[pa[1]]
            dx_b = p[pb[2]] - p[pb[0]]
            dy_b = p[pb[3]] - p[pb[1]]
            return [dx_a * dx_b + dy_a * dy_b]

        if c.ctype == ConstraintType.PARALLEL:
            e_a = self._entities[eids[0]]
            e_b = self._entities[eids[1]]
            pa, pb = e_a.param_indices, e_b.param_indices
            dx_a = p[pa[2]] - p[pa[0]]
            dy_a = p[pa[3]] - p[pa[1]]
            dx_b = p[pb[2]] - p[pb[0]]
            dy_b = p[pb[3]] - p[pb[1]]
            return [dx_a * dy_b - dy_a * dx_b]

        if c.ctype == ConstraintType.EQUAL_LENGTH:
            e_a = self._entities[eids[0]]
            e_b = self._entities[eids[1]]
            pa, pb = e_a.param_indices, e_b.param_indices
            len_a = math.sqrt((p[pa[2]] - p[pa[0]]) ** 2 + (p[pa[3]] - p[pa[1]]) ** 2)
            len_b = math.sqrt((p[pb[2]] - p[pb[0]]) ** 2 + (p[pb[3]] - p[pb[1]]) ** 2)
            return [len_a - len_b]

        if c.ctype == ConstraintType.RADIUS:
            e = self._entities[eids[0]]
            r_idx = e.param_indices[2]
            return [p[r_idx] - c.value]

        if c.ctype == ConstraintType.TANGENT_LC:
            e_line = self._entities[eids[0]]
            e_circ = self._entities[eids[1]]
            lp = e_line.param_indices
            cp = e_circ.param_indices
            lx1, ly1, lx2, ly2 = p[lp[0]], p[lp[1]], p[lp[2]], p[lp[3]]
            ccx, ccy, cr = p[cp[0]], p[cp[1]], p[cp[2]]
            dx, dy = lx2 - lx1, ly2 - ly1
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-20:
                return [0.0]
            signed_dist = ((ccx - lx1) * dy - (ccy - ly1) * dx) / math.sqrt(length_sq)
            return [abs(signed_dist) - cr]

        if c.ctype == ConstraintType.TANGENT_CC:
            e_a = self._entities[eids[0]]
            e_b = self._entities[eids[1]]
            pa, pb = e_a.param_indices, e_b.param_indices
            dx = p[pa[0]] - p[pb[0]]
            dy = p[pa[1]] - p[pb[1]]
            dist = math.sqrt(dx * dx + dy * dy)
            r_sum = p[pa[2]] + p[pb[2]]
            r_diff = abs(p[pa[2]] - p[pb[2]])
            err_ext = dist - r_sum
            err_int = dist - r_diff
            return [err_ext] if abs(err_ext) < abs(err_int) else [err_int]

        if c.ctype == ConstraintType.POINT_ON_LINE:
            ix_p, iy_p = self._resolve_point(eids[0], "")
            e_line = self._entities[eids[1]]
            lp = e_line.param_indices
            lx1, ly1, lx2, ly2 = p[lp[0]], p[lp[1]], p[lp[2]], p[lp[3]]
            dx, dy = lx2 - lx1, ly2 - ly1
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-20:
                return [0.0]
            cross = (p[ix_p] - lx1) * dy - (p[iy_p] - ly1) * dx
            return [cross / math.sqrt(length_sq)]

        if c.ctype == ConstraintType.FIXED:
            ix, iy = self._resolve_point(eids[0], "")
            tx = float(sels[0]) if len(sels) > 0 and sels[0] else p[ix]
            ty = float(sels[1]) if len(sels) > 1 and sels[1] else p[iy]
            return [p[ix] - tx, p[iy] - ty]

        if c.ctype == ConstraintType.SYMMETRIC:
            ix_a, iy_a = self._resolve_point(eids[0], "")
            ix_b, iy_b = self._resolve_point(eids[1], "")
            e_line = self._entities[eids[2]]
            lp = e_line.param_indices
            lx1, ly1, lx2, ly2 = p[lp[0]], p[lp[1]], p[lp[2]], p[lp[3]]
            mx = (p[ix_a] + p[ix_b]) / 2.0
            my = (p[iy_a] + p[iy_b]) / 2.0
            dx, dy = lx2 - lx1, ly2 - ly1
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-20:
                return [0.0, 0.0]
            inv_len = 1.0 / math.sqrt(length_sq)
            cross_mid = ((mx - lx1) * dy - (my - ly1) * dx) * inv_len
            abx = p[ix_b] - p[ix_a]
            aby = p[iy_b] - p[iy_a]
            dot_perp = (abx * dx + aby * dy) * inv_len
            return [cross_mid, dot_perp]

        if c.ctype == ConstraintType.MIDPOINT:
            ix_p, iy_p = self._resolve_point(eids[0], "")
            e_line = self._entities[eids[1]]
            lp = e_line.param_indices
            mx = (p[lp[0]] + p[lp[2]]) / 2.0
            my = (p[lp[1]] + p[lp[3]]) / 2.0
            return [p[ix_p] - mx, p[iy_p] - my]

        return [0.0]

    def _eval_constraint_error(self, c: Constraint, p: np.ndarray) -> float:
        """Scalar error for diagnostics: sum of squared linear residuals."""
        residuals = self._eval_residuals(c, p)
        return sum(r * r for r in residuals)

    # -- Solver ---------------------------------------------------------------

    def _already_satisfied(self, tol: float = 1e-8) -> bool:
        """Fast check: are all constraints satisfied within *tol*?"""
        p = np.array(self._params, dtype=np.float64)
        for c in self._constraints:
            if not c.driving:
                continue
            for r in self._eval_residuals(c, p):
                if abs(r) > tol:
                    return False
        return True

    def solve(self, max_iter: int = 100, tol: float = 1e-10) -> bool:
        """
        Solve the constraint system using ``least_squares`` (TRF).

        Returns ``True`` if all constraints are satisfied within *tol*.
        The internal parameter vector is updated in-place.
        """
        if not self._constraints:
            return True

        # Skip the solver entirely when already converged
        if self._already_satisfied(tol=1e-8):
            return True

        p0 = np.array(self._params, dtype=np.float64)
        n = len(p0)

        # Build per-parameter bounds for least_squares
        lower = np.full(n, -np.inf)
        upper = np.full(n, np.inf)
        for i in self._fixed:
            lower[i] = p0[i]
            upper[i] = p0[i]

        result = least_squares(
            lambda x: self._build_residual_vector(x),
            p0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_iter,
            ftol=tol,
            xtol=tol,
            gtol=tol,
        )

        self._params = result.x.tolist()

        max_residual = float(np.max(np.abs(result.fun)))
        return max_residual < 1e-6

    def solve_drag(
        self,
        drag_eid: int,
        target_x: float,
        target_y: float,
        max_iter: int = 30,
    ) -> bool:
        """
        Solve while dragging a point toward a target position.

        Sets the point to *target* as an initial guess, then runs the
        solver with a small iteration budget (interactive drag should be
        fast, not perfect).  The drag target is treated as a *hint* —
        it biases the solver's starting point but does not add any
        constraint, so it never conflicts with existing constraints.
        """
        self.set_point(drag_eid, target_x, target_y)
        self.solve(max_iter=max_iter)
        return True

    # -- Serialisation ---------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the full solver state to a dict."""
        return {
            "params": list(self._params),
            "entities": {
                str(eid): {
                    "eid": e.eid,
                    "kind": e.kind.name,
                    "param_indices": e.param_indices,
                    "prim_index": e.prim_index,
                }
                for eid, e in self._entities.items()
            },
            "constraints": [c.to_dict() for c in self._constraints],
            "fixed": sorted(self._fixed),
            "next_eid": self._next_eid,
            "next_cid": self._next_cid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConstraintSolver":
        """Reconstruct a solver from a dict."""
        solver = cls()
        solver._params = list(d["params"])
        for key, ed in d["entities"].items():
            e = GeoEntity(
                eid=ed["eid"],
                kind=EntityKind[ed["kind"]],
                param_indices=ed["param_indices"],
                prim_index=ed.get("prim_index", -1),
            )
            solver._entities[e.eid] = e
        solver._constraints = [Constraint.from_dict(cd) for cd in d["constraints"]]
        solver._fixed = set(d.get("fixed", []))
        solver._next_eid = d.get("next_eid", 0)
        solver._next_cid = d.get("next_cid", 0)
        return solver

    # -- Diagnostic -----------------------------------------------------------

    def constraint_errors(self) -> List[Tuple[int, float]]:
        """Return (cid, error) for each constraint (sum of squared residuals)."""
        p = np.array(self._params, dtype=np.float64)
        result = []
        for c in self._constraints:
            err = self._eval_constraint_error(c, p)
            result.append((c.cid, float(err)))
        return result

    def is_fully_constrained(self, tol: float = 1e-6) -> bool:
        """True if DOF == 0 and all constraints satisfied."""
        if self.dof > 0:
            return False
        errors = self.constraint_errors()
        return all(abs(e) < tol for _, e in errors)

    def __repr__(self) -> str:
        n_ent = len(self._entities)
        n_con = len(self._constraints)
        n_par = len(self._params)
        return (
            f"ConstraintSolver(entities={n_ent}, constraints={n_con}, "
            f"params={n_par}, dof≈{self.dof})"
        )
