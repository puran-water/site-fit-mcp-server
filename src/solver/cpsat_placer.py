"""OR-Tools CP-SAT based placement solver.

Uses NoOverlap2D constraint for structure placement with:
- Hard constraints: No overlap, boundary containment
- Soft constraints: Topology flow direction, adjacency preferences
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from ..models.structures import StructureFootprint, PlacedStructure
from ..models.rules import RuleSet
from ..topology.placement_hints import PlacementHints
from .solution_pool import SolutionCollector

logger = logging.getLogger(__name__)


@dataclass
class PlacementSolverConfig:
    """Configuration for the placement solver."""

    # Grid resolution in meters (1.0 = 1m grid)
    grid_resolution: float = 1.0

    # Time limit in seconds
    max_time_seconds: float = 60.0

    # Maximum solutions to collect
    max_solutions: int = 100

    # Random seed for reproducibility
    seed: int = 42

    # Penalty weights
    flow_violation_weight: int = 10  # Penalty for flow direction violations
    adjacency_weight: int = 5        # Bonus for adjacent placement
    compactness_weight: int = 1      # Weight for compactness objective

    # Circle inflation factor for bounding boxes (to ensure clearance)
    circle_inflation: float = 1.1

    # Whether to enable OR-Tools symmetry breaking
    symmetry_breaking: bool = True

    # Parallel workers (0 = auto)
    num_workers: int = 0


@dataclass
class StructureVars:
    """CP-SAT variables for a single structure."""

    structure_id: str
    x_var: cp_model.IntVar
    y_var: cp_model.IntVar
    orientation_var: Optional[cp_model.IntVar] = None  # None for circles
    x_interval: Optional[cp_model.IntervalVar] = None
    y_interval: Optional[cp_model.IntervalVar] = None

    # Dimensions at each orientation (for rectangles)
    dims_by_orientation: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Whether this is a circular structure
    is_circular: bool = False

    # Actual dimensions (grid units)
    width_grid: int = 0
    height_grid: int = 0


@dataclass
class SolverResult:
    """Result from the placement solver."""

    status: str  # "optimal", "feasible", "infeasible", "timeout"
    solutions: List[List[PlacedStructure]]
    solve_time_seconds: float
    num_solutions_found: int
    objective_value: Optional[float] = None
    statistics: Dict[str, any] = field(default_factory=dict)


class PlacementSolver:
    """CP-SAT based structure placement solver.

    Uses OR-Tools NoOverlap2D constraint to ensure non-overlapping placements
    within the buildable area, with soft penalties for topology violations.
    """

    def __init__(
        self,
        structures: List[StructureFootprint],
        bounds: Tuple[float, float, float, float],  # min_x, min_y, max_x, max_y
        rules: RuleSet,
        hints: Optional[PlacementHints] = None,
        config: Optional[PlacementSolverConfig] = None,
    ):
        """Initialize solver.

        Args:
            structures: List of structures to place
            bounds: Bounding box for valid placement region
            rules: Engineering rules for clearances
            hints: Topology-derived placement hints
            config: Solver configuration
        """
        self.structures = structures
        self.bounds = bounds
        self.rules = rules
        self.hints = hints or PlacementHints()
        self.config = config or PlacementSolverConfig()

        # CP-SAT model
        self.model = cp_model.CpModel()

        # Structure variables
        self.struct_vars: Dict[str, StructureVars] = {}

        # Objectives
        self.objective_terms: List[cp_model.LinearExpr] = []

        # Build the model
        self._build_model()

    def _to_grid(self, value: float) -> int:
        """Convert meters to grid units."""
        return int(value / self.config.grid_resolution)

    def _from_grid(self, value: int) -> float:
        """Convert grid units to meters."""
        return value * self.config.grid_resolution

    def _build_model(self):
        """Build the CP-SAT model with all constraints."""
        min_x, min_y, max_x, max_y = self.bounds

        # Grid bounds
        grid_min_x = self._to_grid(min_x)
        grid_min_y = self._to_grid(min_y)
        grid_max_x = self._to_grid(max_x)
        grid_max_y = self._to_grid(max_y)

        logger.info(
            f"Building model: {len(self.structures)} structures, "
            f"grid bounds ({grid_min_x}, {grid_min_y}) to ({grid_max_x}, {grid_max_y})"
        )

        # Create variables for each structure
        for struct in self.structures:
            self._create_structure_vars(
                struct, grid_min_x, grid_min_y, grid_max_x, grid_max_y
            )

        # Add NoOverlap2D constraint
        self._add_no_overlap_constraint()

        # Add clearance constraints
        self._add_clearance_constraints()

        # Add soft topology constraints
        self._add_topology_constraints()

        # Set objective
        self._set_objective()

    def _create_structure_vars(
        self,
        struct: StructureFootprint,
        min_x: int,
        min_y: int,
        max_x: int,
        max_y: int,
    ):
        """Create decision variables for a structure."""
        is_circle = struct.is_circle

        # Get dimensions
        if is_circle:
            fp = struct.footprint  # type: ignore
            dim = self._to_grid(fp.d * self.config.circle_inflation)
            width, height = dim, dim
            orientations = [0]  # Circles have no rotation
        else:
            fp = struct.footprint  # type: ignore
            width = self._to_grid(fp.w)
            height = self._to_grid(fp.h)
            orientations = struct.orientations_deg

        # X and Y position variables (center of structure)
        # Adjust bounds to keep structure fully inside
        half_w = max(width, height) // 2 + 1  # Conservative for any rotation
        half_h = half_w

        # Get equipment-specific boundary setback (in addition to buildable area setback)
        # The buildable area already has property_line_default setback applied,
        # so we only need to add the EXTRA setback for this equipment type
        equip_setback = self.rules.get_equipment_to_boundary(struct.type)
        default_setback = self.rules.setbacks.property_line_default
        extra_setback = max(0, equip_setback - default_setback)
        extra_setback_grid = self._to_grid(extra_setback)

        # Apply extra boundary margin for this structure
        x_margin = half_w + extra_setback_grid
        y_margin = half_h + extra_setback_grid

        x_var = self.model.NewIntVar(
            min_x + x_margin, max_x - x_margin, f"x_{struct.id}"
        )
        y_var = self.model.NewIntVar(
            min_y + y_margin, max_y - y_margin, f"y_{struct.id}"
        )

        # Orientation variable (for rectangles with multiple orientations)
        if is_circle or len(orientations) == 1:
            o_var = None
            actual_width = width
            actual_height = height
        else:
            # Create orientation variable
            o_var = self.model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(list(range(len(orientations)))),
                f"o_{struct.id}",
            )
            # Width/height depend on orientation (handled in intervals)
            actual_width = max(width, height)
            actual_height = actual_width

        # Store structure vars
        self.struct_vars[struct.id] = StructureVars(
            structure_id=struct.id,
            x_var=x_var,
            y_var=y_var,
            orientation_var=o_var,
            is_circular=is_circle,
            width_grid=width,
            height_grid=height,
            dims_by_orientation={
                0: (width, height),
                90: (height, width),
                180: (width, height),
                270: (height, width),
            },
        )

    def _add_no_overlap_constraint(self):
        """Add NoOverlap2D constraint for all structures."""
        x_intervals = []
        y_intervals = []

        for struct_id, sv in self.struct_vars.items():
            # Use max dimensions for interval (conservative for any rotation)
            w = max(sv.width_grid, sv.height_grid)
            h = w

            # Create interval variables
            # Interval: [start, start + size)
            x_interval = self.model.NewIntervalVar(
                sv.x_var - w // 2,  # start
                w,                   # size
                sv.x_var + (w - w // 2),  # end
                f"xi_{struct_id}",
            )
            y_interval = self.model.NewIntervalVar(
                sv.y_var - h // 2,
                h,
                sv.y_var + (h - h // 2),
                f"yi_{struct_id}",
            )

            sv.x_interval = x_interval
            sv.y_interval = y_interval

            x_intervals.append(x_interval)
            y_intervals.append(y_interval)

        # Add NoOverlap2D constraint
        if x_intervals:
            self.model.AddNoOverlap2D(x_intervals, y_intervals)

    def _add_clearance_constraints(self):
        """Add minimum clearance constraints between structures.

        Note: NoOverlap2D handles non-overlap. For additional clearance,
        we expand the intervals or add distance constraints.
        """
        # Already handled by inflating circular structures
        # For additional clearances, we'd need to add explicit distance constraints

        struct_list = list(self.struct_vars.items())
        for i, (id1, sv1) in enumerate(struct_list):
            struct1 = next(s for s in self.structures if s.id == id1)
            for id2, sv2 in struct_list[i + 1:]:
                struct2 = next(s for s in self.structures if s.id == id2)

                # Get required clearance
                clearance = self.rules.get_clearance(struct1.type, struct2.type)
                clearance_grid = self._to_grid(clearance)

                if clearance_grid > 0:
                    # Add soft penalty for close placement
                    # (Hard distance constraints can make problem infeasible)
                    self._add_soft_distance_penalty(sv1, sv2, clearance_grid)

    def _add_soft_distance_penalty(
        self,
        sv1: StructureVars,
        sv2: StructureVars,
        min_dist_grid: int,
    ):
        """Add penalty for structures closer than minimum clearance.

        Uses Manhattan distance approximation since CP-SAT can't handle
        Euclidean distance directly. The actual Shapely validation in
        Phase 5 will catch any remaining violations.
        """
        # Get structure dimensions for interval inflation
        dims1 = sv1.dims_by_orientation.get(0, (0, 0))
        dims2 = sv2.dims_by_orientation.get(0, (0, 0))

        # Half-sizes
        half_w1 = dims1[0] // 2 if dims1[0] else 1
        half_w2 = dims2[0] // 2 if dims2[0] else 1

        # Inflate intervals by clearance amount
        # This makes NoOverlap2D enforce clearance as non-overlap
        clearance_half = min_dist_grid // 2

        # Create inflated intervals for this pair
        # Note: CP-SAT NoOverlap2D already uses intervals, so we add
        # additional constraints to ensure centers are far enough apart

        # Minimum center-to-center distance (conservative approximation)
        min_center_dist = half_w1 + half_w2 + min_dist_grid

        # Add disjunctive constraint: |x1 - x2| + |y1 - y2| >= min_center_dist
        # Linearized: at least one of the following must be true:
        # - x1 >= x2 + min_center_dist
        # - x2 >= x1 + min_center_dist
        # - y1 >= y2 + min_center_dist
        # - y2 >= y1 + min_center_dist

        # Create boolean variables for each direction
        b_x1_right = self.model.NewBoolVar(f"clear_{sv1.structure_id}_{sv2.structure_id}_x1r")
        b_x2_right = self.model.NewBoolVar(f"clear_{sv1.structure_id}_{sv2.structure_id}_x2r")
        b_y1_above = self.model.NewBoolVar(f"clear_{sv1.structure_id}_{sv2.structure_id}_y1a")
        b_y2_above = self.model.NewBoolVar(f"clear_{sv1.structure_id}_{sv2.structure_id}_y2a")

        # If b_x1_right, then x1 >= x2 + min_center_dist
        self.model.Add(sv1.x_var >= sv2.x_var + min_center_dist).OnlyEnforceIf(b_x1_right)
        self.model.Add(sv2.x_var >= sv1.x_var + min_center_dist).OnlyEnforceIf(b_x2_right)
        self.model.Add(sv1.y_var >= sv2.y_var + min_center_dist).OnlyEnforceIf(b_y1_above)
        self.model.Add(sv2.y_var >= sv1.y_var + min_center_dist).OnlyEnforceIf(b_y2_above)

        # At least one must be true (structures must be separated in at least one axis)
        self.model.AddBoolOr([b_x1_right, b_x2_right, b_y1_above, b_y2_above])

    def _add_topology_constraints(self):
        """Add soft constraints based on process topology."""
        if not self.hints.flow_precedence:
            return

        # Flow direction: upstream should be west of downstream
        for upstream_id, downstream_id in self.hints.flow_precedence:
            if upstream_id not in self.struct_vars or downstream_id not in self.struct_vars:
                continue

            sv_up = self.struct_vars[upstream_id]
            sv_down = self.struct_vars[downstream_id]

            # Soft penalty if downstream is west of upstream
            # violation = max(0, x_up - x_down)
            violation = self.model.NewIntVar(
                0, 1000, f"flow_viol_{upstream_id}_{downstream_id}"
            )
            self.model.Add(violation >= sv_up.x_var - sv_down.x_var)
            self.model.Add(violation >= 0)

            # Add to objective
            self.objective_terms.append(
                self.config.flow_violation_weight * violation
            )

        # Adjacency preferences
        for (id1, id2), weight in self.hints.adjacency_weights.items():
            if id1 not in self.struct_vars or id2 not in self.struct_vars:
                continue

            sv1 = self.struct_vars[id1]
            sv2 = self.struct_vars[id2]

            # Minimize Manhattan distance for adjacent pairs
            # dist = |x1-x2| + |y1-y2|
            x_diff = self.model.NewIntVar(-1000, 1000, f"xdiff_{id1}_{id2}")
            y_diff = self.model.NewIntVar(-1000, 1000, f"ydiff_{id1}_{id2}")
            x_abs = self.model.NewIntVar(0, 1000, f"xabs_{id1}_{id2}")
            y_abs = self.model.NewIntVar(0, 1000, f"yabs_{id1}_{id2}")

            self.model.Add(x_diff == sv1.x_var - sv2.x_var)
            self.model.Add(y_diff == sv1.y_var - sv2.y_var)
            self.model.AddAbsEquality(x_abs, x_diff)
            self.model.AddAbsEquality(y_abs, y_diff)

            # Add weighted distance to objective (negative weight = minimize)
            adj_weight = int(weight * self.config.adjacency_weight)
            if adj_weight > 0:
                self.objective_terms.append(adj_weight * (x_abs + y_abs))

    def _set_objective(self):
        """Set the optimization objective."""
        # Compactness: minimize total bounding box
        all_x = [sv.x_var for sv in self.struct_vars.values()]
        all_y = [sv.y_var for sv in self.struct_vars.values()]

        if all_x and self.config.compactness_weight > 0:
            max_x = self.model.NewIntVar(0, 10000, "max_x")
            min_x = self.model.NewIntVar(0, 10000, "min_x")
            max_y = self.model.NewIntVar(0, 10000, "max_y")
            min_y = self.model.NewIntVar(0, 10000, "min_y")

            self.model.AddMaxEquality(max_x, all_x)
            self.model.AddMinEquality(min_x, all_x)
            self.model.AddMaxEquality(max_y, all_y)
            self.model.AddMinEquality(min_y, all_y)

            span_x = self.model.NewIntVar(0, 10000, "span_x")
            span_y = self.model.NewIntVar(0, 10000, "span_y")
            self.model.Add(span_x == max_x - min_x)
            self.model.Add(span_y == max_y - min_y)

            self.objective_terms.append(
                self.config.compactness_weight * (span_x + span_y)
            )

        # Set objective to minimize sum of all terms
        if self.objective_terms:
            self.model.Minimize(sum(self.objective_terms))

    def solve(self) -> SolverResult:
        """Solve the placement problem.

        Returns:
            SolverResult with solutions and statistics
        """
        import time

        solver = cp_model.CpSolver()

        # Configure solver
        solver.parameters.max_time_in_seconds = self.config.max_time_seconds
        solver.parameters.random_seed = self.config.seed

        if self.config.num_workers > 0:
            solver.parameters.num_workers = self.config.num_workers

        # Enable advanced features for NoOverlap2D
        solver.parameters.use_timetabling_in_no_overlap_2d = True
        solver.parameters.use_energetic_reasoning_in_no_overlap_2d = True

        if self.config.symmetry_breaking:
            solver.parameters.symmetry_level = 2

        # Collect multiple solutions
        collector = SolutionCollector(
            model=self.model,
            struct_vars=self.struct_vars,
            structures=self.structures,
            grid_resolution=self.config.grid_resolution,
            max_solutions=self.config.max_solutions,
        )

        start_time = time.time()
        status = solver.Solve(self.model, collector)
        solve_time = time.time() - start_time

        # Map status
        status_map = {
            cp_model.OPTIMAL: "optimal",
            cp_model.FEASIBLE: "feasible",
            cp_model.INFEASIBLE: "infeasible",
            cp_model.MODEL_INVALID: "invalid",
            cp_model.UNKNOWN: "timeout",
        }
        status_str = status_map.get(status, "unknown")

        # Get solutions
        solutions = collector.get_solutions()

        logger.info(
            f"Solver finished: status={status_str}, "
            f"solutions={len(solutions)}, time={solve_time:.2f}s"
        )

        return SolverResult(
            status=status_str,
            solutions=solutions,
            solve_time_seconds=solve_time,
            num_solutions_found=len(solutions),
            objective_value=solver.ObjectiveValue() if status in [cp_model.OPTIMAL, cp_model.FEASIBLE] else None,
            statistics={
                "branches": solver.NumBranches(),
                "conflicts": solver.NumConflicts(),
                "wall_time": solver.WallTime(),
            },
        )


def create_solver_from_request(
    structures: List[StructureFootprint],
    buildable_bounds: Tuple[float, float, float, float],
    rules: RuleSet,
    hints: Optional[PlacementHints] = None,
    max_solutions: int = 10,
    max_time_seconds: float = 60.0,
    seed: int = 42,
) -> PlacementSolver:
    """Convenience function to create a solver from request parameters.

    Args:
        structures: Structures to place
        buildable_bounds: (min_x, min_y, max_x, max_y) of buildable area
        rules: Engineering rules
        hints: Topology placement hints
        max_solutions: Maximum solutions to find
        max_time_seconds: Time limit
        seed: Random seed

    Returns:
        Configured PlacementSolver
    """
    config = PlacementSolverConfig(
        max_solutions=max_solutions,
        max_time_seconds=max_time_seconds,
        seed=seed,
    )

    return PlacementSolver(
        structures=structures,
        bounds=buildable_bounds,
        rules=rules,
        hints=hints,
        config=config,
    )
