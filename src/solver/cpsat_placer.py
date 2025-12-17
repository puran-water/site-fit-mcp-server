"""OR-Tools CP-SAT based placement solver.

Uses NoOverlap2D constraint for structure placement with:
- Hard constraints: No overlap, boundary containment
- Soft constraints: Topology flow direction, adjacency preferences
"""

import logging
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from ..models.rules import RuleSet
from ..models.structures import PlacedStructure, StructureFootprint
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

    # Randomize search for solution diversity
    randomize_search: bool = True

    # Size of variable pool for random branching (larger = more diversity)
    search_random_variable_pool_size: int = 10


@dataclass
class StructureVars:
    """CP-SAT variables for a single structure."""

    structure_id: str
    x_var: cp_model.IntVar
    y_var: cp_model.IntVar
    orientation_var: cp_model.IntVar | None = None  # None for circles
    x_interval: cp_model.IntervalVar | None = None
    y_interval: cp_model.IntervalVar | None = None

    # Dimensions at each orientation (for rectangles)
    dims_by_orientation: dict[int, tuple[int, int]] = field(default_factory=dict)

    # Whether this is a circular structure
    is_circular: bool = False

    # Whether this is a pinned structure with fixed position
    is_pinned: bool = False

    # Actual dimensions (grid units)
    width_grid: int = 0
    height_grid: int = 0

    # Optional intervals for rotation (keyed by orientation index)
    # Both x and y share the same presence literal per orientation
    optional_x_intervals: dict[int, cp_model.IntervalVar] = field(default_factory=dict)
    optional_y_intervals: dict[int, cp_model.IntervalVar] = field(default_factory=dict)
    orientation_presence: dict[int, cp_model.IntVar] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Result from the placement solver."""

    status: str  # "optimal", "feasible", "infeasible", "timeout"
    solutions: list[list[PlacedStructure]]
    solve_time_seconds: float
    num_solutions_found: int
    objective_value: float | None = None
    statistics: dict[str, any] = field(default_factory=dict)
    solution_objectives: list[float | None] = field(default_factory=list)


class PlacementSolver:
    """CP-SAT based structure placement solver.

    Uses OR-Tools NoOverlap2D constraint to ensure non-overlapping placements
    within the buildable area, with soft penalties for topology violations.
    """

    def __init__(
        self,
        structures: list[StructureFootprint],
        bounds: tuple[float, float, float, float],  # min_x, min_y, max_x, max_y
        rules: RuleSet,
        hints: PlacementHints | None = None,
        config: PlacementSolverConfig | None = None,
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
        self.struct_vars: dict[str, StructureVars] = {}

        # Objectives
        self.objective_terms: list[cp_model.LinearExpr] = []

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
        """Create decision variables for a structure.

        For rotatable structures, uses optional intervals with reified bounds.
        Each orientation has its own presence literal shared by both x and y intervals.

        For pinned structures with fixed_position, creates constant variables
        at the specified location (still added to NoOverlap2D for collision detection).
        """
        is_circle = struct.is_circle
        is_pinned = struct.pinned and struct.fixed_position is not None

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

        # Get equipment-specific boundary setback (in addition to buildable area setback)
        equip_setback = self.rules.get_equipment_to_boundary(struct.type)
        default_setback = self.rules.setbacks.property_line_default
        extra_setback = max(0, equip_setback - default_setback)
        extra_setback_grid = self._to_grid(extra_setback)

        # Handle pinned structures with fixed position
        if is_pinned:
            fixed_x = self._to_grid(struct.fixed_position.x)
            fixed_y = self._to_grid(struct.fixed_position.y)
            fixed_rot = struct.fixed_position.rotation_deg

            # Create constant variables at fixed position
            x_var = self.model.NewConstant(fixed_x)
            y_var = self.model.NewConstant(fixed_y)

            # Override orientations to only the fixed rotation
            if not is_circle:
                orientations = [fixed_rot]

            logger.debug(
                f"Pinned structure {struct.id} at fixed position "
                f"({fixed_x}, {fixed_y}) rotation {fixed_rot}°"
            )
        else:
            # For rotatable structures, use max dimension for initial domain bounds
            # (reified constraints will tighten per-orientation)
            max_half = max(width, height) // 2 + 1
            x_margin = max_half + extra_setback_grid
            y_margin = max_half + extra_setback_grid

            x_var = self.model.NewIntVar(
                min_x + x_margin, max_x - x_margin, f"x_{struct.id}"
            )
            y_var = self.model.NewIntVar(
                min_y + y_margin, max_y - y_margin, f"y_{struct.id}"
            )

        # Build dimensions by orientation mapping
        dims_by_orientation = {}
        for deg in orientations:
            if deg in [0, 180]:
                dims_by_orientation[deg] = (width, height)
            else:  # 90, 270
                dims_by_orientation[deg] = (height, width)

        # Store initial vars (intervals added in _add_no_overlap_constraint)
        sv = StructureVars(
            structure_id=struct.id,
            x_var=x_var,
            y_var=y_var,
            orientation_var=None,  # Set below for rotatable
            is_circular=is_circle,
            is_pinned=is_pinned,
            width_grid=width,
            height_grid=height,
            dims_by_orientation=dims_by_orientation,
        )

        # Handle rotation with optional intervals and reified bounds
        if not is_circle and len(orientations) > 1:
            # Create presence literal for each orientation
            # CRITICAL: Both x and y intervals share the SAME presence literal
            o_vars = []
            for i, deg in enumerate(orientations):
                o_var = self.model.NewBoolVar(f"o_{struct.id}_{deg}")
                o_vars.append(o_var)
                sv.orientation_presence[i] = o_var

                # Get dimensions for this orientation
                w, h = dims_by_orientation[deg]
                half_w = w // 2
                half_h = h // 2

                # Reify containment bounds per orientation
                # (tighter than max dimension bounds)
                self.model.Add(
                    x_var >= min_x + half_w + extra_setback_grid
                ).OnlyEnforceIf(o_var)
                self.model.Add(
                    x_var <= max_x - half_w - extra_setback_grid
                ).OnlyEnforceIf(o_var)
                self.model.Add(
                    y_var >= min_y + half_h + extra_setback_grid
                ).OnlyEnforceIf(o_var)
                self.model.Add(
                    y_var <= max_y - half_h - extra_setback_grid
                ).OnlyEnforceIf(o_var)

            # Exactly one orientation must be active
            self.model.AddExactlyOne(o_vars)

            # Store orientation variable as the index (for solution extraction)
            sv.orientation_var = self.model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(list(range(len(orientations)))),
                f"oidx_{struct.id}",
            )
            # Link orientation index to presence bools
            for i, o_var in enumerate(o_vars):
                self.model.Add(sv.orientation_var == i).OnlyEnforceIf(o_var)

        self.struct_vars[struct.id] = sv

    def _add_no_overlap_constraint(self):
        """Add NoOverlap2D constraint for all structures.

        For rotatable structures, uses optional intervals with presence literals.
        ALL optional intervals (from all orientations) are added to NoOverlap2D -
        CP-SAT handles the constraint correctly based on which are "present".

        CLEARANCE: Intervals are expanded by half the max clearance needed,
        so NoOverlap2D naturally enforces clearance (both structures expand,
        giving full clearance between them).
        """
        x_intervals = []
        y_intervals = []

        for struct_id, sv in self.struct_vars.items():
            # Compute clearance expansion for this structure
            # Use ceiling division to ensure full clearance (add 1 to round up)
            max_clear = self._compute_max_clearance(struct_id)
            clearance_half = (max_clear + 1) // 2  # Ceiling division

            # Check if structure has rotation options
            has_rotation = len(sv.orientation_presence) > 1

            if has_rotation:
                # Create optional intervals for each orientation
                # CRITICAL: Both x and y intervals share the SAME presence literal
                for i, (deg, (w, h)) in enumerate(sv.dims_by_orientation.items()):
                    if i not in sv.orientation_presence:
                        continue  # Skip orientations not in the allowed list

                    presence = sv.orientation_presence[i]
                    # Add clearance expansion to half-dimensions
                    half_w = w // 2 + clearance_half
                    half_h = h // 2 + clearance_half
                    expanded_w = w + 2 * clearance_half
                    expanded_h = h + 2 * clearance_half

                    # Create optional interval for this orientation (with clearance)
                    x_int = self.model.NewOptionalIntervalVar(
                        sv.x_var - half_w,    # start
                        expanded_w,            # size (includes clearance)
                        sv.x_var + (expanded_w - half_w),  # end
                        presence,              # is_present literal
                        f"xi_{struct_id}_{deg}",
                    )
                    y_int = self.model.NewOptionalIntervalVar(
                        sv.y_var - half_h,
                        expanded_h,
                        sv.y_var + (expanded_h - half_h),
                        presence,
                        f"yi_{struct_id}_{deg}",
                    )

                    sv.optional_x_intervals[i] = x_int
                    sv.optional_y_intervals[i] = y_int

                    x_intervals.append(x_int)
                    y_intervals.append(y_int)

                # Also store the first interval as the "default" for backward compat
                if 0 in sv.optional_x_intervals:
                    sv.x_interval = sv.optional_x_intervals[0]
                    sv.y_interval = sv.optional_y_intervals[0]
            else:
                # Non-rotatable structure: use fixed interval with clearance
                w, h = sv.width_grid, sv.height_grid
                half_w = w // 2 + clearance_half
                half_h = h // 2 + clearance_half
                expanded_w = w + 2 * clearance_half
                expanded_h = h + 2 * clearance_half

                x_interval = self.model.NewIntervalVar(
                    sv.x_var - half_w,
                    expanded_w,
                    sv.x_var + (expanded_w - half_w),
                    f"xi_{struct_id}",
                )
                y_interval = self.model.NewIntervalVar(
                    sv.y_var - half_h,
                    expanded_h,
                    sv.y_var + (expanded_h - half_h),
                    f"yi_{struct_id}",
                )

                sv.x_interval = x_interval
                sv.y_interval = y_interval

                x_intervals.append(x_interval)
                y_intervals.append(y_interval)

        # Add NoOverlap2D constraint with all intervals (including optional)
        if x_intervals:
            self.model.AddNoOverlap2D(x_intervals, y_intervals)

    def _add_clearance_constraints(self):
        """Add minimum clearance constraints between structures.

        CONSERVATIVE APPROACH: Instead of pairwise Manhattan constraints,
        we compute the max clearance each structure needs and expand its
        intervals accordingly. This is conservative (may reject valid placements)
        but dramatically reduces Shapely validator rejects.

        For each structure, we find the maximum clearance it requires to any
        other structure, then expand its interval by half that clearance.
        When two structures' expanded intervals don't overlap, they have
        at least max_clearance/2 + max_clearance/2 = max_clearance between them.
        """
        # Already applied via interval expansion in _add_no_overlap_constraint
        # The expansion is computed per-structure based on max clearance needed
        pass

    def _compute_max_clearance(self, struct_id: str) -> int:
        """Compute maximum clearance this structure needs to any other.

        Returns:
            Maximum clearance in grid units (ceiling to ensure full coverage)
        """
        struct = next((s for s in self.structures if s.id == struct_id), None)
        if struct is None:
            return 0

        max_clearance = 0

        for other in self.structures:
            if other.id == struct_id:
                continue
            clearance = self.rules.get_clearance(struct.type, other.type)
            clearance_grid = self._to_grid(clearance)
            max_clearance = max(max_clearance, clearance_grid)

        return max_clearance

    def _compute_global_max_clearance(self) -> int:
        """Compute global maximum clearance across all structure pairs.

        This is used for conservative interval expansion where all structures
        are expanded uniformly.

        Returns:
            Maximum clearance in grid units
        """
        max_clearance = 0
        for s1 in self.structures:
            for s2 in self.structures:
                if s1.id >= s2.id:  # Skip self and duplicates
                    continue
                clearance = self.rules.get_clearance(s1.type, s2.type)
                clearance_grid = self._to_grid(clearance)
                max_clearance = max(max_clearance, clearance_grid)
        return max_clearance

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

        # Enable randomized search for solution diversity
        if self.config.randomize_search:
            solver.parameters.randomize_search = True
            solver.parameters.search_random_variable_pool_size = (
                self.config.search_random_variable_pool_size
            )

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
            solution_objectives=collector.get_objectives(),
        )


def create_solver_from_request(
    structures: list[StructureFootprint],
    buildable_bounds: tuple[float, float, float, float],
    rules: RuleSet,
    hints: PlacementHints | None = None,
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
