"""OR-Tools CP-SAT based placement solver."""

from .cpsat_placer import (
    PlacementSolver,
    PlacementSolverConfig,
    SolverResult,
)
from .solution_pool import (
    SolutionCollector,
    SolutionPool,
)
from .diversity import (
    SolutionFingerprint,
    filter_diverse_solutions,
    compute_solution_distance,
)

__all__ = [
    # Main solver
    "PlacementSolver",
    "PlacementSolverConfig",
    "SolverResult",
    # Solution collection
    "SolutionCollector",
    "SolutionPool",
    # Diversity filtering
    "SolutionFingerprint",
    "filter_diverse_solutions",
    "compute_solution_distance",
]
