"""OR-Tools CP-SAT based placement solver."""

from .cpsat_placer import (
    PlacementSolver,
    PlacementSolverConfig,
    SolverResult,
)
from .diversity import (
    SolutionFingerprint,
    compute_solution_distance,
    filter_diverse_solutions,
)
from .grid_candidates import (
    CandidateGrid,
    candidates_to_element_tables,
    compute_candidates_for_structures,
    compute_valid_candidates,
)
from .solution_pool import (
    SolutionCollector,
    SolutionPool,
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
    # Grid candidates
    "CandidateGrid",
    "compute_valid_candidates",
    "compute_candidates_for_structures",
    "candidates_to_element_tables",
]
