"""Solution collection and pooling for CP-SAT solver."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ortools.sat.python import cp_model

from ..models.structures import StructureFootprint, PlacedStructure

if TYPE_CHECKING:
    from .cpsat_placer import StructureVars

logger = logging.getLogger(__name__)


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """Collects multiple solutions from CP-SAT solver.

    Stores solutions as they are found, up to max_solutions limit.
    """

    def __init__(
        self,
        model: cp_model.CpModel,
        struct_vars: Dict[str, "StructureVars"],  # Forward reference
        structures: List[StructureFootprint],
        grid_resolution: float = 1.0,
        max_solutions: int = 100,
    ):
        """Initialize collector.

        Args:
            model: CP-SAT model
            struct_vars: Dict mapping structure ID to StructureVars
            structures: List of structure definitions
            grid_resolution: Grid cell size in meters
            max_solutions: Maximum solutions to collect
        """
        super().__init__()
        self.model = model
        self.struct_vars = struct_vars
        self.structures = {s.id: s for s in structures}
        self.grid_resolution = grid_resolution
        self.max_solutions = max_solutions

        self._solutions: List[List[PlacedStructure]] = []
        self._objectives: List[Optional[float]] = []
        self._solution_count = 0

    def on_solution_callback(self):
        """Called when a new solution is found."""
        self._solution_count += 1

        if len(self._solutions) >= self.max_solutions:
            # Stop searching
            self.StopSearch()
            return

        # Extract solution
        placements = []
        for struct_id, sv in self.struct_vars.items():
            x_grid = self.Value(sv.x_var)
            y_grid = self.Value(sv.y_var)

            # Convert back to meters
            x = x_grid * self.grid_resolution
            y = y_grid * self.grid_resolution

            # Get orientation
            struct = self.structures[struct_id]
            if sv.orientation_var is not None:
                o_idx = self.Value(sv.orientation_var)
                # Map index to actual degrees
                rotation = struct.orientations_deg[o_idx] if o_idx < len(struct.orientations_deg) else 0
            elif sv.is_pinned and struct.fixed_position is not None:
                # Pinned structure with fixed rotation
                rotation = struct.fixed_position.rotation_deg
            else:
                # Circle or single-orientation structure
                rotation = 0

            # Create PlacedStructure
            struct = self.structures[struct_id]
            placed = PlacedStructure(
                structure=struct,
                x=x,
                y=y,
                rotation_deg=rotation,
            )
            placements.append(placed)

        self._solutions.append(placements)

        # Capture objective value for this solution
        if self.model.HasObjective():
            obj_val = self.ObjectiveValue()
            self._objectives.append(obj_val)
        else:
            obj_val = None
            self._objectives.append(None)

        logger.debug(
            f"Solution {self._solution_count}: "
            f"objective={obj_val if obj_val is not None else 'N/A'}"
        )

    def get_solutions(self) -> List[List[PlacedStructure]]:
        """Get all collected solutions."""
        return self._solutions

    def get_solutions_with_objectives(
        self,
    ) -> List[Tuple[List[PlacedStructure], Optional[float]]]:
        """Get all solutions paired with their objective values.

        Returns:
            List of (placements, objective_value) tuples
        """
        return list(zip(self._solutions, self._objectives))

    def get_objectives(self) -> List[Optional[float]]:
        """Get all objective values in order."""
        return self._objectives

    @property
    def solution_count(self) -> int:
        """Get number of solutions found."""
        return self._solution_count


@dataclass
class SolutionEntry:
    """A solution entry in the pool."""

    placements: List[PlacedStructure]
    objective_value: Optional[float] = None
    rank: int = 0
    fingerprint: Optional["SolutionFingerprint"] = None  # Forward ref


class SolutionPool:
    """Pool of solutions with ranking and filtering.

    Maintains a set of diverse solutions sorted by objective value.
    """

    def __init__(self, max_size: int = 100):
        """Initialize pool.

        Args:
            max_size: Maximum solutions to keep in pool
        """
        self.max_size = max_size
        self._entries: List[SolutionEntry] = []

    def add(
        self,
        placements: List[PlacedStructure],
        objective_value: Optional[float] = None,
    ) -> bool:
        """Add a solution to the pool.

        Args:
            placements: Solution placements
            objective_value: Objective value (lower is better)

        Returns:
            True if solution was added (not duplicate)
        """
        # Check for duplicates using placement hash
        new_hash = self._compute_placement_hash(placements)
        for entry in self._entries:
            existing_hash = self._compute_placement_hash(entry.placements)
            if new_hash == existing_hash:
                return False

        entry = SolutionEntry(
            placements=placements,
            objective_value=objective_value,
        )
        self._entries.append(entry)

        # Sort by objective (lower is better)
        # Use float('inf') only for None values; 0 is a valid best objective
        self._entries.sort(key=lambda e: float('inf') if e.objective_value is None else e.objective_value)

        # Trim to max size
        if len(self._entries) > self.max_size:
            self._entries = self._entries[:self.max_size]

        # Update ranks
        for i, e in enumerate(self._entries):
            e.rank = i

        return True

    def _compute_placement_hash(self, placements: List[PlacedStructure]) -> tuple:
        """Compute hash of placement positions for deduplication."""
        # Round to grid resolution (1m) for comparison
        positions = []
        for p in sorted(placements, key=lambda x: x.structure_id):
            positions.append((
                p.structure_id,
                round(p.x),
                round(p.y),
                p.rotation_deg,
            ))
        return tuple(positions)

    def get_top_n(self, n: int) -> List[SolutionEntry]:
        """Get top N solutions by objective value."""
        return self._entries[:n]

    def get_diverse(self, n: int, min_distance: float = 0.1) -> List[SolutionEntry]:
        """Get N diverse solutions using greedy selection.

        Args:
            n: Number of solutions to select
            min_distance: Minimum fingerprint distance between solutions

        Returns:
            List of diverse solution entries
        """
        if len(self._entries) <= n:
            return self._entries

        # Use diversity module for selection
        from .diversity import filter_diverse_solutions, SolutionFingerprint

        # Compute fingerprints
        fingerprinted = []
        for entry in self._entries:
            if entry.fingerprint is None:
                entry.fingerprint = SolutionFingerprint.from_placements(
                    entry.placements
                )
            fingerprinted.append(entry)

        # Greedy diverse selection
        selected = filter_diverse_solutions(
            solutions=fingerprinted,
            target_count=n,
            min_distance=min_distance,
        )

        return selected

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)


# Forward reference resolution
from .diversity import SolutionFingerprint  # noqa: E402, F811
