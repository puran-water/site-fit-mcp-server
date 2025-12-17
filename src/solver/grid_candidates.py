"""Grid-based candidate computation for structure placement.

Uses Shapely prepared geometry and STRtree for efficient polygon containment
and keepout intersection tests. This module provides valid placement candidates
that respect site boundaries and keepout zones at the solver level.
"""

import logging
from dataclasses import dataclass

from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


@dataclass
class CandidateGrid:
    """Grid of valid placement candidates for a structure.

    Stores the valid (x, y) positions where a structure center can be placed,
    along with metadata for CP-SAT integration.
    """

    structure_id: str
    candidates: list[tuple[float, float]]  # (x, y) in meters
    grid_resolution: float
    width: float  # Structure width in meters
    height: float  # Structure height in meters

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def x_coords(self) -> list[float]:
        """Get list of x coordinates."""
        return [c[0] for c in self.candidates]

    @property
    def y_coords(self) -> list[float]:
        """Get list of y coordinates."""
        return [c[1] for c in self.candidates]

    def to_grid_coords(self) -> tuple[list[int], list[int]]:
        """Convert to integer grid coordinates for CP-SAT.

        Returns:
            Tuple of (x_grid_coords, y_grid_coords)
        """
        scale = 1.0 / self.grid_resolution
        x_grid = [int(c[0] * scale) for c in self.candidates]
        y_grid = [int(c[1] * scale) for c in self.candidates]
        return x_grid, y_grid


def compute_valid_candidates(
    buildable: Polygon,
    structure_width: float,
    structure_height: float,
    grid_resolution: float = 1.0,
    keepouts: list[Polygon] | None = None,
    structure_id: str = "unknown",
    orientations: list[int] | None = None,
) -> CandidateGrid:
    """Compute valid placement candidates for a structure.

    Uses prepared Shapely geometry for efficient containment checks. The buildable
    polygon is shrunk by half the structure size to ensure full containment.

    Args:
        buildable: Polygon representing the buildable area (already has boundary
                   setbacks applied)
        structure_width: Structure width in meters
        structure_height: Structure height in meters
        grid_resolution: Grid spacing in meters (default 1.0m)
        keepouts: Optional list of keepout zone polygons to avoid
        structure_id: Identifier for logging
        orientations: List of valid orientations in degrees (e.g., [0, 90]).
                      If provided, uses max dimension for shrinking to allow any rotation.

    Returns:
        CandidateGrid with valid placement positions

    Example:
        >>> from shapely.geometry import box
        >>> buildable = box(0, 0, 100, 80)
        >>> grid = compute_valid_candidates(buildable, 10, 8, grid_resolution=2.0)
        >>> print(f"Found {grid.num_candidates} valid positions")
    """
    # Determine effective dimensions (max if rotation allowed)
    if orientations and len(orientations) > 1:
        # Structure can rotate, need clearance for any orientation
        max_dim = max(structure_width, structure_height)
        effective_w = max_dim
        effective_h = max_dim
    else:
        effective_w = structure_width
        effective_h = structure_height

    # Shrink buildable area by half structure size to ensure full containment
    shrink_x = effective_w / 2
    shrink_y = effective_h / 2
    shrink_amount = max(shrink_x, shrink_y)

    valid_region = buildable.buffer(-shrink_amount)

    if valid_region.is_empty or not valid_region.is_valid:
        logger.warning(
            f"Structure {structure_id} ({structure_width}x{structure_height}m) "
            f"is too large for buildable area after shrinking by {shrink_amount}m"
        )
        return CandidateGrid(
            structure_id=structure_id,
            candidates=[],
            grid_resolution=grid_resolution,
            width=structure_width,
            height=structure_height,
        )

    # Prepare geometry for fast containment checks
    prepared = prep(valid_region)

    # Build STRtree for keepouts if provided (fast intersection tests)
    keepout_tree = None
    expanded_keepouts = []
    if keepouts:
        # Expand keepouts by half structure size to prevent overlap
        for ko in keepouts:
            expanded = ko.buffer(shrink_amount)
            if expanded.is_valid and not expanded.is_empty:
                expanded_keepouts.append(expanded)
        if expanded_keepouts:
            keepout_tree = STRtree(expanded_keepouts)

    # Generate grid candidates
    min_x, min_y, max_x, max_y = valid_region.bounds
    candidates = []

    x = min_x
    while x <= max_x:
        y = min_y
        while y <= max_y:
            pt = Point(x, y)

            # Check containment in valid region
            if prepared.contains(pt):
                # Check no intersection with keepouts
                if keepout_tree is not None:
                    # query() returns indices of geometries that may intersect
                    nearby_indices = keepout_tree.query(pt)
                    in_keepout = any(
                        expanded_keepouts[idx].contains(pt)
                        for idx in nearby_indices
                    )
                    if not in_keepout:
                        candidates.append((x, y))
                else:
                    candidates.append((x, y))

            y += grid_resolution
        x += grid_resolution

    logger.debug(
        f"Structure {structure_id}: {len(candidates)} candidates "
        f"in {(max_x - min_x) / grid_resolution:.0f}x{(max_y - min_y) / grid_resolution:.0f} grid"
    )

    return CandidateGrid(
        structure_id=structure_id,
        candidates=candidates,
        grid_resolution=grid_resolution,
        width=structure_width,
        height=structure_height,
    )


def compute_candidates_for_structures(
    structures: list,  # List[StructureFootprint]
    buildable: Polygon,
    grid_resolution: float = 1.0,
    keepouts: list[Polygon] | None = None,
) -> dict[str, CandidateGrid]:
    """Compute valid candidates for all structures.

    Args:
        structures: List of StructureFootprint objects
        buildable: Buildable area polygon
        grid_resolution: Grid spacing in meters
        keepouts: Optional list of keepout polygons

    Returns:
        Dictionary mapping structure_id to CandidateGrid
    """
    result = {}

    for struct in structures:
        if struct.is_circle:
            fp = struct.footprint
            w = h = fp.d
        else:
            fp = struct.footprint
            w = fp.w
            h = fp.h

        grid = compute_valid_candidates(
            buildable=buildable,
            structure_width=w,
            structure_height=h,
            grid_resolution=grid_resolution,
            keepouts=keepouts,
            structure_id=struct.id,
            orientations=struct.orientations_deg if not struct.is_circle else None,
        )
        result[struct.id] = grid

    return result


def candidates_to_element_tables(
    candidate_grid: CandidateGrid,
    grid_resolution: float,
) -> tuple[list[int], list[int]]:
    """Convert candidates to lookup tables for CP-SAT AddElement constraint.

    The AddElement constraint allows:
        model.AddElement(index_var, table, result_var)
    Where result_var = table[index_var]

    Args:
        candidate_grid: Grid of valid candidates
        grid_resolution: Grid resolution for scaling

    Returns:
        Tuple of (x_table, y_table) as integer grid coordinates

    Example:
        >>> x_table, y_table = candidates_to_element_tables(grid, 1.0)
        >>> # In CP-SAT:
        >>> idx = model.NewIntVar(0, len(x_table) - 1, "idx")
        >>> model.AddElement(idx, x_table, x_var)
        >>> model.AddElement(idx, y_table, y_var)
    """
    return candidate_grid.to_grid_coords()
