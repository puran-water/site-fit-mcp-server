"""Pairwise clearance calculations and violation detection."""

from dataclasses import dataclass

from shapely.strtree import STRtree

from ..models.rules import RuleSet
from ..models.structures import PlacedStructure
from .polygon_ops import PolygonLike


@dataclass
class ClearanceViolation:
    """A clearance constraint violation between two structures."""

    structure1_id: str
    structure2_id: str
    actual_distance: float
    required_distance: float
    violation_amount: float  # How much under required clearance
    structure1_type: str
    structure2_type: str

    @property
    def is_critical(self) -> bool:
        """Check if violation is critical (overlap or very close)."""
        return self.actual_distance < 0.5  # Less than 0.5m

    def __str__(self) -> str:
        return (
            f"Clearance violation: {self.structure1_id} <-> {self.structure2_id}: "
            f"{self.actual_distance:.2f}m (required {self.required_distance:.2f}m, "
            f"violation {self.violation_amount:.2f}m)"
        )


def compute_pairwise_distances(
    structures: list[PlacedStructure],
) -> dict[tuple[str, str], float]:
    """Compute distance between all pairs of structures.

    Uses Shapely for accurate polygon-to-polygon distance.

    Args:
        structures: List of placed structures with positions

    Returns:
        Dict mapping (id1, id2) pairs to distances (id1 < id2 alphabetically)
    """
    distances = {}

    # Build geometries
    geometries = {}
    for s in structures:
        geometries[s.structure_id] = s.to_shapely_polygon()

    # Compute pairwise distances
    structure_ids = sorted(geometries.keys())
    for i, id1 in enumerate(structure_ids):
        geom1 = geometries[id1]
        for id2 in structure_ids[i + 1:]:
            geom2 = geometries[id2]
            dist = geom1.distance(geom2)
            distances[(id1, id2)] = dist

    return distances


def check_clearance_violations(
    structures: list[PlacedStructure],
    rules: RuleSet,
    structure_types: dict[str, str] | None = None,
) -> list[ClearanceViolation]:
    """Check all pairwise clearances against rules.

    Args:
        structures: List of placed structures
        rules: RuleSet with clearance requirements
        structure_types: Optional dict mapping structure_id to equipment type
            (if not provided, uses PlacedStructure fields)

    Returns:
        List of ClearanceViolation objects for any violations
    """
    violations = []
    structure_types = structure_types or {}

    # Build lookup for structure info
    struct_map = {s.structure_id: s for s in structures}

    # Compute all distances
    distances = compute_pairwise_distances(structures)

    # Check against required clearances
    for (id1, id2), actual_dist in distances.items():
        # Get equipment types
        s1 = struct_map[id1]
        s2 = struct_map[id2]

        type1 = structure_types.get(id1) or getattr(s1, 'equipment_type', 'default')
        type2 = structure_types.get(id2) or getattr(s2, 'equipment_type', 'default')

        # Get required clearance from rules
        required = rules.get_clearance(type1, type2)

        if actual_dist < required:
            violations.append(ClearanceViolation(
                structure1_id=id1,
                structure2_id=id2,
                actual_distance=actual_dist,
                required_distance=required,
                violation_amount=required - actual_dist,
                structure1_type=type1,
                structure2_type=type2,
            ))

    return violations


def get_minimum_clearance(
    structure: PlacedStructure,
    other_structures: list[PlacedStructure],
    use_strtree: bool = True,
) -> tuple[float, str | None]:
    """Get minimum clearance from structure to any other structure.

    Uses STRtree spatial index for efficient nearest neighbor queries
    when there are many structures.

    Args:
        structure: The structure to check
        other_structures: List of other structures
        use_strtree: Use STRtree for large structure lists (default True)

    Returns:
        Tuple of (min_distance, closest_structure_id)
    """
    # Filter out self from other_structures
    filtered = [s for s in other_structures if s.structure_id != structure.structure_id]
    if not filtered:
        return (float('inf'), None)

    geom = structure.to_shapely_polygon()

    # For small lists, linear scan is faster than building STRtree
    if not use_strtree or len(filtered) < 10:
        min_dist = float('inf')
        closest_id = None
        for other in filtered:
            other_geom = other.to_shapely_polygon()
            dist = geom.distance(other_geom)
            if dist < min_dist:
                min_dist = dist
                closest_id = other.structure_id
        return (min_dist, closest_id)

    # Use STRtree for larger lists
    geometries = []
    id_map = {}
    for s in filtered:
        other_geom = s.to_shapely_polygon()
        geometries.append(other_geom)
        id_map[id(other_geom)] = s.structure_id

    tree = STRtree(geometries)

    # Find nearest geometry
    nearest_geom = tree.nearest(geom)
    if nearest_geom is None:
        return (float('inf'), None)

    nearest_id = id_map.get(id(nearest_geom))
    dist = geom.distance(nearest_geom)

    return (dist, nearest_id)


def check_overlap(
    structures: list[PlacedStructure],
    tolerance: float = 0.01,
) -> list[tuple[str, str, float]]:
    """Check for overlapping structures.

    Uses STRtree spatial index for efficient candidate filtering.
    For n structures, reduces from O(n²) to O(n log n + k) where k is
    the number of actual candidate pairs.

    Args:
        structures: List of placed structures
        tolerance: Intersection area threshold to consider overlap

    Returns:
        List of (id1, id2, overlap_area) tuples for overlapping pairs
    """
    if len(structures) < 2:
        return []

    # Build geometries and spatial index
    geometries = []
    id_list = []
    for s in structures:
        geom = s.to_shapely_polygon()
        geometries.append(geom)
        id_list.append(s.structure_id)

    tree = STRtree(geometries)

    # Use STRtree to find overlapping pairs efficiently
    overlaps = []
    checked_pairs: set = set()

    for i, geom1 in enumerate(geometries):
        id1 = id_list[i]

        # Query for geometries that intersect with geom1
        candidate_indices = tree.query(geom1, predicate='intersects')

        for j in candidate_indices:
            if j <= i:  # Skip self and already-checked pairs
                continue

            pair_key = (i, j) if i < j else (j, i)
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            id2 = id_list[j]
            geom2 = geometries[j]

            # Compute actual intersection area
            intersection = geom1.intersection(geom2)
            overlap_area = intersection.area
            if overlap_area > tolerance:
                overlaps.append((id1, id2, overlap_area))

    return overlaps


def compute_clearance_matrix(
    structures: list[PlacedStructure],
    rules: RuleSet,
    structure_types: dict[str, str] | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute detailed clearance matrix with actual and required distances.

    Args:
        structures: List of placed structures
        rules: RuleSet with clearance requirements
        structure_types: Optional dict mapping structure_id to equipment type

    Returns:
        Dict mapping (id1, id2) to {actual, required, margin}
    """
    structure_types = structure_types or {}
    struct_map = {s.structure_id: s for s in structures}
    distances = compute_pairwise_distances(structures)

    matrix = {}
    for (id1, id2), actual_dist in distances.items():
        s1 = struct_map[id1]
        s2 = struct_map[id2]

        type1 = structure_types.get(id1) or getattr(s1, 'equipment_type', 'default')
        type2 = structure_types.get(id2) or getattr(s2, 'equipment_type', 'default')

        required = rules.get_clearance(type1, type2)

        matrix[(id1, id2)] = {
            'actual': actual_dist,
            'required': required,
            'margin': actual_dist - required,
            'type1': type1,
            'type2': type2,
        }

    return matrix


def check_boundary_clearance(
    structure: PlacedStructure,
    boundary: PolygonLike,
    required_setback: float,
) -> tuple[bool, float]:
    """Check if structure maintains required setback from boundary.

    Args:
        structure: Structure to check
        boundary: Site boundary polygon
        required_setback: Required setback distance

    Returns:
        Tuple of (is_valid, actual_distance_to_boundary)
    """
    geom = structure.to_shapely_polygon()

    # Distance to boundary exterior
    dist = boundary.exterior.distance(geom)

    return (dist >= required_setback, dist)


def find_nearest_neighbors(
    structures: list[PlacedStructure],
    k: int = 3,
) -> dict[str, list[tuple[str, float]]]:
    """Find k nearest neighbors for each structure.

    Uses STRtree spatial index for efficient queries.

    Args:
        structures: List of placed structures
        k: Number of neighbors to find

    Returns:
        Dict mapping structure_id to list of (neighbor_id, distance)
    """
    if not structures or k < 1:
        return {}

    # Build geometries and spatial index
    geometries = []
    id_map = {}
    for i, s in enumerate(structures):
        geom = s.to_shapely_polygon()
        geometries.append(geom)
        id_map[id(geom)] = s.structure_id

    tree = STRtree(geometries)

    neighbors = {}
    for s in structures:
        geom = s.to_shapely_polygon()

        # Query nearest (returns indices)
        # STRtree.nearest returns geometries, not indices in newer shapely
        nearest_geoms = tree.nearest(geom, num_results=k + 1)  # +1 to exclude self

        nearby = []
        for near_geom in nearest_geoms:
            near_id = id_map.get(id(near_geom))
            if near_id and near_id != s.structure_id:
                dist = geom.distance(near_geom)
                nearby.append((near_id, dist))

        # Sort by distance and take top k
        nearby.sort(key=lambda x: x[1])
        neighbors[s.structure_id] = nearby[:k]

    return neighbors


def validate_no_overlap_2d_compatible(
    structures: list[PlacedStructure],
) -> tuple[bool, list[str]]:
    """Validate that structures can be represented in NoOverlap2D constraint.

    NoOverlap2D requires axis-aligned rectangles. This checks if:
    - All structures have valid bounding boxes
    - No structures have incompatible rotations

    Args:
        structures: List of placed structures

    Returns:
        Tuple of (is_compatible, list of issues)
    """
    issues = []

    for s in structures:
        # Check bounds are valid
        try:
            bounds = s.get_bounds()
            min_x, min_y, max_x, max_y = bounds
            if max_x <= min_x or max_y <= min_y:
                issues.append(f"Structure {s.structure_id} has invalid bounds")
        except Exception as e:
            issues.append(f"Structure {s.structure_id} bounds error: {e}")

        # Check rotation is axis-aligned
        if s.rotation_deg not in [0, 90, 180, 270]:
            issues.append(
                f"Structure {s.structure_id} has non-axis-aligned rotation "
                f"({s.rotation_deg}°)"
            )

    return (len(issues) == 0, issues)
