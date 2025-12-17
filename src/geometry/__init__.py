"""Geometry operations for site layout using Shapely and pyclipper."""

from .clearance import (
    ClearanceViolation,
    check_clearance_violations,
    compute_pairwise_distances,
    get_minimum_clearance,
)
from .containment import (
    ContainmentResult,
    check_containment,
    check_placement_valid,
    get_valid_placement_region,
)
from .polygon_ops import (
    buffer_polygon,
    compute_buildable_area,
    inset_polygon,
    polygon_from_coords,
    polygon_to_coords,
    subtract_polygons,
    union_polygons,
)

__all__ = [
    # Polygon operations
    "compute_buildable_area",
    "inset_polygon",
    "buffer_polygon",
    "polygon_from_coords",
    "polygon_to_coords",
    "union_polygons",
    "subtract_polygons",
    # Containment checks
    "check_containment",
    "check_placement_valid",
    "get_valid_placement_region",
    "ContainmentResult",
    # Clearance calculations
    "compute_pairwise_distances",
    "check_clearance_violations",
    "get_minimum_clearance",
    "ClearanceViolation",
]
