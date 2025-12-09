"""Geometry operations for site layout using Shapely and pyclipper."""

from .polygon_ops import (
    compute_buildable_area,
    inset_polygon,
    buffer_polygon,
    polygon_from_coords,
    polygon_to_coords,
    union_polygons,
    subtract_polygons,
)
from .containment import (
    check_containment,
    check_placement_valid,
    get_valid_placement_region,
    ContainmentResult,
)
from .clearance import (
    compute_pairwise_distances,
    check_clearance_violations,
    get_minimum_clearance,
    ClearanceViolation,
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
