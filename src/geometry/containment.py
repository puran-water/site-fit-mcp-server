"""Containment checks for structure placement within site boundaries."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep

from .polygon_ops import (
    PolygonLike,
    create_rectangle,
    create_circle,
    inset_polygon,
)


class ContainmentStatus(Enum):
    """Status of containment check."""
    FULLY_CONTAINED = "fully_contained"
    PARTIALLY_OUTSIDE = "partially_outside"
    FULLY_OUTSIDE = "fully_outside"
    INVALID_GEOMETRY = "invalid_geometry"


@dataclass
class ContainmentResult:
    """Result of a containment check."""

    status: ContainmentStatus
    overlap_ratio: float  # 0.0 to 1.0, fraction of structure inside boundary
    outside_area: float   # Area of structure outside boundary
    message: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if placement is valid (fully contained)."""
        return self.status == ContainmentStatus.FULLY_CONTAINED


def check_containment(
    structure: PolygonLike,
    boundary: PolygonLike,
    tolerance: float = 0.01,
) -> ContainmentResult:
    """Check if structure is fully contained within boundary.

    Args:
        structure: Structure footprint polygon
        boundary: Site boundary or buildable area polygon
        tolerance: Small tolerance for floating point comparison (meters)

    Returns:
        ContainmentResult with status and metrics
    """
    if structure is None or structure.is_empty:
        return ContainmentResult(
            status=ContainmentStatus.INVALID_GEOMETRY,
            overlap_ratio=0.0,
            outside_area=0.0,
            message="Structure geometry is empty",
        )

    if boundary is None or boundary.is_empty:
        return ContainmentResult(
            status=ContainmentStatus.INVALID_GEOMETRY,
            overlap_ratio=0.0,
            outside_area=structure.area,
            message="Boundary geometry is empty",
        )

    if not structure.is_valid or not boundary.is_valid:
        return ContainmentResult(
            status=ContainmentStatus.INVALID_GEOMETRY,
            overlap_ratio=0.0,
            outside_area=0.0,
            message="Invalid geometry",
        )

    # Use prepared geometry for faster containment check
    prepared_boundary = prep(boundary)

    # Check if fully contained
    if prepared_boundary.contains(structure):
        return ContainmentResult(
            status=ContainmentStatus.FULLY_CONTAINED,
            overlap_ratio=1.0,
            outside_area=0.0,
        )

    # Check for overlap
    if not prepared_boundary.intersects(structure):
        return ContainmentResult(
            status=ContainmentStatus.FULLY_OUTSIDE,
            overlap_ratio=0.0,
            outside_area=structure.area,
        )

    # Partial overlap - compute metrics
    intersection = structure.intersection(boundary)
    overlap_area = intersection.area if not intersection.is_empty else 0.0
    structure_area = structure.area

    overlap_ratio = overlap_area / structure_area if structure_area > 0 else 0.0
    outside_area = structure_area - overlap_area

    # Check if "close enough" to fully contained (within tolerance)
    if outside_area < tolerance * tolerance:
        return ContainmentResult(
            status=ContainmentStatus.FULLY_CONTAINED,
            overlap_ratio=1.0,
            outside_area=0.0,
        )

    return ContainmentResult(
        status=ContainmentStatus.PARTIALLY_OUTSIDE,
        overlap_ratio=overlap_ratio,
        outside_area=outside_area,
        message=f"Structure {outside_area:.2f}m² outside boundary",
    )


def check_placement_valid(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: int,
    buildable_area: PolygonLike,
    is_circular: bool = False,
) -> ContainmentResult:
    """Check if a placement is valid within buildable area.

    Convenience function that creates structure geometry and checks containment.

    Args:
        center_x: X coordinate of structure center
        center_y: Y coordinate of structure center
        width: Structure width (or diameter for circles)
        height: Structure height (ignored for circles)
        rotation_deg: Rotation in degrees
        buildable_area: Polygon representing valid placement region
        is_circular: If True, treat as circular structure

    Returns:
        ContainmentResult with validation status
    """
    if is_circular:
        structure = create_circle(center_x, center_y, width)
    else:
        structure = create_rectangle(center_x, center_y, width, height, rotation_deg)

    return check_containment(structure, buildable_area)


def get_valid_placement_region(
    buildable_area: PolygonLike,
    structure_width: float,
    structure_height: float,
    is_circular: bool = False,
) -> PolygonLike:
    """Get region where structure center can be placed.

    Insets the buildable area by half the structure dimensions
    to get valid center positions.

    Args:
        buildable_area: Available buildable area
        structure_width: Structure width
        structure_height: Structure height
        is_circular: If True, use width as diameter

    Returns:
        Polygon where structure center can be validly placed
    """
    if is_circular:
        # For circles, inset by radius
        inset_dist = structure_width / 2
    else:
        # For rectangles, use the larger dimension (conservative)
        # This assumes any rotation might be used
        inset_dist = max(structure_width, structure_height) / 2

    return inset_polygon(buildable_area, inset_dist)


def get_placement_bounds(
    buildable_area: PolygonLike,
    structure_width: float,
    structure_height: float,
    is_circular: bool = False,
) -> Tuple[float, float, float, float]:
    """Get bounding box for valid placement centers.

    Args:
        buildable_area: Available buildable area
        structure_width: Structure width
        structure_height: Structure height
        is_circular: If True, use width as diameter

    Returns:
        Tuple of (min_x, min_y, max_x, max_y) for valid center positions
    """
    valid_region = get_valid_placement_region(
        buildable_area, structure_width, structure_height, is_circular
    )

    if valid_region is None or valid_region.is_empty:
        return (0.0, 0.0, 0.0, 0.0)

    return valid_region.bounds


def check_point_in_buildable(
    x: float,
    y: float,
    buildable_area: PolygonLike,
) -> bool:
    """Quick check if a point is within buildable area.

    Args:
        x: X coordinate
        y: Y coordinate
        buildable_area: Buildable area polygon

    Returns:
        True if point is inside buildable area
    """
    if buildable_area is None or buildable_area.is_empty:
        return False

    point = Point(x, y)
    return buildable_area.contains(point)


def get_valid_cells(
    buildable_area: PolygonLike,
    grid_resolution: float = 1.0,
    structure_width: float = 0.0,
    structure_height: float = 0.0,
) -> List[Tuple[int, int]]:
    """Get list of valid grid cells for structure placement.

    Creates a grid over the buildable area and returns cells
    where a structure center could be validly placed.

    Args:
        buildable_area: Available buildable area
        grid_resolution: Grid cell size in meters
        structure_width: Structure width (0 for point containment)
        structure_height: Structure height (0 for point containment)

    Returns:
        List of (grid_x, grid_y) tuples for valid cells
    """
    if buildable_area is None or buildable_area.is_empty:
        return []

    # Get valid region for structure centers
    if structure_width > 0 or structure_height > 0:
        max_dim = max(structure_width, structure_height)
        valid_region = inset_polygon(buildable_area, max_dim / 2)
    else:
        valid_region = buildable_area

    if valid_region.is_empty:
        return []

    # Get bounds
    min_x, min_y, max_x, max_y = valid_region.bounds

    # Prepare geometry for fast containment checks
    prepared = prep(valid_region)

    # Generate valid cells
    valid_cells = []
    grid_x = 0
    x = min_x
    while x <= max_x:
        grid_y = 0
        y = min_y
        while y <= max_y:
            if prepared.contains(Point(x, y)):
                valid_cells.append((grid_x, grid_y))
            y += grid_resolution
            grid_y += 1
        x += grid_resolution
        grid_x += 1

    return valid_cells


def grid_to_coords(
    grid_x: int,
    grid_y: int,
    origin_x: float,
    origin_y: float,
    grid_resolution: float,
) -> Tuple[float, float]:
    """Convert grid cell to world coordinates.

    Args:
        grid_x: Grid X index
        grid_y: Grid Y index
        origin_x: X coordinate of grid origin
        origin_y: Y coordinate of grid origin
        grid_resolution: Grid cell size

    Returns:
        Tuple of (world_x, world_y)
    """
    return (
        origin_x + grid_x * grid_resolution,
        origin_y + grid_y * grid_resolution,
    )


def coords_to_grid(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    grid_resolution: float,
) -> Tuple[int, int]:
    """Convert world coordinates to grid cell.

    Args:
        x: World X coordinate
        y: World Y coordinate
        origin_x: X coordinate of grid origin
        origin_y: Y coordinate of grid origin
        grid_resolution: Grid cell size

    Returns:
        Tuple of (grid_x, grid_y)
    """
    return (
        int((x - origin_x) / grid_resolution),
        int((y - origin_y) / grid_resolution),
    )
