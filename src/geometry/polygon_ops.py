"""Polygon operations using Shapely and pyclipper.

Provides robust polygon inset, buffer, union, and subtraction operations
for computing buildable areas and structure footprints.
"""

from __future__ import annotations

import logging

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

try:
    import pyclipper
    HAS_PYCLIPPER = True
except ImportError:
    HAS_PYCLIPPER = False
    logging.warning("pyclipper not available, using Shapely buffer for insets")

logger = logging.getLogger(__name__)

# Type aliases
Coords = list[tuple[float, float]]
PolygonLike = Polygon | MultiPolygon


def polygon_from_coords(coords: Coords) -> Polygon:
    """Create Shapely Polygon from coordinate list.

    Args:
        coords: List of (x, y) tuples forming polygon exterior

    Returns:
        Shapely Polygon
    """
    return Polygon(coords)


def polygon_to_coords(polygon: Polygon) -> Coords:
    """Extract coordinates from Shapely Polygon.

    Args:
        polygon: Shapely Polygon

    Returns:
        List of (x, y) tuples
    """
    return list(polygon.exterior.coords)


def inset_polygon(
    polygon: PolygonLike,
    distance: float,
    use_pyclipper: bool = True,
) -> PolygonLike:
    """Inset (shrink) polygon by given distance.

    Uses pyclipper for robust offsetting when available,
    falls back to Shapely negative buffer.

    Args:
        polygon: Polygon to inset
        distance: Inset distance in same units as polygon (positive = shrink)
        use_pyclipper: Use pyclipper if available

    Returns:
        Inset polygon (may be MultiPolygon if inset splits the shape)
    """
    if distance <= 0:
        return polygon

    if use_pyclipper and HAS_PYCLIPPER:
        return _inset_with_pyclipper(polygon, distance)
    else:
        return _inset_with_shapely(polygon, distance)


def _inset_with_pyclipper(polygon: PolygonLike, distance: float) -> PolygonLike:
    """Inset polygon using pyclipper (more robust for complex shapes)."""
    # pyclipper uses integer coordinates, scale up
    scale = 1000.0  # mm precision

    pco = pyclipper.PyclipperOffset()

    if isinstance(polygon, MultiPolygon):
        # Handle each polygon separately
        results = []
        for geom in polygon.geoms:
            result = _inset_with_pyclipper(geom, distance)
            if not result.is_empty:
                results.append(result)
        if not results:
            return Polygon()
        return unary_union(results)

    # Convert exterior to pyclipper format (scaled integers)
    exterior_coords = [
        (int(x * scale), int(y * scale))
        for x, y in polygon.exterior.coords[:-1]  # Exclude closing point
    ]

    # Add exterior path
    pco.AddPath(
        exterior_coords,
        pyclipper.JT_MITER,  # Miter join for sharp corners
        pyclipper.ET_CLOSEDPOLYGON,
    )

    # Handle holes (interior rings)
    for interior in polygon.interiors:
        hole_coords = [
            (int(x * scale), int(y * scale))
            for x, y in interior.coords[:-1]
        ]
        pco.AddPath(
            hole_coords,
            pyclipper.JT_MITER,
            pyclipper.ET_CLOSEDPOLYGON,
        )

    # Execute offset (negative for inset)
    try:
        solution = pco.Execute(-distance * scale)
    except Exception as e:
        logger.warning(f"pyclipper offset failed: {e}, using Shapely fallback")
        return _inset_with_shapely(polygon, distance)

    if not solution:
        return Polygon()

    # Convert back to Shapely
    polygons = []
    for path in solution:
        coords = [(x / scale, y / scale) for x, y in path]
        if len(coords) >= 3:
            poly = Polygon(coords)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)

    if not polygons:
        return Polygon()
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return MultiPolygon(polygons)


def _inset_with_shapely(polygon: PolygonLike, distance: float) -> PolygonLike:
    """Inset polygon using Shapely negative buffer."""
    result = polygon.buffer(-distance, join_style=2)  # Miter join

    if result.is_empty:
        return Polygon()

    # Ensure valid geometry
    if not result.is_valid:
        result = make_valid(result)

    return result


def buffer_polygon(
    polygon: PolygonLike,
    distance: float,
    cap_style: int = 1,  # round
    join_style: int = 1,  # round
) -> PolygonLike:
    """Expand polygon by given distance (buffer).

    Args:
        polygon: Polygon to expand
        distance: Buffer distance (positive = expand)
        cap_style: 1=round, 2=flat, 3=square
        join_style: 1=round, 2=mitre, 3=bevel

    Returns:
        Buffered polygon
    """
    result = polygon.buffer(distance, cap_style=cap_style, join_style=join_style)

    if not result.is_valid:
        result = make_valid(result)

    return result


def union_polygons(polygons: list[PolygonLike]) -> PolygonLike:
    """Compute union of multiple polygons.

    Args:
        polygons: List of polygons to union

    Returns:
        Unified polygon (may be MultiPolygon)
    """
    if not polygons:
        return Polygon()

    valid_polygons = []
    for p in polygons:
        if p is None or p.is_empty:
            continue
        if not p.is_valid:
            p = make_valid(p)
        valid_polygons.append(p)

    if not valid_polygons:
        return Polygon()

    return unary_union(valid_polygons)


def subtract_polygons(
    base: PolygonLike,
    subtract: list[PolygonLike],
) -> PolygonLike:
    """Subtract multiple polygons from base polygon.

    Args:
        base: Base polygon to subtract from
        subtract: List of polygons to subtract

    Returns:
        Result polygon (may be MultiPolygon or empty)
    """
    if base is None or base.is_empty:
        return Polygon()

    result = base
    for sub in subtract:
        if sub is None or sub.is_empty:
            continue
        if not sub.is_valid:
            sub = make_valid(sub)
        result = result.difference(sub)
        if result.is_empty:
            return Polygon()

    if not result.is_valid:
        result = make_valid(result)

    return result


def compute_buildable_area(
    boundary: PolygonLike,
    setback: float,
    keepouts: list[PolygonLike] | None = None,
    existing: list[PolygonLike] | None = None,
    keepout_buffers: list[float] | None = None,
    existing_buffers: list[float] | None = None,
) -> PolygonLike:
    """Compute the buildable area within a site boundary.

    Applies:
    1. Inset boundary by setback distance
    2. Subtract keepout zones (with optional buffers)
    3. Subtract existing structures (with optional buffers)

    Args:
        boundary: Site boundary polygon
        setback: Property line setback distance
        keepouts: Zones where building is prohibited
        keepout_buffers: Buffer distances for each keepout (defaults to 0)
        existing: Existing structures that cannot be moved
        existing_buffers: Buffer distances around existing (defaults to setback)

    Returns:
        Buildable area polygon (may be MultiPolygon)
    """
    keepouts = keepouts or []
    existing = existing or []
    keepout_buffers = keepout_buffers or [0.0] * len(keepouts)
    existing_buffers = existing_buffers or [setback] * len(existing)

    # Step 1: Inset boundary by setback
    buildable = inset_polygon(boundary, setback)

    if buildable.is_empty:
        logger.warning(f"Setback of {setback}m results in empty buildable area")
        return Polygon()

    # Step 2: Subtract buffered keepouts
    subtract_zones = []
    for i, keepout in enumerate(keepouts):
        if keepout is None or keepout.is_empty:
            continue
        buffer_dist = keepout_buffers[i] if i < len(keepout_buffers) else 0.0
        if buffer_dist > 0:
            keepout = buffer_polygon(keepout, buffer_dist)
        subtract_zones.append(keepout)

    # Step 3: Subtract buffered existing structures
    for i, existing_struct in enumerate(existing):
        if existing_struct is None or existing_struct.is_empty:
            continue
        buffer_dist = existing_buffers[i] if i < len(existing_buffers) else setback
        if buffer_dist > 0:
            existing_struct = buffer_polygon(existing_struct, buffer_dist)
        subtract_zones.append(existing_struct)

    # Perform subtraction
    if subtract_zones:
        buildable = subtract_polygons(buildable, subtract_zones)

    return buildable


def get_polygon_area(polygon: PolygonLike) -> float:
    """Get area of polygon in square units.

    Args:
        polygon: Polygon to measure

    Returns:
        Area in square units (e.g., m^2)
    """
    if polygon is None or polygon.is_empty:
        return 0.0
    return polygon.area


def get_polygon_bounds(polygon: PolygonLike) -> tuple[float, float, float, float]:
    """Get bounding box of polygon.

    Args:
        polygon: Polygon to measure

    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    if polygon is None or polygon.is_empty:
        return (0.0, 0.0, 0.0, 0.0)
    return polygon.bounds


def create_rectangle(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    rotation_deg: int = 0,
) -> Polygon:
    """Create a rectangle polygon centered at given point.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        width: Rectangle width
        height: Rectangle height
        rotation_deg: Rotation in degrees (0, 90, 180, 270)

    Returns:
        Rectangle Polygon
    """
    # Handle rotation by swapping dimensions
    if rotation_deg in [90, 270]:
        width, height = height, width

    half_w = width / 2
    half_h = height / 2

    coords = [
        (center_x - half_w, center_y - half_h),
        (center_x + half_w, center_y - half_h),
        (center_x + half_w, center_y + half_h),
        (center_x - half_w, center_y + half_h),
        (center_x - half_w, center_y - half_h),  # Close ring
    ]

    return Polygon(coords)


def create_circle(
    center_x: float,
    center_y: float,
    diameter: float,
    resolution: int = 32,
) -> Polygon:
    """Create a circular polygon centered at given point.

    Args:
        center_x: X coordinate of center
        center_y: Y coordinate of center
        diameter: Circle diameter
        resolution: Number of points to approximate circle

    Returns:
        Circular Polygon
    """
    from shapely.geometry import Point

    center = Point(center_x, center_y)
    return center.buffer(diameter / 2, resolution=resolution)
