"""Dock zone generation for vehicle access to structures."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, Point, LineString, box

from ..models.structures import PlacedStructure, AccessRequirement

logger = logging.getLogger(__name__)


class DockEdge(Enum):
    """Which edge of structure the dock is on."""
    NORTH = "north"  # Top edge (+y)
    SOUTH = "south"  # Bottom edge (-y)
    EAST = "east"    # Right edge (+x)
    WEST = "west"    # Left edge (-x)


@dataclass
class DockZone:
    """A vehicle dock/apron zone adjacent to a structure."""

    structure_id: str
    edge: DockEdge
    geometry: Polygon  # Rectangular dock area
    access_point: Tuple[float, float]  # Point where road connects
    vehicle_type: str
    required: bool

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of dock zone."""
        centroid = self.geometry.centroid
        return (centroid.x, centroid.y)

    def to_geojson_feature(self) -> dict:
        """Convert to GeoJSON Feature."""
        coords = list(self.geometry.exterior.coords)
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "kind": "dock_zone",
                "structure_id": self.structure_id,
                "edge": self.edge.value,
                "vehicle_type": self.vehicle_type,
                "required": self.required,
                "access_point": list(self.access_point),
            },
        }


def get_dock_edge(
    structure: PlacedStructure,
    preference: str,  # "long_side", "short_side", "any"
    blocked_edges: Optional[List[DockEdge]] = None,
) -> DockEdge:
    """Determine which edge to place dock on.

    Args:
        structure: Placed structure
        preference: Edge preference from AccessRequirement
        blocked_edges: Edges that can't have docks (adjacent to boundary, etc.)

    Returns:
        Best DockEdge for dock placement
    """
    blocked = set(blocked_edges or [])
    available = [e for e in DockEdge if e not in blocked]

    if not available:
        # All edges blocked, use south as fallback
        logger.warning(f"All edges blocked for {structure.structure_id}, using SOUTH")
        return DockEdge.SOUTH

    w, h = structure.get_current_dims()

    # Determine which edges are "long" and "short"
    if w >= h:
        # Wider than tall: N/S are long edges, E/W are short
        long_edges = [DockEdge.NORTH, DockEdge.SOUTH]
        short_edges = [DockEdge.EAST, DockEdge.WEST]
    else:
        # Taller than wide: E/W are long edges, N/S are short
        long_edges = [DockEdge.EAST, DockEdge.WEST]
        short_edges = [DockEdge.NORTH, DockEdge.SOUTH]

    if preference == "long_side":
        # Prefer long edges
        for edge in long_edges:
            if edge in available:
                return edge
        # Fallback to any available
        return available[0]

    elif preference == "short_side":
        # Prefer short edges
        for edge in short_edges:
            if edge in available:
                return edge
        return available[0]

    else:  # "any"
        # Prefer south (typical access), then west, east, north
        priority = [DockEdge.SOUTH, DockEdge.WEST, DockEdge.EAST, DockEdge.NORTH]
        for edge in priority:
            if edge in available:
                return edge
        return available[0]


def create_dock_geometry(
    structure: PlacedStructure,
    edge: DockEdge,
    dock_length: float,
    dock_width: float,
) -> Tuple[Polygon, Tuple[float, float]]:
    """Create dock zone geometry adjacent to structure edge.

    Args:
        structure: Placed structure
        edge: Which edge to place dock on
        dock_length: Dock length (perpendicular to structure edge)
        dock_width: Dock width (parallel to structure edge)

    Returns:
        Tuple of (dock_polygon, access_point)
    """
    min_x, min_y, max_x, max_y = structure.get_bounds()
    cx, cy = structure.get_center()

    # Compute dock bounds based on edge
    if edge == DockEdge.SOUTH:
        # Dock below structure
        dock_min_x = cx - dock_width / 2
        dock_max_x = cx + dock_width / 2
        dock_min_y = min_y - dock_length
        dock_max_y = min_y
        access_point = (cx, dock_min_y)

    elif edge == DockEdge.NORTH:
        # Dock above structure
        dock_min_x = cx - dock_width / 2
        dock_max_x = cx + dock_width / 2
        dock_min_y = max_y
        dock_max_y = max_y + dock_length
        access_point = (cx, dock_max_y)

    elif edge == DockEdge.WEST:
        # Dock left of structure
        dock_min_x = min_x - dock_length
        dock_max_x = min_x
        dock_min_y = cy - dock_width / 2
        dock_max_y = cy + dock_width / 2
        access_point = (dock_min_x, cy)

    else:  # EAST
        # Dock right of structure
        dock_min_x = max_x
        dock_max_x = max_x + dock_length
        dock_min_y = cy - dock_width / 2
        dock_max_y = cy + dock_width / 2
        access_point = (dock_max_x, cy)

    dock_poly = box(dock_min_x, dock_min_y, dock_max_x, dock_max_y)
    return dock_poly, access_point


def generate_dock_zones(
    placements: List[PlacedStructure],
    default_dock_length: float = 15.0,
    default_dock_width: float = 6.0,
) -> List[DockZone]:
    """Generate dock zones for all structures with access requirements.

    Args:
        placements: List of placed structures
        default_dock_length: Default dock length in meters
        default_dock_width: Default dock width in meters

    Returns:
        List of DockZone objects
    """
    dock_zones = []

    for placement in placements:
        access = placement.structure.access

        if access is None:
            continue  # No access requirement

        # Determine dock dimensions
        dock_length = access.dock_length if access.dock_length else default_dock_length
        dock_width = access.dock_width if access.dock_width else default_dock_width

        # Determine which edge
        edge = get_dock_edge(placement, access.dock_edge)

        # Create geometry
        dock_poly, access_pt = create_dock_geometry(
            placement, edge, dock_length, dock_width
        )

        dock_zones.append(DockZone(
            structure_id=placement.structure_id,
            edge=edge,
            geometry=dock_poly,
            access_point=access_pt,
            vehicle_type=access.vehicle,
            required=access.required,
        ))

    logger.info(f"Generated {len(dock_zones)} dock zones")
    return dock_zones


def check_dock_overlaps(
    dock_zones: List[DockZone],
    structures: List[PlacedStructure],
    tolerance: float = 0.1,
) -> List[Tuple[str, str, float]]:
    """Check for overlaps between dock zones and structures.

    Args:
        dock_zones: List of dock zones
        structures: List of placed structures
        tolerance: Overlap area threshold

    Returns:
        List of (dock_structure_id, overlapping_structure_id, overlap_area)
    """
    overlaps = []

    for dock in dock_zones:
        for struct in structures:
            # Don't check overlap with own structure
            if struct.structure_id == dock.structure_id:
                continue

            struct_poly = struct.to_shapely_polygon()
            if dock.geometry.intersects(struct_poly):
                intersection = dock.geometry.intersection(struct_poly)
                overlap_area = intersection.area
                if overlap_area > tolerance:
                    overlaps.append((
                        dock.structure_id,
                        struct.structure_id,
                        overlap_area,
                    ))

    return overlaps


def adjust_dock_for_overlap(
    dock: DockZone,
    structure: PlacedStructure,
    other_structures: List[PlacedStructure],
) -> Optional[DockZone]:
    """Try to adjust dock zone to avoid overlaps.

    Attempts to find an alternative edge that doesn't overlap.

    Args:
        dock: Original dock zone
        structure: Structure the dock belongs to
        other_structures: Other structures to avoid

    Returns:
        Adjusted DockZone or None if no valid position found
    """
    # Try each edge in priority order
    for edge in [DockEdge.SOUTH, DockEdge.WEST, DockEdge.EAST, DockEdge.NORTH]:
        if edge == dock.edge:
            continue  # Already tried this edge

        # Create new dock at this edge
        access = structure.structure.access
        if access is None:
            return None

        dock_poly, access_pt = create_dock_geometry(
            structure, edge, access.dock_length, access.dock_width
        )

        # Check for overlaps
        has_overlap = False
        for other in other_structures:
            if other.structure_id == structure.structure_id:
                continue
            other_poly = other.to_shapely_polygon()
            if dock_poly.intersects(other_poly):
                has_overlap = True
                break

        if not has_overlap:
            return DockZone(
                structure_id=dock.structure_id,
                edge=edge,
                geometry=dock_poly,
                access_point=access_pt,
                vehicle_type=dock.vehicle_type,
                required=dock.required,
            )

    return None  # No valid position found
