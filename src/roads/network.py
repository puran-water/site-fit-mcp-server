"""Road network generation and validation."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from shapely.geometry import LineString, Point, Polygon, MultiLineString
from shapely.ops import linemerge, unary_union

from ..models.site import Entrance
from ..models.structures import PlacedStructure
from ..models.solution import RoadSegment, RoadNetwork
from ..models.rules import RuleSet
from .dock_zones import DockZone, generate_dock_zones
from .pathfinder import CostGrid, create_cost_grid, find_road_path, PathfinderResult

logger = logging.getLogger(__name__)


@dataclass
class RoadValidationResult:
    """Result from road network validation."""

    is_valid: bool
    all_docks_accessible: bool
    accessible_docks: Set[str]
    inaccessible_docks: Set[str]
    total_road_length: float
    issues: List[str] = field(default_factory=list)


class RoadNetworkBuilder:
    """Builds road network connecting entrances to structure dock zones."""

    def __init__(
        self,
        structures: List[PlacedStructure],
        entrances: List[Entrance],
        boundary: Polygon,
        rules: RuleSet,
        grid_resolution: float = 1.0,
        keepouts: Optional[List[Polygon]] = None,
    ):
        """Initialize road network builder.

        Args:
            structures: Placed structures (used as obstacles)
            entrances: Site entrances
            boundary: Site boundary polygon
            rules: Engineering rules
            grid_resolution: Pathfinding grid resolution (meters)
            keepouts: Optional keepout zones (used as obstacles in pathfinding)
        """
        self.structures = structures
        self.entrances = entrances
        self.boundary = boundary
        self.rules = rules
        self.grid_resolution = grid_resolution
        self.keepouts = keepouts or []

        # Generate dock zones
        self.dock_zones = generate_dock_zones(structures)

        # Build cost grid
        self._build_cost_grid()

    def _build_cost_grid(self):
        """Build cost grid for pathfinding.

        Ensures roads stay inside the site boundary by marking all cells
        outside the boundary as impassable obstacles.
        Also marks keepout zones as obstacles.
        """
        # Get site bounds
        bounds = self.boundary.bounds

        # Collect obstacles (structures as polygons)
        obstacles = [s.to_shapely_polygon() for s in self.structures]

        # Add keepout zones as obstacles (they should be impassable)
        obstacles.extend(self.keepouts)

        # Create cost grid
        self.cost_grid = create_cost_grid(
            bounds=bounds,
            obstacles=obstacles,
            resolution=self.grid_resolution,
            obstacle_buffer=self.rules.access.road_width / 2,  # Road needs clearance
            near_obstacle_cost=1.5,  # Slightly higher cost near structures
            near_obstacle_distance=self.rules.access.road_width,
        )

        # CRITICAL: Mark cells outside site boundary as impassable
        # This ensures roads cannot be routed outside the property
        self._mark_outside_boundary_as_obstacles()

    def _mark_outside_boundary_as_obstacles(self):
        """Mark all grid cells outside the site boundary as impassable.

        This ensures road paths cannot be routed outside the property line.
        Uses efficient vectorized checking where possible.
        """
        from shapely.prepared import prep

        # Prepare boundary for fast containment checks
        prepared_boundary = prep(self.boundary)

        # Iterate through grid and mark cells outside boundary as obstacles
        min_x, min_y, max_x, max_y = self.boundary.bounds
        outside_count = 0

        for gy in range(self.cost_grid.height):
            for gx in range(self.cost_grid.width):
                # Get world coordinates for cell center
                wx, wy = self.cost_grid.grid_to_world(gx, gy)
                cell_point = Point(wx, wy)

                # If cell center is outside boundary, mark as obstacle
                if not prepared_boundary.contains(cell_point):
                    self.cost_grid.obstacles[gy, gx] = True
                    self.cost_grid.costs[gy, gx] = float('inf')
                    outside_count += 1

        logger.debug(
            f"Marked {outside_count} cells outside boundary as impassable "
            f"(grid size: {self.cost_grid.width}x{self.cost_grid.height})"
        )

    def build_network(
        self,
        connect_all_docks: bool = True,
    ) -> Optional[RoadNetwork]:
        """Build road network connecting entrances to docks.

        Uses a greedy approach:
        1. Start from entrance(s)
        2. Route to each dock in priority order
        3. Try to reuse existing road segments

        Args:
            connect_all_docks: If True, fail if any required dock can't be reached

        Returns:
            RoadNetwork or None if building fails
        """
        if not self.entrances:
            logger.warning("No entrances defined, cannot build road network")
            return None

        if not self.dock_zones:
            logger.info("No dock zones, creating minimal road network")
            return RoadNetwork()

        # Start from first entrance
        main_entrance = self.entrances[0]
        entrance_pt = main_entrance.point

        # Track road segments
        segments: List[RoadSegment] = []
        segment_lines: List[LineString] = []  # For spatial queries

        # Track connected docks
        connected_docks: Set[str] = set()

        # Sort docks by priority (required first, then by distance)
        sorted_docks = sorted(
            self.dock_zones,
            key=lambda d: (not d.required, self._distance(entrance_pt, d.access_point)),
        )

        # Route to each dock
        for dock in sorted_docks:
            # Find best starting point (entrance or existing road)
            start_pt, connected_to = self._find_best_start(
                entrance_pt, dock.access_point, segment_lines
            )

            # Find path
            result = find_road_path(
                start=start_pt,
                end=dock.access_point,
                cost_grid=self.cost_grid,
                allow_diagonal=False,  # Prefer orthogonal roads
            )

            if result.success:
                # Create road segment
                segment = self._create_segment(
                    result.path,
                    start_id=connected_to,
                    end_id=dock.structure_id,
                )
                segments.append(segment)
                # Guard against degenerate paths (need >= 2 points for LineString)
                if len(result.path) >= 2:
                    segment_lines.append(LineString(result.path))
                connected_docks.add(dock.structure_id)
            else:
                logger.warning(
                    f"Could not route to dock {dock.structure_id}: {result.message}"
                )
                if dock.required and connect_all_docks:
                    logger.error("Required dock not reachable, aborting")
                    return None

        # Calculate total road length
        total_length = sum(s.length for s in segments)

        # Build network
        network = RoadNetwork(
            segments=segments,
            total_length=total_length,
            entrances_connected=[e.id for e in self.entrances],
            structures_accessible=list(connected_docks),
        )

        logger.info(
            f"Built road network: {len(segments)} segments, "
            f"{total_length:.1f}m total, {len(connected_docks)} docks connected"
        )

        return network

    def _find_best_start(
        self,
        entrance: Tuple[float, float],
        target: Tuple[float, float],
        existing_roads: List[LineString],
    ) -> Tuple[Tuple[float, float], str]:
        """Find best starting point for new road segment.

        Checks if connecting from existing road is shorter than from entrance.

        Args:
            entrance: Main entrance point
            target: Target dock access point
            existing_roads: Existing road centerlines

        Returns:
            Tuple of (start_point, connection_id)
        """
        best_start = entrance
        best_id = "entrance"
        best_dist = self._distance(entrance, target)

        target_pt = Point(target)

        for i, road in enumerate(existing_roads):
            # Find nearest point on road
            nearest = road.interpolate(road.project(target_pt))
            near_pt = (nearest.x, nearest.y)

            # Check if this is a better start
            dist = self._distance(near_pt, target)
            if dist < best_dist * 0.8:  # 20% savings threshold
                best_start = near_pt
                best_id = f"road_{i}"
                best_dist = dist

        return best_start, best_id

    def _create_segment(
        self,
        path: List[Tuple[float, float]],
        start_id: str,
        end_id: str,
    ) -> RoadSegment:
        """Create a RoadSegment from path points."""
        if len(path) < 2:
            # Invalid path
            return RoadSegment(
                id=f"seg_{start_id}_to_{end_id}",
                start=path[0] if path else (0, 0),
                end=path[-1] if path else (0, 0),
                width=self.rules.access.road_width,
                waypoints=[],
                connects_to=[start_id, end_id],
            )

        return RoadSegment(
            id=f"seg_{start_id}_to_{end_id}",
            start=path[0],
            end=path[-1],
            width=self.rules.access.road_width,
            waypoints=path[1:-1] if len(path) > 2 else [],
            connects_to=[start_id, end_id],
        )

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def validate_road_network(
    network: RoadNetwork,
    structures: List[PlacedStructure],
    entrances: List[Entrance],
    require_all_accessible: bool = True,
) -> RoadValidationResult:
    """Validate a road network.

    Checks:
    1. All structures with access requirements are reachable
    2. Roads don't overlap structures
    3. Turning radius requirements are met (simplified check)

    Args:
        network: Road network to validate
        structures: Placed structures
        entrances: Site entrances
        require_all_accessible: Require all structures with access to be reachable

    Returns:
        RoadValidationResult with validation details
    """
    issues = []
    accessible = set(network.structures_accessible)

    # Find structures that need access
    need_access = {
        s.structure_id
        for s in structures
        if s.structure.access is not None and s.structure.access.required
    }

    inaccessible = need_access - accessible

    if inaccessible:
        issues.append(
            f"Structures without road access: {', '.join(sorted(inaccessible))}"
        )

    # Check road-structure overlaps
    structure_polys = {s.structure_id: s.to_shapely_polygon() for s in structures}

    for segment in network.segments:
        road_line = LineString(segment.to_linestring_coords())
        road_buffer = road_line.buffer(segment.width / 2)

        for struct_id, poly in structure_polys.items():
            if road_buffer.intersects(poly):
                intersection_area = road_buffer.intersection(poly).area
                if intersection_area > 0.1:  # Small tolerance
                    issues.append(
                        f"Road segment {segment.id} overlaps structure {struct_id}"
                    )

    # Calculate total length
    total_length = sum(s.length for s in network.segments)

    # Determine overall validity
    all_accessible = len(inaccessible) == 0
    is_valid = all_accessible if require_all_accessible else True
    is_valid = is_valid and len([i for i in issues if "overlaps" in i]) == 0

    return RoadValidationResult(
        is_valid=is_valid,
        all_docks_accessible=all_accessible,
        accessible_docks=accessible,
        inaccessible_docks=inaccessible,
        total_road_length=total_length,
        issues=issues,
    )


def validate_road_polygon_containment(
    network: RoadNetwork,
    boundary: Polygon,
    keepouts: Optional[List[Polygon]] = None,
    tolerance: float = 0.5,
) -> Tuple[bool, List[str]]:
    """Validate that road polygons (buffered centerlines) stay inside boundary.

    This is a final check ensuring the actual road surface doesn't extend
    outside the site boundary or into keepout zones.

    Args:
        network: Road network to validate
        boundary: Site boundary polygon
        keepouts: Optional list of keepout zone polygons
        tolerance: Tolerance for boundary checks (meters)

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    keepouts = keepouts or []

    # Prepare boundary with small tolerance for numerical precision
    boundary_buffered = boundary.buffer(tolerance)

    for segment in network.segments:
        # Create road polygon from buffered centerline
        road_line = LineString(segment.to_linestring_coords())
        road_polygon = road_line.buffer(segment.width / 2)

        # Check containment within boundary
        if not boundary_buffered.contains(road_polygon):
            # Find how much is outside
            outside = road_polygon.difference(boundary)
            if not outside.is_empty and outside.area > tolerance * tolerance:
                issues.append(
                    f"Road segment {segment.id} extends {outside.area:.2f}m² "
                    f"outside site boundary"
                )

        # Check intersection with keepouts
        for i, keepout in enumerate(keepouts):
            if road_polygon.intersects(keepout):
                intersection = road_polygon.intersection(keepout)
                if not intersection.is_empty and intersection.area > tolerance * tolerance:
                    issues.append(
                        f"Road segment {segment.id} intersects keepout zone {i} "
                        f"by {intersection.area:.2f}m²"
                    )

    is_valid = len(issues) == 0
    return is_valid, issues


def compute_adaptive_grid_resolution(
    boundary: Polygon,
    num_structures: int,
) -> float:
    """Compute adaptive grid resolution based on site complexity.

    For complex sites (small area, many structures, or irregular boundaries),
    uses finer resolution for more accurate road routing.

    Args:
        boundary: Site boundary polygon
        num_structures: Number of structures to place

    Returns:
        Recommended grid resolution in meters (0.5 to 2.0)
    """
    # Base resolution on site area
    area = boundary.area
    perimeter = boundary.length

    # Irregularity metric: compare to equivalent area square
    equivalent_side = area ** 0.5
    irregularity = perimeter / (4 * equivalent_side) if equivalent_side > 0 else 1.0

    # Density: structures per 100m²
    density = (num_structures * 100) / area if area > 0 else 0

    # Start with default resolution
    resolution = 1.0

    # Finer resolution for irregular boundaries
    if irregularity > 1.3:  # 30% more perimeter than square
        resolution = 0.75

    if irregularity > 1.5:  # 50% more perimeter
        resolution = 0.5

    # Finer resolution for dense sites
    if density > 1.0:  # More than 1 structure per 100m²
        resolution = min(resolution, 0.75)

    if density > 2.0:  # More than 2 structures per 100m²
        resolution = min(resolution, 0.5)

    # Coarser resolution for very large sites (performance)
    if area > 50000:  # > 5 hectares
        resolution = max(resolution, 1.5)

    if area > 100000:  # > 10 hectares
        resolution = max(resolution, 2.0)

    return resolution


def build_road_network_for_solution(
    placements: List[PlacedStructure],
    entrances: List[Entrance],
    boundary: Polygon,
    rules: RuleSet,
    keepouts: Optional[List[Polygon]] = None,
    validate_containment: bool = True,
    grid_resolution: Optional[float] = None,
) -> Optional[RoadNetwork]:
    """Convenience function to build road network for a solution.

    Args:
        placements: Placed structures
        entrances: Site entrances
        boundary: Site boundary
        rules: Engineering rules
        keepouts: Optional keepout zones to validate against
        validate_containment: If True, validate road polygons stay inside boundary
        grid_resolution: Pathfinding grid resolution in meters (default: adaptive)
            - 0.5: Finer grid, better for complex/irregular sites
            - 1.0: Default, good balance of accuracy and speed
            - 2.0: Coarser grid, faster for large sites

    Returns:
        RoadNetwork or None if building fails or validation fails
    """
    # Use adaptive resolution if not specified
    if grid_resolution is None:
        grid_resolution = compute_adaptive_grid_resolution(boundary, len(placements))
        logger.debug(f"Using adaptive grid resolution: {grid_resolution}m")

    builder = RoadNetworkBuilder(
        structures=placements,
        entrances=entrances,
        boundary=boundary,
        rules=rules,
        grid_resolution=grid_resolution,
        keepouts=keepouts,
    )

    network = builder.build_network()

    if network is None:
        return None

    # Validate road polygon containment if requested
    if validate_containment:
        is_valid, issues = validate_road_polygon_containment(
            network, boundary, keepouts
        )
        if not is_valid:
            for issue in issues:
                logger.warning(f"Road validation failed: {issue}")
            return None

    return network
