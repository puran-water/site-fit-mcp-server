"""A* pathfinding for road network routing."""

import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


@dataclass
class PathfinderResult:
    """Result from pathfinding."""

    success: bool
    path: List[Tuple[float, float]]  # List of waypoints
    cost: float
    length: float  # Path length in meters
    message: Optional[str] = None

    def to_linestring(self) -> Optional[LineString]:
        """Convert path to Shapely LineString."""
        if not self.path or len(self.path) < 2:
            return None
        return LineString(self.path)


@dataclass
class GridCell:
    """A cell in the cost grid."""

    x: int
    y: int
    cost: float  # Movement cost (high near obstacles)
    is_obstacle: bool = False


class CostGrid:
    """2D grid for A* pathfinding with obstacle awareness."""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],  # min_x, min_y, max_x, max_y
        resolution: float = 1.0,  # Grid cell size in meters
        base_cost: float = 1.0,
        obstacle_cost: float = float('inf'),
    ):
        """Initialize cost grid.

        Args:
            bounds: World coordinate bounds
            resolution: Grid cell size in meters
            base_cost: Base movement cost per cell
            obstacle_cost: Cost for obstacle cells (inf = impassable)
        """
        self.bounds = bounds
        self.resolution = resolution
        self.base_cost = base_cost
        self.obstacle_cost = obstacle_cost

        min_x, min_y, max_x, max_y = bounds
        self.width = int(np.ceil((max_x - min_x) / resolution))
        self.height = int(np.ceil((max_y - min_y) / resolution))

        # Initialize cost array (all cells passable)
        self.costs = np.full((self.height, self.width), base_cost, dtype=float)
        self.obstacles = np.zeros((self.height, self.width), dtype=bool)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        min_x, min_y, _, _ = self.bounds
        gx = int((x - min_x) / self.resolution)
        gy = int((y - min_y) / self.resolution)
        return (
            max(0, min(gx, self.width - 1)),
            max(0, min(gy, self.height - 1)),
        )

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        min_x, min_y, _, _ = self.bounds
        x = min_x + (gx + 0.5) * self.resolution
        y = min_y + (gy + 0.5) * self.resolution
        return (x, y)

    def add_obstacle(self, polygon: Polygon, buffer: float = 0.0):
        """Mark cells covered by polygon as obstacles.

        Args:
            polygon: Obstacle polygon in world coordinates
            buffer: Additional buffer around obstacle
        """
        if buffer > 0:
            polygon = polygon.buffer(buffer)

        # Get polygon bounds
        poly_bounds = polygon.bounds
        if not poly_bounds:
            return

        pmin_x, pmin_y, pmax_x, pmax_y = poly_bounds

        # Convert to grid bounds
        gx1, gy1 = self.world_to_grid(pmin_x, pmin_y)
        gx2, gy2 = self.world_to_grid(pmax_x, pmax_y)

        # Mark cells inside polygon
        for gy in range(gy1, min(gy2 + 1, self.height)):
            for gx in range(gx1, min(gx2 + 1, self.width)):
                wx, wy = self.grid_to_world(gx, gy)
                if polygon.contains(Point(wx, wy)):
                    self.obstacles[gy, gx] = True
                    self.costs[gy, gx] = self.obstacle_cost

    def add_cost_zone(self, polygon: Polygon, cost_multiplier: float):
        """Add cost multiplier to cells in polygon (soft avoidance).

        Args:
            polygon: Zone polygon
            cost_multiplier: Multiply base cost by this factor
        """
        poly_bounds = polygon.bounds
        if not poly_bounds:
            return

        pmin_x, pmin_y, pmax_x, pmax_y = poly_bounds
        gx1, gy1 = self.world_to_grid(pmin_x, pmin_y)
        gx2, gy2 = self.world_to_grid(pmax_x, pmax_y)

        for gy in range(gy1, min(gy2 + 1, self.height)):
            for gx in range(gx1, min(gx2 + 1, self.width)):
                if not self.obstacles[gy, gx]:
                    wx, wy = self.grid_to_world(gx, gy)
                    if polygon.contains(Point(wx, wy)):
                        self.costs[gy, gx] *= cost_multiplier

    def get_cost(self, gx: int, gy: int) -> float:
        """Get movement cost at grid cell."""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.costs[gy, gx]
        return self.obstacle_cost

    def is_passable(self, gx: int, gy: int) -> bool:
        """Check if grid cell is passable."""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return not self.obstacles[gy, gx]
        return False


def create_cost_grid(
    bounds: Tuple[float, float, float, float],
    obstacles: List[Polygon],
    resolution: float = 1.0,
    obstacle_buffer: float = 1.0,
    near_obstacle_cost: float = 2.0,
    near_obstacle_distance: float = 3.0,
) -> CostGrid:
    """Create a cost grid with obstacles.

    Args:
        bounds: World coordinate bounds
        obstacles: List of obstacle polygons
        resolution: Grid resolution in meters
        obstacle_buffer: Buffer around obstacles
        near_obstacle_cost: Cost multiplier near obstacles
        near_obstacle_distance: Distance for near-obstacle zone

    Returns:
        Configured CostGrid
    """
    grid = CostGrid(bounds, resolution)

    # Add obstacles with buffer
    for obs in obstacles:
        grid.add_obstacle(obs, buffer=obstacle_buffer)

        # Add cost zone near obstacles
        if near_obstacle_distance > 0:
            near_zone = obs.buffer(near_obstacle_distance)
            # Don't increase cost of actual obstacle cells
            near_zone = near_zone.difference(obs.buffer(obstacle_buffer))
            if not near_zone.is_empty:
                grid.add_cost_zone(near_zone, near_obstacle_cost)

    logger.info(
        f"Created cost grid: {grid.width}x{grid.height}, "
        f"{np.sum(grid.obstacles)} obstacle cells"
    )
    return grid


def find_road_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    cost_grid: CostGrid,
    allow_diagonal: bool = False,
) -> PathfinderResult:
    """Find path from start to end using A* algorithm.

    Args:
        start: Start point in world coordinates
        end: End point in world coordinates
        cost_grid: CostGrid with obstacles
        allow_diagonal: Allow diagonal movement

    Returns:
        PathfinderResult with path or failure message
    """
    # Convert to grid coordinates
    start_grid = cost_grid.world_to_grid(*start)
    end_grid = cost_grid.world_to_grid(*end)

    # Check start/end validity
    if not cost_grid.is_passable(*start_grid):
        return PathfinderResult(
            success=False,
            path=[],
            cost=float('inf'),
            length=0.0,
            message="Start position is blocked",
        )

    if not cost_grid.is_passable(*end_grid):
        return PathfinderResult(
            success=False,
            path=[],
            cost=float('inf'),
            length=0.0,
            message="End position is blocked",
        )

    # A* pathfinding
    path_grid = _astar(start_grid, end_grid, cost_grid, allow_diagonal)

    if path_grid is None:
        return PathfinderResult(
            success=False,
            path=[],
            cost=float('inf'),
            length=0.0,
            message="No path found",
        )

    # Convert grid path to world coordinates
    path_world = [cost_grid.grid_to_world(gx, gy) for gx, gy in path_grid]

    # Ensure exact start and end points
    path_world[0] = start
    path_world[-1] = end

    # Simplify path (remove unnecessary waypoints)
    path_simplified = _simplify_path(path_world, cost_grid)

    # Calculate total cost and length
    total_cost = sum(
        cost_grid.get_cost(*cost_grid.world_to_grid(*pt))
        for pt in path_world
    )
    length = LineString(path_simplified).length if len(path_simplified) >= 2 else 0.0

    return PathfinderResult(
        success=True,
        path=path_simplified,
        cost=total_cost,
        length=length,
    )


def _astar(
    start: Tuple[int, int],
    end: Tuple[int, int],
    grid: CostGrid,
    allow_diagonal: bool,
) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding on grid.

    Args:
        start: Start cell (gx, gy)
        end: End cell (gx, gy)
        grid: Cost grid
        allow_diagonal: Allow diagonal moves

    Returns:
        List of grid cells forming path, or None if no path
    """
    # Handle trivial case: start equals end
    # Return 2-point path to ensure valid LineString creation
    if start == end:
        return [start, end]

    # Neighbors: 4-directional or 8-directional
    if allow_diagonal:
        neighbors = [
            (0, 1), (1, 0), (0, -1), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]
        diag_cost = 1.414  # sqrt(2)
    else:
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        diag_cost = 1.0

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Manhattan distance for 4-dir, Euclidean for 8-dir
        if allow_diagonal:
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Priority queue: (f_score, counter, cell)
    counter = 0
    open_set = [(heuristic(start, end), counter, start)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start: 0}
    closed_set: Set[Tuple[int, int]] = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == end:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        if current in closed_set:
            continue
        closed_set.add(current)

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if not grid.is_passable(*neighbor):
                continue
            if neighbor in closed_set:
                continue

            # Movement cost (diagonal vs orthogonal)
            move_cost = diag_cost if (dx != 0 and dy != 0) else 1.0
            cell_cost = grid.get_cost(*neighbor)
            tentative_g = g_score[current] + move_cost * cell_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, end)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    return None  # No path found


def _simplify_path(
    path: List[Tuple[float, float]],
    grid: CostGrid,
) -> List[Tuple[float, float]]:
    """Simplify path by removing unnecessary waypoints.

    Uses line-of-sight check to skip intermediate points.

    Args:
        path: Original path
        grid: Cost grid for obstacle checking

    Returns:
        Simplified path with fewer waypoints
    """
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    i = 0

    while i < len(path) - 1:
        # Try to skip to furthest visible point
        best_j = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if _has_line_of_sight(path[i], path[j], grid):
                best_j = j
                break

        simplified.append(path[best_j])
        i = best_j

    return simplified


def _has_line_of_sight(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    grid: CostGrid,
) -> bool:
    """Check if there's clear line of sight between two points.

    Args:
        p1: Start point
        p2: End point
        grid: Cost grid with obstacles

    Returns:
        True if line between points doesn't cross obstacles
    """
    line = LineString([p1, p2])
    num_samples = int(line.length / grid.resolution) + 1

    for i in range(num_samples + 1):
        t = i / max(1, num_samples)
        pt = line.interpolate(t, normalized=True)
        gx, gy = grid.world_to_grid(pt.x, pt.y)
        if not grid.is_passable(gx, gy):
            return False

    return True
