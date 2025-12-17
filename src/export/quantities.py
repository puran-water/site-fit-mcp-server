"""Quantity takeoff computations for site layout solutions.

Generates ROM (Rough Order of Magnitude) quantities for civil engineering
handoff and cost estimation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import csv
import io
import math

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from ..models.solution import SiteFitSolution, RoadNetwork, Placement


@dataclass
class QuantityTakeoff:
    """ROM quantity takeoff data from a site layout solution."""

    # Pad/structure areas by type
    pad_area_by_type: Dict[str, float] = field(default_factory=dict)
    total_pad_area_m2: float = 0.0

    # Road quantities
    road_length_m: float = 0.0
    road_area_m2: float = 0.0
    road_intersection_count: int = 0
    max_dead_end_length_m: float = 0.0

    # Pipe proxies (from topology)
    pipe_length_by_type: Dict[str, float] = field(default_factory=dict)
    total_pipe_proxy_length_m: float = 0.0

    # Site boundary
    fence_perimeter_m: float = 0.0
    buildable_area_m2: float = 0.0
    site_utilization_pct: float = 0.0

    # Constructability indicators
    min_throat_width_m: float = float("inf")
    structure_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pad_area_by_type": self.pad_area_by_type,
            "total_pad_area_m2": round(self.total_pad_area_m2, 2),
            "road_length_m": round(self.road_length_m, 2),
            "road_area_m2": round(self.road_area_m2, 2),
            "road_intersection_count": self.road_intersection_count,
            "max_dead_end_length_m": round(self.max_dead_end_length_m, 2),
            "pipe_length_by_type": {k: round(v, 2) for k, v in self.pipe_length_by_type.items()},
            "total_pipe_proxy_length_m": round(self.total_pipe_proxy_length_m, 2),
            "fence_perimeter_m": round(self.fence_perimeter_m, 2),
            "buildable_area_m2": round(self.buildable_area_m2, 2),
            "site_utilization_pct": round(self.site_utilization_pct, 2),
            "min_throat_width_m": round(self.min_throat_width_m, 2) if self.min_throat_width_m != float("inf") else None,
            "structure_count": self.structure_count,
        }

    def to_csv_string(self) -> str:
        """Export quantities as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header section
        writer.writerow(["Site Layout Quantity Takeoff"])
        writer.writerow([])

        # Summary section
        writer.writerow(["Summary"])
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerow(["Total Pad Area", round(self.total_pad_area_m2, 2), "m2"])
        writer.writerow(["Road Length", round(self.road_length_m, 2), "m"])
        writer.writerow(["Road Area", round(self.road_area_m2, 2), "m2"])
        writer.writerow(["Pipe Proxy Length", round(self.total_pipe_proxy_length_m, 2), "m"])
        writer.writerow(["Fence Perimeter", round(self.fence_perimeter_m, 2), "m"])
        writer.writerow(["Site Utilization", round(self.site_utilization_pct, 2), "%"])
        writer.writerow(["Structure Count", self.structure_count, ""])
        writer.writerow([])

        # Pad areas by type
        writer.writerow(["Pad Areas by Equipment Type"])
        writer.writerow(["Equipment Type", "Area (m2)", "Count"])
        for eq_type, area in sorted(self.pad_area_by_type.items()):
            writer.writerow([eq_type, round(area, 2), ""])
        writer.writerow([])

        # Pipe lengths by type
        if self.pipe_length_by_type:
            writer.writerow(["Pipe Proxy Lengths by Type"])
            writer.writerow(["Pipe Type", "Length (m)"])
            for pipe_type, length in sorted(self.pipe_length_by_type.items()):
                writer.writerow([pipe_type, round(length, 2)])
            writer.writerow([])

        # Road metrics
        writer.writerow(["Road Network Metrics"])
        writer.writerow(["Metric", "Value", "Unit"])
        writer.writerow(["Intersection Count", self.road_intersection_count, ""])
        writer.writerow(["Max Dead End Length", round(self.max_dead_end_length_m, 2), "m"])
        if self.min_throat_width_m != float("inf"):
            writer.writerow(["Min Throat Width", round(self.min_throat_width_m, 2), "m"])

        return output.getvalue()


def compute_quantities(
    solution: SiteFitSolution,
    boundary: Optional[Polygon] = None,
    structure_types: Optional[Dict[str, str]] = None,
    topology_edges: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None,
    buildable_area_m2: float = 0.0,
) -> QuantityTakeoff:
    """Compute quantity takeoff from a site layout solution.

    Args:
        solution: The site layout solution
        boundary: Site boundary polygon for fence perimeter
        structure_types: Mapping of structure_id to equipment type
        topology_edges: List of (from_id, to_id, metadata) for pipe routing
        buildable_area_m2: Buildable area for utilization calculation

    Returns:
        QuantityTakeoff with computed quantities
    """
    structure_types = structure_types or {}
    takeoff = QuantityTakeoff()

    # Compute pad areas
    takeoff = _compute_pad_areas(takeoff, solution.placements, structure_types)

    # Compute road quantities
    if solution.road_network:
        takeoff = _compute_road_quantities(takeoff, solution.road_network)

    # Compute pipe proxies
    if topology_edges:
        takeoff = _compute_pipe_proxies(takeoff, solution.placements, topology_edges)

    # Compute boundary metrics
    if boundary:
        takeoff.fence_perimeter_m = boundary.exterior.length
        takeoff.buildable_area_m2 = buildable_area_m2 or boundary.area

    # Compute site utilization
    if takeoff.buildable_area_m2 > 0:
        takeoff.site_utilization_pct = (takeoff.total_pad_area_m2 / takeoff.buildable_area_m2) * 100

    # Compute min throat width
    takeoff = _compute_min_throat_width(takeoff, solution.placements)

    takeoff.structure_count = len(solution.placements)

    return takeoff


def _compute_pad_areas(
    takeoff: QuantityTakeoff,
    placements: List[Placement],
    structure_types: Dict[str, str],
) -> QuantityTakeoff:
    """Compute pad/structure areas by equipment type."""
    area_by_type: Dict[str, float] = {}
    total_area = 0.0

    for p in placements:
        eq_type = structure_types.get(p.structure_id, p.structure_id.split("-")[0] if "-" in p.structure_id else "unknown")

        if p.is_circle:
            area = math.pi * (p.width / 2) ** 2
        else:
            area = p.width * p.height

        area_by_type[eq_type] = area_by_type.get(eq_type, 0.0) + area
        total_area += area

    takeoff.pad_area_by_type = area_by_type
    takeoff.total_pad_area_m2 = total_area
    return takeoff


def _compute_road_quantities(
    takeoff: QuantityTakeoff,
    road_network: RoadNetwork,
) -> QuantityTakeoff:
    """Compute road network quantities."""
    takeoff.road_length_m = road_network.total_length

    # Compute road area by buffering centerlines
    road_polygons = []
    endpoint_counts: Dict[Tuple[float, float], int] = {}

    for seg in road_network.segments:
        coords = seg.to_linestring_coords()
        if len(coords) >= 2:
            line = LineString(coords)
            buffered = line.buffer(seg.width / 2, cap_style=2)  # Flat caps
            road_polygons.append(buffered)

            # Count endpoints for intersection detection
            start = (round(coords[0][0], 1), round(coords[0][1], 1))
            end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
            endpoint_counts[start] = endpoint_counts.get(start, 0) + 1
            endpoint_counts[end] = endpoint_counts.get(end, 0) + 1

    if road_polygons:
        combined = unary_union(road_polygons)
        takeoff.road_area_m2 = combined.area

    # Count intersections (nodes with 3+ connections)
    takeoff.road_intersection_count = sum(1 for count in endpoint_counts.values() if count >= 3)

    # Find max dead end length (endpoints with only 1 connection)
    max_dead_end = 0.0
    for seg in road_network.segments:
        coords = seg.to_linestring_coords()
        if len(coords) >= 2:
            start = (round(coords[0][0], 1), round(coords[0][1], 1))
            end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
            if endpoint_counts.get(start, 0) == 1 or endpoint_counts.get(end, 0) == 1:
                max_dead_end = max(max_dead_end, seg.length)

    takeoff.max_dead_end_length_m = max_dead_end

    return takeoff


def _compute_pipe_proxies(
    takeoff: QuantityTakeoff,
    placements: List[Placement],
    topology_edges: List[Tuple[str, str, Dict[str, Any]]],
) -> QuantityTakeoff:
    """Compute pipe proxy lengths from topology.

    Uses Manhattan distance between connected equipment as proxy for pipe length.
    """
    # Build placement lookup
    placement_map = {p.structure_id: p for p in placements}

    pipe_by_type: Dict[str, float] = {}
    total_pipe = 0.0

    for from_id, to_id, metadata in topology_edges:
        from_p = placement_map.get(from_id)
        to_p = placement_map.get(to_id)

        if from_p and to_p:
            # Manhattan distance as pipe length proxy
            dx = abs(to_p.x - from_p.x)
            dy = abs(to_p.y - from_p.y)
            dist = dx + dy

            # Classify pipe type from metadata
            pipe_type = metadata.get("type", "process")
            if "gravity" in pipe_type.lower() or "drain" in pipe_type.lower():
                pipe_type = "gravity"
            elif "gas" in pipe_type.lower() or "air" in pipe_type.lower():
                pipe_type = "gas"
            elif "sludge" in pipe_type.lower():
                pipe_type = "sludge"
            else:
                pipe_type = "pressure"

            pipe_by_type[pipe_type] = pipe_by_type.get(pipe_type, 0.0) + dist
            total_pipe += dist

    takeoff.pipe_length_by_type = pipe_by_type
    takeoff.total_pipe_proxy_length_m = total_pipe

    return takeoff


def _compute_min_throat_width(
    takeoff: QuantityTakeoff,
    placements: List[Placement],
) -> QuantityTakeoff:
    """Compute minimum throat width between adjacent structures.

    This is a constructability indicator - narrow passages make equipment access difficult.
    """
    min_width = float("inf")

    # Check all pairs for minimum gap
    for i, p1 in enumerate(placements):
        for p2 in placements[i + 1:]:
            # Get bounding boxes
            b1 = p1.get_bounds()
            b2 = p2.get_bounds()

            # Check X gap (when Y ranges overlap)
            y_overlap = not (b1[3] < b2[1] or b2[3] < b1[1])
            if y_overlap:
                x_gap = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
                if x_gap == 0:
                    x_gap = abs(b2[0] - b1[2]) if b2[0] > b1[2] else abs(b1[0] - b2[2])
                if x_gap > 0:
                    min_width = min(min_width, x_gap)

            # Check Y gap (when X ranges overlap)
            x_overlap = not (b1[2] < b2[0] or b2[2] < b1[0])
            if x_overlap:
                y_gap = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                if y_gap == 0:
                    y_gap = abs(b2[1] - b1[3]) if b2[1] > b1[3] else abs(b1[1] - b2[3])
                if y_gap > 0:
                    min_width = min(min_width, y_gap)

    takeoff.min_throat_width_m = min_width

    return takeoff
