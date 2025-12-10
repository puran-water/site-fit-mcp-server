"""Solution and output models for site-fit results."""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
import json


class RoadSegment(BaseModel):
    """A segment of the road network."""

    id: str = Field(..., description="Segment identifier")
    start: Tuple[float, float] = Field(..., description="Start point [x, y]")
    end: Tuple[float, float] = Field(..., description="End point [x, y]")
    width: float = Field(default=6.0, description="Road width in meters")
    waypoints: List[Tuple[float, float]] = Field(
        default_factory=list, description="Intermediate waypoints for curved roads"
    )
    connects_to: List[str] = Field(
        default_factory=list, description="IDs of connected road segments or structures"
    )

    @property
    def length(self) -> float:
        """Calculate segment length including waypoints."""
        import math
        points = [self.start] + self.waypoints + [self.end]
        total = 0.0
        for i in range(len(points) - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def to_linestring_coords(self) -> List[Tuple[float, float]]:
        """Get coordinates as LineString format for GeoJSON."""
        return [self.start] + self.waypoints + [self.end]


class RoadNetwork(BaseModel):
    """Complete road network for a site layout solution."""

    segments: List[RoadSegment] = Field(default_factory=list, description="Road segments")
    total_length: float = Field(default=0.0, description="Total road length in meters")
    entrances_connected: List[str] = Field(
        default_factory=list, description="IDs of site entrances connected"
    )
    structures_accessible: List[str] = Field(
        default_factory=list, description="IDs of structures with road access"
    )

    @computed_field
    @property
    def coverage_ratio(self) -> float:
        """Ratio of structures accessible to total structures."""
        if not self.structures_accessible:
            return 0.0
        # This would need total structures count from context
        return 1.0  # Placeholder - validated solutions should have 100%


class SolutionMetrics(BaseModel):
    """Quantitative metrics for evaluating a site layout solution."""

    # Core metrics
    pipe_length_weighted: float = Field(
        default=0.0, description="Weighted sum of process connection lengths"
    )
    road_length: float = Field(default=0.0, description="Total road network length in meters")
    site_utilization: float = Field(
        default=0.0, ge=0, le=1, description="Ratio of used area to buildable area"
    )
    compactness: float = Field(
        default=0.0, ge=0, le=1, description="Convex hull efficiency (occupied/hull area)"
    )

    # Constraint satisfaction
    min_clearance_achieved: float = Field(
        default=0.0, ge=0, description="Minimum achieved clearance between structures"
    )
    clearance_violations: int = Field(
        default=0, ge=0, description="Number of clearance constraint violations"
    )
    access_violations: int = Field(
        default=0, ge=0, description="Number of structures without road access"
    )

    # Topology adherence
    topology_penalty: float = Field(
        default=0.0, ge=0, description="Penalty for violating process flow direction"
    )
    adjacency_score: float = Field(
        default=0.0, description="Score for adjacency preference satisfaction"
    )

    @computed_field
    @property
    def is_feasible(self) -> bool:
        """Check if solution is feasible (no hard constraint violations)."""
        return self.clearance_violations == 0 and self.access_violations == 0

    def overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute weighted overall score (lower is better)."""
        w = weights or {
            "pipe_length": 1.0,
            "road_length": 0.5,
            "compactness": -0.3,  # Higher compactness is better
            "topology_penalty": 2.0,
        }
        return (
            w.get("pipe_length", 0) * self.pipe_length_weighted
            + w.get("road_length", 0) * self.road_length
            + w.get("compactness", 0) * (1 - self.compactness)
            + w.get("topology_penalty", 0) * self.topology_penalty
        )


class Placement(BaseModel):
    """A single structure placement within a solution."""

    structure_id: str = Field(..., description="Structure ID from input")
    x: float = Field(..., description="X coordinate of structure center")
    y: float = Field(..., description="Y coordinate of structure center")
    rotation_deg: int = Field(default=0, description="Rotation in degrees")
    width: float = Field(..., description="Actual width at current rotation")
    height: float = Field(..., description="Actual height at current rotation")
    shape: str = Field(default="rect", description="Footprint shape: 'rect' or 'circle'")

    @property
    def is_circle(self) -> bool:
        """Check if this is a circular structure."""
        return self.shape == "circle"

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get axis-aligned bounding box (min_x, min_y, max_x, max_y)."""
        half_w, half_h = self.width / 2, self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )

    def to_geojson_geometry(self) -> Dict[str, Any]:
        """Convert to GeoJSON geometry (circle as 32-point polygon, rect as 5-point polygon)."""
        import math

        if self.is_circle:
            # Create circular polygon approximation
            radius = self.width / 2
            num_segments = 32
            coords = []
            for i in range(num_segments):
                angle = 2 * math.pi * i / num_segments
                px = self.x + radius * math.cos(angle)
                py = self.y + radius * math.sin(angle)
                coords.append([px, py])
            coords.append(coords[0])  # Close the ring
            return {"type": "Polygon", "coordinates": [coords]}
        else:
            # Rectangle
            half_w, half_h = self.width / 2, self.height / 2
            coords = [
                [self.x - half_w, self.y - half_h],
                [self.x + half_w, self.y - half_h],
                [self.x + half_w, self.y + half_h],
                [self.x - half_w, self.y + half_h],
                [self.x - half_w, self.y - half_h],  # Close ring
            ]
            return {"type": "Polygon", "coordinates": [coords]}


class SiteFitSolution(BaseModel):
    """Complete site layout solution."""

    id: str = Field(..., description="Unique solution identifier")
    job_id: str = Field(..., description="Parent job identifier")
    rank: int = Field(default=0, ge=0, description="Solution rank (0 = best)")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Core solution data
    placements: List[Placement] = Field(..., description="Structure placements")
    road_network: Optional[RoadNetwork] = Field(default=None, description="Road network solution")
    metrics: SolutionMetrics = Field(default_factory=SolutionMetrics)

    # GeoJSON export
    features_geojson: Optional[Dict[str, Any]] = Field(
        default=None, description="GeoJSON FeatureCollection for visualization"
    )

    # Explanation for diversity
    diversity_note: Optional[str] = Field(
        default=None, description="Why this solution differs from others"
    )

    def get_placement(self, structure_id: str) -> Optional[Placement]:
        """Get placement for a specific structure."""
        for p in self.placements:
            if p.structure_id == structure_id:
                return p
        return None

    def to_geojson_feature_collection(
        self,
        include_site: bool = True,
        include_roads: bool = True,
        include_labels: bool = True,
    ) -> Dict[str, Any]:
        """Generate GeoJSON FeatureCollection for this solution.

        If features_geojson is already computed, returns it.
        Otherwise generates minimal version from placements.
        """
        if self.features_geojson is not None:
            return self.features_geojson

        features = []

        # Structure placements as polygons (circles as true circular polygons)
        for p in self.placements:
            features.append({
                "type": "Feature",
                "geometry": p.to_geojson_geometry(),
                "properties": {
                    "id": p.structure_id,
                    "kind": "structure",
                    "shape": p.shape,
                    "x": p.x,
                    "y": p.y,
                    "rotation": p.rotation_deg,
                    "width": p.width,
                    "height": p.height,
                },
            })

        # Road network as LineStrings
        if include_roads and self.road_network:
            for seg in self.road_network.segments:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": seg.to_linestring_coords(),
                    },
                    "properties": {
                        "id": seg.id,
                        "kind": "road",
                        "width": seg.width,
                        "length": seg.length,
                    },
                })

        return {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "solution_id": self.id,
                "job_id": self.job_id,
                "rank": self.rank,
                "metrics": self.metrics.model_dump(),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize solution to JSON string."""
        return self.model_dump_json(indent=indent)
