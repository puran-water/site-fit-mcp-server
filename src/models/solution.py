"""Solution and output models for site-fit results."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class RoadSegment(BaseModel):
    """A segment of the road network."""

    id: str = Field(..., description="Segment identifier")
    start: tuple[float, float] = Field(..., description="Start point [x, y]")
    end: tuple[float, float] = Field(..., description="End point [x, y]")
    width: float = Field(default=6.0, description="Road width in meters")
    waypoints: list[tuple[float, float]] = Field(
        default_factory=list, description="Intermediate waypoints for curved roads"
    )
    connects_to: list[str] = Field(
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

    def to_linestring_coords(self) -> list[tuple[float, float]]:
        """Get coordinates as LineString format for GeoJSON."""
        return [self.start] + self.waypoints + [self.end]


class RoadNetwork(BaseModel):
    """Complete road network for a site layout solution."""

    segments: list[RoadSegment] = Field(default_factory=list, description="Road segments")
    total_length: float = Field(default=0.0, description="Total road length in meters")
    entrances_connected: list[str] = Field(
        default_factory=list, description="IDs of site entrances connected"
    )
    structures_accessible: list[str] = Field(
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

    # ROM metrics (new)
    pipe_length_by_type: dict[str, float] = Field(
        default_factory=dict,
        description="Pipe length breakdown by type: {gravity, pressure, gas, sludge}"
    )
    road_area_m2: float = Field(
        default=0.0, ge=0, description="Total road surface area in square meters"
    )
    max_dead_end_length: float = Field(
        default=0.0, ge=0, description="Longest dead-end road segment in meters"
    )
    intersection_count: int = Field(
        default=0, ge=0, description="Number of road intersections (3+ way)"
    )
    min_throat_width: float | None = Field(
        default=None, ge=0, description="Narrowest passage between structures in meters"
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

    # Service envelope metrics (soft penalties)
    envelope_overlap_count: int = Field(
        default=0, ge=0, description="Number of service envelope overlaps"
    )
    envelope_overlap_area_m2: float = Field(
        default=0.0, ge=0, description="Total overlapping envelope area in square meters"
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

    def overall_score(self, weights: dict[str, float] | None = None) -> float:
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

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get axis-aligned bounding box (min_x, min_y, max_x, max_y)."""
        half_w, half_h = self.width / 2, self.height / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )

    def to_geojson_geometry(self) -> dict[str, Any]:
        """Convert to GeoJSON geometry with rotation applied.

        For circles: 32-point polygon approximation (rotation has no effect)
        For rectangles: 5-point polygon with rotation applied for non-90° multiples

        Note: For 90° multiples (0, 90, 180, 270), width/height are already
        rotation-adjusted (swapped for 90/270), so no rotation matrix needed.
        Only apply rotation matrix for arbitrary angles like 45°.
        """
        import math

        if self.is_circle:
            # Create circular polygon approximation (rotation doesn't affect circles)
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
            # Rectangle - width/height already account for 90° multiples
            half_w, half_h = self.width / 2, self.height / 2

            # Base corners relative to center
            corners = [
                (-half_w, -half_h),
                (half_w, -half_h),
                (half_w, half_h),
                (-half_w, half_h),
            ]

            # Only apply rotation matrix for non-90° multiples (e.g., 45°)
            # For 0, 90, 180, 270: width/height swap already handles orientation
            if self.rotation_deg % 90 != 0:
                angle_rad = math.radians(self.rotation_deg)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                # Rotation matrix: [cos θ, -sin θ; sin θ, cos θ]
                corners = [
                    (cx * cos_a - cy * sin_a, cx * sin_a + cy * cos_a)
                    for cx, cy in corners
                ]

            # Translate to placement center and format for GeoJSON
            coords = [[self.x + cx, self.y + cy] for cx, cy in corners]
            coords.append(coords[0])  # Close ring

            return {"type": "Polygon", "coordinates": [coords]}


class SiteFitSolution(BaseModel):
    """Complete site layout solution."""

    id: str = Field(..., description="Unique solution identifier")
    job_id: str = Field(..., description="Parent job identifier")
    rank: int = Field(default=0, ge=0, description="Solution rank (0 = best)")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Core solution data
    placements: list[Placement] = Field(..., description="Structure placements")
    road_network: RoadNetwork | None = Field(default=None, description="Road network solution")
    metrics: SolutionMetrics = Field(default_factory=SolutionMetrics)

    # GeoJSON export
    features_geojson: dict[str, Any] | None = Field(
        default=None, description="GeoJSON FeatureCollection for visualization"
    )

    # Explanation for diversity
    diversity_note: str | None = Field(
        default=None, description="Why this solution differs from others"
    )

    def get_placement(self, structure_id: str) -> Placement | None:
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
    ) -> dict[str, Any]:
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
