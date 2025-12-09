"""Site boundary and constraint models."""

from typing import List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator


class GeoJSONPoint(BaseModel):
    """GeoJSON Point geometry."""

    type: Literal["Point"] = "Point"
    coordinates: Tuple[float, float] = Field(..., description="[x, y] coordinates in meters")


class GeoJSONPolygon(BaseModel):
    """GeoJSON Polygon geometry.

    Coordinates are a list of linear rings (first is exterior, rest are holes).
    Each ring is a list of [x, y] coordinate pairs.
    """

    type: Literal["Polygon"] = "Polygon"
    coordinates: List[List[Tuple[float, float]]] = Field(
        ..., description="List of rings, each ring is a list of [x, y] coordinates"
    )

    @field_validator("coordinates")
    @classmethod
    def validate_rings(cls, v: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """Validate that at least one ring exists and rings are closed."""
        if not v:
            raise ValueError("Polygon must have at least one ring (exterior)")
        for ring in v:
            if len(ring) < 4:
                raise ValueError("Ring must have at least 4 points (closed polygon)")
            if ring[0] != ring[-1]:
                raise ValueError("Ring must be closed (first point == last point)")
        return v

    @property
    def exterior(self) -> List[Tuple[float, float]]:
        """Get exterior ring coordinates."""
        return self.coordinates[0]

    @property
    def interiors(self) -> List[List[Tuple[float, float]]]:
        """Get interior rings (holes) if any."""
        return self.coordinates[1:] if len(self.coordinates) > 1 else []


class Entrance(BaseModel):
    """Site entrance/gate for vehicle access."""

    id: str = Field(..., description="Unique identifier for entrance")
    point: Tuple[float, float] = Field(..., description="[x, y] coordinates of entrance center")
    width: float = Field(default=6.0, ge=3.0, le=20.0, description="Gate width in meters")
    direction: Optional[Literal["N", "S", "E", "W"]] = Field(
        default=None, description="Direction entrance faces (optional)"
    )

    @property
    def x(self) -> float:
        return self.point[0]

    @property
    def y(self) -> float:
        return self.point[1]


class Keepout(BaseModel):
    """No-build zone within site boundary.

    Represents wetlands, utilities, archaeological sites, etc.
    """

    id: str = Field(..., description="Unique identifier")
    geometry: GeoJSONPolygon = Field(..., description="Keepout zone boundary")
    reason: str = Field(
        ..., description="Reason for keepout: wetland, utility, archaeological, setback, etc."
    )
    buffer: float = Field(
        default=0.0, ge=0.0, description="Additional buffer distance around keepout in meters"
    )


class ExistingStructure(BaseModel):
    """Existing structure that cannot be moved.

    Used for brownfield sites with existing infrastructure.
    """

    id: str = Field(..., description="Unique identifier")
    footprint: GeoJSONPolygon = Field(..., description="Structure footprint polygon")
    height: Optional[float] = Field(default=None, ge=0.0, description="Structure height in meters")
    clearance_required: float = Field(
        default=3.0, ge=0.0, description="Minimum clearance from existing structure in meters"
    )
    is_tie_in_point: bool = Field(
        default=False, description="Whether this is a utility tie-in point"
    )


class SiteBoundary(BaseModel):
    """Complete site definition with boundary, entrances, and constraints.

    This is the main site configuration object passed to the solver.
    """

    units: Literal["m", "ft"] = Field(default="m", description="Coordinate units")
    boundary: GeoJSONPolygon = Field(..., description="Site perimeter boundary")
    entrances: List[Entrance] = Field(
        default_factory=list, description="Site entrances for vehicle access"
    )
    keepouts: List[Keepout] = Field(default_factory=list, description="No-build zones")
    existing: List[ExistingStructure] = Field(
        default_factory=list, description="Existing structures (brownfield)"
    )

    @field_validator("entrances")
    @classmethod
    def validate_entrances(cls, v: List[Entrance]) -> List[Entrance]:
        """Validate at least one entrance exists."""
        if not v:
            raise ValueError("Site must have at least one entrance")
        return v

    @property
    def has_keepouts(self) -> bool:
        return len(self.keepouts) > 0

    @property
    def is_brownfield(self) -> bool:
        return len(self.existing) > 0

    def get_entrance_by_id(self, entrance_id: str) -> Optional[Entrance]:
        """Get entrance by ID."""
        for entrance in self.entrances:
            if entrance.id == entrance_id:
                return entrance
        return None
