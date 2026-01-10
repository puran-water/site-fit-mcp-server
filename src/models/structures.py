"""Structure footprint and placement models."""

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RectFootprint(BaseModel):
    """Rectangular structure footprint."""

    shape: Literal["rect"] = "rect"
    w: float = Field(..., gt=0, description="Width in meters (x-dimension)")
    h: float = Field(..., gt=0, description="Height in meters (y-dimension)")

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def perimeter(self) -> float:
        return 2 * (self.w + self.h)

    @property
    def long_side(self) -> float:
        return max(self.w, self.h)

    @property
    def short_side(self) -> float:
        return min(self.w, self.h)

    def get_dims_at_rotation(self, rotation_deg: int) -> tuple[float, float]:
        """Get width, height at given rotation (0, 90, 180, 270)."""
        if rotation_deg in (90, 270):
            return self.h, self.w
        return self.w, self.h


class CircleFootprint(BaseModel):
    """Circular structure footprint (tanks, digesters)."""

    shape: Literal["circle"] = "circle"
    d: float = Field(..., gt=0, description="Diameter in meters")

    @property
    def radius(self) -> float:
        return self.d / 2

    @property
    def area(self) -> float:
        return math.pi * self.radius**2

    @property
    def circumference(self) -> float:
        return math.pi * self.d

    def get_bounding_square(self, padding: float = 1.0) -> tuple[float, float]:
        """Get bounding square dimensions with optional padding factor."""
        side = self.d * padding
        return side, side


class AccessRequirement(BaseModel):
    """Vehicle access requirement for a structure."""

    vehicle: str = Field(
        ..., description="Vehicle type: dump_truck, tanker, forklift, crane, fire_truck"
    )
    dock_edge: Literal["long_side", "short_side", "any"] = Field(
        default="any", description="Which edge requires dock access"
    )
    dock_length: float = Field(
        default=15.0, gt=0, description="Required dock/apron length in meters"
    )
    dock_width: float = Field(
        default=6.0, gt=0, description="Required dock/apron width in meters"
    )
    required: bool = Field(default=True, description="Whether access is mandatory")
    turning_radius: float | None = Field(
        default=None, ge=0, description="Minimum turning radius for vehicle"
    )


class FixedPosition(BaseModel):
    """Fixed position for pinned structures."""

    x: float = Field(..., description="X coordinate of structure center")
    y: float = Field(..., description="Y coordinate of structure center")
    rotation_deg: int = Field(default=0, description="Rotation in degrees")

    @field_validator("rotation_deg")
    @classmethod
    def validate_rotation(cls, v: int) -> int:
        """Validate rotation is a valid value."""
        if v not in {0, 90, 180, 270}:
            raise ValueError(f"Rotation must be 0, 90, 180, or 270, got {v}")
        return v


class ServiceEnvelopes(BaseModel):
    """Service envelope requirements for maintenance and operation access.

    Service envelopes define additional clearance zones around equipment
    for maintenance access, crane access, and laydown areas.
    Overlapping envelopes incur a soft penalty (not a hard constraint).
    """

    maintenance_offset: float = Field(
        default=0.0, ge=0.0,
        description="Additional clearance for maintenance access (all sides)"
    )
    crane_access_edge: Literal["N", "S", "E", "W", "long", "short"] | None = Field(
        default=None,
        description="Edge requiring crane access strip"
    )
    crane_strip_width: float = Field(
        default=6.0, ge=0.0,
        description="Width of crane access strip in meters"
    )
    crane_strip_length: float = Field(
        default=20.0, ge=0.0,
        description="Length of crane access strip in meters"
    )
    laydown_area: tuple[float, float] | None = Field(
        default=None,
        description="Laydown area dimensions (width, length) adjacent to structure"
    )
    laydown_edge: Literal["N", "S", "E", "W", "long", "short"] | None = Field(
        default=None,
        description="Edge where laydown area should be placed"
    )


class StructureFootprint(BaseModel):
    """Complete structure definition with footprint and constraints."""

    id: str = Field(..., description="Unique structure identifier")
    type: str = Field(
        ..., description="Structure type from BFD: aeration_tank, digester, dewatering_building, etc."
    )
    footprint: RectFootprint | CircleFootprint = Field(
        ..., description="Footprint geometry"
    )
    orientations_deg: list[int] = Field(
        default=[0, 90, 180, 270],
        description="Allowed orientations in degrees (only applies to rectangles)"
    )
    height: float | None = Field(default=None, ge=0, description="Structure height in meters")
    dome_height_m: float | None = Field(
        default=None, ge=0,
        description="Dome/membrane cover height in meters (for digesters). "
                    "The 'height' parameter represents SHELL height; dome is added on top. "
                    "If not provided, FreeCAD uses DOME_RATIO * diameter as fallback."
    )
    access: AccessRequirement | None = Field(
        default=None, description="Vehicle access requirement"
    )
    equipment_tag: str | None = Field(
        default=None, description="Equipment tag from BFD (e.g., 230-AS-01)"
    )
    area_number: int | None = Field(
        default=None, description="Process area number from BFD hierarchy"
    )

    # Pinned placement support (Tier 2)
    pinned: bool = Field(
        default=False,
        description="If true, structure must be placed at fixed_position"
    )
    fixed_position: FixedPosition | None = Field(
        default=None,
        description="Fixed position for pinned structures"
    )
    allowed_zone: list[list[float]] | None = Field(
        default=None,
        description="Constraint polygon limiting placement area [[x,y], ...]"
    )

    # Service envelope support (Tier 2)
    service_envelopes: ServiceEnvelopes | None = Field(
        default=None,
        description="Service envelope requirements for maintenance/crane access"
    )

    @field_validator("orientations_deg")
    @classmethod
    def validate_orientations(cls, v: list[int]) -> list[int]:
        """Validate orientation values."""
        valid = {0, 90, 180, 270}
        for o in v:
            if o not in valid:
                raise ValueError(f"Orientation must be 0, 90, 180, or 270, got {o}")
        return sorted(set(v))

    @property
    def is_circle(self) -> bool:
        return isinstance(self.footprint, CircleFootprint)

    @property
    def is_rect(self) -> bool:
        return isinstance(self.footprint, RectFootprint)

    def get_bounding_dims(self, padding: float = 1.0) -> tuple[float, float]:
        """Get bounding box dimensions.

        For circles, returns square bounding box.
        For rectangles, returns max dimensions across all rotations.
        """
        if self.is_circle:
            return self.footprint.get_bounding_square(padding)  # type: ignore
        else:
            # For rectangles, max dims to cover all rotations
            fp = self.footprint  # type: ignore
            max_dim = max(fp.w, fp.h) * padding
            return max_dim, max_dim

    def get_area(self) -> float:
        """Get footprint area in square meters."""
        return self.footprint.area


class PlacedStructure(BaseModel):
    """A structure with assigned position and orientation."""

    structure: StructureFootprint = Field(..., description="Structure definition")
    x: float = Field(..., description="X coordinate of structure center")
    y: float = Field(..., description="Y coordinate of structure center")
    rotation_deg: int = Field(default=0, description="Rotation in degrees (0, 90, 180, 270)")

    @model_validator(mode="after")
    def validate_rotation(self) -> "PlacedStructure":
        """Validate rotation is allowed for this structure."""
        if self.structure.is_circle:
            # Circles ignore rotation
            object.__setattr__(self, "rotation_deg", 0)
        elif self.rotation_deg not in self.structure.orientations_deg:
            raise ValueError(
                f"Rotation {self.rotation_deg} not in allowed orientations "
                f"{self.structure.orientations_deg}"
            )
        return self

    @property
    def id(self) -> str:
        return self.structure.id

    @property
    def type(self) -> str:
        return self.structure.type

    def get_current_dims(self) -> tuple[float, float]:
        """Get width, height at current rotation."""
        if self.structure.is_circle:
            fp = self.structure.footprint  # type: ignore
            return fp.d, fp.d
        else:
            fp = self.structure.footprint  # type: ignore
            return fp.get_dims_at_rotation(self.rotation_deg)

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get axis-aligned bounding box (min_x, min_y, max_x, max_y)."""
        w, h = self.get_current_dims()
        half_w, half_h = w / 2, h / 2
        return (
            self.x - half_w,
            self.y - half_h,
            self.x + half_w,
            self.y + half_h,
        )

    def get_center(self) -> tuple[float, float]:
        """Get center point."""
        return self.x, self.y

    @property
    def structure_id(self) -> str:
        """Get structure ID (alias for clearance module compatibility)."""
        return self.structure.id

    @property
    def equipment_type(self) -> str:
        """Get equipment type for clearance calculations."""
        return self.structure.type

    @property
    def width(self) -> float:
        """Get current width at rotation."""
        return self.get_current_dims()[0]

    @property
    def height(self) -> float:
        """Get current height at rotation."""
        return self.get_current_dims()[1]

    def to_shapely_polygon(self):
        """Convert to Shapely Polygon for geometry operations.

        Returns:
            shapely.geometry.Polygon
        """
        from shapely.geometry import Point, Polygon

        if self.structure.is_circle:
            # Create circular polygon
            fp = self.structure.footprint  # type: ignore
            center = Point(self.x, self.y)
            return center.buffer(fp.d / 2, resolution=32)
        else:
            # Create rectangular polygon
            w, h = self.get_current_dims()
            half_w, half_h = w / 2, h / 2
            coords = [
                (self.x - half_w, self.y - half_h),
                (self.x + half_w, self.y - half_h),
                (self.x + half_w, self.y + half_h),
                (self.x - half_w, self.y + half_h),
                (self.x - half_w, self.y - half_h),  # Close ring
            ]
            return Polygon(coords)
