"""NFPA 820 hazardous area zone calculation and validation.

NFPA 820 - Standard for Fire Protection in Wastewater Treatment and Collection Facilities
defines hazardous area classifications for wastewater and biogas facilities.

Class I, Division 1: Ignitable concentrations of flammable gases/vapors exist under normal conditions
Class I, Division 2: Ignitable concentrations may exist under abnormal conditions

This module generates buffer zones around hazardous equipment and validates that
non-hazardous-rated equipment is placed outside these zones.
"""

from dataclasses import dataclass
from enum import Enum

from shapely.geometry import Point, Polygon, mapping
from shapely.ops import unary_union

from ..models.rules import RuleSet
from ..models.solution import Placement


class HazardZoneType(Enum):
    """NFPA 820 hazardous area classification."""

    CLASS_I_DIV_1 = "class_i_div_1"
    CLASS_I_DIV_2 = "class_i_div_2"


@dataclass
class HazardZone:
    """A computed hazard zone polygon."""

    zone_type: HazardZoneType
    source_equipment_id: str
    source_equipment_type: str
    polygon: Polygon
    radius: float  # Zone radius in meters
    vertical_extent: float  # Vertical extent above grade in meters

    def to_geojson_feature(self) -> dict:
        """Convert to GeoJSON Feature.

        Uses shapely.geometry.mapping() to correctly handle:
        - Interior rings (holes) in polygons (e.g., Division 2 annuli)
        - MultiPolygon geometries with proper coordinate nesting
        """
        # Use shapely's mapping for correct GeoJSON geometry
        geometry = mapping(self.polygon)

        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "feature_type": "hazard_zone",
                "zone_type": self.zone_type.value,
                "source_equipment_id": self.source_equipment_id,
                "source_equipment_type": self.source_equipment_type,
                "radius_m": self.radius,
                "vertical_extent_m": self.vertical_extent,
                "nfpa820_classification": (
                    "Class I, Division 1" if self.zone_type == HazardZoneType.CLASS_I_DIV_1
                    else "Class I, Division 2"
                ),
            },
        }


def compute_hazard_zones(
    placements: list[Placement],
    rules: RuleSet,
    structure_types: dict[str, str] | None = None,
) -> list[HazardZone]:
    """Compute NFPA 820 hazard zones for all placed structures.

    Args:
        placements: List of equipment placements
        rules: RuleSet with NFPA 820 zone configurations
        structure_types: Optional mapping of structure_id to equipment type

    Returns:
        List of HazardZone objects for both Division 1 and Division 2 zones
    """
    structure_types = structure_types or {}
    zones: list[HazardZone] = []

    for placement in placements:
        # Determine equipment type
        equip_type = structure_types.get(
            placement.structure_id,
            getattr(placement, 'equipment_type', 'default')
        )

        # Get NFPA 820 zone config for this equipment type
        zone_config = rules.get_nfpa820_zone(equip_type)
        if zone_config is None:
            continue

        # Get equipment footprint (not just centroid) for accurate buffering
        # This ensures the hazard zone extends from the equipment perimeter,
        # not just the center point - critical for large rectangular equipment
        footprint = _get_placement_footprint(placement)

        # Create Division 1 zone if radius > 0
        if zone_config.class_i_div_1_radius > 0:
            # Buffer the actual footprint, not just the centroid
            div1_polygon = footprint.buffer(zone_config.class_i_div_1_radius)
            zones.append(HazardZone(
                zone_type=HazardZoneType.CLASS_I_DIV_1,
                source_equipment_id=placement.structure_id,
                source_equipment_type=equip_type,
                polygon=div1_polygon,
                radius=zone_config.class_i_div_1_radius,
                vertical_extent=zone_config.vertical_above,
            ))

        # Create Division 2 zone if radius > 0
        if zone_config.class_i_div_2_radius > 0:
            # Buffer the footprint for Division 2
            div2_polygon = footprint.buffer(zone_config.class_i_div_2_radius)
            # Division 2 is the annular ring outside Division 1
            if zone_config.class_i_div_1_radius > 0:
                div1_inner = footprint.buffer(zone_config.class_i_div_1_radius)
                div2_polygon = div2_polygon.difference(div1_inner)
            zones.append(HazardZone(
                zone_type=HazardZoneType.CLASS_I_DIV_2,
                source_equipment_id=placement.structure_id,
                source_equipment_type=equip_type,
                polygon=div2_polygon,
                radius=zone_config.class_i_div_2_radius,
                vertical_extent=zone_config.vertical_above,
            ))

    return zones


def _get_placement_centroid(placement: Placement) -> tuple[float, float]:
    """Get the centroid of a placement.

    Note: Placement.x and Placement.y are already center coordinates.
    """
    return (placement.x, placement.y)


def _get_placement_footprint(placement: Placement) -> Polygon:
    """Get the Shapely polygon footprint for a placement.

    Creates accurate polygons:
    - For circles: Circular buffer around center point (32-point approximation)
    - For rectangles: Axis-aligned bounding box (rotation applied if needed)

    Args:
        placement: Placement object with x, y, width, height, shape

    Returns:
        Shapely Polygon representing the structure footprint
    """
    import math

    if placement.is_circle:
        # Circular structure - create circular polygon
        radius = placement.width / 2
        return Point(placement.x, placement.y).buffer(radius, resolution=32)
    else:
        # Rectangular structure - create polygon with rotation
        half_w, half_h = placement.width / 2, placement.height / 2

        # Base corners relative to center
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]

        # Apply rotation matrix for non-90° multiples
        # (for 0, 90, 180, 270: width/height are already swapped)
        if placement.rotation_deg % 90 != 0:
            angle_rad = math.radians(placement.rotation_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            corners = [
                (cx * cos_a - cy * sin_a, cx * sin_a + cy * cos_a)
                for cx, cy in corners
            ]

        # Translate to placement center
        coords = [(placement.x + cx, placement.y + cy) for cx, cy in corners]
        coords.append(coords[0])  # Close the ring

        return Polygon(coords)


def get_combined_hazard_zones(
    zones: list[HazardZone],
    zone_type: HazardZoneType | None = None,
) -> Polygon | None:
    """Combine multiple hazard zones into a single polygon.

    Args:
        zones: List of HazardZone objects
        zone_type: Filter by zone type (None = all)

    Returns:
        Combined polygon or None if no zones
    """
    filtered = zones
    if zone_type is not None:
        filtered = [z for z in zones if z.zone_type == zone_type]

    if not filtered:
        return None

    polygons = [z.polygon for z in filtered]
    return unary_union(polygons)


@dataclass
class HazardZoneViolation:
    """A violation where non-hazardous equipment is in a hazard zone."""

    equipment_id: str
    equipment_type: str
    zone_type: HazardZoneType
    source_equipment_id: str
    distance_into_zone: float  # How far into the zone

    def __str__(self) -> str:
        return (
            f"NFPA 820 Violation: {self.equipment_id} ({self.equipment_type}) "
            f"is within {self.zone_type.value} zone from {self.source_equipment_id} "
            f"({self.distance_into_zone:.2f}m into zone)"
        )


def validate_hazard_zone_exclusions(
    placements: list[Placement],
    zones: list[HazardZone],
    rules: RuleSet,
    structure_types: dict[str, str] | None = None,
) -> list[HazardZoneViolation]:
    """Validate that excluded equipment types are outside hazard zones.

    Args:
        placements: All equipment placements
        zones: Computed hazard zones
        rules: RuleSet with exclusion list
        structure_types: Optional mapping of structure_id to equipment type

    Returns:
        List of violations where excluded equipment is in hazard zones
    """
    structure_types = structure_types or {}
    violations: list[HazardZoneViolation] = []

    # Build combined zone polygons by type
    div1_combined = get_combined_hazard_zones(zones, HazardZoneType.CLASS_I_DIV_1)
    div2_combined = get_combined_hazard_zones(zones, HazardZoneType.CLASS_I_DIV_2)

    for placement in placements:
        equip_type = structure_types.get(
            placement.structure_id,
            getattr(placement, 'equipment_type', 'default')
        )

        # Check if this equipment type must be excluded from hazard zones
        if not rules.is_hazard_zone_exclusion(equip_type):
            continue

        centroid = Point(_get_placement_centroid(placement))

        # Check Division 1 zones (most restrictive)
        if div1_combined and div1_combined.contains(centroid):
            # Find which source zone this is from
            for zone in zones:
                if zone.zone_type == HazardZoneType.CLASS_I_DIV_1:
                    if zone.polygon.contains(centroid):
                        violations.append(HazardZoneViolation(
                            equipment_id=placement.structure_id,
                            equipment_type=equip_type,
                            zone_type=HazardZoneType.CLASS_I_DIV_1,
                            source_equipment_id=zone.source_equipment_id,
                            distance_into_zone=zone.radius - centroid.distance(
                                Point(_get_placement_centroid_from_zone(zone))
                            ),
                        ))
                        break

        # Check Division 2 zones
        if div2_combined and div2_combined.contains(centroid):
            for zone in zones:
                if zone.zone_type == HazardZoneType.CLASS_I_DIV_2:
                    if zone.polygon.contains(centroid):
                        violations.append(HazardZoneViolation(
                            equipment_id=placement.structure_id,
                            equipment_type=equip_type,
                            zone_type=HazardZoneType.CLASS_I_DIV_2,
                            source_equipment_id=zone.source_equipment_id,
                            distance_into_zone=zone.radius - centroid.distance(
                                zone.polygon.centroid
                            ),
                        ))
                        break

    return violations


def _get_placement_centroid_from_zone(zone: HazardZone) -> tuple[float, float]:
    """Get centroid from zone polygon centroid."""
    return (zone.polygon.centroid.x, zone.polygon.centroid.y)
