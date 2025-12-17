"""Tests for footprint-based NFPA 820 hazard zone calculation.

Verifies that hazard zones are computed from equipment footprint perimeter,
not just the centroid - critical for accurate zones around large rectangular equipment.
"""

import math
import pytest
from shapely.geometry import Point, Polygon

from src.hazards.nfpa820_zones import (
    compute_hazard_zones,
    _get_placement_footprint,
    get_combined_hazard_zones,
    HazardZone,
    HazardZoneType,
)
from src.models.solution import Placement
from src.models.rules import RuleSet, NFPA820ZoneConfig


class TestGetPlacementFootprint:
    """Test footprint polygon generation from placements."""

    def test_circular_placement_footprint(self):
        """Circular placement generates circular polygon."""
        placement = Placement(
            structure_id="TK-001",
            x=50.0,
            y=50.0,
            width=10.0,
            height=10.0,
            rotation_deg=0,
            shape="circle",
        )

        footprint = _get_placement_footprint(placement)

        # Should be approximately circular
        assert footprint.is_valid
        # Area should be π * r² = π * 25 ≈ 78.54
        assert footprint.area == pytest.approx(math.pi * 25, rel=0.01)

        # Centroid at center
        assert footprint.centroid.x == pytest.approx(50.0, abs=0.1)
        assert footprint.centroid.y == pytest.approx(50.0, abs=0.1)

    def test_rectangular_placement_footprint(self):
        """Rectangular placement generates rectangular polygon."""
        placement = Placement(
            structure_id="BLD-001",
            x=50.0,
            y=50.0,
            width=20.0,
            height=10.0,
            rotation_deg=0,
            shape="rect",
        )

        footprint = _get_placement_footprint(placement)

        # Should be rectangular
        assert footprint.is_valid
        # Area should be 20 * 10 = 200
        assert footprint.area == pytest.approx(200.0, rel=0.01)

        # Bounding box
        minx, miny, maxx, maxy = footprint.bounds
        assert minx == pytest.approx(40.0, abs=0.1)  # 50 - 10
        assert maxx == pytest.approx(60.0, abs=0.1)  # 50 + 10
        assert miny == pytest.approx(45.0, abs=0.1)  # 50 - 5
        assert maxy == pytest.approx(55.0, abs=0.1)  # 50 + 5

    def test_rotated_rectangular_footprint(self):
        """90° rotated rectangle swaps width/height."""
        placement = Placement(
            structure_id="BLD-001",
            x=50.0,
            y=50.0,
            width=10.0,  # After 90° rotation, this is swapped
            height=20.0,
            rotation_deg=90,
            shape="rect",
        )

        footprint = _get_placement_footprint(placement)

        # Area should still be 200
        assert footprint.area == pytest.approx(200.0, rel=0.01)


class TestHazardZoneFromFootprint:
    """Test hazard zone calculation from equipment footprint."""

    @pytest.fixture
    def rules_with_zones(self):
        """RuleSet with NFPA 820 zone configurations."""
        rules = RuleSet(
            nfpa820_zones={
                "digester": NFPA820ZoneConfig(
                    class_i_div_1_radius=3.0,
                    class_i_div_2_radius=7.6,
                    vertical_above=5.5,
                ),
                "aeration_tank": NFPA820ZoneConfig(
                    class_i_div_1_radius=0.0,  # No Division 1
                    class_i_div_2_radius=3.0,
                    vertical_above=3.0,
                ),
            }
        )
        return rules

    def test_circular_equipment_zone(self, rules_with_zones):
        """Circular digester hazard zone extends from perimeter."""
        placements = [
            Placement(
                structure_id="DIG-001",
                x=50.0,
                y=50.0,
                width=12.0,  # diameter
                height=12.0,
                rotation_deg=0,
                shape="circle",
            )
        ]

        zones = compute_hazard_zones(
            placements=placements,
            rules=rules_with_zones,
            structure_types={"DIG-001": "digester"},
        )

        # Should have both Division 1 and Division 2 zones
        div1_zones = [z for z in zones if z.zone_type == HazardZoneType.CLASS_I_DIV_1]
        div2_zones = [z for z in zones if z.zone_type == HazardZoneType.CLASS_I_DIV_2]

        assert len(div1_zones) == 1
        assert len(div2_zones) == 1

        div1 = div1_zones[0]
        div2 = div2_zones[0]

        # Division 1 zone: footprint (r=6) + buffer (3) = r=9 total
        # Area = π * 9² ≈ 254.47
        # But the zone is footprint.buffer(3), so it's the circular buffer
        # For circular footprint: area = π * (6+3)² = π * 81 ≈ 254.47
        expected_div1_area = math.pi * 81
        assert div1.polygon.area == pytest.approx(expected_div1_area, rel=0.05)

        # Division 2 is annular ring (donut) between div1 and div2 radii
        # Outer: footprint.buffer(7.6) = π * (6+7.6)² = π * 184.96 ≈ 581.0
        # Inner (Division 1): π * 81
        # Annulus area ≈ 581 - 254 ≈ 327
        div2_outer_area = math.pi * (6 + 7.6) ** 2
        expected_div2_area = div2_outer_area - expected_div1_area
        assert div2.polygon.area == pytest.approx(expected_div2_area, rel=0.05)

    def test_rectangular_equipment_zone_from_perimeter(self, rules_with_zones):
        """Rectangular tank hazard zone extends from all edges, not centroid.

        This is the key test: a 20m x 12m tank with 3m Division 1 radius
        should produce a zone that is 3m from ALL edges, not a circular
        zone centered on the tank centroid.
        """
        placements = [
            Placement(
                structure_id="DIG-002",
                x=50.0,
                y=50.0,
                width=20.0,
                height=12.0,
                rotation_deg=0,
                shape="rect",
            )
        ]

        zones = compute_hazard_zones(
            placements=placements,
            rules=rules_with_zones,
            structure_types={"DIG-002": "digester"},
        )

        div1_zones = [z for z in zones if z.zone_type == HazardZoneType.CLASS_I_DIV_1]
        assert len(div1_zones) == 1
        div1 = div1_zones[0]

        # Footprint is 20x12 rectangle centered at (50, 50)
        # Division 1 buffer is 3m from perimeter
        # Result is a rounded rectangle approximately 26x18 (20+6, 12+6)
        # with rounded corners

        # Test key points that should be INSIDE the zone:
        # - Points 2m from tank corners (within 3m buffer)
        inside_points = [
            (50 + 10 + 2, 50),      # 2m from east edge
            (50 - 10 - 2, 50),      # 2m from west edge
            (50, 50 + 6 + 2),       # 2m from north edge
            (50, 50 - 6 - 2),       # 2m from south edge
        ]
        for px, py in inside_points:
            assert div1.polygon.contains(Point(px, py)), \
                f"Point ({px}, {py}) should be inside Division 1 zone"

        # Test key points that should be OUTSIDE the zone:
        outside_points = [
            (50 + 10 + 4, 50),      # 4m from east edge (> 3m buffer)
            (50 - 10 - 4, 50),      # 4m from west edge
            (50, 50 + 6 + 4),       # 4m from north edge
            (50, 50 - 6 - 4),       # 4m from south edge
        ]
        for px, py in outside_points:
            assert not div1.polygon.contains(Point(px, py)), \
                f"Point ({px}, {py}) should be outside Division 1 zone"

    def test_zone_shape_not_circular_for_rectangle(self, rules_with_zones):
        """Hazard zone for rectangle is NOT circular.

        Old centroid-based code would produce circular zones.
        Footprint-based code produces rounded rectangles.
        """
        placements = [
            Placement(
                structure_id="DIG-003",
                x=50.0,
                y=50.0,
                width=30.0,  # Very elongated
                height=5.0,
                rotation_deg=0,
                shape="rect",
            )
        ]

        zones = compute_hazard_zones(
            placements=placements,
            rules=rules_with_zones,
            structure_types={"DIG-003": "digester"},
        )

        div1 = [z for z in zones if z.zone_type == HazardZoneType.CLASS_I_DIV_1][0]

        # For elongated rectangle, the zone should also be elongated
        minx, miny, maxx, maxy = div1.polygon.bounds
        zone_width = maxx - minx   # Should be ~36 (30 + 2*3)
        zone_height = maxy - miny  # Should be ~11 (5 + 2*3)

        # Zone should NOT be square/circular
        assert zone_width > zone_height * 2, \
            "Zone should be elongated (not circular) for rectangular equipment"

        # Verify expected dimensions
        assert zone_width == pytest.approx(36.0, abs=1.0)
        assert zone_height == pytest.approx(11.0, abs=1.0)


class TestHazardZoneGeoJSON:
    """Test GeoJSON export of hazard zones."""

    def test_geojson_feature_structure(self):
        """GeoJSON feature has correct structure."""
        zone = HazardZone(
            zone_type=HazardZoneType.CLASS_I_DIV_1,
            source_equipment_id="DIG-001",
            source_equipment_type="digester",
            polygon=Point(50, 50).buffer(10),
            radius=3.0,
            vertical_extent=5.5,
        )

        feature = zone.to_geojson_feature()

        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"
        assert "coordinates" in feature["geometry"]

        props = feature["properties"]
        assert props["feature_type"] == "hazard_zone"
        assert props["zone_type"] == "class_i_div_1"
        assert props["source_equipment_id"] == "DIG-001"
        assert props["radius_m"] == 3.0
        assert props["vertical_extent_m"] == 5.5
        assert "Class I, Division 1" in props["nfpa820_classification"]

    def test_annular_zone_has_hole(self):
        """Division 2 annular zone exports with interior ring (hole)."""
        # Create a donut polygon (annulus)
        outer = Point(50, 50).buffer(10)
        inner = Point(50, 50).buffer(5)
        annulus = outer.difference(inner)

        zone = HazardZone(
            zone_type=HazardZoneType.CLASS_I_DIV_2,
            source_equipment_id="DIG-001",
            source_equipment_type="digester",
            polygon=annulus,
            radius=10.0,
            vertical_extent=5.5,
        )

        feature = zone.to_geojson_feature()

        # Should have exterior and interior rings
        coords = feature["geometry"]["coordinates"]
        # For polygon with hole: [[exterior], [interior]]
        assert len(coords) == 2, "Annular zone should have exterior and interior rings"


class TestCombinedHazardZones:
    """Test combining multiple hazard zones."""

    def test_combine_overlapping_zones(self):
        """Overlapping zones merge into single polygon."""
        zones = [
            HazardZone(
                zone_type=HazardZoneType.CLASS_I_DIV_1,
                source_equipment_id="DIG-001",
                source_equipment_type="digester",
                polygon=Point(50, 50).buffer(10),
                radius=10.0,
                vertical_extent=5.5,
            ),
            HazardZone(
                zone_type=HazardZoneType.CLASS_I_DIV_1,
                source_equipment_id="DIG-002",
                source_equipment_type="digester",
                polygon=Point(60, 50).buffer(10),  # Overlaps with first
                radius=10.0,
                vertical_extent=5.5,
            ),
        ]

        combined = get_combined_hazard_zones(zones, HazardZoneType.CLASS_I_DIV_1)

        assert combined is not None
        assert combined.is_valid
        # Combined area should be less than sum of individual areas (overlap)
        individual_sum = zones[0].polygon.area + zones[1].polygon.area
        assert combined.area < individual_sum

    def test_filter_by_zone_type(self):
        """Filtering returns only matching zone type."""
        zones = [
            HazardZone(
                zone_type=HazardZoneType.CLASS_I_DIV_1,
                source_equipment_id="DIG-001",
                source_equipment_type="digester",
                polygon=Point(50, 50).buffer(3),
                radius=3.0,
                vertical_extent=5.5,
            ),
            HazardZone(
                zone_type=HazardZoneType.CLASS_I_DIV_2,
                source_equipment_id="DIG-001",
                source_equipment_type="digester",
                polygon=Point(50, 50).buffer(7.6).difference(Point(50, 50).buffer(3)),
                radius=7.6,
                vertical_extent=5.5,
            ),
        ]

        div1_only = get_combined_hazard_zones(zones, HazardZoneType.CLASS_I_DIV_1)
        div2_only = get_combined_hazard_zones(zones, HazardZoneType.CLASS_I_DIV_2)

        assert div1_only is not None
        assert div2_only is not None
        # Division 1 zone (smaller) should have less area than Division 2
        assert div1_only.area < div2_only.area

    def test_empty_zones_returns_none(self):
        """Empty zone list returns None."""
        combined = get_combined_hazard_zones([], HazardZoneType.CLASS_I_DIV_1)
        assert combined is None
