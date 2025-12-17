"""Tests for per-structure turning radius validation.

Tests that structures with AccessRequirement.turning_radius have their
specific turning radius requirements enforced for dock approaches.
"""

import pytest
from shapely.geometry import Polygon

from src.roads.network import (
    validate_road_network,
    RoadValidationResult,
)
from src.models.site import Entrance
from src.models.structures import (
    StructureFootprint,
    PlacedStructure,
    RectFootprint,
    CircleFootprint,
    AccessRequirement,
)
from src.models.solution import RoadNetwork, RoadSegment


class TestPerStructureTurningRadius:
    """Test per-structure turning radius validation."""

    @pytest.fixture
    def entrance(self):
        """Site entrance."""
        return Entrance(id="ENT-001", point=(0.0, 50.0), width=6.0)

    @pytest.fixture
    def structure_with_turning_requirement(self):
        """Structure requiring 15m turning radius."""
        struct = StructureFootprint(
            id="TK-001",
            type="tank",
            footprint=CircleFootprint(d=10.0),
            access=AccessRequirement(
                vehicle="tanker",
                turning_radius=15.0,  # Requires 15m turning radius
            ),
        )
        return PlacedStructure(
            structure=struct,
            x=50.0,
            y=50.0,
            rotation_deg=0,
        )

    @pytest.fixture
    def structure_without_turning_requirement(self):
        """Structure without specific turning requirement."""
        struct = StructureFootprint(
            id="BLD-001",
            type="building",
            footprint=RectFootprint(w=20.0, h=10.0),
            access=AccessRequirement(
                vehicle="forklift",
                # No turning_radius specified - uses global default
            ),
        )
        return PlacedStructure(
            structure=struct,
            x=80.0,
            y=50.0,
            rotation_deg=0,
        )

    def test_validates_with_structure_specific_radius(
        self, entrance, structure_with_turning_requirement
    ):
        """Validation uses structure-specific turning radius."""
        # Road segment with tight corner connecting to TK-001
        segment = RoadSegment(
            id="seg_entrance_to_TK-001",
            start=(0.0, 50.0),
            end=(50.0, 50.0),
            width=6.0,
            waypoints=[(10.0, 50.0), (10.0, 55.0)],  # Tight 90° turn with 5m legs
            connects_to=["entrance", "TK-001"],
        )

        network = RoadNetwork(
            segments=[segment],
            total_length=50.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["TK-001"],
        )

        result = validate_road_network(
            network=network,
            structures=[structure_with_turning_requirement],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=12.0,  # Global default
            validate_turning=True,
            structure_turning_radii={"TK-001": 15.0},
        )

        # Should fail because structure requires 15m but corner only achieves ~5m
        assert not result.is_valid
        assert any("TK-001" in issue for issue in result.issues)

    def test_global_radius_used_when_no_structure_specific(
        self, entrance, structure_without_turning_requirement
    ):
        """Global radius used for structures without specific requirements."""
        # Straight road to structure (no corners = always valid for turning)
        segment = RoadSegment(
            id="seg_entrance_to_BLD-001",
            start=(0.0, 50.0),
            end=(55.0, 50.0),  # Ends before structure center
            width=6.0,
            waypoints=[],  # Straight segment - no corners
            connects_to=["entrance", "BLD-001"],
        )

        network = RoadNetwork(
            segments=[segment],
            total_length=55.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["BLD-001"],
        )

        result = validate_road_network(
            network=network,
            structures=[structure_without_turning_requirement],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=11.0,  # Global default
            validate_turning=True,
            structure_turning_radii={},  # No structure-specific requirements
        )

        # Should pass - straight road with no corners
        assert result.is_valid

    def test_stricter_structure_requirement_takes_precedence(
        self, entrance, structure_with_turning_requirement
    ):
        """Structure requirement takes precedence over global when stricter."""
        # Corner that meets 12m global but not 15m structure requirement
        segment = RoadSegment(
            id="seg_entrance_to_TK-001",
            start=(0.0, 50.0),
            end=(50.0, 50.0),
            width=6.0,
            waypoints=[(13.0, 50.0), (13.0, 63.0)],  # 13m legs → ~13m achievable
            connects_to=["entrance", "TK-001"],
        )

        network = RoadNetwork(
            segments=[segment],
            total_length=50.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["TK-001"],
        )

        result = validate_road_network(
            network=network,
            structures=[structure_with_turning_requirement],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=12.0,  # Global: 12m
            validate_turning=True,
            structure_turning_radii={"TK-001": 15.0},  # Structure: 15m (stricter)
        )

        # Should fail because TK-001 needs 15m but only ~13m achievable
        assert not result.is_valid
        # Error should mention the structure's requirement
        assert any("15" in issue and "TK-001" in issue for issue in result.issues)

    def test_passing_with_adequate_turning_radius(
        self, entrance, structure_with_turning_requirement
    ):
        """Validation passes when turning radius is adequate."""
        # Straight road to dock zone (avoids structure overlap)
        # Structure is circular d=10 at (50, 50), so edge is at x=45
        segment = RoadSegment(
            id="seg_entrance_to_TK-001",
            start=(0.0, 50.0),
            end=(40.0, 50.0),  # Ends at dock zone, before structure
            width=6.0,
            waypoints=[],  # Straight - always valid for turning
            connects_to=["entrance", "TK-001"],
        )

        network = RoadNetwork(
            segments=[segment],
            total_length=40.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["TK-001"],
        )

        result = validate_road_network(
            network=network,
            structures=[structure_with_turning_requirement],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=12.0,
            validate_turning=True,
            structure_turning_radii={"TK-001": 15.0},
        )

        # Should pass - straight road with no corners
        assert result.is_valid

    def test_straight_segment_always_valid(self, entrance, structure_with_turning_requirement):
        """Straight road segment has no corners, always valid."""
        # Road ends at dock zone before structure (d=10 tank at 50,50 → edge at x=45)
        segment = RoadSegment(
            id="seg_entrance_to_TK-001",
            start=(0.0, 50.0),
            end=(40.0, 50.0),  # End before structure
            width=6.0,
            waypoints=[],  # No waypoints = straight segment
            connects_to=["entrance", "TK-001"],
        )

        network = RoadNetwork(
            segments=[segment],
            total_length=40.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["TK-001"],
        )

        result = validate_road_network(
            network=network,
            structures=[structure_with_turning_requirement],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=12.0,
            validate_turning=True,
            structure_turning_radii={"TK-001": 100.0},  # Even very large requirement
        )

        # Should pass - no corners to validate
        assert result.is_valid

    def test_multiple_structures_different_requirements(self, entrance):
        """Multiple structures with different turning requirements."""
        struct1 = StructureFootprint(
            id="TK-001",
            type="tank",
            footprint=CircleFootprint(d=10.0),
            access=AccessRequirement(vehicle="tanker", turning_radius=15.0),
        )
        # Tank at (30, 70) - edge at x=25
        placed1 = PlacedStructure(structure=struct1, x=30.0, y=70.0, rotation_deg=0)

        struct2 = StructureFootprint(
            id="TK-002",
            type="tank",
            footprint=CircleFootprint(d=10.0),
            access=AccessRequirement(vehicle="forklift", turning_radius=8.0),
        )
        # Tank at (70, 70) - edge at x=65
        placed2 = PlacedStructure(structure=struct2, x=70.0, y=70.0, rotation_deg=0)

        # Segment to TK-001 with tight corner (fails 15m requirement)
        # Path: (0,50) -> (15,50) -> (15,60) -> (20,60) - away from structure
        seg1 = RoadSegment(
            id="seg_entrance_to_TK-001",
            start=(0.0, 50.0),
            end=(20.0, 60.0),  # Ends near dock, not on structure
            width=6.0,
            waypoints=[(8.0, 50.0), (8.0, 60.0)],  # 8m legs → ~8m achievable
            connects_to=["entrance", "TK-001"],
        )

        # Segment to TK-002 with adequate corner (passes 8m requirement)
        # Straight road
        seg2 = RoadSegment(
            id="seg_entrance_to_TK-002",
            start=(0.0, 50.0),
            end=(60.0, 60.0),  # Ends at dock zone
            width=6.0,
            waypoints=[],  # Straight - always valid
            connects_to=["entrance", "TK-002"],
        )

        network = RoadNetwork(
            segments=[seg1, seg2],
            total_length=100.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["TK-001", "TK-002"],
        )

        result = validate_road_network(
            network=network,
            structures=[placed1, placed2],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=6.0,  # Global
            validate_turning=True,
            structure_turning_radii={"TK-001": 15.0, "TK-002": 8.0},
        )

        # Should fail for TK-001 (needs 15m, achieves ~8m with 8m legs)
        # TK-002 should be fine (straight road)
        assert not result.is_valid
        assert any("TK-001" in issue for issue in result.issues)
        assert not any("TK-002" in issue for issue in result.issues)


class TestValidationResultMessages:
    """Test validation result message formatting."""

    @pytest.fixture
    def entrance(self):
        return Entrance(id="ENT-001", point=(0.0, 50.0), width=6.0)

    @pytest.fixture
    def placed_structure(self):
        struct = StructureFootprint(
            id="TK-001",
            type="tank",
            footprint=CircleFootprint(d=10.0),
            access=AccessRequirement(vehicle="tanker", turning_radius=15.0),
        )
        return PlacedStructure(structure=struct, x=50.0, y=50.0, rotation_deg=0)

    def test_error_message_includes_required_radius(self, entrance, placed_structure):
        """Error message includes the required turning radius."""
        segment = RoadSegment(
            id="seg_entrance_to_TK-001",
            start=(0.0, 50.0),
            end=(50.0, 50.0),
            width=6.0,
            waypoints=[(5.0, 50.0), (5.0, 55.0)],  # Very tight corner
            connects_to=["entrance", "TK-001"],
        )

        network = RoadNetwork(
            segments=[segment],
            total_length=50.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["TK-001"],
        )

        result = validate_road_network(
            network=network,
            structures=[placed_structure],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=12.0,
            validate_turning=True,
            structure_turning_radii={"TK-001": 15.0},
        )

        assert not result.is_valid
        # Message should include required radius and structure ID
        issues_text = " ".join(result.issues)
        assert "15" in issues_text  # Required radius
        assert "TK-001" in issues_text  # Structure ID

    def test_error_message_includes_segment_id(self, entrance, placed_structure):
        """Error message includes the failing segment ID."""
        segment = RoadSegment(
            id="seg_entrance_to_TK-001",
            start=(0.0, 50.0),
            end=(50.0, 50.0),
            width=6.0,
            waypoints=[(5.0, 50.0), (5.0, 55.0)],
            connects_to=["entrance", "TK-001"],
        )

        network = RoadNetwork(
            segments=[segment],
            total_length=50.0,
            entrances_connected=["ENT-001"],
            structures_accessible=["TK-001"],
        )

        result = validate_road_network(
            network=network,
            structures=[placed_structure],
            entrances=[entrance],
            require_all_accessible=False,
            min_turning_radius=12.0,
            validate_turning=True,
            structure_turning_radii={"TK-001": 15.0},
        )

        assert not result.is_valid
        assert any("seg_entrance_to_TK-001" in issue for issue in result.issues)
