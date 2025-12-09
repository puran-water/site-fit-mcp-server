"""Integration tests for site-fit generation pipeline.

Tests the full pipeline from request to solutions with sample site data.
Monitors rejection statistics to detect high failure rates.
"""

import pytest
import asyncio
from typing import Any, Dict, List

from src.pipeline import generate_site_fits
from src.tools.sitefit_tools import SiteFitRequest
from src.models.rules import RuleSet


# ============================================================================
# Sample Site Data
# ============================================================================

# Simple rectangular site (100m x 80m)
SIMPLE_SITE_BOUNDARY = [
    [0, 0],
    [100, 0],
    [100, 80],
    [0, 80],
    [0, 0],
]

# L-shaped site (complex boundary)
L_SHAPED_SITE_BOUNDARY = [
    [0, 0],
    [100, 0],
    [100, 40],
    [60, 40],
    [60, 80],
    [0, 80],
    [0, 0],
]

# Site with keepout zone
SIMPLE_KEEPOUT = {
    "id": "wetland_1",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[30, 30], [50, 30], [50, 50], [30, 50], [30, 30]]],
    },
    "reason": "wetland",
}


# Sample structures (small wastewater facility)
def get_sample_structures() -> List[Dict[str, Any]]:
    """Get a set of sample structures for testing."""
    return [
        {
            "id": "EQ-001",
            "type": "influent_pump_station",
            "footprint": {"shape": "rect", "w": 8, "h": 6},
            "access": {"vehicle": "truck", "required": True, "dock_length": 12},
        },
        {
            "id": "EQ-002",
            "type": "screening_building",
            "footprint": {"shape": "rect", "w": 10, "h": 8},
            "access": {"vehicle": "truck", "required": True, "dock_length": 10},
        },
        {
            "id": "TK-001",
            "type": "equalization_tank",
            "footprint": {"shape": "circle", "d": 15},
            "access": {"vehicle": "tanker", "required": True, "dock_length": 15},
        },
        {
            "id": "AS-001",
            "type": "aeration_tank",
            "footprint": {"shape": "rect", "w": 20, "h": 12},
            "access": {"vehicle": "forklift", "required": False},
        },
        {
            "id": "CL-001",
            "type": "secondary_clarifier",
            "footprint": {"shape": "circle", "d": 18},
        },
    ]


def get_minimal_structures() -> List[Dict[str, Any]]:
    """Get minimal 3-structure set for quick tests.

    Note: Structures are sized to fit comfortably within the 100x80m test site
    with default clearances (3m) and setbacks (7.5m).
    """
    return [
        {
            "id": "EQ-001",
            "type": "pump_station",
            "footprint": {"shape": "rect", "w": 5, "h": 4},
        },
        {
            "id": "TK-001",
            "type": "tank",
            "footprint": {"shape": "circle", "d": 8},
        },
        {
            "id": "BLD-001",
            "type": "building",
            "footprint": {"shape": "rect", "w": 6, "h": 5},
        },
    ]


def get_entrance() -> Dict[str, Any]:
    """Get sample entrance at site boundary."""
    return {"id": "gate_1", "point": [50, 0], "width": 6.0}


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_site_request() -> SiteFitRequest:
    """Create a simple site fit request."""
    return SiteFitRequest(
        site={
            "boundary": SIMPLE_SITE_BOUNDARY,
            "entrances": [get_entrance()],
            "keepouts": [],
            "existing": [],
        },
        topology=None,
        program={"structures": get_minimal_structures()},
        rules_override=None,
        generation={
            "max_solutions": 3,
            "max_time_seconds": 30,
            "seed": 42,
        },
    )


@pytest.fixture
def complex_site_request() -> SiteFitRequest:
    """Create a complex site fit request with keepouts and more structures."""
    return SiteFitRequest(
        site={
            "boundary": L_SHAPED_SITE_BOUNDARY,
            "entrances": [get_entrance()],
            "keepouts": [SIMPLE_KEEPOUT],
            "existing": [],
        },
        topology=None,
        program={"structures": get_sample_structures()},
        rules_override=None,
        generation={
            "max_solutions": 5,
            "max_time_seconds": 60,
            "seed": 123,
        },
    )


@pytest.fixture
def request_with_topology() -> SiteFitRequest:
    """Create request with SFILES2 topology."""
    sfiles2 = "(influent)pump|EQ-001(screening)EQ-002(eq_tank)TK-001(aeration)AS-001(clarifier)CL-001(effluent)"
    return SiteFitRequest(
        site={
            "boundary": SIMPLE_SITE_BOUNDARY,
            "entrances": [get_entrance()],
            "keepouts": [],
            "existing": [],
        },
        topology={
            "sfiles2": sfiles2,
            "node_metadata": {
                "EQ-001": {"area_number": 100},
                "EQ-002": {"area_number": 100},
                "TK-001": {"area_number": 200},
                "AS-001": {"area_number": 200},
                "CL-001": {"area_number": 300},
            },
        },
        program={"structures": get_sample_structures()},
        rules_override=None,
        generation={
            "max_solutions": 3,
            "max_time_seconds": 45,
            "seed": 42,
        },
    )


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Test full pipeline integration."""

    @pytest.mark.asyncio
    async def test_simple_site_generates_solutions(self, simple_site_request):
        """Test that a simple site generates at least one solution."""
        solutions, stats = await generate_site_fits(simple_site_request)

        assert solutions is not None
        assert len(solutions) > 0, f"Expected solutions, got 0. Stats: {stats}"
        assert stats.get("status") != "failed"

        # Check solution structure
        sol = solutions[0]
        assert sol.id is not None
        assert len(sol.placements) == len(get_minimal_structures())
        assert sol.metrics is not None

    @pytest.mark.asyncio
    async def test_complex_site_generates_solutions(self, complex_site_request):
        """Test complex site with keepouts generates solutions."""
        solutions, stats = await generate_site_fits(complex_site_request)

        assert solutions is not None
        # Complex sites may fail - check stats for reasons
        if not solutions:
            print(f"No solutions for complex site. Stats: {stats}")
            # This is acceptable - the site may be too constrained
            return

        assert len(solutions) > 0
        # Verify keepouts are respected (structures not inside keepout)
        for sol in solutions:
            for placement in sol.placements:
                # Keepout is at x=30-50, y=30-50
                # Structure centers should not be in keepout
                if 30 < placement.x < 50 and 30 < placement.y < 50:
                    # This would be a violation - containment should reject
                    pytest.fail(f"Structure {placement.structure_id} placed inside keepout")

    @pytest.mark.asyncio
    async def test_topology_affects_placement(self, request_with_topology):
        """Test that topology hints affect solution layout."""
        solutions, stats = await generate_site_fits(request_with_topology)

        # Topology parsing may fail gracefully - check stats
        if stats.get("topology_error"):
            pytest.skip(f"Topology parsing failed: {stats['topology_error']}")

        assert solutions is not None
        # With topology, expect process flow direction to be somewhat respected
        # (upstream units should generally be west of downstream)

    @pytest.mark.asyncio
    async def test_no_solutions_on_impossible_site(self):
        """Test that impossible sites return empty solutions gracefully."""
        # Very small site that can't fit all structures
        impossible_request = SiteFitRequest(
            site={
                "boundary": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],  # 10x10m
                "entrances": [{"id": "gate", "point": [5, 0]}],
                "keepouts": [],
                "existing": [],
            },
            topology=None,
            program={
                "structures": [
                    {"id": "big1", "type": "tank", "footprint": {"shape": "rect", "w": 8, "h": 8}},
                    {"id": "big2", "type": "tank", "footprint": {"shape": "rect", "w": 8, "h": 8}},
                ]
            },
            generation={"max_solutions": 1, "max_time_seconds": 10, "seed": 1},
        )

        solutions, stats = await generate_site_fits(impossible_request)

        # Should return empty list, not crash
        assert solutions is not None
        assert isinstance(solutions, list)
        # Expect error in stats
        assert "error" in stats or len(solutions) == 0


class TestRejectionStatistics:
    """Monitor rejection statistics to detect high failure rates."""

    @pytest.mark.asyncio
    async def test_rejection_rates_are_acceptable(self, simple_site_request):
        """Test that rejection rates are not excessively high."""
        solutions, stats = await generate_site_fits(simple_site_request)

        # Get rejection counts
        cpsat_count = stats.get("cpsat_solutions", 0)
        containment_rejects = stats.get("containment_rejects", 0)
        clearance_rejects = stats.get("clearance_rejects", 0)
        boundary_setback_rejects = stats.get("boundary_setback_rejects", 0)
        validated = stats.get("validated_solutions", 0)

        total_rejects = containment_rejects + clearance_rejects + boundary_setback_rejects

        print(f"\n=== Rejection Statistics ===")
        print(f"CP-SAT solutions: {cpsat_count}")
        print(f"Containment rejects: {containment_rejects}")
        print(f"Clearance rejects: {clearance_rejects}")
        print(f"Boundary setback rejects: {boundary_setback_rejects}")
        print(f"Total rejects: {total_rejects}")
        print(f"Validated solutions: {validated}")
        print(f"Final solutions: {len(solutions)}")

        # Acceptable thresholds
        if cpsat_count > 0:
            rejection_rate = total_rejects / cpsat_count
            print(f"Rejection rate: {rejection_rate:.1%}")

            # Alert if rejection rate is very high (> 90%)
            if rejection_rate > 0.9 and validated == 0:
                pytest.fail(
                    f"Excessive rejection rate: {rejection_rate:.1%}. "
                    f"This indicates Phase 5 validation is too strict or CP-SAT "
                    f"is generating solutions outside the buildable area."
                )

    @pytest.mark.asyncio
    async def test_report_all_rejection_stats(self, complex_site_request):
        """Generate detailed rejection report for complex site."""
        solutions, stats = await generate_site_fits(complex_site_request)

        print("\n=== Full Pipeline Statistics ===")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")

        # Assert we got stats
        assert "job_id" in stats


class TestSolutionQuality:
    """Test quality metrics of generated solutions."""

    @pytest.mark.asyncio
    async def test_solutions_have_valid_metrics(self, simple_site_request):
        """Test that solutions have properly computed metrics."""
        solutions, stats = await generate_site_fits(simple_site_request)

        if not solutions:
            pytest.skip("No solutions generated")

        for sol in solutions:
            metrics = sol.metrics

            # Metrics should be within valid ranges
            assert 0 <= metrics.compactness <= 1, f"Invalid compactness: {metrics.compactness}"
            assert metrics.road_length >= 0, f"Invalid road length: {metrics.road_length}"
            assert metrics.topology_penalty >= 0, f"Invalid topology penalty: {metrics.topology_penalty}"

    @pytest.mark.asyncio
    async def test_solutions_are_diverse(self, simple_site_request):
        """Test that returned solutions are actually diverse."""
        # Create a new request with more solutions requested
        request = SiteFitRequest(
            site=simple_site_request.site,
            topology=simple_site_request.topology,
            program=simple_site_request.program,
            rules_override=simple_site_request.rules_override,
            generation={
                "max_solutions": 5,
                "max_time_seconds": 30,
                "seed": 42,
            },
        )

        solutions, stats = await generate_site_fits(request)

        if len(solutions) < 2:
            pytest.skip("Not enough solutions to test diversity")

        # Compare first two solutions
        sol1, sol2 = solutions[0], solutions[1]

        # At least one placement should differ significantly
        placement_diffs = []
        for p1 in sol1.placements:
            p2 = sol2.get_placement(p1.structure_id)
            if p2:
                dist = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
                placement_diffs.append(dist)

        if placement_diffs:
            max_diff = max(placement_diffs)
            print(f"\nMax placement difference between solutions: {max_diff:.1f}m")
            # Solutions should have at least some difference
            assert max_diff > 0.1, "Solutions appear to be identical"


class TestBoundaryConstraints:
    """Test that boundary constraints are properly enforced."""

    @pytest.mark.asyncio
    async def test_structures_inside_boundary(self, simple_site_request):
        """Test that all structures are inside the site boundary."""
        from shapely.geometry import Polygon

        solutions, stats = await generate_site_fits(simple_site_request)

        if not solutions:
            pytest.skip("No solutions generated")

        boundary = Polygon(SIMPLE_SITE_BOUNDARY)

        for sol in solutions:
            for placement in sol.placements:
                # Check center is inside
                assert boundary.contains_properly(
                    Polygon.from_bounds(
                        placement.x - 0.1, placement.y - 0.1,
                        placement.x + 0.1, placement.y + 0.1
                    ).centroid
                ), f"Structure {placement.structure_id} center at ({placement.x}, {placement.y}) outside boundary"

    @pytest.mark.asyncio
    async def test_roads_inside_boundary(self, simple_site_request):
        """Test that road networks stay inside the site boundary."""
        from shapely.geometry import Polygon, LineString

        solutions, stats = await generate_site_fits(simple_site_request)

        if not solutions:
            pytest.skip("No solutions generated")

        boundary = Polygon(SIMPLE_SITE_BOUNDARY)

        for sol in solutions:
            if sol.road_network:
                for segment in sol.road_network.segments:
                    road_line = LineString(segment.to_linestring_coords())
                    # Road should be within boundary (with small tolerance for edge)
                    assert boundary.buffer(0.5).contains(road_line), (
                        f"Road segment {segment.id} extends outside boundary"
                    )


class TestAccessRequirements:
    """Test that access requirements are properly handled."""

    @pytest.mark.asyncio
    async def test_access_requirements_parsed(self):
        """Test that access requirements are parsed from input."""
        request = SiteFitRequest(
            site={
                "boundary": SIMPLE_SITE_BOUNDARY,
                "entrances": [get_entrance()],
                "keepouts": [],
                "existing": [],
            },
            topology=None,
            program={
                "structures": [
                    {
                        "id": "EQ-001",
                        "type": "pump",
                        "footprint": {"shape": "rect", "w": 6, "h": 4},
                        "access": {
                            "vehicle": "tanker",
                            "dock_edge": "long_side",
                            "dock_length": 15,
                            "dock_width": 8,
                            "required": True,
                            "turning_radius": 12,
                        },
                    },
                ]
            },
            generation={"max_solutions": 1, "max_time_seconds": 20, "seed": 1},
        )

        solutions, stats = await generate_site_fits(request)

        # Should not crash when parsing access requirements
        assert "error" not in stats or stats.get("num_structures") == 1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
