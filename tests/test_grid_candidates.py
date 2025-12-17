"""Tests for grid-based candidate computation."""

import pytest
from shapely.geometry import Polygon, box

from src.solver.grid_candidates import (
    CandidateGrid,
    compute_valid_candidates,
    compute_candidates_for_structures,
    candidates_to_element_tables,
)


class TestComputeValidCandidates:
    """Test compute_valid_candidates function."""

    def test_simple_rectangular_site(self):
        """Test candidate generation for simple rectangular site."""
        buildable = box(0, 0, 100, 80)
        grid = compute_valid_candidates(
            buildable=buildable,
            structure_width=10,
            structure_height=8,
            grid_resolution=5.0,
            structure_id="TK-001",
        )

        assert grid.num_candidates > 0
        assert grid.structure_id == "TK-001"
        assert grid.width == 10
        assert grid.height == 8

        # All candidates should be inside the shrunk region
        shrink = max(10, 8) / 2  # 5m shrink
        for x, y in grid.candidates:
            assert x >= shrink
            assert x <= 100 - shrink
            assert y >= shrink
            assert y <= 80 - shrink

    def test_structure_too_large_returns_empty(self):
        """Test that oversized structures return empty candidates."""
        buildable = box(0, 0, 10, 10)  # 10x10m site
        grid = compute_valid_candidates(
            buildable=buildable,
            structure_width=15,  # Larger than site
            structure_height=15,
            grid_resolution=1.0,
            structure_id="TOO-BIG",
        )

        assert grid.num_candidates == 0

    def test_keepout_zones_excluded(self):
        """Test that candidates inside keepout zones are excluded."""
        buildable = box(0, 0, 100, 80)
        keepout = box(30, 30, 50, 50)  # 20x20m keepout in center

        grid = compute_valid_candidates(
            buildable=buildable,
            structure_width=4,
            structure_height=4,
            grid_resolution=2.0,
            keepouts=[keepout],
            structure_id="TK-002",
        )

        assert grid.num_candidates > 0

        # No candidates should be in the keepout zone (accounting for structure size)
        shrink = 2  # half of structure size
        for x, y in grid.candidates:
            # Point shouldn't be inside the expanded keepout
            assert not (30 - shrink < x < 50 + shrink and 30 - shrink < y < 50 + shrink)

    def test_multiple_keepouts(self):
        """Test handling of multiple keepout zones."""
        buildable = box(0, 0, 100, 80)
        keepouts = [
            box(10, 10, 20, 20),
            box(70, 50, 90, 70),
        ]

        grid = compute_valid_candidates(
            buildable=buildable,
            structure_width=4,
            structure_height=4,
            grid_resolution=2.0,
            keepouts=keepouts,
            structure_id="TK-003",
        )

        assert grid.num_candidates > 0

    def test_rotatable_structure_uses_max_dimension(self):
        """Test that rotatable structures use max dimension for shrinking."""
        buildable = box(0, 0, 100, 80)

        # Non-rotatable (single orientation)
        grid_fixed = compute_valid_candidates(
            buildable=buildable,
            structure_width=20,
            structure_height=10,
            grid_resolution=5.0,
            orientations=[0],
            structure_id="FIXED",
        )

        # Rotatable (multiple orientations)
        grid_rotatable = compute_valid_candidates(
            buildable=buildable,
            structure_width=20,
            structure_height=10,
            grid_resolution=5.0,
            orientations=[0, 90],
            structure_id="ROTATABLE",
        )

        # Rotatable structure should have fewer candidates (uses max(20,10) for shrink)
        assert grid_rotatable.num_candidates <= grid_fixed.num_candidates

    def test_grid_resolution_affects_density(self):
        """Test that grid resolution affects candidate density."""
        buildable = box(0, 0, 100, 80)

        grid_coarse = compute_valid_candidates(
            buildable=buildable,
            structure_width=5,
            structure_height=5,
            grid_resolution=10.0,
            structure_id="COARSE",
        )

        grid_fine = compute_valid_candidates(
            buildable=buildable,
            structure_width=5,
            structure_height=5,
            grid_resolution=2.0,
            structure_id="FINE",
        )

        # Fine grid should have more candidates
        assert grid_fine.num_candidates > grid_coarse.num_candidates

    def test_l_shaped_site(self):
        """Test candidate generation for L-shaped (complex) site."""
        # L-shaped polygon
        l_shaped = Polygon([
            (0, 0), (100, 0), (100, 40), (60, 40), (60, 80), (0, 80), (0, 0)
        ])

        grid = compute_valid_candidates(
            buildable=l_shaped,
            structure_width=8,
            structure_height=6,
            grid_resolution=5.0,
            structure_id="L-STRUCT",
        )

        assert grid.num_candidates > 0

        # Verify candidates are inside L-shape (not in the cut-out corner)
        # Cut-out is roughly x>60, y>40
        for x, y in grid.candidates:
            if x > 60 and y > 40:
                pytest.fail(f"Candidate ({x}, {y}) is in the cut-out corner")


class TestCandidateGrid:
    """Test CandidateGrid dataclass methods."""

    def test_x_coords_property(self):
        """Test x_coords property returns correct list."""
        grid = CandidateGrid(
            structure_id="TEST",
            candidates=[(10, 20), (15, 25), (20, 30)],
            grid_resolution=1.0,
            width=5,
            height=5,
        )

        assert grid.x_coords == [10, 15, 20]

    def test_y_coords_property(self):
        """Test y_coords property returns correct list."""
        grid = CandidateGrid(
            structure_id="TEST",
            candidates=[(10, 20), (15, 25), (20, 30)],
            grid_resolution=1.0,
            width=5,
            height=5,
        )

        assert grid.y_coords == [20, 25, 30]

    def test_to_grid_coords(self):
        """Test conversion to integer grid coordinates."""
        grid = CandidateGrid(
            structure_id="TEST",
            candidates=[(10.0, 20.0), (15.0, 25.0)],
            grid_resolution=5.0,
            width=5,
            height=5,
        )

        x_grid, y_grid = grid.to_grid_coords()

        assert x_grid == [2, 3]  # 10/5=2, 15/5=3
        assert y_grid == [4, 5]  # 20/5=4, 25/5=5


class TestCandidatesToElementTables:
    """Test candidates_to_element_tables function."""

    def test_creates_valid_lookup_tables(self):
        """Test that element tables are created correctly."""
        grid = CandidateGrid(
            structure_id="TEST",
            candidates=[(0, 0), (5, 5), (10, 10)],
            grid_resolution=1.0,
            width=2,
            height=2,
        )

        x_table, y_table = candidates_to_element_tables(grid, 1.0)

        assert len(x_table) == 3
        assert len(y_table) == 3
        assert x_table == [0, 5, 10]
        assert y_table == [0, 5, 10]


class TestComputeCandidatesForStructures:
    """Test batch candidate computation for multiple structures."""

    def test_computes_for_all_structures(self):
        """Test that candidates are computed for all structures."""
        from dataclasses import dataclass
        from typing import Optional, List

        # Mock structure objects
        @dataclass
        class MockCircleFootprint:
            d: float

        @dataclass
        class MockRectFootprint:
            w: float
            h: float

        @dataclass
        class MockStructure:
            id: str
            type: str
            footprint: object
            is_circle: bool = False
            orientations_deg: Optional[List[int]] = None

        structures = [
            MockStructure(
                id="TK-001",
                type="tank",
                footprint=MockCircleFootprint(d=10),
                is_circle=True,
            ),
            MockStructure(
                id="BLD-001",
                type="building",
                footprint=MockRectFootprint(w=8, h=6),
                is_circle=False,
                orientations_deg=[0, 90],
            ),
        ]

        buildable = box(0, 0, 100, 80)

        result = compute_candidates_for_structures(
            structures=structures,
            buildable=buildable,
            grid_resolution=5.0,
            keepouts=None,
        )

        assert "TK-001" in result
        assert "BLD-001" in result
        assert result["TK-001"].num_candidates > 0
        assert result["BLD-001"].num_candidates > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
