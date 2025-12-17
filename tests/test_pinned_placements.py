"""Tests for pinned placement functionality.

Tests that structures with pinned=True are placed at their exact fixed_position
coordinates and that the solver validates them against other constraints.
"""

import pytest
from shapely.geometry import Polygon

from src.models.structures import (
    StructureFootprint,
    RectFootprint,
    CircleFootprint,
    FixedPosition,
)
from src.solver.cpsat_placer import PlacementSolver, PlacementSolverConfig, SolverResult
from src.models.rules import RuleSet


class TestFixedPositionModel:
    """Test the FixedPosition data model."""

    def test_valid_rotation_values(self):
        """Valid rotations are 0, 90, 180, 270."""
        for rot in [0, 90, 180, 270]:
            pos = FixedPosition(x=10.0, y=20.0, rotation_deg=rot)
            assert pos.rotation_deg == rot

    def test_invalid_rotation_raises(self):
        """Invalid rotation values raise ValidationError."""
        with pytest.raises(ValueError):
            FixedPosition(x=10.0, y=20.0, rotation_deg=45)

    def test_default_rotation_is_zero(self):
        """Default rotation is 0."""
        pos = FixedPosition(x=10.0, y=20.0)
        assert pos.rotation_deg == 0


class TestStructureFootprintPinned:
    """Test pinned field on StructureFootprint."""

    def test_pinned_with_fixed_position(self):
        """Pinned structure with valid fixed_position."""
        struct = StructureFootprint(
            id="TK-001",
            type="tank",
            footprint=CircleFootprint(d=10.0),
            pinned=True,
            fixed_position=FixedPosition(x=50.0, y=50.0),
        )
        assert struct.pinned is True
        assert struct.fixed_position.x == 50.0
        assert struct.fixed_position.y == 50.0

    def test_pinned_false_by_default(self):
        """Structures are not pinned by default."""
        struct = StructureFootprint(
            id="TK-002",
            type="tank",
            footprint=CircleFootprint(d=10.0),
        )
        assert struct.pinned is False
        assert struct.fixed_position is None

    def test_rect_pinned_with_rotation(self):
        """Rectangular structure pinned with rotation."""
        struct = StructureFootprint(
            id="BLD-001",
            type="building",
            footprint=RectFootprint(w=20.0, h=10.0),
            pinned=True,
            fixed_position=FixedPosition(x=100.0, y=50.0, rotation_deg=90),
        )
        assert struct.pinned is True
        assert struct.fixed_position.rotation_deg == 90


class TestPlacementSolverPinned:
    """Test PlacementSolver handling of pinned structures."""

    @pytest.fixture
    def simple_bounds(self):
        """Simple square bounds: (min_x, min_y, max_x, max_y)."""
        return (0.0, 0.0, 100.0, 100.0)

    @pytest.fixture
    def default_rules(self):
        """Default engineering rules."""
        return RuleSet()

    @pytest.fixture
    def solver_config(self):
        """Fast solver config for testing."""
        return PlacementSolverConfig(
            max_time_seconds=30,
            max_solutions=3,
            grid_resolution=0.5,
        )

    def test_pinned_structure_exact_position(self, simple_bounds, default_rules, solver_config):
        """Pinned structure is placed at exact coordinates."""
        structures = [
            StructureFootprint(
                id="TK-001",
                type="tank",
                footprint=CircleFootprint(d=10.0),
                pinned=True,
                fixed_position=FixedPosition(x=50.0, y=50.0),
            )
        ]

        solver = PlacementSolver(
            structures=structures,
            bounds=simple_bounds,
            rules=default_rules,
            config=solver_config,
        )

        result = solver.solve()
        assert result.status in ("optimal", "feasible")
        assert len(result.solutions) >= 1

        # Get first solution's placements
        placements = result.solutions[0]
        placed_tank = next(p for p in placements if p.structure_id == "TK-001")

        # Should be at exact fixed position (within grid resolution)
        assert placed_tank.x == pytest.approx(50.0, abs=solver_config.grid_resolution)
        assert placed_tank.y == pytest.approx(50.0, abs=solver_config.grid_resolution)

    def test_pinned_rect_rotation_preserved(self, simple_bounds, default_rules, solver_config):
        """Pinned rectangular structure preserves rotation."""
        structures = [
            StructureFootprint(
                id="BLD-001",
                type="building",
                footprint=RectFootprint(w=20.0, h=10.0),
                pinned=True,
                fixed_position=FixedPosition(x=50.0, y=50.0, rotation_deg=90),
            )
        ]

        solver = PlacementSolver(
            structures=structures,
            bounds=simple_bounds,
            rules=default_rules,
            config=solver_config,
        )

        result = solver.solve()
        assert result.status in ("optimal", "feasible")
        assert len(result.solutions) >= 1

        placements = result.solutions[0]
        placed = next(p for p in placements if p.structure_id == "BLD-001")
        assert placed.rotation_deg == 90

    def test_mixed_pinned_and_free(self, simple_bounds, default_rules, solver_config):
        """Mix of pinned and free structures."""
        structures = [
            StructureFootprint(
                id="TK-001",
                type="tank",
                footprint=CircleFootprint(d=10.0),
                pinned=True,
                fixed_position=FixedPosition(x=25.0, y=50.0),
            ),
            StructureFootprint(
                id="TK-002",
                type="tank",
                footprint=CircleFootprint(d=10.0),
                # Not pinned - solver chooses position
            ),
        ]

        solver = PlacementSolver(
            structures=structures,
            bounds=simple_bounds,
            rules=default_rules,
            config=solver_config,
        )

        result = solver.solve()
        assert result.status in ("optimal", "feasible")
        assert len(result.solutions) >= 1

        placements = result.solutions[0]
        pinned = next(p for p in placements if p.structure_id == "TK-001")
        free = next(p for p in placements if p.structure_id == "TK-002")

        # Pinned at exact position
        assert pinned.x == pytest.approx(25.0, abs=solver_config.grid_resolution)
        assert pinned.y == pytest.approx(50.0, abs=solver_config.grid_resolution)

        # Free structure placed somewhere valid (not overlapping pinned)
        # Distance between centers should be > sum of radii (10.0)
        dx = free.x - pinned.x
        dy = free.y - pinned.y
        distance = (dx**2 + dy**2) ** 0.5
        # With clearances, the minimum distance may be larger
        assert distance >= 9.5  # Allow small tolerance for grid rounding

    def test_pinned_position_is_honored_even_if_unusual(self, default_rules, solver_config):
        """Pinned position is placed exactly as specified.

        Note: The solver currently trusts pinned positions and doesn't validate
        them against bounds. This is intentional for brownfield scenarios where
        existing equipment may be outside the "buildable area" polygon but still
        needs to be represented in the model. Validation against site boundary
        is done at the pipeline level.
        """
        structures = [
            StructureFootprint(
                id="TK-001",
                type="tank",
                footprint=CircleFootprint(d=10.0),
                pinned=True,
                # Position at edge of boundary
                fixed_position=FixedPosition(x=95.0, y=50.0),
            )
        ]

        bounds = (0.0, 0.0, 100.0, 100.0)

        solver = PlacementSolver(
            structures=structures,
            bounds=bounds,
            rules=default_rules,
            config=solver_config,
        )

        result = solver.solve()
        assert result.status in ("optimal", "feasible")
        assert len(result.solutions) >= 1

        # Verify the pinned position is honored exactly
        placed = result.solutions[0][0]
        assert placed.x == pytest.approx(95.0, abs=solver_config.grid_resolution)
        assert placed.y == pytest.approx(50.0, abs=solver_config.grid_resolution)

    def test_overlapping_pinned_structures_infeasible(self, simple_bounds, default_rules, solver_config):
        """Two pinned structures that overlap are infeasible."""
        structures = [
            StructureFootprint(
                id="TK-001",
                type="tank",
                footprint=CircleFootprint(d=10.0),
                pinned=True,
                fixed_position=FixedPosition(x=50.0, y=50.0),
            ),
            StructureFootprint(
                id="TK-002",
                type="tank",
                footprint=CircleFootprint(d=10.0),
                pinned=True,
                # Overlaps with TK-001 (center distance 5m < sum of radii 10m)
                fixed_position=FixedPosition(x=55.0, y=50.0),
            ),
        ]

        solver = PlacementSolver(
            structures=structures,
            bounds=simple_bounds,
            rules=default_rules,
            config=solver_config,
        )

        result = solver.solve()
        # Should have no valid solutions due to overlap
        assert result.status == "infeasible" or len(result.solutions) == 0
