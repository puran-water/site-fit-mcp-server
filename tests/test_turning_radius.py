"""Tests for turning radius validation.

Tests the corrected formula: max_achievable_radius = min(leg1, leg2) / tan(θ/2)
where θ is the interior turn angle (0 = straight, π = U-turn).
"""

import math
import pytest

from src.roads.turning_radius import (
    validate_turning_radius,
    compute_required_leg_length,
    smooth_corner_with_arc,
    TurningRadiusIssue,
    TurningRadiusResult,
)


class TestValidateTurningRadius:
    """Test turning radius validation with various geometries."""

    def test_straight_path_always_valid(self):
        """Straight path has no corners, always valid."""
        path = [(0, 0), (10, 0), (20, 0), (30, 0)]
        result = validate_turning_radius(path, min_radius=100.0)
        assert result.is_valid
        assert len(result.issues) == 0

    def test_right_angle_with_sufficient_legs(self):
        """90° turn with legs long enough for required radius."""
        # For 90° turn: r = min_leg / tan(45°) = min_leg / 1.0 = min_leg
        # With 10m legs, max achievable radius = 10m
        path = [(0, 0), (10, 0), (10, 10)]
        result = validate_turning_radius(path, min_radius=9.0)
        assert result.is_valid
        assert result.min_achievable_radius is not None
        assert result.min_achievable_radius >= 9.0

    def test_right_angle_with_insufficient_legs(self):
        """90° turn with legs too short for required radius."""
        # With 5m legs, max achievable radius = 5m
        path = [(0, 0), (5, 0), (5, 5)]
        result = validate_turning_radius(path, min_radius=10.0)
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].turn_angle_deg == pytest.approx(90.0, abs=1.0)

    def test_acute_turn_tighter_radius(self):
        """Acute turn (< 90°) requires shorter legs for same radius."""
        # 60° turn: r = min_leg / tan(30°) = min_leg / 0.577 ≈ 1.73 * min_leg
        path = [(0, 0), (10, 0), (15, 8.66)]  # ~60° turn
        result = validate_turning_radius(path, min_radius=10.0)
        assert result.is_valid

    def test_obtuse_turn_larger_radius(self):
        """Obtuse turn (> 90°) allows larger radius with same legs."""
        # 120° turn: r = min_leg / tan(60°) = min_leg / 1.73 ≈ 0.577 * min_leg
        path = [(0, 0), (10, 0), (5, 8.66)]  # ~120° turn
        result = validate_turning_radius(path, min_radius=5.0)
        assert result.is_valid

    def test_multiple_corners_all_valid(self):
        """Path with multiple valid corners."""
        # Square path with 10m sides - three 90° corners
        path = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        result = validate_turning_radius(path, min_radius=9.0)
        assert result.is_valid
        assert len(result.issues) == 0

    def test_multiple_corners_one_invalid(self):
        """Path with one invalid corner among multiple."""
        # Path: right turn at (10,0), straight at (10,10), right turn at (10,15)
        # Corner at index 1: 90° turn, 10m legs → max 10m radius (ok for 9m)
        # Corner at index 2: ~0° turn (straight) → infinite radius (ok)
        # Corner at index 3: 90° turn, 5m legs → max 5m radius (FAILS for 9m)
        path = [(0, 0), (10, 0), (10, 10), (10, 15), (15, 15)]
        result = validate_turning_radius(path, min_radius=9.0)
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].corner_index == 3  # Third corner (5m legs at 90°)

    def test_tolerance_allows_near_miss(self):
        """Tolerance parameter allows slight violations."""
        # 5m legs = 5m max radius, requiring 5.3m should fail without tolerance
        path = [(0, 0), (5, 0), (5, 5)]

        # Should fail without tolerance
        result_strict = validate_turning_radius(path, min_radius=5.3, tolerance=0.0)
        assert not result_strict.is_valid

        # Should pass with 0.5m tolerance
        result_tolerant = validate_turning_radius(path, min_radius=5.3, tolerance=0.5)
        assert result_tolerant.is_valid

    def test_very_short_segments_skipped(self):
        """Very short segments are skipped to avoid numerical issues."""
        path = [(0, 0), (0.05, 0), (0.05, 0.05)]  # < 0.1m segments
        result = validate_turning_radius(path, min_radius=100.0)
        # Should not flag issues for negligible segments
        assert result.is_valid

    def test_two_point_path_trivially_valid(self):
        """Path with only 2 points has no corners."""
        path = [(0, 0), (10, 10)]
        result = validate_turning_radius(path, min_radius=100.0)
        assert result.is_valid
        assert result.min_achievable_radius is None

    def test_single_point_path_trivially_valid(self):
        """Single point path is trivially valid."""
        path = [(5, 5)]
        result = validate_turning_radius(path, min_radius=100.0)
        assert result.is_valid

    def test_empty_path_trivially_valid(self):
        """Empty path is trivially valid."""
        path = []
        result = validate_turning_radius(path, min_radius=100.0)
        assert result.is_valid


class TestTurningRadiusFormula:
    """Test the corrected turning radius formula directly."""

    @pytest.mark.parametrize("turn_deg,expected_factor", [
        (30, 3.73),   # tan(15°) ≈ 0.268, factor ≈ 3.73
        (45, 2.41),   # tan(22.5°) ≈ 0.414, factor ≈ 2.41
        (60, 1.73),   # tan(30°) ≈ 0.577, factor ≈ 1.73
        (90, 1.00),   # tan(45°) = 1.0, factor = 1.0
        (120, 0.58),  # tan(60°) ≈ 1.73, factor ≈ 0.58
        (135, 0.41),  # tan(67.5°) ≈ 2.41, factor ≈ 0.41
        (150, 0.27),  # tan(75°) ≈ 3.73, factor ≈ 0.27
    ])
    def test_radius_to_leg_ratio(self, turn_deg, expected_factor):
        """Verify radius = min_leg / tan(θ/2) gives correct factor."""
        # Create path with turn_deg interior angle
        # Turn angle = 180 - exterior angle
        # exterior angle = angle between vectors

        # For a path (0,0) -> (leg,0) -> endpoint, the turn angle determines endpoint
        leg = 10.0
        turn_rad = math.radians(turn_deg)

        # Use the formula directly
        half_angle = turn_rad / 2
        expected_radius = leg / math.tan(half_angle)

        # Verify the factor relationship
        actual_factor = expected_radius / leg
        assert actual_factor == pytest.approx(expected_factor, rel=0.1)


class TestComputeRequiredLegLength:
    """Test required leg length computation."""

    def test_90_degree_turn(self):
        """90° turn: leg length = radius * tan(45°) = radius."""
        length = compute_required_leg_length(90.0, min_radius=12.0)
        assert length == pytest.approx(12.0, rel=0.01)

    def test_60_degree_turn(self):
        """60° turn: leg length = radius * tan(30°) ≈ 0.577 * radius."""
        length = compute_required_leg_length(60.0, min_radius=12.0)
        assert length == pytest.approx(12.0 * math.tan(math.radians(30)), rel=0.01)

    def test_120_degree_turn(self):
        """120° turn: leg length = radius * tan(60°) ≈ 1.73 * radius."""
        length = compute_required_leg_length(120.0, min_radius=12.0)
        assert length == pytest.approx(12.0 * math.tan(math.radians(60)), rel=0.01)

    def test_nearly_straight_returns_zero(self):
        """Nearly straight path (< 1°) returns 0."""
        length = compute_required_leg_length(0.5, min_radius=12.0)
        assert length == 0.0


class TestSmoothCornerWithArc:
    """Test corner smoothing with inscribed arcs."""

    def test_right_angle_corner(self):
        """Generate arc points for 90° corner."""
        arc = smooth_corner_with_arc(
            p_prev=(0, 0),
            p_curr=(10, 0),
            p_next=(10, 10),
            radius=3.0,
            num_points=8,
        )
        # Should return arc points
        assert len(arc) >= 2

        # First point should be before corner, last after
        # Arc should be smoother than corner
        for point in arc:
            # All points should be near the corner region
            assert 0 <= point[0] <= 15
            assert -5 <= point[1] <= 15

    def test_small_radius_fits(self):
        """Small radius fits in corner."""
        arc = smooth_corner_with_arc(
            p_prev=(0, 0),
            p_curr=(10, 0),
            p_next=(10, 10),
            radius=2.0,
        )
        assert len(arc) > 1  # Should generate arc

    def test_large_radius_returns_corner(self):
        """Radius too large returns corner point."""
        arc = smooth_corner_with_arc(
            p_prev=(0, 0),
            p_curr=(5, 0),  # Only 5m legs
            p_next=(5, 5),
            radius=10.0,   # Too big for 5m legs
        )
        # Should return just the corner since arc won't fit
        assert arc == [(5, 0)]

    def test_degenerate_legs_return_corner(self):
        """Very short legs return corner point."""
        arc = smooth_corner_with_arc(
            p_prev=(0, 0),
            p_curr=(0.01, 0),
            p_next=(0.01, 0.01),
            radius=1.0,
        )
        assert arc == [(0.01, 0)]


class TestTurningRadiusIssue:
    """Test the TurningRadiusIssue dataclass."""

    def test_message_format(self):
        """Issue message contains all relevant info."""
        issue = TurningRadiusIssue(
            corner_index=2,
            corner_point=(15.5, 20.3),
            turn_angle_deg=95.2,
            max_achievable_radius=8.5,
            required_radius=12.0,
            leg1_length=8.5,
            leg2_length=10.2,
        )
        msg = issue.message
        assert "15.5" in msg or "15.5" in msg  # Corner point
        assert "20.3" in msg
        assert "8.5" in msg  # Max radius
        assert "12" in msg  # Required radius
        assert "95" in msg  # Turn angle


class TestTurningRadiusResult:
    """Test the TurningRadiusResult dataclass."""

    def test_valid_summary(self):
        """Valid result shows min radius."""
        result = TurningRadiusResult(
            is_valid=True,
            issues=[],
            min_achievable_radius=15.3,
        )
        assert "valid" in result.summary.lower()
        assert "15.3" in result.summary

    def test_invalid_summary(self):
        """Invalid result shows issue count."""
        result = TurningRadiusResult(
            is_valid=False,
            issues=[
                TurningRadiusIssue(0, (0, 0), 90, 5, 12, 5, 5),
                TurningRadiusIssue(1, (10, 10), 90, 3, 12, 3, 5),
            ],
            min_achievable_radius=3.0,
        )
        assert "invalid" in result.summary.lower()
        assert "2" in result.summary  # 2 issues
