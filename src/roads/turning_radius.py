"""Turning radius validation for road networks.

Validates that all corners in a road path can accommodate the required
vehicle turning radius. Uses inscribed circle geometry.

CRITICAL: The formula used here was corrected based on code review.
The correct formula is: max_achievable_radius = min(leg1, leg2) / tan(θ/2)
where θ is the deflection angle (0 = straight, π = U-turn).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TurningRadiusIssue:
    """Details about a turning radius violation."""

    corner_index: int
    corner_point: Tuple[float, float]
    turn_angle_deg: float
    max_achievable_radius: float
    required_radius: float
    leg1_length: float
    leg2_length: float

    @property
    def message(self) -> str:
        return (
            f"Corner at ({self.corner_point[0]:.1f}, {self.corner_point[1]:.1f}): "
            f"max radius {self.max_achievable_radius:.1f}m < required {self.required_radius}m "
            f"(turn angle {self.turn_angle_deg:.1f}°, legs {self.leg1_length:.1f}m/{self.leg2_length:.1f}m)"
        )


@dataclass
class TurningRadiusResult:
    """Result of turning radius validation."""

    is_valid: bool
    issues: List[TurningRadiusIssue]
    min_achievable_radius: Optional[float] = None  # Tightest corner

    @property
    def summary(self) -> str:
        if self.is_valid:
            return f"Path valid (min radius: {self.min_achievable_radius:.1f}m)"
        return f"Path invalid: {len(self.issues)} corner(s) fail turning radius"


def validate_turning_radius(
    path: List[Tuple[float, float]],
    min_radius: float,
    tolerance: float = 0.5,
) -> TurningRadiusResult:
    """Validate all corners in a path meet turning radius requirements.

    Uses inscribed circle geometry to compute the maximum achievable turning
    radius at each corner based on:
    - The lengths of the two road segments meeting at the corner
    - The angle between them

    CORRECTED FORMULA (per Codex review):
        max_achievable_radius = min(leg1, leg2) / tan(θ/2)

    where θ is the deflection angle (angle between direction vectors):
    - θ = 0 means straight (no turn needed, infinite radius ok)
    - θ = π/2 means 90° turn (right angle)
    - θ = π means 180° U-turn (impossible, radius → 0)

    Args:
        path: List of (x, y) coordinates defining the path
        min_radius: Minimum required turning radius in meters
        tolerance: Tolerance in meters for near-misses (default 0.5m)

    Returns:
        TurningRadiusResult with validation status and any issues

    Example:
        >>> path = [(0, 0), (10, 0), (10, 10)]  # Right angle turn
        >>> result = validate_turning_radius(path, min_radius=5.0)
        >>> print(result.is_valid)
    """
    if len(path) < 3:
        # Need at least 3 points to have a corner
        return TurningRadiusResult(is_valid=True, issues=[], min_achievable_radius=None)

    issues = []
    min_achievable = float('inf')

    for i in range(1, len(path) - 1):
        p_prev = path[i - 1]
        p_curr = path[i]
        p_next = path[i + 1]

        # Compute leg lengths
        leg1 = math.dist(p_prev, p_curr)
        leg2 = math.dist(p_curr, p_next)

        # Skip if legs are too short to compute meaningfully
        if leg1 < 0.1 or leg2 < 0.1:
            continue

        # Compute vectors
        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        # Normalize vectors
        v1_len = math.sqrt(v1[0]**2 + v1[1]**2)
        v2_len = math.sqrt(v2[0]**2 + v2[1]**2)

        if v1_len < 0.001 or v2_len < 0.001:
            continue

        v1_norm = (v1[0] / v1_len, v1[1] / v1_len)
        v2_norm = (v2[0] / v2_len, v2[1] / v2_len)

        # Compute dot product to get angle
        dot = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
        dot = max(-1.0, min(1.0, dot))  # Clamp for numerical stability

        # Deflection angle: angle between incoming and outgoing direction vectors
        # cos(θ) = dot, where θ is the angle between vectors
        # θ = 0 means straight (no turn), θ = π means U-turn
        deflection_angle = math.acos(dot)
        turn_angle = deflection_angle  # How much we actually turn

        # Nearly straight (turn < ~5 degrees) - no radius constraint
        if turn_angle < 0.09:  # ~5 degrees
            continue

        # CORRECTED FORMULA:
        # max_achievable_radius = min(leg1, leg2) / tan(turn_angle / 2)
        half_angle = turn_angle / 2

        if half_angle > 0.001:  # Avoid division by zero
            tan_half = math.tan(half_angle)
            if tan_half > 0.001:
                max_achievable_radius = min(leg1, leg2) / tan_half
            else:
                max_achievable_radius = float('inf')
        else:
            max_achievable_radius = float('inf')

        # Track minimum
        if max_achievable_radius < min_achievable:
            min_achievable = max_achievable_radius

        # Check if corner fails
        if max_achievable_radius < min_radius - tolerance:
            issues.append(TurningRadiusIssue(
                corner_index=i,
                corner_point=p_curr,
                turn_angle_deg=math.degrees(turn_angle),
                max_achievable_radius=max_achievable_radius,
                required_radius=min_radius,
                leg1_length=leg1,
                leg2_length=leg2,
            ))

    return TurningRadiusResult(
        is_valid=len(issues) == 0,
        issues=issues,
        min_achievable_radius=min_achievable if min_achievable != float('inf') else None,
    )


def compute_required_leg_length(
    turn_angle_deg: float,
    min_radius: float,
) -> float:
    """Compute minimum road segment length needed for a given turn angle and radius.

    Useful for planning: how long must road segments be to achieve the turn?

    Args:
        turn_angle_deg: Turn angle in degrees (0 = straight, 90 = right angle, 180 = U-turn)
        min_radius: Required turning radius in meters

    Returns:
        Minimum leg length in meters

    Example:
        >>> length = compute_required_leg_length(90, 12.0)
        >>> print(f"Need {length:.1f}m segments for 90° turn with 12m radius")
    """
    turn_angle_rad = math.radians(turn_angle_deg)
    if turn_angle_rad < 0.01:
        return 0.0  # Nearly straight, no constraint

    half_angle = turn_angle_rad / 2
    tan_half = math.tan(half_angle)

    if tan_half < 0.001:
        return 0.0

    return min_radius * tan_half


def smooth_corner_with_arc(
    p_prev: Tuple[float, float],
    p_curr: Tuple[float, float],
    p_next: Tuple[float, float],
    radius: float,
    num_points: int = 8,
) -> List[Tuple[float, float]]:
    """Generate arc points to smooth a corner with given radius.

    Replaces a sharp corner with an inscribed circular arc.

    Args:
        p_prev: Point before corner
        p_curr: Corner point
        p_next: Point after corner
        radius: Arc radius in meters
        num_points: Number of points on the arc (default 8)

    Returns:
        List of points forming the arc (excluding p_curr)

    Note:
        This is for visualization/future path smoothing. The arc will
        "cut the corner" and may not follow the exact path.
    """
    # Compute vectors
    v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
    v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

    leg1 = math.sqrt(v1[0]**2 + v1[1]**2)
    leg2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if leg1 < 0.1 or leg2 < 0.1:
        return [p_curr]

    # Normalize
    v1_norm = (v1[0] / leg1, v1[1] / leg1)
    v2_norm = (v2[0] / leg2, v2[1] / leg2)

    # Compute angle
    dot = v1_norm[0] * v2_norm[0] + v1_norm[1] * v2_norm[1]
    dot = max(-1.0, min(1.0, dot))
    exterior_angle = math.acos(dot)
    turn_angle = math.pi - exterior_angle

    if turn_angle < 0.01:
        return [p_curr]

    half_angle = turn_angle / 2
    tan_half = math.tan(half_angle)

    if tan_half < 0.001:
        return [p_curr]

    # Distance from corner to tangent points
    tangent_dist = radius * tan_half

    # Check if tangent points would be within the legs
    if tangent_dist > min(leg1, leg2):
        # Can't fit the requested radius, return corner as-is
        return [p_curr]

    # Tangent points
    t1 = (p_curr[0] - v1_norm[0] * tangent_dist, p_curr[1] - v1_norm[1] * tangent_dist)
    t2 = (p_curr[0] + v2_norm[0] * tangent_dist, p_curr[1] + v2_norm[1] * tangent_dist)

    # Bisector direction (points toward arc center)
    bisector = (
        -(v1_norm[0] + v2_norm[0]),
        -(v1_norm[1] + v2_norm[1]),
    )
    bisector_len = math.sqrt(bisector[0]**2 + bisector[1]**2)
    if bisector_len < 0.001:
        return [p_curr]
    bisector_norm = (bisector[0] / bisector_len, bisector[1] / bisector_len)

    # Distance from corner to arc center
    center_dist = radius / math.sin(half_angle) if math.sin(half_angle) > 0.001 else radius

    # Arc center
    center = (
        p_curr[0] + bisector_norm[0] * center_dist,
        p_curr[1] + bisector_norm[1] * center_dist,
    )

    # Generate arc points
    # Start angle: from center to t1
    start_angle = math.atan2(t1[1] - center[1], t1[0] - center[0])
    end_angle = math.atan2(t2[1] - center[1], t2[0] - center[0])

    # Ensure we go the short way around
    delta = end_angle - start_angle
    if delta > math.pi:
        delta -= 2 * math.pi
    elif delta < -math.pi:
        delta += 2 * math.pi

    arc_points = []
    for j in range(num_points + 1):
        t = j / num_points
        angle = start_angle + t * delta
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        arc_points.append((x, y))

    return arc_points
