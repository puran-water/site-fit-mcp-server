"""Road network generation and validation."""

from .dock_zones import (
    DockZone,
    generate_dock_zones,
    get_dock_edge,
)
from .network import (
    RoadNetworkBuilder,
    RoadValidationResult,
    validate_road_network,
)
from .pathfinder import (
    PathfinderResult,
    create_cost_grid,
    find_road_path,
)
from .turning_radius import (
    TurningRadiusIssue,
    TurningRadiusResult,
    compute_required_leg_length,
    validate_turning_radius,
)

__all__ = [
    # Dock zones
    "DockZone",
    "generate_dock_zones",
    "get_dock_edge",
    # Pathfinding
    "find_road_path",
    "create_cost_grid",
    "PathfinderResult",
    # Network
    "RoadNetworkBuilder",
    "validate_road_network",
    "RoadValidationResult",
    # Turning radius
    "validate_turning_radius",
    "TurningRadiusResult",
    "TurningRadiusIssue",
    "compute_required_leg_length",
]
