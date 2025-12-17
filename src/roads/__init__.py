"""Road network generation and validation."""

from .dock_zones import (
    DockZone,
    generate_dock_zones,
    get_dock_edge,
)
from .pathfinder import (
    find_road_path,
    create_cost_grid,
    PathfinderResult,
)
from .network import (
    RoadNetworkBuilder,
    validate_road_network,
    RoadValidationResult,
)
from .turning_radius import (
    validate_turning_radius,
    TurningRadiusResult,
    TurningRadiusIssue,
    compute_required_leg_length,
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
