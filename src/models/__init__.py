"""Pydantic models for site-fit-mcp-server."""

from .rules import AccessRules, RuleSet, SetbackRules
from .site import (
    Entrance,
    ExistingStructure,
    GeoJSONPoint,
    GeoJSONPolygon,
    Keepout,
    SiteBoundary,
)
from .solution import (
    Placement,
    RoadNetwork,
    RoadSegment,
    SiteFitSolution,
    SolutionMetrics,
)
from .structures import (
    AccessRequirement,
    CircleFootprint,
    PlacedStructure,
    RectFootprint,
    StructureFootprint,
)
from .topology import TopologyEdge, TopologyGraph, TopologyNode

__all__ = [
    # Site
    "SiteBoundary",
    "Entrance",
    "Keepout",
    "ExistingStructure",
    "GeoJSONPolygon",
    "GeoJSONPoint",
    # Structures
    "StructureFootprint",
    "RectFootprint",
    "CircleFootprint",
    "AccessRequirement",
    "PlacedStructure",
    # Rules
    "RuleSet",
    "AccessRules",
    "SetbackRules",
    # Solution
    "SiteFitSolution",
    "Placement",
    "SolutionMetrics",
    "RoadNetwork",
    "RoadSegment",
    # Topology
    "TopologyGraph",
    "TopologyNode",
    "TopologyEdge",
]
