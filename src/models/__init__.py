"""Pydantic models for site-fit-mcp-server."""

from .site import (
    SiteBoundary,
    Entrance,
    Keepout,
    ExistingStructure,
    GeoJSONPolygon,
    GeoJSONPoint,
)
from .structures import (
    StructureFootprint,
    RectFootprint,
    CircleFootprint,
    AccessRequirement,
    PlacedStructure,
)
from .rules import RuleSet, AccessRules, SetbackRules
from .solution import (
    SiteFitSolution,
    Placement,
    SolutionMetrics,
    RoadNetwork,
    RoadSegment,
)
from .topology import TopologyGraph, TopologyNode, TopologyEdge

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
