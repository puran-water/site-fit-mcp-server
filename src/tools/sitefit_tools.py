"""MCP tool request/response schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SiteFitDiversityConfig(BaseModel):
    """Configuration for solution diversity."""

    min_delta: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum fingerprint distance between diverse solutions (0-1)",
    )
    centroid_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for position-based diversity",
    )
    ordering_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for ordering-based diversity",
    )
    cluster_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for cluster-based diversity",
    )


class SiteFitGenerationConfig(BaseModel):
    """Configuration for site fit generation."""

    max_solutions: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of solutions to return",
    )
    max_time_seconds: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Maximum solve time in seconds",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )
    diversity: SiteFitDiversityConfig = Field(
        default_factory=SiteFitDiversityConfig,
        description="Diversity filtering configuration",
    )
    require_road_access: bool = Field(
        default=True,
        description="Reject solutions where docks can't be reached by roads",
    )


class SiteInput(BaseModel):
    """Site boundary and constraints input."""

    boundary: List[List[float]] = Field(
        ...,
        description="Site boundary as list of [x, y] coordinates (closed polygon)",
    )
    entrances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Site entrances with id, point, and optional width",
    )
    keepouts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Keep-out zones with id, geometry, and reason",
    )
    existing: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Existing structures that cannot be moved",
    )


class TopologyInput(BaseModel):
    """Process topology input."""

    sfiles2: Optional[str] = Field(
        default=None,
        description="SFILES2 string describing process topology",
    )
    node_metadata: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Additional metadata for topology nodes",
    )


class ProgramInput(BaseModel):
    """Building program (structures to place)."""

    structures: List[Dict[str, Any]] = Field(
        ...,
        description="List of structures with footprint and requirements",
    )


class SiteFitRequest(BaseModel):
    """Complete request for site fit generation."""

    site: SiteInput = Field(..., description="Site boundary and constraints")
    topology: Optional[TopologyInput] = Field(
        default=None,
        description="Process topology from SFILES2",
    )
    program: ProgramInput = Field(..., description="Building program")
    rules_override: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Override default engineering rules",
    )
    generation: SiteFitGenerationConfig = Field(
        default_factory=SiteFitGenerationConfig,
        description="Generation configuration",
    )


class SolutionSummary(BaseModel):
    """Summary of a single solution."""

    id: str
    rank: int
    metrics: Dict[str, Any]
    diversity_note: Optional[str] = None


class SiteFitResponse(BaseModel):
    """Response from site fit generation."""

    job_id: str
    status: str  # "completed", "failed", "timeout"
    message: Optional[str] = None
    num_solutions: int = 0
    solutions: List[SolutionSummary] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
