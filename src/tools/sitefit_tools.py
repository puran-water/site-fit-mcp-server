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
    node_map: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping from topology node IDs to structure IDs (e.g., {'reactor-1': 'RX-001'})",
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


class GISLoadRequest(BaseModel):
    """Request for loading site data from a GIS file."""

    file_path: str = Field(
        ...,
        description="Path to the GIS file (Shapefile, GeoJSON, GeoPackage, etc.)",
    )
    boundary_layer: Optional[str] = Field(
        default=None,
        description="Layer name for site boundary (auto-detect if None)",
    )
    keepout_layers: Optional[List[str]] = Field(
        default=None,
        description="Layer names for keepout zones (auto-detect if None)",
    )
    entrance_layer: Optional[str] = Field(
        default=None,
        description="Layer name for entrance points (auto-detect if None)",
    )
    target_crs: Optional[str] = Field(
        default=None,
        description="Target CRS for output (e.g., 'EPSG:32632'). None = keep original.",
    )
    auto_detect: bool = Field(
        default=True,
        description="Auto-detect layers based on naming conventions",
    )


class GISLoadResponse(BaseModel):
    """Response from loading a GIS file."""

    success: bool
    boundary: Optional[List[List[float]]] = Field(
        default=None,
        description="Site boundary coordinates (closed polygon)",
    )
    boundary_area: float = Field(
        default=0.0,
        description="Boundary area in source units (sq meters if CRS is projected)",
    )
    entrances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Entrances loaded from file",
    )
    keepouts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Keepout zones loaded from file",
    )
    source_crs: Optional[str] = Field(
        default=None,
        description="Source coordinate reference system",
    )
    layers_found: List[str] = Field(
        default_factory=list,
        description="All layers found in the file",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated during loading",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if loading failed",
    )


class GISLayerInfo(BaseModel):
    """Information about a layer in a GIS file."""

    name: str
    geometry_type: Optional[str] = None
    feature_count: Optional[int] = None
    crs: Optional[str] = None
    properties: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class GISListLayersResponse(BaseModel):
    """Response from listing layers in a GIS file."""

    success: bool
    file_path: str
    layers: List[GISLayerInfo] = Field(default_factory=list)
    error: Optional[str] = None
