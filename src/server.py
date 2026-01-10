"""FastMCP server for site-fit generation.

Exposes MCP tools for generating site layout solutions and integrates
with FastAPI for static file serving (viewer).
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from mcp.server.fastmcp import FastMCP

from .pipeline import generate_site_fits
from .response_filters import (
    DetailLevel,
    filter_metrics,
    filter_solution_summary,
    filter_statistics,
    filter_topology_result,
)
from .tools.sitefit_tools import (
    SiteFitRequest,
)

# Path to static files
STATIC_DIR = Path(__file__).parent.parent / "static"

# Persistence configuration
PERSISTENCE_DIR = Path.home() / ".sitefit" / "jobs"
SCHEMA_VERSION = "1.0"
DEFAULT_RETENTION_DAYS = 7

# Configure structured logging (JSON to stderr for MCP compatibility)
# stdio servers must NOT log to stdout as it interferes with JSON-RPC
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Ensure stdlib logging goes to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)

logger = structlog.get_logger(__name__)

# Create MCP server (following Python naming convention: {service}_mcp)
mcp = FastMCP(
    name="sitefit_mcp",
    instructions="Generate multiple feasible site layouts for wastewater/biogas facilities. "
    "Use sitefit_generate to create layouts, sitefit_get_solution for details, "
    "and sitefit_list_solutions to paginate through results.",
)

# In-memory storage for jobs and solutions
_jobs: dict[str, dict[str, Any]] = {}
_solutions: dict[str, Any] = {}


# =============================================================================
# Persistence Functions (atomic writes to ~/.sitefit/jobs/)
# =============================================================================

def _persist_job(job_id: str, job_data: dict) -> None:
    """Persist job and its solutions to disk with atomic writes.

    Uses temp file + rename pattern to prevent corruption on crash.
    Each job is stored in its own directory with:
    - job.json: Job metadata and request
    - {solution_id}.json: Each solution's full data
    """
    try:
        job_dir = PERSISTENCE_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Add schema version and timestamp for future migration
        persist_data = {
            **job_data,
            "_schema_version": SCHEMA_VERSION,
            "_persisted_at": datetime.utcnow().isoformat(),
        }

        # Atomic write: temp file + rename
        target = job_dir / "job.json"
        with tempfile.NamedTemporaryFile(
            mode='w', dir=job_dir, delete=False, suffix='.tmp'
        ) as tmp:
            json.dump(persist_data, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.rename(tmp.name, target)

        # Persist each solution
        for sol_id in job_data.get("solution_ids", []):
            if sol_id in _solutions:
                _persist_solution(job_dir, sol_id, _solutions[sol_id])

        logger.info(
            "job_persisted",
            job_id=job_id,
            solution_count=len(job_data.get("solution_ids", [])),
            path=str(job_dir),
        )

    except Exception as e:
        logger.warning("job_persist_failed", job_id=job_id, error=str(e))


def _persist_solution(job_dir: Path, sol_id: str, sol_data: dict) -> None:
    """Persist a single solution with atomic write."""
    try:
        target = job_dir / f"{sol_id}.json"
        with tempfile.NamedTemporaryFile(
            mode='w', dir=job_dir, delete=False, suffix='.tmp'
        ) as tmp:
            json.dump(sol_data, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.rename(tmp.name, target)
    except Exception as e:
        logger.warning("solution_persist_failed", solution_id=sol_id, error=str(e))


def _load_job(job_id: str) -> dict | None:
    """Load job from disk if exists, restoring to memory.

    Returns None if job doesn't exist on disk.
    Also loads all associated solutions into _solutions dict.
    """
    job_dir = PERSISTENCE_DIR / job_id
    job_file = job_dir / "job.json"

    if not job_file.exists():
        return None

    try:
        with open(job_file) as f:
            job_data = json.load(f)

        # Strip internal fields before returning
        job_data.pop("_schema_version", None)
        job_data.pop("_persisted_at", None)

        # Load all solutions for this job
        for sol_id in job_data.get("solution_ids", []):
            sol_file = job_dir / f"{sol_id}.json"
            if sol_file.exists() and sol_id not in _solutions:
                with open(sol_file) as f:
                    _solutions[sol_id] = json.load(f)

        # Restore to memory
        _jobs[job_id] = job_data
        logger.info(
            "job_loaded_from_disk",
            job_id=job_id,
            solution_count=len(job_data.get("solution_ids", [])),
        )

        return job_data

    except Exception as e:
        logger.warning("job_load_failed", job_id=job_id, error=str(e))
        return None


def _cleanup_old_jobs(max_age_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Remove jobs older than max_age_days.

    Returns the number of jobs cleaned up.
    """
    if not PERSISTENCE_DIR.exists():
        return 0

    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    cleaned = 0

    try:
        for job_dir in PERSISTENCE_DIR.iterdir():
            if not job_dir.is_dir():
                continue

            job_file = job_dir / "job.json"
            if not job_file.exists():
                continue

            try:
                with open(job_file) as f:
                    data = json.load(f)
                persisted_str = data.get("_persisted_at", "2000-01-01T00:00:00")
                persisted = datetime.fromisoformat(persisted_str.replace("Z", "+00:00").replace("+00:00", ""))

                if persisted < cutoff:
                    shutil.rmtree(job_dir)
                    cleaned += 1
                    logger.info("job_cleaned_up", job_id=job_dir.name)
            except Exception as e:
                logger.warning("job_cleanup_check_failed", job_id=job_dir.name, error=str(e))

    except Exception as e:
        logger.warning("jobs_cleanup_failed", error=str(e))

    return cleaned


def _try_load_from_disk(job_id: str) -> bool:
    """Try to load a job from disk if not in memory.

    Returns True if job was loaded (or already in memory).
    """
    if job_id in _jobs:
        return True
    return _load_job(job_id) is not None


@mcp.tool(
    annotations={
        "readOnlyHint": False,  # Creates and stores solutions
        "destructiveHint": False,  # Does not delete existing data
        "idempotentHint": True,  # Same seed produces same results
        "openWorldHint": False,  # Does not call external services
    }
)
async def sitefit_generate(
    site_boundary: list[list[float]],
    structures: list[dict[str, Any]],
    entrances: list[dict[str, Any]] | None = None,
    keepouts: list[dict[str, Any]] | None = None,
    existing: list[dict[str, Any]] | None = None,
    sfiles2: str | None = None,
    rules_override: dict[str, Any] | None = None,
    max_solutions: int = 5,
    max_time_seconds: float = 60.0,
    seed: int = 42,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Generate site layout solutions for wastewater/biogas facilities.

    Creates multiple diverse placement solutions using constraint programming
    (OR-Tools CP-SAT) with Shapely geometry validation and A* road routing.

    Args:
        site_boundary: Site boundary polygon as [[x,y], ...] coordinates (meters)
        structures: List of structures with id, type, footprint {shape, w, h or d}
        entrances: Site entrances with {id, point: [x,y], width}
        keepouts: Keep-out zones with {id, geometry: GeoJSON, reason}
        existing: Existing structures for brownfield sites with {id, footprint: GeoJSON,
                  clearance_required, is_tie_in_point}
        sfiles2: Optional SFILES2 string for process topology constraints
        rules_override: Optional rule overrides for setbacks/clearances
        max_solutions: Maximum solutions to return (1-50, default 5)
        max_time_seconds: Maximum solve time (5-600 seconds, default 60)
        seed: Random seed for reproducibility (default 42)
        detail_level: Response detail - "compact" (default) for essential fields,
                      "full" for all metrics and statistics

    Returns:
        Dict with job_id, status, num_solutions, solutions list, and statistics
    """
    # Build request
    request = SiteFitRequest(
        site={
            "boundary": site_boundary,
            "entrances": entrances or [],
            "keepouts": keepouts or [],
            "existing": existing or [],
        },
        topology={"sfiles2": sfiles2} if sfiles2 else None,
        program={"structures": structures},
        rules_override=rules_override,
        generation={
            "max_solutions": max_solutions,
            "max_time_seconds": max_time_seconds,
            "seed": seed,
        },
    )

    # Run generation
    try:
        solutions, stats = await generate_site_fits(request)

        # Store solutions and original request for contract export
        job_id = stats.get("job_id", "unknown")
        job_data = {
            "status": "completed",
            "stats": stats,
            "solution_ids": [s.id for s in solutions],
            "request": request.model_dump(),  # Store for contract export
        }
        _jobs[job_id] = job_data

        for sol in solutions:
            _solutions[sol.id] = sol.model_dump()

        # Persist to disk for recovery after restart
        _persist_job(job_id, job_data)

        # Build response with filtered fields based on detail_level
        solution_summaries = []
        for s in solutions:
            summary = {
                "id": s.id,
                "rank": s.rank,
                "metrics": filter_metrics(s.metrics.model_dump(), detail_level),
            }
            if detail_level == "full":
                summary["diversity_note"] = s.diversity_note
            solution_summaries.append(summary)

        response = {
            "job_id": job_id,
            "status": "completed",
            "num_solutions": len(solutions),
            "solutions": solution_summaries,
        }

        # Include statistics based on detail_level
        if detail_level == "full":
            response["statistics"] = stats
        else:
            response["statistics"] = filter_statistics(stats, detail_level)

        return response

    except Exception as e:
        logger.exception("generation_failed", error=str(e))
        return {
            "isError": True,
            "job_id": "error",
            "status": "failed",
            "error": str(e),
            "suggestion": "Check site_boundary is a valid closed polygon and structures have valid footprints",
            "num_solutions": 0,
            "solutions": [],
            "statistics": {},
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,  # Creates and stores solutions
        "destructiveHint": False,  # Does not delete existing data
        "idempotentHint": True,  # Same seed produces same results
        "openWorldHint": False,  # Does not call external services
    }
)
async def sitefit_generate_from_request(
    request: dict[str, Any],
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Generate site layout solutions from a complete SiteFitRequest object.

    Alternative to sitefit_generate that accepts the full nested request schema.
    Useful when programmatically constructing requests or integrating with other systems.

    Args:
        request: Full SiteFitRequest object with site, program, topology, rules_override, generation
        detail_level: Response detail - "compact" (default) for essential fields,
                      "full" for all metrics and statistics

    Returns:
        Dict with job_id, status, num_solutions, solutions list, and statistics

    Example request:
        {
            "site": {"boundary": [[0,0], [100,0], [100,80], [0,80], [0,0]], "entrances": [], "keepouts": []},
            "program": {"structures": [{"id": "TK-001", "type": "tank", "footprint": {"shape": "circle", "d": 12}}]},
            "topology": {"sfiles2": "(tank)", "node_map": {"tank": "TK-001"}},
            "generation": {"max_solutions": 5, "seed": 42}
        }
    """
    try:
        # Validate and construct request
        site_fit_request = SiteFitRequest(**request)

        # Run generation
        solutions, stats = await generate_site_fits(site_fit_request)

        # Store solutions and original request for contract export
        job_id = stats.get("job_id", "unknown")
        job_data = {
            "status": "completed",
            "stats": stats,
            "solution_ids": [s.id for s in solutions],
            "request": site_fit_request.model_dump(),  # Store for contract export
        }
        _jobs[job_id] = job_data

        for sol in solutions:
            _solutions[sol.id] = sol.model_dump()

        # Persist to disk for recovery after restart
        _persist_job(job_id, job_data)

        # Build response with filtered fields based on detail_level
        solution_summaries = []
        for s in solutions:
            summary = {
                "id": s.id,
                "rank": s.rank,
                "metrics": filter_metrics(s.metrics.model_dump(), detail_level),
            }
            if detail_level == "full":
                summary["diversity_note"] = s.diversity_note
            solution_summaries.append(summary)

        response = {
            "job_id": job_id,
            "status": "completed",
            "num_solutions": len(solutions),
            "solutions": solution_summaries,
        }

        # Include statistics based on detail_level
        if detail_level == "full":
            response["statistics"] = stats
        else:
            response["statistics"] = filter_statistics(stats, detail_level)

        return response

    except Exception as e:
        logger.exception("generation_from_request_failed", error=str(e))
        return {
            "isError": True,
            "job_id": "error",
            "status": "failed",
            "error": str(e),
            "suggestion": "Validate request matches SiteFitRequest schema",
            "num_solutions": 0,
            "solutions": [],
            "statistics": {},
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,  # Only reads stored data
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_get_solution(
    solution_id: str,
    include_geojson: bool = True,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Get full details of a specific solution by ID.

    Retrieves placement coordinates, metrics, road network, and optional
    GeoJSON feature collection for visualization.

    Args:
        solution_id: Solution ID from sitefit_generate response
        include_geojson: Include GeoJSON feature collection (default True)
        detail_level: Response detail - "compact" (default) for essential fields,
                      "full" for all metrics and metadata

    Returns:
        Full solution with placements, metrics, road_network, and features_geojson
    """
    # Try to load from disk if not in memory
    if solution_id not in _solutions:
        # Solution ID format is typically "{job_id}-{index}"
        # Try to extract job_id and load from disk
        if "-" in solution_id:
            job_id = solution_id.rsplit("-", 1)[0]
            _try_load_from_disk(job_id)

    if solution_id not in _solutions:
        return {
            "isError": True,
            "error": f"Solution {solution_id} not found",
            "suggestion": "Use sitefit_list_solutions to get valid solution IDs from a job",
        }

    solution = _solutions[solution_id]

    # Build filtered response
    result = {
        "id": solution.get("id"),
        "job_id": solution.get("job_id"),
        "rank": solution.get("rank"),
        "placements": solution.get("placements", []),  # Always include full placements
        "road_network": solution.get("road_network"),
        "metrics": filter_metrics(solution.get("metrics", {}), detail_level),
    }

    # Add optional fields for full mode
    if detail_level == "full":
        result["created_at"] = solution.get("created_at")
        result["diversity_note"] = solution.get("diversity_note")

    # Include GeoJSON if requested
    if include_geojson and "features_geojson" in solution:
        result["features_geojson"] = solution["features_geojson"]

    return result


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_list_solutions(
    job_id: str,
    limit: int = 20,
    offset: int = 0,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """List solutions for a job with pagination.

    Returns summaries (id, rank, metrics) for quick browsing.
    Use sitefit_get_solution for full details.

    Args:
        job_id: Job ID from sitefit_generate
        limit: Maximum solutions to return (default 20)
        offset: Offset for pagination (default 0)
        detail_level: Response detail - "compact" (default) for essential fields,
                      "full" for all metrics

    Returns:
        Dict with job_id, total, offset, limit, solutions, has_more, next_offset
    """
    # Try to load from disk if not in memory
    _try_load_from_disk(job_id)

    if job_id not in _jobs:
        return {
            "isError": True,
            "error": f"Job {job_id} not found",
            "suggestion": "Use sitefit_generate to create a job first",
            "solutions": [],
        }

    job = _jobs[job_id]
    solution_ids = job.get("solution_ids", [])

    # Apply pagination
    paginated_ids = solution_ids[offset : offset + limit]

    solutions = []
    for sol_id in paginated_ids:
        if sol_id in _solutions:
            sol = _solutions[sol_id]
            summary = {
                "id": sol["id"],
                "rank": sol["rank"],
                "metrics": filter_metrics(sol.get("metrics", {}), detail_level),
            }
            if detail_level == "full":
                summary["diversity_note"] = sol.get("diversity_note")
            solutions.append(summary)

    total = len(solution_ids)
    has_more = offset + limit < total

    return {
        "job_id": job_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "count": len(solutions),
        "solutions": solutions,
        "has_more": has_more,
        "next_offset": offset + limit if has_more else None,
    }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_job_status(
    job_id: str,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Get the status of a site-fit generation job.

    Check if a job is queued, running, completed, or failed.
    For completed jobs, includes solution count and summary statistics.

    Args:
        job_id: Job ID from sitefit_generate
        detail_level: Response detail - "compact" (default) for essential stats,
                      "full" for all statistics

    Returns:
        Dict with job_id, status, progress (0-100), and job details
    """
    # Try to load from disk if not in memory
    _try_load_from_disk(job_id)

    if job_id not in _jobs:
        return {
            "isError": True,
            "error": f"Job {job_id} not found",
            "suggestion": "Use sitefit_generate to create a job first",
        }

    job = _jobs[job_id]
    status = job.get("status", "unknown")
    stats = job.get("stats", {})

    result = {
        "job_id": job_id,
        "status": status,
        "progress": 100 if status == "completed" else (0 if status == "failed" else 50),
    }

    if status == "completed":
        result["num_solutions"] = len(job.get("solution_ids", []))
        # Only include statistics in full mode per plan
        if detail_level == "full":
            result["statistics"] = stats
    elif status == "failed":
        result["error"] = job.get("error", "Unknown error")

    return result


@mcp.tool(
    annotations={
        "readOnlyHint": True,  # Only reads and formats existing data
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_export(
    solution_id: str,
    format: str = "geojson",
    include_roads: bool = True,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Export a solution to various formats.

    Converts solution data to GeoJSON, SVG, contract, or summary formats
    for use in external tools and viewers.

    Args:
        solution_id: Solution ID from sitefit_generate
        format: Export format - 'geojson', 'svg', 'contract', or 'summary' (default: geojson)
        include_roads: Include road network in contract format (default: True)
        detail_level: Response detail - "compact" (default) omits extra metadata,
                      "full" includes all metadata

    Returns:
        Dict with format, data (content), and metadata
    """
    # Try to load from disk if not in memory
    if solution_id not in _solutions:
        if "-" in solution_id:
            job_id = solution_id.rsplit("-", 1)[0]
            _try_load_from_disk(job_id)

    if solution_id not in _solutions:
        return {
            "isError": True,
            "error": f"Solution {solution_id} not found",
            "suggestion": "Use sitefit_list_solutions to get valid solution IDs",
        }

    solution = _solutions[solution_id]
    format = format.lower()

    if format == "geojson":
        result = {
            "format": "geojson",
            "content_type": "application/geo+json",
            "data": solution.get("features_geojson", {}),
        }
        if detail_level == "full":
            result["metadata"] = {
                "solution_id": solution_id,
                "rank": solution.get("rank"),
            }
        return result

    elif format == "svg":
        # Use SVG export module
        try:
            from shapely.geometry import Polygon

            from .export.svg import export_solution_to_svg
            from .models.solution import SiteFitSolution

            # Reconstruct solution object
            sol = SiteFitSolution(**solution)

            # Get boundary from GeoJSON
            boundary_feature = next(
                (f for f in solution.get("features_geojson", {}).get("features", [])
                 if f.get("properties", {}).get("kind") == "boundary"),
                None
            )

            if boundary_feature:
                boundary = Polygon(boundary_feature["geometry"]["coordinates"][0])
            else:
                # Fallback to bounding box of all features
                boundary = Polygon([[0, 0], [200, 0], [200, 150], [0, 150], [0, 0]])

            svg_content = export_solution_to_svg(
                solution=sol,
                boundary=boundary,
                show_labels=True,
                show_roads=True,
            )

            result = {
                "format": "svg",
                "content_type": "image/svg+xml",
                "data": svg_content,
            }
            if detail_level == "full":
                result["metadata"] = {
                    "solution_id": solution_id,
                    "rank": solution.get("rank"),
                }
            return result
        except Exception as e:
            logger.exception("svg_export_failed", solution_id=solution_id, error=str(e))
            return {
                "isError": True,
                "error": f"SVG export failed: {str(e)}",
                "suggestion": "Try exporting as 'geojson' instead",
            }

    elif format == "summary":
        metrics = solution.get("metrics", {})
        placements = solution.get("placements", [])

        data = {
            "solution_id": solution_id,
            "rank": solution.get("rank"),
            "num_structures": len(placements),
            "metrics": filter_metrics(metrics, detail_level),
            "has_road_network": solution.get("road_network") is not None,
            "structure_ids": [p.get("structure_id") for p in placements],
        }
        if detail_level == "full":
            data["diversity_note"] = solution.get("diversity_note")

        result = {
            "format": "summary",
            "content_type": "application/json",
            "data": data,
        }
        if detail_level == "full":
            result["metadata"] = {"solution_id": solution_id}
        return result

    elif format == "contract":
        # Contract format for FreeCAD integration
        # Find the job that contains this solution to get original request
        job_request = None
        for job_id, job_data in _jobs.items():
            if solution_id in job_data.get("solution_ids", []):
                job_request = job_data.get("request")
                break

        # Extract site info from request or GeoJSON
        site_data: dict[str, Any] = {"boundary": [], "entrances": [], "keepouts": []}
        structures_data: list[dict[str, Any]] = []

        if job_request:
            site_data = job_request.get("site", {})
            structures_data = job_request.get("program", {}).get("structures", [])
        else:
            # Fallback: extract from GeoJSON features (e.g., after server restart)
            for feature in solution.get("features_geojson", {}).get("features", []):
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                kind = props.get("kind")

                if kind == "boundary" and geom.get("type") == "Polygon":
                    site_data["boundary"] = geom["coordinates"][0]
                elif kind == "entrance":
                    site_data["entrances"].append({
                        "id": props.get("id", ""),
                        "point": geom.get("coordinates", []),
                        "width": props.get("width", 6.0),
                    })
                elif kind == "keepout":
                    site_data["keepouts"].append({
                        "id": props.get("id", ""),
                        "geometry": geom,
                        "reason": props.get("reason", ""),
                    })
                elif kind == "structure":
                    # Extract structure dimensions from GeoJSON geometry
                    struct_id = props.get("id", props.get("structure_id", ""))
                    struct_type = props.get("type", "unknown")
                    height = props.get("height", 5.0)
                    dome_height_m = props.get("dome_height_m")  # May be None

                    # Determine shape and dimensions from geometry
                    if geom.get("type") == "Polygon":
                        coords = geom.get("coordinates", [[]])[0]
                        if len(coords) >= 4:
                            # Calculate bounding box
                            xs = [c[0] for c in coords]
                            ys = [c[1] for c in coords]
                            width = max(xs) - min(xs)
                            length = max(ys) - min(ys)

                            # Check if roughly circular (width ~= length and many points)
                            if len(coords) > 8 and abs(width - length) < 0.1 * max(width, length):
                                footprint = {"shape": "circle", "d": (width + length) / 2}
                            else:
                                footprint = {"shape": "rect", "w": width, "h": length}

                            struct_dict = {
                                "id": struct_id,
                                "type": struct_type,
                                "footprint": footprint,
                                "height": height,
                            }
                            # Include dome_height_m if present (for digester dome covers)
                            if dome_height_m is not None:
                                struct_dict["dome_height_m"] = dome_height_m
                            structures_data.append(struct_dict)

        # Build placements with 'id' instead of 'structure_id' for FreeCAD
        placements = solution.get("placements", [])
        contract_placements = [
            {
                "id": p.get("structure_id"),
                "x": p.get("x"),
                "y": p.get("y"),
                "rotation_deg": p.get("rotation_deg", 0),
            }
            for p in placements
        ]

        # Build road network if requested
        road_network_data = None
        if include_roads and solution.get("road_network"):
            rn = solution["road_network"]
            road_network_data = {
                "segments": [
                    {
                        "id": seg.get("id"),
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "width": seg.get("width", 6.0),
                        "waypoints": seg.get("waypoints", []),
                        "connects_to": seg.get("connects_to", []),
                    }
                    for seg in rn.get("segments", [])
                ],
                "total_length": rn.get("total_length", 0.0),
                "entrances_connected": rn.get("entrances_connected", []),
                "structures_accessible": rn.get("structures_accessible", []),
            }

        # Build contract data - always include essential fields for FreeCAD
        # Uses Spatial Contract v1.0 schema
        contract_data = {
            "contract_version": "1.0.0",
            "project": {
                "name": "",
                "id": solution_id,
                "revision": "A",
            },
            "site": {
                "boundary": site_data.get("boundary", []),
                "units": "meters",
                "crs": "local",
                "entrances": site_data.get("entrances", []),
                "keepouts": site_data.get("keepouts", []),
            },
            "program": {
                "structures": structures_data,
            },
            "placements": contract_placements,
            "road_network": road_network_data,
            "provenance": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "solver_version": "1.0.0",
                "solution_id": solution_id,
                "job_id": job_id if job_id else "",
            },
        }

        # Add metrics if available
        if solution.get("metrics"):
            contract_data["metrics"] = solution["metrics"]

        # Only include extended metadata in full mode
        if detail_level == "full":
            contract_data["provenance"]["source"] = "sitefit_mcp"
            contract_data["provenance"]["rank"] = solution.get("rank", 0)

        result = {
            "format": "contract",
            "content_type": "application/json",
            "data": contract_data,
        }
        if detail_level == "full":
            result["metadata"] = {
                "solution_id": solution_id,
                "rank": solution.get("rank"),
            }
        return result

    else:
        return {
            "isError": True,
            "error": f"Unknown format '{format}'",
            "suggestion": "Valid formats are: 'geojson', 'svg', 'contract', 'summary'",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_export_contract(
    solution_id: str,
    include_roads: bool = True,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Export a solution as contract JSON for FreeCAD integration.

    Convenience wrapper for sitefit_export(format='contract') that produces
    a contract JSON ready for direct use with freecad-mcp's import_sitefit_contract.

    The contract includes:
    - Site boundary, entrances, and keepouts
    - Structure definitions with dimensions (from original request)
    - Placements with 'id' (mapped from structure_id) for FreeCAD compatibility
    - Road network geometry (optional)

    Args:
        solution_id: Solution ID from sitefit_generate
        include_roads: Include road network geometry (default: True)
        detail_level: Response detail - "compact" (default) for essential fields,
            "full" for all fields including metadata

    Returns:
        Contract JSON with project, site, program, placements, road_network, metadata
    """
    return await sitefit_export(solution_id, format="contract", include_roads=include_roads, detail_level=detail_level)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_export_batch(
    job_id: str,
    solution_ids: list[str] | None = None,
    include_roads: bool = True,
    include_structures: bool = True,
    limit: int = 10,
    offset: int = 0,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Export multiple solutions from a job as contract JSONs in one call.

    Eliminates the need for agents to loop through sitefit_get_solution for each
    solution. Returns complete contracts ready for FreeCAD's present_layout_options.

    Each contract includes:
    - Site boundary, entrances, and keepouts
    - Structure definitions with dimensions
    - Placements with coordinates and rotations
    - Road network geometry (optional)

    Args:
        job_id: Job ID from sitefit_generate
        solution_ids: Optional list of specific solution IDs to export.
                      If None, exports all solutions from the job.
        include_roads: Include road network in each contract (default: True)
        include_structures: Include structure definitions (default: True)
        limit: Maximum contracts to return per call (default: 10, max: 50)
        offset: Offset for pagination (default: 0)
        detail_level: Response detail - "compact" (default) for essential fields,
                      "full" for all fields including metadata

    Returns:
        Dict with contracts array, total count, pagination info, and has_more flag
    """
    # Try to load from disk if not in memory
    _try_load_from_disk(job_id)

    if job_id not in _jobs:
        return {
            "isError": True,
            "error": f"Job {job_id} not found",
            "suggestion": "Use sitefit_generate to create a job first",
            "contracts": [],
        }

    job = _jobs[job_id]
    all_solution_ids = job.get("solution_ids", [])

    # Filter to requested solutions if specified
    if solution_ids:
        target_ids = [sid for sid in solution_ids if sid in all_solution_ids]
        if not target_ids:
            return {
                "isError": True,
                "error": "None of the specified solution_ids belong to this job",
                "suggestion": f"Valid solution IDs for job {job_id}: {all_solution_ids[:5]}...",
                "contracts": [],
            }
    else:
        target_ids = all_solution_ids

    # Apply pagination
    total = len(target_ids)
    limit = min(limit, 50)  # Cap at 50 to prevent payload explosion
    paginated_ids = target_ids[offset : offset + limit]
    has_more = offset + limit < total

    # Build contracts for each solution
    contracts = []
    errors = []

    for sol_id in paginated_ids:
        try:
            # Use existing export function
            result = await sitefit_export(
                solution_id=sol_id,
                format="contract",
                include_roads=include_roads,
                detail_level=detail_level,
            )

            if result.get("isError"):
                errors.append({"solution_id": sol_id, "error": result.get("error")})
                continue

            contract_data = result.get("data", {})

            # Optionally strip structures to reduce payload
            if not include_structures and "program" in contract_data:
                contract_data["program"]["structures"] = []

            # Add solution metadata for FreeCAD layer creation
            # Include both underscore-prefixed (contract internal) and plain names
            # for FreeCAD's present_layout_options/import_solutions_as_layers compatibility
            contract_data["_solution_id"] = sol_id
            contract_data["solution_id"] = sol_id  # FreeCAD alias
            if sol_id in _solutions:
                contract_data["_rank"] = _solutions[sol_id].get("rank", 0)
                contract_data["rank"] = _solutions[sol_id].get("rank", 0)  # FreeCAD alias
                contract_data["_metrics"] = _solutions[sol_id].get("metrics", {})
                contract_data["metrics"] = _solutions[sol_id].get("metrics", {})  # FreeCAD alias

            # Add top-level structures alias from program.structures for FreeCAD
            if "program" in contract_data and "structures" in contract_data["program"]:
                contract_data["structures"] = contract_data["program"]["structures"]

            contracts.append(contract_data)

        except Exception as e:
            logger.warning("batch_export_solution_failed", solution_id=sol_id, error=str(e))
            errors.append({"solution_id": sol_id, "error": str(e)})

    return {
        "job_id": job_id,
        "contracts": contracts,
        "total": total,
        "count": len(contracts),
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
        "next_offset": offset + limit if has_more else None,
        "errors": errors if errors else None,
    }


@mcp.tool(
    annotations={
        "readOnlyHint": False,  # Creates files on disk
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_export_pack(
    solution_id: str,
    formats: list[str] = ["geojson", "csv"],
    output_dir: str | None = None,
    project_name: str = "",
    drawing_number: str = "",
) -> dict[str, Any]:
    """Export solution as a complete deliverable package with multiple formats.

    Generates bundled exports for proposal exhibits and civil handoff.
    This is the GUARANTEED headless export method - works without FreeCAD GUI.

    Available formats:
    - pdf: Plan sheet with layout, scale bar, legend, quantities (uses ReportLab - headless safe)
    - dxf: CAD file with layers: BOUNDARY, BUILDLIMIT, KEEPOUTS, STRUCTURES, ROADS
    - csv: ROM quantities - pad areas, road length, pipe proxies, fence perimeter
    - geojson: GeoJSON for visualization and GIS integration
    - svg: SVG vector graphics for web viewing (no external dependencies)

    Args:
        solution_id: Solution ID from sitefit_generate
        formats: List of formats to generate - 'pdf', 'dxf', 'csv', 'geojson', 'svg' (default: csv, geojson)
        output_dir: Directory for output files (uses temp directory if not specified)
        project_name: Project name for PDF title block
        drawing_number: Drawing number for PDF title block

    Returns:
        Dict with success, files (format -> path mapping), quantities, and any errors

    Examples:
        Export full deliverable pack:
        ```json
        {
            "solution_id": "job_abc123-0",
            "formats": ["pdf", "dxf", "csv", "geojson", "svg"],
            "project_name": "Acme WWTP Expansion",
            "drawing_number": "100-GA-001"
        }
        ```
    """
    # Try to load from disk if not in memory
    if solution_id not in _solutions:
        if "-" in solution_id:
            job_id = solution_id.rsplit("-", 1)[0]
            _try_load_from_disk(job_id)

    if solution_id not in _solutions:
        return {
            "isError": True,
            "error": f"Solution {solution_id} not found",
            "suggestion": "Use sitefit_list_solutions to get valid solution IDs",
        }

    try:
        from shapely.geometry import Polygon

        from .export.pack import export_pack
        from .models.solution import SiteFitSolution

        # Reconstruct solution object
        solution_data = _solutions[solution_id]
        sol = SiteFitSolution(**solution_data)

        # Get boundary from GeoJSON
        boundary_feature = next(
            (f for f in solution_data.get("features_geojson", {}).get("features", [])
             if f.get("properties", {}).get("kind") == "boundary"),
            None
        )

        if boundary_feature:
            boundary = Polygon(boundary_feature["geometry"]["coordinates"][0])
        else:
            # Fallback: compute bounding box from placements
            placements = solution_data.get("placements", [])
            if placements:
                xs = [p["x"] for p in placements]
                ys = [p["y"] for p in placements]
                margin = 20
                boundary = Polygon([
                    [min(xs) - margin, min(ys) - margin],
                    [max(xs) + margin, min(ys) - margin],
                    [max(xs) + margin, max(ys) + margin],
                    [min(xs) - margin, max(ys) + margin],
                    [min(xs) - margin, min(ys) - margin],
                ])
            else:
                boundary = Polygon([[0, 0], [200, 0], [200, 150], [0, 150], [0, 0]])

        # Extract structure types from features
        structure_types = {}
        for feature in solution_data.get("features_geojson", {}).get("features", []):
            props = feature.get("properties", {})
            if props.get("kind") == "structure":
                struct_id = props.get("id")
                struct_type = props.get("type", "unknown")
                if struct_id:
                    structure_types[struct_id] = struct_type

        # Run export pack
        result = export_pack(
            solution=sol,
            boundary=boundary,
            formats=formats,
            output_dir=output_dir,
            structure_types=structure_types,
            project_name=project_name or f"Site Layout {solution_id}",
            drawing_number=drawing_number or solution_id[:8].upper(),
        )

        return {
            "success": result.success,
            "solution_id": solution_id,
            "formats_generated": result.formats_generated,
            "files": result.files,
            "quantities": result.quantities,
            "errors": result.errors if result.errors else None,
        }

    except ImportError as e:
        return {
            "isError": True,
            "error": f"Export module not available: {e}",
            "suggestion": "Some formats require optional dependencies. Install with: pip install 'site-fit-mcp[export]'",
        }
    except Exception as e:
        logger.exception("export_pack_failed", solution_id=solution_id, error=str(e))
        return {
            "isError": True,
            "error": f"Export failed: {str(e)}",
            "suggestion": "Check that the solution has valid geometry data",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def ruleset_list() -> dict[str, Any]:
    """List available rulesets for site layout generation.

    Returns names and descriptions of available engineering rule configurations.

    Returns:
        Dict with rulesets array containing {name, description} objects
    """
    from .rules.loader import list_rulesets

    try:
        rulesets = list_rulesets()
        return {
            "rulesets": rulesets,
            "count": len(rulesets),
        }
    except Exception as e:
        logger.exception("rulesets_list_failed", error=str(e))
        return {
            "isError": True,
            "error": str(e),
            "rulesets": [],
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def ruleset_get(
    name: str = "default",
) -> dict[str, Any]:
    """Get a ruleset configuration and JSON schema.

    Returns the full ruleset with setback/clearance values and the
    JSON schema for validation and UI generation.

    Args:
        name: Ruleset name (use ruleset_list to see available options)

    Returns:
        Dict with name, rules (configuration), and schema (JSON Schema)
    """
    from .rules.loader import load_ruleset

    try:
        rules = load_ruleset(name)
        return {
            "name": name,
            "rules": rules.model_dump(),
            "schema": rules.model_json_schema(),
        }
    except FileNotFoundError:
        return {
            "isError": True,
            "error": f"Ruleset '{name}' not found",
            "suggestion": "Use ruleset_list to see available rulesets",
        }
    except Exception as e:
        logger.exception("ruleset_load_failed", ruleset_name=name, error=str(e))
        return {
            "isError": True,
            "error": str(e),
            "suggestion": "Check ruleset YAML syntax",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,  # Only parses, doesn't store
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def topology_parse_sfiles2(
    sfiles2: str,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Parse and validate an SFILES2 process topology string.

    Tokenizes the SFILES2 notation and extracts nodes (equipment) and
    edges (connections) for validation before use in sitefit_generate.

    Args:
        sfiles2: SFILES2 string (e.g., "(influent)pump|P-101(tank)T-101")
        detail_level: Response detail - "compact" (default) for summary only,
                      "full" includes nodes, edges, and tokens arrays

    Returns:
        Dict with valid (bool), nodes, edges, tokens, num_nodes, num_edges.
        If invalid, includes error message.
    """
    from .topology.sfiles_parser import parse_sfiles_topology, tokenize_sfiles

    try:
        topology = parse_sfiles_topology(sfiles2)
        tokens = tokenize_sfiles(sfiles2)

        result = {
            "valid": True,
            "num_nodes": len(topology.nodes),
            "num_edges": len(topology.edges),
        }

        # Include full arrays only in full mode
        if detail_level == "full":
            result["nodes"] = [n.model_dump() for n in topology.nodes]
            result["edges"] = [e.model_dump() for e in topology.edges]
            result["tokens"] = tokens

        return result
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "num_nodes": 0,
            "num_edges": 0,
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,  # Only reads files
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_load_gis_file(
    file_path: str,
    boundary_layer: str | None = None,
    keepout_layers: list[str] | None = None,
    entrance_layer: str | None = None,
    target_crs: str | None = None,
    auto_detect: bool = True,
) -> dict[str, Any]:
    """Load site definition from a GIS file.

    Imports site boundary, keepout zones, and entrances from various GIS formats
    including Shapefile, GeoJSON, GeoPackage, KML, and File Geodatabase.

    Args:
        file_path: Path to the GIS file
        boundary_layer: Layer name for site boundary (auto-detect if None)
        keepout_layers: Layer names for keepout zones (auto-detect if None)
        entrance_layer: Layer name for entrance points (auto-detect if None)
        target_crs: Target CRS for output (e.g., 'EPSG:32632'). None = keep original.
        auto_detect: Auto-detect layers based on naming conventions

    Returns:
        Dict with boundary, keepouts, entrances, source_crs, layers_found, warnings
    """
    try:
        from .loaders.gis_loader import load_site_from_file
    except ImportError as e:
        return {
            "success": False,
            "error": "GIS loading requires fiona. Install with: pip install 'site-fit-mcp[gis]'",
            "import_error": str(e),
        }

    try:
        result = load_site_from_file(
            file_path=file_path,
            boundary_layer=boundary_layer,
            keepout_layers=keepout_layers,
            entrance_layer=entrance_layer,
            target_crs=target_crs,
            auto_detect=auto_detect,
        )

        return {
            "success": True,
            "boundary": result.boundary,
            "boundary_area": result.boundary_area,
            "entrances": result.entrances,
            "keepouts": result.keepouts,
            "source_crs": result.source_crs,
            "layers_found": result.layers_found,
            "warnings": result.warnings,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"File not found: {e}",
        }
    except Exception as e:
        logger.exception("gis_file_load_failed", file_path=file_path, error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_list_gis_layers(
    file_path: str,
) -> dict[str, Any]:
    """List all layers in a GIS file with metadata.

    Returns layer names, geometry types, feature counts, and CRS information.
    Useful for inspecting files before loading.

    Args:
        file_path: Path to the GIS file

    Returns:
        Dict with layers array containing name, geometry_type, feature_count, crs
    """
    try:
        from .loaders.gis_loader import list_gis_layers
    except ImportError as e:
        return {
            "success": False,
            "error": "GIS loading requires fiona. Install with: pip install 'site-fit-mcp[gis]'",
            "import_error": str(e),
        }

    try:
        layers = list_gis_layers(file_path)
        return {
            "success": True,
            "file_path": file_path,
            "layers": layers,
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"File not found: {e}",
        }
    except Exception as e:
        logger.exception("gis_layers_list_failed", file_path=file_path, error=str(e))
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def sitefit_compare_solutions(
    solution_id_a: str,
    solution_id_b: str,
    detail_level: DetailLevel = "compact",
) -> dict[str, Any]:
    """Compare two solutions and explain differences (Phase 5: Solution Diff).

    Useful for understanding why different solutions were ranked differently
    and what changed between layout options.

    Args:
        solution_id_a: First solution ID (typically lower rank)
        solution_id_b: Second solution ID (typically higher rank)
        detail_level: Response detail level ("compact" or "full")

    Returns:
        Comparison dict with summary, moved structures, metric deltas, and road changes
    """
    logger.info(
        "compare_solutions_started",
        solution_id_a=solution_id_a,
        solution_id_b=solution_id_b,
    )

    # Fetch both solutions
    sol_a = _solutions.get(solution_id_a)
    sol_b = _solutions.get(solution_id_b)

    if not sol_a:
        return {
            "success": False,
            "error": f"Solution not found: {solution_id_a}",
        }
    if not sol_b:
        return {
            "success": False,
            "error": f"Solution not found: {solution_id_b}",
        }

    # Get placements (dict by structure_id)
    placements_a = {p.get("structure_id", p.get("id", "")): p for p in sol_a.get("placements", [])}
    placements_b = {p.get("structure_id", p.get("id", "")): p for p in sol_b.get("placements", [])}

    # Find moved structures
    moved_structures = []
    rotation_changes = []

    all_structure_ids = set(placements_a.keys()) | set(placements_b.keys())

    for struct_id in all_structure_ids:
        p_a = placements_a.get(struct_id)
        p_b = placements_b.get(struct_id)

        if p_a and p_b:
            x_a, y_a = p_a.get("x", 0), p_a.get("y", 0)
            x_b, y_b = p_b.get("x", 0), p_b.get("y", 0)
            rot_a = p_a.get("rotation_deg", p_a.get("rotation", 0))
            rot_b = p_b.get("rotation_deg", p_b.get("rotation", 0))

            delta_x = round(x_b - x_a, 2)
            delta_y = round(y_b - y_a, 2)

            # Structure moved if delta > 0.1m
            if abs(delta_x) > 0.1 or abs(delta_y) > 0.1:
                moved_structures.append({
                    "id": struct_id,
                    "delta_x": delta_x,
                    "delta_y": delta_y,
                    "a_position": [round(x_a, 2), round(y_a, 2)],
                    "b_position": [round(x_b, 2), round(y_b, 2)],
                })

            # Rotation changed
            if rot_a != rot_b:
                rotation_changes.append({
                    "id": struct_id,
                    "from_deg": rot_a,
                    "to_deg": rot_b,
                })
        elif p_a and not p_b:
            moved_structures.append({
                "id": struct_id,
                "status": "removed_in_b",
                "a_position": [round(p_a.get("x", 0), 2), round(p_a.get("y", 0), 2)],
            })
        elif p_b and not p_a:
            moved_structures.append({
                "id": struct_id,
                "status": "added_in_b",
                "b_position": [round(p_b.get("x", 0), 2), round(p_b.get("y", 0), 2)],
            })

    # Compute metric deltas
    metrics_a = sol_a.get("metrics", {})
    metrics_b = sol_b.get("metrics", {})

    # Handle Pydantic model or dict
    if hasattr(metrics_a, "model_dump"):
        metrics_a = metrics_a.model_dump()
    if hasattr(metrics_b, "model_dump"):
        metrics_b = metrics_b.model_dump()

    metric_deltas = {}
    key_metrics = ["road_length", "compactness", "pipe_length_weighted", "site_utilization", "topology_penalty"]

    for key in key_metrics:
        val_a = metrics_a.get(key, 0) or 0
        val_b = metrics_b.get(key, 0) or 0
        delta = val_b - val_a
        if abs(delta) > 0.001:
            metric_deltas[key] = round(delta, 3)

    # Road network changes
    road_a = sol_a.get("road_network") or {}
    road_b = sol_b.get("road_network") or {}

    if hasattr(road_a, "model_dump"):
        road_a = road_a.model_dump()
    if hasattr(road_b, "model_dump"):
        road_b = road_b.model_dump()

    segments_a = road_a.get("segments", []) if road_a else []
    segments_b = road_b.get("segments", []) if road_b else []

    seg_ids_a = {s.get("id", f"seg_{i}") for i, s in enumerate(segments_a)}
    seg_ids_b = {s.get("id", f"seg_{i}") for i, s in enumerate(segments_b)}

    road_network_changes = {
        "segments_in_a": len(segments_a),
        "segments_in_b": len(segments_b),
        "segments_added": list(seg_ids_b - seg_ids_a),
        "segments_removed": list(seg_ids_a - seg_ids_b),
        "length_a": round(road_a.get("total_length", 0), 2) if road_a else 0,
        "length_b": round(road_b.get("total_length", 0), 2) if road_b else 0,
    }
    road_network_changes["length_delta"] = round(
        road_network_changes["length_b"] - road_network_changes["length_a"], 2
    )

    # Build summary
    summary_parts = []
    if moved_structures:
        summary_parts.append(f"{len(moved_structures)} structure(s) repositioned")
    if rotation_changes:
        summary_parts.append(f"{len(rotation_changes)} rotation change(s)")
    if road_network_changes["length_delta"] != 0:
        delta_str = f"+{road_network_changes['length_delta']}" if road_network_changes['length_delta'] > 0 else str(road_network_changes['length_delta'])
        summary_parts.append(f"road length {delta_str}m")
    if metric_deltas.get("compactness"):
        delta_str = f"+{metric_deltas['compactness']}" if metric_deltas['compactness'] > 0 else str(metric_deltas['compactness'])
        summary_parts.append(f"compactness {delta_str}")

    summary = "; ".join(summary_parts) if summary_parts else "Solutions are identical"

    result = {
        "success": True,
        "solution_a": solution_id_a,
        "solution_b": solution_id_b,
        "rank_a": sol_a.get("rank"),
        "rank_b": sol_b.get("rank"),
        "summary": summary,
        "moved_structures": moved_structures,
        "rotation_changes": rotation_changes,
        "metric_deltas": metric_deltas,
        "road_network_changes": road_network_changes,
    }

    # Add full metrics in full mode
    if detail_level == "full":
        result["metrics_a"] = metrics_a
        result["metrics_b"] = metrics_b

    logger.info(
        "compare_solutions_complete",
        solution_id_a=solution_id_a,
        solution_id_b=solution_id_b,
        moved_count=len(moved_structures),
        rotation_count=len(rotation_changes),
    )

    return result


def run_server():
    """Run the MCP server (MCP transport only)."""
    # Cleanup old jobs on startup (7-day retention policy)
    cleaned = _cleanup_old_jobs()
    if cleaned > 0:
        logger.info("startup_cleanup_complete", jobs_cleaned=cleaned)

    asyncio.run(mcp.run())


def run_with_static_server(host: str = "0.0.0.0", port: int = 8765):
    """Run MCP server with static file serving via FastAPI.

    This mode serves:
    - MCP endpoints at /mcp (SSE transport)
    - REST API at /api/* (for viewer)
    - Static viewer files at /

    Args:
        host: Host to bind to
        port: Port to listen on
    """
    # Cleanup old jobs on startup (7-day retention policy)
    cleaned = _cleanup_old_jobs()
    if cleaned > 0:
        logger.info("startup_cleanup_complete", jobs_cleaned=cleaned)

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.responses import FileResponse

    # Create FastAPI app
    app = FastAPI(
        title="Site-Fit MCP Server",
        description="Generate site layouts for wastewater/biogas facilities",
        version="0.1.0",
    )

    # Add CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount MCP SSE endpoint
    app.mount("/mcp", mcp.sse_app())

    # ========================================================================
    # REST API endpoints for viewer
    # ========================================================================

    @app.get("/api/jobs")
    async def api_list_jobs():
        """List all jobs."""
        jobs_list = []
        for job_id, job in _jobs.items():
            jobs_list.append({
                "job_id": job_id,
                "status": job.get("status", "unknown"),
                "num_solutions": len(job.get("solution_ids", [])),
            })
        return {"jobs": jobs_list}

    @app.get("/api/jobs/{job_id}")
    async def api_get_job(job_id: str):
        """Get job status and details."""
        result = await sitefit_job_status(job_id)
        if result.get("isError"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.get("/api/jobs/{job_id}/solutions")
    async def api_list_solutions(job_id: str, limit: int = 50, offset: int = 0):
        """List solutions for a job."""
        result = await sitefit_list_solutions(job_id, limit, offset)
        if result.get("isError"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.get("/api/solutions/{solution_id}")
    async def api_get_solution(solution_id: str, include_geojson: bool = True):
        """Get full solution details."""
        result = await sitefit_get_solution(solution_id, include_geojson)
        if result.get("isError"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.get("/api/solutions/{solution_id}/export/{format}")
    async def api_export_solution(solution_id: str, format: str):
        """Export solution to various formats."""
        result = await sitefit_export(solution_id, format)
        if result.get("isError"):
            raise HTTPException(status_code=400, detail=result["error"])
        return result

    @app.post("/api/generate")
    async def api_generate(request: dict):
        """Generate site layouts (REST wrapper for MCP tool)."""
        try:
            result = await sitefit_generate(
                site_boundary=request.get("site_boundary", []),
                structures=request.get("structures", []),
                entrances=request.get("entrances"),
                keepouts=request.get("keepouts"),
                sfiles2=request.get("sfiles2"),
                rules_override=request.get("rules_override"),
                max_solutions=request.get("max_solutions", 5),
                max_time_seconds=request.get("max_time_seconds", 60.0),
                seed=request.get("seed", 42),
            )
            if result.get("isError"):
                raise HTTPException(status_code=400, detail=result["error"])
            return result
        except Exception as e:
            logger.exception("api_generate_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Static file serving (must come after API routes)
    # ========================================================================

    if STATIC_DIR.exists():
        @app.get("/")
        async def serve_index():
            """Serve the main viewer page."""
            return FileResponse(STATIC_DIR / "index.html")

        # Serve static files directly (CSS, JS)
        @app.get("/styles.css")
        async def serve_styles():
            return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")

        @app.get("/app.js")
        async def serve_app_js():
            return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")

    logger.info(
        "server_starting",
        host=host,
        port=port,
        viewer_url=f"http://{host}:{port}/",
        api_url=f"http://{host}:{port}/api/",
        mcp_url=f"http://{host}:{port}/mcp",
    )

    uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point.

    Supports different modes:
    - Default: MCP stdio transport (for MCP clients)
    - --serve: HTTP server with static files and MCP SSE
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        # Parse optional host:port
        host = "0.0.0.0"
        port = 8765

        if len(sys.argv) > 2:
            addr = sys.argv[2]
            if ":" in addr:
                host, port_str = addr.split(":", 1)
                port = int(port_str)
            else:
                port = int(addr)

        run_with_static_server(host, port)
    else:
        # Default: MCP stdio mode
        run_server()


if __name__ == "__main__":
    main()
