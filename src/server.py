"""FastMCP server for site-fit generation.

Exposes MCP tools for generating site layout solutions and integrates
with FastAPI for static file serving (viewer).
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .pipeline import generate_site_fits
from .tools.sitefit_tools import (
    SiteFitRequest,
)

# Path to static files
STATIC_DIR = Path(__file__).parent.parent / "static"

# Configure logging to stderr (required for MCP stdio transport)
# stdio servers must NOT log to stdout as it interferes with JSON-RPC
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # Critical: use stderr, not stdout
)
logger = logging.getLogger(__name__)

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
        _jobs[job_id] = {
            "status": "completed",
            "stats": stats,
            "solution_ids": [s.id for s in solutions],
            "request": request.model_dump(),  # Store for contract export
        }

        for sol in solutions:
            _solutions[sol.id] = sol.model_dump()

        # Build response
        return {
            "job_id": job_id,
            "status": "completed",
            "num_solutions": len(solutions),
            "solutions": [
                {
                    "id": s.id,
                    "rank": s.rank,
                    "metrics": s.metrics.model_dump(),
                    "diversity_note": s.diversity_note,
                }
                for s in solutions
            ],
            "statistics": stats,
        }

    except Exception as e:
        logger.exception("Generation failed")
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
) -> dict[str, Any]:
    """Generate site layout solutions from a complete SiteFitRequest object.

    Alternative to sitefit_generate that accepts the full nested request schema.
    Useful when programmatically constructing requests or integrating with other systems.

    Args:
        request: Full SiteFitRequest object with site, program, topology, rules_override, generation

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
        _jobs[job_id] = {
            "status": "completed",
            "stats": stats,
            "solution_ids": [s.id for s in solutions],
            "request": site_fit_request.model_dump(),  # Store for contract export
        }

        for sol in solutions:
            _solutions[sol.id] = sol.model_dump()

        # Build response
        return {
            "job_id": job_id,
            "status": "completed",
            "num_solutions": len(solutions),
            "solutions": [
                {
                    "id": s.id,
                    "rank": s.rank,
                    "metrics": s.metrics.model_dump(),
                    "diversity_note": s.diversity_note,
                }
                for s in solutions
            ],
            "statistics": stats,
        }

    except Exception as e:
        logger.exception("Generation failed")
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
) -> dict[str, Any]:
    """Get full details of a specific solution by ID.

    Retrieves placement coordinates, metrics, road network, and optional
    GeoJSON feature collection for visualization.

    Args:
        solution_id: Solution ID from sitefit_generate response
        include_geojson: Include GeoJSON feature collection (default True)

    Returns:
        Full solution with placements, metrics, road_network, and features_geojson
    """
    if solution_id not in _solutions:
        return {
            "isError": True,
            "error": f"Solution {solution_id} not found",
            "suggestion": "Use sitefit_list_solutions to get valid solution IDs from a job",
        }

    solution = _solutions[solution_id]

    if not include_geojson and "features_geojson" in solution:
        # Return without GeoJSON for smaller response
        result = {k: v for k, v in solution.items() if k != "features_geojson"}
        return result

    return solution


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
) -> dict[str, Any]:
    """List solutions for a job with pagination.

    Returns summaries (id, rank, metrics) for quick browsing.
    Use sitefit_get_solution for full details.

    Args:
        job_id: Job ID from sitefit_generate
        limit: Maximum solutions to return (default 20)
        offset: Offset for pagination (default 0)

    Returns:
        Dict with job_id, total, offset, limit, solutions, has_more, next_offset
    """
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
            solutions.append({
                "id": sol["id"],
                "rank": sol["rank"],
                "metrics": sol.get("metrics", {}),
                "diversity_note": sol.get("diversity_note"),
            })

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
) -> dict[str, Any]:
    """Get the status of a site-fit generation job.

    Check if a job is queued, running, completed, or failed.
    For completed jobs, includes solution count and summary statistics.

    Args:
        job_id: Job ID from sitefit_generate

    Returns:
        Dict with job_id, status, progress (0-100), and job details
    """
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
        result["statistics"] = {
            "cpsat_solutions": stats.get("cpsat_solutions", 0),
            "validated_solutions": stats.get("validated_solutions", 0),
            "total_time_seconds": stats.get("total_time_seconds", 0),
        }
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
) -> dict[str, Any]:
    """Export a solution to various formats.

    Converts solution data to GeoJSON, SVG, contract, or summary formats
    for use in external tools and viewers.

    Args:
        solution_id: Solution ID from sitefit_generate
        format: Export format - 'geojson', 'svg', 'contract', or 'summary' (default: geojson)
        include_roads: Include road network in contract format (default: True)

    Returns:
        Dict with format, data (content), and metadata
    """
    if solution_id not in _solutions:
        return {
            "isError": True,
            "error": f"Solution {solution_id} not found",
            "suggestion": "Use sitefit_list_solutions to get valid solution IDs",
        }

    solution = _solutions[solution_id]
    format = format.lower()

    if format == "geojson":
        return {
            "format": "geojson",
            "content_type": "application/geo+json",
            "data": solution.get("features_geojson", {}),
            "metadata": {
                "solution_id": solution_id,
                "rank": solution.get("rank"),
            },
        }

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

            return {
                "format": "svg",
                "content_type": "image/svg+xml",
                "data": svg_content,
                "metadata": {
                    "solution_id": solution_id,
                    "rank": solution.get("rank"),
                },
            }
        except Exception as e:
            logger.exception("SVG export failed")
            return {
                "isError": True,
                "error": f"SVG export failed: {str(e)}",
                "suggestion": "Try exporting as 'geojson' instead",
            }

    elif format == "summary":
        metrics = solution.get("metrics", {})
        placements = solution.get("placements", [])

        return {
            "format": "summary",
            "content_type": "application/json",
            "data": {
                "solution_id": solution_id,
                "rank": solution.get("rank"),
                "num_structures": len(placements),
                "metrics": metrics,
                "diversity_note": solution.get("diversity_note"),
                "has_road_network": solution.get("road_network") is not None,
                "structure_ids": [p.get("structure_id") for p in placements],
            },
            "metadata": {
                "solution_id": solution_id,
            },
        }

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
            # Fallback: extract from GeoJSON features
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

        return {
            "format": "contract",
            "content_type": "application/json",
            "data": {
                "project": {
                    "name": "",
                    "id": solution_id,
                    "revision": "A",
                },
                "site": {
                    "boundary": site_data.get("boundary", []),
                    "entrances": site_data.get("entrances", []),
                    "keepouts": site_data.get("keepouts", []),
                },
                "program": {
                    "structures": structures_data,
                },
                "placements": contract_placements,
                "road_network": road_network_data,
                "metadata": {
                    "source": "sitefit_mcp",
                    "solution_id": solution_id,
                    "rank": solution.get("rank", 0),
                },
            },
            "metadata": {
                "solution_id": solution_id,
                "rank": solution.get("rank"),
            },
        }

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

    Returns:
        Contract JSON with project, site, program, placements, road_network, metadata
    """
    return await sitefit_export(solution_id, format="contract", include_roads=include_roads)


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

    Generates bundled exports for proposal exhibits and civil handoff:
    - PDF plan sheet with layout diagram, scale bar, legend, and quantities
    - DXF CAD file with layers: BOUNDARY, BUILDLIMIT, KEEPOUTS, STRUCTURES, ROADS
    - CSV with ROM quantities: pad areas, road length, pipe proxies, fence perimeter
    - GeoJSON for visualization and GIS integration

    Args:
        solution_id: Solution ID from sitefit_generate
        formats: List of formats to generate - 'pdf', 'dxf', 'csv', 'geojson' (default: csv, geojson)
        output_dir: Directory for output files (uses temp directory if not specified)
        project_name: Project name for PDF title block
        drawing_number: Drawing number for PDF title block

    Returns:
        Dict with success, files (format -> path mapping), quantities, and any errors
    """
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
        logger.exception("Export pack failed")
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
        logger.exception("Failed to list rulesets")
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
        logger.exception(f"Failed to load ruleset '{name}'")
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
) -> dict[str, Any]:
    """Parse and validate an SFILES2 process topology string.

    Tokenizes the SFILES2 notation and extracts nodes (equipment) and
    edges (connections) for validation before use in sitefit_generate.

    Args:
        sfiles2: SFILES2 string (e.g., "(influent)pump|P-101(tank)T-101")

    Returns:
        Dict with valid (bool), nodes, edges, tokens, num_nodes, num_edges.
        If invalid, includes error message.
    """
    from .topology.sfiles_parser import parse_sfiles_topology, tokenize_sfiles

    try:
        topology = parse_sfiles_topology(sfiles2)
        tokens = tokenize_sfiles(sfiles2)

        return {
            "valid": True,
            "nodes": [n.model_dump() for n in topology.nodes],
            "edges": [e.model_dump() for e in topology.edges],
            "tokens": tokens,
            "num_nodes": len(topology.nodes),
            "num_edges": len(topology.edges),
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "nodes": [],
            "edges": [],
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
        logger.exception(f"Failed to load GIS file: {file_path}")
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
        logger.exception(f"Failed to list GIS layers: {file_path}")
        return {
            "success": False,
            "error": str(e),
        }


def run_server():
    """Run the MCP server (MCP transport only)."""

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
            logger.exception("API generate failed")
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

    logger.info(f"Starting Site-Fit server on http://{host}:{port}")
    logger.info(f"  - Viewer:     http://{host}:{port}/")
    logger.info(f"  - REST API:   http://{host}:{port}/api/")
    logger.info(f"  - MCP (SSE):  http://{host}:{port}/mcp")

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
