"""Response filtering utilities for reducing context consumption.

Provides compact vs full response modes for MCP tools. Compact mode (default)
returns only actionable information needed for workflows. Full mode includes
all fields for debugging.
"""

from typing import Any, Literal

# Type alias for detail level parameter
DetailLevel = Literal["compact", "full"]

# Fields to keep in compact metrics response
# Always includes is_feasible for workflow decision-making
COMPACT_METRICS = {"compactness", "road_length", "is_feasible"}

# Fields to keep in compact statistics response
COMPACT_STATS = {
    "job_id",
    "cpsat_solutions",
    "validated_solutions",
    "total_time_seconds",
    "num_structures",
}


def filter_metrics(metrics: dict[str, Any], detail_level: DetailLevel) -> dict[str, Any]:
    """Filter solution metrics based on detail level.

    Args:
        metrics: Full metrics dictionary from SolutionMetrics.model_dump()
        detail_level: "compact" for essential fields, "full" for all fields

    Returns:
        Filtered metrics dictionary
    """
    if detail_level == "full":
        return metrics
    return {k: v for k, v in metrics.items() if k in COMPACT_METRICS}


def filter_statistics(stats: dict[str, Any], detail_level: DetailLevel) -> dict[str, Any]:
    """Filter generation statistics based on detail level.

    Args:
        stats: Full statistics dictionary from generate_site_fits()
        detail_level: "compact" for essential fields, "full" for all fields

    Returns:
        Filtered statistics dictionary
    """
    if detail_level == "full":
        return stats
    return {k: v for k, v in stats.items() if k in COMPACT_STATS}


def filter_solution_summary(
    solution: dict[str, Any],
    detail_level: DetailLevel
) -> dict[str, Any]:
    """Filter solution summary for list responses.

    Args:
        solution: Solution dict with id, rank, metrics, diversity_note
        detail_level: "compact" for essential fields, "full" for all fields

    Returns:
        Filtered solution summary
    """
    if detail_level == "full":
        return solution

    return {
        "id": solution.get("id"),
        "rank": solution.get("rank"),
        "metrics": filter_metrics(solution.get("metrics", {}), detail_level),
    }


def filter_placement(
    placement: dict[str, Any],
    detail_level: DetailLevel
) -> dict[str, Any]:
    """Filter placement data.

    Note: Dimensions (width, height, shape) are always kept since caller
    may not have persisted original structure definitions.

    Args:
        placement: Placement dict with structure_id, x, y, rotation_deg, etc.
        detail_level: "compact" or "full" (currently no difference for placements)

    Returns:
        Placement dictionary (unchanged - dimensions always needed)
    """
    # Keep all placement fields - dimensions needed even in compact mode
    # per design decision: "Don't drop placement dimensions"
    return placement


def filter_topology_result(
    result: dict[str, Any],
    detail_level: DetailLevel
) -> dict[str, Any]:
    """Filter topology parse result.

    Args:
        result: Parse result with valid, nodes, edges, tokens, etc.
        detail_level: "compact" omits nodes/edges/tokens arrays

    Returns:
        Filtered result
    """
    if detail_level == "full":
        return result

    # Compact: just validation summary
    compact = {
        "valid": result.get("valid"),
        "num_nodes": result.get("num_nodes"),
        "num_edges": result.get("num_edges"),
    }

    # Always include error if present
    if "error" in result:
        compact["error"] = result["error"]

    return compact
