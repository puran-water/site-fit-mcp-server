"""Site-Fit MCP Server - Generate feasible site layouts for process facilities.

This package provides:
- MCP tools for site layout generation
- OR-Tools CP-SAT based placement solver
- SFILES2 topology integration
- GeoJSON export and Leaflet viewer

Core functionality can be imported without MCP server dependencies:
    from src.pipeline import generate_site_fits
    from src.solver import PlacementSolver

To get the MCP server instance:
    from src import get_mcp
    mcp = get_mcp()
"""

__version__ = "0.1.0"


def get_mcp():
    """Get the MCP server instance (lazy import to avoid coupling).

    Returns:
        FastMCP: The configured MCP server instance.

    Example:
        from src import get_mcp
        mcp = get_mcp()
    """
    from .server import mcp
    return mcp


# Expose core modules for direct import without MCP dependency
def get_pipeline():
    """Get the pipeline module for direct use."""
    from . import pipeline
    return pipeline


def get_solver():
    """Get the solver module for direct use."""
    from . import solver
    return solver


__all__ = ["get_mcp", "get_pipeline", "get_solver", "__version__"]
