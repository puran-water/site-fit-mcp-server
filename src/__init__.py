"""Site-Fit MCP Server - Generate feasible site layouts for process facilities.

This package provides:
- MCP tools for site layout generation
- OR-Tools CP-SAT based placement solver
- SFILES2 topology integration
- GeoJSON export and Leaflet viewer
"""

__version__ = "0.1.0"

from .server import mcp

__all__ = ["mcp", "__version__"]
