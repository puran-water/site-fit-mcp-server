# Site-Fit MCP Server

MCP server for constraint-based site layout optimization of wastewater/biogas facilities.

## Overview

Site-Fit generates multiple feasible site layouts ("test fits") for wastewater and biogas treatment facilities. It uses:

- **OR-Tools CP-SAT** for constraint-based structure placement with NoOverlap2D
- **Shapely/pyclipper** for geometry validation and polygon operations
- **A\* pathfinding** for road network generation constrained to site boundary
- **SFILES2 topology** for process flow constraints (optional)
- **Diversity filtering** via fingerprinting and DBSCAN clustering

## Features

- Generate multiple diverse layout solutions
- Automatic road network generation with turning radius constraints
- Respect property setbacks, equipment clearances, and keepout zones
- Per-equipment boundary setbacks (e.g., digesters require larger setbacks)
- Process topology-aware placement hints from SFILES2
- GeoJSON and SVG export for visualization
- Built-in Leaflet viewer with CRS.Simple for planar coordinates

## Installation

```bash
# Clone the repository
git clone https://github.com/puran-water/site-fit-mcp-server.git
cd site-fit-mcp-server

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

### MCP Mode (stdio)

For integration with MCP clients like Claude Desktop:

```bash
site-fit-mcp
```

### HTTP Mode with Viewer

For standalone use with the built-in web viewer:

```bash
site-fit-serve --port 8765
```

Then open http://localhost:8765 in your browser.

## MCP Tools

| Tool | Description |
|------|-------------|
| `sitefit_generate` | Generate site layout solutions from site boundary, structures, and optional topology |
| `sitefit_get_solution` | Get full solution details including GeoJSON features |
| `sitefit_list_solutions` | List solutions for a job with pagination |
| `sitefit_job_status` | Get status and progress of a generation job |
| `sitefit_export` | Export solution to GeoJSON, SVG, or summary format |
| `sitefit_generate_from_request` | Generate from nested SiteFitRequest object |
| `ruleset_list` | List available engineering rulesets |
| `ruleset_get` | Get ruleset configuration and JSON schema |
| `topology_parse_sfiles2` | Parse and validate SFILES2 topology strings |

## API Reference

### sitefit_generate

Generate site layout solutions.

**Parameters:**
- `site` - Site definition with boundary, entrances, keepouts, existing structures
- `program` - List of structures to place with footprints and access requirements
- `topology` (optional) - SFILES2 string for process flow constraints
- `rules_override` (optional) - Override default engineering rules
- `generation` - Generation parameters (max_solutions, max_time_seconds, seed)

**Example:**
```json
{
  "site": {
    "boundary": [[0,0], [100,0], [100,80], [0,80], [0,0]],
    "entrances": [{"id": "gate_1", "point": [50, 0], "width": 6.0}],
    "keepouts": [],
    "existing": []
  },
  "program": {
    "structures": [
      {"id": "TK-001", "type": "tank", "footprint": {"shape": "circle", "d": 12}},
      {"id": "PS-001", "type": "pump_station", "footprint": {"shape": "rect", "w": 6, "h": 4}}
    ]
  },
  "generation": {
    "max_solutions": 5,
    "max_time_seconds": 60,
    "seed": 42
  }
}
```

### Structure Footprints

**Rectangular:**
```json
{"shape": "rect", "w": 10, "h": 8}
```

**Circular:**
```json
{"shape": "circle", "d": 15}
```

### Access Requirements

Structures can specify vehicle access requirements:
```json
{
  "id": "DIG-001",
  "type": "digester",
  "footprint": {"shape": "circle", "d": 20},
  "access": {
    "vehicle": "tanker",
    "dock_edge": "any",
    "dock_length": 15.0,
    "required": true
  }
}
```

### Topology Node Mapping

When using SFILES2 topology, you can map topology node IDs to structure IDs:
```json
{
  "topology": {
    "sfiles2": "(reactor)->(tank)->(pump)",
    "node_map": {
      "reactor-1": "RX-001",
      "tank-1": "TK-001",
      "pump-1": "PS-001"
    }
  }
}
```

This allows topology-derived placement hints (flow direction, adjacency) to correctly apply to your structures.

## Engineering Rules

Default rules are defined in `src/rulesets/default.yaml`:

- **Property setbacks**: Distance from site boundary (default: 7.5m)
- **Equipment setbacks**: Type-specific boundary distances (e.g., digesters: 15m)
- **Equipment clearances**: Minimum distances between equipment types
- **Access rules**: Road width (6m), turning radius (12m), dock depth (15m)

Override rules via the `rules_override` parameter in `sitefit_generate`.

## Architecture

```
src/
├── server.py              # FastMCP server + FastAPI integration
├── pipeline.py            # Main generation pipeline
├── models/                # Pydantic schemas
│   ├── site.py           # Site, Boundary, Keepout, Entrance
│   ├── structures.py     # Structure footprints, access requirements
│   ├── rules.py          # Setback/clearance rule schemas
│   ├── topology.py       # SFILES topology wrapper
│   └── solution.py       # SiteFitSolution, metrics, GeoJSON
├── geometry/              # Geometry operations
│   ├── polygon_ops.py    # Shapely/pyclipper wrappers
│   ├── containment.py    # Boundary containment checks
│   └── clearance.py      # Pairwise distance calculations
├── solver/                # OR-Tools CP-SAT solver
│   ├── cpsat_placer.py   # Placement solver with NoOverlap2D
│   ├── solution_pool.py  # Multi-solution enumeration
│   └── diversity.py      # Solution fingerprinting & filtering
├── topology/              # SFILES2 parsing
│   ├── sfiles_parser.py  # SFILES2 graph parsing with branch handling
│   ├── graph_analysis.py # SCC, topological order, clustering
│   └── placement_hints.py # Topology-derived placement guidance
├── roads/                 # Road network generation
│   ├── dock_zones.py     # Access dock generation
│   ├── pathfinder.py     # A* road routing
│   └── network.py        # Road network validation
├── rules/                 # Rule management
│   └── loader.py         # YAML ruleset loading and merging
├── export/                # Export utilities
│   ├── geojson.py        # GeoJSON FeatureCollection export
│   └── svg.py            # SVG preview generation
└── tools/                 # MCP tool definitions
    └── sitefit_tools.py
```

## Performance

| Facility Size | Structures | Total Time | Solutions |
|---------------|------------|------------|-----------|
| Small         | 5-10       | ~1s        | 3-5       |
| Medium        | 10-15      | ~6s        | 5         |
| Large         | 15-20      | ~1-2s      | 5         |
| Very Large    | 20-25      | ~30s       | 3-5       |

Performance depends on site complexity, structure density, and road routing requirements.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

The project uses standard Python formatting conventions.

## Recent Updates

### v0.2.0 (2025-12-14)
- **YAML Rulesets**: Pipeline now loads rules from `src/rulesets/default.yaml` instead of hardcoded defaults
- **Wheel Packaging**: Rulesets moved into `src/` package for proper wheel distribution
- **Solution Ranking**: Objective values now properly threaded through pipeline for correct ranking
- **Rotation Export**: GeoJSON/SVG exports now correctly handle rotated rectangles (non-90° angles)
- **Clearance Naming**: Renamed misleading `_add_soft_distance_penalty` to `_add_hard_clearance_constraint`
- **Objective Sorting**: Fixed edge case where zero-valued objectives were treated as worst
- **Road Pathfinding**: Fixed LineString error when start==end by ensuring 2-point minimum paths

### v0.1.0 (2025-12-10)
- **SVG Export**: Fixed to work with Placement model (creates polygons from x/y/width/height)
- **Ruleset Tools**: `ruleset_list` and `ruleset_get` now use YAML loader module
- **Keepout Zones**: Added as road network obstacles (roads won't route through keepouts)
- **Topology Mapping**: Added `node_map` field to link topology IDs to structure IDs
- **GeoJSON Export**: Circles now export as true 32-segment polygons, not rectangles
- **API Schema**: Added `sitefit_generate_from_request` tool for nested request objects

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
