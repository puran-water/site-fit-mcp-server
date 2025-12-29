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
| `sitefit_export` | Export solution to GeoJSON, SVG, contract, or summary format |
| `sitefit_export_contract` | Export solution as Spatial Contract JSON for FreeCAD integration |
| `sitefit_export_pack` | Export bundle with DXF, GeoJSON, CSV quantities, and optional PDF |
| `sitefit_generate_from_request` | Generate from nested SiteFitRequest object |
| `sitefit_load_gis_file` | Load site boundary/keepouts/entrances from GIS files |
| `sitefit_list_gis_layers` | List layers in a GIS file with metadata |
| `ruleset_list` | List available engineering rulesets |
| `ruleset_get` | Get ruleset configuration and JSON schema |
| `topology_parse_sfiles2` | Parse and validate SFILES2 topology strings |

### Response Detail Level

All tools support a `detail_level` parameter to reduce context consumption:

| Value | Description |
|-------|-------------|
| `"compact"` (default) | Returns only essential fields for workflows. ~60-80% context reduction. |
| `"full"` | Returns all metrics, statistics, and debug information. |

**Compact mode includes:**
- Core identifiers: `job_id`, `solution_id`, `status`
- Key metrics: `compactness`, `road_length`, `is_feasible`
- Essential placement data: `structure_id`, `x`, `y`, `rotation_deg`, `width`, `height`, `shape`
- Error information when present

**Full mode adds:**
- Complete metrics (20+ fields)
- Full statistics (solve times, candidate counts)
- Diversity notes and debug information

Example:
```python
# Compact response (default) - for production workflows
result = sitefit_generate(site_boundary=..., structures=..., detail_level="compact")

# Full response - for debugging
result = sitefit_generate(site_boundary=..., structures=..., detail_level="full")
```

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
    "turning_radius": 15.0,
    "required": true
  }
}
```

The `turning_radius` field specifies the minimum turning radius for dock approaches to this structure. The solver validates road segments connecting to the structure against this requirement.

### Pinned Placements

Structures can be pinned to fixed positions for brownfield sites or iterative design:
```json
{
  "id": "TK-001",
  "type": "tank",
  "footprint": {"shape": "circle", "d": 12},
  "pinned": true,
  "fixed_position": {
    "x": 50.0,
    "y": 50.0,
    "rotation_deg": 0
  }
}
```

Pinned structures are placed at exact coordinates. The solver validates them against other constraints (NoOverlap2D) but does not move them.

### Service Envelopes

Structures can specify maintenance and crane access requirements:
```json
{
  "id": "RX-001",
  "type": "reactor",
  "footprint": {"shape": "rect", "w": 8, "h": 6},
  "service_envelopes": {
    "maintenance_offset": 3.0,
    "crane_access_edge": "long",
    "crane_strip_width": 6.0,
    "crane_strip_length": 20.0,
    "laydown_area": [10.0, 8.0],
    "laydown_edge": "E"
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
- **NFPA 820 hazard zones**: Hazardous area classifications for biogas facilities

Override rules via the `rules_override` parameter in `sitefit_generate`.

### NFPA 820 Hazardous Area Zones

Site-fit automatically generates NFPA 820 hazardous area classifications for wastewater and biogas facilities. These zones are included in the GeoJSON export.

**Zone Classifications:**
- **Class I, Division 1**: Ignitable concentrations exist under normal conditions
- **Class I, Division 2**: Ignitable concentrations may exist under abnormal conditions

**Default Zone Radii (configurable in ruleset):**

| Equipment Type | Class I Div 1 | Class I Div 2 |
|----------------|---------------|---------------|
| Digester | Interior only | 3.0m (10 ft) |
| Wet well | Interior only | 3.0m (10 ft) |
| Digester gas piping | 1.5m (5 ft) | 3.0m (10 ft) |
| Covered clarifier | Interior only | 3.0m (10 ft) |
| Odor control | - | 0.9m (3 ft) |
| Flare | 1.5m (5 ft) | 4.5m (15 ft) |

**Excluded Equipment:**
Electrical buildings, control buildings, motor control centers, offices, and laboratories must be placed outside hazard zones.

**Override Example:**
```json
{
  "rules_override": {
    "nfpa820_zones": {
      "digester": {
        "class_i_div_2_radius": 5.0
      }
    }
  }
}
```

## Architecture

```
src/
├── server.py              # FastMCP server + FastAPI integration
├── pipeline.py            # Main generation pipeline
├── models/                # Pydantic schemas
│   ├── site.py           # Site, Boundary, Keepout, Entrance
│   ├── structures.py     # Structure footprints, access requirements
│   ├── rules.py          # Setback/clearance rule schemas + NFPA 820
│   ├── topology.py       # SFILES topology wrapper
│   └── solution.py       # SiteFitSolution, metrics, GeoJSON
├── geometry/              # Geometry operations
│   ├── polygon_ops.py    # Shapely/pyclipper wrappers
│   ├── containment.py    # Boundary containment checks
│   └── clearance.py      # Pairwise distance calculations (STRtree optimized)
├── solver/                # OR-Tools CP-SAT solver
│   ├── cpsat_placer.py   # Placement solver with NoOverlap2D
│   ├── solution_pool.py  # Multi-solution enumeration
│   ├── diversity.py      # Solution fingerprinting & filtering
│   └── grid_candidates.py # Grid-based candidate generation
├── topology/              # SFILES2 parsing
│   ├── sfiles_parser.py  # SFILES2 graph parsing with branch handling
│   ├── graph_analysis.py # SCC, topological order, clustering
│   └── placement_hints.py # Topology-derived placement guidance
├── roads/                 # Road network generation
│   ├── dock_zones.py     # Access dock generation
│   ├── pathfinder.py     # A* road routing
│   ├── network.py        # Road network validation (Steiner tree optimized)
│   └── turning_radius.py # Turning radius validation at corners
├── hazards/               # Safety zone calculations
│   └── nfpa820_zones.py  # NFPA 820 hazardous area classification
├── loaders/               # File format loaders
│   └── gis_loader.py     # Shapefile/GeoPackage/GeoJSON import
├── rules/                 # Rule management
│   └── loader.py         # YAML ruleset loading and merging
├── rulesets/              # Ruleset definitions
│   └── default.yaml      # Default engineering rules
├── export/                # Export utilities
│   ├── geojson.py        # GeoJSON FeatureCollection export
│   ├── svg.py            # SVG preview generation
│   ├── dxf.py            # DXF CAD export with layers
│   ├── quantities.py     # CSV quantity takeoffs
│   ├── pdf_report.py     # PDF plan sheet generation
│   └── pack.py           # Bundle export orchestrator
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

## GIS Integration

### Loading Solutions in QGIS

Site-fit exports solutions as GeoJSON, which can be directly loaded into QGIS for visualization and analysis:

```python
# PyQGIS script to load site-fit solution
from qgis.core import QgsVectorLayer, QgsProject

# Load the exported GeoJSON
layer = QgsVectorLayer("/path/to/solution.geojson", "Site Layout", "ogr")

# Add to project
if layer.isValid():
    QgsProject.instance().addMapLayer(layer)
else:
    print("Failed to load layer")
```

Or simply drag and drop the `.geojson` file into QGIS.

**Layer Styling Tips:**
- Use categorized symbology on the `feature_type` property to distinguish structures, roads, and boundary
- Enable labels using the `id` property for structure identification
- Set road features to line symbology with appropriate width (6m default)

### Coordinate Reference System

Site-fit uses a local coordinate system in meters. When loading into QGIS:
1. Set the project CRS to a local projected CRS (e.g., UTM zone for your area)
2. Or use a generic meters-based CRS like EPSG:3857 for visualization only

For georeferenced outputs, transform the GeoJSON coordinates to your target CRS before loading.

### Importing from GIS Files

Site-fit can directly import site boundaries, keepouts, and entrances from various GIS formats:

**Supported Formats:**
- Shapefile (.shp)
- GeoJSON (.geojson, .json)
- GeoPackage (.gpkg)
- KML (.kml)
- File Geodatabase (.gdb)

**Installation:**
```bash
pip install 'site-fit-mcp[gis]'
```

**Usage via MCP:**
```python
# List layers in a file
result = await sitefit_list_gis_layers("/path/to/parcel.shp")
print(result["layers"])

# Load site data
site_data = await sitefit_load_gis_file(
    file_path="/path/to/parcel.shp",
    boundary_layer=None,  # Auto-detect
    keepout_layers=None,  # Auto-detect from names like "easement", "wetland"
    target_crs="EPSG:32632",  # Optional CRS transformation
)

# Use loaded data with sitefit_generate
solutions = await sitefit_generate(
    site_boundary=site_data["boundary"],
    keepouts=site_data["keepouts"],
    entrances=site_data["entrances"],
    structures=[...]
)
```

**Auto-Detection:**
Layer auto-detection uses naming conventions:
- **Boundary**: "boundary", "parcel", "lot", "property", "site"
- **Keepouts**: "keepout", "easement", "wetland", "flood", "buffer", "setback"
- **Entrances**: "entrance", "access", "gate", "driveway"

## FreeCAD Integration

Site-fit can export solutions as Spatial Contract JSON for seamless import into FreeCAD via freecad-mcp.

### Contract Export

```python
# Generate layouts
result = sitefit_generate(
    site_boundary=[[0,0], [100,0], [100,80], [0,80], [0,0]],
    structures=[
        {"id": "TK-101", "type": "tank", "footprint": {"shape": "circle", "d": 15}},
        {"id": "BLDG-001", "type": "building", "footprint": {"shape": "rect", "w": 12, "h": 15}},
    ],
    entrances=[{"id": "gate", "point": [50, 0], "width": 6}],
    max_solutions=3
)

# Export as contract (includes structure dimensions, placements, roads)
contract = sitefit_export_contract(
    solution_id=result["solutions"][0]["solution_id"],
    include_roads=True
)

# Use with freecad-mcp's import_sitefit_contract tool
```

### Contract Schema

The contract format includes:
- **site**: Boundary, entrances, keepouts
- **program.structures**: Equipment with dimensions (`id`, `type`, `footprint`)
- **placements**: Solved positions (`id`, `x`, `y`, `rotation_deg`)
- **road_network**: Segments with start, end, waypoints

Note: Placements use `id` (not `structure_id`) for FreeCAD compatibility.

## Recent Updates

### v0.5.0 (2025-12-29)
- **Context Optimization**: Added `detail_level` parameter to all tools (default: `"compact"`)
- **Compact Mode**: ~60-80% context reduction in typical responses
- **Response Filters**: New `response_filters.py` module for field filtering
- **Metrics Filtering**: Compact mode returns only 3 key metrics vs 20+ in full mode
- **Statistics Filtering**: `sitefit_job_status` omits statistics in compact mode

### v0.4.0 (2025-12-27)
- **Contract Export**: New `sitefit_export_contract` for FreeCAD integration via freecad-mcp
- **Contract Format**: Added `format="contract"` option to `sitefit_export`
- **Structure Dimensions**: Contract export includes original footprint dimensions from request

### v0.3.0 (2025-12-17)
- **Export Pack Tool**: New `sitefit_export_pack` for bundled DXF/GeoJSON/CSV/PDF exports
- **Pinned Placements**: Support `pinned: true` with `fixed_position` for brownfield sites
- **Service Envelopes**: Maintenance, crane access, and laydown area requirements
- **Per-Structure Turning Radius**: `access.turning_radius` enforced per dock approach
- **Footprint-Based Hazard Zones**: NFPA 820 zones now calculated from equipment perimeter, not centroid
- **Enhanced ROM Metrics**: Categorical breakdowns in `SolutionMetrics`
- **Existing Structures**: Parse `site.existing` as obstacles for road routing and placement

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
