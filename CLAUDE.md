# Site-Fit MCP Server

Constraint-based site layout solver exposed via Model Context Protocol (MCP).

## Architecture

```
site-fit-mcp-server/
├── src/
│   ├── server.py              # FastMCP server with all MCP tools
│   ├── models/                # Pydantic models for site, solution, etc.
│   ├── solver/                # Layout generation algorithms
│   ├── export/                # Export formats (DXF, PDF, GeoJSON, SVG)
│   │   ├── dxf.py            # Civil engineering DXF layers
│   │   ├── pdf_reportlab.py  # Headless PDF generation
│   │   └── pack.py           # Multi-format export bundler
│   ├── contract/             # Spatial Contract validation
│   │   └── validator.py      # JSON Schema validation + migration
│   └── loaders/              # GIS file loaders
├── schemas/
│   └── spatial_contract_v1.json  # Formal JSON Schema for data interchange
└── docs/
    └── completed-plans/
        └── workflow-hardening-plan.md  # Implementation plan (completed)
```

## Completed Plans

| Plan | Status | Date |
|------|--------|------|
| [Workflow Hardening Plan](docs/completed-plans/workflow-hardening-plan.md) | Complete | 2026-01-09 |

## Development Progress

### Workflow Hardening (Completed 2026-01-09)

All 5 phases implemented and verified:

**Phase 1: Core Fixes + Observability**
- `sitefit_export_batch`: Batch export with paging and FreeCAD-compatible aliases
- Roads in `import_solutions_as_layers` with active/inactive styling
- Persistence with atomic writes to `~/.sitefit/jobs/`
- Draft API wrapper for FreeCAD compatibility
- Structured logging via structlog (JSON output)

**Phase 2: TechDraw Reliability**
- `techdraw_preflight`: Check GUI/Xvfb availability before export
- `sitefit_export_pack`: Guaranteed headless export (ReportLab PDF, ezdxf DXF)
- Structured diagnostics with error codes

**Phase 3: Contract Formalization**
- JSON Schema v1.0 at `schemas/spatial_contract_v1.json`
- Schema validation wired into import/export flows
- Migration support from legacy (0.9) contracts

**Phase 4: Civil Access + Schema Evolution**
- Road corridor fields in schema (edge_left, edge_right, surface_type)
- FreeCAD road rendering with visual hierarchy
- DXF export with civil layers: ROAD_CENTERLINE, ROAD_EDGE, ROAD_SURFACE, SETBACKS

**Phase 5: Polish**
- `sitefit_compare_solutions`: Diff tool for comparing layouts

## Key MCP Tools

| Tool | Purpose |
|------|---------|
| `sitefit_generate` | Generate layout solutions from constraints |
| `sitefit_list_solutions` | List solutions for a job |
| `sitefit_get_solution` | Get full solution details |
| `sitefit_export` | Export single solution (geojson, svg, contract, summary) |
| `sitefit_export_batch` | Export multiple solutions as contracts (for FreeCAD) |
| `sitefit_export_pack` | Headless multi-format export bundle |
| `sitefit_export_contract` | Export contract JSON for FreeCAD import |
| `sitefit_compare_solutions` | Compare two solutions and explain differences |

## Integration with FreeCAD MCP

Site-fit produces Spatial Contract JSON which FreeCAD MCP consumes:

```
site-fit-mcp-server                    freecad-mcp
─────────────────                      ───────────
sitefit_generate()
       │
       ▼
sitefit_export_batch()  ────────────►  present_layout_options()
       │                                      │
       │                                      ▼
       │                               import_solutions_as_layers()
       │                                      │
       ▼                                      ▼
Spatial Contract JSON                  FreeCAD Document with
  - site.boundary                      toggleable solution layers
  - placements[]
  - road_network
  - structures[]
```

## Running the Server

```bash
# Install dependencies
uv sync

# Run MCP server
uv run python -m site_fit_mcp_server
```

## Schema Location

The Spatial Contract v1.0 JSON Schema is at:
- `schemas/spatial_contract_v1.json`

FreeCAD MCP references this schema for validation.
