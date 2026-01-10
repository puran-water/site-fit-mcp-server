# Site-Fit + FreeCAD Engineering Workflow Hardening Plan

## Executive Summary

The user's feedback is **accurate and well-founded**. Independent verification (Claude + Codex) confirms all four identified issues:

1. **Multi-solution propagation failure** - `sitefit_generate` returns summaries; agents skip the required `sitefit_get_solution` loop
2. **Roads missing in FreeCAD layers** - `import_solutions_as_layers` completely ignores road networks
3. **TechDraw brittleness** - Requires GUI (FreeCAD #5710 open since 2017), template discovery fragile
4. **"Demo" quality** - No persistence, no contract versioning, no compliance audit trail

---

## Verified Technical Findings

### Site-Fit MCP Server (`site-fit-mcp-server/`)

| Finding | Location | Impact |
|---------|----------|--------|
| **In-memory storage only** | `server.py:47-49` | Solutions lost on restart |
| **Generate returns summaries** | `server.py` sitefit_generate | Must loop get_solution |
| **No batch export** | No such endpoint | Agent loop overhead |
| **Roads in contract export** | `server.py:572-714` | Works if called correctly |
| **No formal contract schema** | Inferred from code | Version drift risk |

### FreeCAD MCP Server (`freecad-mcp/`)

| Finding | Location | Impact |
|---------|----------|--------|
| **Layers path = no roads** | `contract_tools.py:1633` | Roads completely omitted |
| **Contract path = roads** | `contract_tools.py:1107,1541-1592` | Works correctly |
| **Draft API inconsistent** | makeWire vs make_wire | Both work (aliases) |
| **TechDraw requires GUI** | TechDrawGui imports | No headless PDF |
| **Template fallback fragile** | Lines 198-214 | May silently fail |

### FreeCAD API (via DeepWiki + GitHub)

| Finding | Source | Impact |
|---------|--------|--------|
| **Draft.make_wire is canonical** | DeepWiki | makeWire is alias, both work |
| **Headless PDF = "won't fix"** | FreeCAD #5710 (open since 2017) | Requires Xvfb or GUI |
| **Template discovery** | Preferences `TemplateFile`/`TemplateDir` | Falls back to resource dir |

---

## Implementation Plan (Codex-Revised)

### Phase 1: Critical Fixes + Observability

**Goal:** Make existing pipeline reliable with debugging visibility from day one.

#### 1A. Add Batch Solution Export with Paging

**File:** `site-fit-mcp-server/src/server.py`

```python
@mcp.tool()
async def sitefit_export_batch(
    job_id: str,
    solution_ids: list[str] | None = None,  # Filter to specific solutions
    include_roads: bool = True,
    include_structures: bool = True,
    format: Literal["contracts", "solutions"] = "contracts",
    limit: int = 10,  # Prevent payload explosion
    offset: int = 0
) -> dict:
    """Export solutions from a job with paging support."""
    return {
        "contracts": [...],
        "total": int,
        "limit": int,
        "offset": int,
        "has_more": bool
    }
```

**Acceptance criteria:**
- Single call returns N contracts (up to limit)
- Supports paging for large solution sets
- Each contract includes placements + road_network
- No agent-side loop required for typical cases

#### 1B. Add Roads to `import_solutions_as_layers`

**File:** `freecad-mcp/src/freecad_mcp/contract_tools.py`

Modify function at line 1633:

```python
async def import_solutions_as_layers(
    doc_name: str,
    solutions: list[dict],  # Now expects: {..., "road_network": {...}}
    site_boundary: list[list[float]] | None = None,
    keepouts: list[dict] | None = None,
    active_layer_index: int = 0,
    create_roads: bool = True,  # NEW
) -> dict:
```

Road creation per layer:
1. Extract `road_network` from each solution
2. Create Draft wires for each segment (centerline)
3. Style: gray (#808080), width=2, dashed for inactive layers
4. Group under `{layer_name}/Roads`

**Acceptance criteria:**
- Each layer includes its road network as Draft wires
- Road styling distinguishes active vs hidden layers
- Toggle visibility works for roads and equipment together

#### 1C. Add Solution Persistence with Atomic Writes

**File:** `site-fit-mcp-server/src/server.py`

```python
import json
import tempfile
import fcntl
from pathlib import Path
from datetime import datetime, timedelta

PERSISTENCE_DIR = Path.home() / ".sitefit" / "jobs"
SCHEMA_VERSION = "1.0"

def _persist_job(job_id: str, job_data: dict):
    """Atomic write: temp file + rename to prevent corruption."""
    job_dir = PERSISTENCE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Add schema version for future migration
    job_data["_schema_version"] = SCHEMA_VERSION
    job_data["_persisted_at"] = datetime.utcnow().isoformat()

    # Atomic write pattern
    target = job_dir / "job.json"
    with tempfile.NamedTemporaryFile(
        mode='w', dir=job_dir, delete=False, suffix='.tmp'
    ) as tmp:
        json.dump(job_data, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.rename(tmp.name, target)

    # Persist each solution
    for sol_id in job_data.get("solution_ids", []):
        if sol_id in _solutions:
            _persist_solution(job_dir, sol_id, _solutions[sol_id])

def _load_job(job_id: str) -> dict | None:
    """Load job from disk, with schema migration if needed."""
    job_dir = PERSISTENCE_DIR / job_id
    job_file = job_dir / "job.json"
    if not job_file.exists():
        return None
    with open(job_file) as f:
        data = json.load(f)
    return _migrate_job_schema(data)

def _cleanup_old_jobs(max_age_days: int = 7):
    """Remove jobs older than max_age_days."""
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    for job_dir in PERSISTENCE_DIR.iterdir():
        job_file = job_dir / "job.json"
        if job_file.exists():
            with open(job_file) as f:
                data = json.load(f)
            persisted = datetime.fromisoformat(data.get("_persisted_at", "2000-01-01"))
            if persisted < cutoff:
                shutil.rmtree(job_dir)
```

**Acceptance criteria:**
- Jobs survive server restart
- Atomic writes prevent corruption on crash
- Schema version enables future migration
- 7-day default retention with cleanup

#### 1D. Draft API Wrapper (Low Priority)

**File:** `freecad-mcp/src/freecad_mcp/contract_tools.py`

Both `Draft.make_wire` and `Draft.makeWire` are aliases (confirmed via DeepWiki). Wrapper is optional but harmless:

```python
def _make_wire_compat(vectors, closed=False, face=False):
    """Version-compatible wire creation (both names work)."""
    import Draft
    make_fn = getattr(Draft, 'make_wire', None) or getattr(Draft, 'makeWire')
    return make_fn(vectors, closed=closed, face=face)
```

#### 1E. Structured Logging (Moved from Phase 5)

**Files:** Both `site-fit-mcp-server/src/server.py` and `freecad-mcp/src/freecad_mcp/*.py`

```python
import structlog
logger = structlog.get_logger()

# Configure once at startup
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

# Usage in tools
logger.info("batch_export_started", job_id=job_id, solution_count=len(solution_ids))
logger.info("solution_persisted", job_id=job_id, solution_id=sol_id, path=str(target))
logger.warning("techdraw_gui_unavailable", doc_name=doc_name, fallback="sitefit_export_pack")
logger.error("contract_validation_failed", version=version, expected="1.0.0")
```

**Acceptance criteria:**
- All major operations logged with structured JSON
- Includes job_id, solution_id, timestamps
- Errors include actionable context

---

### Phase 2: TechDraw Reliability

**Goal:** Make 2D drawing export either reliable or gracefully skipped.

#### 2A. TechDraw Preflight with Xvfb Detection

**File:** `freecad-mcp/src/freecad_mcp/techdraw_tools.py`

```python
@mcp.tool()
async def techdraw_preflight(doc_name: str) -> dict:
    """Check TechDraw readiness before export."""
    import os
    import shutil

    gui_available = _check_gui_available()
    xvfb_available = shutil.which("xvfb-run") is not None
    display_set = "DISPLAY" in os.environ

    recommendations = []
    if not gui_available:
        if xvfb_available:
            recommendations.append("Run with: xvfb-run freecad ...")
        else:
            recommendations.append("Install Xvfb: apt install xvfb")
        recommendations.append("Or use sitefit_export_pack for headless PDF")

    return {
        "gui_available": gui_available,
        "xvfb_available": xvfb_available,
        "display_set": display_set,
        "techdraw_available": _check_techdraw_module(),
        "template_path": _find_template(doc_name),
        "visible_objects": _count_visible_objects(doc_name),
        "freecad_version": FreeCAD.Version(),
        "can_export_pdf": gui_available or (xvfb_available and display_set),
        "recommendations": recommendations
    }
```

#### 2B. Non-TechDraw Fallback Export

**File:** `site-fit-mcp-server/src/tools/export_tools.py`

Enhance `sitefit_export_pack` as the **guaranteed** deliverable path:

```python
async def sitefit_export_pack(
    solution_id: str,
    formats: list[str] = ["pdf", "dxf", "geojson"],
    output_dir: str | None = None,
    project_name: str = "",
    drawing_number: str = "",
    scale: str = "1:200"  # For PDF layout
) -> dict:
    """Headless export pack - works without FreeCAD GUI."""
```

- **PDF**: ReportLab for layout (not scale-critical, but reliable)
- **DXF**: ezdxf (already implemented, works well)
- **SVG**: svgwrite for web viewing
- **GeoJSON**: Shapely (already implemented)

TechDraw = "enhanced deliverable" when GUI available.

#### 2C. Structured TechDraw Diagnostics

**File:** `freecad-mcp/src/freecad_mcp/techdraw_tools.py`

On any export failure:

```python
{
    "success": False,
    "error": "PDF export failed: TechDrawGui requires Qt GUI",
    "error_code": "TECHDRAW_NO_GUI",
    "diagnostics": {
        "freecad_version": "0.21.2",
        "gui_mode": False,
        "xvfb_available": True,
        "display_env": None,
        "template_found": True,
        "objects_in_view": 15
    },
    "recommendations": [
        "Run FreeCAD with Xvfb: xvfb-run -a freecad ...",
        "Or set QT_QPA_PLATFORM=offscreen",
        "Or use sitefit_export_pack for guaranteed PDF output"
    ],
    "fallback_available": "sitefit_export_pack"
}
```

---

### Phase 3: Contract Formalization with Migration

**Goal:** Versioned, traceable data contract with graceful migration.

#### 3A. Spatial Contract v1.0 JSON Schema

**New file:** `site-fit-mcp-server/schemas/spatial_contract_v1.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://sitefit.local/schemas/spatial_contract_v1.json",
  "title": "Spatial Contract",
  "version": "1.0.0",
  "type": "object",
  "required": ["contract_version", "project", "site", "placements"],
  "properties": {
    "contract_version": {"const": "1.0.0"},
    "project": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "revision": {"type": "string"}
      }
    },
    "site": {
      "type": "object",
      "required": ["boundary", "units"],
      "properties": {
        "boundary": {
          "type": "array",
          "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
        },
        "crs": {"type": "string", "description": "EPSG code or 'local'"},
        "units": {"enum": ["meters", "feet"]},
        "north_rotation_deg": {"type": "number", "default": 0}
      }
    },
    "program": {
      "type": "object",
      "properties": {
        "structures": {"type": "array"}
      }
    },
    "placements": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "x", "y"],
        "properties": {
          "id": {"type": "string"},
          "x": {"type": "number"},
          "y": {"type": "number"},
          "rotation_deg": {"type": "integer", "default": 0}
        }
      }
    },
    "road_network": {
      "type": "object",
      "properties": {
        "segments": {"type": "array"},
        "total_length": {"type": "number"}
      }
    },
    "provenance": {
      "type": "object",
      "properties": {
        "inputs_hash": {"type": "string"},
        "ruleset_id": {"type": "string"},
        "solver_seed": {"type": "integer"},
        "generated_at": {"type": "string", "format": "date-time"}
      }
    }
  }
}
```

#### 3B. Contract Validation with Migration Path

**Approach:** Start lenient, add strict mode later (per Codex recommendation).

```python
SUPPORTED_VERSIONS = ["1.0.0", "0.9"]  # 0.9 = legacy unversioned

def _validate_contract(contract: dict, strict: bool = False) -> dict:
    """
    Validate and optionally migrate contract.

    Args:
        contract: The contract to validate
        strict: If True, reject non-1.0.0 versions. If False, attempt migration.

    Returns:
        Migrated contract (or original if already valid)

    Raises:
        ValueError: If contract is invalid or migration fails
    """
    version = contract.get("contract_version", "0.9")  # Assume legacy if missing

    if version == "1.0.0":
        jsonschema.validate(contract, SPATIAL_CONTRACT_V1_SCHEMA)
        return contract

    if strict:
        raise ValueError(
            f"Contract version '{version}' not supported in strict mode. "
            f"Expected '1.0.0'. Re-export from site-fit."
        )

    # Migration path
    logger.warning("contract_migration", from_version=version, to_version="1.0.0")
    migrated = _migrate_contract(contract, version)
    jsonschema.validate(migrated, SPATIAL_CONTRACT_V1_SCHEMA)
    return migrated

def _migrate_contract(contract: dict, from_version: str) -> dict:
    """Migrate old contract format to v1.0.0."""
    if from_version == "0.9":
        # Legacy format: structure_id -> id, add missing fields
        return {
            "contract_version": "1.0.0",
            "project": contract.get("project", {"id": "", "name": "", "revision": "A"}),
            "site": {
                "boundary": contract.get("site", {}).get("boundary", []),
                "units": "meters",
                "crs": "local"
            },
            "placements": [
                {**p, "id": p.pop("structure_id", p.get("id"))}
                for p in contract.get("placements", [])
            ],
            "road_network": contract.get("road_network"),
            "provenance": {"generated_at": datetime.utcnow().isoformat()}
        }
    raise ValueError(f"No migration path from version '{from_version}'")
```

**Strict rejection becomes opt-in** (enable after migration path is stable):

```python
# In FreeCAD import:
contract = _validate_contract(raw_contract, strict=False)  # Default: migrate

# User can opt-in to strict:
contract = _validate_contract(raw_contract, strict=True)  # Reject old versions
```

---

### Phase 4: Civil Access + Schema Evolution

**Goal:** Roads that civil engineers can use, coupled with schema.

#### 4A. Road Corridor Representation in Schema

Extend `road_network` in v1.0 schema (backward compatible):

```python
"road_network": {
    "segments": [{
        "id": str,
        "centerline": [[x, y], ...],  # Required
        "width": float,               # Pavement width in meters
        "edge_left": [[x, y], ...],   # Computed offset (optional)
        "edge_right": [[x, y], ...],  # Computed offset (optional)
        "surface_type": "asphalt" | "gravel" | "concrete"
    }],
    "intersections": [{
        "id": str,
        "center": [x, y],
        "corner_radii": [float, ...]  # For each quadrant
    }],
    "total_length": float,
    "entrances_connected": [str],
    "structures_accessible": [str]
}
```

#### 4B. FreeCAD Road Rendering

**File:** `freecad-mcp/src/freecad_mcp/contract_tools.py`

Create roads with visual hierarchy:

```python
def _create_road_geometry(doc, segment: dict, layer_name: str):
    """Create road as centerline + edges."""
    # 1. Centerline (dashed gray)
    centerline = Draft.makeWire(
        [FreeCAD.Vector(p[0]*1000, p[1]*1000, 0) for p in segment["centerline"]],
        closed=False
    )
    centerline.ViewObject.LineColor = (0.5, 0.5, 0.5)
    centerline.ViewObject.DrawStyle = "Dashed"

    # 2. Edge of pavement (solid black) - if available
    if "edge_left" in segment:
        edge_left = Draft.makeWire(...)
        edge_left.ViewObject.LineColor = (0.0, 0.0, 0.0)
        edge_left.ViewObject.LineWidth = 1.5

    # 3. Optional filled face for visual clarity
    if segment.get("surface_type"):
        # Create face between edges
        ...
```

#### 4C. DXF Export with Road Layers

DXF layer structure for civil handoff:

```
BOUNDARY          - Site boundary polygon
STRUCTURES        - Equipment footprints
ROAD_CENTERLINE   - Road centerlines (for alignment)
ROAD_EDGE         - Edge of pavement
ROAD_SURFACE      - Hatched pavement areas (optional)
SETBACKS          - Build limits
KEEPOUTS          - No-build zones
```

---

### Phase 5: Polish

**Goal:** Developer experience and debugging tools.

#### 5A. Solution Diff Report

```python
@mcp.tool()
async def sitefit_compare_solutions(
    solution_id_a: str,
    solution_id_b: str
) -> dict:
    """Compare two solutions and explain differences."""
    return {
        "summary": "Solution B moves 3 structures and shortens roads by 12m",
        "moved_structures": [
            {"id": "TK-101", "delta_x": 5.2, "delta_y": -3.1},
            ...
        ],
        "rotation_changes": [
            {"id": "BLDG-001", "from": 0, "to": 90}
        ],
        "metric_deltas": {
            "road_length": -12.3,
            "compactness": 0.05,
            "pipe_length_proxy": -8.7
        },
        "road_network_changes": {
            "segments_added": 0,
            "segments_removed": 1,
            "rerouted": ["seg_3"]
        }
    }
```

---

## Verification Plan

### After Phase 1:
1. Generate job with 10 solutions
2. Call `sitefit_export_batch(limit=5)` - verify 5 contracts + `has_more=true`
3. Call `present_layout_options` - verify layers with roads in FreeCAD
4. Restart server, re-fetch job - verify solutions persist
5. Check logs for structured JSON output

### After Phase 2:
1. Run in headless mode (no Xvfb)
2. Call `techdraw_preflight` - verify recommendations include Xvfb
3. Call `sitefit_export_pack` - verify PDF generated
4. Run with `xvfb-run` - verify TechDraw exports work

### After Phase 3:
1. Export contract - verify `contract_version: "1.0.0"`
2. Import legacy contract (no version) - verify migration works
3. Import with `strict=True` - verify rejection with clear error

### After Phase 4:
1. Generate layout with roads
2. Export to DXF - verify road layers present
3. Open in AutoCAD/Civil3D - verify layers usable

---

## Files to Modify

### Site-Fit MCP Server
| File | Changes |
|------|---------|
| `src/server.py` | Batch export with paging, persistence with atomic writes, structured logging |
| `src/tools/export_tools.py` | Enhanced export_pack |
| `schemas/spatial_contract_v1.json` | New file: formal schema |

### FreeCAD MCP Server
| File | Changes |
|------|---------|
| `src/freecad_mcp/contract_tools.py` | Roads in layers, road rendering, migration |
| `src/freecad_mcp/techdraw_tools.py` | Preflight with Xvfb, diagnostics |

### Skill Documentation
| File | Changes |
|------|---------|
| `~/skills/site-fit-workflow/SKILL.md` | Update for batch export |
| `~/skills/site-fit-workflow/references/contract-schema.md` | Document v1.0 schema |

---

## Final Prioritization

Incorporating Codex recommendations for "utility fast":

| Phase | Items | Rationale |
|-------|-------|-----------|
| **1** | 1A-1E | Core fixes + logging for debugging everything else |
| **2** | 2A-2C | Deliverables work reliably (or fail clearly) |
| **3** | 3A-3B | Formalization with migration (not strict reject yet) |
| **4** | 4A-4C | Civil-grade roads coupled with schema |
| **5** | 5A | Polish (diff tool) |

---

## User Decisions

| Question | Decision |
|----------|----------|
| **Persistence storage** | File-based JSON (`~/.sitefit/jobs/`) with atomic writes |
| **Contract versioning** | Migration-first, strict rejection as opt-in later |
| **Phase priority** | Reliability first, roads second, formalization third |

---

## Codex Validations Incorporated

| Codex Finding | How Addressed |
|---------------|---------------|
| Batch export payload risk | Added paging with `limit`/`offset` |
| Roads not in contract schema | Phase 4A couples road schema with visualization |
| TechDraw preflight won't fix GUI | Added Xvfb detection + clear recommendations |
| Strict rejection = migration trap | Changed to migration-first with optional strict mode |
| Observability too late | Moved to Phase 1E |
| File persistence needs atomic writes | Added temp file + rename pattern |
| Draft API wrapper low-value | Marked as low priority, simplified |
