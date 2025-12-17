# Refactoring Plan: IFC-Bonsai Integration for 3D Site Test Fits

## Executive Summary

Extend ifc-bonsai-mcp for process industries (wastewater, waste-to-energy) and integrate it with site-fit-mcp-server to enable human-in-the-loop 3D test fits with Blender visualization and editing.

**User Decisions:**
- Fork ifc-bonsai-mcp and maintain separately
- Detailed equipment models with nozzles, platforms, accurate geometry
- Both 3D visualization AND 3D constraint solving
- Both Blender native editing AND chat-based parameter tuning

---

## Architecture Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌────────────────┐
│  Claude/MCP Client  │────▶│  site-fit-mcp       │────▶│  IFC Export    │
│                     │     │  (2D/3D solver)     │     │  + State Store │
└─────────────────────┘     └─────────────────────┘     └───────┬────────┘
                                                                │
                                     ┌──────────────────────────┘
                                     ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌────────────────┐
│  process-ifc-mcp    │◀───▶│  Session State      │◀───▶│  Blender       │
│  (forked bonsai)    │     │  (Redis/SQLite)     │     │  + Bonsai      │
└─────────────────────┘     └─────────────────────┘     └────────────────┘
```

---

## Phase 1: Fork and Extend ifc-bonsai-mcp (Weeks 1-4)

### 1.1 Repository Setup
**New repo:** `process-ifc-mcp` (fork of Show2Instruct/ifc-bonsai-mcp)

```
process-ifc-mcp/
├── src/process_ifc_mcp/
│   ├── server.py
│   ├── mcp_functions/
│   │   ├── api_tools.py          # Original tools
│   │   └── process_tools/        # NEW
│   │       ├── wastewater_tools.py
│   │       ├── wte_tools.py
│   │       ├── common_tools.py
│   │       └── nozzle_tools.py
│   ├── equipment/                # NEW
│   │   ├── registry.py           # DEXPI-IFC mapping
│   │   ├── wastewater/           # Clarifiers, digesters, etc.
│   │   ├── wte/                  # Boilers, turbines, etc.
│   │   └── common/               # Tanks, pumps, HX
│   ├── geometry/                 # NEW
│   │   ├── primitives.py         # Trimesh builders
│   │   ├── nozzle_generator.py
│   │   └── ifc_converter.py
│   └── models/
│       └── equipment_params.py   # Pydantic schemas
```

### 1.2 New MCP Tools for Process Equipment

**Wastewater Tools:**
```python
@mcp.tool()
def create_clarifier(name, clarifier_type, diameter, sidewall_depth, ...)
@mcp.tool()
def create_digester(name, digester_type, diameter, height, roof_type, ...)
@mcp.tool()
def create_aeration_tank(name, tank_type, length, width, depth, ...)
```

**WTE Tools:**
```python
@mcp.tool()
def create_boiler(name, boiler_type, thermal_capacity_mw, ...)
@mcp.tool()
def create_turbine_generator(name, turbine_type, power_output_mw, ...)
@mcp.tool()
def create_scrubber(name, scrubber_type, gas_flow_m3h, ...)
@mcp.tool()
def create_cooling_tower(name, tower_type, cooling_capacity_mw, ...)
@mcp.tool()
def create_stack(name, height, base_diameter, top_diameter, ...)
```

**Common Tools:**
```python
@mcp.tool()
def create_process_tank(name, tank_type, diameter, height, roof_type, ...)
@mcp.tool()
def create_pump(name, pump_type, flow_m3h, head_m, ...)
@mcp.tool()
def create_heat_exchanger(name, hx_type, duty_kw, ...)
@mcp.tool()
def add_nozzle_to_equipment(equipment_guid, nozzle_spec)
```

### 1.3 DEXPI to IFC Mapping

| DEXPI Class | IFC4x3 Class | PredefinedType |
|-------------|--------------|----------------|
| Tank | IfcTank | VESSEL, STORAGE |
| Separator | IfcFlowTreatmentDevice | USERDEFINED |
| Pump | IfcPump | ENDSUCTION, SUBMERSIBLEPUMP |
| Boiler | IfcBoiler | STEAM |
| HeatExchanger | IfcHeatExchanger | SHELLANDTUBE, PLATE |
| CoolingTower | IfcCoolingTower | MECHANICALFORCEDDRAFT |
| Chimney | IfcChimney | - |

### 1.4 Geometry Generation (Trimesh → IFC)

Use existing `create_trimesh_ifc` pattern from ifc-bonsai-mcp:
- Generate parametric meshes with Trimesh
- Convert to IFC via IfcOpenShell ShapeBuilder
- Include nozzle geometry, platforms, ladders

---

## Phase 2: 3D Solver Extension for site-fit (Weeks 3-6)

### 2.1 Model Extensions

**File: `src/models/structures.py`**
```python
class Elevation3DConfig(BaseModel):
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    z_preferred: Optional[float] = None
    elevation_mode: Literal["ground", "elevated", "underground", "floating"] = "ground"
    can_support_load: bool = False
    clearance_above: float = 3.0

class StructureFootprint(BaseModel):
    # ... existing fields ...
    height: Optional[float] = None  # Already exists!
    elevation_3d: Optional[Elevation3DConfig] = None  # NEW
    weight_kg: Optional[float] = None  # NEW
```

**File: `src/models/solution.py`**
```python
class Placement3D(BaseModel):
    structure_id: str
    x: float
    y: float
    z: float = 0.0  # NEW
    rotation_deg: int = 0
    width: float
    height: float
    depth: float = 0.0  # Vertical extent, NEW
```

**File: `src/models/rules.py`**
```python
class VerticalClearanceRules(BaseModel):
    max_site_elevation: float = 50.0
    default_clearance_above: float = 3.0
    equipment_vertical_clearance: Dict[str, float] = {...}

class StackingRules(BaseModel):
    allowed_stacking: Dict[str, Dict[str, float]] = {...}
    prohibited_stacking: List[Tuple[str, str]] = [...]

class GradeRules(BaseModel):
    gravity_flow_slope: float = 0.01
    max_conveyor_slope: float = 0.18

class RuleSet(BaseModel):
    # ... existing ...
    vertical_clearance: VerticalClearanceRules  # NEW
    stacking: StackingRules  # NEW
    grade: GradeRules  # NEW
    enable_3d: bool = False  # NEW
```

### 2.2 3D CP-SAT Solver

**New File: `src/solver/cpsat_placer_3d.py`**

```python
class PlacementSolver3D(PlacementSolver):
    def __init__(self, ..., z_bounds: Tuple[float, float]):
        self.z_bounds = z_bounds
        self.struct_vars_3d: Dict[str, StructureVars3D] = {}

    def _create_structure_vars(self, struct, ...):
        # Create z_var for each structure
        z_var = self.model.NewIntVar(z_min_grid, z_max_grid, f"z_{struct.id}")
        z_interval = self.model.NewIntervalVar(z_var, depth_grid, ...)

    def _add_3d_overlap_constraints(self):
        # For each pair:
        # IF XY overlaps THEN Z must not overlap
        # Use reified booleans for conditional constraints

    def _add_vertical_clearance_constraints(self):
        # Apply vertical_clearance rules from RuleSet

    def _add_grade_constraints(self):
        # Gravity flow constraints based on topology
```

**Solver Factory:**
```python
class PlacementSolverFactory:
    @staticmethod
    def create(...) -> Union[PlacementSolver, PlacementSolver3D]:
        if rules.enable_3d and any(s.elevation_3d for s in structures):
            return PlacementSolver3D(...)
        return PlacementSolver(...)
```

### 2.3 3D Validation

**New File: `src/geometry/clearance_3d.py`**
```python
def compute_3d_distance(p1: Placement3D, p2: Placement3D) -> Tuple[float, float, float]:
    """Return (distance_3d, distance_xy, distance_z)"""

def check_3d_clearance_violations(placements, rules, structure_types):
    """Check all pairwise 3D clearances"""

def validate_stacking_relationships(placements, rules, structure_types):
    """Validate stacking is physically feasible"""
```

---

## Phase 3: Integration Architecture (Weeks 5-8)

### 3.1 Communication Pattern

**Separate servers with shared state:**
- site-fit-mcp: Port 8765 (constraint solving)
- process-ifc-mcp: Port 8766 (3D visualization)
- Shared state: Redis or SQLite

**Session State Schema:**
```python
class IntegrationSession:
    session_id: str
    job_id: str
    solution_id: str
    ifc_file_path: Optional[str]
    entity_map: Dict[str, str]  # site-fit ID → IFC GUID
    coordinate_transform: CoordinateTransformer
    pending_changes: List[PlacementChange]
    validation_status: str
```

### 3.2 New Tools for site-fit-mcp

```python
@mcp.tool()
async def sitefit_export_to_ifc(
    solution_id: str,
    output_path: Optional[str] = None,
    include_roads: bool = True,
    default_height: float = 5.0,
) -> Dict[str, Any]:
    """Export solution to IFC for Blender visualization"""

@mcp.tool()
async def sitefit_validate_placements(
    solution_id: str,
    placements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate edited placements from Blender"""

@mcp.tool()
async def sitefit_update_solution(
    solution_id: str,
    placements: List[Dict[str, Any]],
    reoptimize: bool = False,
    fixed_structures: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Update solution with user edits"""

@mcp.tool()
async def sitefit_create_integration_session(
    solution_id: str,
) -> Dict[str, Any]:
    """Create session for Blender editing workflow"""
```

### 3.3 New Tools for process-ifc-mcp

```python
@mcp.tool()
async def bonsai_load_sitefit_solution(
    ifc_path: str,
    session_id: str,
    camera_preset: str = "top_orthographic",
) -> Dict[str, Any]:
    """Load site-fit IFC in Blender"""

@mcp.tool()
async def bonsai_capture_positions(
    session_id: str,
) -> Dict[str, Any]:
    """Capture current positions from Blender"""

@mcp.tool()
async def bonsai_highlight_violations(
    violations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Highlight constraint violations in viewport"""

@mcp.tool()
async def bonsai_sync_to_sitefit(
    session_id: str,
    commit: bool = False,
) -> Dict[str, Any]:
    """Sync changes back to site-fit for validation"""
```

### 3.4 Edit-Validate Workflow

```
1. sitefit_generate() → multiple 2D/3D solutions
2. sitefit_create_integration_session(solution_id)
3. sitefit_export_to_ifc(solution_id) → IFC file
4. bonsai_load_sitefit_solution(ifc_path, session_id)
5. User edits in Blender viewport
6. bonsai_capture_positions(session_id)
7. sitefit_validate_placements(solution_id, placements)
8. bonsai_highlight_violations(violations)
9. Iterate until valid
10. sitefit_update_solution(solution_id, placements, commit=True)
```

### 3.5 Coordinate System Handling

```python
class CoordinateTransformer:
    def sitefit_to_ifc(self, x, y, z=0.0) -> Tuple[float, float, float]:
        # Apply rotation + translation to IFC coords

    def ifc_to_sitefit(self, x, y, z) -> Tuple[float, float]:
        # Transform back to site-fit 2D coords

    def rotation_ifc_to_sitefit(self, ifc_rotation_deg) -> int:
        # Snap to 0/90/180/270
```

---

## Critical Files to Modify

### site-fit-mcp-server:
1. `src/models/structures.py` - Add Elevation3DConfig, extend StructureFootprint
2. `src/models/solution.py` - Add Placement3D class
3. `src/models/rules.py` - Add VerticalClearanceRules, StackingRules, GradeRules
4. `src/solver/cpsat_placer.py` - Create cpsat_placer_3d.py extending this
5. `src/geometry/clearance.py` - Create clearance_3d.py following this pattern
6. `src/pipeline.py` - Integrate 3D validation, solver factory
7. `src/server.py` - Add IFC export + integration tools
8. `src/export/geojson.py` - Pattern for new `src/export/ifc.py`

### process-ifc-mcp (new fork):
1. `src/process_ifc_mcp/mcp_functions/process_tools/*.py` - Equipment tools
2. `src/process_ifc_mcp/equipment/*.py` - Geometry generators
3. `src/process_ifc_mcp/geometry/*.py` - Trimesh/IFC utilities
4. `src/process_ifc_mcp/models/equipment_params.py` - Parameter schemas

---

## Implementation Sequence (3D Visualization First)

**Priority:** Get visual feedback loop working early, then enhance incrementally.

| Week | Phase | Tasks |
|------|-------|-------|
| 1 | **IFC Export Foundation** | Add `sitefit_export_to_ifc` tool to site-fit; create basic `src/export/ifc.py` module; extrude existing footprints (height field) to 3D boxes/cylinders |
| 2 | **Fork + Basic Blender Integration** | Fork ifc-bonsai-mcp as `process-ifc-mcp`; implement `bonsai_load_sitefit_solution`; basic camera setup + color coding |
| 3 | **Edit-Validate Loop** | `bonsai_capture_positions`, `sitefit_validate_placements`, `bonsai_highlight_violations`; coordinate transformer |
| 4 | **Session Management + Sync** | Integration session state; `sitefit_update_solution`; complete roundtrip workflow |
| 5 | **3D Data Models** | Add Elevation3DConfig, Placement3D, 3D rules to site-fit; simple vertical clearance validation |
| 6 | **3D Solver (Basic)** | PlacementSolver3D with z-variables; basic 3D non-overlap constraints; 2D fallback |
| 7 | **Common Equipment** | Tanks, pumps, heat exchangers with parametric geometry; nozzle generation |
| 8 | **Wastewater Equipment** | Clarifiers, digesters, aeration tanks with detailed geometry |
| 9 | **WTE Equipment + Polish** | Boilers, turbines, scrubbers, stacks; advanced 3D solver constraints (stacking, grade); testing |

**Milestone Checkpoints:**
- Week 2: Can export site-fit solution to IFC and view in Blender
- Week 4: Can edit positions in Blender and validate back through site-fit
- Week 6: 3D solver generating solutions with elevation constraints
- Week 9: Full process equipment library with human-in-the-loop editing

---

## Testing Strategy

1. **Unit tests** for each equipment geometry generator
2. **Integration tests** for DEXPI → IFC → Blender roundtrip
3. **3D solver tests**: stacking, vertical clearance, grade constraints
4. **Workflow tests**: complete edit-validate loop
5. **Visual validation**: screenshot comparison for equipment geometry

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Complex equipment geometry | Start with simple parametric shapes, iterate |
| IFC schema limitations | Use USERDEFINED + custom Psets |
| 3D solver performance | Provide 2D fallback mode |
| Coordinate system errors | Comprehensive transformer tests |
| Blender communication latency | Batch operations where possible |

---

## References

- **ifc-bonsai-mcp:** https://github.com/Show2Instruct/ifc-bonsai-mcp
- **DeepWiki documentation:** Used for architecture analysis
- **IFC4x3 Schema:** https://ifc43-docs.standards.buildingsmart.org/
- **IfcOpenShell:** https://ifcopenshell.org/
