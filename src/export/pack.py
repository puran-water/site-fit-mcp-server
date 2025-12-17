"""Export pack bundler for site layout solutions.

Generates a complete deliverable package with multiple export formats:
- PDF plan sheet
- DXF CAD file
- CSV quantities
- GeoJSON
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
import tempfile

from shapely.geometry import Polygon

from ..models.solution import SiteFitSolution
from .quantities import compute_quantities, QuantityTakeoff
from .geojson import solution_to_geojson


@dataclass
class ExportPackResult:
    """Result of export pack generation."""

    success: bool = True
    formats_generated: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)  # format -> file path
    errors: Dict[str, str] = field(default_factory=dict)  # format -> error message
    quantities: Optional[Dict[str, Any]] = None


def export_pack(
    solution: SiteFitSolution,
    boundary: Polygon,
    formats: List[str],
    output_dir: Optional[str] = None,
    structure_types: Optional[Dict[str, str]] = None,
    topology_edges: Optional[List[Tuple[str, str, Dict[str, Any]]]] = None,
    buildable_area_m2: float = 0.0,
    buildable_polygon: Optional[Polygon] = None,
    keepouts: Optional[List[Dict[str, Any]]] = None,
    entrances: Optional[List[Dict[str, Any]]] = None,
    hazard_zones: Optional[List[Any]] = None,
    project_name: str = "",
    drawing_number: str = "",
    tight_constraints: Optional[List[str]] = None,
) -> ExportPackResult:
    """Generate export pack with multiple formats.

    Args:
        solution: The site layout solution
        boundary: Site boundary polygon
        formats: List of formats to generate: 'pdf', 'dxf', 'csv', 'geojson'
        output_dir: Directory for output files (uses temp dir if None)
        structure_types: Mapping of structure_id to equipment type
        topology_edges: List of (from_id, to_id, metadata) for pipe routing
        buildable_area_m2: Buildable area for utilization calculation
        buildable_polygon: Buildable area polygon for DXF BUILDLIMIT layer
        keepouts: Keepout zone definitions for export
        entrances: Entrance definitions for export
        hazard_zones: NFPA 820 hazard zones for export
        project_name: Project name for reports
        drawing_number: Drawing number for reports
        tight_constraints: List of bottleneck constraints for reports

    Returns:
        ExportPackResult with file paths and any errors
    """
    result = ExportPackResult()
    structure_types = structure_types or {}
    topology_edges = topology_edges or []
    tight_constraints = tight_constraints or []

    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(tempfile.mkdtemp(prefix="sitefit_export_"))

    # Compute quantities (used by multiple formats)
    takeoff = compute_quantities(
        solution=solution,
        boundary=boundary,
        structure_types=structure_types,
        topology_edges=topology_edges,
        buildable_area_m2=buildable_area_m2,
    )
    result.quantities = takeoff.to_dict()

    # Generate each requested format
    for fmt in formats:
        fmt = fmt.lower()
        try:
            if fmt == "csv":
                file_path = _export_csv(takeoff, output_path, solution.id)
                result.files["csv"] = str(file_path)
                result.formats_generated.append("csv")

            elif fmt == "geojson":
                file_path = _export_geojson(
                    solution, output_path, boundary, keepouts, entrances, hazard_zones
                )
                result.files["geojson"] = str(file_path)
                result.formats_generated.append("geojson")

            elif fmt == "dxf":
                file_path = _export_dxf(
                    solution, output_path, boundary, buildable_polygon,
                    keepouts, entrances, hazard_zones, structure_types
                )
                if file_path:
                    result.files["dxf"] = str(file_path)
                    result.formats_generated.append("dxf")
                else:
                    result.errors["dxf"] = "ezdxf not installed"

            elif fmt == "pdf":
                file_path = _export_pdf(
                    solution, output_path, boundary, takeoff,
                    project_name, drawing_number, tight_constraints
                )
                if file_path:
                    result.files["pdf"] = str(file_path)
                    result.formats_generated.append("pdf")
                else:
                    result.errors["pdf"] = "weasyprint not installed"

            else:
                result.errors[fmt] = f"Unknown format: {fmt}"

        except Exception as e:
            result.errors[fmt] = str(e)

    result.success = len(result.errors) == 0

    return result


def _export_csv(
    takeoff: QuantityTakeoff,
    output_path: Path,
    solution_id: str,
) -> Path:
    """Export quantities as CSV."""
    file_path = output_path / f"{solution_id}_quantities.csv"
    csv_content = takeoff.to_csv_string()
    file_path.write_text(csv_content)
    return file_path


def _export_geojson(
    solution: SiteFitSolution,
    output_path: Path,
    boundary: Polygon,
    keepouts: Optional[List[Dict[str, Any]]],
    entrances: Optional[List[Dict[str, Any]]],
    hazard_zones: Optional[List[Any]],
) -> Path:
    """Export solution as GeoJSON."""
    file_path = output_path / f"{solution.id}_layout.geojson"

    # Use existing solution GeoJSON or generate minimal version
    geojson = solution.to_geojson_feature_collection(
        include_site=True,
        include_roads=True,
        include_labels=True,
    )

    # Add boundary if not present
    has_boundary = any(
        f.get("properties", {}).get("kind") == "boundary"
        for f in geojson.get("features", [])
    )
    if not has_boundary:
        geojson["features"].insert(0, {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(boundary.exterior.coords)],
            },
            "properties": {
                "kind": "boundary",
                "layer": "BOUNDARY",
            },
        })

    with open(file_path, "w") as f:
        json.dump(geojson, f, indent=2)

    return file_path


def _export_dxf(
    solution: SiteFitSolution,
    output_path: Path,
    boundary: Polygon,
    buildable_polygon: Optional[Polygon],
    keepouts: Optional[List[Dict[str, Any]]],
    entrances: Optional[List[Dict[str, Any]]],
    hazard_zones: Optional[List[Any]],
    structure_types: Dict[str, str],
) -> Optional[Path]:
    """Export solution as DXF with engineering layers."""
    try:
        from .dxf import save_solution_to_dxf
    except ImportError:
        return None

    file_path = output_path / f"{solution.id}_layout.dxf"

    save_solution_to_dxf(
        solution=solution,
        filepath=str(file_path),
        boundary=boundary,
        buildable_polygon=buildable_polygon,
        keepouts=keepouts,
        entrances=entrances,
        hazard_zones=hazard_zones,
        structure_types=structure_types,
    )

    return file_path


def _export_pdf(
    solution: SiteFitSolution,
    output_path: Path,
    boundary: Polygon,
    takeoff: QuantityTakeoff,
    project_name: str,
    drawing_number: str,
    tight_constraints: List[str],
) -> Optional[Path]:
    """Export solution as PDF plan sheet."""
    try:
        from .pdf_report import generate_pdf_report, PDFReportConfig
    except ImportError:
        return None

    file_path = output_path / f"{solution.id}_plan.pdf"

    config = PDFReportConfig(
        title="Site Layout Plan",
        project_name=project_name,
        drawing_number=drawing_number or solution.id,
    )

    pdf_bytes = generate_pdf_report(
        solution=solution,
        boundary=boundary,
        takeoff=takeoff,
        config=config,
        tight_constraints=tight_constraints,
    )

    file_path.write_bytes(pdf_bytes)

    return file_path
