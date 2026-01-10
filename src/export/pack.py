"""Export pack bundler for site layout solutions.

Generates a complete deliverable package with multiple export formats:
- PDF plan sheet
- DXF CAD file
- CSV quantities
- GeoJSON
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import Polygon

from ..models.solution import SiteFitSolution
from .quantities import QuantityTakeoff, compute_quantities


@dataclass
class ExportPackResult:
    """Result of export pack generation."""

    success: bool = True
    formats_generated: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # format -> file path
    errors: dict[str, str] = field(default_factory=dict)  # format -> error message
    quantities: dict[str, Any] | None = None


def export_pack(
    solution: SiteFitSolution,
    boundary: Polygon,
    formats: list[str],
    output_dir: str | None = None,
    structure_types: dict[str, str] | None = None,
    topology_edges: list[tuple[str, str, dict[str, Any]]] | None = None,
    buildable_area_m2: float = 0.0,
    buildable_polygon: Polygon | None = None,
    keepouts: list[dict[str, Any]] | None = None,
    entrances: list[dict[str, Any]] | None = None,
    hazard_zones: list[Any] | None = None,
    project_name: str = "",
    drawing_number: str = "",
    tight_constraints: list[str] | None = None,
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
                    result.errors["pdf"] = "reportlab and weasyprint not installed"

            elif fmt == "svg":
                file_path = _export_svg(
                    solution, output_path, boundary, structure_types
                )
                result.files["svg"] = str(file_path)
                result.formats_generated.append("svg")

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
    keepouts: list[dict[str, Any]] | None,
    entrances: list[dict[str, Any]] | None,
    hazard_zones: list[Any] | None,
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
    buildable_polygon: Polygon | None,
    keepouts: list[dict[str, Any]] | None,
    entrances: list[dict[str, Any]] | None,
    hazard_zones: list[Any] | None,
    structure_types: dict[str, str],
) -> Path | None:
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


def _export_svg(
    solution: SiteFitSolution,
    output_path: Path,
    boundary: Polygon,
    structure_types: dict[str, str],
) -> Path:
    """Export solution as SVG for web viewing.

    Pure Python SVG generation - no external dependencies required.
    """
    file_path = output_path / f"{solution.id}_layout.svg"

    # Calculate bounds and viewBox
    bounds = boundary.bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    padding = max(width, height) * 0.05

    # SVG viewBox with some padding
    vb_minx = minx - padding
    vb_miny = miny - padding
    vb_width = width + 2 * padding
    vb_height = height + 2 * padding

    # Start SVG
    svg_lines = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" ',
        f'     viewBox="{vb_minx:.2f} {-vb_miny - vb_height:.2f} {vb_width:.2f} {vb_height:.2f}"',
        f'     width="{vb_width * 5:.0f}" height="{vb_height * 5:.0f}">',
        f'  <defs>',
        f'    <style>',
        f'      .boundary {{ fill: #f5f5f5; stroke: #333; stroke-width: 0.5; }}',
        f'      .structure-rect {{ fill: #cce5cc; stroke: #4a7c4a; stroke-width: 0.3; }}',
        f'      .structure-circle {{ fill: #b3d9ff; stroke: #336699; stroke-width: 0.3; }}',
        f'      .road {{ fill: none; stroke: #808080; stroke-width: 1; stroke-dasharray: 2,1; }}',
        f'      .label {{ font-family: Arial, sans-serif; font-size: 2px; text-anchor: middle; }}',
        f'    </style>',
        f'  </defs>',
        f'  <g transform="scale(1,-1)">',  # Flip Y axis
    ]

    # Draw boundary
    coords = list(boundary.exterior.coords)
    points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    svg_lines.append(f'    <polygon class="boundary" points="{points_str}" />')

    # Draw road network
    road_network = getattr(solution, "road_network", None)
    if road_network and hasattr(road_network, "segments"):
        for segment in road_network.segments:
            centerline = getattr(segment, "centerline", [])
            if len(centerline) >= 2:
                points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in centerline)
                svg_lines.append(f'    <polyline class="road" points="{points_str}" />')

    # Draw placements
    for placement in solution.placements:
        px, py = placement.x, placement.y
        structure = next(
            (s for s in (solution.structures or []) if s.id == placement.structure_id),
            None
        )

        if structure and structure.footprint:
            footprint = structure.footprint
            shape = footprint.get("shape", "rectangle")

            if shape == "circle":
                d = footprint.get("d", 10)
                radius = d / 2
                svg_lines.append(
                    f'    <circle class="structure-circle" cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" />'
                )
            else:
                w = footprint.get("w", 10)
                l = footprint.get("l", 10)
                svg_lines.append(
                    f'    <rect class="structure-rect" '
                    f'x="{px - w/2:.2f}" y="{py - l/2:.2f}" '
                    f'width="{w:.2f}" height="{l:.2f}" />'
                )
        else:
            # Generic marker
            svg_lines.append(
                f'    <circle class="structure-circle" cx="{px:.2f}" cy="{py:.2f}" r="2" />'
            )

        # Label (flip Y for text)
        svg_lines.append(
            f'    <text class="label" x="{px:.2f}" y="{-py:.2f}" '
            f'transform="scale(1,-1)">{placement.structure_id}</text>'
        )

    svg_lines.append('  </g>')
    svg_lines.append('</svg>')

    file_path.write_text("\n".join(svg_lines))
    return file_path


def _export_pdf(
    solution: SiteFitSolution,
    output_path: Path,
    boundary: Polygon,
    takeoff: QuantityTakeoff,
    project_name: str,
    drawing_number: str,
    tight_constraints: list[str],
) -> Path | None:
    """Export solution as PDF plan sheet.

    Tries ReportLab (headless-safe) first, falls back to weasyprint.
    """
    file_path = output_path / f"{solution.id}_plan.pdf"

    # Try headless-safe ReportLab first
    try:
        from .pdf_reportlab import PDFConfig, generate_pdf_headless

        config = PDFConfig(
            title="Site Layout Plan",
            project_name=project_name,
            drawing_number=drawing_number or solution.id,
        )

        pdf_bytes = generate_pdf_headless(
            solution=solution,
            boundary=boundary,
            takeoff=takeoff,
            config=config,
            tight_constraints=tight_constraints,
        )

        file_path.write_bytes(pdf_bytes)
        return file_path

    except ImportError:
        pass  # ReportLab not available, try weasyprint

    # Fallback to weasyprint (may require GUI dependencies)
    try:
        from .pdf_report import PDFReportConfig, generate_pdf_report

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

    except ImportError:
        return None
