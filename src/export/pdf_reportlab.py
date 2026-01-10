"""Headless PDF plan sheet generation using ReportLab.

This module provides a GUI-free alternative to weasyprint-based PDF generation.
It's designed for server environments without X11/Qt dependencies.

Uses ReportLab for PDF generation and svglib for SVG rendering.
"""

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from shapely.geometry import Polygon

from ..models.solution import SiteFitSolution
from .quantities import QuantityTakeoff


@dataclass
class PDFConfig:
    """Configuration for headless PDF report generation."""

    page_size: str = "A3"  # A3 landscape for plan sheets
    title: str = "Site Layout Plan"
    project_name: str = ""
    drawing_number: str = ""
    revision: str = "A"
    scale: str | None = None  # Auto-calculated if None
    include_quantities: bool = True
    include_constraints: bool = True
    company_name: str = ""
    prepared_by: str = ""


def generate_pdf_headless(
    solution: SiteFitSolution,
    boundary: Polygon,
    takeoff: QuantityTakeoff,
    config: PDFConfig | None = None,
    tight_constraints: list[str] | None = None,
) -> bytes:
    """Generate PDF plan sheet using ReportLab (headless-safe).

    Args:
        solution: The site layout solution
        boundary: Site boundary polygon
        takeoff: Computed quantity takeoff
        config: PDF report configuration
        tight_constraints: List of top 5 bottleneck constraints

    Returns:
        PDF bytes ready for file writing

    Raises:
        ImportError: If reportlab is not installed
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A3, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        from reportlab.platypus import Paragraph, Table, TableStyle
    except ImportError:
        raise ImportError(
            "reportlab is required for headless PDF export. "
            "Install with: pip install 'site-fit-mcp-server[pdf-headless]'"
        )

    config = config or PDFConfig()
    tight_constraints = tight_constraints or []

    # Page setup - A3 landscape
    page_width, page_height = landscape(A3)
    margin = 15 * mm

    # Calculate bounds and scale
    bounds = boundary.bounds  # (minx, miny, maxx, maxy)
    site_width_m = bounds[2] - bounds[0]
    site_height_m = bounds[3] - bounds[1]

    # Drawing area (left 2/3 of page for layout, right 1/3 for legend/quantities)
    draw_area_width = (page_width - 3 * margin) * 0.65
    draw_area_height = page_height - 4 * margin - 30 * mm  # Leave space for title block

    # Calculate scale
    scale_x = draw_area_width / (site_width_m * 1000)  # points per meter
    scale_y = draw_area_height / (site_height_m * 1000)
    scale_factor = min(scale_x, scale_y) * 0.85  # 85% to leave margin

    # Determine nice scale string
    scale_ratio = int(1000 / (scale_factor / (72 / 25.4))) if scale_factor > 0 else 1000
    scale_string = config.scale or f"1:{_nice_scale(scale_ratio)}"

    # Create PDF
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A3))

    # Drawing area origin (bottom-left of layout area)
    origin_x = margin
    origin_y = margin + 25 * mm  # Above title block

    # Draw border
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.rect(margin, margin, page_width - 2 * margin, page_height - 2 * margin)

    # Title block (bottom)
    _draw_title_block(
        c,
        margin,
        margin,
        page_width - 2 * margin,
        25 * mm,
        config,
        scale_string,
    )

    # Draw layout
    _draw_site_layout(
        c,
        solution,
        boundary,
        bounds,
        origin_x + 5 * mm,
        origin_y + 5 * mm,
        draw_area_width - 10 * mm,
        draw_area_height - 10 * mm,
        scale_factor,
    )

    # Legend and quantities (right panel)
    panel_x = margin + draw_area_width + 10 * mm
    panel_width = page_width - panel_x - margin
    _draw_legend_panel(
        c,
        panel_x,
        origin_y,
        panel_width,
        draw_area_height,
        takeoff,
        tight_constraints,
        config,
    )

    # Scale bar
    _draw_scale_bar(
        c,
        origin_x + 10 * mm,
        origin_y + 10 * mm,
        scale_factor,
        scale_string,
    )

    # North arrow
    _draw_north_arrow(
        c,
        origin_x + draw_area_width - 20 * mm,
        origin_y + draw_area_height - 15 * mm,
    )

    c.save()
    buffer.seek(0)
    return buffer.read()


def _nice_scale(scale: int) -> int:
    """Round scale to a nice engineering value."""
    nice_scales = [50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]
    for ns in nice_scales:
        if scale <= ns * 1.2:
            return ns
    return scale


def _draw_title_block(
    c,
    x: float,
    y: float,
    width: float,
    height: float,
    config: PDFConfig,
    scale_string: str,
):
    """Draw title block at bottom of page."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    # Border
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.5)
    c.rect(x, y, width, height)

    # Vertical dividers
    dividers = [width * 0.4, width * 0.6, width * 0.75, width * 0.9]
    for dx in dividers:
        c.line(x + dx, y, x + dx, y + height)

    # Text content
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 5 * mm, y + height - 8 * mm, config.title)

    c.setFont("Helvetica", 9)
    c.drawString(x + 5 * mm, y + height - 15 * mm, f"Project: {config.project_name}")
    c.drawString(x + 5 * mm, y + 3 * mm, f"Company: {config.company_name or 'N/A'}")

    # Drawing number
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + dividers[0] + 3 * mm, y + height - 8 * mm, "DWG NO.")
    c.setFont("Helvetica", 10)
    c.drawString(x + dividers[0] + 3 * mm, y + height - 16 * mm, config.drawing_number or "---")

    # Scale
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + dividers[1] + 3 * mm, y + height - 8 * mm, "SCALE")
    c.setFont("Helvetica", 10)
    c.drawString(x + dividers[1] + 3 * mm, y + height - 16 * mm, scale_string)

    # Revision
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + dividers[2] + 3 * mm, y + height - 8 * mm, "REV")
    c.setFont("Helvetica", 10)
    c.drawString(x + dividers[2] + 3 * mm, y + height - 16 * mm, config.revision)

    # Date
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + dividers[3] + 3 * mm, y + height - 8 * mm, "DATE")
    c.setFont("Helvetica", 10)
    c.drawString(x + dividers[3] + 3 * mm, y + height - 16 * mm, datetime.now().strftime("%Y-%m-%d"))


def _draw_site_layout(
    c,
    solution: SiteFitSolution,
    boundary: Polygon,
    bounds: tuple[float, float, float, float],
    x: float,
    y: float,
    width: float,
    height: float,
    scale_factor: float,
):
    """Draw the site layout diagram."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    minx, miny, maxx, maxy = bounds

    def transform(px: float, py: float) -> tuple[float, float]:
        """Transform site coordinates to PDF coordinates."""
        tx = x + (px - minx) * scale_factor
        ty = y + (py - miny) * scale_factor
        return tx, ty

    # Draw boundary
    c.setStrokeColor(colors.black)
    c.setLineWidth(1.5)
    c.setFillColor(colors.Color(0.95, 0.95, 0.95))

    coords = list(boundary.exterior.coords)
    path = c.beginPath()
    tx, ty = transform(coords[0][0], coords[0][1])
    path.moveTo(tx, ty)
    for coord in coords[1:]:
        tx, ty = transform(coord[0], coord[1])
        path.lineTo(tx, ty)
    path.close()
    c.drawPath(path, fill=1, stroke=1)

    # Draw road network
    road_network = getattr(solution, "road_network", None)
    if road_network and hasattr(road_network, "segments"):
        c.setStrokeColor(colors.Color(0.5, 0.5, 0.5))
        c.setLineWidth(2)
        c.setDash([4, 2])
        for segment in road_network.segments:
            centerline = getattr(segment, "centerline", [])
            if len(centerline) >= 2:
                path = c.beginPath()
                tx, ty = transform(centerline[0][0], centerline[0][1])
                path.moveTo(tx, ty)
                for pt in centerline[1:]:
                    tx, ty = transform(pt[0], pt[1])
                    path.lineTo(tx, ty)
                c.drawPath(path, fill=0, stroke=1)
        c.setDash([])

    # Draw placements
    for placement in solution.placements:
        px, py = placement.x, placement.y
        tx, ty = transform(px, py)

        # Get structure info
        structure = next(
            (s for s in (solution.structures or []) if s.id == placement.structure_id),
            None
        )

        if structure:
            footprint = structure.footprint
            shape = footprint.get("shape", "rectangle") if footprint else "rectangle"

            if shape == "circle":
                d = footprint.get("d", 10) if footprint else 10
                radius = d * scale_factor / 2
                c.setFillColor(colors.Color(0.7, 0.85, 1.0))
                c.setStrokeColor(colors.Color(0.2, 0.4, 0.6))
                c.setLineWidth(1)
                c.circle(tx, ty, radius, fill=1, stroke=1)
            else:
                w = footprint.get("w", 10) if footprint else 10
                l = footprint.get("l", 10) if footprint else 10
                rect_w = w * scale_factor
                rect_l = l * scale_factor
                c.setFillColor(colors.Color(0.8, 0.9, 0.8))
                c.setStrokeColor(colors.Color(0.3, 0.5, 0.3))
                c.setLineWidth(1)
                c.rect(tx - rect_w / 2, ty - rect_l / 2, rect_w, rect_l, fill=1, stroke=1)

            # Label
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 6)
            c.drawCentredString(tx, ty - 3, placement.structure_id)
        else:
            # Generic marker
            c.setFillColor(colors.Color(0.9, 0.9, 0.9))
            c.setStrokeColor(colors.black)
            c.circle(tx, ty, 3 * mm, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 5)
            c.drawCentredString(tx, ty - 2, placement.structure_id[:8])


def _draw_legend_panel(
    c,
    x: float,
    y: float,
    width: float,
    height: float,
    takeoff: QuantityTakeoff,
    tight_constraints: list[str],
    config: PDFConfig,
):
    """Draw legend and quantities panel."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    # Border
    c.setStrokeColor(colors.Color(0.8, 0.8, 0.8))
    c.setLineWidth(0.5)
    c.rect(x, y, width, height)

    # Legend title
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.black)
    c.drawString(x + 5 * mm, y + height - 10 * mm, "LEGEND")

    # Legend items
    legend_y = y + height - 20 * mm
    c.setFont("Helvetica", 8)

    # Boundary
    c.setFillColor(colors.Color(0.95, 0.95, 0.95))
    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.rect(x + 5 * mm, legend_y - 2 * mm, 8 * mm, 4 * mm, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.drawString(x + 16 * mm, legend_y, "Site Boundary")
    legend_y -= 10 * mm

    # Structure (rectangle)
    c.setFillColor(colors.Color(0.8, 0.9, 0.8))
    c.setStrokeColor(colors.Color(0.3, 0.5, 0.3))
    c.rect(x + 5 * mm, legend_y - 2 * mm, 8 * mm, 4 * mm, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.drawString(x + 16 * mm, legend_y, "Equipment (Rect)")
    legend_y -= 10 * mm

    # Structure (circle)
    c.setFillColor(colors.Color(0.7, 0.85, 1.0))
    c.setStrokeColor(colors.Color(0.2, 0.4, 0.6))
    c.circle(x + 9 * mm, legend_y, 2 * mm, fill=1, stroke=1)
    c.setFillColor(colors.black)
    c.drawString(x + 16 * mm, legend_y, "Equipment (Circ)")
    legend_y -= 10 * mm

    # Road
    c.setStrokeColor(colors.Color(0.5, 0.5, 0.5))
    c.setLineWidth(2)
    c.setDash([4, 2])
    c.line(x + 5 * mm, legend_y, x + 13 * mm, legend_y)
    c.setDash([])
    c.setFillColor(colors.black)
    c.drawString(x + 16 * mm, legend_y, "Access Road")
    legend_y -= 15 * mm

    # Quantities section
    if config.include_quantities:
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 5 * mm, legend_y, "QUANTITIES")
        legend_y -= 10 * mm

        c.setFont("Helvetica", 8)
        quantities = takeoff.to_dict()

        qty_items = [
            ("Total Pad Area", f"{quantities.get('total_pad_area_m2', 0):.1f} m²"),
            ("Road Length", f"{quantities.get('road_length_m', 0):.1f} m"),
            ("Pipe Proxy", f"{quantities.get('pipe_length_proxy_m', 0):.1f} m"),
            ("Fence Perimeter", f"{quantities.get('fence_perimeter_m', 0):.1f} m"),
            ("Utilization", f"{quantities.get('utilization_pct', 0):.1f}%"),
        ]

        for label, value in qty_items:
            c.drawString(x + 5 * mm, legend_y, f"{label}:")
            c.drawRightString(x + width - 5 * mm, legend_y, value)
            legend_y -= 8 * mm

    # Constraints section
    if config.include_constraints and tight_constraints:
        legend_y -= 5 * mm
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 5 * mm, legend_y, "CONSTRAINTS")
        legend_y -= 10 * mm

        c.setFont("Helvetica", 7)
        for constraint in tight_constraints[:5]:
            # Wrap long text
            text = constraint[:35] + "..." if len(constraint) > 35 else constraint
            c.drawString(x + 5 * mm, legend_y, f"• {text}")
            legend_y -= 7 * mm


def _draw_scale_bar(
    c,
    x: float,
    y: float,
    scale_factor: float,
    scale_string: str,
):
    """Draw a scale bar."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    # Calculate 10m bar length in points
    bar_length = 10 * scale_factor  # 10 meters

    # If bar would be too small or too large, adjust
    if bar_length < 20 * mm:
        bar_meters = 5
        bar_length = 5 * scale_factor
    elif bar_length > 80 * mm:
        bar_meters = 20
        bar_length = 20 * scale_factor
    else:
        bar_meters = 10

    # Draw bar
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.black)
    c.setLineWidth(1)

    # Alternating black/white segments
    segment_length = bar_length / 2
    c.rect(x, y, segment_length, 3 * mm, fill=1, stroke=1)
    c.setFillColor(colors.white)
    c.rect(x + segment_length, y, segment_length, 3 * mm, fill=1, stroke=1)

    # Labels
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 7)
    c.drawString(x, y - 4 * mm, "0")
    c.drawCentredString(x + segment_length, y - 4 * mm, f"{bar_meters // 2}")
    c.drawString(x + bar_length, y - 4 * mm, f"{bar_meters}m")

    # Scale text
    c.setFont("Helvetica", 8)
    c.drawString(x, y + 5 * mm, f"Scale {scale_string}")


def _draw_north_arrow(c, x: float, y: float):
    """Draw a north arrow."""
    from reportlab.lib import colors
    from reportlab.lib.units import mm

    arrow_height = 10 * mm
    arrow_width = 6 * mm

    # Arrow shape
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.black)
    c.setLineWidth(1)

    # Draw arrow
    path = c.beginPath()
    path.moveTo(x, y + arrow_height)  # Top point
    path.lineTo(x - arrow_width / 2, y)  # Bottom left
    path.lineTo(x, y + arrow_height / 3)  # Notch
    path.lineTo(x + arrow_width / 2, y)  # Bottom right
    path.close()
    c.drawPath(path, fill=1, stroke=1)

    # N label
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(x, y + arrow_height + 3 * mm, "N")
