"""PDF plan sheet generation for site layout solutions.

Generates engineering-style plan sheets with:
- Layout diagram with scale bar and north arrow
- Legend
- Quantities table
- Constraint summary
"""

from dataclasses import dataclass
from datetime import datetime

from shapely.geometry import Polygon

from ..models.solution import SiteFitSolution
from .quantities import QuantityTakeoff


@dataclass
class PDFReportConfig:
    """Configuration for PDF report generation."""

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


def generate_pdf_report(
    solution: SiteFitSolution,
    boundary: Polygon,
    takeoff: QuantityTakeoff,
    config: PDFReportConfig | None = None,
    tight_constraints: list[str] | None = None,
) -> bytes:
    """Generate PDF plan sheet for a site layout solution.

    Args:
        solution: The site layout solution
        boundary: Site boundary polygon
        takeoff: Computed quantity takeoff
        config: PDF report configuration
        tight_constraints: List of top 5 bottleneck constraints

    Returns:
        PDF bytes ready for file writing

    Raises:
        ImportError: If weasyprint is not installed
    """
    try:
        from weasyprint import CSS, HTML
    except ImportError:
        raise ImportError(
            "weasyprint is required for PDF export. "
            "Install with: pip install site-fit-mcp-server[pdf]"
        )

    config = config or PDFReportConfig()
    tight_constraints = tight_constraints or []

    # Calculate bounds and scale
    bounds = boundary.bounds  # (minx, miny, maxx, maxy)
    width_m = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]

    # Auto-calculate scale for A3 (420mm x 297mm, use 380x260 for drawing area)
    draw_width_mm = 380
    draw_height_mm = 260

    scale_x = draw_width_mm / (width_m * 1000)  # mm per m
    scale_y = draw_height_mm / (height_m * 1000)
    scale_factor = min(scale_x, scale_y) * 0.9  # 90% to leave margin

    # SVG viewBox dimensions
    svg_width = width_m * scale_factor * 1000  # in mm scaled to px (1px = 1mm for simplicity)
    svg_height = height_m * scale_factor * 1000

    # Determine nice scale string
    scale_ratio = int(1000 / scale_factor) if scale_factor > 0 else 1000
    scale_string = config.scale or f"1:{_nice_scale(scale_ratio)}"

    # Generate SVG layout
    svg_content = _generate_layout_svg(
        solution,
        boundary,
        bounds,
        svg_width,
        svg_height,
        scale_factor,
    )

    # Generate HTML
    html_content = _generate_html_report(
        svg_content,
        takeoff,
        config,
        scale_string,
        tight_constraints,
        svg_width,
        svg_height,
    )

    # Convert to PDF
    html = HTML(string=html_content)
    css = CSS(string=_get_report_css())

    return html.write_pdf(stylesheets=[css])


def _nice_scale(ratio: int) -> int:
    """Round scale ratio to a nice engineering scale."""
    nice_scales = [100, 200, 250, 500, 1000, 1500, 2000, 2500, 5000, 10000]
    for scale in nice_scales:
        if ratio <= scale:
            return scale
    return ratio


def _generate_layout_svg(
    solution: SiteFitSolution,
    boundary: Polygon,
    bounds: tuple[float, float, float, float],
    svg_width: float,
    svg_height: float,
    scale_factor: float,
) -> str:
    """Generate SVG layout diagram."""
    minx, miny, maxx, maxy = bounds

    def tx(x: float) -> float:
        """Transform X coordinate to SVG space."""
        return (x - minx) * scale_factor * 1000

    def ty(y: float) -> float:
        """Transform Y coordinate to SVG space (inverted Y)."""
        return svg_height - (y - miny) * scale_factor * 1000

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_width}mm" height="{svg_height}mm" '
        f'viewBox="0 0 {svg_width} {svg_height}">'
    ]

    # Background
    svg_parts.append(f'<rect x="0" y="0" width="{svg_width}" height="{svg_height}" fill="white"/>')

    # Site boundary
    boundary_coords = list(boundary.exterior.coords)
    points = " ".join(f"{tx(x)},{ty(y)}" for x, y in boundary_coords)
    svg_parts.append(
        f'<polygon points="{points}" fill="none" stroke="#333" stroke-width="0.5" stroke-dasharray="2,1"/>'
    )

    # Structures
    for p in solution.placements:
        cx, cy = tx(p.x), ty(p.y)
        w = p.width * scale_factor * 1000
        h = p.height * scale_factor * 1000

        if p.is_circle:
            r = w / 2
            svg_parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" '
                f'fill="#4A90D9" fill-opacity="0.7" stroke="#2E6BA8" stroke-width="0.3"/>'
            )
        else:
            svg_parts.append(
                f'<rect x="{cx - w/2}" y="{cy - h/2}" width="{w}" height="{h}" '
                f'fill="#7BC47F" fill-opacity="0.7" stroke="#4A8A4E" stroke-width="0.3" '
                f'transform="rotate({-p.rotation_deg} {cx} {cy})"/>'
            )

        # Label
        label = p.structure_id
        font_size = max(2, min(6, w / 4))
        svg_parts.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{font_size}px" font-family="Arial, sans-serif" fill="#333">{label}</text>'
        )

    # Roads
    if solution.road_network:
        for seg in solution.road_network.segments:
            coords = seg.to_linestring_coords()
            points = " ".join(f"{tx(x)},{ty(y)}" for x, y in coords)
            svg_parts.append(
                f'<polyline points="{points}" fill="none" stroke="#888" stroke-width="1.5"/>'
            )

    # North arrow (top right corner)
    arrow_x = svg_width - 15
    arrow_y = 15
    svg_parts.append(
        f'<g transform="translate({arrow_x},{arrow_y})">'
        f'<polygon points="0,-10 -5,5 0,0 5,5" fill="#333"/>'
        f'<text x="0" y="12" text-anchor="middle" font-size="6px" font-family="Arial">N</text>'
        f'</g>'
    )

    # Scale bar (bottom left)
    scale_bar_length_m = _nice_scale_bar_length(bounds)
    scale_bar_px = scale_bar_length_m * scale_factor * 1000
    bar_x, bar_y = 10, svg_height - 10
    svg_parts.append(
        f'<g transform="translate({bar_x},{bar_y})">'
        f'<rect x="0" y="-3" width="{scale_bar_px}" height="3" fill="#333"/>'
        f'<line x1="0" y1="-3" x2="0" y2="-6" stroke="#333" stroke-width="0.5"/>'
        f'<line x1="{scale_bar_px}" y1="-3" x2="{scale_bar_px}" y2="-6" stroke="#333" stroke-width="0.5"/>'
        f'<text x="{scale_bar_px/2}" y="-8" text-anchor="middle" font-size="5px" font-family="Arial">{scale_bar_length_m}m</text>'
        f'</g>'
    )

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def _nice_scale_bar_length(bounds: tuple[float, float, float, float]) -> int:
    """Calculate nice scale bar length for the site size."""
    width = bounds[2] - bounds[0]
    target = width / 5  # About 1/5 of site width
    nice_lengths = [5, 10, 20, 25, 50, 100, 200, 500, 1000]
    for length in nice_lengths:
        if length >= target:
            return length
    return int(target)


def _generate_html_report(
    svg_content: str,
    takeoff: QuantityTakeoff,
    config: PDFReportConfig,
    scale_string: str,
    tight_constraints: list[str],
    svg_width: float,
    svg_height: float,
) -> str:
    """Generate complete HTML report."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Quantities table rows
    qty_rows = []
    if config.include_quantities:
        qty_rows.append(f'<tr><td>Total Pad Area</td><td>{takeoff.total_pad_area_m2:.1f} m&sup2;</td></tr>')
        qty_rows.append(f'<tr><td>Road Length</td><td>{takeoff.road_length_m:.1f} m</td></tr>')
        qty_rows.append(f'<tr><td>Road Area</td><td>{takeoff.road_area_m2:.1f} m&sup2;</td></tr>')
        qty_rows.append(f'<tr><td>Fence Perimeter</td><td>{takeoff.fence_perimeter_m:.1f} m</td></tr>')
        qty_rows.append(f'<tr><td>Site Utilization</td><td>{takeoff.site_utilization_pct:.1f}%</td></tr>')
        qty_rows.append(f'<tr><td>Structure Count</td><td>{takeoff.structure_count}</td></tr>')
        if takeoff.total_pipe_proxy_length_m > 0:
            qty_rows.append(f'<tr><td>Pipe Proxy Length</td><td>{takeoff.total_pipe_proxy_length_m:.1f} m</td></tr>')

    qty_table = f'''
    <table class="quantities">
        <thead><tr><th colspan="2">Quantities</th></tr></thead>
        <tbody>{"".join(qty_rows)}</tbody>
    </table>
    ''' if qty_rows else ""

    # Constraints list
    constraints_html = ""
    if config.include_constraints and tight_constraints:
        items = "".join(f"<li>{c}</li>" for c in tight_constraints[:5])
        constraints_html = f'''
        <div class="constraints">
            <h4>Tight Constraints</h4>
            <ol>{items}</ol>
        </div>
        '''

    # Pad areas by type
    pad_rows = []
    for eq_type, area in sorted(takeoff.pad_area_by_type.items()):
        pad_rows.append(f'<tr><td>{eq_type}</td><td>{area:.1f} m&sup2;</td></tr>')

    pad_table = f'''
    <table class="pad-areas">
        <thead><tr><th colspan="2">Pad Areas by Type</th></tr></thead>
        <tbody>{"".join(pad_rows)}</tbody>
    </table>
    ''' if pad_rows else ""

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{config.title}</title>
    </head>
    <body>
        <div class="page">
            <div class="title-block">
                <div class="title">{config.title}</div>
                <div class="project">{config.project_name}</div>
                <div class="meta">
                    <span>Drawing: {config.drawing_number}</span>
                    <span>Rev: {config.revision}</span>
                    <span>Scale: {scale_string}</span>
                    <span>Date: {date_str}</span>
                </div>
            </div>

            <div class="drawing-area">
                {svg_content}
            </div>

            <div class="sidebar">
                <div class="legend">
                    <h4>Legend</h4>
                    <div class="legend-item">
                        <span class="swatch rect"></span>
                        <span>Rectangular Structure</span>
                    </div>
                    <div class="legend-item">
                        <span class="swatch circle"></span>
                        <span>Circular Structure (Tank)</span>
                    </div>
                    <div class="legend-item">
                        <span class="swatch road"></span>
                        <span>Road</span>
                    </div>
                    <div class="legend-item">
                        <span class="swatch boundary"></span>
                        <span>Site Boundary</span>
                    </div>
                </div>

                {qty_table}
                {pad_table}
                {constraints_html}
            </div>

            <div class="footer">
                <div class="company">{config.company_name}</div>
                <div class="prepared">Prepared by: {config.prepared_by}</div>
                <div class="generated">Generated by Site-Fit MCP Server</div>
            </div>
        </div>
    </body>
    </html>
    '''


def _get_report_css() -> str:
    """Get CSS styles for the report."""
    return '''
    @page {
        size: A3 landscape;
        margin: 10mm;
    }

    body {
        margin: 0;
        padding: 0;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 10pt;
    }

    .page {
        width: 100%;
        height: 100%;
        display: grid;
        grid-template-columns: 1fr 80mm;
        grid-template-rows: auto 1fr auto;
        gap: 5mm;
    }

    .title-block {
        grid-column: 1 / -1;
        border-bottom: 1px solid #333;
        padding-bottom: 3mm;
    }

    .title {
        font-size: 18pt;
        font-weight: bold;
    }

    .project {
        font-size: 14pt;
        color: #555;
    }

    .meta {
        display: flex;
        gap: 15mm;
        font-size: 9pt;
        color: #666;
        margin-top: 2mm;
    }

    .drawing-area {
        grid-column: 1;
        border: 0.5pt solid #ccc;
        overflow: hidden;
    }

    .sidebar {
        grid-column: 2;
        display: flex;
        flex-direction: column;
        gap: 5mm;
    }

    .legend {
        border: 0.5pt solid #ccc;
        padding: 3mm;
    }

    .legend h4 {
        margin: 0 0 2mm 0;
        font-size: 10pt;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 2mm;
        margin: 1mm 0;
    }

    .swatch {
        width: 8mm;
        height: 4mm;
        display: inline-block;
    }

    .swatch.rect {
        background: #7BC47F;
        border: 0.5pt solid #4A8A4E;
    }

    .swatch.circle {
        background: #4A90D9;
        border: 0.5pt solid #2E6BA8;
        border-radius: 50%;
    }

    .swatch.road {
        background: #888;
        height: 2mm;
    }

    .swatch.boundary {
        background: none;
        border: 0.5pt dashed #333;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8pt;
    }

    th, td {
        border: 0.5pt solid #ccc;
        padding: 1mm 2mm;
        text-align: left;
    }

    th {
        background: #f0f0f0;
    }

    .constraints {
        border: 0.5pt solid #ccc;
        padding: 3mm;
    }

    .constraints h4 {
        margin: 0 0 2mm 0;
        font-size: 10pt;
    }

    .constraints ol {
        margin: 0;
        padding-left: 5mm;
        font-size: 8pt;
    }

    .footer {
        grid-column: 1 / -1;
        border-top: 1px solid #333;
        padding-top: 2mm;
        display: flex;
        justify-content: space-between;
        font-size: 8pt;
        color: #666;
    }
    '''
