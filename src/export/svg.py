"""SVG export for site layout preview generation.

Generates SVG diagrams from site layout solutions for quick preview
without requiring a full browser-based viewer.
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from shapely.geometry import Polygon, LineString, Point

from ..models.solution import SiteFitSolution
from ..models.structures import PlacedStructure

logger = logging.getLogger(__name__)

# Default SVG styling
DEFAULT_STYLES = {
    "boundary": {
        "fill": "#e8f4fc",
        "stroke": "#2266cc",
        "stroke_width": 2,
        "fill_opacity": 0.3,
    },
    "keepout": {
        "fill": "#ffcccc",
        "stroke": "#cc0000",
        "stroke_width": 1,
        "fill_opacity": 0.5,
        "stroke_dasharray": "5,3",
    },
    "structure_rect": {
        "fill": "#ff9933",
        "stroke": "#cc6600",
        "stroke_width": 1.5,
        "fill_opacity": 0.8,
    },
    "structure_circle": {
        "fill": "#66b3ff",
        "stroke": "#0066cc",
        "stroke_width": 1.5,
        "fill_opacity": 0.8,
    },
    "road": {
        "fill": "#999999",
        "stroke": "#666666",
        "stroke_width": 0.5,
        "fill_opacity": 0.7,
    },
    "dock": {
        "fill": "#cccccc",
        "stroke": "#666666",
        "stroke_width": 0.5,
        "fill_opacity": 0.5,
        "stroke_dasharray": "3,2",
    },
    "entrance": {
        "fill": "#00cc00",
        "stroke": "#009900",
        "stroke_width": 1,
    },
    "label": {
        "font_family": "Arial, sans-serif",
        "font_size": 8,
        "fill": "#333333",
        "text_anchor": "middle",
    },
}


@dataclass
class SVGViewBox:
    """SVG viewport definition."""
    min_x: float
    min_y: float
    width: float
    height: float
    padding: float = 10.0

    @classmethod
    def from_bounds(
        cls,
        bounds: Tuple[float, float, float, float],
        padding: float = 10.0,
    ) -> "SVGViewBox":
        """Create viewbox from (minx, miny, maxx, maxy) bounds."""
        minx, miny, maxx, maxy = bounds
        return cls(
            min_x=minx - padding,
            min_y=miny - padding,
            width=maxx - minx + 2 * padding,
            height=maxy - miny + 2 * padding,
            padding=padding,
        )

    def to_attr(self) -> str:
        """Return viewBox attribute value."""
        return f"{self.min_x} {self.min_y} {self.width} {self.height}"


def export_solution_to_svg(
    solution: SiteFitSolution,
    boundary: Polygon,
    keepouts: Optional[List[Polygon]] = None,
    width: int = 800,
    height: int = 600,
    show_labels: bool = True,
    show_roads: bool = True,
    show_docks: bool = True,
    styles: Optional[dict] = None,
) -> str:
    """Export a site layout solution to SVG format.

    Args:
        solution: SiteFitSolution with placements and road network
        boundary: Site boundary polygon
        keepouts: Optional list of keepout zone polygons
        width: SVG width in pixels
        height: SVG height in pixels
        show_labels: Whether to show structure labels
        show_roads: Whether to show road network
        show_docks: Whether to show dock zones
        styles: Optional style overrides

    Returns:
        SVG string
    """
    # Merge styles
    merged_styles = {**DEFAULT_STYLES}
    if styles:
        for key, value in styles.items():
            if key in merged_styles:
                merged_styles[key] = {**merged_styles[key], **value}
            else:
                merged_styles[key] = value

    # Calculate viewbox from boundary
    viewbox = SVGViewBox.from_bounds(boundary.bounds, padding=15.0)

    # Start SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" ',
        f'width="{width}" height="{height}" ',
        f'viewBox="{viewbox.to_attr()}" ',
        f'preserveAspectRatio="xMidYMid meet">',
        "",
        "<!-- Site-Fit Generated SVG -->",
        f"<!-- Solution ID: {solution.id} -->",
        "",
    ]

    # Add defs for patterns/gradients
    svg_parts.append(_generate_defs())

    # Flip Y-axis for conventional coordinates (Y increases upward)
    svg_parts.append(f'<g transform="scale(1, -1) translate(0, {-viewbox.min_y * 2 - viewbox.height})">')

    # Draw boundary
    svg_parts.append(_polygon_to_svg(boundary, "boundary", merged_styles["boundary"]))

    # Draw keepouts
    if keepouts:
        for i, keepout in enumerate(keepouts):
            svg_parts.append(_polygon_to_svg(
                keepout, f"keepout_{i}", merged_styles["keepout"]
            ))

    # Draw road network
    if show_roads and solution.road_network:
        for segment in solution.road_network.segments:
            if hasattr(segment, "polygon") and segment.polygon:
                svg_parts.append(_polygon_to_svg(
                    segment.polygon, f"road_{segment.id}", merged_styles["road"]
                ))

    # Draw dock zones
    if show_docks:
        for placement in solution.placements:
            if hasattr(placement, "dock_zone") and placement.dock_zone:
                svg_parts.append(_polygon_to_svg(
                    placement.dock_zone, f"dock_{placement.structure_id}",
                    merged_styles["dock"]
                ))

    # Draw structures
    for placement in solution.placements:
        footprint = placement.footprint_polygon
        is_circle = placement.structure.footprint.shape == "circle"
        style_key = "structure_circle" if is_circle else "structure_rect"
        svg_parts.append(_polygon_to_svg(
            footprint, placement.structure_id, merged_styles[style_key]
        ))

    # Close transform group
    svg_parts.append("</g>")

    # Draw labels (separate group, not flipped)
    if show_labels:
        svg_parts.append('<g class="labels">')
        for placement in solution.placements:
            # Transform coordinates for label position
            cx, cy = placement.x, placement.y
            # Flip Y for label
            label_y = viewbox.min_y + viewbox.height - (cy - viewbox.min_y)
            svg_parts.append(_label_to_svg(
                placement.structure_id, cx, label_y, merged_styles["label"]
            ))
        svg_parts.append("</g>")

    # Close SVG
    svg_parts.append("</svg>")

    return "\n".join(svg_parts)


def _generate_defs() -> str:
    """Generate SVG defs section for patterns and gradients."""
    return """<defs>
    <pattern id="keepout-pattern" patternUnits="userSpaceOnUse" width="10" height="10">
        <line x1="0" y1="0" x2="10" y2="10" stroke="#cc0000" stroke-width="0.5"/>
    </pattern>
</defs>"""


def _polygon_to_svg(polygon: Polygon, id: str, style: dict) -> str:
    """Convert Shapely polygon to SVG path element."""
    if polygon.is_empty:
        return ""

    # Get exterior coordinates
    coords = list(polygon.exterior.coords)
    if not coords:
        return ""

    # Build path data
    path_data = f"M {coords[0][0]},{coords[0][1]}"
    for x, y in coords[1:]:
        path_data += f" L {x},{y}"
    path_data += " Z"

    # Build style string
    style_parts = []
    if "fill" in style:
        style_parts.append(f'fill="{style["fill"]}"')
    if "fill_opacity" in style:
        style_parts.append(f'fill-opacity="{style["fill_opacity"]}"')
    if "stroke" in style:
        style_parts.append(f'stroke="{style["stroke"]}"')
    if "stroke_width" in style:
        style_parts.append(f'stroke-width="{style["stroke_width"]}"')
    if "stroke_dasharray" in style:
        style_parts.append(f'stroke-dasharray="{style["stroke_dasharray"]}"')

    style_str = " ".join(style_parts)

    return f'<path id="{id}" d="{path_data}" {style_str}/>'


def _label_to_svg(text: str, x: float, y: float, style: dict) -> str:
    """Create SVG text element for label."""
    font_family = style.get("font_family", "Arial")
    font_size = style.get("font_size", 8)
    fill = style.get("fill", "#333333")
    anchor = style.get("text_anchor", "middle")

    return (
        f'<text x="{x}" y="{y}" font-family="{font_family}" '
        f'font-size="{font_size}" fill="{fill}" text-anchor="{anchor}">'
        f'{text}</text>'
    )


def export_comparison_svg(
    solutions: List[SiteFitSolution],
    boundary: Polygon,
    cols: int = 3,
    cell_width: int = 300,
    cell_height: int = 250,
) -> str:
    """Export multiple solutions as a comparison grid SVG.

    Args:
        solutions: List of solutions to compare
        boundary: Site boundary polygon
        cols: Number of columns in grid
        cell_width: Width of each cell in pixels
        cell_height: Height of each cell in pixels

    Returns:
        SVG string with grid layout
    """
    num_solutions = len(solutions)
    rows = (num_solutions + cols - 1) // cols

    total_width = cols * cell_width
    total_height = rows * cell_height

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" ',
        f'width="{total_width}" height="{total_height}">',
        "",
        "<!-- Site-Fit Solution Comparison Grid -->",
        "",
    ]

    for i, solution in enumerate(solutions):
        row = i // cols
        col = i % cols
        x_offset = col * cell_width
        y_offset = row * cell_height

        # Create sub-SVG for each solution
        sub_svg = export_solution_to_svg(
            solution=solution,
            boundary=boundary,
            width=cell_width - 20,
            height=cell_height - 40,
            show_labels=True,
            show_roads=True,
            show_docks=False,  # Simpler view for comparison
        )

        # Embed as nested SVG
        svg_parts.append(f'<g transform="translate({x_offset + 10}, {y_offset + 30})">')
        svg_parts.append(f'<text x="{(cell_width - 20) / 2}" y="-15" '
                        f'text-anchor="middle" font-family="Arial" font-size="12" '
                        f'font-weight="bold">Solution {i + 1} (Rank {solution.rank})</text>')
        # Strip outer svg tags from sub_svg
        inner = sub_svg.split(">", 1)[1].rsplit("</svg>", 1)[0]
        svg_parts.append(f'<svg width="{cell_width - 20}" height="{cell_height - 40}">')
        svg_parts.append(inner)
        svg_parts.append("</svg>")
        svg_parts.append("</g>")

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)
