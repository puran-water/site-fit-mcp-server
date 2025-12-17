"""DXF export utilities for site-fit solutions.

Requires the optional 'dxf' extra: pip install site-fit-mcp-server[dxf]
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

from shapely.geometry import Polygon, Point, LineString

if TYPE_CHECKING:
    import ezdxf
    from ezdxf.document import Drawing
    from ..models.solution import SiteFitSolution
    from ..models.site import Keepout, Entrance


logger = logging.getLogger(__name__)


def solution_to_dxf(
    solution: "SiteFitSolution",
    boundary: Optional[Polygon] = None,
    entrances: Optional[List["Entrance"]] = None,
    keepouts: Optional[List["Keepout"]] = None,
    units: str = "m",
) -> "Drawing":
    """Export a site-fit solution to DXF format for CAD integration.

    Creates a DXF document with layers:
    - BOUNDARY: Site boundary polygon
    - KEEPOUTS: Keepout/exclusion zones (hatched)
    - ENTRANCES: Entrance points
    - STRUCTURES: Equipment footprints with annotations
    - ROADS: Road centerlines and corridors
    - HAZARD_ZONES: NFPA 820 hazard zones (if present)

    Args:
        solution: SiteFitSolution to export
        boundary: Optional site boundary polygon
        entrances: Optional list of entrances
        keepouts: Optional list of keepout zones
        units: Unit system ('m' for meters, 'ft' for feet)

    Returns:
        ezdxf.Drawing object ready to save

    Raises:
        ImportError: If ezdxf is not installed

    Example:
        from src.export.dxf import solution_to_dxf

        doc = solution_to_dxf(solution, boundary, entrances, keepouts)
        doc.saveas("site_layout.dxf")
    """
    try:
        import ezdxf
        from ezdxf import units as dxf_units
        from ezdxf.enums import TextEntityAlignment
    except ImportError:
        raise ImportError(
            "ezdxf is required for DXF export. "
            "Install with: pip install site-fit-mcp-server[dxf]"
        )

    # Create new DXF document (R2010 format for wide compatibility)
    doc = ezdxf.new("R2010")

    # Set units
    if units == "m":
        doc.units = dxf_units.M
    elif units == "ft":
        doc.units = dxf_units.FT
    else:
        doc.units = dxf_units.M

    msp = doc.modelspace()

    # Create layers with appropriate colors
    _create_layers(doc)

    # Add boundary
    if boundary is not None:
        _add_boundary(msp, boundary)

    # Add keepouts
    if keepouts:
        _add_keepouts(msp, keepouts)

    # Add entrances
    if entrances:
        _add_entrances(msp, entrances)

    # Add structures
    for placement in solution.placements:
        _add_structure(msp, placement, doc)

    # Add roads
    if solution.road_network:
        _add_road_network(msp, solution.road_network)

    return doc


def _create_layers(doc: "Drawing") -> None:
    """Create standard layers for site layout."""
    import ezdxf

    layers = [
        ("BOUNDARY", 7),          # White
        ("BUILDLIMIT", 3),        # Green - buildable area after setbacks
        ("KEEPOUTS", 1),          # Red
        ("ENTRANCES", 3),         # Green
        ("STRUCTURES", 5),        # Blue
        ("STRUCTURE_LABELS", 7),  # White
        ("ROADS", 4),             # Cyan
        ("HAZARD_ZONES", 1),      # Red (for hazard zones)
        ("HAZARD_DIV1", 30),      # Orange (Class I Div 1)
        ("HAZARD_DIV2", 40),      # Light red (Class I Div 2)
        ("CENTERLINES", 2),       # Yellow
    ]

    for name, color in layers:
        if name not in doc.layers:
            doc.layers.add(name, color=color)


def _add_boundary(msp, boundary: Polygon) -> None:
    """Add site boundary as a polyline."""
    coords = list(boundary.exterior.coords)
    # Convert to list of (x, y) tuples
    points = [(c[0], c[1]) for c in coords]
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "BOUNDARY"})


def _add_keepouts(msp, keepouts: List["Keepout"]) -> None:
    """Add keepout zones as hatched polygons."""
    for ko in keepouts:
        coords = ko.geometry.get("coordinates", [[]])
        if coords and coords[0]:
            points = [(c[0], c[1]) for c in coords[0]]
            # Add outline
            poly = msp.add_lwpolyline(
                points, close=True,
                dxfattribs={"layer": "KEEPOUTS"}
            )
            # Add hatch for visibility
            try:
                hatch = msp.add_hatch(
                    color=1,  # Red
                    dxfattribs={"layer": "KEEPOUTS"}
                )
                hatch.paths.add_polyline_path(points, is_closed=True)
                hatch.set_pattern_fill("ANSI31", scale=0.5)
            except Exception as e:
                logger.debug(f"Could not add hatch for keepout {ko.id}: {e}")


def _add_entrances(msp, entrances: List["Entrance"]) -> None:
    """Add entrances as points with markers."""
    for ent in entrances:
        x, y = ent.point[0], ent.point[1]
        # Add circle marker
        msp.add_circle(
            center=(x, y),
            radius=2.0,
            dxfattribs={"layer": "ENTRANCES"}
        )
        # Add label
        msp.add_text(
            ent.id,
            height=1.5,
            dxfattribs={
                "layer": "ENTRANCES",
                "insert": (x + 3, y),
            }
        )


def _add_structure(msp, placement, doc: "Drawing") -> None:
    """Add a structure placement with annotation."""
    from ezdxf.enums import TextEntityAlignment

    # Get structure polygon
    poly = placement.to_shapely_polygon()

    if placement.shape == "circle":
        # Add as circle
        msp.add_circle(
            center=(placement.x, placement.y),
            radius=placement.width / 2,
            dxfattribs={"layer": "STRUCTURES"}
        )
    else:
        # Add as polyline (rectangle or rotated rectangle)
        coords = list(poly.exterior.coords)
        points = [(c[0], c[1]) for c in coords]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "STRUCTURES"})

    # Add label at center
    msp.add_text(
        placement.structure_id,
        height=1.0,
        dxfattribs={
            "layer": "STRUCTURE_LABELS",
            "insert": (placement.x, placement.y),
        }
    )

    # Add center point marker
    msp.add_point(
        (placement.x, placement.y),
        dxfattribs={"layer": "CENTERLINES"}
    )


def _add_road_network(msp, road_network) -> None:
    """Add road network as polylines."""
    for segment in road_network.segments:
        coords = segment.to_linestring_coords()
        if len(coords) >= 2:
            points = [(c[0], c[1]) for c in coords]
            # Centerline
            msp.add_lwpolyline(
                points,
                dxfattribs={"layer": "ROADS"}
            )


def _add_buildable_area(msp, buildable: Polygon) -> None:
    """Add buildable area (after setbacks) as a dashed polyline."""
    coords = list(buildable.exterior.coords)
    points = [(c[0], c[1]) for c in coords]
    msp.add_lwpolyline(
        points, close=True,
        dxfattribs={
            "layer": "BUILDLIMIT",
            "linetype": "DASHED",
        }
    )


def _add_hazard_zones(msp, hazard_zones: List[Any]) -> None:
    """Add NFPA 820 hazard zones as hatched polygons."""
    for zone in hazard_zones:
        # Get zone type and polygon
        zone_type = getattr(zone, 'zone_type', None)
        polygon = getattr(zone, 'polygon', None)

        if polygon is None:
            continue

        # Determine layer based on zone type
        if zone_type is not None:
            zone_type_str = zone_type.value if hasattr(zone_type, 'value') else str(zone_type)
            if 'div_1' in zone_type_str or 'div1' in zone_type_str.lower():
                layer = "HAZARD_DIV1"
                hatch_color = 30  # Orange
            else:
                layer = "HAZARD_DIV2"
                hatch_color = 40  # Light red
        else:
            layer = "HAZARD_ZONES"
            hatch_color = 1  # Red

        # Add polygon boundary
        coords = list(polygon.exterior.coords)
        points = [(c[0], c[1]) for c in coords]
        msp.add_lwpolyline(
            points, close=True,
            dxfattribs={"layer": layer}
        )

        # Add hatch for visibility
        try:
            hatch = msp.add_hatch(
                color=hatch_color,
                dxfattribs={"layer": layer}
            )
            hatch.paths.add_polyline_path(points, is_closed=True)
            hatch.set_pattern_fill("ANSI32", scale=1.0)  # Different pattern from keepouts
        except Exception as e:
            logger.debug(f"Could not add hatch for hazard zone: {e}")


def save_solution_to_dxf(
    solution: "SiteFitSolution",
    filepath: str,
    boundary: Optional[Polygon] = None,
    buildable_polygon: Optional[Polygon] = None,
    entrances: Optional[List["Entrance"]] = None,
    keepouts: Optional[List["Keepout"]] = None,
    hazard_zones: Optional[List[Any]] = None,
    structure_types: Optional[Dict[str, str]] = None,
    units: str = "m",
) -> str:
    """Export solution to DXF file with all engineering layers.

    Args:
        solution: SiteFitSolution to export
        filepath: Output file path (should end in .dxf)
        boundary: Optional site boundary
        buildable_polygon: Optional buildable area (after setbacks)
        entrances: Optional entrances
        keepouts: Optional keepouts
        hazard_zones: Optional NFPA 820 hazard zones
        structure_types: Optional mapping of structure_id to type
        units: Unit system

    Returns:
        The filepath where the file was saved

    Example:
        save_solution_to_dxf(solution, "output/layout.dxf", boundary, buildable)
    """
    try:
        import ezdxf
        from ezdxf import units as dxf_units
    except ImportError:
        raise ImportError(
            "ezdxf is required for DXF export. "
            "Install with: pip install site-fit-mcp-server[dxf]"
        )

    # Create new DXF document (R2010 format for wide compatibility)
    doc = ezdxf.new("R2010")

    # Set units
    if units == "m":
        doc.units = dxf_units.M
    elif units == "ft":
        doc.units = dxf_units.FT
    else:
        doc.units = dxf_units.M

    msp = doc.modelspace()

    # Create layers with appropriate colors
    _create_layers(doc)

    # Add boundary
    if boundary is not None:
        _add_boundary(msp, boundary)

    # Add buildable limit
    if buildable_polygon is not None:
        _add_buildable_area(msp, buildable_polygon)

    # Add keepouts
    if keepouts:
        _add_keepouts(msp, keepouts)

    # Add hazard zones
    if hazard_zones:
        _add_hazard_zones(msp, hazard_zones)

    # Add entrances
    if entrances:
        _add_entrances(msp, entrances)

    # Add structures
    for placement in solution.placements:
        _add_structure(msp, placement, doc)

    # Add roads
    if solution.road_network:
        _add_road_network(msp, solution.road_network)

    doc.saveas(filepath)
    return filepath
