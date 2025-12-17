"""GIS file loading for site-fit.

Supports loading site boundaries, keepouts, and entrances from various GIS formats:
- Shapefile (.shp)
- GeoJSON (.geojson, .json)
- GeoPackage (.gpkg)
- KML (.kml)
- File Geodatabase (.gdb)

Uses Fiona/GDAL for file reading and PyProj for CRS transformations.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import transform

logger = logging.getLogger(__name__)


@dataclass
class GISLoadResult:
    """Result from loading a GIS file."""

    boundary: list[list[float]] | None = None
    boundary_area: float = 0.0
    keepouts: list[dict[str, Any]] = field(default_factory=list)
    entrances: list[dict[str, Any]] = field(default_factory=list)
    source_crs: str | None = None
    warnings: list[str] = field(default_factory=list)
    layers_found: list[str] = field(default_factory=list)


def load_site_from_file(
    file_path: str | Path,
    boundary_layer: str | None = None,
    keepout_layers: list[str] | None = None,
    entrance_layer: str | None = None,
    target_crs: str | None = None,
    auto_detect: bool = True,
) -> GISLoadResult:
    """Load site definition from a GIS file.

    Args:
        file_path: Path to the GIS file (Shapefile, GeoJSON, GeoPackage, etc.)
        boundary_layer: Layer name for site boundary (auto-detect if None)
        keepout_layers: Layer names for keepout zones (auto-detect if None)
        entrance_layer: Layer name for entrance points (auto-detect if None)
        target_crs: Target CRS for output coordinates (None = keep original)
        auto_detect: Auto-detect layers based on naming conventions

    Returns:
        GISLoadResult with boundary, keepouts, entrances, and metadata

    Raises:
        ImportError: If fiona is not installed
        FileNotFoundError: If file doesn't exist
    """
    try:
        import fiona
    except ImportError:
        raise ImportError(
            "fiona is required for GIS file loading. "
            "Install with: pip install 'site-fit-mcp[gis]' or pip install fiona"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GIS file not found: {file_path}")

    result = GISLoadResult()

    # List available layers
    try:
        layers = fiona.listlayers(str(file_path))
        result.layers_found = layers
        logger.info(f"Found {len(layers)} layer(s) in {file_path.name}: {layers}")
    except Exception as e:
        # Single-layer file (e.g., plain GeoJSON or Shapefile)
        layers = [None]
        logger.debug(f"Single-layer file: {e}")

    # Get transformer if target CRS specified
    transformer = None
    if target_crs:
        try:
            from pyproj import Transformer
            # Will be set per-layer based on source CRS
        except ImportError:
            result.warnings.append(
                "pyproj not installed - skipping CRS transformation"
            )
            target_crs = None

    # Auto-detect layers if requested
    if auto_detect and layers[0] is not None:
        if boundary_layer is None:
            boundary_layer = detect_boundary_layer(layers)
        if keepout_layers is None:
            keepout_layers = detect_keepout_layers(layers)
        if entrance_layer is None:
            entrance_layer = detect_entrance_layer(layers)

    # Load boundary
    boundary_layer_to_load = boundary_layer or (layers[0] if len(layers) == 1 else None)
    if boundary_layer_to_load is not None or layers[0] is None:
        try:
            with fiona.open(str(file_path), layer=boundary_layer_to_load) as src:
                result.source_crs = src.crs.to_string() if src.crs else None

                # Create transformer if needed
                if target_crs and src.crs:
                    try:
                        from pyproj import Transformer
                        transformer = Transformer.from_crs(
                            src.crs.to_string(), target_crs, always_xy=True
                        )
                    except Exception as e:
                        result.warnings.append(f"CRS transformation failed: {e}")

                # Find the largest polygon as boundary
                largest_poly = None
                largest_area = 0.0

                for feature in src:
                    geom = shape(feature["geometry"])

                    # Transform if needed
                    if transformer:
                        geom = transform(transformer.transform, geom)

                    # Handle different geometry types
                    if isinstance(geom, (Polygon, MultiPolygon)):
                        if isinstance(geom, MultiPolygon):
                            # Use the largest polygon from multipolygon
                            geom = max(geom.geoms, key=lambda p: p.area)

                        if geom.area > largest_area:
                            largest_area = geom.area
                            largest_poly = geom

                if largest_poly:
                    result.boundary = [list(coord) for coord in largest_poly.exterior.coords]
                    result.boundary_area = largest_area
                    logger.info(f"Loaded boundary: {largest_area:.2f} sq units")

        except Exception as e:
            result.warnings.append(f"Failed to load boundary: {e}")
            logger.error(f"Boundary load error: {e}")

    # Load keepouts
    if keepout_layers:
        for ko_layer in keepout_layers:
            try:
                with fiona.open(str(file_path), layer=ko_layer) as src:
                    # Reset transformer for each layer to avoid applying wrong CRS
                    transformer = None
                    if target_crs and src.crs:
                        try:
                            from pyproj import Transformer
                            transformer = Transformer.from_crs(
                                src.crs.to_string(), target_crs, always_xy=True
                            )
                        except Exception:
                            pass  # transformer stays None

                    for i, feature in enumerate(src):
                        geom = shape(feature["geometry"])

                        if transformer:
                            geom = transform(transformer.transform, geom)

                        if isinstance(geom, (Polygon, MultiPolygon)):
                            if isinstance(geom, MultiPolygon):
                                for j, poly in enumerate(geom.geoms):
                                    result.keepouts.append({
                                        "id": f"{ko_layer}_{i}_{j}",
                                        "geometry": {
                                            "type": "Polygon",
                                            "coordinates": [list(poly.exterior.coords)],
                                        },
                                        "reason": _get_feature_reason(feature, ko_layer),
                                    })
                            else:
                                result.keepouts.append({
                                    "id": f"{ko_layer}_{i}",
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [list(geom.exterior.coords)],
                                    },
                                    "reason": _get_feature_reason(feature, ko_layer),
                                })

                logger.info(f"Loaded {len(result.keepouts)} keepouts from {ko_layer}")

            except Exception as e:
                result.warnings.append(f"Failed to load keepouts from {ko_layer}: {e}")

    # Load entrances
    if entrance_layer:
        try:
            with fiona.open(str(file_path), layer=entrance_layer) as src:
                # Reset transformer for entrance layer to avoid applying wrong CRS
                transformer = None
                if target_crs and src.crs:
                    try:
                        from pyproj import Transformer
                        transformer = Transformer.from_crs(
                            src.crs.to_string(), target_crs, always_xy=True
                        )
                    except Exception:
                        pass  # transformer stays None

                for i, feature in enumerate(src):
                    geom = shape(feature["geometry"])

                    if transformer:
                        geom = transform(transformer.transform, geom)

                    if isinstance(geom, Point):
                        props = feature.get("properties", {}) or {}
                        result.entrances.append({
                            "id": props.get("id", f"entrance_{i}"),
                            "point": [geom.x, geom.y],
                            "width": props.get("width", 6.0),
                        })

            logger.info(f"Loaded {len(result.entrances)} entrances from {entrance_layer}")

        except Exception as e:
            result.warnings.append(f"Failed to load entrances from {entrance_layer}: {e}")

    return result


def detect_boundary_layer(layers: list[str]) -> str | None:
    """Detect the boundary layer from layer names.

    Looks for layers with names suggesting they contain site boundaries.
    """
    boundary_keywords = [
        "boundary", "parcel", "lot", "property", "site",
        "extent", "outline", "perimeter", "bounds",
    ]

    for layer in layers:
        layer_lower = layer.lower()
        for keyword in boundary_keywords:
            if keyword in layer_lower:
                logger.info(f"Auto-detected boundary layer: {layer}")
                return layer

    # Return first layer if no match and only one layer
    if len(layers) == 1:
        return layers[0]

    return None


def detect_keepout_layers(layers: list[str]) -> list[str]:
    """Detect keepout layers from layer names.

    Looks for layers with names suggesting they contain restricted areas.
    """
    keepout_keywords = [
        "keepout", "exclusion", "setback", "easement", "wetland",
        "flood", "buffer", "restricted", "no-build", "nobuild",
        "conservation", "utility", "drainage",
    ]

    matches = []
    for layer in layers:
        layer_lower = layer.lower()
        for keyword in keepout_keywords:
            if keyword in layer_lower:
                matches.append(layer)
                logger.info(f"Auto-detected keepout layer: {layer}")
                break

    return matches


def detect_entrance_layer(layers: list[str]) -> str | None:
    """Detect the entrance layer from layer names.

    Looks for layers with names suggesting they contain access points.
    """
    entrance_keywords = [
        "entrance", "entry", "access", "gate", "driveway",
        "ingress", "egress", "opening",
    ]

    for layer in layers:
        layer_lower = layer.lower()
        for keyword in entrance_keywords:
            if keyword in layer_lower:
                logger.info(f"Auto-detected entrance layer: {layer}")
                return layer

    return None


def _get_feature_reason(feature: dict[str, Any], layer_name: str) -> str:
    """Extract a reason/description from feature properties."""
    props = feature.get("properties", {}) or {}

    # Try common attribute names
    for key in ["reason", "type", "description", "name", "class", "category"]:
        if key in props and props[key]:
            return str(props[key])

    # Fall back to layer name
    return layer_name


def list_gis_layers(file_path: str | Path) -> list[dict[str, Any]]:
    """List all layers in a GIS file with basic metadata.

    Args:
        file_path: Path to the GIS file

    Returns:
        List of dicts with layer name, geometry type, and feature count
    """
    try:
        import fiona
    except ImportError:
        raise ImportError(
            "fiona is required for GIS file loading. "
            "Install with: pip install 'site-fit-mcp[gis]' or pip install fiona"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GIS file not found: {file_path}")

    results = []

    try:
        layers = fiona.listlayers(str(file_path))
    except Exception:
        layers = [None]

    for layer in layers:
        try:
            with fiona.open(str(file_path), layer=layer) as src:
                results.append({
                    "name": layer or file_path.stem,
                    "geometry_type": src.schema.get("geometry", "Unknown"),
                    "feature_count": len(src),
                    "crs": src.crs.to_string() if src.crs else None,
                    "properties": list(src.schema.get("properties", {}).keys()),
                })
        except Exception as e:
            results.append({
                "name": layer or file_path.stem,
                "error": str(e),
            })

    return results
