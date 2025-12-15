"""GeoJSON export utilities for site-fit solutions."""

from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon, Point, LineString, mapping

from ..models.solution import SiteFitSolution, Placement, RoadNetwork, RoadSegment
from ..models.structures import PlacedStructure
from ..models.site import SiteBoundary, Entrance, Keepout


def solution_to_geojson(
    solution: SiteFitSolution,
    boundary: Optional[Polygon] = None,
    entrances: Optional[List[Entrance]] = None,
    keepouts: Optional[List[Keepout]] = None,
    include_labels: bool = True,
) -> Dict[str, Any]:
    """Convert a complete solution to GeoJSON FeatureCollection.

    Args:
        solution: SiteFitSolution to export
        boundary: Optional site boundary polygon
        entrances: Optional list of entrances
        keepouts: Optional list of keepout zones
        include_labels: Include label points for structures

    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []

    # Add boundary if provided
    if boundary is not None:
        features.append({
            "type": "Feature",
            "geometry": mapping(boundary),
            "properties": {
                "kind": "boundary",
                "layer": "site",
            },
        })

    # Add keepouts
    if keepouts:
        for ko in keepouts:
            coords = ko.geometry.get("coordinates", [[]])
            if coords and coords[0]:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coords,
                    },
                    "properties": {
                        "kind": "keepout",
                        "id": ko.id,
                        "reason": ko.reason,
                        "layer": "site",
                    },
                })

    # Add entrances
    if entrances:
        for ent in entrances:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": list(ent.point),
                },
                "properties": {
                    "kind": "entrance",
                    "id": ent.id,
                    "width": ent.width,
                    "layer": "site",
                },
            })

    # Add structure placements
    for p in solution.placements:
        features.append(_placement_to_feature(p))

        # Add label point if requested
        if include_labels:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [p.x, p.y],
                },
                "properties": {
                    "kind": "label",
                    "id": p.structure_id,
                    "text": p.structure_id,
                    "layer": "labels",
                },
            })

    # Add road network
    if solution.road_network:
        features.extend(road_network_to_features(solution.road_network))

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "solution_id": solution.id,
            "job_id": solution.job_id,
            "rank": solution.rank,
            "metrics": solution.metrics.model_dump() if solution.metrics else {},
        },
    }


def placements_to_geojson(
    placements: List[PlacedStructure],
) -> Dict[str, Any]:
    """Convert placements to GeoJSON FeatureCollection.

    Args:
        placements: List of PlacedStructure objects

    Returns:
        GeoJSON FeatureCollection dict
    """
    features = []

    for p in placements:
        # Get polygon from placed structure
        poly = p.to_shapely_polygon()
        features.append({
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "kind": "structure",
                "id": p.structure_id,
                "type": p.equipment_type,
                "rotation": p.rotation_deg,
                "center": [p.x, p.y],
                "layer": "structures",
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def road_network_to_geojson(
    network: RoadNetwork,
) -> Dict[str, Any]:
    """Convert road network to GeoJSON FeatureCollection.

    Args:
        network: RoadNetwork object

    Returns:
        GeoJSON FeatureCollection dict
    """
    features = road_network_to_features(network)

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "total_length": network.total_length,
            "entrances_connected": network.entrances_connected,
            "structures_accessible": network.structures_accessible,
        },
    }


def road_network_to_features(network: RoadNetwork) -> List[Dict[str, Any]]:
    """Convert road network to list of GeoJSON features.

    Args:
        network: RoadNetwork object

    Returns:
        List of GeoJSON Feature dicts
    """
    features = []

    for segment in network.segments:
        coords = segment.to_linestring_coords()

        # Road centerline
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [list(c) for c in coords],
            },
            "properties": {
                "kind": "road",
                "id": segment.id,
                "width": segment.width,
                "length": segment.length,
                "connects_to": segment.connects_to,
                "layer": "roads",
            },
        })

        # Optional: road polygon (buffered centerline)
        # This could be added for more accurate visualization

    return features


def _placement_to_feature(placement: Placement) -> Dict[str, Any]:
    """Convert a Placement to GeoJSON Feature.

    Delegates to the Placement's to_geojson_geometry() method which
    handles circles, rectangles, and rotation correctly.

    Args:
        placement: Placement object

    Returns:
        GeoJSON Feature dict
    """
    return {
        "type": "Feature",
        "geometry": placement.to_geojson_geometry(),
        "properties": {
            "kind": "structure",
            "id": placement.structure_id,
            "x": placement.x,
            "y": placement.y,
            "rotation": placement.rotation_deg,
            "width": placement.width,
            "height": placement.height,
            "shape": placement.shape,
            "layer": "structures",
        },
    }


def merge_geojson_collections(
    *collections: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge multiple GeoJSON FeatureCollections.

    Args:
        *collections: GeoJSON FeatureCollection dicts

    Returns:
        Merged GeoJSON FeatureCollection
    """
    all_features = []

    for coll in collections:
        if coll and "features" in coll:
            all_features.extend(coll["features"])

    return {
        "type": "FeatureCollection",
        "features": all_features,
    }


def filter_geojson_by_layer(
    geojson: Dict[str, Any],
    layers: List[str],
) -> Dict[str, Any]:
    """Filter GeoJSON features by layer property.

    Args:
        geojson: GeoJSON FeatureCollection
        layers: List of layer names to include

    Returns:
        Filtered GeoJSON FeatureCollection
    """
    features = geojson.get("features", [])
    filtered = [
        f for f in features
        if f.get("properties", {}).get("layer") in layers
    ]

    return {
        "type": "FeatureCollection",
        "features": filtered,
        "properties": geojson.get("properties", {}),
    }
