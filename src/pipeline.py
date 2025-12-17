"""Main site-fit generation pipeline.

Orchestrates the full pipeline from request to solutions:
1. Parse topology from SFILES2
2. Compute buildable area
3. Run CP-SAT placement solver
4. Validate with Shapely (true clearances)
5. Generate road networks
6. Filter for diversity
7. Export to GeoJSON
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from shapely.geometry import Polygon

from .models.site import SiteBoundary, Entrance, Keepout
from .models.structures import (
    StructureFootprint, PlacedStructure, RectFootprint, CircleFootprint,
    FixedPosition, ServiceEnvelopes,
)
from .models.rules import RuleSet
from .rules.loader import load_ruleset
from .models.solution import SiteFitSolution, SolutionMetrics, Placement
from .models.topology import TopologyGraph
from .geometry.polygon_ops import compute_buildable_area, polygon_from_coords
from .geometry.clearance import check_clearance_violations, check_boundary_clearance
from .geometry.containment import check_containment, ContainmentStatus
from .models.structures import AccessRequirement
from .topology.sfiles_parser import parse_sfiles_topology
from .topology.placement_hints import compute_placement_hints, PlacementHints
from .solver.cpsat_placer import PlacementSolver, PlacementSolverConfig
from .solver.diversity import SolutionFingerprint, filter_diverse_solutions
from .solver.solution_pool import SolutionPool, SolutionEntry
from .roads.network import build_road_network_for_solution
from .tools.sitefit_tools import SiteFitRequest, SiteFitResponse, SolutionSummary

logger = logging.getLogger(__name__)


# Progress callback type
ProgressCallback = Callable[[str, float], None]


async def generate_site_fits(
    request: SiteFitRequest,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[List[SiteFitSolution], Dict[str, Any]]:
    """Main pipeline: generate site fit solutions from request.

    Args:
        request: SiteFitRequest with site, topology, program, and rules
        progress_callback: Optional callback for progress updates (message, percent)

    Returns:
        Tuple of (solutions, statistics)
    """
    job_id = str(uuid.uuid4())[:8]
    stats: Dict[str, Any] = {"job_id": job_id}
    start_time = time.time()

    def report_progress(message: str, percent: float):
        if progress_callback:
            progress_callback(message, percent)
        logger.info(f"[{job_id}] {message} ({percent:.0f}%)")

    # PHASE 1: Parse inputs
    report_progress("Parsing inputs...", 5)

    # Parse site boundary
    boundary_coords = request.site.boundary
    site_boundary = polygon_from_coords([tuple(c) for c in boundary_coords])

    # Parse entrances
    entrances = []
    for ent_data in request.site.entrances:
        entrances.append(Entrance(
            id=ent_data.get("id", f"entrance_{len(entrances)}"),
            point=tuple(ent_data["point"]),
            width=ent_data.get("width", 6.0),
        ))

    # Parse keepouts
    keepouts = []
    keepout_polys = []
    for ko_data in request.site.keepouts:
        ko_coords = ko_data.get("geometry", {}).get("coordinates", [[]])[0]
        if ko_coords:
            poly = polygon_from_coords([tuple(c) for c in ko_coords])
            keepout_polys.append(poly)
            keepouts.append(Keepout(
                id=ko_data.get("id", f"keepout_{len(keepouts)}"),
                geometry={"type": "Polygon", "coordinates": [ko_coords]},
                reason=ko_data.get("reason", "unspecified"),
            ))

    # Parse existing structures (brownfield)
    existing_polys = []
    existing_buffers = []
    tie_in_points = []  # Existing structures that are utility tie-in points
    for ex_data in request.site.existing:
        # Handle both "footprint" (GeoJSON) and "geometry" (legacy) keys
        footprint_data = ex_data.get("footprint") or ex_data.get("geometry", {})
        ex_coords = footprint_data.get("coordinates", [[]])[0]
        if ex_coords:
            poly = polygon_from_coords([tuple(c) for c in ex_coords])
            existing_polys.append(poly)
            # Use clearance_required or default to 3.0m
            clearance = ex_data.get("clearance_required", 3.0)
            existing_buffers.append(clearance)
            # Track tie-in points for road routing
            if ex_data.get("is_tie_in_point", False):
                tie_in_points.append({
                    "id": ex_data.get("id", f"tiein_{len(tie_in_points)}"),
                    "point": (poly.centroid.x, poly.centroid.y),
                    "polygon": poly,
                })

    # Parse structures
    structures = []
    for s_data in request.program.structures:
        footprint_data = s_data.get("footprint", {})
        shape = footprint_data.get("shape", "rect")

        if shape == "circle":
            footprint = CircleFootprint(d=footprint_data.get("d", 10.0))
        else:
            footprint = RectFootprint(
                w=footprint_data.get("w", 10.0),
                h=footprint_data.get("h", 10.0),
            )

        # Parse access requirements if provided (full parameter support)
        access_data = s_data.get("access")
        access_req = None
        if access_data:
            access_req = AccessRequirement(
                vehicle=access_data.get("vehicle", "truck"),
                dock_edge=access_data.get("dock_edge", "any"),
                dock_length=access_data.get("dock_length", 15.0),
                dock_width=access_data.get("dock_width", 6.0),
                required=access_data.get("required", True),
                turning_radius=access_data.get("turning_radius"),
            )

        # Parse pinned placement if provided (Tier 2)
        fixed_pos = None
        fixed_pos_data = s_data.get("fixed_position")
        if fixed_pos_data:
            fixed_pos = FixedPosition(
                x=fixed_pos_data.get("x", 0.0),
                y=fixed_pos_data.get("y", 0.0),
                rotation_deg=fixed_pos_data.get("rotation_deg", 0),
            )

        # Parse service envelopes if provided (Tier 2)
        svc_envelopes = None
        svc_data = s_data.get("service_envelopes")
        if svc_data:
            svc_envelopes = ServiceEnvelopes(
                maintenance_offset=svc_data.get("maintenance_offset", 0.0),
                crane_access_edge=svc_data.get("crane_access_edge"),
                crane_strip_width=svc_data.get("crane_strip_width", 6.0),
                crane_strip_length=svc_data.get("crane_strip_length", 20.0),
                laydown_area=tuple(svc_data["laydown_area"]) if svc_data.get("laydown_area") else None,
                laydown_edge=svc_data.get("laydown_edge"),
            )

        structures.append(StructureFootprint(
            id=s_data.get("id", f"structure_{len(structures)}"),
            type=s_data.get("type", "unknown"),
            footprint=footprint,
            orientations_deg=s_data.get("orientations_deg", [0, 90, 180, 270]),
            height=s_data.get("height"),
            equipment_tag=s_data.get("equipment_tag"),
            area_number=s_data.get("area_number"),
            access=access_req,
            pinned=s_data.get("pinned", False),
            fixed_position=fixed_pos,
            allowed_zone=s_data.get("allowed_zone"),
            service_envelopes=svc_envelopes,
        ))

    stats["num_structures"] = len(structures)
    stats["num_keepouts"] = len(keepouts)
    stats["num_entrances"] = len(entrances)
    stats["num_existing"] = len(existing_polys)
    stats["num_tie_in_points"] = len(tie_in_points)
    stats["is_brownfield"] = len(existing_polys) > 0
    stats["num_pinned"] = sum(1 for s in structures if s.pinned)
    stats["num_with_service_envelopes"] = sum(1 for s in structures if s.service_envelopes)

    # PHASE 2: Parse topology
    report_progress("Parsing topology...", 10)

    hints = PlacementHints()
    if request.topology and request.topology.sfiles2:
        try:
            topology = parse_sfiles_topology(
                request.topology.sfiles2,
                node_metadata=request.topology.node_metadata,
            )
            hints = compute_placement_hints(topology)

            # Translate hints using node_map if provided
            # This links topology node IDs (e.g., "reactor-1") to structure IDs (e.g., "RX-001")
            if request.topology.node_map:
                hints = hints.translate_with_node_map(request.topology.node_map)
                stats["topology_node_map_used"] = True

            stats["topology_nodes"] = len(topology.nodes)
            stats["topology_edges"] = len(topology.edges)
        except Exception as e:
            logger.warning(f"Failed to parse topology: {e}")
            stats["topology_error"] = str(e)

    # PHASE 3: Compute buildable area
    report_progress("Computing buildable area...", 15)

    # Load rules from YAML with optional overrides
    rules = load_ruleset("default")
    if request.rules_override:
        rules = rules.merge_override(request.rules_override)

    setback = rules.setbacks.property_line_default
    buildable = compute_buildable_area(
        boundary=site_boundary,
        setback=setback,
        keepouts=keepout_polys,
        existing=existing_polys if existing_polys else None,
        existing_buffers=existing_buffers if existing_buffers else None,
    )

    if buildable.is_empty:
        logger.error("Buildable area is empty after setbacks")
        return [], {"error": "No buildable area", **stats}

    buildable_bounds = buildable.bounds
    stats["buildable_area_m2"] = buildable.area

    # PHASE 4: Run CP-SAT solver
    report_progress("Running constraint solver...", 20)

    solver_config = PlacementSolverConfig(
        max_solutions=request.generation.max_solutions * 3,  # Over-generate
        max_time_seconds=request.generation.max_time_seconds,
        seed=request.generation.seed,
    )

    solver = PlacementSolver(
        structures=structures,
        bounds=buildable_bounds,
        rules=rules,
        hints=hints,
        config=solver_config,
    )

    solver_result = solver.solve()

    stats["solver_status"] = solver_result.status
    stats["solver_time_seconds"] = solver_result.solve_time_seconds
    stats["cpsat_solutions"] = solver_result.num_solutions_found
    stats.update(solver_result.statistics)

    if not solver_result.solutions:
        logger.error(f"Solver found no solutions: {solver_result.status}")
        return [], {"error": f"No solutions found: {solver_result.status}", **stats}

    report_progress(f"Found {len(solver_result.solutions)} CP-SAT solutions", 50)

    # PHASE 5: Validate with Shapely (containment, clearances, boundary setbacks)
    report_progress("Validating containment and clearances...", 55)

    # Build structure type lookup for boundary setbacks
    structure_types = {s.id: s.type for s in structures}

    validated = []  # List of (placements, objective_value) tuples
    containment_rejects = 0
    clearance_rejects = 0
    boundary_setback_rejects = 0

    # Get per-solution objectives (may be empty if solver didn't track them)
    sol_objectives = solver_result.solution_objectives or []

    for i, placements in enumerate(solver_result.solutions):
        # Get objective for this solution (if available)
        objective = sol_objectives[i] if i < len(sol_objectives) else None
        solution_valid = True

        # 5a. Check each placement is fully inside buildable area
        for p in placements:
            poly = p.to_shapely_polygon()
            containment = check_containment(poly, buildable)
            if not containment.is_valid:
                logger.debug(
                    f"Solution {i}: {p.structure_id} fails containment "
                    f"({containment.status.value}, {containment.outside_area:.2f}m² outside)"
                )
                containment_rejects += 1
                solution_valid = False
                break

        if not solution_valid:
            continue

        # 5b. Check equipment-to-boundary setbacks
        for p in placements:
            eq_type = getattr(p, 'equipment_type', None) or structure_types.get(p.structure_id, 'default')
            required_setback = rules.get_equipment_to_boundary(eq_type)
            is_valid, actual_dist = check_boundary_clearance(p, site_boundary, required_setback)
            if not is_valid:
                logger.debug(
                    f"Solution {i}: {p.structure_id} ({eq_type}) too close to boundary "
                    f"({actual_dist:.2f}m vs required {required_setback:.2f}m)"
                )
                boundary_setback_rejects += 1
                solution_valid = False
                break

        if not solution_valid:
            continue

        # 5c. Check true pairwise clearances (especially for circles)
        violations = check_clearance_violations(placements, rules)

        if violations:
            logger.debug(f"Solution {i} has {len(violations)} clearance violations")
            clearance_rejects += 1
            continue

        validated.append((placements, objective))

        if len(validated) >= request.generation.max_solutions * 2:
            break

    stats["validated_solutions"] = len(validated)
    stats["containment_rejects"] = containment_rejects
    stats["clearance_rejects"] = clearance_rejects
    stats["boundary_setback_rejects"] = boundary_setback_rejects
    report_progress(f"{len(validated)} solutions passed validation", 60)

    if not validated:
        logger.error("No solutions passed validation")
        reject_summary = (
            f"containment={containment_rejects}, "
            f"clearance={clearance_rejects}, "
            f"boundary_setback={boundary_setback_rejects}"
        )
        return [], {"error": f"No valid solutions after validation ({reject_summary})", **stats}

    # PHASE 6: Generate road networks
    report_progress("Generating road networks...", 65)

    solutions_with_roads = []  # List of (placements, road_network, objective) tuples
    road_generation_failures = 0
    for i, (placements, objective) in enumerate(validated):
        if request.generation.require_road_access and entrances:
            road_network = build_road_network_for_solution(
                placements=placements,
                entrances=entrances,
                boundary=site_boundary,
                rules=rules,
                keepouts=keepout_polys,  # Pass keepouts for road validation
                validate_containment=True,
            )

            if road_network is None:
                logger.debug(f"Solution {i} failed road network generation")
                road_generation_failures += 1
                continue
        else:
            road_network = None

        solutions_with_roads.append((placements, road_network, objective))

        report_progress(
            f"Generated roads for {len(solutions_with_roads)} solutions",
            65 + (i / len(validated)) * 15,
        )

    stats["solutions_with_roads"] = len(solutions_with_roads)
    stats["road_generation_failures"] = road_generation_failures

    if not solutions_with_roads:
        logger.error("No solutions have valid road networks")
        return [], {"error": "No solutions with valid roads", **stats}

    # PHASE 7: Diversity filtering
    report_progress("Selecting diverse solutions...", 80)

    # Create solution pool with objective values for proper ranking
    pool = SolutionPool(max_size=len(solutions_with_roads))
    for placements, road_network, objective in solutions_with_roads:
        pool.add(placements, objective_value=objective)

    # Compute fingerprints and filter
    for entry in pool:
        entry.fingerprint = SolutionFingerprint.from_placements(entry.placements)

    diverse_entries = pool.get_diverse(
        n=request.generation.max_solutions,
        min_distance=request.generation.diversity.min_delta,
    )

    stats["diverse_solutions"] = len(diverse_entries)

    # PHASE 8: Build final solutions
    report_progress("Building final solutions...", 90)

    final_solutions = []
    for rank, entry in enumerate(diverse_entries):
        # Find matching road network
        road_network = None
        for placements, rn, _obj in solutions_with_roads:
            if placements == entry.placements:
                road_network = rn
                break

        # Compute metrics
        metrics = _compute_metrics(
            entry.placements, road_network, hints, buildable_area=buildable.area
        )

        # Create solution
        solution = SiteFitSolution(
            id=f"{job_id}-{rank}",
            job_id=job_id,
            rank=rank,
            placements=[
                Placement(
                    structure_id=p.structure_id,
                    x=p.x,
                    y=p.y,
                    rotation_deg=p.rotation_deg,
                    width=p.width,
                    height=p.height,
                    shape=p.structure.footprint.shape,  # Preserve shape for GeoJSON
                )
                for p in entry.placements
            ],
            road_network=road_network,
            metrics=metrics,
            diversity_note=_generate_diversity_note(entry, diverse_entries, rank),
        )

        # Generate GeoJSON (with hazard zones and keepouts for viewer)
        structure_types = {s.id: s.type for s in structures}
        solution.features_geojson = _generate_geojson(
            solution, site_boundary, entrances, rules, structure_types, keepouts
        )

        final_solutions.append(solution)

    stats["final_solutions"] = len(final_solutions)
    stats["total_time_seconds"] = time.time() - start_time

    report_progress("Complete!", 100)

    return final_solutions, stats


def _compute_metrics(
    placements: List[PlacedStructure],
    road_network,
    hints: PlacementHints,
    buildable_area: float = 0.0,
    edge_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> SolutionMetrics:
    """Compute solution metrics including ROM quantities.

    Args:
        placements: List of placed structures
        road_network: Road network (if computed)
        hints: PlacementHints with topology info
        buildable_area: Total buildable area in m² for site utilization
        edge_metadata: Optional edge metadata for pipe type classification
            keyed by 'from_id:to_id', e.g., {'EQ-001:TK-001': {'type': 'gravity'}}

    Returns:
        SolutionMetrics with computed values including ROM metrics
    """
    from .topology.placement_hints import compute_flow_violation_score
    from shapely.ops import unary_union
    from shapely.geometry import LineString

    edge_metadata = edge_metadata or {}

    # Build position lookup
    positions = {p.structure_id: (p.x, p.y) for p in placements}

    # Compactness (convex hull ratio)
    if placements:
        all_polys = [p.to_shapely_polygon() for p in placements]
        combined = unary_union(all_polys)
        hull = combined.convex_hull
        compactness = combined.area / hull.area if hull.area > 0 else 0.0
        total_footprint = combined.area
    else:
        compactness = 0.0
        total_footprint = 0.0

    # Site utilization (structure footprint / buildable area)
    site_utilization = total_footprint / buildable_area if buildable_area > 0 else 0.0

    # Pipe length weighted and by type
    pipe_length_weighted = 0.0
    pipe_length_by_type: Dict[str, float] = {}

    for upstream, downstream in hints.flow_precedence:
        if upstream in positions and downstream in positions:
            x1, y1 = positions[upstream]
            x2, y2 = positions[downstream]
            # Manhattan distance (pipes typically run orthogonally)
            dist = abs(x2 - x1) + abs(y2 - y1)
            pipe_length_weighted += dist

            # Classify pipe type from metadata
            edge_key = f"{upstream}:{downstream}"
            meta = edge_metadata.get(edge_key, {})
            pipe_type = _classify_pipe_type(meta.get("type", "process"))
            pipe_length_by_type[pipe_type] = pipe_length_by_type.get(pipe_type, 0.0) + dist

    # Also consider adjacency weights
    for (n1, n2), weight in hints.adjacency_weights.items():
        if n1 in positions and n2 in positions:
            x1, y1 = positions[n1]
            x2, y2 = positions[n2]
            dist = abs(x2 - x1) + abs(y2 - y1)
            pipe_length_weighted += dist * weight

    # Road metrics
    road_length = 0.0
    road_area_m2 = 0.0
    max_dead_end_length = 0.0
    intersection_count = 0

    if road_network and road_network.segments:
        road_length = road_network.total_length

        # Compute road area and intersection metrics
        road_polys = []
        endpoint_counts: Dict[Tuple[float, float], int] = {}

        for seg in road_network.segments:
            coords = seg.to_linestring_coords()
            if len(coords) >= 2:
                line = LineString(coords)
                buffered = line.buffer(seg.width / 2, cap_style=2)
                road_polys.append(buffered)

                # Count endpoints
                start = (round(coords[0][0], 1), round(coords[0][1], 1))
                end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
                endpoint_counts[start] = endpoint_counts.get(start, 0) + 1
                endpoint_counts[end] = endpoint_counts.get(end, 0) + 1

        if road_polys:
            road_combined = unary_union(road_polys)
            road_area_m2 = road_combined.area

        # Count intersections (3+ connections)
        intersection_count = sum(1 for c in endpoint_counts.values() if c >= 3)

        # Find max dead end
        for seg in road_network.segments:
            coords = seg.to_linestring_coords()
            if len(coords) >= 2:
                start = (round(coords[0][0], 1), round(coords[0][1], 1))
                end = (round(coords[-1][0], 1), round(coords[-1][1], 1))
                if endpoint_counts.get(start, 0) == 1 or endpoint_counts.get(end, 0) == 1:
                    max_dead_end_length = max(max_dead_end_length, seg.length)

    # Compute min throat width between structures
    min_throat_width = _compute_min_throat_width(placements)

    # Flow violation
    topology_penalty = compute_flow_violation_score(hints, positions)

    return SolutionMetrics(
        pipe_length_weighted=pipe_length_weighted,
        pipe_length_by_type=pipe_length_by_type,
        road_length=road_length,
        road_area_m2=road_area_m2,
        max_dead_end_length=max_dead_end_length,
        intersection_count=intersection_count,
        min_throat_width=min_throat_width if min_throat_width < float("inf") else None,
        site_utilization=site_utilization,
        compactness=compactness,
        topology_penalty=topology_penalty,
    )


def _classify_pipe_type(pipe_type_str: str) -> str:
    """Classify pipe type from metadata string."""
    pipe_type_lower = pipe_type_str.lower()
    if "gravity" in pipe_type_lower or "drain" in pipe_type_lower:
        return "gravity"
    elif "gas" in pipe_type_lower or "air" in pipe_type_lower or "biogas" in pipe_type_lower:
        return "gas"
    elif "sludge" in pipe_type_lower:
        return "sludge"
    else:
        return "pressure"


def _compute_min_throat_width(placements: List[PlacedStructure]) -> float:
    """Compute minimum throat width between adjacent structures."""
    min_width = float("inf")

    for i, p1 in enumerate(placements):
        for p2 in placements[i + 1:]:
            b1 = p1.get_bounds()
            b2 = p2.get_bounds()

            # Check X gap when Y ranges overlap
            y_overlap = not (b1[3] < b2[1] or b2[3] < b1[1])
            if y_overlap:
                x_gap = abs(b2[0] - b1[2]) if b2[0] > b1[2] else abs(b1[0] - b2[2])
                if x_gap > 0:
                    min_width = min(min_width, x_gap)

            # Check Y gap when X ranges overlap
            x_overlap = not (b1[2] < b2[0] or b2[2] < b1[0])
            if x_overlap:
                y_gap = abs(b2[1] - b1[3]) if b2[1] > b1[3] else abs(b1[1] - b2[3])
                if y_gap > 0:
                    min_width = min(min_width, y_gap)

    return min_width


def _generate_diversity_note(
    entry: SolutionEntry,
    all_entries: List[SolutionEntry],
    rank: int,
) -> Optional[str]:
    """Generate note explaining why this solution is different."""
    if rank == 0:
        return "Best overall solution by objective"

    if entry.fingerprint is None:
        return None

    # Compare to best solution
    best = all_entries[0]
    if best.fingerprint is None:
        return None

    dist = entry.fingerprint.distance_to(best.fingerprint)

    if dist > 0.5:
        return "Significantly different layout arrangement"
    elif dist > 0.3:
        return "Moderately different structure positions"
    else:
        return "Alternative placement with minor variations"


def _generate_geojson(
    solution: SiteFitSolution,
    boundary: Polygon,
    entrances: List[Entrance],
    rules: Optional[RuleSet] = None,
    structure_types: Optional[Dict[str, str]] = None,
    keepouts: Optional[List[Keepout]] = None,
) -> Dict[str, Any]:
    """Generate full GeoJSON for solution visualization.

    Args:
        solution: SiteFitSolution with placements
        boundary: Site boundary polygon
        entrances: List of entrances
        rules: Optional RuleSet for NFPA 820 zone computation
        structure_types: Optional mapping of structure_id to equipment type
        keepouts: Optional list of keepout zones to include in output
    """
    features = []

    # Site boundary
    boundary_coords = list(boundary.exterior.coords)
    features.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [boundary_coords]},
        "properties": {"kind": "boundary", "layer": "site"},
    })

    # Keepouts (added for viewer visibility)
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

    # Entrances
    for ent in entrances:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(ent.point)},
            "properties": {"kind": "entrance", "id": ent.id, "layer": "site"},
        })

    # NFPA 820 hazard zones (if rules provided and zones configured)
    if rules and rules.nfpa820_zones:
        try:
            from .hazards.nfpa820_zones import compute_hazard_zones
            hazard_zones = compute_hazard_zones(
                solution.placements, rules, structure_types
            )
            for zone in hazard_zones:
                feature = zone.to_geojson_feature()
                features.append(feature)
        except ImportError:
            logger.warning("Hazard zone module not available")

    # Structures (circles as true circular polygons, rectangles as 4-corner polygons)
    for p in solution.placements:
        features.append({
            "type": "Feature",
            "geometry": p.to_geojson_geometry(),
            "properties": {
                "kind": "structure",
                "id": p.structure_id,
                "shape": p.shape,
                "rotation": p.rotation_deg,
            },
        })

    # Roads
    if solution.road_network:
        for seg in solution.road_network.segments:
            coords = seg.to_linestring_coords()
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "kind": "road",
                    "id": seg.id,
                    "width": seg.width,
                },
            })

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "solution_id": solution.id,
            "job_id": solution.job_id,
            "rank": solution.rank,
        },
    }
