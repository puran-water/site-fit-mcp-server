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
from .models.structures import StructureFootprint, PlacedStructure, RectFootprint, CircleFootprint
from .models.rules import RuleSet
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

        structures.append(StructureFootprint(
            id=s_data.get("id", f"structure_{len(structures)}"),
            type=s_data.get("type", "unknown"),
            footprint=footprint,
            orientations_deg=s_data.get("orientations_deg", [0, 90, 180, 270]),
            height=s_data.get("height"),
            equipment_tag=s_data.get("equipment_tag"),
            area_number=s_data.get("area_number"),
            access=access_req,
        ))

    stats["num_structures"] = len(structures)
    stats["num_keepouts"] = len(keepouts)
    stats["num_entrances"] = len(entrances)

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
            stats["topology_nodes"] = len(topology.nodes)
            stats["topology_edges"] = len(topology.edges)
        except Exception as e:
            logger.warning(f"Failed to parse topology: {e}")
            stats["topology_error"] = str(e)

    # PHASE 3: Compute buildable area
    report_progress("Computing buildable area...", 15)

    # Load rules with overrides
    rules = RuleSet()
    if request.rules_override:
        rules = rules.merge_override(request.rules_override)

    setback = rules.setbacks.property_line_default
    buildable = compute_buildable_area(
        boundary=site_boundary,
        setback=setback,
        keepouts=keepout_polys,
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

    validated = []
    containment_rejects = 0
    clearance_rejects = 0
    boundary_setback_rejects = 0

    for i, placements in enumerate(solver_result.solutions):
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

        validated.append(placements)

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

    solutions_with_roads = []
    road_generation_failures = 0
    for i, placements in enumerate(validated):
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

        solutions_with_roads.append((placements, road_network))

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

    # Create solution pool
    pool = SolutionPool(max_size=len(solutions_with_roads))
    for placements, road_network in solutions_with_roads:
        pool.add(placements)

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
        for placements, rn in solutions_with_roads:
            if placements == entry.placements:
                road_network = rn
                break

        # Compute metrics
        metrics = _compute_metrics(entry.placements, road_network, hints)

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
                )
                for p in entry.placements
            ],
            road_network=road_network,
            metrics=metrics,
            diversity_note=_generate_diversity_note(entry, diverse_entries, rank),
        )

        # Generate GeoJSON
        solution.features_geojson = _generate_geojson(
            solution, site_boundary, entrances
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
) -> SolutionMetrics:
    """Compute solution metrics."""
    from .topology.placement_hints import compute_flow_violation_score

    # Compactness (convex hull ratio)
    from shapely.ops import unary_union

    if placements:
        all_polys = [p.to_shapely_polygon() for p in placements]
        combined = unary_union(all_polys)
        hull = combined.convex_hull
        compactness = combined.area / hull.area if hull.area > 0 else 0.0
    else:
        compactness = 0.0

    # Road length
    road_length = road_network.total_length if road_network else 0.0

    # Flow violation
    positions = {p.structure_id: (p.x, p.y) for p in placements}
    topology_penalty = compute_flow_violation_score(hints, positions)

    return SolutionMetrics(
        pipe_length_weighted=0.0,  # Would need actual pipe routing
        road_length=road_length,
        site_utilization=0.0,  # Would need total buildable area
        compactness=compactness,
        topology_penalty=topology_penalty,
    )


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
) -> Dict[str, Any]:
    """Generate full GeoJSON for solution visualization."""
    features = []

    # Site boundary
    boundary_coords = list(boundary.exterior.coords)
    features.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [boundary_coords]},
        "properties": {"kind": "boundary"},
    })

    # Entrances
    for ent in entrances:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": list(ent.point)},
            "properties": {"kind": "entrance", "id": ent.id},
        })

    # Structures
    for p in solution.placements:
        min_x, min_y = p.x - p.width / 2, p.y - p.height / 2
        max_x, max_y = p.x + p.width / 2, p.y + p.height / 2
        coords = [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y],
        ]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "kind": "structure",
                "id": p.structure_id,
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
