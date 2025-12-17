"""Solution fingerprinting and diversity filtering.

Ensures returned solutions are meaningfully different from each other
by computing multi-dimensional fingerprints and filtering based on distance.
"""

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..models.structures import PlacedStructure

logger = logging.getLogger(__name__)


@dataclass
class SolutionFingerprint:
    """Multi-dimensional fingerprint for comparing solution similarity.

    Components:
    1. Normalized centroids (40% weight) - Where structures are placed
    2. Relative orderings (40% weight) - Which structures are left/above others
    3. Cluster assignments (20% weight) - How structures group spatially
    """

    # Normalized (x, y) positions for each structure
    normalized_positions: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Set of ordering relationships: (id1, id2, 'east'|'north')
    orderings: set[tuple[str, str, str]] = field(default_factory=set)

    # Cluster ID for each structure (from DBSCAN or similar)
    cluster_assignments: dict[str, int] = field(default_factory=dict)

    # Bounding box used for normalization
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    @classmethod
    def from_placements(
        cls,
        placements: list[PlacedStructure],
        cluster_eps: float = 15.0,
    ) -> "SolutionFingerprint":
        """Create fingerprint from placement list.

        Args:
            placements: List of placed structures
            cluster_eps: DBSCAN epsilon for clustering (meters)

        Returns:
            SolutionFingerprint
        """
        if not placements:
            return cls()

        # Compute bounding box
        xs = [p.x for p in placements]
        ys = [p.y for p in placements]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Avoid division by zero
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)

        bounds = (min_x, min_y, max_x, max_y)

        # Normalized positions
        normalized = {}
        for p in placements:
            nx = (p.x - min_x) / width
            ny = (p.y - min_y) / height
            normalized[p.structure_id] = (nx, ny)

        # Compute orderings
        orderings = set()
        for p1, p2 in combinations(placements, 2):
            id1, id2 = p1.structure_id, p2.structure_id
            # Ensure consistent ordering by sorting IDs
            if id1 > id2:
                id1, id2 = id2, id1
                p1, p2 = p2, p1

            # East-west relationship
            if p1.x < p2.x - 1.0:  # 1m threshold
                orderings.add((id1, id2, 'east'))
            elif p2.x < p1.x - 1.0:
                orderings.add((id2, id1, 'east'))

            # North-south relationship
            if p1.y < p2.y - 1.0:
                orderings.add((id1, id2, 'north'))
            elif p2.y < p1.y - 1.0:
                orderings.add((id2, id1, 'north'))

        # Cluster assignments using DBSCAN
        clusters = _compute_clusters(placements, eps=cluster_eps)

        return cls(
            normalized_positions=normalized,
            orderings=orderings,
            cluster_assignments=clusters,
            bounds=bounds,
        )

    def distance_to(
        self,
        other: "SolutionFingerprint",
        centroid_weight: float = 0.4,
        ordering_weight: float = 0.4,
        cluster_weight: float = 0.2,
    ) -> float:
        """Compute distance to another fingerprint.

        Args:
            other: Other fingerprint to compare
            centroid_weight: Weight for centroid distance (default 0.4)
            ordering_weight: Weight for ordering distance (default 0.4)
            cluster_weight: Weight for cluster distance (default 0.2)

        Returns:
            Distance in [0, 1] range (0 = identical, 1 = completely different)
        """
        # Centroid distance (Euclidean on normalized positions)
        centroid_dist = _compute_centroid_distance(
            self.normalized_positions, other.normalized_positions
        )

        # Ordering distance (Jaccard on ordering sets)
        ordering_dist = _compute_jaccard_distance(
            self.orderings, other.orderings
        )

        # Cluster distance (Rand index based)
        cluster_dist = _compute_cluster_distance(
            self.cluster_assignments, other.cluster_assignments
        )

        # Weighted combination
        total = (
            centroid_weight * centroid_dist
            + ordering_weight * ordering_dist
            + cluster_weight * cluster_dist
        )

        return min(1.0, max(0.0, total))


def _compute_clusters(
    placements: list[PlacedStructure],
    eps: float = 15.0,
    min_samples: int = 1,
) -> dict[str, int]:
    """Compute spatial clusters using DBSCAN.

    Args:
        placements: Structure placements
        eps: Maximum distance between points in cluster
        min_samples: Minimum points for cluster

    Returns:
        Dict mapping structure_id to cluster_id
    """
    if not placements:
        return {}

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # Fallback: each structure in own cluster
        return {p.structure_id: i for i, p in enumerate(placements)}

    # Prepare coordinates
    coords = np.array([[p.x, p.y] for p in placements])

    # Run DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)

    # Map to structure IDs
    clusters = {}
    for i, p in enumerate(placements):
        clusters[p.structure_id] = int(clustering.labels_[i])

    return clusters


def _compute_centroid_distance(
    pos1: dict[str, tuple[float, float]],
    pos2: dict[str, tuple[float, float]],
) -> float:
    """Compute average Euclidean distance between normalized positions."""
    if not pos1 or not pos2:
        return 1.0

    common_ids = set(pos1.keys()) & set(pos2.keys())
    if not common_ids:
        return 1.0

    total_dist = 0.0
    for sid in common_ids:
        x1, y1 = pos1[sid]
        x2, y2 = pos2[sid]
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        total_dist += dist

    # Normalize by max possible distance (sqrt(2) for unit square)
    avg_dist = total_dist / len(common_ids)
    return min(1.0, avg_dist / np.sqrt(2))


def _compute_jaccard_distance(
    set1: set[tuple[str, str, str]],
    set2: set[tuple[str, str, str]],
) -> float:
    """Compute Jaccard distance between ordering sets."""
    if not set1 and not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0.0

    jaccard_similarity = intersection / union
    return 1.0 - jaccard_similarity


def _compute_cluster_distance(
    clusters1: dict[str, int],
    clusters2: dict[str, int],
) -> float:
    """Compute distance based on cluster agreement (1 - Rand Index)."""
    if not clusters1 or not clusters2:
        return 1.0

    common_ids = set(clusters1.keys()) & set(clusters2.keys())
    if len(common_ids) < 2:
        return 0.0  # Can't compute with < 2 points

    # Compute Rand Index
    agreements = 0
    total_pairs = 0

    ids = list(common_ids)
    for i, id1 in enumerate(ids):
        for id2 in ids[i + 1:]:
            same_c1 = clusters1[id1] == clusters1[id2]
            same_c2 = clusters2[id1] == clusters2[id2]

            if same_c1 == same_c2:
                agreements += 1
            total_pairs += 1

    if total_pairs == 0:
        return 0.0

    rand_index = agreements / total_pairs
    return 1.0 - rand_index


def compute_solution_distance(
    placements1: list[PlacedStructure],
    placements2: list[PlacedStructure],
) -> float:
    """Compute distance between two solutions.

    Convenience function that creates fingerprints and computes distance.

    Args:
        placements1: First solution placements
        placements2: Second solution placements

    Returns:
        Distance in [0, 1] range
    """
    fp1 = SolutionFingerprint.from_placements(placements1)
    fp2 = SolutionFingerprint.from_placements(placements2)
    return fp1.distance_to(fp2)


def filter_diverse_solutions(
    solutions: list["SolutionEntry"],
    target_count: int,
    min_distance: float = 0.1,
) -> list["SolutionEntry"]:
    """Select diverse solutions using greedy max-min distance selection.

    Algorithm:
    1. Start with best solution (rank 0)
    2. Iteratively add solution with maximum minimum distance to selected set
    3. Continue until target_count reached or no solution exceeds min_distance

    Args:
        solutions: List of SolutionEntry objects with fingerprints
        target_count: Target number of solutions
        min_distance: Minimum distance threshold

    Returns:
        List of diverse solutions
    """
    if not solutions:
        return []

    if len(solutions) <= target_count:
        return solutions

    # Start with best solution
    selected = [solutions[0]]
    remaining = list(solutions[1:])

    while len(selected) < target_count and remaining:
        # Find solution with max min-distance to selected set
        best_candidate = None
        best_min_dist = -1.0

        for candidate in remaining:
            if candidate.fingerprint is None:
                continue

            # Compute min distance to all selected
            min_dist = float('inf')
            for sel in selected:
                if sel.fingerprint is None:
                    continue
                dist = candidate.fingerprint.distance_to(sel.fingerprint)
                min_dist = min(min_dist, dist)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate

        # Check if best candidate exceeds threshold
        if best_candidate is None or best_min_dist < min_distance:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

        logger.debug(
            f"Selected solution {len(selected)}: min_dist={best_min_dist:.3f}"
        )

    return selected


# Forward reference for type hints
from .solution_pool import SolutionEntry  # noqa: E402, F811
