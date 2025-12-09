"""Placement hints derived from process topology.

Converts graph analysis results into soft constraints and preferences
for the CP-SAT placement solver.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from ..models.topology import TopologyGraph
from .graph_analysis import (
    compute_topological_ranks,
    compute_sccs,
    compute_area_clusters,
    get_adjacency_pairs,
    get_flow_precedence_pairs,
    compute_node_degrees,
    identify_critical_path,
)


@dataclass
class PlacementHints:
    """Soft constraints and preferences derived from process topology.

    Used by the CP-SAT solver to guide placement decisions:
    - target_ranks: Preferred x-position layers
    - adjacency_weights: Pair proximity preferences
    - flow_precedence: Directional constraints
    - cluster_assignments: Area-based groupings
    """

    # Node ID -> preferred x-layer (0 = west/inlet)
    target_ranks: Dict[str, int] = field(default_factory=dict)

    # (node1, node2) -> proximity weight (higher = should be closer)
    adjacency_weights: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # List of (upstream, downstream) pairs for flow direction
    flow_precedence: List[Tuple[str, str]] = field(default_factory=list)

    # Cluster ID -> list of node IDs (should be grouped)
    cluster_assignments: Dict[int, List[str]] = field(default_factory=dict)

    # Nodes on critical path (most important for alignment)
    critical_path: List[str] = field(default_factory=list)

    # Node ID -> (in_degree, out_degree) for prioritization
    node_degrees: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    # Node ID -> SCC ID (recycle loops)
    scc_membership: Dict[str, int] = field(default_factory=dict)

    # Source nodes (should be near west edge)
    source_nodes: Set[str] = field(default_factory=set)

    # Sink nodes (should be near east edge)
    sink_nodes: Set[str] = field(default_factory=set)

    def get_rank_for_node(self, node_id: str) -> int:
        """Get target rank for a node, defaulting to 0."""
        return self.target_ranks.get(node_id, 0)

    def get_adjacency_weight(self, node1: str, node2: str) -> float:
        """Get adjacency weight for a pair of nodes."""
        key = tuple(sorted([node1, node2]))
        return self.adjacency_weights.get(key, 0.0)

    def are_in_same_cluster(self, node1: str, node2: str) -> bool:
        """Check if two nodes are in the same area cluster."""
        for cluster_nodes in self.cluster_assignments.values():
            if node1 in cluster_nodes and node2 in cluster_nodes:
                return True
        return False

    def are_in_same_scc(self, node1: str, node2: str) -> bool:
        """Check if two nodes are in the same SCC (recycle loop)."""
        scc1 = self.scc_membership.get(node1)
        scc2 = self.scc_membership.get(node2)
        return scc1 is not None and scc1 == scc2

    def get_high_priority_nodes(self, top_n: int = 5) -> List[str]:
        """Get nodes with highest connectivity (most edges).

        These are hub nodes that should be central in the layout.
        """
        if not self.node_degrees:
            return []

        # Sort by total degree (in + out)
        sorted_nodes = sorted(
            self.node_degrees.items(),
            key=lambda x: x[1][0] + x[1][1],
            reverse=True,
        )
        return [node_id for node_id, _ in sorted_nodes[:top_n]]

    @property
    def num_ranks(self) -> int:
        """Get number of distinct ranks (x-layers)."""
        if not self.target_ranks:
            return 1
        return max(self.target_ranks.values()) + 1


def compute_placement_hints(
    topology: TopologyGraph,
    area_cluster_weight: float = 2.0,
    scc_cluster_weight: float = 3.0,
) -> PlacementHints:
    """Compute placement hints from process topology.

    Analyzes the topology graph to extract:
    - Topological ranks for flow-based x-positioning
    - Adjacency weights for proximity preferences
    - Area-based clustering for grouped placement
    - SCC membership for recycle loop handling

    Args:
        topology: TopologyGraph parsed from SFILES2
        area_cluster_weight: Weight bonus for same-area adjacency
        scc_cluster_weight: Weight bonus for same-SCC adjacency

    Returns:
        PlacementHints with all computed soft constraints
    """
    hints = PlacementHints()

    # Convert to NetworkX for analysis
    nx_graph = topology.to_networkx()

    # Compute topological ranks
    hints.target_ranks = compute_topological_ranks(nx_graph)

    # Compute SCC membership
    hints.scc_membership = compute_sccs(nx_graph)

    # Compute area clusters
    hints.cluster_assignments = compute_area_clusters(topology)

    # Compute adjacency weights from edges
    adjacency_pairs = get_adjacency_pairs(topology)
    for n1, n2, weight in adjacency_pairs:
        key = tuple(sorted([n1, n2]))
        hints.adjacency_weights[key] = weight

    # Boost weights for same-area pairs
    for cluster_nodes in hints.cluster_assignments.values():
        for i, n1 in enumerate(cluster_nodes):
            for n2 in cluster_nodes[i + 1:]:
                key = tuple(sorted([n1, n2]))
                current = hints.adjacency_weights.get(key, 0.0)
                hints.adjacency_weights[key] = current + area_cluster_weight

    # Boost weights for same-SCC pairs
    scc_groups: Dict[int, List[str]] = {}
    for node_id, scc_id in hints.scc_membership.items():
        if scc_id not in scc_groups:
            scc_groups[scc_id] = []
        scc_groups[scc_id].append(node_id)

    for scc_nodes in scc_groups.values():
        if len(scc_nodes) > 1:
            for i, n1 in enumerate(scc_nodes):
                for n2 in scc_nodes[i + 1:]:
                    key = tuple(sorted([n1, n2]))
                    current = hints.adjacency_weights.get(key, 0.0)
                    hints.adjacency_weights[key] = current + scc_cluster_weight

    # Compute flow precedence
    hints.flow_precedence = get_flow_precedence_pairs(topology)

    # Compute node degrees
    hints.node_degrees = compute_node_degrees(nx_graph)

    # Identify critical path
    hints.critical_path = identify_critical_path(nx_graph)

    # Identify sources and sinks
    for node_id, (in_deg, out_deg) in hints.node_degrees.items():
        if in_deg == 0:
            hints.source_nodes.add(node_id)
        if out_deg == 0:
            hints.sink_nodes.add(node_id)

    return hints


def ranks_to_x_bands(
    hints: PlacementHints,
    site_width: float,
    margin: float = 10.0,
) -> Dict[str, Tuple[float, float]]:
    """Convert topological ranks to x-coordinate bands.

    Maps each rank to a horizontal band of the site, ensuring
    process flow proceeds generally from west to east.

    Args:
        hints: PlacementHints with target_ranks
        site_width: Total site width in meters
        margin: Edge margin in meters

    Returns:
        Dict mapping node ID to (min_x, max_x) band
    """
    if not hints.target_ranks:
        # Single band for entire site
        return {}

    num_ranks = hints.num_ranks
    usable_width = site_width - 2 * margin
    band_width = usable_width / num_ranks

    bands = {}
    for node_id, rank in hints.target_ranks.items():
        min_x = margin + rank * band_width
        max_x = margin + (rank + 1) * band_width
        bands[node_id] = (min_x, max_x)

    return bands


def get_cluster_centroids(
    hints: PlacementHints,
    structure_positions: Dict[str, Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """Compute centroid of each area cluster based on placed structures.

    Useful for validating cluster cohesion after placement.

    Args:
        hints: PlacementHints with cluster_assignments
        structure_positions: Dict mapping node ID to (x, y) position

    Returns:
        Dict mapping cluster ID to centroid (x, y)
    """
    centroids = {}

    for cluster_id, node_ids in hints.cluster_assignments.items():
        positions = [
            structure_positions[nid]
            for nid in node_ids
            if nid in structure_positions
        ]
        if positions:
            cx = sum(p[0] for p in positions) / len(positions)
            cy = sum(p[1] for p in positions) / len(positions)
            centroids[cluster_id] = (cx, cy)

    return centroids


def compute_flow_violation_score(
    hints: PlacementHints,
    structure_positions: Dict[str, Tuple[float, float]],
) -> float:
    """Compute penalty score for flow direction violations.

    Penalizes cases where downstream units are west of upstream units.

    Args:
        hints: PlacementHints with flow_precedence
        structure_positions: Dict mapping node ID to (x, y) position

    Returns:
        Total violation score (0 = no violations)
    """
    total_violation = 0.0

    for upstream, downstream in hints.flow_precedence:
        if upstream in structure_positions and downstream in structure_positions:
            up_x = structure_positions[upstream][0]
            down_x = structure_positions[downstream][0]

            # Penalty if downstream is west of upstream
            if down_x < up_x:
                total_violation += up_x - down_x

    return total_violation
