"""Topology parsing and analysis for SFILES2 integration."""

from .graph_analysis import (
    compute_area_clusters,
    compute_node_degrees,
    compute_sccs,
    compute_topological_ranks,
    get_adjacency_pairs,
    get_flow_precedence_pairs,
    identify_critical_path,
)
from .placement_hints import (
    PlacementHints,
    compute_flow_violation_score,
    compute_placement_hints,
    get_cluster_centroids,
    ranks_to_x_bands,
)
from .sfiles_parser import SfilesParseError, parse_sfiles_topology, tokenize_sfiles

__all__ = [
    # Parser
    "parse_sfiles_topology",
    "SfilesParseError",
    "tokenize_sfiles",
    # Graph analysis
    "compute_topological_ranks",
    "compute_sccs",
    "compute_area_clusters",
    "get_adjacency_pairs",
    "get_flow_precedence_pairs",
    "compute_node_degrees",
    "identify_critical_path",
    # Placement hints
    "PlacementHints",
    "compute_placement_hints",
    "ranks_to_x_bands",
    "get_cluster_centroids",
    "compute_flow_violation_score",
]
