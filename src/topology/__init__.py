"""Topology parsing and analysis for SFILES2 integration."""

from .sfiles_parser import parse_sfiles_topology, SfilesParseError, tokenize_sfiles
from .graph_analysis import (
    compute_topological_ranks,
    compute_sccs,
    compute_area_clusters,
    get_adjacency_pairs,
    get_flow_precedence_pairs,
    compute_node_degrees,
    identify_critical_path,
)
from .placement_hints import (
    PlacementHints,
    compute_placement_hints,
    ranks_to_x_bands,
    get_cluster_centroids,
    compute_flow_violation_score,
)

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
