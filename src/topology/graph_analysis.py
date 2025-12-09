"""Graph analysis functions for process topology.

Computes topological ranks, strongly connected components, and area clusters
for placement optimization hints.
"""

from typing import Dict, List, Set, Tuple
import networkx as nx

from ..models.topology import TopologyGraph


def compute_topological_ranks(graph: nx.DiGraph) -> Dict[str, int]:
    """Compute topological ranks for all nodes.

    Rank 0 = source nodes (no incoming edges)
    Higher ranks = further downstream in process flow

    For cyclic graphs, condenses SCCs first then ranks condensation graph.

    Args:
        graph: NetworkX directed graph

    Returns:
        Dict mapping node ID to rank (0-indexed)
    """
    if graph.number_of_nodes() == 0:
        return {}

    # Handle cycles via condensation
    condensation = nx.condensation(graph)

    # Get topological order of condensation graph
    try:
        topo_order = list(nx.topological_sort(condensation))
    except nx.NetworkXUnfeasible:
        # Fallback: use BFS from nodes with no predecessors
        topo_order = list(range(len(condensation.nodes())))

    # Map condensation node -> rank
    cond_to_rank = {node: i for i, node in enumerate(topo_order)}

    # Map original nodes to ranks
    ranks = {}
    for cond_node in condensation.nodes():
        members = condensation.nodes[cond_node]["members"]
        rank = cond_to_rank.get(cond_node, 0)
        for node_id in members:
            ranks[str(node_id)] = rank

    return ranks


def compute_sccs(graph: nx.DiGraph) -> Dict[str, int]:
    """Compute strongly connected components.

    Nodes in the same SCC form recycle loops and should be placed adjacently.

    Args:
        graph: NetworkX directed graph

    Returns:
        Dict mapping node ID to SCC ID (0-indexed)
    """
    scc_mapping = {}
    sccs = list(nx.strongly_connected_components(graph))

    for scc_id, scc in enumerate(sccs):
        for node_id in scc:
            scc_mapping[str(node_id)] = scc_id

    return scc_mapping


def compute_area_clusters(
    topology: TopologyGraph,
) -> Dict[int, List[str]]:
    """Group nodes by area number for clustered placement.

    Args:
        topology: TopologyGraph with area_number attributes

    Returns:
        Dict mapping area number to list of node IDs
    """
    clusters: Dict[int, List[str]] = {}

    for node in topology.nodes:
        if node.area_number is not None:
            if node.area_number not in clusters:
                clusters[node.area_number] = []
            clusters[node.area_number].append(node.id)

    return clusters


def get_adjacency_pairs(
    topology: TopologyGraph,
    weight_by_stream_count: bool = True,
) -> List[Tuple[str, str, float]]:
    """Get pairs of nodes that should be placed adjacently.

    Returns weighted pairs based on:
    - Direct edges (process connections)
    - Shared SCC membership (recycle loops)
    - Multiple streams between same units

    Args:
        topology: TopologyGraph to analyze
        weight_by_stream_count: If True, weight pairs by edge count

    Returns:
        List of (node1_id, node2_id, weight) tuples, sorted by weight descending
    """
    pairs: Dict[Tuple[str, str], float] = {}

    # Add edge-based pairs
    for edge in topology.edges:
        key = tuple(sorted([edge.source, edge.target]))
        if key not in pairs:
            pairs[key] = 0.0
        # Higher weight for material streams than signal
        weight = 2.0 if edge.stream_type == "material" else 1.0
        pairs[key] += weight

    # Add SCC-based pairs (recycle loops should be close)
    scc_groups: Dict[int, List[str]] = {}
    for node in topology.nodes:
        if node.scc_id is not None:
            if node.scc_id not in scc_groups:
                scc_groups[node.scc_id] = []
            scc_groups[node.scc_id].append(node.id)

    for scc_id, members in scc_groups.items():
        if len(members) > 1:
            # All pairs within SCC get bonus weight
            for i, n1 in enumerate(members):
                for n2 in members[i + 1:]:
                    key = tuple(sorted([n1, n2]))
                    if key not in pairs:
                        pairs[key] = 0.0
                    pairs[key] += 3.0  # Recycle loop bonus

    # Convert to sorted list
    result = [(k[0], k[1], v) for k, v in pairs.items()]
    result.sort(key=lambda x: x[2], reverse=True)

    return result


def get_flow_precedence_pairs(
    topology: TopologyGraph,
) -> List[Tuple[str, str]]:
    """Get pairs where first node should be upstream (west) of second.

    Based on process flow direction - useful for soft constraints in solver.

    Args:
        topology: TopologyGraph to analyze

    Returns:
        List of (upstream_id, downstream_id) tuples
    """
    precedence = []

    for edge in topology.edges:
        if edge.stream_type == "material":
            # Source should be upstream (west/left) of target
            precedence.append((edge.source, edge.target))

    return precedence


def compute_node_degrees(graph: nx.DiGraph) -> Dict[str, Tuple[int, int]]:
    """Compute in-degree and out-degree for each node.

    Useful for identifying:
    - Sources (in_degree=0): inlet/feed structures
    - Sinks (out_degree=0): outlet/discharge structures
    - Hubs (high total degree): central process units

    Args:
        graph: NetworkX directed graph

    Returns:
        Dict mapping node ID to (in_degree, out_degree)
    """
    degrees = {}
    for node in graph.nodes():
        degrees[str(node)] = (graph.in_degree(node), graph.out_degree(node))
    return degrees


def identify_critical_path(graph: nx.DiGraph) -> List[str]:
    """Identify the critical path through the process.

    The critical path is the longest path from any source to any sink.
    Units on this path are most important for process flow direction.

    Args:
        graph: NetworkX directed graph

    Returns:
        List of node IDs forming the critical path
    """
    if graph.number_of_nodes() == 0:
        return []

    # Find sources (no incoming edges)
    sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if not sources:
        # Graph is cyclic, pick arbitrary start
        sources = [list(graph.nodes())[0]]

    # Find sinks (no outgoing edges)
    sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    if not sinks:
        sinks = [list(graph.nodes())[-1]]

    # Find longest path using DAG longest path (after condensation for cycles)
    try:
        # For DAG, use built-in longest path
        if nx.is_directed_acyclic_graph(graph):
            path = nx.dag_longest_path(graph)
            return [str(n) for n in path]
    except Exception:
        pass

    # Fallback: BFS from sources to find longest simple path
    longest_path = []
    for source in sources:
        for sink in sinks:
            try:
                for path in nx.all_simple_paths(graph, source, sink):
                    if len(path) > len(longest_path):
                        longest_path = path
                    # Limit search to avoid combinatorial explosion
                    if len(longest_path) > 10:
                        break
            except nx.NetworkXNoPath:
                continue
            if len(longest_path) > 10:
                break

    return [str(n) for n in longest_path]
