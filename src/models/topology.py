"""Topology models for process flowsheet integration."""

from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field
import networkx as nx


class TopologyNode(BaseModel):
    """A node in the process topology graph (unit operation)."""

    id: str = Field(..., description="Node identifier (from SFILES)")
    unit_type: str = Field(..., description="Unit type: reactor, tank, pump, etc.")
    name: Optional[str] = Field(default=None, description="Descriptive name")
    equipment_tag: Optional[str] = Field(
        default=None, description="Equipment tag (e.g., 230-AS-01)"
    )
    area_number: Optional[int] = Field(default=None, description="Process area number")
    category: Optional[str] = Field(default=None, description="Process category")
    subcategory: Optional[str] = Field(default=None, description="Process subcategory")

    # Computed attributes (populated during graph analysis)
    rank: Optional[int] = Field(default=None, description="Topological rank (0 = inlet)")
    scc_id: Optional[int] = Field(default=None, description="Strongly connected component ID")

    @property
    def semantic_id(self) -> str:
        """Get semantic ID for display."""
        if self.equipment_tag:
            return self.equipment_tag
        return self.id


class TopologyEdge(BaseModel):
    """An edge in the process topology graph (stream)."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    stream_type: str = Field(
        default="material", description="Stream type: material, energy, signal"
    )
    stream_name: Optional[str] = Field(default=None, description="Stream name/number")
    tags: Dict[str, List[str]] = Field(
        default_factory=lambda: {"he": [], "col": []},
        description="SFILES v2 tags for heat exchangers, columns, etc.",
    )
    weight: float = Field(default=1.0, ge=0, description="Edge weight for optimization")


class TopologyGraph(BaseModel):
    """Process topology graph parsed from SFILES2."""

    nodes: List[TopologyNode] = Field(default_factory=list, description="Topology nodes")
    edges: List[TopologyEdge] = Field(default_factory=list, description="Topology edges")

    # Analysis results
    topological_ranks: Dict[str, int] = Field(
        default_factory=dict, description="Node ID -> topological rank"
    )
    scc_mapping: Dict[str, int] = Field(
        default_factory=dict, description="Node ID -> SCC ID for recycle detection"
    )
    area_clusters: Dict[int, List[str]] = Field(
        default_factory=dict, description="Area number -> list of node IDs"
    )

    def get_node(self, node_id: str) -> Optional[TopologyNode]:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_edges_from(self, node_id: str) -> List[TopologyEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[TopologyEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target == node_id]

    def get_adjacent_nodes(self, node_id: str) -> Set[str]:
        """Get IDs of all adjacent nodes (in or out)."""
        adjacent = set()
        for e in self.edges:
            if e.source == node_id:
                adjacent.add(e.target)
            if e.target == node_id:
                adjacent.add(e.source)
        return adjacent

    def get_edge(self, source: str, target: str) -> Optional[TopologyEdge]:
        """Get edge between two nodes if it exists."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX DiGraph for analysis."""
        G = nx.DiGraph()

        # Add nodes with attributes
        for node in self.nodes:
            G.add_node(
                node.id,
                unit_type=node.unit_type,
                name=node.name,
                equipment_tag=node.equipment_tag,
                area_number=node.area_number,
                category=node.category,
                subcategory=node.subcategory,
                rank=node.rank,
                scc_id=node.scc_id,
            )

        # Add edges with attributes
        for edge in self.edges:
            G.add_edge(
                edge.source,
                edge.target,
                stream_type=edge.stream_type,
                stream_name=edge.stream_name,
                tags=edge.tags,
                weight=edge.weight,
            )

        return G

    @classmethod
    def from_networkx(cls, G: nx.DiGraph) -> "TopologyGraph":
        """Create from NetworkX DiGraph."""
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            nodes.append(TopologyNode(
                id=str(node_id),
                unit_type=attrs.get("unit_type", "unknown"),
                name=attrs.get("name"),
                equipment_tag=attrs.get("equipment_tag"),
                area_number=attrs.get("area_number"),
                category=attrs.get("category"),
                subcategory=attrs.get("subcategory"),
                rank=attrs.get("rank"),
                scc_id=attrs.get("scc_id"),
            ))

        edges = []
        for source, target, attrs in G.edges(data=True):
            edges.append(TopologyEdge(
                source=str(source),
                target=str(target),
                stream_type=attrs.get("stream_type", "material"),
                stream_name=attrs.get("stream_name"),
                tags=attrs.get("tags", {"he": [], "col": []}),
                weight=attrs.get("weight", 1.0),
            ))

        return cls(nodes=nodes, edges=edges)

    def compute_ranks_and_sccs(self) -> None:
        """Compute topological ranks and SCCs for all nodes.

        Updates topological_ranks, scc_mapping, and node.rank/node.scc_id.
        """
        G = self.to_networkx()

        # Compute SCCs (for detecting recycle loops)
        sccs = list(nx.strongly_connected_components(G))
        for scc_id, scc in enumerate(sccs):
            for node_id in scc:
                self.scc_mapping[node_id] = scc_id
                node = self.get_node(node_id)
                if node:
                    node.scc_id = scc_id

        # Compute condensation graph for ranking
        condensation = nx.condensation(G)

        # Topological sort of condensation graph
        try:
            topo_order = list(nx.topological_sort(condensation))
        except nx.NetworkXUnfeasible:
            # Graph has cycles that couldn't be condensed - use simple ordering
            topo_order = list(range(len(sccs)))

        # Assign ranks based on condensation order
        scc_to_rank = {scc_id: rank for rank, scc_id in enumerate(topo_order)}

        for node in self.nodes:
            if node.id in self.scc_mapping:
                scc_id = self.scc_mapping[node.id]
                # Find which condensation node this SCC maps to
                for cond_node in condensation.nodes():
                    if node.id in condensation.nodes[cond_node]["members"]:
                        node.rank = list(nx.topological_sort(condensation)).index(cond_node)
                        self.topological_ranks[node.id] = node.rank
                        break
            else:
                node.rank = 0
                self.topological_ranks[node.id] = 0

        # Group by area
        for node in self.nodes:
            if node.area_number is not None:
                if node.area_number not in self.area_clusters:
                    self.area_clusters[node.area_number] = []
                self.area_clusters[node.area_number].append(node.id)

    def get_edge_list_with_weights(self) -> List[Tuple[str, str, float]]:
        """Get list of (source, target, weight) tuples for optimization."""
        return [(e.source, e.target, e.weight) for e in self.edges]
