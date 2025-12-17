"""SFILES2 string parser for extracting process topology.

Uses the Flowsheet_Class from SFILES2 library for parsing.
Falls back to a simple regex-based parser if library not available.
"""

import logging
import re
from typing import Any

import networkx as nx

from ..models.topology import TopologyEdge, TopologyGraph, TopologyNode

logger = logging.getLogger(__name__)


class SfilesParseError(Exception):
    """Error parsing SFILES string."""

    pass


def parse_sfiles_topology(
    sfiles_string: str,
    node_metadata: dict[str, dict[str, Any]] | None = None,
) -> TopologyGraph:
    """Parse SFILES2 string into a TopologyGraph.

    Attempts to use the SFILES2 library (Flowsheet_Class) if available.
    Falls back to simple regex parsing otherwise.

    Args:
        sfiles_string: SFILES2 format string
        node_metadata: Optional dict mapping node IDs to metadata
            (equipment_tag, area_number, category, etc.)

    Returns:
        TopologyGraph with nodes, edges, and computed ranks/SCCs

    Raises:
        SfilesParseError: If parsing fails
    """
    if not sfiles_string or not sfiles_string.strip():
        raise SfilesParseError("Empty SFILES string")

    node_metadata = node_metadata or {}

    try:
        # Try using SFILES2 library
        return _parse_with_flowsheet_class(sfiles_string, node_metadata)
    except ImportError:
        logger.warning("SFILES2 library not available, using fallback parser")
        return _parse_with_fallback(sfiles_string, node_metadata)
    except Exception as e:
        logger.warning(f"SFILES2 library failed ({e}), using fallback parser")
        return _parse_with_fallback(sfiles_string, node_metadata)


def _parse_with_flowsheet_class(
    sfiles_string: str,
    node_metadata: dict[str, dict[str, Any]],
) -> TopologyGraph:
    """Parse using Flowsheet_Class from SFILES2 library."""
    from Flowsheet_Class.flowsheet import Flowsheet

    # Create flowsheet and parse
    flowsheet = Flowsheet()
    flowsheet.sfiles = sfiles_string
    flowsheet.create_from_sfiles()

    # Extract NetworkX graph
    nx_graph: nx.DiGraph = flowsheet.state

    # Convert to TopologyGraph
    nodes = []
    for node_id, attrs in nx_graph.nodes(data=True):
        node_id_str = str(node_id)

        # Get unit type from node name (strip numbers)
        unit_type = re.sub(r"-?\d+$", "", node_id_str)

        # Merge with provided metadata
        meta = node_metadata.get(node_id_str, {})

        nodes.append(TopologyNode(
            id=node_id_str,
            unit_type=attrs.get("unit_type", unit_type),
            name=attrs.get("name", node_id_str),
            equipment_tag=meta.get("equipment_tag") or attrs.get("equipment_tag"),
            area_number=meta.get("area_number") or attrs.get("area_number"),
            category=meta.get("category") or attrs.get("category"),
            subcategory=meta.get("subcategory") or attrs.get("subcategory"),
        ))

    edges = []
    for source, target, attrs in nx_graph.edges(data=True):
        tags = attrs.get("tags", {"he": [], "col": []})
        if not isinstance(tags, dict):
            tags = {"he": [], "col": []}

        edges.append(TopologyEdge(
            source=str(source),
            target=str(target),
            stream_type=attrs.get("stream_type", "material"),
            stream_name=attrs.get("stream_name"),
            tags=tags,
            weight=attrs.get("weight", 1.0),
        ))

    topology = TopologyGraph(nodes=nodes, edges=edges)
    topology.compute_ranks_and_sccs()

    return topology


def _parse_with_fallback(
    sfiles_string: str,
    node_metadata: dict[str, dict[str, Any]],
) -> TopologyGraph:
    """Fallback regex-based parser for basic SFILES strings.

    Handles simple sequential flows and basic branches.
    Does not support all SFILES v2 features (heat integration tags, etc.).
    """
    # SFILES element patterns
    unit_pattern = r"\(([^)]+)\)"  # (unit-name)
    # Patterns for future SFILES v2 features (tags, cycles)
    # tag_pattern = r"\{([^}]+)\}"  # {tag}
    # cycle_start = r"<(\d+)"  # <1
    # cycle_end = r"(\d+)(?![^(]*\))"  # 1 (not inside parens)

    # Extract all units
    units = re.findall(unit_pattern, sfiles_string)

    if not units:
        raise SfilesParseError(f"No units found in SFILES string: {sfiles_string}")

    # Build nodes
    nodes = []
    seen_ids = set()

    for unit in units:
        # Handle numbered units (e.g., reactor-1)
        base_type = re.sub(r"-?\d+$", "", unit)
        node_id = unit

        # Ensure unique IDs
        if node_id in seen_ids:
            counter = 1
            while f"{node_id}-{counter}" in seen_ids:
                counter += 1
            node_id = f"{node_id}-{counter}"
        seen_ids.add(node_id)

        meta = node_metadata.get(node_id, {})

        nodes.append(TopologyNode(
            id=node_id,
            unit_type=base_type,
            name=unit,
            equipment_tag=meta.get("equipment_tag"),
            area_number=meta.get("area_number"),
            category=meta.get("category"),
            subcategory=meta.get("subcategory"),
        ))

    # Build edges from sequential units
    # This is a simplified approach - full SFILES parsing is complex
    edges = []
    node_ids = [n.id for n in nodes]

    # Simple sequential connections
    for i in range(len(node_ids) - 1):
        # Check if there's a branch between these units
        # For simplicity, assume sequential unless branch detected
        edges.append(TopologyEdge(
            source=node_ids[i],
            target=node_ids[i + 1],
            stream_type="material",
            weight=1.0,
        ))

    # Handle branches [...]
    # Branches create parallel paths from a branch point
    branch_edges = _parse_branch_edges(sfiles_string, node_ids)
    for source_id, target_id in branch_edges:
        # Avoid duplicate edges
        edge_exists = any(
            e.source == source_id and e.target == target_id
            for e in edges
        )
        if not edge_exists:
            edges.append(TopologyEdge(
                source=source_id,
                target=target_id,
                stream_type="material",
                weight=1.0,
            ))

    # Handle recycles <n ... n
    cycle_refs = _find_cycles(sfiles_string)
    for cycle_start_idx, cycle_end_idx in cycle_refs:
        if cycle_start_idx < len(node_ids) and cycle_end_idx < len(node_ids):
            edges.append(TopologyEdge(
                source=node_ids[cycle_end_idx],
                target=node_ids[cycle_start_idx],
                stream_type="material",
                weight=1.0,
            ))

    topology = TopologyGraph(nodes=nodes, edges=edges)
    topology.compute_ranks_and_sccs()

    return topology


def _find_branches(sfiles_string: str) -> list[tuple[int, int]]:
    """Find branch positions in SFILES string.

    Returns list of (start_index, end_index) for each [...] block.
    """
    branches = []
    depth = 0
    start = -1

    for i, char in enumerate(sfiles_string):
        if char == "[":
            if depth == 0:
                start = i
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                branches.append((start, i))
                start = -1

    return branches


def _parse_branch_edges(
    sfiles_string: str,
    node_ids: list[str],
) -> list[tuple[str, str]]:
    """Parse branch structures and return additional edges.

    SFILES branch format: (A)[(B)(C)|&(D)(E)]
    This means: A splits to both B->C and D->E paths

    The | separator divides parallel branches.
    &| indicates a merge point (incoming branch).

    Returns list of (source_id, target_id) tuples for branch edges.
    """
    edges = []
    unit_pattern = r"\(([^)]+)\)"

    # Find all branch blocks
    branch_positions = _find_branches(sfiles_string)

    for branch_start, branch_end in branch_positions:
        # Find the unit immediately before the branch (branch point)
        prefix = sfiles_string[:branch_start]
        prefix_units = re.findall(unit_pattern, prefix)

        if not prefix_units:
            continue

        branch_point_unit = prefix_units[-1]
        branch_point_idx = _find_unit_index(branch_point_unit, node_ids)
        if branch_point_idx < 0:
            continue

        # Extract branch content
        branch_content = sfiles_string[branch_start + 1:branch_end]

        # Split by | to get parallel paths (but not &| which is merge)
        # Use regex to split on | but not &| or n|
        paths = re.split(r"(?<![&n])\|", branch_content)

        for path in paths:
            path = path.strip()
            if not path:
                continue

            # Handle merge marker &| at start of path
            if path.startswith("&"):
                path = path[1:]  # Remove & prefix

            # Find first unit in this path
            path_units = re.findall(unit_pattern, path)
            if path_units:
                first_unit = path_units[0]
                first_unit_idx = _find_unit_index(first_unit, node_ids)
                if first_unit_idx >= 0:
                    # Edge from branch point to first unit of this path
                    edges.append((node_ids[branch_point_idx], node_ids[first_unit_idx]))

        # Find unit after the branch (merge point)
        suffix = sfiles_string[branch_end + 1:]
        suffix_units = re.findall(unit_pattern, suffix)

        if suffix_units:
            merge_point_unit = suffix_units[0]
            merge_point_idx = _find_unit_index(merge_point_unit, node_ids)

            if merge_point_idx >= 0:
                # Connect last unit of each branch path to merge point
                for path in paths:
                    path_units = re.findall(unit_pattern, path)
                    if path_units:
                        last_unit = path_units[-1]
                        last_unit_idx = _find_unit_index(last_unit, node_ids)
                        if last_unit_idx >= 0:
                            edges.append((node_ids[last_unit_idx], node_ids[merge_point_idx]))

    return edges


def _find_unit_index(unit_name: str, node_ids: list[str]) -> int:
    """Find index of a unit in the node_ids list.

    Handles both exact matches and partial matches for numbered units.
    """
    # Try exact match first
    if unit_name in node_ids:
        return node_ids.index(unit_name)

    # Try matching base name (for units that got renamed due to duplicates)
    for i, node_id in enumerate(node_ids):
        if node_id == unit_name or node_id.startswith(unit_name + "-"):
            return i
        # Also check if unit_name is a prefix match
        base = re.sub(r"-\d+$", "", node_id)
        if base == unit_name:
            return i

    return -1


def _find_cycles(sfiles_string: str) -> list[tuple[int, int]]:
    """Find recycle loop references in SFILES string.

    Looks for patterns like <1 ... 1 indicating a recycle stream.
    Returns list of (start_unit_index, end_unit_index) for each cycle.
    """
    cycles = []

    # Find cycle start markers <n
    starts = re.finditer(r"<(\d+)", sfiles_string)
    start_markers = {m.group(1): m.start() for m in starts}

    # Find cycle end markers (just the number)
    for cycle_id, start_pos in start_markers.items():
        # Look for the matching end marker after the start
        end_pattern = rf"(?<!\<)(?<!\d){cycle_id}(?!\d)"
        for m in re.finditer(end_pattern, sfiles_string):
            if m.start() > start_pos:
                # Count units between start and end positions
                units_before_start = len(re.findall(r"\([^)]+\)", sfiles_string[:start_pos]))
                units_before_end = len(re.findall(r"\([^)]+\)", sfiles_string[:m.start()]))
                if units_before_start < units_before_end:
                    cycles.append((units_before_start, units_before_end - 1))
                break

    return cycles


def tokenize_sfiles(sfiles_string: str) -> list[dict[str, Any]]:
    """Tokenize SFILES string into elements.

    Returns list of tokens with type and value.
    Useful for debugging and detailed parsing.
    """
    # Full SFILES tokenization pattern
    pattern = r"(\([^)]+\)|\{[^}]+\}|[<%_]+\d+|\]|\[|<&\||(?<!<)&\||n\||(?<!&)(?<!n)\||&(?!\|)|\d+)"

    tokens = []
    for match in re.finditer(pattern, sfiles_string):
        token = match.group(0)

        if token.startswith("("):
            tokens.append({"type": "unit", "value": token[1:-1]})
        elif token.startswith("{"):
            tokens.append({"type": "tag", "value": token[1:-1]})
        elif token == "[":
            tokens.append({"type": "branch_start", "value": token})
        elif token == "]":
            tokens.append({"type": "branch_end", "value": token})
        elif token.startswith("<") and token[1:].isdigit():
            tokens.append({"type": "cycle_start", "value": int(token[1:])})
        elif token.isdigit():
            tokens.append({"type": "cycle_end", "value": int(token)})
        elif token == "<&|" or token == "&|":
            tokens.append({"type": "incoming_branch", "value": token})
        elif token == "n|":
            tokens.append({"type": "new_flowsheet", "value": token})
        else:
            tokens.append({"type": "unknown", "value": token})

    return tokens
