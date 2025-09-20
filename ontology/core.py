# pyrator/ontology/core.py
from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class Node:
    """A single ontology node."""

    id: str
    name: str | None = None
    depth: int = 0
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class Ontology:
    """Versioned ontology (tree or DAG) with fast ancestor queries and closure utilities."""

    version: str
    nodes: dict[str, Node] = field(default_factory=dict)
    parents: dict[str, set[str]] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=dict)
    closure: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        version: str,
        nodes: dict[str, dict[str, Any]] | dict[str, Node],
        edges: Iterable[tuple[str, str]],
    ) -> "Ontology":
        """Build an Ontology from node dicts and (parent, child) edges."""
        temp_nodes: dict[str, Node] = {
            nid: payload
            if isinstance(payload, Node)
            else Node(
                id=nid,
                name=payload.get("name"),
                meta={k: v for k, v in payload.items() if k != "name"} or None,
            )
            for nid, payload in nodes.items()
        }

        parents: dict[str, set[str]] = {nid: set() for nid in temp_nodes}
        children: dict[str, set[str]] = {nid: set() for nid in temp_nodes}
        for p, c in edges:
            if p not in temp_nodes:
                raise KeyError(f"Parent id not found in nodes: {p}")
            if c not in temp_nodes:
                raise KeyError(f"Child id not found in nodes: {c}")
            parents[c].add(p)
            children[p].add(c)

        _ensure_acyclic(temp_nodes.keys(), children)
        topo_order = _toposort(temp_nodes.keys(), parents, children)

        closure: dict[str, set[str]] = {nid: {nid} for nid in temp_nodes}
        final_nodes: dict[str, Node] = {}
        for nid in topo_order:
            parent_nodes = parents.get(nid, set())
            for p_id in parent_nodes:
                closure[nid] |= closure[p_id]

            max_parent_depth = -1
            if parent_nodes:
                max_parent_depth = max(final_nodes[p_id].depth for p_id in parent_nodes)

            old_node = temp_nodes[nid]
            final_nodes[nid] = Node(
                id=nid, name=old_node.name, depth=max_parent_depth + 1, meta=old_node.meta
            )

        return cls(
            version=version, nodes=final_nodes, parents=parents, children=children, closure=closure
        )

    def expand_with_ancestors(self, labels: Iterable[str], strict: bool = False) -> set[str]:
        """Return labels ∪ all their ancestors (closure)."""
        if strict:
            self.validate_labels_exist(labels)
        out: set[str] = set()
        for lab in labels:
            out.update(self.closure.get(lab, {lab}))
        return out

    def compress_to_frontier(self, labels: Iterable[str]) -> set[str]:
        """Remove any label that is an ancestor of another selected label."""
        labs = set(labels)
        ancestors_in_set = set()
        for x in labs:
            ancestors_in_set.update(self.closure.get(x, {x}) - {x})
        return labs - ancestors_in_set

    def validate_labels_exist(self, labels: Iterable[str]) -> None:
        """Raise KeyError if any label is not found in the ontology."""
        missing = [label for label in labels if label not in self.nodes]
        if missing:
            raise KeyError(f"Unknown labels: {missing}")

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Return True if `a` is an ancestor of `b` (or a==b)."""
        return ancestor in self.closure.get(descendant, {descendant})

    def lowest_common_ancestors(self, labels: Iterable[str]) -> set[str]:
        """Return the set of LCAs (may be >1 in DAGs)."""
        labs = {label for label in labels if label in self.nodes}
        if not labs or len(labs) == 1:
            return labs.copy()

        it = iter(labs)
        common_ancestors = self.closure[next(it)].copy()
        for label in it:
            common_ancestors.intersection_update(self.closure[label])

        if not common_ancestors:
            return set()

        not_lowest = set()
        for x in common_ancestors:
            for child in self.children.get(x, set()):
                if child in common_ancestors:
                    not_lowest.add(x)
                    break
        return common_ancestors - not_lowest

    def ancestors(self, node_id: str) -> set[str]:
        """Returns all ancestors of a node, excluding the node itself."""
        return self.closure.get(node_id, {node_id}) - {node_id}

    def parents_of(self, node_id: str) -> set[str]:
        """Returns the direct parents of a node."""
        return self.parents.get(node_id, set())

    def get_descendants(self, node_id: str) -> set[str]:
        """
        Returns all descendants of a node by traversing its children.
        """
        if node_id not in self.nodes:
            return set()

        descendants = set()
        queue = deque(self.children.get(node_id, set()))

        while queue:
            child_id = queue.popleft()
            if child_id not in descendants:
                descendants.add(child_id)
                # Add the grandchildren to the queue to be processed
                queue.extend(self.children.get(child_id, set()))

        return descendants


def truncate_id_to_depth(node_id: str, depth: int, sep: str = "_") -> str:
    """Truncate a path-like ID 'a_b_c' to a given depth (1-based)."""
    parts = node_id.split(sep)
    if len(parts) - 1 <= depth:
        return node_id
    return sep.join(parts[: 1 + depth])


def _ensure_acyclic(nodes: Collection[str], children: dict[str, set[str]]) -> None:
    """Raise ValueError if a cycle is detected using DFS."""
    path = set()
    visited = set()

    def dfs(u: str) -> None:
        path.add(u)
        visited.add(u)
        for v in children.get(u, set()):
            if v in path:
                raise ValueError(f"Cycle detected involving {u} -> {v}")
            if v not in visited:
                dfs(v)
        path.remove(u)

    for n in nodes:
        if n not in visited:
            dfs(n)


def _toposort(
    nodes: Collection[str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> list[str]:
    """Kahn's algorithm for topological sorting."""
    node_count = len(nodes)
    in_degree = {n: len(parents.get(n, set())) for n in nodes}
    queue = deque([n for n, d in in_degree.items() if d == 0])
    out: list[str] = []

    while queue:
        u = queue.popleft()
        out.append(u)
        for v in children.get(u, set()):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(out) != node_count:
        raise ValueError("Graph is not a DAG (cycle suspected)")
    return out
