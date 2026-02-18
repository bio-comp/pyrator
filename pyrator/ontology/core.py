# pyrator/ontology/core.py
from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(slots=True, frozen=True)
class Node:
    """A single ontology node."""

    id: str
    name: str | None = None
    depth: int = 0
    ic: float = 0.0
    meta: dict[str, Any] | None = None


@dataclass(slots=True)
class Ontology:
    """Versioned ontology (tree or DAG) with fast ancestor queries and closure utilities."""

    version: str
    nodes: dict[str, Node] = field(default_factory=dict)
    parents: dict[str, set[str]] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=dict)
    closure: dict[str, set[str]] = field(default_factory=dict)
    lca_policy: Literal["max_ic", "max_depth"] = "max_ic"
    path_policy: Literal["undirected_tr", "spanning_tree"] = "undirected_tr"
    ic_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(  # noqa: C901
        cls,
        version: str,
        nodes: dict[str, dict[str, Any]] | dict[str, Node],
        edges: Iterable[tuple[str, str]],
        corpus_counts: dict[str, int] | None = None,
        smoothing_alpha: float = 1.0,
        lca_policy: Literal["max_ic", "max_depth"] = "max_ic",
        path_policy: Literal["undirected_tr", "spanning_tree"] = "undirected_tr",
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
                id=nid,
                name=old_node.name,
                depth=max_parent_depth + 1,
                meta=old_node.meta,
            )

        # IC Calculation
        ic_metadata = {"alpha": smoothing_alpha, "log_base": "e"}
        if corpus_counts is not None:
            # $L$: base nodes (leaves or atomic concepts)
            # For simplicity, we define base nodes as leaves if not specified.
            # But the README says: "For DAGs, map each corpus occurrence to exactly one $v\in L$."
            # We'll assume $L$ is all nodes that have counts in corpus_counts.

            # $count_{desc}(c) = \sum_{v \in desc(c) \cap L} count_{corpus}(v)$
            # Compute descendant closure for IC calculation

            desc_closure: dict[str, set[str]] = {nid: {nid} for nid in final_nodes}
            # Reverse topo order for descendants
            for nid in reversed(topo_order):
                for child_id in children.get(nid, set()):
                    desc_closure[nid] |= desc_closure[child_id]

            total_corpus_count = sum(corpus_counts.values())
            num_nodes = len(final_nodes)

            for nid in final_nodes:
                # count_desc(c)
                node_descendants = desc_closure[nid]
                desc_count = sum(corpus_counts.get(d, 0) for d in node_descendants)

                # p_hat(c) = (count_desc(c) + alpha) / (total_corpus_count + alpha * |V|)
                p_hat = (desc_count + smoothing_alpha) / (
                    total_corpus_count + smoothing_alpha * num_nodes
                )
                ic = -math.log(p_hat)

                # Update node with IC
                old = final_nodes[nid]
                final_nodes[nid] = Node(
                    id=old.id,
                    name=old.name,
                    depth=old.depth,
                    ic=ic,
                    meta=old.meta,
                )

        return cls(
            version=version,
            nodes=final_nodes,
            parents=parents,
            children=children,
            closure=closure,
            lca_policy=lca_policy,
            path_policy=path_policy,
            ic_metadata=ic_metadata,
        )

    def get_lca(self, u: str, v: str) -> str | None:
        """
        Return the single LCA of two nodes based on the ontology's lca_policy.
        """
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"One or both labels not found in ontology: {u}, {v}")

        lcas = self.lowest_common_ancestors([u, v])
        if not lcas:
            return None

        if len(lcas) == 1:
            return next(iter(lcas))

        if self.lca_policy == "max_ic":
            return max(lcas, key=lambda nid: self.nodes[nid].ic)
        else:  # max_depth
            return max(lcas, key=lambda nid: self.nodes[nid].depth)

    def get_distance(self, u: str, v: str, metric: str = "path") -> float:  # noqa: C901
        """
        Calculate the distance between two nodes using the specified metric.

        Metrics:
            - "path": depth(u) + depth(v) - 2 * depth(LCA)
            - "lca": 1 - s_lca(u, v)
            - "lin": 1 - s_lin(u, v)
            - "resnik_norm": 1 - IC(LCA) / max_ic (normalized Resnik distance)
        """
        if u == v:
            return 0.0

        # Ensure nodes exist
        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"One or both labels not found in ontology: {u}, {v}")

        if metric == "path":
            # Optimization: check if one is ancestor of other immediately
            if self.is_ancestor(u, v):
                return float(self.nodes[v].depth - self.nodes[u].depth)
            if self.is_ancestor(v, u):
                return float(self.nodes[u].depth - self.nodes[v].depth)

            lca = self.get_lca(u, v)
            if lca is None:
                return float(self.nodes[u].depth + self.nodes[v].depth)
            return float(self.nodes[u].depth + self.nodes[v].depth - 2 * self.nodes[lca].depth)

        if metric == "lca":
            return 1.0 - self.get_similarity(u, v, metric="lca")

        if metric == "lin":
            return 1.0 - self.get_similarity(u, v, metric="lin")

        if metric == "resnik_norm":
            lca = self.get_lca(u, v)
            if lca is None:
                return 1.0
            max_ic = max(n.ic for n in self.nodes.values())
            if max_ic == 0:
                return 0.0
            return 1.0 - (self.nodes[lca].ic / max_ic)

        raise ValueError(f"Unknown distance metric: {metric}")

    def get_similarity(self, u: str, v: str, metric: str = "lin") -> float:  # noqa: C901
        """
        Calculate the similarity between two nodes using the specified metric.

        Metrics:
            - "lca": 2 * depth(LCA) / (depth(u) + depth(v))
            - "lin": 2 * IC(LCA) / (IC(u) + IC(v))
            - "resnik": IC(LCA) (Note: raw Resnik is not normalized)
            - "resnik_norm": IC(LCA) / max_ic (normalized similarity in [0, 1])
        """
        if u == v:
            return 1.0

        if u not in self.nodes or v not in self.nodes:
            raise KeyError(f"One or both labels not found in ontology: {u}, {v}")

        lca = self.get_lca(u, v)
        if lca is None:
            return 0.0

        if metric == "lca":
            denom = self.nodes[u].depth + self.nodes[v].depth
            if denom == 0:
                return 1.0 if u == v else 0.0
            return (2 * self.nodes[lca].depth) / denom

        if metric == "lin":
            denom = self.nodes[u].ic + self.nodes[v].ic
            if denom < 1e-12:
                return 1.0 if u == v else 0.0
            return (2 * self.nodes[lca].ic) / denom

        if metric == "resnik":
            return self.nodes[lca].ic

        if metric == "resnik_norm":
            max_ic = max(n.ic for n in self.nodes.values())
            if max_ic == 0:
                return 1.0 if u == v else 0.0
            return self.nodes[lca].ic / max_ic

        raise ValueError(f"Unknown similarity metric: {metric}")

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
        """Returns all descendants of a node by traversing its children."""
        if node_id not in self.nodes:
            return set()

        descendants = set()
        queue = deque(self.children.get(node_id, set()))

        while queue:
            child_id = queue.popleft()
            if child_id not in descendants:
                descendants.add(child_id)
                queue.extend(self.children.get(child_id, set()))

        return descendants

    def get_depth(self, node_id: str) -> int:
        """Return the depth of a node (root depth = 0)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node not found in ontology: {node_id}")
        return self.nodes[node_id].depth

    def get_information_content(self, node_id: str) -> float:
        """Return the information content (IC) of a node."""
        if node_id not in self.nodes:
            raise KeyError(f"Node not found in ontology: {node_id}")
        return self.nodes[node_id].ic

    @classmethod
    def from_csv(cls, path: str | Path, **kwargs: Any) -> "Ontology":
        """
        Build an Ontology from a CSV file.
        Expected format: columns for node_id, parent_id, etc.
        (Implementation depends on exact expected CSV schema, but usually it's edges).
        """
        # Placeholder for basic implementation or specific logic
        # For now, we'll assume a simple (parent, child) edge list CSV
        import pandas as pd

        df = pd.read_csv(path)
        # Assume columns 'parent', 'child'
        edges = list(df[["parent", "child"]].itertuples(index=False, name=None))
        # Collect all nodes
        nodes = {nid: {} for nid in set(df["parent"]) | set(df["child"])}
        return cls.build(version="1.0", nodes=nodes, edges=edges, **kwargs)

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "Ontology":
        """Build an Ontology from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        # data should have 'nodes' and 'edges'
        return cls.build(
            version=data.get("version", "1.0"), nodes=data["nodes"], edges=data["edges"], **kwargs
        )

    def metrics(self) -> list[dict[str, Any]]:
        """Return a list of available metrics and their details."""
        return [
            {
                "key": "path",
                "type": "distance",
                "definition": "depth(u) + depth(v) - 2 * depth(LCA)",
            },
            {"key": "lca", "type": "distance", "definition": "1 - s_lca"},
            {"key": "lin", "type": "distance", "definition": "1 - s_lin"},
            {"key": "resnik_norm", "type": "distance", "definition": "1 - IC(LCA) / max_ic"},
            {
                "key": "lca",
                "type": "similarity",
                "definition": "2 * depth(LCA) / (depth(u) + depth(v))",
            },
            {"key": "lin", "type": "similarity", "definition": "2 * IC(LCA) / (IC(u) + IC(v))"},
            {"key": "resnik", "type": "similarity", "definition": "IC(LCA)"},
            {"key": "resnik_norm", "type": "similarity", "definition": "IC(LCA) / max_ic"},
        ]


def truncate_id_to_depth(node_id: str, depth: int, sep: str = "_") -> str:
    """Truncate a path-like ID 'a_b_c' to a given depth (1-based)."""
    parts = node_id.split(sep)
    if len(parts) - 1 <= depth:
        return node_id
    return sep.join(parts[: 1 + depth])


def _ensure_acyclic(nodes: Collection[str], children: dict[str, set[str]]) -> None:
    """Raise ValueError if a cycle is detected using iterative DFS."""
    # 0 = unvisited, 1 = active path (gray), 2 = done (black)
    state: dict[str, int] = {n: 0 for n in nodes}

    for start in nodes:
        if state[start] != 0:
            continue

        state[start] = 1
        stack: list[tuple[str, Iterator[str]]] = [(start, iter(children.get(start, set())))]

        while stack:
            node_id, iterator = stack[-1]
            try:
                child_id = next(iterator)
            except StopIteration:
                state[node_id] = 2
                stack.pop()
                continue

            child_state = state.get(child_id, 0)
            if child_state == 1:
                raise ValueError(
                    f"Cycle detected: edge {child_id} -> {node_id} closes a cycle (back edge found)"
                )
            if child_state == 0:
                state[child_id] = 1
                stack.append((child_id, iter(children.get(child_id, set()))))


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
        raise ValueError("Cycle detected: Graph is not a DAG")
    return out
