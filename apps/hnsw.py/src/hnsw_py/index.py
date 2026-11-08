"""A pure-Python implementation of Hierarchical Navigable Small World graphs.

The graph construction and search follow Malkov & Yashunin, "Efficient and
robust approximate nearest neighbor search using HNSW graphs" (2016).

Only the standard library is used. Vector arithmetic relies on ``math.sumprod``
and ``math.dist``; the graph itself is plain Python lists, sets, and heaps.
"""

from __future__ import annotations

import heapq
import math
import random
import sys
from typing import Iterable, Sequence

from .distance import DISTANCES, Vector, normalize


class HNSWIndex:
    """Approximate nearest-neighbour index backed by an HNSW graph.

    Parameters
    ----------
    dim:
        Vector dimensionality. Every added vector and query must match it.
    space:
        One of ``"cosine"``, ``"l2"``, or ``"ip"``.
    m:
        Number of neighbours per node on upper layers. Layer 0 keeps ``2 * m``.
    ef_construction:
        Candidate-list size used while inserting.
    ef_search:
        Default candidate-list size used while querying.
    seed:
        Seed for deterministic random level assignment.
    heuristic:
        Use the paper's neighbour-selection heuristic instead of simply taking
        the nearest ``m`` candidates.
    """

    def __init__(
        self,
        dim: int,
        space: str = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 100,
        heuristic: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if space not in DISTANCES:
            available = ", ".join(sorted(DISTANCES))
            raise ValueError(f"Unsupported space '{space}'. Use one of: {available}")
        if m < 2:
            raise ValueError("m must be at least 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be at least 1")
        if ef_search < 1:
            raise ValueError("ef_search must be at least 1")

        self.dim = int(dim)
        self.space = space
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.heuristic = bool(heuristic)

        self._distance = DISTANCES[space]
        self._rng = random.Random(seed)
        self._level_mult = 1.0 / math.log(self.m)
        self._max_links = self.m
        self._max_links0 = 2 * self.m

        self._vectors: list[list[float]] = []
        self._labels: list[int] = []
        self._levels: list[int] = []
        self._links: list[list[list[int]]] = []

        self._entry_point: int | None = None
        self._max_level = -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_items(
        self,
        vectors: Iterable[Vector],
        labels: Iterable[int] | None = None,
    ) -> None:
        """Add a batch of vectors, optionally with matching labels."""
        batch = list(vectors)
        if labels is None:
            start = len(self._vectors)
            batch_labels = list(range(start, start + len(batch)))
        else:
            batch_labels = [int(label) for label in labels]
            if len(batch_labels) != len(batch):
                raise ValueError("labels and vectors must have the same length")

        for vector, label in zip(batch, batch_labels):
            self.add_item(vector, label)

    def add_item(self, vector: Vector, label: int | None = None) -> None:
        """Insert one vector into the graph."""
        prepared = self._prepare(vector)
        node = len(self._vectors)
        level = self._random_level()

        self._vectors.append(prepared)
        self._labels.append(node if label is None else int(label))
        self._levels.append(level)
        self._links.append([[] for _ in range(level + 1)])

        if self._entry_point is None:
            self._entry_point = node
            self._max_level = level
            return

        entry = self._entry_point
        for layer in range(self._max_level, level, -1):
            entry = self._greedy(prepared, entry, layer)

        entry_points = [entry]
        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(
                prepared,
                entry_points,
                self.ef_construction,
                layer,
            )
            neighbors = self._select_neighbors(candidates, self.m)
            self._links[node][layer] = neighbors

            limit = self._max_links0 if layer == 0 else self._max_links
            for neighbor in neighbors:
                neighbor_links = self._links[neighbor][layer]
                neighbor_links.append(node)
                if len(neighbor_links) > limit:
                    self._links[neighbor][layer] = self._prune(
                        neighbor,
                        layer,
                        limit,
                    )

            entry_points = [node_id for _, node_id in candidates]

        if level > self._max_level:
            self._entry_point = node
            self._max_level = level

    def set_ef(self, ef: int) -> None:
        """Set the candidate-list size used by subsequent queries."""
        if ef < 1:
            raise ValueError("ef must be at least 1")
        self.ef_search = int(ef)

    def search(self, query: Vector, k: int = 10) -> tuple[list[int], list[float]]:
        """Return ``k`` nearest labels and their distances for one query."""
        if not self._vectors:
            raise RuntimeError("Cannot search an empty index")
        if k < 1:
            raise ValueError("k must be at least 1")

        prepared = self._prepare(query)
        entry = self._entry_point
        assert entry is not None

        for layer in range(self._max_level, 0, -1):
            entry = self._greedy(prepared, entry, layer)

        candidates = self._search_layer(
            prepared,
            [entry],
            max(self.ef_search, k),
            0,
        )
        top = candidates[:k]
        labels = [self._labels[node] for _, node in top]
        distances = [distance for distance, _ in top]
        return labels, distances

    def search_batch(
        self,
        queries: Iterable[Vector],
        k: int = 10,
    ) -> list[tuple[list[int], list[float]]]:
        """Search every query and return one ``(labels, distances)`` pair each."""
        return [self.search(query, k) for query in queries]

    def knn_query(
        self,
        queries: Iterable[Vector],
        k: int = 10,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Batch alias that mirrors ``hnswlib``'s ``knn_query`` shape."""
        results = self.search_batch(queries, k)
        labels = [labels for labels, _ in results]
        distances = [distances for _, distances in results]
        return labels, distances

    def index_size_bytes(self) -> int:
        """Approximate memory footprint of the vectors and graph.

        Python objects carry a lot of per-element overhead, so this is an
        estimate based on container sizes plus typical float/int object sizes.
        """
        size = 0
        for vector in self._vectors:
            size += sys.getsizeof(vector) + len(vector) * 24
        for node_links in self._links:
            for layer_links in node_links:
                size += sys.getsizeof(layer_links) + len(layer_links) * 28
        size += sys.getsizeof(self._vectors)
        size += sys.getsizeof(self._labels)
        size += sys.getsizeof(self._links)
        return size

    def __len__(self) -> int:
        return len(self._vectors)

    # ------------------------------------------------------------------
    # Construction internals
    # ------------------------------------------------------------------
    def _prepare(self, vector: Vector) -> list[float]:
        if len(vector) != self.dim:
            raise ValueError(
                f"Expected a {self.dim}-dimensional vector, got {len(vector)}"
            )
        values = [float(value) for value in vector]
        if self.space == "cosine":
            return normalize(values)
        return values

    def _random_level(self) -> int:
        sample = max(self._rng.random(), 1e-12)
        return int(-math.log(sample) * self._level_mult)

    def _greedy(self, query: Vector, entry: int, layer: int) -> int:
        distance = self._distance
        vectors = self._vectors
        links = self._links

        current = entry
        current_distance = distance(query, vectors[current])

        while True:
            best = current
            best_distance = current_distance
            for neighbor in links[current][layer]:
                neighbor_distance = distance(query, vectors[neighbor])
                if neighbor_distance < best_distance:
                    best = neighbor
                    best_distance = neighbor_distance

            if best == current:
                return current
            current = best
            current_distance = best_distance

    def _search_layer(
        self,
        query: Vector,
        entry_points: Sequence[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        ef = max(1, int(ef))
        distance = self._distance
        vectors = self._vectors
        links = self._links

        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []

        for node in entry_points:
            node_distance = distance(query, vectors[node])
            candidates.append((node_distance, node))
            results.append((-node_distance, node))

        heapq.heapify(candidates)
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            candidate_distance, candidate = heapq.heappop(candidates)
            if len(results) >= ef and candidate_distance > -results[0][0]:
                break

            for neighbor in links[candidate][layer]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                neighbor_distance = distance(query, vectors[neighbor])
                if len(results) < ef or neighbor_distance < -results[0][0]:
                    heapq.heappush(candidates, (neighbor_distance, neighbor))
                    heapq.heappush(results, (-neighbor_distance, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        return sorted((-distance_value, node) for distance_value, node in results)

    def _select_neighbors(
        self,
        candidates: Sequence[tuple[float, int]],
        limit: int,
    ) -> list[int]:
        ordered = sorted(candidates)
        if not self.heuristic:
            return [node for _, node in ordered[:limit]]

        selected: list[int] = []
        selected_set: set[int] = set()

        for candidate_distance, candidate in ordered:
            if len(selected) >= limit:
                break
            if not selected:
                selected.append(candidate)
                selected_set.add(candidate)
                continue

            keep = True
            for other in selected:
                if (
                    self._distance(
                        self._vectors[candidate],
                        self._vectors[other],
                    )
                    < candidate_distance
                ):
                    keep = False
                    break

            if keep:
                selected.append(candidate)
                selected_set.add(candidate)

        if len(selected) < limit:
            for _, candidate in ordered:
                if candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                if len(selected) >= limit:
                    break

        return selected

    def _prune(self, node: int, layer: int, limit: int) -> list[int]:
        """Keep the nearest ``limit`` links when a neighbour list overflows.

        Pruning uses a plain nearest-link cut rather than the diversity
        heuristic. Lists only overflow by one entry at a time, and the
        heuristic's O(limit^2) cross-distance pass dominates construction time
        for little benefit in that case.
        """
        base = self._vectors[node]
        candidates = [
            (self._distance(base, self._vectors[neighbor]), neighbor)
            for neighbor in self._links[node][layer]
        ]
        candidates.sort()
        return [neighbor for _, neighbor in candidates[:limit]]
