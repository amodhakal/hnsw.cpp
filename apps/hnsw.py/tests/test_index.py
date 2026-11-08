from __future__ import annotations

import math
import random
import unittest

from hnsw_py import HNSWIndex
from hnsw_py.distance import (
    cosine_distance,
    inner_product_distance,
    l2_distance,
    normalize,
)


def _make_vectors(count: int, dim: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        for _ in range(count)
    ]


def _distance(space: str, a: list[float], b: list[float]) -> float:
    if space == "cosine":
        return cosine_distance(normalize(a), normalize(b))
    if space == "l2":
        return l2_distance(a, b)
    return inner_product_distance(a, b)


def _brute_force(
    vectors: list[list[float]],
    query: list[float],
    k: int,
    space: str,
) -> list[int]:
    distances = [
        _distance(space, vector, query)
        for vector in vectors
    ]
    return sorted(range(len(vectors)), key=lambda index: distances[index])[:k]


class HNSWIndexTests(unittest.TestCase):
    def test_recall_against_brute_force(self) -> None:
        vectors = _make_vectors(300, 16, seed=7)
        queries = _make_vectors(20, 16, seed=8)

        for space in ("cosine", "l2", "ip"):
            with self.subTest(space=space):
                index = HNSWIndex(
                    dim=16,
                    space=space,
                    m=8,
                    ef_construction=100,
                    ef_search=100,
                    seed=42,
                )
                index.add_items(vectors)

                hits = 0
                for query in queries:
                    labels, _ = index.search(query, k=5)
                    exact = _brute_force(vectors, query, 5, space)
                    hits += len(set(labels) & set(exact))

                recall = hits / (len(queries) * 5)
                self.assertGreaterEqual(recall, 0.9)

    def test_deterministic_with_same_seed(self) -> None:
        vectors = _make_vectors(200, 8, seed=1)
        queries = _make_vectors(10, 8, seed=2)

        first = HNSWIndex(dim=8, m=8, seed=123)
        first.add_items(vectors)
        second = HNSWIndex(dim=8, m=8, seed=123)
        second.add_items(vectors)

        for query in queries:
            self.assertEqual(first.search(query, k=5), second.search(query, k=5))

    def test_custom_labels_round_trip(self) -> None:
        vectors = _make_vectors(20, 4, seed=3)
        labels = list(range(1000, 1020))
        index = HNSWIndex(dim=4, m=4, seed=5)
        index.add_items(vectors, labels)

        found, _ = index.search(vectors[0], k=3)
        self.assertTrue(set(found).issubset(set(labels)))
        self.assertIn(1000, found)

    def test_k_larger_than_index(self) -> None:
        vectors = _make_vectors(5, 4, seed=4)
        index = HNSWIndex(dim=4, m=4, seed=5)
        index.add_items(vectors)

        labels, distances = index.search(vectors[0], k=10)
        self.assertEqual(len(labels), 5)
        self.assertEqual(len(distances), 5)

    def test_empty_index_raises(self) -> None:
        index = HNSWIndex(dim=4)
        with self.assertRaises(RuntimeError):
            index.search([0.0, 0.0, 0.0, 0.0])

    def test_dimension_mismatch_raises(self) -> None:
        index = HNSWIndex(dim=4)
        with self.assertRaises(ValueError):
            index.add_item([0.0, 0.0, 0.0])

    def test_invalid_space_raises(self) -> None:
        with self.assertRaises(ValueError):
            HNSWIndex(dim=4, space="manhattan")

    def test_set_ef(self) -> None:
        vectors = _make_vectors(50, 8, seed=6)
        index = HNSWIndex(dim=8, m=4, seed=7)
        index.add_items(vectors)

        index.set_ef(80)
        self.assertEqual(index.ef_search, 80)
        labels, _ = index.search(vectors[0], k=5)
        self.assertEqual(len(labels), 5)

        with self.assertRaises(ValueError):
            index.set_ef(0)

    def test_index_size_is_positive(self) -> None:
        vectors = _make_vectors(20, 4, seed=9)
        index = HNSWIndex(dim=4, m=4, seed=10)
        index.add_items(vectors)
        self.assertGreater(index.index_size_bytes(), 0)

    def test_batch_matches_single(self) -> None:
        vectors = _make_vectors(100, 8, seed=11)
        queries = _make_vectors(5, 8, seed=12)
        index = HNSWIndex(dim=8, m=8, seed=13)
        index.add_items(vectors)

        batch_labels, batch_distances = index.knn_query(queries, k=5)
        for row, query in enumerate(queries):
            labels, distances = index.search(query, k=5)
            self.assertEqual(batch_labels[row], labels)
            self.assertEqual(batch_distances[row], distances)


if __name__ == "__main__":
    unittest.main()
