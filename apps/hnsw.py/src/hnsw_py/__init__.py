"""Pure-Python HNSW approximate nearest-neighbour search."""

from __future__ import annotations

import random

from .index import HNSWIndex

__all__ = ["HNSWIndex"]
__version__ = "0.1.0"


def main() -> None:
    """Small self-contained demo used by the ``hnsw-py`` console script."""
    rng = random.Random(0)
    dim = 16
    count = 200
    vectors = [
        [rng.gauss(0.0, 1.0) for _ in range(dim)]
        for _ in range(count)
    ]

    index = HNSWIndex(
        dim=dim,
        m=8,
        ef_construction=100,
        ef_search=50,
        seed=1,
    )
    index.add_items(vectors)

    labels, distances = index.search(vectors[0], k=5)
    print(f"indexed {len(index)} vectors")
    print(f"nearest labels: {labels}")
    print(f"distances: {[round(distance, 4) for distance in distances]}")
