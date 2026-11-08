from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    train: np.ndarray
    queries: np.ndarray


def make_synthetic(
    n: int,
    dim: int,
    queries: int,
    clusters: int,
    cluster_std: float,
    seed: int,
) -> Dataset:
    """Generate clustered Gaussian vectors, similar to ann-benchmarks data."""
    if n <= 0 or dim <= 0 or queries <= 0:
        raise ValueError("n, dim, and queries must be positive")
    if clusters <= 0:
        raise ValueError("clusters must be positive")

    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim)).astype(np.float32)

    def sample(count: int) -> np.ndarray:
        assignments = rng.integers(0, clusters, size=count)
        noise = rng.normal(
            scale=cluster_std,
            size=(count, dim),
        ).astype(np.float32)
        return np.ascontiguousarray(
            centers[assignments] + noise,
            dtype=np.float32,
        )

    return Dataset(train=sample(n), queries=sample(queries))
