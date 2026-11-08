from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-30)


def exact_knn(
    train: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str,
    chunk_size: int = 64,
) -> np.ndarray:
    """Return exact top-k labels for each query.

    Cosine is computed on normalized vectors. L2 ranking uses
    ``||x||^2 - 2 q.x`` (the query norm is constant and can be omitted).
    """
    train = np.asarray(train, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)

    if k < 1 or k > train.shape[0]:
        raise ValueError(f"k must be between 1 and {train.shape[0]}")

    if metric == "cosine":
        train = _normalize(train)
        queries = _normalize(queries)
        train_norms: np.ndarray | None = None
    elif metric == "l2":
        train_norms = np.einsum("ij,ij->i", train, train)
    elif metric == "ip":
        train_norms = None
    else:
        raise ValueError(f"Unsupported metric '{metric}'")

    result = np.empty((queries.shape[0], k), dtype=np.int64)

    for start in range(0, queries.shape[0], chunk_size):
        stop = min(start + chunk_size, queries.shape[0])
        scores = queries[start:stop] @ train.T
        if metric == "l2":
            assert train_norms is not None
            scores = train_norms[None, :] - 2.0 * scores
            indices = np.argpartition(scores, kth=k - 1, axis=1)[:, :k]
        else:
            indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        result[start:stop] = indices

    return result
