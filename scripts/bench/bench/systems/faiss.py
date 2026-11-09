from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import faiss  # ty: ignore[import-not-found]
import numpy as np

from ..interface import BenchmarkSystem


@dataclass
class _Index:
    index: Any
    normalize: bool


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norms, 1e-30), dtype=np.float32)


def _metric(space: str) -> int:
    if space in {"cosine", "ip"}:
        return int(faiss.METRIC_INNER_PRODUCT)
    if space == "l2":
        return int(faiss.METRIC_L2)
    raise ValueError(f"Unsupported metric '{space}'")


def _build(train: np.ndarray, params: Mapping[str, Any]) -> tuple[Any, float]:
    space = str(params.get("space", "cosine"))
    normalize = space == "cosine"

    index = faiss.IndexHNSWFlat(
        int(train.shape[1]),
        int(params.get("m", 16)),
        _metric(space),
    )
    index.hnsw.efConstruction = int(params.get("ef_construction", 200))
    index.hnsw.efSearch = int(params.get("ef", 50))

    vectors = _normalize(train) if normalize else np.ascontiguousarray(
        train, dtype=np.float32
    )

    started = time.perf_counter()
    index.add(vectors)
    build_seconds = time.perf_counter() - started

    return _Index(index=index, normalize=normalize), build_seconds


def _set_query_params(handle: _Index, params: Mapping[str, Any]) -> None:
    if "ef" in params:
        handle.index.hnsw.efSearch = int(params["ef"])


def _search_batch(
    handle: _Index, queries: np.ndarray, k: int, _threads: int
) -> np.ndarray:
    vectors = (
        _normalize(queries)
        if handle.normalize
        else np.ascontiguousarray(queries, dtype=np.float32)
    )
    _, labels = handle.index.search(vectors, int(k))
    return labels


def _search_one(handle: _Index, query: np.ndarray, k: int) -> np.ndarray:
    vector = query.reshape(1, -1)
    vector = (
        _normalize(vector)
        if handle.normalize
        else np.ascontiguousarray(vector, dtype=np.float32)
    )
    _, labels = handle.index.search(vector, int(k))
    return labels[0]


def _index_size_bytes(handle: _Index) -> int | None:
    try:
        return int(faiss.serialize_index(handle.index).nbytes)
    except Exception:
        return None


FAISS = BenchmarkSystem(
    name="faiss",
    label="FAISS (HNSW)",
    build=_build,
    set_query_params=_set_query_params,
    search_batch=_search_batch,
    search_one=_search_one,
    index_size_bytes=_index_size_bytes,
    default_params={
        "space": "cosine",
        "m": 16,
        "ef_construction": 200,
        "query_mode": "sequential",
        "query_threads": 1,
        "sweep": {"ef": [10, 20, 40, 80, 160, 320]},
    },
)
