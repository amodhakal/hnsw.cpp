from __future__ import annotations

import time
from typing import Any, Mapping

import hnsw_cpp  # ty: ignore[import-not-found]
import numpy as np

from ..interface import BenchmarkSystem


def _build(train: np.ndarray, params: Mapping[str, Any]) -> tuple[Any, float]:
    index = hnsw_cpp.HNSWIndex(
        dim=int(train.shape[1]),
        space=str(params.get("space", "cosine")),
        m=int(params.get("m", 16)),
        ef_construction=int(params.get("ef_construction", 100)),
        ef_search=int(params.get("ef", 50)),
        seed=int(params.get("seed", 100)),
        heuristic=bool(params.get("heuristic", True)),
        visited_mode=str(params.get("visited_mode", "bloom")),
    )

    labels = np.arange(train.shape[0], dtype=np.uint64)

    started = time.perf_counter()
    index.add_items(train, labels)
    build_seconds = time.perf_counter() - started

    return index, build_seconds


def _set_query_params(index: Any, params: Mapping[str, Any]) -> None:
    if "ef" in params:
        index.set_ef(int(params["ef"]))


def _search_batch(
    index: Any,
    queries: np.ndarray,
    k: int,
    _threads: int,
) -> np.ndarray:
    labels, _ = index.knn_query(queries, k=int(k))
    return labels


def _search_one(index: Any, query: np.ndarray, k: int) -> np.ndarray:
    labels, _ = index.search(query, k=int(k))
    return labels


def _index_size_bytes(index: Any) -> int | None:
    try:
        return int(index.index_size_bytes())
    except Exception:
        return None


HNSW_CPP = BenchmarkSystem(
    name="hnsw-cpp",
    label="hnsw.cpp (C++)",
    build=_build,
    set_query_params=_set_query_params,
    search_batch=_search_batch,
    search_one=_search_one,
    index_size_bytes=_index_size_bytes,
    default_params={
        "space": "cosine",
        "m": 16,
        "ef_construction": 100,
        "seed": 100,
        "query_mode": "sequential",
        "query_threads": 1,
        "sweep": {"ef": [10, 40, 160]},
    },
)
