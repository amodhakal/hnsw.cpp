from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from hnsw_py import HNSWIndex

from ..interface import BenchmarkSystem


def _build(train: np.ndarray, params: Mapping[str, Any]) -> tuple[Any, float]:
    index = HNSWIndex(
        dim=int(train.shape[1]),
        space=str(params.get("space", "cosine")),
        m=int(params.get("m", 16)),
        ef_construction=int(params.get("ef_construction", 100)),
        ef_search=int(params.get("ef", 50)),
        seed=int(params.get("seed", 100)),
        heuristic=bool(params.get("heuristic", True)),
    )

    # The library is stdlib-only, so convert the NumPy matrix once before the
    # timer starts. The benchmark measures index construction, not conversion.
    vectors = train.tolist()
    labels = list(range(len(vectors)))

    started = time.perf_counter()
    index.add_items(vectors, labels)
    build_seconds = time.perf_counter() - started

    return index, build_seconds


def _set_query_params(index: Any, params: Mapping[str, Any]) -> None:
    if "ef" in params:
        index.set_ef(int(params["ef"]))


def _search_batch(
    index: Any,
    queries: np.ndarray,
    k: int,
    threads: int,
) -> np.ndarray:
    results = index.search_batch(queries.tolist(), k=int(k))
    return np.asarray(
        [labels for labels, _ in results],
        dtype=np.int64,
    )


def _search_one(index: Any, query: np.ndarray, k: int) -> np.ndarray:
    labels, _ = index.search(query.tolist(), k=int(k))
    return np.asarray(labels, dtype=np.int64)


def _index_size_bytes(index: Any) -> int | None:
    try:
        return int(index.index_size_bytes())
    except Exception:
        return None


HNSW_PY = BenchmarkSystem(
    name="hnsw-py",
    label="hnsw.py (pure Python)",
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
