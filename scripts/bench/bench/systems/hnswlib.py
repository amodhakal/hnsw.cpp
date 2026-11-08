from __future__ import annotations

import time
from typing import Any, Mapping

import hnswlib  # ty: ignore[import-not-found]
import numpy as np

from ..interface import BenchmarkSystem


def _build(train: np.ndarray, params: Mapping[str, Any]) -> tuple[Any, float]:
    index = hnswlib.Index(
        space=str(params.get("space", "cosine")),
        dim=int(train.shape[1]),
    )
    index.init_index(
        max_elements=int(train.shape[0]),
        M=int(params.get("m", 16)),
        ef_construction=int(params.get("ef_construction", 200)),
        random_seed=int(params.get("seed", 100)),
    )

    build_threads = params.get("build_threads")
    if build_threads is not None:
        index.set_num_threads(int(build_threads))

    labels = np.arange(train.shape[0], dtype=np.uint64)
    started = time.perf_counter()
    index.add_items(train, labels)
    build_seconds = time.perf_counter() - started

    return index, build_seconds


def _set_query_params(index: Any, params: Mapping[str, Any]) -> None:
    if "ef" in params:
        index.set_ef(int(params["ef"]))


def _search_batch(
    index: Any, queries: np.ndarray, k: int, threads: int
) -> np.ndarray:
    labels, _ = index.knn_query(queries, k=int(k), num_threads=int(threads))
    return labels


def _search_one(index: Any, query: np.ndarray, k: int) -> np.ndarray:
    labels, _ = index.knn_query(query, k=int(k), num_threads=1)
    return labels[0]


def _index_size_bytes(index: Any) -> int | None:
    try:
        return int(index.index_file_size())
    except Exception:
        return None


HNSWLIB = BenchmarkSystem(
    name="hnswlib",
    label="hnswlib",
    build=_build,
    set_query_params=_set_query_params,
    search_batch=_search_batch,
    search_one=_search_one,
    index_size_bytes=_index_size_bytes,
    default_params={
        "space": "cosine",
        "m": 16,
        "ef_construction": 200,
        "build_threads": 8,
        "query_threads": 1,
        "query_mode": "sequential",
        "sweep": {"ef": [10, 20, 40, 80, 160, 320]},
    },
)
