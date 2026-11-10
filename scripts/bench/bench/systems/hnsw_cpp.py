from __future__ import annotations

import time
from typing import Any, Mapping

import hnsw_cpp  # ty: ignore[import-not-found]
import numpy as np

from ..interface import BenchmarkSystem

_REQUIRED_PARAMS = (
    "space",
    "m",
    "ef_construction",
    "ef_search",
    "seed",
    "heuristic",
    "visited_mode",
    "alpha",
    "generosity",
    "subdist",
    "subdist_prefix",
    "subdist_samples",
    "subdist_max_stddev",
)


def _build(train: np.ndarray, params: Mapping[str, Any]) -> tuple[Any, float]:
    # Every tuning constant is required. They live in the benchmark presets so
    # the preset is the single source of truth; a missing key fails loudly
    # instead of silently falling back to a library default.
    missing = [key for key in _REQUIRED_PARAMS if key not in params]
    if missing:
        raise KeyError(
            "hnsw-cpp preset is missing required constants: "
            + ", ".join(missing)
        )

    index = hnsw_cpp.HNSWIndex(
        dim=int(train.shape[1]),
        space=str(params["space"]),
        m=int(params["m"]),
        ef_construction=int(params["ef_construction"]),
        ef_search=int(params["ef_search"]),
        seed=int(params["seed"]),
        heuristic=bool(params["heuristic"]),
        visited_mode=str(params["visited_mode"]),
        alpha=float(params["alpha"]),
        generosity=float(params["generosity"]),
        subdist=bool(params["subdist"]),
        subdist_prefix=int(params["subdist_prefix"]),
        subdist_samples=int(params["subdist_samples"]),
        subdist_max_stddev=float(params["subdist_max_stddev"]),
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
    label="hnsw.cpp",
    build=_build,
    set_query_params=_set_query_params,
    search_batch=_search_batch,
    search_one=_search_one,
    index_size_bytes=_index_size_bytes,
    # Harness-only defaults. Library tuning constants are supplied by presets.
    default_params={
        "query_mode": "sequential",
        "query_threads": 1,
        "sweep": {"ef": [10, 20, 40, 80, 160, 320]},
    },
)
