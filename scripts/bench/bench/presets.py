from __future__ import annotations

# Plain Python presets. No TOML, no inheritance.
#
# Benchmark tuning constants live here, not in the systems: each entry below
# fully specifies the values that system is built and queried with. This makes
# the preset the single source of truth and keeps the comparison apples-to-apples
# (for example, every HNSW system builds with ef_construction=200).
#
# Adding a system means adding an explicit entry here. If you want to reuse a
# system's defaults instead, set "benchmarks": "all" and put the constants in
# that system's ``default_params``.

_HNSW_SWEEP = {"ef": [10, 20, 40, 80, 160, 320]}

_HNSWLIB = {
    "library": "hnswlib",
    "label": "hnswlib",
    "params": {
        "space": "cosine",
        "m": 16,
        "ef_construction": 200,
        "build_threads": 8,
        "query_threads": 1,
        "query_mode": "sequential",
        "sweep": _HNSW_SWEEP,
    },
}

_HNSW_CPP = {
    "library": "hnsw-cpp",
    "label": "hnsw.cpp",
    "params": {
        "space": "cosine",
        "m": 16,
        "ef_construction": 200,
        # Search-time ef. The sweep below overrides it before every measurement.
        "ef_search": 50,
        "seed": 100,
        "heuristic": True,
        "visited_mode": "bloom",
        "alpha": 1.0,
        "generosity": 1.0,
        "subdist": True,
        "subdist_prefix": 32,
        "subdist_samples": 10000,
        "subdist_max_stddev": 0.1,
        "query_mode": "sequential",
        "query_threads": 1,
        "sweep": _HNSW_SWEEP,
    },
}

_FAISS = {
    "library": "faiss",
    "label": "FAISS (HNSW)",
    "params": {
        "space": "cosine",
        "m": 16,
        "ef_construction": 200,
        "query_mode": "sequential",
        "query_threads": 1,
        "sweep": _HNSW_SWEEP,
    },
}

PRESETS = {
    "default": {
        "dataset": {
            "n": 1_000_000,
            "dim": 128,
            "queries": 10_000,
            "clusters": 1_000,
            "cluster_std": 0.05,
            "seed": 42,
        },
        "metrics": {
            "k": 10,
            "reps": 5,
            "warmup": 100,
            "target_recall": 0.99,
            "max_ef": 2048,
        },
        "output": {"dir": "results", "format": "png"},
        "benchmarks": [_HNSWLIB, _HNSW_CPP, _FAISS],
    },
    "fast": {
        "dataset": {
            "n": 80_000,
            "dim": 128,
            "queries": 10_000,
            "clusters": 100,
            "cluster_std": 0.05,
            "seed": 42,
        },
        "metrics": {
            "k": 10,
            "reps": 1,
            "warmup": 20,
            "target_recall": 0.99,
            "max_ef": 512,
        },
        "output": {"dir": "results", "format": "png"},
        "benchmarks": [_HNSWLIB, _HNSW_CPP, _FAISS],
    },
}
