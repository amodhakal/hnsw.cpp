from __future__ import annotations

# Plain Python presets. No TOML, no inheritance.
#
# ``"benchmarks": "all"`` expands to one entry per system registered with the
# config manager, using each system's default_params.
#
# To compare specific configurations, replace "all" with an explicit list:
#   "benchmarks": [
#       {"library": "hnswlib", "label": "hnswlib M=16", "params": {"m": 16}},
#       {"library": "hnswlib", "label": "hnswlib M=32", "params": {"m": 32}},
#   ]

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
        "metrics": {"k": 10, "reps": 5, "warmup": 100},
        "output": {"dir": "results", "format": "png"},
        "benchmarks": "all",
    },
    "smoke": {
        "dataset": {
            "n": 100_000,
            "dim": 128,
            "queries": 2_000,
            "clusters": 100,
            "cluster_std": 0.05,
            "seed": 42,
        },
        "metrics": {"k": 10, "reps": 1, "warmup": 20},
        "output": {"dir": "results", "format": "png"},
        "benchmarks": [
            {
                "library": "hnswlib",
                "label": "hnswlib M=16",
                "params": {"sweep": {"ef": [10, 40, 160]}},
            }
        ],
    },
    "compare_m": {
        "dataset": {
            "n": 1_000_000,
            "dim": 128,
            "queries": 10_000,
            "clusters": 1_000,
            "cluster_std": 0.05,
            "seed": 42,
        },
        "metrics": {"k": 10, "reps": 5, "warmup": 100},
        "output": {"dir": "results", "format": "png"},
        "benchmarks": [
            {
                "library": "hnswlib",
                "label": "hnswlib M=16",
                "params": {"m": 16, "ef_construction": 200},
            },
            {
                "library": "hnswlib",
                "label": "hnswlib M=32",
                "params": {"m": 32, "ef_construction": 200},
            },
        ],
    },
}
