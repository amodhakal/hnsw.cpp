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
        "metrics": {
            "k": 10,
            "reps": 5,
            "warmup": 100,
            "target_recall": 0.99,
            "max_ef": 2048,
        },
        "output": {"dir": "results", "format": "png"},
        "benchmarks": "all",
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
        "benchmarks": "all",
    },
}
