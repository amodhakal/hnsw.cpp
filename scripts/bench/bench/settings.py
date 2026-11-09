from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    n: int
    dim: int
    queries: int
    clusters: int
    cluster_std: float
    seed: int


@dataclass
class MetricsConfig:
    k: int
    reps: int
    warmup: int
    target_recall: float | None = None
    max_ef: int = 512


@dataclass
class OutputConfig:
    dir: str
    format: str
    label: str | None = None


@dataclass
class BenchmarkEntry:
    library: str
    label: str
    params: dict


@dataclass
class RunConfig:
    preset: str
    dataset: DatasetConfig
    metrics: MetricsConfig
    output: OutputConfig
    benchmarks: list[BenchmarkEntry]
