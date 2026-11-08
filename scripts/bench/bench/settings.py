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


@dataclass
class OutputConfig:
    dir: str
    format: str


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
