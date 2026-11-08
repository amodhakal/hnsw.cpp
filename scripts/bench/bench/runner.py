from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import ConfigManager
from .datasets import Dataset, make_synthetic
from .groundtruth import exact_knn
from .metrics import latency_percentiles_ms, median, recall_at_k
from .settings import BenchmarkEntry, RunConfig


@dataclass
class QueryPoint:
    params: dict
    recall: float
    qps: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class Series:
    label: str
    library: str
    build_seconds: float
    index_bytes: int | None
    points: list[QueryPoint] = field(default_factory=list)


@dataclass
class RunResult:
    preset: str
    timestamp: datetime
    n: int
    dim: int
    queries: int
    groundtruth_seconds: float
    series: list[Series]
    config: RunConfig
    graph_path: Path | None = None
    latest_graph_path: Path | None = None
    csv_path: Path | None = None
    json_path: Path | None = None


def _sweep_points(sweep: Any) -> list[dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = list(sweep)
    values = [
        value if isinstance(value, list) else [value]
        for value in sweep.values()
    ]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _benchmark_metric(entry: BenchmarkEntry) -> str:
    return str(entry.params.get("space", "cosine"))


class BenchmarkRunner:
    """Runs every configured benchmark and produces one combined graph."""

    def __init__(self, config: RunConfig, manager: ConfigManager) -> None:
        self.config = config
        self.manager = manager

    def _dataset(self) -> Dataset:
        cfg = self.config.dataset
        return make_synthetic(
            n=cfg.n,
            dim=cfg.dim,
            queries=cfg.queries,
            clusters=cfg.clusters,
            cluster_std=cfg.cluster_std,
            seed=cfg.seed,
        )

    def _groundtruth(
        self,
        data: Dataset,
        metric: str,
        cache: dict[str, np.ndarray],
    ) -> np.ndarray:
        if metric not in cache:
            started = time.perf_counter()
            cache[metric] = exact_knn(
                data.train,
                data.queries,
                self.config.metrics.k,
                metric,
            )
            elapsed = time.perf_counter() - started
            print(
                f"[ground truth] metric={metric} "
                f"queries={data.queries.shape[0]} "
                f"time={elapsed:.2f}s"
            )
        return cache[metric]

    def _warmup(
        self,
        system: Any,
        index: Any,
        queries: np.ndarray,
        k: int,
        mode: str,
        threads: int,
        count: int,
    ) -> None:
        if count <= 0:
            return
        subset = queries[: min(count, len(queries))]
        if mode == "batch":
            system.search_batch(index, subset, k, threads)
        else:
            for query in subset:
                system.search_one(index, query, k)

    def _measure(
        self,
        system: Any,
        index: Any,
        queries: np.ndarray,
        k: int,
        mode: str,
        threads: int,
    ) -> tuple[np.ndarray, float, list[float]]:
        if mode == "batch":
            started = time.perf_counter()
            labels = system.search_batch(index, queries, k, threads)
            elapsed = time.perf_counter() - started
            return labels, elapsed, []

        latencies_ms: list[float] = []
        labels = np.empty((len(queries), k), dtype=np.int64)
        started = time.perf_counter()
        for row, query in enumerate(queries):
            query_started = time.perf_counter()
            labels[row] = system.search_one(index, query, k)
            latencies_ms.append((time.perf_counter() - query_started) * 1000.0)
        elapsed = time.perf_counter() - started
        return labels, elapsed, latencies_ms

    def run(self) -> RunResult:
        data = self._dataset()
        print(
            f"[dataset] train={data.train.shape[0]}x{data.train.shape[1]} "
            f"queries={data.queries.shape[0]}"
        )

        gt_cache: dict[str, np.ndarray] = {}
        gt_seconds = 0.0
        for metric in {_benchmark_metric(entry) for entry in self.config.benchmarks}:
            started = time.perf_counter()
            self._groundtruth(data, metric, gt_cache)
            gt_seconds += time.perf_counter() - started

        series: list[Series] = []
        k = self.config.metrics.k

        for entry in self.config.benchmarks:
            system = self.manager.get_system(entry.library)
            metric = _benchmark_metric(entry)
            truth = gt_cache[metric]
            params = entry.params
            mode = str(params.get("query_mode", "sequential"))
            threads = int(params.get("query_threads", 1))

            print(f"\n[benchmark] {entry.label} ({entry.library}, {metric})")
            index, build_seconds = system.build(data.train, params)
            index_bytes = system.index_size_bytes(index)
            print(
                f"  build={build_seconds:.2f}s "
                f"index={_format_bytes(index_bytes)}"
            )

            result = Series(
                label=entry.label,
                library=entry.library,
                build_seconds=build_seconds,
                index_bytes=index_bytes,
            )

            try:
                for point in _sweep_points(params.get("sweep", {})):
                    system.set_query_params(index, point)
                    self._warmup(
                        system,
                        index,
                        data.queries,
                        k,
                        mode,
                        threads,
                        self.config.metrics.warmup,
                    )

                    rep_qps: list[float] = []
                    rep_latencies: list[float] = []
                    labels: np.ndarray | None = None
                    for _ in range(max(1, self.config.metrics.reps)):
                        rep_labels, elapsed, latencies = self._measure(
                            system,
                            index,
                            data.queries,
                            k,
                            mode,
                            threads,
                        )
                        if labels is None:
                            labels = rep_labels
                        rep_qps.append(len(data.queries) / max(elapsed, 1e-12))
                        rep_latencies.extend(latencies)

                    assert labels is not None
                    if rep_latencies:
                        avg_ms = float(np.mean(rep_latencies))
                        p50, p95, p99 = latency_percentiles_ms(rep_latencies)
                    else:
                        avg_ms = p50 = p95 = p99 = float("nan")

                    point_result = QueryPoint(
                        params=dict(point),
                        recall=recall_at_k(labels, truth, k),
                        qps=median(rep_qps),
                        avg_ms=avg_ms,
                        p50_ms=p50,
                        p95_ms=p95,
                        p99_ms=p99,
                    )
                    result.points.append(point_result)
                    print(
                        f"  {_params_label(point)}: "
                        f"recall@{k}={point_result.recall:.4f} "
                        f"qps={point_result.qps:.1f} "
                        f"p50={point_result.p50_ms:.3f}ms"
                    )
            finally:
                if system.close is not None:
                    system.close(index)

            series.append(result)

        run_result = RunResult(
            preset=self.config.preset,
            timestamp=datetime.now(),
            n=data.train.shape[0],
            dim=data.train.shape[1],
            queries=data.queries.shape[0],
            groundtruth_seconds=gt_seconds,
            series=series,
            config=self.config,
        )

        from .report import write_outputs

        write_outputs(run_result)
        return run_result


def _params_label(params: Mapping[str, Any]) -> str:
    if not params:
        return "default"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"
