from __future__ import annotations

import statistics

import numpy as np


def recall_at_k(
    predicted: np.ndarray,
    truth: np.ndarray,
    k: int,
) -> float:
    if predicted.shape[0] != truth.shape[0]:
        raise ValueError("Predicted and truth arrays must have the same length")
    hits = 0
    for predicted_row, truth_row in zip(predicted, truth):
        hits += len(set(predicted_row[:k].tolist()) & set(truth_row[:k].tolist()))
    return hits / float(predicted.shape[0] * k)


def latency_percentiles_ms(latencies: list[float]) -> tuple[float, float, float]:
    if not latencies:
        return (float("nan"), float("nan"), float("nan"))
    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    return (float(p50), float(p95), float(p99))


def median(values: list[float]) -> float:
    return float(statistics.median(values))
