"""Distance helpers for the pure-Python HNSW index.

Cosine and inner-product ranking only need the dot product, so they use
``math.sumprod`` on Python 3.12+ (with a pure-Python fallback earlier). L2 uses
``math.dist``. The C-backed helpers keep the Python-level graph code usable at
benchmark sizes.
"""

from __future__ import annotations

import math
from typing import Sequence

try:  # Python 3.12+
    from math import sumprod as _sumprod
except ImportError:  # pragma: no cover - fallback for Python 3.11

    def _sumprod(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(left * right for left, right in zip(a, b))


Vector = Sequence[float]


def normalize(vector: Vector) -> list[float]:
    """Return a unit-length copy of ``vector`` (zero vectors stay zero)."""
    norm = math.sqrt(_sumprod(vector, vector))
    if norm == 0.0:
        return [0.0] * len(vector)
    inverse = 1.0 / norm
    return [value * inverse for value in vector]


def cosine_distance(a: Vector, b: Vector) -> float:
    """Cosine distance for already-normalized vectors."""
    return 1.0 - _sumprod(a, b)


def inner_product_distance(a: Vector, b: Vector) -> float:
    """Inner-product distance (larger dot product means closer)."""
    return 1.0 - _sumprod(a, b)


def l2_distance(a: Vector, b: Vector) -> float:
    """Squared Euclidean distance."""
    distance = math.dist(a, b)
    return distance * distance


DISTANCES = {
    "cosine": cosine_distance,
    "l2": l2_distance,
    "ip": inner_product_distance,
}
