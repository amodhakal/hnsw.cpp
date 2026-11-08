from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class BenchmarkSystem:
    """The interface a vector-search library must implement.

    This is a value, not a base class: no inheritance is involved. Create one
    instance per library and hand it to ``ConfigManager.register_system``.
    """

    name: str
    build: Callable[[np.ndarray, Mapping[str, Any]], tuple[Any, float]]
    set_query_params: Callable[[Any, Mapping[str, Any]], None]
    search_batch: Callable[[Any, np.ndarray, int, int], np.ndarray]
    search_one: Callable[[Any, np.ndarray, int], np.ndarray]
    index_size_bytes: Callable[[Any], int | None]
    default_params: Mapping[str, Any] = field(default_factory=dict)
    label: str | None = None
    close: Callable[[Any], None] | None = None
