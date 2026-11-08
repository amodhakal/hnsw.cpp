from __future__ import annotations

from .hnsw_py import HNSW_PY
from .hnswlib import HNSWLIB

# Add new system instances here. They are registered with the config manager
# automatically and included by presets that use ``benchmarks = "all"``.
SYSTEMS = (HNSWLIB, HNSW_PY)
