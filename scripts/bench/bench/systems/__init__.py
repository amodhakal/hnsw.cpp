from __future__ import annotations

from .hnsw_cpp import HNSW_CPP
from .hnsw_py import HNSW_PY
from .hnswlib import HNSWLIB

# Add new system instances here. They are registered with the config manager
# automatically and included by presets that use ``benchmarks = "all"``.
SYSTEMS = (HNSWLIB, HNSW_CPP, HNSW_PY)
