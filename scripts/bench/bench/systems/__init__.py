from __future__ import annotations

from .faiss import FAISS
from .hnsw_cpp import HNSW_CPP
from .hnswlib import HNSWLIB

# Add new system instances here. They are registered with the config manager
# automatically and included by presets that use ``benchmarks = "all"``.
SYSTEMS = (HNSWLIB, HNSW_CPP, FAISS)
