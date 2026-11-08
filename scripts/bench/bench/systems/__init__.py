from __future__ import annotations

from .hnswlib import HNSWLIB

# Add new system instances here. They are registered with the config manager
# automatically and included by presets that use ``benchmarks = "all"``.
SYSTEMS = (HNSWLIB,)
