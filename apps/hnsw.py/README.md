# hnsw.py

A pure-Python implementation of Hierarchical Navigable Small World (HNSW)
graphs for approximate nearest-neighbour search.

Only the standard library is used. Vector arithmetic relies on
`math.sumprod` (with a pure-Python fallback before Python 3.12) and
`math.dist`; the graph itself is plain Python lists, sets, and heaps.
Python 3.12+ is recommended for the faster `math.sumprod` path.

## Features

- Multi-layer HNSW graph with greedy descent and `ef`-bounded layer search
- Cosine, L2, and inner-product spaces
- Heuristic neighbour selection with bidirectional links and pruning
- Deterministic level assignment via a seed
- Single-threaded, no external dependencies

## Install

```bash
cd apps/hnsw.py
uv sync
```

## Usage

```python
from hnsw_py import HNSWIndex

index = HNSWIndex(
    dim=128,
    space="cosine",
    m=16,
    ef_construction=100,
    ef_search=50,
    seed=42,
)

index.add_items(vectors, labels=None)
index.set_ef(80)

labels, distances = index.search(query, k=10)
batch_labels, batch_distances = index.knn_query(queries, k=10)
```

`vectors` and `queries` are sequences of float sequences. If you have a NumPy
array, call `.tolist()` first; the library itself does not import NumPy.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `dim` | Vector dimensionality |
| `space` | `cosine`, `l2`, or `ip` |
| `m` | Neighbours per node on upper layers; layer 0 keeps `2 * m` |
| `ef_construction` | Candidate-list size while inserting |
| `ef_search` | Default candidate-list size while querying |
| `seed` | Seed for deterministic random levels |
| `heuristic` | Use the paper's neighbour-selection heuristic |

## Demo

```bash
uv run hnsw-py
```

## Benchmark

The library is registered with the VecSearch benchmark harness as `hnsw-py`.
From the repository root:

```bash
cd scripts/bench
uv sync
uv run vecbench run --preset fast
```

`fast` runs every registered system at 100k × 128 and is sized for an M1 Air
with 8 GB RAM. The 1M `default` preset also includes `hnsw.py`; expect it to be
much slower.

## Limitations

- Single-threaded; the GIL makes Python threads unhelpful here.
- No deletion or update operations.
- Python object overhead makes the index much larger than a native HNSW
  implementation.
- Not intended for million-scale indexes; use it as a readable reference and
  for small-to-medium experiments.
