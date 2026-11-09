# hnsw.cpp

A C++ implementation of Hierarchical Navigable Small World (HNSW) graphs for
approximate nearest-neighbour search.

It is a direct port of [`apps/hnsw.py`](../hnsw.py): the graph construction and
search follow Malkov & Yashunin, "Efficient and robust approximate nearest
neighbor search using HNSW graphs" (2016). The core uses only the C++ standard
library; Python bindings are provided with pybind11.

## Features

- Multi-layer HNSW graph with greedy descent and `ef`-bounded layer search
- Cosine, L2, and inner-product spaces
- Heuristic neighbour selection with bidirectional links and pruning
- Deterministic level assignment via a seed
- Single-threaded, no runtime dependencies
- `int16` vector storage with per-vector scaling
- Epoch-based or Bloom-filter visited sets during layer search

## Build

Building requires CMake 3.18+ and a C++17 compiler. The benchmark harness
builds the extension automatically:

```bash
cd scripts/bench
uv sync
```

After that, editing files under `apps/hnsw.cpp/src/` and running
`uv run vecbench run ...` rebuilds the extension on import. No manual rebuild
step is needed.

To build the standalone C++ demo and tests:

```bash
cmake -S apps/hnsw.cpp -B apps/hnsw.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build apps/hnsw.cpp/build
ctest --test-dir apps/hnsw.cpp/build --output-on-failure
```

## Usage

### C++

```cpp
#include "hnsw/hnsw.hpp"

hnsw::Index index(hnsw::Config{
    128,
    hnsw::Space::Cosine,
    16,
    100,
    50,
    42,
    true,
});

index.add_items(vectors.data(), count);
index.set_ef(80);

hnsw::SearchResult result = index.search(query.data(), 10);
```

`vectors` is a flat `float` array of `count * dim` values. Labels default to
insertion indices; pass a `std::uint64_t*` to `add_items` for custom labels.

### Python

```python
from hnsw_cpp import HNSWIndex

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

`vectors` and `queries` are NumPy arrays. They are converted to contiguous
`float32`; `knn_query` returns two 2-D arrays and is the fast batch path used by
the benchmark.

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
| `visited_mode` | `epoch` for exact per-search marks, `bloom` for a compact probabilistic set |
| `alpha` | Pruning slack for neighbour selection; `1.0` disables slack |

## Demo

```bash
./apps/hnsw.cpp/build/hnsw_demo
```

## Benchmark

The library is registered with the VecSearch benchmark harness as `hnsw-cpp`.
From the repository root:

```bash
cd scripts/bench
uv sync
uv run vecbench run --preset fast
```

`fast` runs every registered system at 80k × 128. The 1M `default` preset also
includes `hnsw.cpp`.

## Limitations

- Single-threaded.
- No deletion or update operations.
- No persistence/serialization.
- Vectors are stored as 16-bit integers with a per-vector scale, so distances
  are approximate even before graph search.
- The graph uses readable nested containers rather than a compact adjacency
  layout, and is intended as a reference implementation rather than a
  SIMD-optimized library.
