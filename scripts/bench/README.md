# scripts/bench

Local **recall vs QPS** benchmarks for vector search libraries. See the
[VecSearch README](../../README.md) for the project overview and the latest
graph.

A `BenchmarkSystem` value describes a library's operations, an instance
is registered with the `ConfigManager`, and a `BenchmarkRunner` runs every
configured benchmark and writes one combined graph.

## Run

```bash
cd scripts/bench
uv sync
uv run vecbench run --preset default
```

The graph is written to `results/` using the local date and time, for example:

```
results/2026-11-08_17-30-45.png
results/2026-11-08_17-30-45.csv
results/2026-11-08_17-30-45.json
```

Each run also refreshes the stable README image
`results/most_recent.png` (or `results/most_recent.jpg` with `--format jpg`).
Dated artifacts are kept for history.

Use `--format jpg` for a JPEG:

```bash
uv run vecbench run --preset default --format jpg
```

## Commands

```bash
uv run vecbench list
uv run vecbench show-config --preset default
uv run vecbench run --preset fast
uv run vecbench run --preset default --n 50000 --queries 500
uv run vecbench run --preset default --ef 10,20,40,80
uv run vecbench run --preset default --set dataset.n=20000
```

## Presets

Presets are plain dictionaries in `bench/presets.py`:

- `default` — 1M × 128, 10k queries, every registered system
- `fast` — 100k × 128, 1k queries, every registered system

`"benchmarks": "all"` expands to one entry per registered system. The graph
contains one line per configured entry.

## Adding a vector search library

1. Add the library under `apps/<name>/` for an in-repo implementation, or
   `vendors/<name>/` for an external/reference library used as a control.
   In-repo libraries also need a path dependency in `scripts/bench/pyproject.toml`.
2. Create `bench/systems/<name>.py` with functions for build, query-parameter
   setup, one-query search, batch search, and index size.
3. Create a `BenchmarkSystem` instance with those functions and
   `default_params`.
4. Add the instance to `SYSTEMS` in `bench/systems/__init__.py`.

It is then registered automatically by `bench/config.py`, and presets using
`"benchmarks": "all"` include it in the next graph. No changes to the runner,
report, metrics, or CLI are needed.

## How measurements work

- Synthetic clustered Gaussian vectors are generated once per run.
- Exact ground truth is computed in chunked numpy using the same metric.
- Each configured benchmark builds its own index, then sweeps its query
  parameters (`ef` for hnswlib and hnsw.py).
- Sequential mode times one query at a time and reports p50/p95/p99 latency.
- Batch mode calls the library's batch query path for throughput.
- QPS is the median over `reps` runs.

The `fast` preset is sized for an M1 Air with 8 GB RAM and completes in under
five minutes, including the pure-Python index. The 1M `default` preset includes
every registered system and is much heavier.
