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
uv run vecbench run --preset smoke
uv run vecbench run --preset default --n 50000 --queries 500
uv run vecbench run --preset default --ef 10,20,40,80
uv run vecbench run --preset default --set dataset.n=20000
```

## Presets

Presets are plain dictionaries in `bench/presets.py`:

- `default` — 100k × 128, 1k queries, every registered system
- `smoke` — 10k × 128, 200 queries, quick validation
- `compare_m` — two hnswlib runs (`M=16` and `M=32`)

`"benchmarks": "all"` expands to one entry per registered system. The graph
contains one line per configured entry.

## Adding a vector search library

1. Create `bench/systems/<name>.py` with functions for build, query-parameter
   setup, one-query search, batch search, and index size.
2. Create a `BenchmarkSystem` instance with those functions and
   `default_params`.
3. Add the instance to `SYSTEMS` in `bench/systems/__init__.py`.

It is then registered automatically by `bench/config.py`, and presets using
`"benchmarks": "all"` include it in the next graph. No changes to the runner,
report, metrics, or CLI are needed.

## How measurements work

- Synthetic clustered Gaussian vectors are generated once per run.
- Exact ground truth is computed in chunked numpy using the same metric.
- Each configured benchmark builds its own index, then sweeps its query
  parameters (`ef` for hnswlib).
- Sequential mode times one query at a time and reports p50/p95/p99 latency.
- Batch mode uses the library's multi-threaded batch query path for throughput.
- QPS is the median over `reps` runs.

Defaults are sized for an M1 Air with 8 GB RAM. The 100k default completes in
roughly 15 seconds on that machine; raise `--n` for a longer, heavier run.
