from __future__ import annotations

import csv
import json
import math
import platform
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from .runner import RunResult, _params_label


def _unique_stem(directory: Path, timestamp: str) -> str:
    stem = timestamp
    suffix = 1
    while any(directory.glob(f"{stem}.*")):
        stem = f"{timestamp}_{suffix}"
        suffix += 1
    return stem


def _slugify_label(label: str | None) -> str:
    if not label:
        return ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")
    return slug[:60].rstrip("-._")


def _extension(fmt: str) -> str:
    return "jpg" if fmt in {"jpg", "jpeg"} else "png"


def write_outputs(result: RunResult) -> None:
    output_dir = Path(result.config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = result.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    slug = _slugify_label(result.config.output.label)
    base_stem = f"{timestamp}_{slug}" if slug else timestamp
    stem = _unique_stem(output_dir, base_stem)
    extension = _extension(result.config.output.format)

    result.graph_path = output_dir / f"{stem}.{extension}"
    result.latest_graph_path = output_dir / f"most_recent.{extension}"
    result.csv_path = output_dir / f"{stem}.csv"
    result.json_path = output_dir / f"{stem}.json"

    _write_csv(result)
    _write_json(result)
    _write_graph(result, result.latest_graph_path)
    _print_summary(result)


def _write_csv(result: RunResult) -> None:
    assert result.csv_path is not None
    with result.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "library",
                "sweep",
                "recall",
                "qps",
                "avg_ms",
                "p50_ms",
                "p95_ms",
                "p99_ms",
                "build_seconds",
                "index_bytes",
            ]
        )
        for series in result.series:
            for point in series.points:
                writer.writerow(
                    [
                        series.label,
                        series.library,
                        json.dumps(point.params, sort_keys=True),
                        point.recall,
                        point.qps,
                        point.avg_ms,
                        point.p50_ms,
                        point.p95_ms,
                        point.p99_ms,
                        series.build_seconds,
                        series.index_bytes,
                    ]
                )


def _write_json(result: RunResult) -> None:
    assert result.json_path is not None
    payload = {
        "preset": result.preset,
        "timestamp": result.timestamp.isoformat(),
        "dataset": {
            "n": result.n,
            "dim": result.dim,
            "queries": result.queries,
        },
        "groundtruth_seconds": result.groundtruth_seconds,
        "config": asdict(result.config),
        "series": [asdict(series) for series in result.series],
        "outputs": {
            "graph": str(result.graph_path),
            "latest_graph": str(result.latest_graph_path),
            "csv": str(result.csv_path),
            "json": str(result.json_path),
        },
    }
    with result.json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def _detect_hardware() -> str:
    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                check=False,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            pass
        else:
            name = completed.stdout.strip()
            if name:
                return name
    return platform.processor() or platform.machine() or "unknown"


def _recall_bounds(result: RunResult) -> tuple[float, float]:
    recalls = [
        point.recall
        for series in result.series
        for point in series.points
    ]
    if not recalls:
        return 0.0, 1.0
    lower = min(0.6, math.floor(min(recalls) * 10) / 10)
    return max(0.0, lower), 1.0


def _write_graph(result: RunResult, latest_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert result.graph_path is not None
    figure, axis = plt.subplots(figsize=(10.0, 6.5))

    for series in result.series:
        ordered = sorted(series.points, key=lambda point: point.qps)
        qps = [max(point.qps, 1e-12) for point in ordered]
        recall = [point.recall for point in ordered]
        axis.plot(
            qps,
            recall,
            marker="o",
            linewidth=2.2,
            markersize=7.0,
            label=series.label,
        )

    k = result.config.metrics.k
    hardware = result.config.output.hardware or _detect_hardware()
    axis.set_xscale("log")
    axis.set_xlabel("Queries per second (QPS)", fontsize=14)
    axis.set_ylabel(f"Recall@{k}", fontsize=14)
    axis.set_ylim(*_recall_bounds(result))
    axis.tick_params(axis="both", which="major", labelsize=12)
    axis.tick_params(axis="both", which="minor", labelsize=11)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(fontsize=12, framealpha=0.9, loc="lower left")
    axis.annotate(
        "faster →",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=13,
        color="0.35",
    )
    axis.set_title(
        f"{_format_count(result.n)} vectors, {result.dim}-dim, {hardware}, k={k}",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    figure.tight_layout()
    figure.savefig(result.graph_path, dpi=160, facecolor="white")
    figure.savefig(latest_path, dpi=160, facecolor="white")
    plt.close(figure)


def _print_summary(result: RunResult) -> None:
    k = result.config.metrics.k
    print("\n" + "=" * 78)
    print(f"Results written to {result.graph_path}")
    print(f"Latest graph: {result.latest_graph_path}")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")
    print("=" * 78)

    for series in result.series:
        print(f"\n{series.label} ({series.library})")
        print(
            f"  build={series.build_seconds:.2f}s "
            f"index={series.index_bytes if series.index_bytes is not None else 'n/a'} bytes"
        )
        header = (
            f"  {'sweep':<20} {'recall@' + str(k):>10} "
            f"{'QPS':>10} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for point in series.points:
            print(
                f"  {_params_label(point.params):<20} "
                f"{point.recall:>10.4f} "
                f"{point.qps:>10.1f} "
                f"{point.p50_ms:>9.3f} "
                f"{point.p95_ms:>9.3f} "
                f"{point.p99_ms:>9.3f}"
            )
    print()
