from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict
from typing import Any

from .config import config_manager
from .runner import BenchmarkRunner


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _parse_set(values: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--set expects PATH=VALUE, got: {item}")
        path, raw_value = item.split("=", 1)
        overrides[path] = _parse_value(raw_value)
    return overrides


def _cmd_run(args: argparse.Namespace) -> None:
    overrides = _parse_set(args.set)

    if args.format is not None:
        overrides["output.format"] = args.format
    if args.output_dir is not None:
        overrides["output.dir"] = args.output_dir
    if args.n is not None:
        overrides["dataset.n"] = args.n
    if args.dim is not None:
        overrides["dataset.dim"] = args.dim
    if args.queries is not None:
        overrides["dataset.queries"] = args.queries
    if args.k is not None:
        overrides["metrics.k"] = args.k
    if args.reps is not None:
        overrides["metrics.reps"] = args.reps
    if args.warmup is not None:
        overrides["metrics.warmup"] = args.warmup

    config = config_manager.resolve(args.preset, overrides)

    if args.ef is not None:
        ef_values = [int(value) for value in args.ef.split(",") if value.strip()]
        if not ef_values:
            raise SystemExit("--ef needs at least one value")
        for entry in config.benchmarks:
            entry.params.setdefault("sweep", {})["ef"] = ef_values

    runner = BenchmarkRunner(config, config_manager)
    runner.run()


def _cmd_list(_: argparse.Namespace) -> None:
    print("Benchmark systems:")
    for name in config_manager.list_systems():
        system = config_manager.get_system(name)
        print(f"  {name:<16} {system.label or ''}")
    print("\nPresets:")
    for name in config_manager.list_presets():
        print(f"  {name}")


def _cmd_show_config(args: argparse.Namespace) -> None:
    config = config_manager.resolve(args.preset, _parse_set(args.set))
    print(json.dumps(asdict(config), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vecbench",
        description="Recall vs QPS benchmarks for vector search libraries",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark preset")
    run_parser.add_argument(
        "--preset",
        default="default",
        help="Python preset name (default: default)",
    )
    run_parser.add_argument(
        "--set",
        action="append",
        metavar="PATH=VALUE",
        help="Override a preset value, e.g. --set dataset.n=50000",
    )
    run_parser.add_argument("--format", choices=["png", "jpg", "jpeg"])
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--n", type=int)
    run_parser.add_argument("--dim", type=int)
    run_parser.add_argument("--queries", type=int)
    run_parser.add_argument("--k", type=int)
    run_parser.add_argument("--reps", type=int)
    run_parser.add_argument("--warmup", type=int)
    run_parser.add_argument(
        "--ef",
        help="Comma-separated ef values applied to every benchmark entry",
    )
    run_parser.set_defaults(func=_cmd_run)

    list_parser = subparsers.add_parser("list", help="List systems and presets")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser(
        "show-config", help="Print the fully resolved preset"
    )
    show_parser.add_argument("--preset", default="default")
    show_parser.add_argument("--set", action="append", metavar="PATH=VALUE")
    show_parser.set_defaults(func=_cmd_show_config)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
