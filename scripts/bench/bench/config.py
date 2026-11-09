from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .interface import BenchmarkSystem
from .presets import PRESETS
from .settings import (
    BenchmarkEntry,
    DatasetConfig,
    MetricsConfig,
    OutputConfig,
    RunConfig,
)
from .systems import SYSTEMS


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _set_path(target: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    current = target
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


class ConfigManager:
    """Registers benchmark-system instances and resolves Python presets.

    Benchmark systems are added with ``register_system``. Presets are plain
    dictionaries, so there is no TOML and no config-class inheritance.
    """

    def __init__(self) -> None:
        self._systems: dict[str, BenchmarkSystem] = {}
        self._presets: dict[str, dict] = {}

    def register_system(self, system: BenchmarkSystem) -> None:
        if system.name in self._systems:
            raise ValueError(f"Benchmark system already registered: {system.name}")
        self._systems[system.name] = system

    def register_preset(self, name: str, preset: dict) -> None:
        self._presets[name] = deepcopy(preset)

    def list_systems(self) -> list[str]:
        return list(self._systems)

    def list_presets(self) -> list[str]:
        return list(self._presets)

    def get_system(self, name: str) -> BenchmarkSystem:
        try:
            return self._systems[name]
        except KeyError:
            available = ", ".join(self._systems) or "<none>"
            raise KeyError(
                f"Unknown benchmark system '{name}'. Available: {available}"
            ) from None

    def _resolve_benchmarks(self, raw: Any) -> list[BenchmarkEntry]:
        raw_entries: list[Mapping[str, Any]]
        if raw == "all":
            raw_entries = [
                {
                    "library": system.name,
                    "label": system.label or system.name,
                    "params": {},
                }
                for system in self._systems.values()
            ]
        elif isinstance(raw, list):
            raw_entries = raw
        else:
            raise ValueError(
                'Preset "benchmarks" must be "all" or a list of entries'
            )

        entries: list[BenchmarkEntry] = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                raise ValueError("Each benchmark entry must be a dictionary")
            library = raw_entry.get("library")
            if not library:
                raise ValueError("Each benchmark entry needs a 'library' field")
            system = self.get_system(str(library))
            raw_params = raw_entry.get("params", {})
            if not isinstance(raw_params, Mapping):
                raise ValueError("Benchmark entry 'params' must be a dictionary")
            params = _deep_merge(dict(system.default_params), raw_params)
            entries.append(
                BenchmarkEntry(
                    library=system.name,
                    label=str(raw_entry.get("label") or system.label or system.name),
                    params=params,
                )
            )
        return entries

    def resolve(
        self,
        preset: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> RunConfig:
        try:
            raw = deepcopy(self._presets[preset])
        except KeyError:
            available = ", ".join(self._presets) or "<none>"
            raise KeyError(
                f"Unknown preset '{preset}'. Available: {available}"
            ) from None

        for path, value in (overrides or {}).items():
            _set_path(raw, path, value)

        dataset = DatasetConfig(**raw["dataset"])
        metrics = MetricsConfig(**raw["metrics"])
        output = OutputConfig(**raw["output"])
        output.format = output.format.lower().lstrip(".")

        if output.format not in {"png", "jpg", "jpeg"}:
            raise ValueError(
                f"Unsupported output format '{output.format}'. Use png, jpg, or jpeg."
            )
        if metrics.k < 1:
            raise ValueError("metrics.k must be at least 1")
        if metrics.max_ef < 1:
            raise ValueError("metrics.max_ef must be at least 1")
        if metrics.target_recall is not None and not (
            0.0 <= metrics.target_recall <= 1.0
        ):
            raise ValueError("metrics.target_recall must be between 0 and 1")

        return RunConfig(
            preset=preset,
            dataset=dataset,
            metrics=metrics,
            output=output,
            benchmarks=self._resolve_benchmarks(raw.get("benchmarks", "all")),
        )


config_manager = ConfigManager()

for _system in SYSTEMS:
    config_manager.register_system(_system)

for _name, _preset in PRESETS.items():
    config_manager.register_preset(_name, _preset)
