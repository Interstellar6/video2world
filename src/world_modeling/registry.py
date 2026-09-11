"""Explicit module registration and discovery, independent of model runtimes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from types import MappingProxyType

from .contracts import ModuleSpec


def contract_digest(spec: ModuleSpec) -> str:
    return hashlib.sha256(json.dumps(spec.as_dict(), sort_keys=True).encode()).hexdigest()


class ModuleRegistry:
    def __init__(self, modules: Iterable[ModuleSpec] = ()):
        self._modules: dict[str, ModuleSpec] = {}
        for spec in modules:
            self.register(spec)

    def register(self, spec: ModuleSpec) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.id):
            raise ValueError(f"invalid module id: {spec.id}")
        if spec.id in self._modules:
            raise ValueError(f"module already registered: {spec.id}")
        if len(set(spec.output_names)) != len(spec.output_names):
            raise ValueError(f"duplicate output roles: {spec.id}")
        owners = {name for module in self._modules.values() for name in module.output_names}
        if owners.intersection(spec.output_names):
            raise ValueError(f"artifact role already has a producer: {spec.id}")
        self._modules[spec.id] = spec

    def get(self, module_id: str) -> ModuleSpec:
        return self._modules[module_id]

    def unregister(self, module_id: str) -> ModuleSpec:
        return self._modules.pop(module_id)

    def load_plugins(self) -> None:
        from importlib.metadata import entry_points

        points = entry_points()
        selected = points.select(group="world_modeling.modules") if hasattr(points, "select") else points.get("world_modeling.modules", [])
        for entry in sorted(selected, key=lambda item: item.name):
            loaded = entry.load()
            specs = loaded() if callable(loaded) else loaded
            for spec in (specs,) if isinstance(specs, ModuleSpec) else specs:
                if not isinstance(spec, ModuleSpec):
                    raise ValueError(f"module plugin {entry.name} must expose ModuleSpec instances")
                self.register(spec)

    def ordered(self, module_ids: Iterable[str], initial_roles: Iterable[str] = ("source_media",)) -> tuple[ModuleSpec, ...]:
        ids = list(module_ids)
        if not ids:
            raise ValueError("pipeline must contain at least one module")
        if len(ids) != len(set(ids)):
            raise ValueError("pipeline contains duplicate module IDs")
        pending = [self.get(name) for name in ids]
        available = set(initial_roles)
        result = []
        while pending:
            ready = next((spec for spec in pending if set(spec.inputs) <= available), None)
            if ready is None:
                missing = {spec.id: sorted(set(spec.inputs) - available) for spec in pending}
                raise ValueError(f"pipeline has missing producers or a dependency cycle: {missing}")
            result.append(ready)
            available.update(ready.output_names)
            pending.remove(ready)
        return tuple(result)

    @property
    def modules(self) -> tuple[ModuleSpec, ...]:
        return tuple(self._modules.values())

    @property
    def specs(self):
        return MappingProxyType(self._modules)

    def discover(self) -> list[dict]:
        return [{**spec.as_dict(), "contract_sha256": contract_digest(spec)} for spec in self.modules]
