#!/usr/bin/env python3
"""Explicit process-local 3D-Fixer selection for the unified module CLI.

Examples, from the repository root with the core package available:
  python scripts/three_d_fixer_cli.py discover
  python scripts/three_d_fixer_cli.py serve observed_context_completion \
      --profile profiles/three-d-fixer.example.json --port 8138
  python scripts/three_d_fixer_cli.py register --registry outputs/services.json \
      --endpoint http://127.0.0.1:8138
  python scripts/three_d_fixer_cli.py run-stage TASK observed_context_completion \
      --profile profiles/three-d-fixer.example.json

Both workers and clients must opt in. No installed global entry point, default
module file or existing profile is changed. Discovery is not model readiness.
"""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from three_d_fixer_module import SPEC
from world_modeling.modules.geometry_completion import SPEC as STREAM_SPEC
from world_modeling.registry import ModuleRegistry


def activate(registry):
    """Explicit replacement, validated before changing the shared registry."""
    if SPEC.id in registry.specs:
        if registry.get(SPEC.id) != SPEC or STREAM_SPEC.id in registry.specs:
            raise ValueError("conflicting observed-context completion registration")
        return registry
    if registry.get(STREAM_SPEC.id) != STREAM_SPEC:
        raise ValueError("refusing to replace a nonstandard geometry completion module")
    ModuleRegistry([SPEC if spec.id == STREAM_SPEC.id else spec for spec in registry.modules])
    registry.unregister(STREAM_SPEC.id)
    registry.register(SPEC)
    return registry


def main(argv=None):
    from world_modeling import modules

    activate(modules.REGISTRY)
    modules.ORDERED_SPECS = modules.REGISTRY.modules
    # SPECS is the registry's live read-only mapping. Existing engine/services
    # references share the same registry object, so discovery agrees everywhere.
    from world_modeling.cli import main as unified_main

    return unified_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
