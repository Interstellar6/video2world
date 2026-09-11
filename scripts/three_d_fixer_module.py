"""Opt-in observed-context completion contract, not a default pipeline module.

Construct a registry without geometry_completion before registering SPEC. The
registry deliberately rejects installing both producers of completion roles.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.contracts import ModuleSpec, role

SPEC = ModuleSpec(
    "observed_context_completion", "3D-Fixer observed-context completion",
    ("assembled_object_views", "assembly_report", "component_masks", "isolated_object_ply", "cameras",
     "scene_depth", "scene_gaussian_ply", "lifting_report"),
    (role("completion_candidates", "generated", "application/json"),
     role("completed_object_meshes", "generated", "model/gltf-binary")),
    "learned_three_d_fixer",
    "Original scene RGB, geometrically verified mask and calibrated observed depth only. "
    "Explicit alternative to Stream3D completion; no generated orbit dependency. "
    "Publishes only after observed-correspondence and heldout reprojection validation.",
)
