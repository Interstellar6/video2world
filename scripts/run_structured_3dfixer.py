#!/usr/bin/env python3
"""Run a pinned 3D-Fixer runner on structured instances from the current scene."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-runner", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--instance-manifest", type=Path, required=True)
    parser.add_argument("--preview-resolution", type=int, default=256)
    parser.add_argument("--preview-frames", type=int, default=6)
    parser.add_argument(
        "--skip-preview-render",
        action="store_true",
        help=(
            "Bypass only the upstream Gaussian preview renderer so the official "
            "texture export can continue. The emitted contact sheet is a placeholder "
            "and must not be used as visual evidence."
        ),
    )
    return parser.parse_args()


def load_runner(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("verified_3dfixer_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_offline_dinov2_adapter() -> str:
    import torch

    torch_home = Path(os.environ["TORCH_HOME"]).resolve()
    local_repo = torch_home / "hub" / "facebookresearch_dinov2_main"
    if not local_repo.is_dir():
        raise FileNotFoundError(local_repo)
    original_load = torch.hub.load

    def local_load(repo_or_dir: Any, *args: Any, **kwargs: Any) -> Any:
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs.pop("source", None)
            return original_load(str(local_repo), *args, source="local", **kwargs)
        return original_load(repo_or_dir, *args, **kwargs)

    torch.hub.load = local_load
    return str(local_repo)


def install_bounded_preview(
    runner: ModuleType,
    *,
    resolution: int,
    num_frames: int,
    skip_render: bool = False,
) -> None:
    if not 64 <= resolution <= 512:
        raise ValueError("preview resolution must be in [64, 512]")
    if not 1 <= num_frames <= 12:
        raise ValueError("preview frames must be in [1, 12]")
    original = runner.render_utils.render_video

    def bounded_render(*args: Any, **kwargs: Any) -> Any:
        if skip_render:
            import numpy as np

            placeholder = np.full((resolution, resolution, 3), 255, dtype=np.uint8)
            return {"color": [placeholder]}
        kwargs["resolution"] = resolution
        kwargs["num_frames"] = num_frames
        return original(*args, **kwargs)

    runner.render_utils.render_video = bounded_render


def configure_instances(raw_instances: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError("instance manifest requires a nonempty instances list")
    configured: list[dict[str, Any]] = []
    for index, item in enumerate(raw_instances):
        if not isinstance(item, dict):
            raise ValueError(f"instances[{index}] must be an object")
        object_id = str(item.get("object_id") or "")
        semantic_boundary = str(item.get("semantic_boundary") or "")
        if not object_id or not semantic_boundary:
            raise ValueError(f"instances[{index}] requires object_id and semantic_boundary")
        frame_id = int(item["frame_id"])
        mask = Path(item["mask_path"]).expanduser().resolve()
        if not mask.is_file():
            raise FileNotFoundError(mask)
        configured.append(
            {
                "name": object_id,
                "frame": f"{frame_id:06d}",
                "mask": mask,
                "semantic_boundary": semantic_boundary,
            }
        )
    return tuple(configured)


def main() -> int:
    args = parse_args()
    if "EXPERIMENT_ROOT" not in os.environ:
        raise RuntimeError("EXPERIMENT_ROOT must point to the 3D-Fixer source/cache root")
    source_root = args.source_root.expanduser().resolve()
    expected_images = source_root / "data_scannet" / "bedroom_4" / "color"
    if not expected_images.is_dir():
        raise FileNotFoundError(expected_images)
    manifest_path = args.instance_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runner = load_runner(args.base_runner.expanduser().resolve())
    offline_dinov2_repo = install_offline_dinov2_adapter()
    install_bounded_preview(
        runner,
        resolution=args.preview_resolution,
        num_frames=args.preview_frames,
        skip_render=args.skip_preview_render,
    )
    runner.HOLI_ROOT = source_root
    runner.OUTPUT_ROOT = args.output_root.expanduser().resolve()
    runner.INSTANCES = configure_instances(manifest.get("instances"))
    runner.main()

    result_path = runner.OUTPUT_ROOT / "run_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "structured_instance_manifest": str(manifest_path),
            "adapter_reuse": str(args.base_runner.expanduser().resolve()),
            "current_source_root": str(source_root),
            "offline_dinov2_repo": offline_dinov2_repo,
            "bounded_preview": {
                "resolution": args.preview_resolution,
                "num_frames": args.preview_frames,
                "affects_generation": False,
                "render_status": (
                    "skipped_upstream_renderer_allocation_bug"
                    if args.skip_preview_render
                    else "rendered"
                ),
                "contact_sheet_evidence": (
                    "placeholder_not_visual_evidence"
                    if args.skip_preview_render
                    else "rendered_preview"
                ),
            },
            "claim_boundary": (
                "3D-Fixer is an in-place single-view completion backend. Outputs use "
                "normalized single-view MoGe scene coordinates and are not global "
                "room-aligned assets, generic mesh repair evidence, or collision meshes."
            ),
        }
    )
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
