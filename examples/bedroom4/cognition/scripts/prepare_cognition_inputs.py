#!/usr/bin/env python3
"""Archive real cognition evidence and prepare VLM-friendly derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

EMBODIED_ROOT = Path("/root/autodl-tmp/embodiedgen_v2_bedroom4_auto_completion_20260714/run/input")
EMBODIED_OBJECTS = [
    ("sam3_nightstand_01", "nightstand"),
    ("sam3_nightstand_02", "nightstand"),
    ("sam3_plant_01", "plant"),
    ("sam3_plant_02", "plant"),
    ("sam3_plant_03", "plant"),
]
PILLOW_ROOT = Path(
    "/data/design/zyx/workspace/video2world_runs/bedroom_4_pillow_delta_20260716/"
    "refinements/top3_score090_erode5_depth005_minvotes2"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(
    path: Path,
    source_path: Path | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    image = Image.open(path)
    record = {
        "path": str(path),
        "source_path": str(source_path) if source_path else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
    }
    if artifact_root is not None:
        record["artifact_relative_path"] = str(path.relative_to(artifact_root))
    return record


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def composite_rgba(source: Path, archived: Path, output: Path) -> dict[str, Any]:
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, archived)
    rgba = Image.open(archived).convert("RGBA")
    background = Image.new("RGBA", rgba.size, (238, 238, 238, 255))
    composite = Image.alpha_composite(background, rgba).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    composite.save(output, format="PNG", optimize=True)
    alpha = np.asarray(rgba.getchannel("A"))
    return {
        "source": file_record(archived, source, output.parent.parent.parent),
        "vl_input": file_record(output, archived, output.parent.parent.parent),
        "alpha_nonzero_ratio": float(np.count_nonzero(alpha) / alpha.size),
        "derivation": "RGBA alpha-composited over neutral RGB(238,238,238); no content generation",
    }


def prepare_pillow(output_root: Path) -> dict[str, Any]:
    frame_source = PILLOW_ROOT / "video2mesh" / "scene" / "frames" / "000064.png"
    mask_source = (
        PILLOW_ROOT
        / "video2mesh"
        / "masks"
        / "2d_eroded5"
        / "sam3_class_pillow"
        / "000064.png"
    )
    frame_archive = output_root / "evidence" / "sam3_pillow_01" / "frame_000064.png"
    mask_archive = output_root / "evidence" / "sam3_pillow_01" / "accepted_mask_000064.png"
    frame_archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(frame_source, frame_archive)
    shutil.copy2(mask_source, mask_archive)

    image = Image.open(frame_archive).convert("RGB")
    mask = Image.open(mask_archive).convert("L")
    mask_array = np.asarray(mask)
    ys, xs = np.nonzero(mask_array > 0)
    if not len(xs):
        raise ValueError("Accepted pillow mask is empty")
    pad = 56
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(image.width, int(xs.max()) + pad + 1),
        min(image.height, int(ys.max()) + pad + 1),
    )
    crop = image.crop(box)
    mask_crop = mask.crop(box)

    rgba = crop.convert("RGBA")
    rgba.putalpha(mask_crop)
    isolated_path = output_root / "vl_inputs" / "sam3_pillow_01_masked.png"
    isolated_path.parent.mkdir(parents=True, exist_ok=True)
    background = Image.new("RGBA", rgba.size, (238, 238, 238, 255))
    Image.alpha_composite(background, rgba).convert("RGB").save(isolated_path, optimize=True)

    context = np.asarray(crop).copy()
    local_mask = np.asarray(mask_crop) > 0
    component_total, _, component_stats, _ = cv2.connectedComponentsWithStats(
        (mask_array > 0).astype(np.uint8), connectivity=8
    )
    component_sizes = sorted(
        (int(component_stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_total)),
        reverse=True,
    )
    contours, _ = cv2.findContours(local_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(context, contours, -1, (235, 52, 52), 4)
    context_path = output_root / "vl_inputs" / "sam3_pillow_01_context.png"
    Image.fromarray(context).save(context_path, optimize=True)

    frame_record = file_record(frame_archive, frame_source, output_root.parent)
    mask_record = file_record(mask_archive, mask_source, output_root.parent)
    derivative_sources = [
        {"role": "rgb_frame", **frame_record},
        {"role": "accepted_binary_mask", **mask_record},
    ]
    return {
        "object_id": "sam3_pillow_01",
        "category": "pillow",
        "instance_scope": "ensemble",
        "instance_semantics": "three_touching_pillows_as_one_accepted_entity_without_stable_per-pillow_ids",
        "mask_component_count": component_total - 1,
        "mask_component_pixels_descending": component_sizes,
        "evidence_kind": "accepted_holi_sam3_frame_mask_crop",
        "source_records": {
            "frame": frame_record,
            "accepted_mask": mask_record,
        },
        "crop_box_xyxy": list(box),
        "vl_images": [
            {
                "role": "accepted_mask_isolated",
                **file_record(isolated_path, mask_archive, output_root.parent),
                "derived_from": derivative_sources,
                "derivation": "crop frame to accepted mask bbox plus 56px padding; alpha=mask>0; composite on synthetic RGB(238,238,238)",
            },
            {
                "role": "scene_context_with_red_mask_boundary",
                **file_record(context_path, frame_archive, output_root.parent),
                "derived_from": derivative_sources,
                "derivation": "same crop; synthetic RGB(235,52,52), 4px contour over mask>0 boundary",
            },
        ],
        "scene_position_inferable": True,
        "spatial_constraint": (
            "Only relations visibly supported inside frame 000064 may be described. "
            "Coordinates and distances are not metric."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    objects = []
    for object_id, category in EMBODIED_OBJECTS:
        source = EMBODIED_ROOT / f"{object_id}_prompted_reference_rgba.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        archived = output_root / "evidence" / object_id / source.name
        vl_input = output_root / "vl_inputs" / f"{object_id}.png"
        records = composite_rgba(source, archived, vl_input)
        objects.append(
            {
                "object_id": object_id,
                "category": category,
                "instance_scope": "single_reconstruction_candidate",
                "evidence_kind": "embodiedgen_prompted_reference_rgba",
                "source_records": {"rgba": records["source"]},
                "vl_images": [
                    {
                        "role": "neutral_background_composite",
                        **records["vl_input"],
                        "derivation": records["derivation"],
                        "derived_from": [{"role": "rgba_reference", **records["source"]}],
                    }
                ],
                "alpha_nonzero_ratio": records["alpha_nonzero_ratio"],
                "derivation": records["derivation"],
                "scene_position_inferable": False,
                "spatial_constraint": (
                    "The isolated RGBA reference contains no scene context. Do not infer room location, "
                    "neighboring objects, metric size, or world orientation."
                ),
            }
        )

    objects.append(prepare_pillow(output_root))
    manifest = {
        "schema_version": 1,
        "scene_id": "bedroom_4",
        "objects": objects,
        "notes": [
            "EmbodiedGen evidence is the exact prompted-reference RGBA used by the reconstruction run.",
            "Pillow evidence uses accepted SAM3 erosion/depth-refined mask frame 000064.",
            "All VLM inputs are deterministic image derivatives; no generative image editing is applied.",
        ],
    }
    write_json(output_root / "input_manifest.json", manifest)
    print(json.dumps({"objects": len(objects), "output": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
