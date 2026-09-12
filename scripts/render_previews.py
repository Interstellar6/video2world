#!/usr/bin/env python3
"""QA previews for a delivery: what the reconstruction and the assets look like.

Every image here is produced by a small CPU renderer instead of an OpenGL
context, because a headless box has no display and the point is a checkable
picture, not a beauty render. Two shapes are drawn:

* point clouds (a scene Gaussian cloud, an object's splat, or a mesh's vertices)
  are perspective-projected and painted far-to-near with a painter's algorithm;
* detections and masks come straight from the observed frames with PIL.

The previews are written into the export root so a delivery shows its own
evidence: a scene overview grid, a per-object turntable, a top-down layout, and
the detection/cutout stills from the source frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.engine import PipelineError, Task, artifact_path, read_json, sha256, write_json  # noqa: E402

BACKGROUND = (255, 255, 255)
DEFAULT_SIZE = 512
DEFAULT_VIEWS = 6


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation whose rows are right, up and forward."""
    forward = [target[i] - eye[i] for i in range(3)]
    norm = math.sqrt(sum(value * value for value in forward))
    if norm <= 0:
        raise PipelineError("camera and target coincide")
    forward = [value / norm for value in forward]
    right = [up[1] * forward[2] - up[2] * forward[1], up[2] * forward[0] - up[0] * forward[2],
             up[0] * forward[1] - up[1] * forward[0]]
    norm = math.sqrt(sum(value * value for value in right))
    if norm <= 1e-12:
        raise PipelineError("camera up vector is parallel to the view direction")
    right = [value / norm for value in right]
    true_up = [forward[1] * right[2] - forward[2] * right[1], forward[2] * right[0] - forward[0] * right[2],
               forward[0] * right[1] - forward[1] * right[0]]
    return [right, true_up, forward]


def orbit_eye(centre, radius, azimuth_degrees: float, elevation_degrees: float):
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    return [centre[0] + radius * math.cos(elevation) * math.cos(azimuth),
            centre[1] + radius * math.sin(elevation),
            centre[2] + radius * math.cos(elevation) * math.sin(azimuth)]


def render_points(points, colors, *, size: int = DEFAULT_SIZE, azimuth: float = 45.0, elevation: float = 20.0,
                  fov_degrees: float = 40.0, splat: int = 1, background=BACKGROUND):
    """Project and paint a coloured point cloud; nearer points win."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise PipelineError("render_points needs an (n, 3) point array")
    if colors.ndim != 2 or len(colors) != len(points) or colors.shape[1] < 3:
        raise PipelineError("render_points needs one colour per point")
    colors = colors[:, :3]
    if colors.dtype.kind == "f":
        scale = 255.0 if float(np.nanmax(colors)) <= 1.0 else 1.0
        colors = np.clip(colors * scale, 0, 255)
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if not len(points):
        raise PipelineError("point cloud has no finite positions")

    low, high = np.percentile(points, [2.0, 98.0], axis=0)
    centre = ((low + high) / 2.0).tolist()
    radius = float(np.linalg.norm((high - low) / 2.0)) or 1.0
    distance = radius / math.tan(math.radians(fov_degrees) / 2.0) * 1.6
    eye = orbit_eye(centre, distance, azimuth, elevation)
    rotation = look_at(eye, centre)
    relative = points - np.asarray(eye)
    camera = relative @ np.asarray(rotation).T
    depth = camera[:, 2]
    focal = (size / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
    visible = depth > 1e-9
    if not visible.any():
        raise PipelineError("no point is in front of the camera")
    u = np.rint(focal * camera[:, 0] / np.where(visible, depth, 1.0) + size / 2.0).astype(np.int64)
    v = np.rint(size / 2.0 - focal * camera[:, 1] / np.where(visible, depth, 1.0)).astype(np.int64)
    inside = visible & (u >= 0) & (u < size) & (v >= 0) & (v < size)

    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:, :] = np.asarray(background, dtype=np.uint8)
    order = np.argsort(-depth[inside], kind="stable")
    u, v, colors = u[inside][order], v[inside][order], colors[inside][order]
    half = max(0, splat // 2)
    for offset_y in range(-half, half + 1):
        for offset_x in range(-half, half + 1):
            target_y = np.clip(v + offset_y, 0, size - 1)
            target_x = np.clip(u + offset_x, 0, size - 1)
            image[target_y, target_x] = colors
    return image


def grid(images, *, columns: int, background=BACKGROUND, gap: int = 8):
    import numpy as np

    if not images:
        raise PipelineError("no images to tile")
    rows = math.ceil(len(images) / columns)
    height, width = images[0].shape[:2]
    canvas = np.empty((rows * height + (rows - 1) * gap, columns * width + (columns - 1) * gap, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(background, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[row * (height + gap):row * (height + gap) + height,
               column * (width + gap):column * (width + gap) + width] = image[:, :, :3]
    return canvas


def painted_fraction(image, background=BACKGROUND) -> float:
    """Share of pixels that are not the background.

    A preview that renders nothing is worse than no preview: it looks like
    evidence. Every delivered image reports its fraction and an empty one is
    refused.
    """
    import numpy as np

    array = np.asarray(image)
    mask = np.abs(array[:, :, :3].astype(np.int64) - np.asarray(background, dtype=np.int64)).sum(axis=2) > 12
    return float(mask.mean())


def save_preview(array, destination: Path, entry: dict) -> dict:
    entry["painted_fraction"] = round(painted_fraction(array), 6)
    if entry["painted_fraction"] <= 0.0:
        raise PipelineError(f"preview rendered no visible geometry: {destination}")
    entry["sha256"] = save_image(array, destination)
    return entry


def save_image(array, destination: Path) -> str:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(destination)
    return sha256(destination)


def draw_detections(image_path: Path, proposals, destination: Path) -> dict:
    """Box and label the frame-local detections on their own source frame."""
    from PIL import Image, ImageDraw

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    palette = [(214, 39, 40), (31, 119, 180), (44, 160, 44), (255, 127, 14), (148, 103, 189), (140, 86, 75)]
    drawn = 0
    for index, item in enumerate(proposals):
        box = item.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        colour = palette[index % len(palette)]
        draw.rectangle(box, outline=colour, width=max(2, image.width // 400))
        draw.text((box[0] + 4, max(0, box[1] - 14)), str(item.get("category") or item.get("object_id") or "object"),
                  fill=colour)
        drawn += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return {"preview_path": str(destination), "sha256": sha256(destination), "boxes": drawn,
            "frame_id": proposals[0].get("frame_id") if proposals else None}


def cutout_strip(image_path: Path, proposals, destination: Path, *, limit: int = 6) -> dict:
    """Paste the detected crops side by side so the masks can be eyeballed."""
    from PIL import Image

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    crops, labels = [], []
    for item in proposals:
        box = item.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (int(round(value)) for value in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.width, x1), min(image.height, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        crop = image.crop((x0, y0, x1, y1))
        crop.thumbnail((256, 256))
        crops.append(crop)
        labels.append(str(item.get("category") or "object"))
        if len(crops) >= limit:
            break
    if not crops:
        raise PipelineError("no usable detection crop for the cutout preview")
    width = sum(crop.width for crop in crops) + 8 * (len(crops) - 1)
    height = max(crop.height for crop in crops)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    offset = 0
    for crop in crops:
        canvas.paste(crop, (offset, 0))
        offset += crop.width + 8
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return {"preview_path": str(destination), "sha256": sha256(destination), "crops": len(crops), "labels": labels}


def point_cloud_from_mesh(path: Path, *, limit: int = 400_000):
    """Vertices and their colours, for a cheap surface preview."""
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not len(mesh.vertices):
        raise PipelineError(f"mesh has no vertices: {path}")
    points = np.asarray(mesh.vertices)
    colours = getattr(mesh.visual, "vertex_colors", None)
    if colours is None or len(colours) != len(points):
        colours = np.full((len(points), 3), 160, dtype=np.uint8)
    colours = np.asarray(colours)[:, :3]
    if len(points) > limit:
        step = len(points) / limit
        keep = np.arange(limit) * step
        keep = keep.astype(np.int64)
        points, colours = points[keep], colours[keep]
    return points, colours


def point_cloud_from_splat(path: Path, *, limit: int = 400_000):
    import numpy as np

    from world_modeling.gaussian_io import read_gaussian_rows

    cloud = read_gaussian_rows(path, required=("x", "y", "z", "f_dc_0"))
    rows, points = cloud.rows, cloud.points
    colours = np.column_stack([np.asarray(rows[name], dtype=np.float64) for name in ("f_dc_0", "f_dc_1", "f_dc_2")])
    colours = np.clip((colours + 1.0) * 127.5, 0, 255)
    if len(points) > limit:
        step = len(points) / limit
        keep = (np.arange(limit) * step).astype(np.int64)
        points, colours = points[keep], colours[keep]
    return points, colours


def turntable(points, colours, *, views: int, size: int, elevation: float = 20.0, columns: int = 3):
    images = [render_points(points, colours, size=size, elevation=elevation,
                            azimuth=360.0 * index / views) for index in range(views)]
    return grid(images, columns=columns)


def read_roles(task: Task) -> dict:
    artifacts = read_json(task.directory / "artifacts.json").get("artifacts", {})
    resolved = {}
    for role, record in artifacts.items():
        try:
            resolved[role] = artifact_path(task, record)
        except (PipelineError, KeyError, OSError):
            continue
    return resolved


def run(args) -> dict:
    import numpy as np

    task = Task(args.task_dir.resolve(), args.task_id) if args.task_id else Task(args.task_dir.resolve().parent.parent,
                                                                               args.task_dir.resolve().name)
    export_root = args.export_root.resolve()
    if not export_root.is_dir():
        raise PipelineError(f"export root does not exist: {export_root}")
    roles = read_roles(task)
    previews, skipped = {}, []
    size, views = args.size, args.views

    object_root = export_root / "object"
    scene_root = export_root / "scene"

    # Scene overview: the delivered surface from three sides, so the whole room
    # can be judged without opening a 30 MB mesh.
    try:
        points, colours = point_cloud_from_mesh(scene_root / "mesh.ply")
        images = [render_points(points, colours, size=size, elevation=elevation, azimuth=azimuth)
                  for elevation, azimuth in ((65.0, 45.0), (20.0, 45.0), (20.0, 225.0))]
        path = scene_root / "scene_overview.jpg"
        previews["scene_overview"] = save_preview(
            grid(images, columns=3), path,
            {"path": str(path.relative_to(export_root)),
             "views": [["top", 45.0], ["front", 45.0], ["back", 225.0]]})
    except (PipelineError, OSError, ValueError, KeyError) as error:
        skipped.append({"preview": "scene_overview", "reason": str(error)})

    # Top-down layout of the observed scene cloud, which is what the old
    # delivery called the layout preview.
    try:
        points, colours = point_cloud_from_splat(roles["scene_gaussian_ply"])
        path = object_root / "object_layout_preview.png"
        previews["object_layout"] = save_preview(
            render_points(points, colours, size=size, elevation=88.0, azimuth=0.0), path,
            {"path": str(path.relative_to(export_root)), "view": "top_down_observed_gaussians"})
    except (PipelineError, OSError, ValueError, KeyError) as error:
        skipped.append({"preview": "object_layout", "reason": str(error)})

    # Per-object turntables and mesh overviews.
    completed = {}
    meshes_record = roles.get("completed_object_meshes")
    if meshes_record is not None:
        for item in read_json(meshes_record).get("objects", []):
            completed[item.get("object_id")] = item
    for object_id, item in sorted(completed.items()):
        directory = export_root / "object" / "object_version_2" / object_id
        if not directory.is_dir():
            continue
        try:
            splat_path = Path(item.get("visual_ply_path") or "")
            splat = task.directory / splat_path if str(splat_path) and not splat_path.is_absolute() else None
            if splat is not None and splat.is_file():
                points, colours = point_cloud_from_splat(splat)
                path = directory / "gaussian_turntable_overview.jpg"
                previews[f"{object_id}_turntable"] = save_preview(
                    turntable(points, colours, views=views, size=size), path,
                    {"path": str(path.relative_to(export_root)), "views": views})
        except (PipelineError, OSError, ValueError, KeyError) as error:
            skipped.append({"preview": f"{object_id}_turntable", "reason": str(error)})
        glb = directory / "asset_mesh.glb"
        if glb.is_file():
            try:
                points, colours = point_cloud_from_mesh(glb)
                path = directory / "glb_mesh_overview.jpg"
                previews[f"{object_id}_mesh"] = save_preview(
                    turntable(points, colours, views=views, size=size), path,
                    {"path": str(path.relative_to(export_root)), "views": views})
            except (PipelineError, OSError, ValueError, KeyError) as error:
                skipped.append({"preview": f"{object_id}_mesh", "reason": str(error)})

    # Detection and cutout stills from the frame with the most proposals.
    proposals_path = roles.get("object_proposals")
    if proposals_path is not None:
        # The grounding document is frame-shaped: frames[*].objects, each already
        # carrying the frame it was read from.
        document = read_json(proposals_path)
        by_frame = {}
        for frame in document.get("frames", []):
            objects = frame.get("objects") or []
            if objects and frame.get("image_path"):
                by_frame[str(frame["frame_id"])] = (frame["image_path"], objects)
        if by_frame:
            frame_id = max(by_frame, key=lambda key: (len(by_frame[key][1]), key))
            image_path, objects = by_frame[frame_id]
            try:
                detect = draw_detections(task.directory / image_path, objects,
                                         object_root / "object_detect_preview.png")
                detect["path"] = "object/object_detect_preview.png"
                detect["frame_id"] = frame_id
                previews["object_detect"] = detect
                cutout = cutout_strip(task.directory / image_path, objects,
                                      object_root / "object_cutout_preview.png")
                cutout["path"] = "object/object_cutout_preview.png"
                cutout["frame_id"] = frame_id
                previews["object_cutout"] = cutout
            except (PipelineError, OSError, ValueError, KeyError) as error:
                skipped.append({"preview": "object_detect", "reason": str(error)})
        else:
            skipped.append({"preview": "object_detect",
                            "reason": "object_proposals carries no frame with objects and an image path"})
    else:
        skipped.append({"preview": "object_detect", "reason": "object_proposals is not published"})

    report = {"schema_version": "1.0", "kind": "video2world-modeling.delivery_previews", "status": "produced",
              "renderer": "cpu_point_painter", "size": size, "views": views,
              "previews": previews, "skipped": skipped}
    write_json(export_root / "qa" / "previews.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--task-id")
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS)
    args = parser.parse_args(argv)
    if not 128 <= args.size <= 2048 or not 3 <= args.views <= 12:
        parser.error("size must be between 128 and 2048; views between 3 and 12")
    try:
        report = run(args)
    except (PipelineError, OSError, ValueError, KeyError) as error:
        print(f"render_previews failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("status", "previews", "skipped")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
