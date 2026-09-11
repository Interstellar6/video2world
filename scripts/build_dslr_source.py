#!/usr/bin/env python3
"""Build a COLMAP ``images/ + sparse/0/`` source dataset from the ScanNet++ DSLR release.

The ScanNet++ DSLR capture ships as
``<scene>/dslr/{resized_undistorted_images,colmap,nerfstudio}``.  The modular
``scene_reconstruction`` module needs an undistorted COLMAP model with a
centered PINHOLE camera, so this script:

* takes the **poses** from ``colmap/images.txt`` verbatim -- they are already in
  the COLMAP world-to-camera convention and share the world frame of
  ``colmap/points3D.txt``;
* takes the **intrinsics** from ``nerfstudio/transforms_undistorted.json``,
  because those describe the undistorted images that are actually shipped;
  ``colmap/cameras.txt`` is OPENCV_FISHEYE and does not apply to them;
* copies the selected frames and writes ``cameras.txt``/``images.txt``/
  ``points3D.txt`` filtered to the points observed by those frames.

Undistortion is a per-pixel remap, so it leaves the extrinsics untouched: the
colmap poses pair with the undistorted intrinsics.  Three gates run before
anything is written, because a silently mixed-up dataset would poison every
downstream module:

1. every selected frame exists in all three sources with matching dimensions and
   a centered PINHOLE principal point;
2. the colmap and nerfstudio trajectories are the *same* capture -- the pairwise
   distance matrix of camera centres must agree, which is what a fixed rigid
   frame change preserves (the poses themselves differ by an axis convention, so
   comparing them element-wise would be meaningless);
3. projecting the sparse points through the colmap poses and undistorted
   intrinsics puts a healthy fraction of them inside the frames, which is the
   direct evidence that poses, intrinsics and points describe one geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
from pathlib import Path


class BuildError(RuntimeError):
    pass


def qvec2rotmat(qvec):
    """COLMAP quaternion (qw, qx, qy, qz) to rotation matrix."""
    w, x, y, z = qvec
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def camera_center(qvec, tvec):
    rotation = qvec2rotmat(qvec)
    return [-sum(rotation[k][i] * tvec[k] for k in range(3)) for i in range(3)]


def jpeg_size(path: Path):
    data = path.read_bytes()
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return width, height
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        index += 2 + struct.unpack(">H", data[index + 2:index + 4])[0]
    raise BuildError(f"{path}: could not read JPEG dimensions")


def parse_colmap_images(path: Path):
    entries = {}
    with path.open() as handle:
        # Blank lines are meaningful: every image record is followed by its
        # POINTS2D line, which is legitimately empty for an image with no
        # observations.  Dropping blanks would shift every record after it into
        # the POINTS2D slot and silently mis-pair the whole file.
        lines = [line for line in handle if not line.startswith("#")]
    # A trailing blank beyond the final POINTS2D line is padding, not a record;
    # drop only as many as it takes to restore the two-lines-per-image pairing.
    while len(lines) % 2 != 0 and lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 2 != 0:
        raise BuildError(f"{path}: expected two lines per image, got {len(lines)} data lines")
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) != 10:
            raise BuildError(f"{path}: unexpected image record: {lines[index].strip()[:80]}")
        name = fields[9]
        if name in entries:
            raise BuildError(f"{path}: duplicate image name {name}")
        entries[name] = {
            "id": int(fields[0]),
            "qvec": [float(v) for v in fields[1:5]],
            "tvec": [float(v) for v in fields[5:8]],
        }
    return entries


def parse_points3d(path: Path):
    points = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 8:
                continue
            track = fields[8:]
            image_ids = {int(track[index]) for index in range(0, len(track) - 1, 2)}
            points.append((int(fields[0]), [float(v) for v in fields[1:4]], [int(v) for v in fields[4:7]], image_ids))
    return points


def select_evenly(count: int, maximum: int):
    if maximum <= 0 or maximum >= count:
        return list(range(count))
    if maximum < 3:
        raise BuildError("--frames must be zero (all) or at least three")
    return [round(index * (count - 1) / (maximum - 1)) for index in range(maximum)]


def parse_colmap_point_ids(path: Path):
    """image name -> set of POINT3D_ID observed in it (the POINTS2D line)."""
    observations = {}
    with path.open() as handle:
        lines = [line for line in handle if not line.startswith("#")]
    while len(lines) % 2 != 0 and lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 2 != 0:
        raise BuildError(f"{path}: expected two lines per image, got {len(lines)} data lines")
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) != 10:
            raise BuildError(f"{path}: unexpected image record")
        points2d = lines[index + 1].split()
        observations[fields[9]] = {int(points2d[position]) for position in range(2, len(points2d), 3)}
    return observations


def select_window(names, observations, window: int, stride: int, minimum_views: int = 3):
    """Pick the contiguous window whose sparse points are seen by the most frames.

    A DSLR release is a walk through the whole space, while stage 1 expects one
    coherent sequence: DA3 multi-view depth and PGSR both need tight baselines
    and repeated coverage. Scoring windows by how many triangulated points at
    least ``minimum_views`` of their frames share picks the segment that is
    actually reconstructed well, instead of an arbitrary slice.
    """
    if window < 3 or window > len(names):
        raise BuildError(f"--window-size must be between 3 and {len(names)}")
    if stride < 1:
        raise BuildError("--window-stride must be positive")
    counts: dict = {}
    for name in names[:window]:
        for point_id in observations.get(name, ()):
            counts[point_id] = counts.get(point_id, 0) + 1
    best = None
    start = 0
    while True:
        score = sum(1 for value in counts.values() if value >= minimum_views)
        if best is None or score > best[1]:
            best = (start, score)
        following = start + window
        if following >= len(names):
            break
        for point_id in observations.get(names[start], ()):
            counts[point_id] -= 1
            if counts[point_id] == 0:
                del counts[point_id]
        for point_id in observations.get(names[following], ()):
            counts[point_id] = counts.get(point_id, 0) + 1
        start += stride
    start, score = best
    return start, score


def distance_matrix(points):
    return [[math.dist(points[i], points[j]) for j in range(len(points))] for i in range(len(points))]


def check_trajectory_agreement(names, colmap_centers, nerfstudio_centers, tolerance):
    """Two fixed-frame descriptions of one capture keep all centre distances."""
    if len(names) < 3:
        raise BuildError("at least three frames are required to compare trajectories")
    left = distance_matrix(colmap_centers)
    right = distance_matrix(nerfstudio_centers)
    scale = max(max(row) for row in left)
    if scale <= 0:
        raise BuildError("camera centres are degenerate; nothing to compare")
    worst, worst_pair = 0.0, None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mismatch = abs(left[i][j] - right[i][j]) / scale
            if mismatch > worst:
                worst, worst_pair = mismatch, (names[i], names[j])
    if worst > tolerance:
        raise BuildError(
            f"colmap and nerfstudio trajectories disagree by {worst:.6g} of the scene extent "
            f"(tolerance {tolerance}) at {worst_pair}; the two files are not one capture"
        )
    return worst


def check_projection(records, points, intrinsics, width, height, minimum_fraction, samples=4000):
    """Project sparse points through the shipped poses and intrinsics.

    The gate is on the *median* frame, not the worst one: a capture legitimately
    contains frames aimed away from the reconstructed sparse cloud, so a single
    low frame is not evidence of a broken dataset.  A wrong pose convention, by
    contrast, drags every frame down -- which is what this catches.
    """
    if len(points) <= samples:
        sample = points
    else:
        step = len(points) / samples
        sample = [points[int(index * step)] for index in range(samples)]
    fx, fy, cx, cy = intrinsics
    fractions = []
    for record in records:
        rotation = qvec2rotmat(record["qvec"])
        tvec = record["tvec"]
        inside = 0
        for _, xyz, _, _ in sample:
            cam = [sum(rotation[i][k] * xyz[k] for k in range(3)) + tvec[i] for i in range(3)]
            if cam[2] <= 0:
                continue
            u = fx * cam[0] / cam[2] + cx
            v = fy * cam[1] / cam[2] + cy
            if 0 <= u < width and 0 <= v < height:
                inside += 1
        fractions.append(inside / len(sample))
    ordered = sorted(fractions)
    median = ordered[len(ordered) // 2]
    if median < minimum_fraction:
        raise BuildError(
            f"the median frame keeps only {median:.3f} of sparse points inside the image "
            f"(minimum {minimum_fraction}); poses, intrinsics and points do not agree"
        )
    return {"median": median, "worst": ordered[0], "best": ordered[-1], "frames": len(ordered)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dslr-dir", required=True, help="<scene>/dslr directory of the ScanNet++ release")
    parser.add_argument("--output", required=True, help="destination dataset directory (images/ + sparse/0/)")
    parser.add_argument("--frames", type=int, default=60, help="how many frames to keep (0 = all)")
    parser.add_argument("--trajectory-tolerance", type=float, default=1e-3, help="max relative trajectory mismatch")
    parser.add_argument("--min-visible-fraction", type=float, default=0.3, help="min share of points inside a frame")
    parser.add_argument("--window-strategy", choices=("even", "coverage"), default="even",
                        help="'even' spreads the selection over the whole capture, 'coverage' takes the best "
                             "contiguous segment, which is what a multi-view reconstruction needs")
    parser.add_argument("--window-size", type=int, default=50, help="frames per candidate segment for --window-strategy coverage")
    parser.add_argument("--window-stride", type=int, default=5, help="segment step for --window-strategy coverage")
    args = parser.parse_args()

    dslr = Path(args.dslr_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    images_dir = dslr / "resized_undistorted_images"
    transforms_path = dslr / "nerfstudio/transforms_undistorted.json"
    colmap_images_path = dslr / "colmap/images.txt"
    colmap_points_path = dslr / "colmap/points3D.txt"
    for path in (images_dir, transforms_path, colmap_images_path, colmap_points_path):
        if not path.exists():
            raise BuildError(f"missing required input: {path}")

    transforms = json.loads(transforms_path.read_text())
    if transforms.get("camera_model") != "PINHOLE":
        raise BuildError(f"expected undistorted PINHOLE intrinsics, got {transforms.get('camera_model')}")
    width, height = int(transforms["w"]), int(transforms["h"])
    fx, fy, cx, cy = (float(transforms[key]) for key in ("fl_x", "fl_y", "cx", "cy"))
    if abs(cx - width / 2) > 1 or abs(cy - height / 2) > 1:
        raise BuildError(f"principal point ({cx}, {cy}) is not centered for {width}x{height}")

    colmap_entries = parse_colmap_images(colmap_images_path)
    frames = sorted(transforms["frames"], key=lambda item: item["file_path"])
    names_all = [frame["file_path"] for frame in frames]
    window_score = None
    if args.window_strategy == "coverage":
        observations = parse_colmap_point_ids(colmap_images_path)
        missing = [name for name in names_all if name not in observations]
        if missing:
            raise BuildError(f"{missing[0]}: no POINTS2D record in colmap/images.txt")
        start, window_score = select_window(names_all, observations, args.window_size, args.window_stride)
        selected = frames[start:start + args.window_size]
        print(json.dumps({"contiguous_window": {"start": names_all[start], "end": names_all[start + args.window_size - 1],
                                                "well_covered_points": window_score}}, sort_keys=True))
    else:
        selected = [frames[index] for index in select_evenly(len(frames), args.frames)]

    records, colmap_centers, nerfstudio_centers = [], [], []
    for frame in selected:
        name = frame["file_path"]
        if name not in colmap_entries:
            raise BuildError(f"{name}: not present in colmap/images.txt")
        image_path = images_dir / name
        if not image_path.is_file():
            raise BuildError(f"{name}: missing image file")
        size = jpeg_size(image_path)
        if size != (width, height):
            raise BuildError(f"{name}: image is {size[0]}x{size[1]} but calibration says {width}x{height}")
        entry = colmap_entries[name]
        if not all(math.isfinite(v) for v in entry["qvec"] + entry["tvec"]):
            raise BuildError(f"{name}: non-finite pose")
        records.append({"name": name, "id": entry["id"], "qvec": entry["qvec"], "tvec": entry["tvec"]})
        colmap_centers.append(camera_center(entry["qvec"], entry["tvec"]))
        nerfstudio_centers.append([float(frame["transform_matrix"][i][3]) for i in range(3)])

    names = [record["name"] for record in records]
    trajectory_mismatch = check_trajectory_agreement(
        names, colmap_centers, nerfstudio_centers, args.trajectory_tolerance
    )

    selected_ids = {record["id"] for record in records}
    points = parse_points3d(colmap_points_path)
    kept = [item for item in points if item[3] & selected_ids]
    if not kept:
        raise BuildError("no sparse points are observed by the selected frames; PGSR would have no initialization")
    projection = check_projection(
        records, kept, (fx, fy, cx, cy), width, height, args.min_visible_fraction
    )

    if output.exists():
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True)
    (output / "sparse/0").mkdir(parents=True)
    for record in records:
        shutil.copy2(images_dir / record["name"], output / "images" / record["name"])

    (output / "sparse/0/cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {width} {height} {fx!r} {fy!r} {cx!r} {cy!r}\n"
    )
    lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        f"# Number of images: {len(records)}, mean observations per image: 0\n",
    ]
    for record in records:
        lines.append(
            f"{record['id']} " + " ".join(f"{v!r}" for v in record["qvec"]) + " "
            + " ".join(f"{v!r}" for v in record["tvec"]) + f" 1 {record['name']}\n"
        )
        lines.append("\n")
    (output / "sparse/0/images.txt").write_text("".join(lines))

    point_lines = [
        "# 3D point list with one line of data per point:\n",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        f"# Number of points: {len(kept)}, mean track length: "
        f"{sum(len(item[3] & selected_ids) for item in kept) / max(1, len(kept)):.4f}\n",
    ]
    for point_id, xyz, rgb, image_ids in kept:
        track = " ".join(f"{image_id} 0" for image_id in sorted(image_ids & selected_ids))
        point_lines.append(
            f"{point_id} " + " ".join(f"{v!r}" for v in xyz) + " " + " ".join(str(v) for v in rgb) + f" 0.0 {track}\n"
        )
    (output / "sparse/0/points3D.txt").write_text("".join(point_lines))

    print(json.dumps({
        "output": str(output),
        "frames": len(records),
        "available_frames": len(frames),
        "images": len(list((output / "images").iterdir())),
        "points": len(kept),
        "points_total": len(points),
        "camera_model": "PINHOLE",
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height},
        "trajectory_mismatch": trajectory_mismatch,
        "sparse_point_visibility": projection,
        "first_frame": records[0]["name"],
        "last_frame": records[-1]["name"],
        "window_strategy": args.window_strategy,
        "well_covered_points": window_score,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as error:
        print(f"build_dslr_source failed: {error}", file=sys.stderr)
        sys.exit(1)
