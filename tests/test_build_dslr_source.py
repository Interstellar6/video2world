from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_dslr_source.py"
SPEC = importlib.util.spec_from_file_location("build_dslr_source_adapter", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

WIDTH, HEIGHT = 64, 48


def rotmat2qvec(rotation):
    """Inverse of the adapter's qvec2rotmat, for building fixtures."""
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (rotation[2][1] - rotation[1][2]) / s, (rotation[0][2] - rotation[2][0]) / s,
             (rotation[1][0] - rotation[0][1]) / s]
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        s = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2
        q = [(rotation[2][1] - rotation[1][2]) / s, 0.25 * s, (rotation[0][1] + rotation[1][0]) / s,
             (rotation[0][2] + rotation[2][0]) / s]
    elif rotation[1][1] > rotation[2][2]:
        s = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2
        q = [(rotation[0][2] - rotation[2][0]) / s, (rotation[0][1] + rotation[1][0]) / s, 0.25 * s,
             (rotation[1][2] + rotation[2][1]) / s]
    else:
        s = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2
        q = [(rotation[1][0] - rotation[0][1]) / s, (rotation[0][2] + rotation[2][0]) / s,
             (rotation[1][2] + rotation[2][1]) / s, 0.25 * s]
    norm = math.sqrt(sum(value * value for value in q))
    return [value / norm for value in q]


def camera_pose(eye, target=(0.3, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """World-to-camera rotation whose rows are (right, down, forward)."""
    forward = [target[index] - eye[index] for index in range(3)]
    norm = math.sqrt(sum(value * value for value in forward))
    forward = [value / norm for value in forward]
    right = [up[1] * forward[2] - up[2] * forward[1], up[2] * forward[0] - up[0] * forward[2],
             up[0] * forward[1] - up[1] * forward[0]]
    norm = math.sqrt(sum(value * value for value in right))
    right = [value / norm for value in right]
    down = [forward[1] * right[2] - forward[2] * right[1], forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0]]
    return [right, down, forward]


def multiply(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def write_fixture(root: Path, frames: int = 6, points_per_axis: int = 5, *, cx=None, cy=None,
                  first_image_wrong_size=False, drop_first_colmap_frame=False,
                  nerfstudio_offset=None, nerfstudio_global_offset=None, colmap_look_away=False,
                  orphan_tracks=False):
    """A synthetic ScanNet++-shaped DSLR directory whose geometry is consistent."""
    from PIL import Image

    dslr = root / "dslr"
    (dslr / "resized_undistorted_images").mkdir(parents=True)
    (dslr / "colmap").mkdir(parents=True)
    (dslr / "nerfstudio").mkdir(parents=True)

    eyes, rotations = [], []
    for index in range(frames):
        angle = 2 * math.pi * index / frames
        eye = [4.0 * math.cos(angle), 2.0 * (index % 3) - 2.0, 4.0 * math.sin(angle)]
        rotation = camera_pose(eye)
        if colmap_look_away:  # same centre, optical axis reversed
            rotation = multiply([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], rotation)
        eyes.append(eye)
        rotations.append(rotation)

    names = [f"DSC{index + 1:05d}.JPG" for index in range(frames)]
    for position, name in enumerate(names):
        size = (WIDTH + 2, HEIGHT) if (first_image_wrong_size and position == 0) else (WIDTH, HEIGHT)
        Image.new("RGB", size, (32, 64, 96)).save(dslr / "resized_undistorted_images" / name)

    image_records = []
    for position, (name, eye, rotation) in enumerate(zip(names, eyes, rotations)):
        tvec = [-sum(rotation[i][k] * eye[k] for k in range(3)) for i in range(3)]
        image_records.append((position + 1, name, rotmat2qvec(rotation), tvec))

    lines = ["# Image list with two lines of data per image:\n",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]
    for position, (image_id, name, qvec, tvec) in enumerate(image_records):
        if drop_first_colmap_frame and position == 0:
            continue
        lines.append(f"{image_id} " + " ".join(repr(value) for value in qvec) + " "
                     + " ".join(repr(value) for value in tvec) + f" 1 {name}\n\n")
    (dslr / "colmap/images.txt").write_text("".join(lines))

    positions = []
    step = 1.2 / max(1, points_per_axis - 1)
    for i in range(points_per_axis):
        for j in range(points_per_axis):
            for k in range(points_per_axis):
                positions.append([-0.6 + i * step, -0.6 + j * step, -0.6 + k * step])
    track_ids = [99] if orphan_tracks else [image_id for image_id, _, _, _ in image_records]
    point_lines = ["# 3D point list with one line of data per point:\n"]
    for index, position in enumerate(positions):
        track = " ".join(f"{image_id} 0" for image_id in track_ids)
        point_lines.append(f"{index + 1} " + " ".join(repr(value) for value in position)
                           + " 10 20 30 0.0 " + track + "\n")
    (dslr / "colmap/points3D.txt").write_text("".join(point_lines))

    world = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    frames_payload = []
    for position, (name, eye, rotation) in enumerate(zip(names, eyes, rotations)):
        c2w_rotation = multiply(world, [[rotation[j][i] for j in range(3)] for i in range(3)])
        translation = list(eye)
        if nerfstudio_offset is not None and position == 0:
            translation = [translation[index] + nerfstudio_offset[index] for index in range(3)]
        if nerfstudio_global_offset is not None:
            translation = [translation[index] + nerfstudio_global_offset[index] for index in range(3)]
        matrix = [c2w_rotation[i] + [translation[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
        frames_payload.append({"file_path": name, "transform_matrix": matrix})
    (dslr / "nerfstudio/transforms_undistorted.json").write_text(json.dumps({
        "camera_model": "PINHOLE",
        "fl_x": 40.0,
        "fl_y": 40.0,
        "cx": WIDTH / 2 if cx is None else cx,
        "cy": HEIGHT / 2 if cy is None else cy,
        "w": WIDTH,
        "h": HEIGHT,
        "frames": frames_payload,
    }))
    return dslr


class BuildDslrSourceTests(unittest.TestCase):
    def run_builder(self, dslr: Path, output: Path):
        argv = [str(SCRIPT), "--dslr-dir", str(dslr), "--output", str(output), "--frames", "0"]
        previous = sys.argv
        sys.argv = argv
        try:
            return builder.main()
        finally:
            sys.argv = previous

    def test_consistent_capture_is_converted_into_a_colmap_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input")
            output = root / "source"
            self.assertEqual(self.run_builder(dslr, output), 0)
            self.assertEqual(len(list((output / "images").iterdir())), 6)
            cameras = (output / "sparse/0/cameras.txt").read_text().splitlines()
            self.assertIn(f"1 PINHOLE {WIDTH} {HEIGHT} 40.0 40.0 {WIDTH / 2} {HEIGHT / 2}", cameras)
            images = [line for line in (output / "sparse/0/images.txt").read_text().splitlines()
                      if not line.startswith("#")]
            self.assertEqual(len(images), 12)
            records = images[::2]
            self.assertEqual(len(records), 6)
            self.assertTrue(all(len(line.split()) == 10 for line in records))
            self.assertTrue(all(records[index].split()[9] == f"DSC{index + 1:05d}.JPG"
                                for index in range(6)))
            points = [line for line in (output / "sparse/0/points3D.txt").read_text().splitlines()
                      if not line.startswith("#")]
            self.assertEqual(len(points), 125)
            self.assertTrue(all(len(line.split()) >= 9 for line in points))

    def test_a_nerfstudio_frame_that_does_not_match_colmap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input", nerfstudio_offset=(3.0, 0.0, 0.0))
            with self.assertRaises(builder.BuildError) as context:
                self.run_builder(dslr, root / "source")
            self.assertIn("not one capture", str(context.exception))

    def test_a_translated_but_rigid_nerfstudio_trajectory_is_accepted(self):
        # A single rigid frame change moves every centre together and must not be
        # mistaken for a mismatch, so all six frames are shifted by the same vector.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input", nerfstudio_global_offset=(5.0, -2.0, 1.0))
            self.assertEqual(self.run_builder(dslr, root / "source"), 0)

    def test_an_off_centre_principal_point_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input", cx=WIDTH / 2 + 5)
            with self.assertRaises(builder.BuildError) as context:
                self.run_builder(dslr, root / "source")
            self.assertIn("not centered", str(context.exception))

    def test_an_image_that_contradicts_the_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input", first_image_wrong_size=True)
            with self.assertRaises(builder.BuildError) as context:
                self.run_builder(dslr, root / "source")
            self.assertIn("but calibration says", str(context.exception))

    def test_a_frame_missing_from_colmap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input", drop_first_colmap_frame=True)
            with self.assertRaises(builder.BuildError) as context:
                self.run_builder(dslr, root / "source")
            self.assertIn("not present in colmap/images.txt", str(context.exception))

    def test_points_without_observations_in_the_selection_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input", orphan_tracks=True)
            with self.assertRaises(builder.BuildError) as context:
                self.run_builder(dslr, root / "source")
            self.assertIn("no sparse points", str(context.exception))

    def test_poses_that_look_away_from_the_points_are_rejected(self):
        # Camera centres are untouched, so the trajectory gate is satisfied: only
        # the projection gate can catch an optical axis that contradicts the cloud.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dslr = write_fixture(root / "input", colmap_look_away=True)
            with self.assertRaises(builder.BuildError) as context:
                self.run_builder(dslr, root / "source")
            self.assertIn("sparse points inside the image", str(context.exception))

    def test_even_selection_keeps_endpoints_and_rejects_tiny_budgets(self):
        self.assertEqual(builder.select_evenly(10, 0), list(range(10)))
        self.assertEqual(builder.select_evenly(10, 10), list(range(10)))
        selected = builder.select_evenly(779, 4)
        self.assertEqual((selected[0], selected[-1]), (0, 778))
        self.assertEqual(len(set(selected)), 4)
        with self.assertRaises(builder.BuildError):
            builder.select_evenly(10, 2)

    def test_blank_points2d_lines_do_not_shift_the_image_records(self):
        # COLMAP writes an empty POINTS2D line for images without observations; a
        # parser that filters blank lines would pair every later record wrongly.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.txt"
            path.write_text("# comment\n1 0.0 0.0 0.0 1.0 1.0 2.0 3.0 1 A.JPG\n\n"
                            "2 0.0 0.0 0.0 1.0 4.0 5.0 6.0 1 B.JPG\n\n")
            entries = builder.parse_colmap_images(path)
            self.assertEqual(sorted(entries), ["A.JPG", "B.JPG"])
            self.assertEqual(entries["B.JPG"]["id"], 2)
            self.assertEqual(entries["B.JPG"]["tvec"], [4.0, 5.0, 6.0])

    def test_jpeg_size_reads_the_frame_header(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.JPG"
            Image.new("RGB", (123, 45)).save(path)
            self.assertEqual(builder.jpeg_size(path), (123, 45))


if __name__ == "__main__":
    unittest.main()
