from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/render_previews.py"
SPEC = importlib.util.spec_from_file_location("render_previews_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class CameraTests(unittest.TestCase):
    def test_look_at_is_orthonormal_and_faces_the_target(self):
        rotation = adapter.look_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0])
        forward = rotation[2]
        self.assertAlmostEqual(forward[2], -1.0, places=9)
        for row in rotation:
            self.assertAlmostEqual(math.sqrt(sum(value * value for value in row)), 1.0, places=9)
        right, up = rotation[0], rotation[1]
        self.assertAlmostEqual(sum(right[i] * up[i] for i in range(3)), 0.0, places=9)

    def test_a_degenerate_camera_is_refused(self):
        with self.assertRaises(adapter.PipelineError):
            adapter.look_at([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        with self.assertRaises(adapter.PipelineError):
            adapter.look_at([0.0, 5.0, 0.0], [0.0, 0.0, 0.0], up=(0.0, 1.0, 0.0))

    def test_the_orbit_radius_is_honoured(self):
        eye = adapter.orbit_eye([1.0, 2.0, 3.0], 4.0, azimuth_degrees=0.0, elevation_degrees=0.0)
        distance = math.dist(eye, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(distance, 4.0, places=9)


class RenderTests(unittest.TestCase):
    def test_a_lone_point_lands_near_the_image_centre(self):
        import numpy as np

        image = adapter.render_points(np.zeros((1, 3)), np.array([[255, 0, 0]]), size=64)
        painted = np.argwhere((image[:, :, 0] > 200) & (image[:, :, 1] < 50))
        self.assertEqual(len(painted), 1)
        self.assertLess(abs(int(painted[0][0]) - 32), 3)
        self.assertLess(abs(int(painted[0][1]) - 32), 3)

    def test_the_nearer_point_wins(self):
        import numpy as np

        # Azimuth zero puts the camera on +x looking at the centroid, so the
        # point at +x is in front of the one at -x and they share a pixel.
        points = np.array([[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]])
        colours = np.array([[255, 0, 0], [0, 0, 255]])
        image = adapter.render_points(points, colours, size=64, azimuth=0.0, elevation=0.0, fov_degrees=60.0)
        painted = image[(image.sum(axis=2) < 600)]
        self.assertTrue(len(painted))
        self.assertTrue((painted[:, 0] > painted[:, 2]).all(), "the near red point must occlude the far blue one")

    def test_a_cloud_without_finite_positions_is_refused(self):
        import numpy as np

        with self.assertRaises(adapter.PipelineError):
            adapter.render_points(np.full((3, 3), np.nan), np.zeros((3, 3)), size=32)
        with self.assertRaises(adapter.PipelineError):
            adapter.render_points(np.zeros((2, 3)), np.zeros((3, 3)), size=32)

    def test_float_colours_are_scaled_to_bytes(self):
        import numpy as np

        image = adapter.render_points(np.zeros((1, 3)), np.array([[1.0, 0.5, 0.0]]), size=32)
        centre = image[16, 16]
        self.assertGreaterEqual(int(centre[0]), 250)
        self.assertAlmostEqual(int(centre[1]), 127, delta=2)

    def test_an_empty_preview_is_refused_and_a_real_one_reports_its_coverage(self):
        import numpy as np

        blank = np.full((32, 32, 3), 255, dtype=np.uint8)
        self.assertEqual(adapter.painted_fraction(blank), 0.0)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(adapter.PipelineError):
                adapter.save_preview(blank, Path(folder) / "blank.png", {})
        image = adapter.render_points(np.zeros((1, 3)), np.array([[10, 10, 10]]), size=32)
        with tempfile.TemporaryDirectory() as folder:
            entry = adapter.save_preview(image, Path(folder) / "point.png", {})
        self.assertGreater(entry["painted_fraction"], 0.0)
        self.assertIn("sha256", entry)

    def test_grid_dimensions_include_the_gaps(self):
        import numpy as np

        tiles = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
        canvas = adapter.grid(tiles, columns=3, gap=4)
        self.assertEqual(canvas.shape, (10 * 2 + 4, 10 * 3 + 8, 3))

    def test_turntable_returns_a_grid_of_the_requested_views(self):
        import numpy as np

        rng = np.random.default_rng(0)
        points = rng.normal(size=(500, 3))
        canvas = adapter.turntable(points, np.full((500, 3), 120), views=6, size=48, columns=3)
        self.assertEqual(canvas.shape[0], 48 * 2 + 8)
        self.assertEqual(canvas.shape[1], 48 * 3 + 16)


class GroundingPreviewTests(unittest.TestCase):
    def test_detection_previews_read_the_frame_shaped_grounding_document(self):
        import hashlib
        import json as jsonlib
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "outputs/task"
            (root / "inputs").mkdir(parents=True)
            Image.new("RGB", (64, 48), (30, 60, 90)).save(root / "inputs/frame.png")
            document = {"frames": [{"frame_id": "000003", "image_path": "inputs/frame.png",
                                    "width": 64, "height": 48,
                                    "objects": [{"category": "bed", "bbox_xyxy": [4, 4, 40, 40]},
                                                {"category": "lamp", "bbox_xyxy": [44, 6, 60, 30]}]}]}
            proposals = root / "proposals.json"
            proposals.write_text(jsonlib.dumps(document))
            digest = hashlib.sha256(proposals.read_bytes()).hexdigest()
            (root / "artifacts.json").write_text(jsonlib.dumps({"artifacts": {
                "object_proposals": {"path": "proposals.json", "sha256": digest,
                                     "size_bytes": proposals.stat().st_size}}}))
            export_root = Path(folder) / "delivery"
            export_root.mkdir()
            report = adapter.run(adapter.argparse.Namespace(
                task_dir=root, task_id=None, export_root=export_root, size=128, views=3))
            self.assertIn("object_detect", report["previews"])
            self.assertIn("object_cutout", report["previews"])
            self.assertEqual(report["previews"]["object_detect"]["boxes"], 2)
            self.assertEqual(report["previews"]["object_detect"]["frame_id"], "000003")
            self.assertTrue((export_root / "object/object_detect_preview.png").is_file())
            self.assertTrue((export_root / "object/object_cutout_preview.png").is_file())


class EntryPointTests(unittest.TestCase):
    def test_argument_bounds_are_enforced(self):
        for extra in (["--size", "64"], ["--views", "2"], ["--views", "99"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                adapter.main(["--task-dir", "/tmp/t", "--export-root", "/tmp/e", *extra])

    def test_a_missing_export_root_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "outputs/task"
            root.mkdir(parents=True)
            code = adapter.main(["--task-dir", str(root), "--export-root", str(Path(folder) / "absent")])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
