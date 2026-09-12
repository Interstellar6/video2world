from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bake_background_texture.py"
SPEC = importlib.util.spec_from_file_location("bake_background_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def make_view(np, *, eye=(0.0, 0.0, 3.0), colour=(200, 40, 40), size=32, depth_value=3.0, facing=True):
    # A camera at +z looking down -z: world z=eye_z maps to camera z=0.
    rotation = np.eye(4)
    rotation[2, 2] = -1.0
    rotation[2, 3] = eye[2]
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = colour
    depth = np.full((size, size), depth_value, dtype=np.float64)
    intrinsics = np.array([[size, 0.0, size / 2], [0.0, size, size / 2], [0.0, 0.0, 1.0]])
    return {"frame_id": "000000", "image": image, "intrinsics": intrinsics,
            "world_to_camera": rotation, "image_to_depth": np.eye(3), "depth": depth,
            "width": size, "height": size, "eye": np.array(eye),
            "absolute_tolerance": 0.05, "relative_tolerance": 0.05}


class BakeTests(unittest.TestCase):
    def test_a_face_in_front_of_the_camera_is_visible_at_the_buffer_depth(self):
        import numpy as np

        view = make_view(np)
        points = np.array([[0.0, 0.0, 0.0]])
        normals = np.array([[0.0, 0.0, 1.0]])
        error = adapter.frame_score(view, points, normals)[0]
        self.assertTrue(np.isfinite(error))
        # a point the depth buffer contradicts is not visible (camera z = 2 vs 3)
        far = np.array([[0.0, 0.0, 1.0]])
        self.assertFalse(np.isfinite(adapter.frame_score(view, far, normals)[0]))

    def test_a_back_facing_normal_is_not_sampled(self):
        import numpy as np

        view = make_view(np)
        points = np.array([[0.0, 0.0, 0.0]])
        away = np.array([[0.0, 0.0, -1.0]])
        self.assertFalse(np.isfinite(adapter.frame_score(view, points, away)[0]))

    def test_sampling_returns_the_frame_colour(self):
        import numpy as np

        view = make_view(np, colour=(17, 200, 90))
        colours = adapter.sample_colours(view, np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.0]]))
        self.assertEqual(colours.shape, (2, 3))
        for row in colours:
            self.assertTrue(abs(int(row[0]) - 17) <= 2)
            self.assertTrue(abs(int(row[1]) - 200) <= 2)
            self.assertTrue(abs(int(row[2]) - 90) <= 2)

    def test_rasterising_a_uv_triangle_covers_its_texels_with_unit_weights(self):
        import numpy as np

        uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        raster = adapter.rasterize_face(uv, 16)
        self.assertIsNotNone(raster)
        texels, weights = raster
        self.assertGreater(len(texels), 16)
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0, atol=1e-9))
        self.assertTrue((texels < 16).all())
        # a degenerate triangle covers nothing
        self.assertIsNone(adapter.rasterize_face(np.zeros((3, 2)), 16))

    def test_uncovered_texels_grow_from_the_covered_neighbourhood(self):
        import numpy as np

        texture = np.zeros((8, 8, 3), dtype=np.uint8)
        covered = np.zeros((8, 8), dtype=bool)
        texture[4, 4] = (10, 20, 30)
        covered[4, 4] = True
        filled, mask = adapter.fill_uncovered(texture, covered, passes=2)
        self.assertTrue(mask.sum() > 1)
        self.assertTrue((filled[3, 4] == np.array([10, 20, 30])).all())
        self.assertTrue((filled[0, 0] == 0).all())

    def test_argument_bounds_are_enforced(self):
        for extra in (["--face-budget", "10"], ["--texture-size", "64"], ["--absolute-tolerance", "0"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                adapter.main(["--task-dir", "/tmp/t", "--output-dir", "/tmp/o", *extra])


if __name__ == "__main__":
    unittest.main()
