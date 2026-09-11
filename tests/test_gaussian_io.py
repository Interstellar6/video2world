from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "src/world_modeling/gaussian_io.py"
SPEC = importlib.util.spec_from_file_location("gaussian_io_adapter", SCRIPT)
gaussians = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gaussians)


def write_ply(path: Path, positions, *, opacity=None, with_face=False, drop=(), extra=()):
    """Write a real Gaussian PLY so the helper is exercised end to end."""
    import numpy as np
    from plyfile import PlyData, PlyElement

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4"), *((name, "f4") for name in extra)]
    fields = [field for field in fields if field[0] not in drop]
    rows = np.zeros(len(positions), dtype=fields)
    names = set(rows.dtype.names or ())
    for index, position in enumerate(positions):
        if {"x", "y", "z"} <= names:
            rows[index]["x"], rows[index]["y"], rows[index]["z"] = position
        if "opacity" in names:
            rows[index]["opacity"] = 1.0 if opacity is None else opacity[index]
    for name in extra:
        if name in names:
            rows[name] = 1.0
    elements = [PlyElement.describe(rows, "vertex")]
    if with_face:
        faces = np.array([([0, 1, 2],)], dtype=[("vertex_indices", "i4", (3,))])
        elements.append(PlyElement.describe(faces, "face"))
    PlyData(elements).write(str(path))


class GaussianIoTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "point_cloud.ply"

    def tearDown(self):
        self.directory.cleanup()

    def test_non_finite_vertices_are_dropped_and_counted(self):
        write_ply(self.path, [[0.0, 0.0, 0.0], [float("nan"), 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        cloud = gaussians.read_gaussian_rows(self.path)
        self.assertEqual(cloud.dropped, 1)
        self.assertEqual(len(cloud.rows), 3)
        self.assertEqual(cloud.points.shape, (3, 3))
        self.assertTrue(self.np.isfinite(cloud.points).all())
        self.assertEqual(cloud.payload.elements[0].name, "vertex")

    def test_a_non_finite_gaussian_attribute_removes_the_whole_row(self):
        # A splat with non-finite opacity renders as garbage even when x/y/z are finite.
        write_ply(self.path, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], opacity=[1.0, float("inf")])
        cloud = gaussians.read_gaussian_rows(self.path)
        self.assertEqual(cloud.dropped, 1)
        self.assertEqual(len(cloud.points), 1)
        self.assertEqual(cloud.points[0].tolist(), [0.0, 0.0, 0.0])

    def test_the_real_pgsr_failure_shape_is_tolerated(self):
        # Mirrors the ScanNet++ DSLR run: 72 MB PGSR cloud where 2.7% of rows are
        # entirely NaN; the stage must keep the finite 97% and record the rest.
        positions = [[float(index), 0.0, 0.0] for index in range(97)]
        positions += [[float("nan")] * 3 for _ in range(3)]
        write_ply(self.path, positions)
        cloud = gaussians.read_gaussian_rows(self.path)
        self.assertEqual(cloud.dropped, 3)
        self.assertEqual(len(cloud.points), 97)

    def test_an_entirely_non_finite_table_stays_fatal(self):
        write_ply(self.path, [[float("nan")] * 3, [float("inf"), 0.0, 0.0]])
        with self.assertRaises(gaussians.GaussianError) as context:
            gaussians.read_gaussian_rows(self.path)
        self.assertIn("every one of the 2 Gaussian vertices", str(context.exception))

    def test_a_missing_coordinate_attribute_stays_fatal(self):
        write_ply(self.path, [[0.0, 0.0, 0.0]], drop=("z",))
        with self.assertRaises(gaussians.GaussianError) as context:
            gaussians.read_gaussian_rows(self.path)
        self.assertIn("missing PLY vertex attributes", str(context.exception))
        self.assertIn("'z'", str(context.exception))

    def test_extra_elements_stay_fatal(self):
        write_ply(self.path, [[0.0, 0.0, 0.0]], with_face=True)
        with self.assertRaises(gaussians.GaussianError) as context:
            gaussians.read_gaussian_rows(self.path)
        self.assertIn("expected one PLY vertex table", str(context.exception))

    def test_required_attributes_are_honoured(self):
        write_ply(self.path, [[0.0, 0.0, 0.0]], extra=("f_dc_0", "scale_0", "rot_0"))
        cloud = gaussians.read_gaussian_rows(
            self.path, required=("x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"))
        self.assertEqual(cloud.dropped, 0)
        self.assertEqual(len(cloud.points), 1)

    def test_the_reported_error_type_is_used(self):
        class Custom(RuntimeError):
            pass

        write_ply(self.path, [[0.0, 0.0, 0.0]])
        with self.assertRaises(Custom):
            gaussians.read_gaussian_rows(self.path, required=("f_dc_0",), error=Custom)


if __name__ == "__main__":
    unittest.main()
