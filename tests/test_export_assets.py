from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_assets.py"
SPEC = importlib.util.spec_from_file_location("export_assets_adapter", SCRIPT)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def write_gaussian_ply(path: Path, positions):
    import numpy as np
    from plyfile import PlyData, PlyElement

    rows = np.zeros(len(positions), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    for index, position in enumerate(positions):
        rows[index]["x"], rows[index]["y"], rows[index]["z"] = position
        rows[index]["opacity"] = 1.0
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))


class GaussianSceneExportTests(unittest.TestCase):
    def build(self, positions):
        directory = Path(tempfile.mkdtemp())
        task = export.Task(directory, "scene")
        (directory / "outputs/scene").mkdir(parents=True)
        source = directory / "outputs/scene/point_cloud.ply"
        write_gaussian_ply(source, positions)
        record = {"path": "point_cloud.ply", "sha256": export.sha256(source),
                  "size_bytes": source.stat().st_size}
        return task, source, record, directory

    def test_non_finite_splats_are_dropped_from_the_delivered_cloud(self):
        import numpy as np
        from plyfile import PlyData

        task, _, record, directory = self.build([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]])
        destination = directory / "export/scene/point_cloud_3dgs.ply"
        entry = export.copy_gaussian_ply(task, record, destination)
        self.assertEqual(entry["non_finite_rows_dropped"], 1)
        self.assertEqual(entry["exported_rows"], 2)
        self.assertEqual(entry["source_sha256"], record["sha256"])
        self.assertEqual(entry["sha256"], export.sha256(destination))
        rows = PlyData.read(str(destination))["vertex"].data
        self.assertEqual(len(rows), 2)
        self.assertTrue(np.isfinite(np.column_stack([rows[key] for key in ("x", "y", "z")])).all())

    def test_a_cloud_without_non_finite_rows_is_copied_unchanged(self):
        task, _, record, directory = self.build([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        destination = directory / "export/scene/point_cloud_3dgs.ply"
        entry = export.copy_gaussian_ply(task, record, destination)
        self.assertEqual(entry["non_finite_rows_dropped"], 0)
        self.assertEqual(entry["sha256"], record["sha256"])

    def test_a_cloud_that_is_entirely_non_finite_is_still_rejected(self):
        task, _, record, directory = self.build([[float("nan")] * 3, [float("inf"), 0.0, 0.0]])
        with self.assertRaises(export.PipelineError):
            export.copy_gaussian_ply(task, record, directory / "export/scene/point_cloud_3dgs.ply")

    def test_a_tampered_source_cloud_is_rejected_before_export(self):
        task, source, record, directory = self.build([[0.0, 0.0, 0.0]])
        write_gaussian_ply(source, [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        with self.assertRaises(export.PipelineError):
            export.copy_gaussian_ply(task, record, directory / "export/scene/point_cloud_3dgs.ply")


if __name__ == "__main__":
    unittest.main()
