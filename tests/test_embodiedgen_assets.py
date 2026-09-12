from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/embodiedgen_assets.py"
SPEC = importlib.util.spec_from_file_location("embodiedgen_assets_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def write_splat(path: Path, positions):
    import numpy as np
    from plyfile import PlyData, PlyElement

    # a splat carries the Gaussian fields the renderer asks for
    rows = np.zeros(len(positions), dtype=[(name, "f4") for name in
                                           ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")])
    for index, position in enumerate(positions):
        rows[index]["x"], rows[index]["y"], rows[index]["z"] = position
        rows[index]["opacity"] = 1.0
        rows[index]["rot_0"] = 1.0
    PlyData([PlyElement.describe(rows, "vertex")]).write(str(path))


class EmbodiedGenBoundaryTests(unittest.TestCase):
    def test_asset_directories_are_refused_for_unsafe_object_ids(self):
        for value in ("", "..", "a/b", "bed bed", None, "x" * 129):
            with self.subTest(value=value), self.assertRaises(adapter.AssetError):
                adapter.asset_directory(Path("/tmp/assets"), value)
        self.assertEqual(adapter.asset_directory(Path("/tmp/assets"), "bed_bedding"),
                         Path("/tmp/assets/bed_bedding"))

    def test_paths_outside_the_task_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / "outputs/task"
            (task / "inputs").mkdir(parents=True)
            (task / "inputs/envelope.json").write_text("{}")
            self.assertTrue(adapter.task_path(task, "inputs/envelope.json").is_file())
            for value in ("../outside.json", str(Path(folder) / "elsewhere.json")):
                with self.subTest(value=value), self.assertRaises(adapter.AssetError):
                    adapter.task_path(task, value)

    def test_an_embodiedgen_checkout_without_the_package_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(adapter.AssetError) as context:
                adapter.load_embodiedgen(Path(folder))
            self.assertIn("not importable", str(context.exception))

    def test_non_finite_splats_are_removed_before_rendering(self):
        import numpy as np
        from plyfile import PlyData

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "splat.ply"
            write_splat(source, [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [1.0, 1.0, 1.0]])
            cleaned, dropped, _ = adapter.clean_splat(source, Path(folder) / "cleaned.ply")
            self.assertEqual(dropped, 1)
            self.assertNotEqual(cleaned, source)
            rows = PlyData.read(str(cleaned))["vertex"].data
            self.assertEqual(len(rows), 2)
            self.assertTrue(np.isfinite(np.column_stack([rows[k] for k in ("x", "y", "z")])).all())

    def test_a_clean_splat_is_reused_in_place(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "splat.ply"
            write_splat(source, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            cleaned, dropped, _ = adapter.clean_splat(source, Path(folder) / "cleaned.ply")
            self.assertEqual(dropped, 0)
            self.assertEqual(cleaned, source)

    def test_render_views_are_a_multiple_of_the_five_elevations(self):
        self.assertEqual(len(adapter.ELEVATIONS), 5)
        with self.assertRaises(SystemExit):
            adapter.main(["--task-dir", "/tmp/t", "--inputs", "i.json", "--outputs", "o.json",
                          "--embodiedgen-root", "/tmp/e", "--num-images", "7"])
        with self.assertRaises(SystemExit):
            adapter.main(["--task-dir", "/tmp/t", "--inputs", "i.json", "--outputs", "o.json",
                          "--embodiedgen-root", "/tmp/e", "--texture-size", "64"])

    def test_a_missing_input_envelope_fails_closed_with_a_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / "outputs/task"
            (task / "inputs").mkdir(parents=True)
            (task / "stages/mesh_postprocess").mkdir(parents=True)
            code = adapter.main(["--task-dir", str(task), "--inputs", "inputs.json",
                                 "--outputs", "stages/mesh_postprocess/provider-artifacts.json",
                                 "--embodiedgen-root", str(Path(folder) / "missing")])
            self.assertEqual(code, 2)

    def test_an_envelope_naming_another_module_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / "outputs/task"
            (task / "inputs").mkdir(parents=True)
            (task / "inputs.json").write_text(json.dumps({"module": "orbit_video", "inputs": {}}))
            code = adapter.main(["--task-dir", str(task), "--inputs", "inputs.json",
                                 "--outputs", "stages/mesh_postprocess/provider-artifacts.json",
                                 "--embodiedgen-root", str(Path(folder) / "missing")])
            self.assertEqual(code, 2)

    def test_reported_paths_must_be_existing_task_relative_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / "outputs/task"
            (task / "stages/mesh_postprocess/assets/bed").mkdir(parents=True)
            (task / "stages/mesh_postprocess/assets/bed/bed.glb").write_bytes(b"glb")
            good = [{"mesh_path": "stages/mesh_postprocess/assets/bed/bed.glb"}]
            adapter.verify_reported_paths(task, good, "repaired_visual_meshes")
            for record in ({"mesh_path": "stages/mesh_postprocess/assets/bed/missing.glb"},
                           {"mesh_path": str(task / "stages/mesh_postprocess/assets/bed/bed.glb")},
                           {"mesh_path": 7},
                           {"textures": [{"path": "assets/bed/texture.png"}]}):
                with self.subTest(record=record), self.assertRaises(adapter.AssetError):
                    adapter.verify_reported_paths(task, [record], "uv_textures")

    def test_the_module_declares_the_same_output_roles_as_the_builtin_provider(self):
        import world_modeling.modules.mesh_postprocess as builtin

        source = SCRIPT.read_text()
        for role in (item.name for item in builtin.SPEC.outputs):
            with self.subTest(role=role):
                self.assertIn(f'"{role}"', source)


class SurfacePointSplatTests(unittest.TestCase):
    """Stream3D publishes surface points, which the renderer needs as Gaussians."""

    def write_points(self, path: Path, count=64):
        import numpy as np
        from plyfile import PlyData, PlyElement

        rows = np.zeros(count, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                      ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")])
        grid = int(count ** (1 / 3)) + 1
        index = 0
        for x in range(grid):
            for y in range(grid):
                for z in range(grid):
                    if index >= count:
                        break
                    rows[index]["x"], rows[index]["y"], rows[index]["z"] = x * 0.1, y * 0.1, z * 0.1
                    rows[index]["red"], rows[index]["green"], rows[index]["blue"] = 10 + index % 200, 40, 200 - index % 150
                    rows[index]["alpha"] = 255
                    index += 1
        PlyData([PlyElement.describe(rows, "vertex")], text=False).write(str(path))
        return path

    def test_surface_points_become_opaque_gaussians_that_keep_their_colour(self):
        import numpy as np
        from plyfile import PlyData

        with tempfile.TemporaryDirectory() as folder:
            source = self.write_points(Path(folder) / "surface.ply", count=27)
            destination = Path(folder) / "splat.ply"
            report = adapter.synthesize_gaussian_ply(source, destination)
            self.assertEqual(report["status"], "synthesized_from_surface_points")
            self.assertEqual(report["points"], 27)
            self.assertGreater(report["median_neighbour_spacing"], 0)
            table = PlyData.read(str(destination))["vertex"].data
            self.assertEqual(len(table), 27)
            self.assertTrue((np.asarray(table["opacity"]) > 4).all())
            self.assertTrue((np.asarray(table["rot_0"]) == 1).all())
            self.assertTrue((np.asarray(table["scale_0"]) <= 0).all())
            # the SH DC term encodes the original byte colour
            source_table = PlyData.read(str(source))["vertex"].data
            recovered = np.rint((np.asarray(table["f_dc_0"]) * adapter.SH_DC_FACTOR + 0.5) * 255).clip(0, 255)
            self.assertTrue(np.allclose(recovered, np.asarray(source_table["red"]), atol=1))

    def test_clean_splat_converts_a_point_cloud_instead_of_refusing_it(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self.write_points(Path(folder) / "surface.ply", count=27)
            cleaned, dropped, report = adapter.clean_splat(source, Path(folder) / "cleaned.ply")
            self.assertEqual(dropped, 0)
            self.assertEqual(report["status"], "synthesized_from_surface_points")
            self.assertTrue(cleaned.is_file())

    def test_a_point_cloud_without_colour_is_still_refused(self):
        import numpy as np
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "grey.ply"
            rows = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
            PlyData([PlyElement.describe(rows, "vertex")]).write(str(source))
            with self.assertRaisesRegex(adapter.AssetError, "no RGB colour|no field of name opacity"):
                adapter.clean_splat(source, Path(folder) / "cleaned.ply")


if __name__ == "__main__":
    unittest.main()
