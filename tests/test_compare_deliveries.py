from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_deliveries.py"
SPEC = importlib.util.spec_from_file_location("compare_deliveries_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def write_ply(path: Path, *, vertices: int, faces: int, colour: bool) -> None:
    properties = "property float x\nproperty float y\nproperty float z\n"
    if colour:
        properties += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ply\nformat ascii 1.0\n"
                    f"element vertex {vertices}\n{properties}"
                    f"element face {faces}\nproperty list uchar int vertex_indices\nend_header\n")


def write_glb(path: Path, *, tris: int, attributes=("POSITION",), images: int = 0) -> None:
    document = {
        "meshes": [{"primitives": [{"attributes": {name: 0 for name in attributes}, "indices": 1}]}],
        "accessors": [{"count": tris * 3}, {"count": tris * 3}],
        "images": [{} for _ in range(images)],
    }
    payload = json.dumps(document).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    chunk = struct.pack("<II", len(payload), 0x4E4F534A) + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(chunk)) + chunk)


def make_delivery(root: Path, *, objects, parts, colour=True, viewer=False):
    write_ply(root / "scene/mesh.ply", vertices=100, faces=200, colour=colour)
    write_ply(root / "scene/point_cloud_3dgs.ply", vertices=50, faces=0, colour=False)
    write_glb(root / "object/background.glb", tris=1000, attributes=("POSITION", "TEXCOORD_0"), images=1)
    for object_id in objects:
        directory = root / "object/object_version_2" / object_id
        write_glb(directory / "asset_mesh.glb", tris=5000, attributes=("POSITION", "COLOR_0"))
        (directory / "asset_splat.ply").write_bytes(b"ply\n")
        (directory / "collision").mkdir(parents=True, exist_ok=True)
        for index in range(2):
            (directory / "collision" / f"hull_{index:03d}.obj").write_text("v 0 0 0\n")
    for object_id in parts:
        write_ply(root / "object/parts" / object_id / f"{object_id}_part_00.ply",
                  vertices=10, faces=0, colour=False)
    write_ply(root / "scene/point_cloud_simple.ply", vertices=20, faces=0, colour=False)
    (root / "scene/scene_overview.jpg").write_bytes(b"jpg")
    (root / "scene/depth").mkdir(parents=True, exist_ok=True)
    (root / "scene/depth/000000.png").write_bytes(b"png")
    (root / "scene/depth/manifest.json").write_text("{}")
    (root / "object").mkdir(parents=True, exist_ok=True)
    for name in ("object_layout_preview.png", "object_detect_preview.png", "object_cutout_preview.png"):
        (root / "object" / name).write_bytes(b"png")
    if viewer:
        (root / "web-demo").mkdir(parents=True, exist_ok=True)
        (root / "web-demo/viewer.json").write_text(json.dumps({"status": "generated"}))
    (root / "export_manifest.json").write_text(json.dumps({"status": "complete", "missing": [],
                                                           "scene_roles": {}, "object_version_2": {}}))


class ComparisonTests(unittest.TestCase):
    def test_ply_header_reports_counts_and_colour(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.ply"
            write_ply(path, vertices=12, faces=34, colour=True)
            header = adapter.ply_header(path)
            self.assertEqual((header["vertices"], header["faces"]), (12, 34))
            self.assertTrue(header["colour"])
            write_ply(path, vertices=1, faces=0, colour=False)
            self.assertFalse(adapter.ply_header(path)["colour"])

    def test_glb_summary_counts_triangles_and_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.glb"
            write_glb(path, tris=7, attributes=("POSITION", "COLOR_0"), images=2)
            summary = adapter.glb_summary(path)
            self.assertEqual(summary["tris"], 7)
            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["attributes"], ["COLOR_0", "POSITION"])

    def test_describe_collects_scene_objects_collisions_and_parts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "delivery"
            make_delivery(root, objects=("bed", "lamp"), parts=("bed",), viewer=True)
            report = adapter.describe(root)
            self.assertTrue(report["has_manifest"])
            self.assertEqual(report["scene"]["scene/mesh.ply"]["faces"], 200)
            self.assertTrue(report["scene"]["scene/mesh.ply"]["colour"])
            self.assertEqual(report["scene"]["object/background.glb"]["tris"], 1000)
            self.assertEqual(sorted(report["objects"]["object_version_2"]), ["bed", "lamp"])
            self.assertEqual(report["objects"]["object_version_2"]["bed"]["collision_hulls"], 2)
            self.assertEqual(report["objects"]["parts"]["count"], 1)
            self.assertIn("scene/scene_overview.jpg", report["previews"])
            self.assertTrue(report["extras"].get("viewer"))

    def test_a_description_taken_on_another_host_can_be_compared_here(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "old"
            candidate = Path(folder) / "new"
            make_delivery(reference, objects=("bed", "lamp"), parts=("bed",), viewer=True)
            make_delivery(candidate, objects=("bed", "lamp"), parts=("bed",), viewer=True)
            captured = Path(folder) / "candidate.json"
            import contextlib
            import io

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(adapter.main(["--describe", str(candidate)]), 0)
            captured.write_text(buffer.getvalue())
            # a description file stands in for the delivery it was taken from
            self.assertEqual(adapter.describe_any(captured), adapter.describe(candidate))
            report = Path(folder) / "comparison.json"
            self.assertEqual(adapter.main(["--reference", str(reference), "--candidate", str(captured),
                                           "--report", str(report), "--require-contract"]), 0)
            loaded = json.loads(report.read_text())
            self.assertEqual(loaded["candidate"]["total_bytes"], adapter.describe(candidate)["total_bytes"])
            self.assertFalse(loaded["candidate"]["contract"]["violations"])

    def test_a_json_file_that_is_not_a_description_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "notes.json"
            path.write_text(json.dumps({"hello": "world"}))
            with self.assertRaises(adapter.PipelineError):
                adapter.describe_any(path)
        self.assertEqual(adapter.main(["--reference", str(path), "--candidate", str(path)]), 2)

    def test_the_table_shows_the_gap_between_two_deliveries(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "old"
            candidate = Path(folder) / "new"
            make_delivery(reference, objects=("bed", "lamp", "plant"), parts=("bed",))
            make_delivery(candidate, objects=("bed",), parts=(), colour=False)
            text = adapter.table(adapter.describe(reference), adapter.describe(candidate))
            self.assertIn("object_version_2 objects", text)
            self.assertIn("3: bed, lamp, plant", text)
            self.assertIn("1: bed", text)
            self.assertIn("no-colour", text)
            self.assertIn("parts", text)

    def test_the_contract_flags_a_delivery_that_is_missing_a_splat(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "delivery"
            make_delivery(root, objects=("bed", "lamp"), parts=("bed",), viewer=True)
            report = adapter.describe(root)
            self.assertEqual(report["contract"]["status"], "contract_met", report["contract"]["violations"])
            (root / "object/object_version_2/lamp/asset_splat.ply").unlink()
            broken = adapter.describe(root)
            self.assertIn("every completed object has a splat", broken["contract"]["violations"])
            self.assertEqual(broken["contract"]["status"], "contract_violated")

    def test_the_contract_flags_a_missing_scene_layer_and_viewer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "delivery"
            make_delivery(root, objects=("bed",), parts=("bed",))
            (root / "scene/point_cloud_3dgs.ply").unlink()
            report = adapter.describe(root)
            self.assertIn("scene gaussians", report["contract"]["violations"])
            self.assertIn("viewer project", report["contract"]["violations"])

    def test_require_contract_returns_a_failure_code(self):
        with tempfile.TemporaryDirectory() as folder:
            reference = Path(folder) / "old"
            candidate = Path(folder) / "new"
            make_delivery(reference, objects=("bed",), parts=("bed",), viewer=True)
            make_delivery(candidate, objects=("bed",), parts=("bed",), viewer=True)
            (candidate / "object/object_version_2/bed/asset_splat.ply").unlink()
            self.assertEqual(adapter.main(["--reference", str(reference), "--candidate", str(candidate)]), 0)
            self.assertEqual(adapter.main(["--reference", str(reference), "--candidate", str(candidate),
                                           "--require-contract"]), 1)

    def test_a_missing_delivery_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(adapter.PipelineError):
                adapter.describe(Path(folder) / "absent")
        self.assertEqual(adapter.main(["--reference", str(Path(folder) / "absent"),
                                       "--candidate", str(Path(folder) / "absent")]), 2)


if __name__ == "__main__":
    unittest.main()
