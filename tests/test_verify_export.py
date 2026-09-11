from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_export.py"
SPEC = importlib.util.spec_from_file_location("verify_export_adapter", SCRIPT)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def write_manifest(root: Path, digest: str, status: str = "complete", missing=None) -> None:
    manifest = {"schema_version": "1.0", "status": status, "missing": missing or [],
                "scene_roles": {"scene/mesh.ply": {"sha256": digest}},
                "object_version_1": {}, "object_version_2": {}, "recomposition": {}}
    (root / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class ExportVerificationTests(unittest.TestCase):
    def run_verify(self, root: Path) -> tuple[int, dict]:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = verify.main(["--export-root", str(root), "--skip-geometry"])
        return code, json.loads(stream.getvalue())

    def test_matching_hashes_pass_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scene").mkdir()
            asset = root / "scene" / "mesh.ply"
            asset.write_bytes(b"exported geometry")
            write_manifest(root, hashlib.sha256(asset.read_bytes()).hexdigest())
            code, report = self.run_verify(root)
            self.assertEqual(code, 0)
            self.assertTrue(report["valid"])
            asset.write_bytes(b"tampered geometry")
            code, report = self.run_verify(root)
            self.assertEqual(code, 1)
            self.assertFalse(report["valid"])
            self.assertTrue(any("sha256 mismatch" in issue for issue in report["issues"]))

    def test_missing_file_incomplete_status_and_absent_manifest_are_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scene").mkdir()
            asset = root / "scene" / "mesh.ply"
            asset.write_bytes(b"data")
            write_manifest(root, hashlib.sha256(b"data").hexdigest())
            asset.unlink()
            code, report = self.run_verify(root)
            self.assertEqual(code, 1)
            self.assertTrue(any("missing file" in issue for issue in report["issues"]))
            asset.write_bytes(b"data")
            write_manifest(root, hashlib.sha256(b"data").hexdigest(), status="partial", missing=["object_version_2"])
            code, report = self.run_verify(root)
            self.assertEqual(code, 1)
            self.assertTrue(any("status" in issue for issue in report["issues"]))
            (root / "export_manifest.json").unlink()
            code, report = self.run_verify(root)
            self.assertEqual(code, 1)
            self.assertIn("missing export_manifest.json", report["issues"])


if __name__ == "__main__":
    unittest.main()
