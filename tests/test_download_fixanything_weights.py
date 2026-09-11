import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import unittest


SPEC = importlib.util.spec_from_file_location("download_fixanything_weights", Path(__file__).resolve().parents[1] / "scripts/download_fixanything_weights.py")
provider = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provider)

try:
    import torch
    from safetensors.torch import load
except ImportError:
    torch = None


class MirrorIdentityTests(unittest.TestCase):
    def test_official_mirror_is_pinned_to_file_revision_and_same_sha256(self):
        source = {"filename": "weights.safetensors", "size": 20, "sha256": "a" * 64}
        mirror = {"Size": 20, "Sha256": "a" * 64, "Revision": "b" * 40}
        selected = provider.validate_mirror_record(source, mirror)
        self.assertEqual(selected["sha256"], source["sha256"])
        self.assertEqual(selected["transport"], "modelscope-direct")
        self.assertIn("Revision=" + "b" * 40, selected["transport_url"])

    def test_mirror_checksum_or_size_mismatch_is_rejected(self):
        source = {"filename": "weights.safetensors", "size": 20, "sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            provider.validate_mirror_record(source, {"Size": 20, "Sha256": "c" * 64, "Revision": "b" * 40})
        with self.assertRaisesRegex(ValueError, "size or identity mismatch"):
            provider.validate_mirror_record(source, {"Size": 21, "Sha256": "a" * 64, "Revision": "b" * 40})


@unittest.skipIf(torch is None, "torch and safetensors required")
class StreamingWeightsTests(unittest.TestCase):
    def fixture(self):
        tensor = torch.tensor([0, 1, -3.25, 1.00390625, 1.01171875], dtype=torch.float32)
        header = json.dumps({"weight": {"dtype": "F32", "shape": [5], "data_offsets": [0, 20]},
                             "__metadata__": {"format": "pt"}}, separators=(",", ":")).encode()
        header += b" " * ((-len(header)) % 8)
        return struct.pack("<Q", len(header)) + header + tensor.numpy().tobytes(), tensor

    def test_stream_conversion_preserves_keys_shapes_and_exact_torch_bf16_values(self):
        source, tensor = self.fixture()
        output = io.BytesIO()
        audit = provider.convert_safetensors(io.BytesIO(source), output, hashlib.sha256(source).hexdigest(), {"test": "fixture"})
        restored = load(output.getvalue())
        self.assertEqual(set(restored), {"weight"})
        self.assertEqual(restored["weight"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(restored["weight"], tensor.to(torch.bfloat16)))
        self.assertTrue(audit["source_sha256_verified"])
        self.assertEqual(audit["source_bytes"], len(source))

    def test_hash_mismatch_rejects_converted_stream(self):
        source, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
            provider.convert_safetensors(io.BytesIO(source), io.BytesIO(), "0" * 64, {})

    def test_truncated_tensor_rejects_stream(self):
        source, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "truncated source"):
            provider.convert_safetensors(io.BytesIO(source[:-1]), io.BytesIO(), hashlib.sha256(source).hexdigest(), {})


if __name__ == "__main__":
    unittest.main()
