from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import trimesh

from scripts.build_coacd_collider import build_coacd_collider


def test_build_coacd_collider_binds_visual_input_and_convex_obj(tmp_path: Path) -> None:
    source = tmp_path / "visual.glb"
    trimesh.creation.icosphere(subdivisions=2).export(source)
    stub = tmp_path / "coacd"
    stub.write_text(
        f"""#!{sys.executable}
import sys
from pathlib import Path
import trimesh
args = sys.argv[1:]
out = Path(args[args.index('-o') + 1])
left = trimesh.creation.box(extents=(1, 2, 3))
right = trimesh.creation.box(extents=(1, 2, 3))
left.apply_translation((-2, 0, 0))
right.apply_translation((2, 0, 0))
trimesh.util.concatenate((left, right)).export(out)
""",
        encoding="utf-8",
    )
    os.chmod(stub, 0o755)
    output = tmp_path / "collider.obj"
    receipt_path = tmp_path / "receipt.json"
    receipt = build_coacd_collider(
        source,
        output,
        receipt_path,
        coacd_executable=stub,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert receipt["status"] == "technical_passed_visual_pending"
    assert receipt["promotion_allowed"] is False
    assert receipt["output_collider"]["collider_topology"] == "convex_decomposition"
    assert receipt["output_collider"]["provenance"] == {"decomposition": "coacd"}
    assert receipt["collider_audit"]["part_count"] == 2
    assert receipt["collider_audit"]["technical_status"] == "passed"
    assert receipt["collider_audit"]["parts"][0]["convex"] is True
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
