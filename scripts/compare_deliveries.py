#!/usr/bin/env python3
"""Compare two delivery directories item by item.

The objective is to make a delivery match an earlier one, which is only
checkable if the differences are measured rather than asserted. This walks two
export roots and reports, per delivery: total size, scene layers with their
vertex/face counts and whether they carry colour, the object assets in each
version with their mesh and splat sizes, collision hull counts, advertised
previews and whether they exist, and the viewer project.

    python scripts/compare_deliveries.py --reference OLD --candidate NEW

Either side may also be a JSON description previously produced with
``--describe``, which is how a delivery measured on the machine that holds it
is compared here without copying hundreds of megabytes between hosts.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from world_modeling.engine import PipelineError, read_json, write_json  # noqa: E402


def ply_header(path: Path, limit: int = 4096) -> dict:
    """Element counts and whether the vertex table carries colour."""
    with path.open("rb") as handle:
        head = handle.read(limit).decode("latin-1", errors="replace")
    header = {"vertices": None, "faces": None, "colour": False}
    for line in head.splitlines():
        if line.startswith("element vertex"):
            header["vertices"] = int(line.split()[-1])
        elif line.startswith("element face"):
            header["faces"] = int(line.split()[-1])
        elif line.startswith("property") and line.split()[-1] in {"red", "green", "blue"}:
            header["colour"] = True
        elif line.startswith("end_header"):
            break
    return header


def glb_summary(path: Path) -> dict:
    """Triangle count, embedded images and vertex attributes of a GLB."""
    data = path.read_bytes()
    offset, summary = 12, {"tris": 0, "attributes": [], "images": 0}
    while offset + 8 <= len(data):
        length, kind = struct.unpack("<II", data[offset:offset + 8])
        chunk = data[offset + 8:offset + 8 + length]
        if kind == 0x4E4F534A:
            document = json.loads(chunk.decode("utf-8"))
            accessors = document.get("accessors", [])
            attributes = set()
            for mesh in document.get("meshes", []):
                for primitive in mesh.get("primitives", []):
                    attributes |= set(primitive.get("attributes", {}))
                    indices = primitive.get("indices")
                    if indices is not None:
                        summary["tris"] += accessors[indices]["count"] // 3
            summary["attributes"] = sorted(attributes)
            summary["images"] = len(document.get("images", []))
        offset += 8 + length
    return summary


def mesh_summary(path: Path) -> dict:
    if path.suffix.lower() == ".ply":
        return ply_header(path)
    if path.suffix.lower() in {".glb", ".gltf"}:
        return glb_summary(path)
    return {}


def describe(root: Path) -> dict:
    if not root.is_dir():
        raise PipelineError(f"delivery directory not found: {root}")
    manifest_path = root / "export_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    report = {"root": str(root), "total_bytes": total, "has_manifest": manifest_path.is_file(),
              "scene": {}, "objects": {}, "collisions": {}, "previews": {}, "extras": {}}

    scene_root = root / "scene"
    depth_frames = 0
    depth_bytes = 0
    if scene_root.is_dir():
        for path in sorted(scene_root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json" or path.name.startswith("."):
                continue
            if path.parent.name == "depth":
                # One row for the depth maps: they are counted, not listed.
                depth_frames += 1
                depth_bytes += path.stat().st_size
                continue
            entry = {"bytes": path.stat().st_size}
            entry.update(mesh_summary(path))
            report["scene"][str(path.relative_to(root))] = entry
    if depth_frames:
        report["scene"]["scene/depth/*.png"] = {"bytes": depth_bytes, "frames": depth_frames}

    for version in ("object_version_1", "object_version_2"):
        directory = root / "object" / version
        if not directory.is_dir():
            continue
        for object_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
            entry = {"files": {}}
            for path in sorted(object_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = str(path.relative_to(object_dir))
                entry["files"][relative] = path.stat().st_size
            hulls = sorted(object_dir.glob("collision/hull_*.obj"))
            if hulls:
                entry["collision_hulls"] = len(hulls)
            for candidate in (object_dir / "asset_mesh.glb", object_dir / f"{object_dir.name}.glb"):
                if candidate.is_file():
                    entry["mesh"] = mesh_summary(candidate)
                    break
            report["objects"].setdefault(version, {})[object_dir.name] = entry

    parts = root / "object" / "parts"
    if parts.is_dir():
        documents = sorted(parts.rglob("*.ply"))
        report["objects"]["parts"] = {"count": len(documents),
                                      "objects": sorted({path.parent.name for path in documents})}

    background = root / "object" / "background.glb"
    if background.is_file():
        report["scene"]["object/background.glb"] = {"bytes": background.stat().st_size,
                                                    **glb_summary(background)}

    for name, relative in (("previews", "qa/previews.json"), ("viewer", "web-demo/viewer.json")):
        document = root / relative
        if document.is_file():
            report["extras"][name] = read_json(document)
    viewer_root = root / "web-demo"
    if viewer_root.is_dir():
        report["extras"]["viewer_project"] = {
            "files": sum(1 for path in viewer_root.rglob("*") if path.is_file()),
            "built": (viewer_root / "dist" / "index.html").is_file(),
            "generated_by_this_repo": (viewer_root / "viewer.json").is_file(),
        }
    report["previews"] = advertises_previews(root, manifest)
    report["contract"] = delivery_contract(root, report)
    return report


def advertises_previews(root: Path, manifest: dict) -> dict:
    """Every preview the export claims, with whether the file is really there."""
    known = {"scene/scene_overview.jpg", "object/object_layout_preview.png",
             "object/object_detect_preview.png", "object/object_cutout_preview.png"}
    previews = {}
    for relative in sorted(known):
        if (root / relative).is_file():
            previews[relative] = True
    for directory in sorted((root / "object" / "object_version_2").glob("*")) if (
            root / "object" / "object_version_2").is_dir() else []:
        for name in ("gaussian_turntable_overview.jpg", "glb_mesh_overview.jpg"):
            if (directory / name).is_file():
                previews[f"object/object_version_2/{directory.name}/{name}"] = True
    return previews


def delivery_contract(root: Path, report: dict) -> dict:
    """What a delivery must contain to be usable, as explicit checks.

    A comparison says how two deliveries differ; this says whether either one is
    complete enough to hand over. It is deliberately about presence and
    structure, not about visual quality, which no script can judge.
    """
    checks = {}

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    scene = report["scene"]
    check("scene surface", any(key.endswith("mesh.ply") for key in scene))
    check("scene gaussians", any("3dgs" in key for key in scene))
    check("scene depth maps", any(key.startswith("scene/depth/") for key in scene))
    versions = report["objects"]
    v2 = versions.get("object_version_2") or {}
    v1 = versions.get("object_version_1") or {}
    check("completed objects", bool(v2), f"{len(v2)} object(s)")
    missing_mesh = sorted(name for name, entry in v2.items() if not entry.get("mesh"))
    check("every completed object has a mesh", not missing_mesh, ", ".join(missing_mesh))
    missing_splat = sorted(name for name, entry in v2.items()
                           if not any(name_ == "asset_splat.ply" for name_ in entry.get("files", {})))
    check("every completed object has a splat", not missing_splat, ", ".join(missing_splat))
    no_hulls = sorted(name for name, entry in v2.items() if not entry.get("collision_hulls"))
    check("every completed object has collision hulls", not no_hulls, ", ".join(no_hulls))
    textured = sorted(name for name, entry in v1.items()
                      if not any(name_.endswith(".mtl") for name_ in entry.get("files", {})))
    check("every textured asset has a material", not textured, ", ".join(textured))
    parts = versions.get("parts") or {}
    check("observed parts", bool(parts.get("count")), f"{parts.get('count', 0)} part(s)")
    check("qa previews", len(report["previews"]) >= 4, f"{len(report['previews'])} present")
    viewer = report["extras"].get("viewer_project")
    check("viewer project", bool(viewer), "present" if viewer else "absent")
    violations = sorted(name for name, entry in checks.items() if not entry["passed"])
    return {"checks": checks, "violations": violations,
            "status": "contract_met" if not violations else "contract_violated"}


def megabytes(value: int | None) -> str:
    return "—" if value is None else f"{value / 1e6:.1f}MB"


def table(reference: dict, candidate: dict) -> str:
    lines = ["| item | reference | candidate |", "|---|---|---|"]

    def row(label: str, left, right) -> None:
        lines.append(f"| {label} | {left} | {right} |")

    row("total size", megabytes(reference["total_bytes"]), megabytes(candidate["total_bytes"]))
    for key in sorted(set(reference["scene"]) | set(candidate["scene"])):
        left, right = reference["scene"].get(key), candidate["scene"].get(key)
        def render(entry):
            if not entry:
                return "missing"
            if entry.get("frames"):
                return f"{entry['frames']} frames {megabytes(entry['bytes'])}"
            if entry.get("faces"):
                return f"{entry['faces']:,} faces{' colour' if entry.get('colour') else ' no-colour'} {megabytes(entry['bytes'])}"
            if entry.get("tris"):
                return f"{entry['tris']:,} tris {entry.get('attributes')} {megabytes(entry['bytes'])}"
            if entry.get("vertices"):
                return f"{entry['vertices']:,} verts {megabytes(entry['bytes'])}"
            return megabytes(entry.get("bytes"))
        row(f"scene `{key}`", render(left), render(right))
    for version in ("object_version_1", "object_version_2", "parts"):
        left = reference["objects"].get(version) or {}
        right = candidate["objects"].get(version) or {}
        if version == "parts":
            row("parts", f"{left.get('count', 0)} parts {left.get('objects', [])}",
                f"{right.get('count', 0)} parts {right.get('objects', [])}")
            continue
        row(f"{version} objects", f"{len(left)}: {', '.join(sorted(left)) or '—'}",
            f"{len(right)}: {', '.join(sorted(right)) or '—'}")
    left_previews = reference["previews"]
    right_previews = candidate["previews"]
    row("previews present", len(left_previews), len(right_previews))
    def viewer_state(entry):
        project = entry["extras"].get("viewer_project")
        if not project:
            return "no"
        origin = "generated" if project["generated_by_this_repo"] else "foreign"
        return f"{origin}, {'built' if project['built'] else 'source only'} ({project['files']} files)"

    row("viewer project", viewer_state(reference), viewer_state(candidate))
    row("contract", reference["contract"]["status"], candidate["contract"]["status"])
    for name in sorted(set(reference["contract"]["checks"]) | set(candidate["contract"]["checks"])):
        left = reference["contract"]["checks"].get(name, {}).get("passed")
        right = candidate["contract"]["checks"].get(name, {}).get("passed")
        if left is not None and right is not None and left == right:
            continue
        row(f"  {name}", "pass" if left else f"fail ({reference['contract']['checks'][name]['detail']})",
            "pass" if right else f"fail ({candidate['contract']['checks'][name]['detail']})")
    return "\n".join(lines)


def describe_any(path: Path) -> dict:
    """Describe a delivery directory, or load a description taken elsewhere."""
    if path.is_file() and path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "contract" not in payload or "total_bytes" not in payload:
            raise PipelineError(f"{path} is not a delivery description")
        return payload
    return describe(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--describe", type=Path,
                        help="print this delivery's description as JSON and exit, so a description taken on "
                             "another host can be compared here without moving the delivery itself")
    parser.add_argument("--report", type=Path, help="write the full JSON comparison here")
    parser.add_argument("--require-contract", action="store_true",
                        help="exit non-zero unless the candidate satisfies the delivery contract")
    args = parser.parse_args(argv)
    if args.describe is not None:
        try:
            print(json.dumps(describe_any(args.describe), indent=2, sort_keys=True))
        except (PipelineError, OSError, ValueError, KeyError) as error:
            print(f"compare_deliveries failed: {error}", file=sys.stderr)
            return 2
        return 0
    if args.reference is None or args.candidate is None:
        print("compare_deliveries failed: --reference and --candidate are both required", file=sys.stderr)
        return 2
    try:
        reference = describe_any(args.reference)
        candidate = describe_any(args.candidate)
    except (PipelineError, OSError, ValueError, KeyError) as error:
        print(f"compare_deliveries failed: {error}", file=sys.stderr)
        return 2
    comparison = {"schema_version": "1.0", "kind": "video2world-modeling.delivery_comparison",
                  "reference": reference, "candidate": candidate}
    if args.report:
        write_json(args.report, comparison)
    print(table(reference, candidate))
    violations = candidate["contract"]["violations"]
    if violations:
        print("candidate contract violations: " + ", ".join(violations), file=sys.stderr)
    if args.require_contract and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
