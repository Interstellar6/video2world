from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "object_lifting.py"
SPEC = importlib.util.spec_from_file_location("object_lifting_adapter", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def options(task):
    return argparse.Namespace(
        task_dir=task, inputs=task / "inputs.json", outputs=task / "provider-artifacts.json",
        confidence_min=0.5, voxel_size=0.04, overlap_radius_voxels=3,
        min_overlap_ratio=0.15, min_scene_support=0.2, max_visible_disagreement=0.25,
        min_views=2, min_positive_views=2, min_mask_pixels=16, min_object_gaussians=10,
        depth_relative_tolerance=0.05, depth_absolute_tolerance=0.01,
        depth_scale_tolerance=0.25, min_depth_agreement=0.2,
        max_observation_points=20000, max_scene_samples=200000,
        composite_policy={},
    )


class PathTests(unittest.TestCase):
    def test_lifting_owns_resolution_of_candidate_roles(self):
        from world_modeling.modules.object_lifting import SPEC as lifting

        self.assertTrue({"component_mask_candidates", "physical_instance_hypotheses"} <= set(lifting.inputs))
        self.assertFalse({"component_masks", "physical_instance_tracks"} & set(lifting.inputs))
        self.assertTrue({"component_masks", "physical_instance_tracks"} <= set(lifting.output_names))

    def test_composite_policy_requires_explicit_group_and_relation(self):
        policy = {"bed": {"pillow": "bedding", "headboard": "structural_part"}}
        self.assertEqual(adapter.composite_policy(policy), policy)
        self.assertEqual(adapter.composite_policy(json.dumps(policy)), policy)
        self.assertEqual(adapter.composite_policy({}), {})
        for value in (None, [], True, {"bed": ["pillow"]}, {"bed": {"pillow": "nearby"}},
                      {"chair": {"cushion": "bedding"}}, {"bed": {"../pillow": "bedding"}}):
            with self.subTest(value=value), self.assertRaises((adapter.LiftingError, ValueError)):
                adapter.composite_policy(value)

    def test_task_boundary_including_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / "task"
            task.mkdir()
            outside = root / "outside.npy"
            outside.write_bytes(b"not loaded")
            (task / "link.npy").symlink_to(outside)
            for path in ("../outside.npy", "link.npy"):
                with self.assertRaises(adapter.LiftingError):
                    adapter.task_path(task, path)


GEOMETRY_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "PIL", "plyfile"))


@unittest.skipUnless(GEOMETRY_DEPS, "requires numpy, scipy, Pillow, and plyfile")
class GeometricLiftingTests(unittest.TestCase):
    def make_task(self, task: Path, *, scale=1, mixed=False):
        import numpy as np
        from PIL import Image
        from plyfile import PlyData, PlyElement

        inputs = task / "inputs"
        inputs.mkdir()
        grid = np.arange(-1.6, 1.6001, 0.025)
        x, y = np.meshgrid(grid, grid)
        positions = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 2)))
        rows = np.zeros(len(positions), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4"), ("f_dc_0", "f4"), ("scale_0", "f4"), ("rot_0", "f4"), ("source_index", "i4")])
        for index, name in enumerate(("x", "y", "z")):
            rows[name] = positions[:, index]
        rows["opacity"], rows["rot_0"], rows["scale_0"] = 1, 1, -4
        rows["f_dc_0"] = np.linspace(-1, 1, len(rows))
        rows["source_index"] = np.arange(len(rows))
        PlyData([PlyElement.describe(rows, "vertex")], comments=["analytic trained-Gaussian attribute fixture"]).write(str(inputs / "scene.ply"))
        mesh_vertices = np.array([(-2., -2., 2.), (2., -2., 2.), (2., 2., 2.), (-2., 2., 2.)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        faces = np.array([([0, 1, 2],), ([0, 2, 3],)], dtype=[("vertex_indices", "i4", (3,))])
        PlyData([PlyElement.describe(mesh_vertices, "vertex"), PlyElement.describe(faces, "face")]).write(str(inputs / "tsdf.ply"))
        calibration = np.array([[40., 0, 31.5], [0, 40., 31.5], [0, 0, 1]])
        cameras, depths, masks, tracks = [], [], [], []
        for index, center in enumerate((-.3, .3)):
            frame_id = str(index)
            pose = np.eye(4)
            pose[0, 3] = -center
            camera = {"frame_id": frame_id, "intrinsics": calibration.tolist(), "world_to_camera": pose.tolist(), "width": 64, "height": 64}
            cameras.append(camera)
            np.save(inputs / f"depth_{index}.npy", np.full((64, 64), 2 * scale, dtype=np.float32))
            np.save(inputs / f"confidence_{index}.npy", np.ones((64, 64), dtype=np.float32))
            depths.append({**camera, "depth_path": f"inputs/depth_{index}.npy", "confidence_path": f"inputs/confidence_{index}.npy"})
            yy, xx = np.indices((64, 64))
            world_x = (xx - 31.5) / 40 * 2 + center
            world_y = (yy - 31.5) / 40 * 2
            for object_id, lower, upper in (("left", -.8, -.25), ("right", .25, .8)):
                mask = (world_x > lower) & (world_x < upper) & (abs(world_y) < .4)
                path = f"inputs/{object_id}_{index}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(task / path)
                digest = adapter.sha256(task / path)
                masks.append({"frame_id": frame_id, "object_id": object_id, "component_id": "surface", "mask_path": path,
                              "mask_sha256": digest, "score": 1.0, "component_group_id": "surface",
                              "observation_id": f"obs_{object_id}_{index}_surface", "identity_scope": "frame_local_observation",
                              "membership_relation": "structural_part", "geometry_validated": False})
                masks.append({"frame_id": frame_id, "object_id": object_id, "component_id": "__object__", "mask_path": path,
                              "mask_sha256": digest, "score": 1.0})
        if mixed:
            tracks = [{"object_id": "mixed", "category": "bed", "association_status": "vlm_identity_hypothesis", "observations": [{"frame_id": "0", "mask_path": "inputs/left_0.png"}, {"frame_id": "1", "mask_path": "inputs/right_1.png"}]}]
            masks = [{**record, "object_id": "mixed"} for record in masks
                     if record["component_id"] == "__object__" and
                     (record["frame_id"], record["object_id"]) in (("0", "left"), ("1", "right"))]
        else:
            tracks = [{"object_id": name, "category": "bed", "association_status": "vlm_identity_hypothesis", "observations": [{"frame_id": str(index), "mask_path": f"inputs/{name}_{index}.png"} for index in range(2)]} for name in ("left", "right")]
        documents = {"cameras": {"frames": cameras, "units": "colmap_reconstruction"}, "scene_depth": {"frames": depths, "units": "colmap_reconstruction", "depth_kind": "camera_z"},
                     "component_mask_candidates": {"frames": cameras, "masks": masks},
                     "physical_instance_hypotheses": {"tracks": tracks, "geometry_validated": False}}
        artifacts = {}
        for name, document in documents.items():
            adapter.write_json(inputs / f"{name}.json", document)
            artifacts[name] = {"path": f"inputs/{name}.json", "evidence": "observed"}
        artifacts["scene_gaussian_ply"] = {"path": "inputs/scene.ply", "evidence": "observed"}
        artifacts["scene_tsdf_mesh"] = {"path": "inputs/tsdf.ply", "evidence": "observed"}
        adapter.write_json(task / "inputs.json", {"module": "object_lifting", "inputs": artifacts})
        return rows

    def make_composite_task(self, task, *, component_frames=(0, 1), relations=("bedding", "bedding"), group="pillow", duplicate_first=False, displaced_second=False):
        import numpy as np
        from PIL import Image

        rows = self.make_task(task)
        candidate_path = task / "inputs/component_mask_candidates.json"
        candidate = adapter.read_json(candidate_path)
        candidate["masks"] = [record for record in candidate["masks"]
                              if record["object_id"] == "left" and record["component_id"] == "__object__"]
        raw_parents = {record["frame_id"]: dict(record) for record in candidate["masks"]}
        hypotheses_path = task / "inputs/physical_instance_hypotheses.json"
        hypotheses = adapter.read_json(hypotheses_path)
        hypotheses["tracks"] = [track for track in hypotheses["tracks"] if track["object_id"] == "left"]
        yy, xx = np.indices((64, 64))
        for index in component_frames:
            center = (-.3, .3)[index]
            world_x = (xx - 31.5) / 40 * 2 + center
            world_y = (yy - 31.5) / 40 * 2
            lower, upper = (.7, 1.25) if index == 1 and displaced_second else (-.8, -.25)
            mask = (world_x > lower) & (world_x < upper) & (world_y > .35) & (world_y < .75)
            path = f"inputs/{group}_{index}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(task / path)
            record = {"frame_id": str(index), "object_id": "left", "component_group_id": group,
                      "component_id": f"{group}__{(0, 7)[index]:03d}", "observation_id": f"obs_{group}_{index}",
                      "identity_scope": "frame_local_observation", "instance_mode": "all",
                      "membership_relation": relations[index], "mask_path": path, "mask_sha256": adapter.sha256(task / path),
                      "score": .8, "geometry_validated": False}
            candidate["masks"].append(record)
            if duplicate_first and index == 0:
                candidate["masks"].append({**record, "component_id": f"{group}__012", "observation_id": f"obs_{group}_0_duplicate"})
        adapter.write_json(candidate_path, candidate)
        adapter.write_json(hypotheses_path, hypotheses)
        return rows, raw_parents

    def assert_resolved_contract(self, task, result):
        self.assertEqual(set(result["outputs"]), {"isolated_object_ply", "carved_scene_ply", "lifting_report", "component_masks", "physical_instance_tracks"})
        report = adapter.read_json(task / result["outputs"]["lifting_report"]["path"])
        documents = {}
        for role in ("component_masks", "physical_instance_tracks"):
            record = report["resolved_files"][role]
            self.assertEqual(record["path"], result["outputs"][role]["path"])
            self.assertEqual(record["sha256"], adapter.sha256(task / record["path"]))
            documents[role] = adapter.read_json(task / record["path"])
            self.assertEqual(documents[role]["acceptance_authority"], result["outputs"]["lifting_report"]["path"])
        parents = {(record["object_id"], str(record["frame_id"])): record for record in documents["component_masks"]["masks"]
                   if record["component_id"] == "__object__"}
        for record in documents["component_masks"]["masks"]:
            self.assertEqual(record["mask_sha256"], adapter.sha256(task / record["mask_path"]))
        for track in documents["physical_instance_tracks"]["tracks"]:
            for item in track["observations"]:
                parent = parents[track["object_id"], str(item["frame_id"])]
                self.assertEqual(item["mask_path"], parent["mask_path"])
                self.assertEqual(item["mask_sha256"], parent["mask_sha256"])
        self.assertFalse(documents["physical_instance_tracks"]["complete_recall_claimed"])
        return report, documents["component_masks"], documents["physical_instance_tracks"]

    def test_two_objects_lift_and_carve_preserving_every_gaussian_attribute(self):
        import numpy as np
        from plyfile import PlyData

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            original = self.make_task(task)
            result = adapter.run(options(task))
            self.assert_resolved_contract(task, result)
            manifest = adapter.read_json(task / result["outputs"]["isolated_object_ply"]["path"])
            self.assertEqual([item["object_id"] for item in manifest["objects"]], ["left", "right"])
            selected_indices = []
            for item in manifest["objects"]:
                lifted = PlyData.read(str(task / item["ply_path"]))["vertex"].data
                np.testing.assert_array_equal(lifted, original[lifted["source_index"]])
                self.assertGreater(len(lifted), 10)
                self.assertTrue(item["parts"])
                selected_indices.extend(lifted["source_index"].tolist())
            self.assertEqual(len(set(selected_indices)), len(selected_indices))
            background = PlyData.read(str(task / result["outputs"]["carved_scene_ply"]["path"]))["vertex"].data
            np.testing.assert_array_equal(background, original[background["source_index"]])
            self.assertFalse(set(selected_indices) & set(background["source_index"]))
            self.assertEqual(len(selected_indices) + len(background), len(original))
            report = adapter.read_json(task / result["outputs"]["lifting_report"]["path"])
            self.assertEqual(report["status"], "passed")
            self.assertTrue(all(item["accepted"] for item in report["tracks"]))

    def test_geometric_two_view_policy_member_extends_parent_and_keeps_resolved_hashes_consistent(self):
        import numpy as np
        from PIL import Image
        from plyfile import PlyData

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            rows, raw_parents = self.make_composite_task(task)
            args = options(task)
            args.composite_policy = {"bed": {"pillow": "bedding"}}
            result = adapter.run(args)
            report, masks, tracks = self.assert_resolved_contract(task, result)
            self.assertEqual(report["status"], "passed")
            member, = report["tracks"][0]["component_association"]["tracks"]
            self.assertTrue(member["geometry_validated"])
            self.assertTrue(member["composition_allowed"])
            self.assertEqual(member["member_ids"], ["obs_pillow_0", "obs_pillow_1"])
            self.assertEqual(member["geometry_report"]["reasons"], [])
            group, = report["tracks"][0]["component_association"]["semantic_groups"]
            self.assertTrue(group["group_geometry_validated"])
            self.assertTrue(group["parent_union_allowed"])
            self.assertFalse(group["geometry_validated"])
            self.assertFalse(group["individual_identity_confirmed"])
            self.assertFalse(group["part_ply_export_allowed"])
            self.assertEqual(group["identity_scope"], "semantic_component_group")
            self.assertEqual(group["unconfirmed_individual_member_ids"], [])
            for key, value in group["geometry_thresholds"].items():
                self.assertEqual(value, getattr(args, key))
            parts = [record for record in masks["masks"] if record["component_id"] != "__object__"]
            self.assertEqual(len(parts), 2)
            self.assertEqual(len({record["component_id"] for record in parts}), 1)
            self.assertEqual({record["source_component_id"] for record in parts}, {"pillow__000", "pillow__007"})
            for parent in (record for record in masks["masks"] if record["component_id"] == "__object__"):
                raw = raw_parents[parent["frame_id"]]
                self.assertEqual(adapter.sha256(task / raw["mask_path"]), raw["mask_sha256"])
                self.assertNotEqual(parent["mask_path"], raw["mask_path"])
                with Image.open(task / raw["mask_path"]) as image:
                    original = np.asarray(image) > 0
                with Image.open(task / parent["mask_path"]) as image:
                    composed = np.asarray(image) > 0
                self.assertTrue(np.all(composed[original]))
                self.assertGreater(int((composed & ~original).sum()), 0)
                self.assertEqual(parent["composition"]["added_source_pixels"], int((composed & ~original).sum()))
                self.assertEqual(len(parent["composition"]["members"]), 1)
            obj, = adapter.read_json(task / result["outputs"]["isolated_object_ply"]["path"])["objects"]
            selected = PlyData.read(str(task / obj["ply_path"]))["vertex"].data
            self.assertTrue((selected["y"] > .4).any())
            np.testing.assert_array_equal(selected, rows[selected["source_index"]])
            self.assertTrue(obj["parts"])
            self.assertTrue(all(part["minimum_positive_views"] == 2 for part in obj["parts"]))
            self.assertTrue(tracks["tracks"][0]["geometry_validated"])

    def test_geometry_alone_or_mismatched_membership_cannot_extend_parent(self):
        from plyfile import PlyData

        cases = [({}, "pillow", ("bedding", "bedding")),
                 ({"bed": {"pillow": "bedding"}}, "pillow", ("bedding", "structural_part")),
                 ({"bed": {"lamp": "structural_part"}}, "lamp", ("supported_by", "supported_by"))]
        for policy, group, relations in cases:
            with self.subTest(policy=policy, group=group, relations=relations), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                _, originals = self.make_composite_task(task, group=group, relations=relations)
                args = options(task)
                args.composite_policy = policy
                result = adapter.run(args)
                report, masks, _ = self.assert_resolved_contract(task, result)
                member, = report["tracks"][0]["component_association"]["tracks"]
                self.assertTrue(member["geometry_validated"])
                self.assertFalse(member["composition_allowed"])
                semantic, = report["tracks"][0]["component_association"]["semantic_groups"]
                self.assertTrue(semantic["group_geometry_validated"])
                self.assertFalse(semantic["composition_allowed"])
                self.assertFalse(semantic["parent_union_allowed"])
                for record in (item for item in masks["masks"] if item["component_id"] == "__object__"):
                    self.assertEqual(record["mask_path"], originals[record["frame_id"]]["mask_path"])
                    self.assertEqual(record["composition"]["added_source_pixels"], 0)
                    self.assertEqual(record["composition"]["members"], [])
                obj, = adapter.read_json(task / result["outputs"]["isolated_object_ply"]["path"])["objects"]
                selected = PlyData.read(str(task / obj["ply_path"]))["vertex"].data
                self.assertFalse((selected["y"] > .4).any())

    def test_single_view_or_geometrically_disconnected_components_do_not_extend_parent(self):
        for extra in ({"component_frames": (0,)}, {"displaced_second": True}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as folder:
                task = Path(folder)
                _, originals = self.make_composite_task(task, **extra)
                args = options(task)
                args.composite_policy = {"bed": {"pillow": "bedding"}}
                result = adapter.run(args)
                report, masks, _ = self.assert_resolved_contract(task, result)
                candidates = report["tracks"][0]["component_association"]["tracks"]
                self.assertTrue(candidates)
                self.assertTrue(all(not candidate["geometry_validated"] for candidate in candidates))
                self.assertTrue(all(not candidate["accepted"] for candidate in candidates))
                group, = report["tracks"][0]["component_association"]["semantic_groups"]
                self.assertFalse(group["group_geometry_validated"])
                self.assertFalse(group["parent_union_allowed"])
                self.assertTrue(group["unconfirmed_individual_member_ids"])
                if "component_frames" in extra:
                    self.assertEqual(group["frame_ids"], ["0"])
                    self.assertIn("insufficient_distinct_frames", group["geometry_report"]["reasons"])
                self.assertFalse(any(record["component_id"] != "__object__" for record in masks["masks"]))
                for record in masks["masks"]:
                    self.assertEqual(record["mask_path"], originals[record["frame_id"]]["mask_path"])
                    self.assertEqual(record["composition"]["added_source_pixels"], 0)

    def test_transitive_same_frame_individual_ambiguity_does_not_veto_validated_semantic_group(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            self.make_composite_task(task, duplicate_first=True)
            args = options(task)
            args.composite_policy = {"bed": {"pillow": "bedding"}}
            result = adapter.run(args)
            report, masks, _ = self.assert_resolved_contract(task, result)
            member, = report["tracks"][0]["component_association"]["tracks"]
            self.assertEqual(member["status"], "ambiguous")
            self.assertEqual(len(member["member_ids"]), 3)
            self.assertIn("same_frame_multiple_observations", member["reasons"])
            self.assertFalse(member["geometry_validated"])
            self.assertTrue(member["group_geometry_validated"])
            group, = report["tracks"][0]["component_association"]["semantic_groups"]
            self.assertTrue(group["parent_union_allowed"])
            self.assertEqual(group["unconfirmed_individual_member_ids"], member["member_ids"])
            self.assertFalse(group["individual_identity_confirmed"])
            self.assertFalse(any(record["component_id"] != "__object__" for record in masks["masks"]))
            self.assertTrue(all(record["composition"]["added_source_pixels"] > 0 for record in masks["masks"]))
            obj, = adapter.read_json(task / result["outputs"]["isolated_object_ply"]["path"])["objects"]
            self.assertEqual(obj["parts"], [])
            self.assertFalse(list(task.rglob("part_*.ply")))

    def test_group_failure_blocks_union_even_when_individual_track_passes(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            _, originals = self.make_composite_task(task)
            path = task / "inputs/component_mask_candidates.json"
            candidates = adapter.read_json(path)
            yy, xx = np.indices((64, 64))
            world_x, world_y = (xx - 31.5) / 40 * 2 + .3, (yy - 31.5) / 40 * 2
            extra = (world_x > .7) & (world_x < 1.25) & (world_y > .35) & (world_y < .75)
            mask_path = "inputs/unrelated_pillow.png"
            Image.fromarray(extra.astype(np.uint8) * 255).save(task / mask_path)
            original = next(record for record in candidates["masks"] if record.get("observation_id") == "obs_pillow_1")
            candidates["masks"].append({**original, "observation_id": "obs_unrelated_pillow", "component_id": "pillow__020",
                                        "mask_path": mask_path, "mask_sha256": adapter.sha256(task / mask_path)})
            adapter.write_json(path, candidates)
            args = options(task)
            args.composite_policy = {"bed": {"pillow": "bedding"}}
            result = adapter.run(args)
            report, masks, _ = self.assert_resolved_contract(task, result)
            association = report["tracks"][0]["component_association"]
            self.assertTrue(any(member["geometry_validated"] for member in association["tracks"]))
            group, = association["semantic_groups"]
            self.assertFalse(group["group_geometry_validated"])
            self.assertTrue(group["composition_allowed"])
            self.assertFalse(group["parent_union_allowed"])
            self.assertFalse(group["geometry_report"]["accepted"])
            self.assertIn("disconnected_geometric_identity_observations", group["geometry_report"]["reasons"])
            self.assertEqual(group["geometry_report"]["visibility_conflicts"]["conflicting_pairs"], 1)
            for record in (item for item in masks["masks"] if item["component_id"] == "__object__"):
                self.assertEqual(record["mask_path"], originals[record["frame_id"]]["mask_path"])
                self.assertEqual(record["composition"]["added_source_pixels"], 0)

    def test_observation_components_split_a_category_level_track(self):
        # One lamp hypothesis covered two disconnected groups of views; the
        # components must come back largest first with their own members.
        observations = [{"frame_id": str(index)} for index in range(5)]
        accepted = {("0", "1"), ("1", "2"), ("0", "2"), ("3", "4")}

        def fake_pair(first, second, frames, voxel_size, tolerance, args):
            key = (first["frame_id"], second["frame_id"])
            return {"accepted": key in accepted, "frame_ids": list(key), "reprojection": []}

        with patch.object(adapter, "association_pair", side_effect=fake_pair):
            components = adapter.observation_components(observations, {}, None, 1.0, 1.0, argparse.Namespace())
        self.assertEqual([[observations[i]["frame_id"] for i in group] for group in components],
                         [["0", "1", "2"], ["3", "4"]])

    def test_every_observation_is_its_own_component_when_nothing_matches(self):
        observations = [{"frame_id": str(index)} for index in range(3)]
        with patch.object(adapter, "association_pair", return_value={"accepted": False, "reprojection": []}):
            components = adapter.observation_components(observations, {}, None, 1.0, 1.0, argparse.Namespace())
        self.assertEqual([len(group) for group in components], [1, 1, 1])

    def test_a_connected_track_stays_one_component(self):
        observations = [{"frame_id": str(index)} for index in range(4)]
        with patch.object(adapter, "association_pair", return_value={"accepted": True, "reprojection": []}):
            components = adapter.observation_components(observations, {}, None, 1.0, 1.0, argparse.Namespace())
        self.assertEqual([len(group) for group in components], [4])

    def test_visibility_conflicts_are_recorded_without_rejecting_a_connected_track(self):
        # A wide-baseline pair can legitimately disagree about occluding a surface
        # it sees from the other side; identity is decided by accepted-pair
        # connectivity, so the conflict is reported and not fatal.
        import numpy as np

        class Frame:
            def __init__(self, name):
                self.frame_id = name
                self.depth = np.ones((2, 2))
                self.valid = np.ones((2, 2), dtype=bool)

        class Tree:
            def query(self, points, distance_upper_bound):
                return np.zeros(len(points)), None

        args = argparse.Namespace(min_views=2, min_mask_pixels=1, min_scene_support=0.5,
                                  overlap_radius_voxels=2, depth_relative_tolerance=0.1)
        observations = [{"frame_id": name, "points": np.zeros((1, 3)), "mask": np.ones((2, 2), dtype=bool),
                         "valid_pixels": 4} for name in ("0", "1")]
        frames = {"0": Frame("0"), "1": Frame("1")}
        pair = {"frame_ids": ["0", "1"], "accepted": True, "baseline": 1.0,
                "forward_overlap_ratio": 0.8, "backward_overlap_ratio": 0.8,
                "reprojection": [{"conflict": True, "visible_disagreement_ratio": 0.3, "positive": 10,
                                  "negative": 3, "occluded": 1, "in_front": 2}]}
        with patch.object(adapter, "association_pair", return_value=pair):
            report = adapter.association_report(observations, frames, Tree(), 0.5, 0.5, args)
        self.assertTrue(report["accepted"], report["reasons"])
        self.assertEqual(report["visibility_conflicts"]["policy"], "recorded_not_blocking")
        self.assertEqual(report["visibility_conflicts"]["conflicting_pairs"], 1)
        self.assertEqual(report["visibility_conflicts"]["total_pairs"], 1)
        self.assertEqual(report["visibility_conflicts"]["conflict_ratio"], 1.0)
        self.assertEqual(report["visibility_conflicts"]["worst_visible_disagreement_ratio"], 0.3)
        self.assertEqual(report["visibility_conflicts"]["frame_pairs"], [["0", "1"]])

    def test_group_union_preserves_exact_pixels_source_hashes_and_proposal_provenance(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            _, parents = self.make_composite_task(task, duplicate_first=True)
            path = task / "inputs/component_mask_candidates.json"
            candidates = adapter.read_json(path)
            for record in candidates["masks"]:
                if record["component_id"] != "__object__":
                    record["quality"] = {"rejection_reasons": [], "box_iou": .75}
                    record["reused_group_gate"] = {"proposal_gate": {"included": True, "box_inside_fraction": .8}}
            adapter.write_json(path, candidates)
            original_hash = adapter.sha256(path)
            args = options(task)
            args.composite_policy = {"bed": {"pillow": "bedding"}}
            result = adapter.run(args)
            report, masks, _ = self.assert_resolved_contract(task, result)
            self.assertEqual(adapter.sha256(path), original_hash)
            group, = report["tracks"][0]["component_association"]["semantic_groups"]
            self.assertFalse(group["generated_pixels_used"])
            for frame in group["frames"]:
                expected = np.zeros((64, 64), dtype=bool)
                for member in frame["members"]:
                    source = next(record for record in candidates["masks"] if record.get("observation_id") == member["observation_id"])
                    self.assertEqual(member["source_record"], source)
                    self.assertEqual(member["membership_relation"], "bedding")
                    self.assertEqual(member["mask_sha256"], adapter.sha256(task / source["mask_path"]))
                    with Image.open(task / source["mask_path"]) as image:
                        expected |= np.asarray(image) > 0
                self.assertFalse(frame["parent_pixels_included"])
                self.assertEqual(frame["pixel_operation"], "logical_or_of_candidate_masks_only")
                with Image.open(task / frame["mask_path"]) as image:
                    np.testing.assert_array_equal(np.asarray(image) > 0, expected)
                self.assertEqual(frame["mask_sha256"], adapter.sha256(task / frame["mask_path"]))
                raw_parent = parents[frame["frame_id"]]
                with Image.open(task / raw_parent["mask_path"]) as image:
                    parent_pixels = np.asarray(image) > 0
                parent = next(record for record in masks["masks"] if record["component_id"] == "__object__" and record["frame_id"] == frame["frame_id"])
                with Image.open(task / parent["mask_path"]) as image:
                    np.testing.assert_array_equal(np.asarray(image) > 0, parent_pixels | expected)
                self.assertEqual(len(parent["composition"]["members"]), len(frame["members"]))

    def test_explicit_proposal_quality_rejection_cannot_contribute_to_group_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            _, originals = self.make_composite_task(task)
            path = task / "inputs/component_mask_candidates.json"
            candidates = adapter.read_json(path)
            rejected = next(record for record in candidates["masks"] if record.get("observation_id") == "obs_pillow_1")
            rejected["quality"] = {"rejection_reasons": ["mask_does_not_match_proposal_box"]}
            adapter.write_json(path, candidates)
            args = options(task)
            args.composite_policy = {"bed": {"pillow": "bedding"}}
            result = adapter.run(args)
            report, masks, _ = self.assert_resolved_contract(task, result)
            association = report["tracks"][0]["component_association"]
            skipped, = association["skipped_observations"]
            self.assertEqual(skipped["source_record"], rejected)
            self.assertEqual(skipped["reason"], "candidate_rejected_by_proposal_or_mask_quality")
            group, = association["semantic_groups"]
            self.assertEqual(group["member_ids"], ["obs_pillow_0"])
            self.assertFalse(group["group_geometry_validated"])
            for record in masks["masks"]:
                self.assertEqual(record["mask_path"], originals[record["frame_id"]]["mask_path"])

    def test_three_view_individual_visible_conflict_remains_rejected_when_semantic_union_passes(self):
        import numpy as np
        from PIL import Image
        from plyfile import PlyData

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            rows, _ = self.make_composite_task(task)
            candidate_path = task / "inputs/component_mask_candidates.json"
            candidates = adapter.read_json(candidate_path)
            candidates["masks"] = [item for item in candidates["masks"] if item["component_id"] == "__object__"]
            cameras = adapter.read_json(task / "inputs/cameras.json")
            depths = adapter.read_json(task / "inputs/scene_depth.json")
            hypotheses = adapter.read_json(task / "inputs/physical_instance_hypotheses.json")
            third_camera = {**cameras["frames"][0], "frame_id": "2", "world_to_camera": np.eye(4).tolist()}
            cameras["frames"].append(third_camera)
            np.save(task / "inputs/depth_2.npy", np.full((64, 64), 2., dtype=np.float32))
            np.save(task / "inputs/confidence_2.npy", np.ones((64, 64), dtype=np.float32))
            depths["frames"].append({**third_camera, "depth_path": "inputs/depth_2.npy", "confidence_path": "inputs/confidence_2.npy"})
            yy, xx = np.indices((64, 64))
            for index, center in enumerate((-.3, .3, 0.)):
                world_x, world_y = (xx - 31.5) / 40 * 2 + center, (yy - 31.5) / 40 * 2
                if index == 2:
                    parent = (world_x > -.8) & (world_x < -.25) & (abs(world_y) < .4)
                    parent_path = "inputs/left_2.png"
                    Image.fromarray(parent.astype(np.uint8) * 255).save(task / parent_path)
                    candidates["masks"].append({**candidates["masks"][0], "frame_id": "2", "mask_path": parent_path,
                                                "mask_sha256": adapter.sha256(task / parent_path)})
                    hypotheses["tracks"][0]["observations"].append({"frame_id": "2", "mask_path": parent_path})
                whole = (world_x > -.8) & (world_x < -.25) & (world_y > .35) & (world_y < .75)
                region = (world_x < -.4) if index == 0 else ((world_x > -.75) & (world_x < -.35)) if index == 1 else (world_x > -.65)
                main = whole & region
                for name, mask in (("main", main), ("remainder", whole & ~main)):
                    path = f"inputs/pillow_{name}_{index}.png"
                    Image.fromarray(mask.astype(np.uint8) * 255).save(task / path)
                    candidates["masks"].append({"object_id": "left", "frame_id": str(index),
                        "component_group_id": "pillow", "component_id": f"pillow_{name}_{index}",
                        "observation_id": f"obs_{name}_{index}", "mask_path": path, "mask_sha256": adapter.sha256(task / path),
                        "membership_relation": "bedding", "quality": {"rejection_reasons": []}, "geometry_validated": False})
            candidates["frames"] = cameras["frames"]
            for filename, document in (("cameras", cameras), ("scene_depth", depths), ("physical_instance_hypotheses", hypotheses),
                                       ("component_mask_candidates", candidates)):
                adapter.write_json(task / f"inputs/{filename}.json", document)
            args = options(task)
            args.composite_policy = {"bed": {"pillow": "bedding"}}
            result = adapter.run(args)
            report, masks, _ = self.assert_resolved_contract(task, result)
            association = report["tracks"][0]["component_association"]
            group, = association["semantic_groups"]
            self.assertTrue(group["group_geometry_validated"])
            self.assertTrue(group["parent_union_allowed"])
            self.assertFalse(group["individual_identity_confirmed"])
            self.assertEqual(group["geometry_report"]["reasons"], [])
            self.assertEqual(len(group["geometry_report"]["pairs"]), 3)
            self.assertTrue(all(pair["accepted"] for pair in group["geometry_report"]["pairs"]))
            conflicted = [member for member in association["tracks"] if "internal_visible_conflict" in member["reasons"]]
            self.assertTrue(conflicted)
            self.assertTrue(all(not member["geometry_validated"] and not member["accepted"] for member in conflicted))
            for member in conflicted:
                self.assertTrue(set(member["member_ids"]) <= set(group["unconfirmed_individual_member_ids"]))
            obj, = adapter.read_json(task / result["outputs"]["isolated_object_ply"]["path"])["objects"]
            self.assertFalse({member["track_id"] for member in conflicted} & {part["component_id"] for part in obj["parts"]})
            self.assertTrue(all(record.get("geometry_validated") is True for record in masks["masks"] if record["component_id"] != "__object__"))
            selected = PlyData.read(str(task / obj["ply_path"]))["vertex"].data
            self.assertTrue((selected["y"] > .4).any())
            np.testing.assert_array_equal(selected, rows[selected["source_index"]])
            self.assertEqual(report["tracks"][0]["gaussian_votes"]["minimum_positive_views"], 2)

    def test_rejected_parent_proposal_cannot_be_rescued_by_semantic_group(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            self.make_composite_task(task)
            path = task / "inputs/component_mask_candidates.json"
            candidates = adapter.read_json(path)
            parent = next(record for record in candidates["masks"] if record["component_id"] == "__object__")
            parent["quality"] = {"rejection_reasons": ["mask_does_not_match_proposal_box"]}
            adapter.write_json(path, candidates)
            args = options(task)
            args.composite_policy = {"bed": {"pillow": "bedding"}}
            with self.assertRaisesRegex(adapter.LiftingError, "no geometrically accepted"):
                adapter.run(args)
            report = adapter.read_json(next(task.glob("stages/object_lifting/run-*/lifting_report.json")))
            self.assertFalse(report["carve_performed"])
            self.assertIn("parent mask did not pass proposal or mask quality", report["tracks"][0]["reasons"])

    def test_same_label_different_physical_objects_cannot_carve(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            self.make_task(task, mixed=True)
            with self.assertRaisesRegex(adapter.LiftingError, "no geometrically accepted"):
                adapter.run(options(task))
            report = adapter.read_json(next(task.glob("stages/object_lifting/run-*/lifting_report.json")))
            self.assertFalse(report["carve_performed"])
            self.assertIn("disconnected_geometric_identity_observations", report["tracks"][0]["reasons"])
            self.assertFalse((task / "provider-artifacts.json").exists())
            self.assertFalse(list(task.rglob("carved_scene.ply")))

    def test_depth_in_different_scene_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            self.make_task(task, scale=2)
            with self.assertRaises(adapter.LiftingError):
                adapter.run(options(task))
            report = adapter.read_json(next(task.glob("stages/object_lifting/run-*/lifting_report.json")))
            self.assertTrue(all(not item["accepted"] for item in report["tracks"]))
            self.assertAlmostEqual(report["frame_depth_alignment"]["0"]["median_depth_over_scene_z"], 2)
            self.assertFalse(list(task.rglob("carved_scene.ply")))

    def test_occluded_and_front_of_surface_points_are_not_carved(self):
        import numpy as np

        frame = adapter.Frame("0", np.full((64, 64), 2.), np.ones((64, 64), dtype=bool), np.array([[40., 0, 31.5], [0, 40., 31.5], [0, 0, 1]]), np.eye(4), None, None, 1)
        mask = np.zeros((64, 64), dtype=bool)
        mask[30:34, 30:34] = True
        points = np.array([[0., 0., 2.], [1., 0., 2.], [0., 0., 4.], [0., 0., 1.]])
        votes = adapter.surface_votes(points, frame, mask, 0.01, 0.05)
        for name, expected in (("positive", [1, 0, 0, 0]), ("negative", [0, 1, 0, 0]), ("occluded", [0, 0, 1, 0]), ("in_front", [0, 0, 0, 1])):
            np.testing.assert_array_equal(votes[name], expected)

    def test_crop_transform_maps_source_mask_to_depth_raster(self):
        import numpy as np
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[32:34, 32:34] = 255
            Image.fromarray(mask).save(task / "mask.png")
            transform = np.array([[.5, 0, -2], [0, .5, 1], [0, 0, 1]])
            frame = adapter.Frame("0", np.ones((32, 32)), np.ones((32, 32), dtype=bool), np.eye(3), np.eye(4), None, None, 1, transform, (64, 64))
            transformed = adapter.load_mask(task, "mask.png", (32, 32), frame)
            self.assertTrue(transformed[17, 14])
            self.assertFalse(transformed[16, 16])

    def test_integer_depth_needs_scale_and_backprojection_uses_confidence(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder)
            np.save(task / "depth.npy", np.full((4, 4), 2000, dtype=np.uint16))
            confidence = np.ones((4, 4))
            confidence[0, 0] = 0
            np.save(task / "confidence.npy", confidence)
            camera = {"intrinsics": [[4., 0, 2], [0, 4., 2], [0, 0, 1]], "world_to_camera": np.eye(4).tolist(), "width": 4, "height": 4}
            record = {**camera, "depth_path": "depth.npy", "confidence_path": "confidence.npy"}
            with self.assertRaisesRegex(adapter.LiftingError, "depth_scale"):
                adapter.load_frame(task, "0", record, camera, options(task))
            frame = adapter.load_frame(task, "0", {**record, "depth_scale": .001}, camera, options(task))
            points = adapter.backproject(frame, np.ones((4, 4), dtype=bool))
            self.assertEqual(len(points), 15)
            np.testing.assert_allclose(points[:, 2], 2)


if __name__ == "__main__":
    unittest.main()
