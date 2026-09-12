from __future__ import annotations

import json
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "profiles"

# The reference delivery trained PGSR to iteration 30000 (with a 7000 fallback),
# which is what the scene Gaussian density target of ~870k points comes from.
REFERENCE_PGSR_ITERATIONS = 30000


def load(name: str) -> dict:
    return json.loads((PROFILES / name).read_text())


def command_of(profile: dict, module: str) -> list:
    return list(profile["providers"][module]["command"])


def option(command: list, flag: str):
    if flag not in command:
        return None
    return command[command.index(flag) + 1]


class ShippedProfileTests(unittest.TestCase):
    def test_the_deployed_profiles_keep_the_reference_scene_recipe(self):
        for name in ("seetacloud.json", "three-d-fixer.skipbg.json", "embodiedgen.skipbg.json"):
            with self.subTest(profile=name):
                profile = load(name)
                command = command_of(profile, "scene_reconstruction")
                self.assertEqual(option(command, "--pgsr-iterations"), str(REFERENCE_PGSR_ITERATIONS))
                self.assertEqual(option(command, "--max-frames"), "0")

    def test_the_object_asset_profiles_use_embodiedgen_and_a_tolerant_inventory(self):
        for name in ("three-d-fixer.skipbg.json", "embodiedgen.skipbg.json"):
            with self.subTest(profile=name):
                profile = load(name)
                self.assertIn("observed_context_completion", profile["providers"])
                self.assertNotIn("geometry_completion", profile["providers"])
                self.assertEqual(option(command_of(profile, "scene_understanding"),
                                         "--component-inventory-policy"), "tolerant")
                self.assertIsNotNone(option(command_of(profile, "scene_understanding"), "--object-hints"))

    def test_the_embodiedgen_profile_binds_the_asset_provider(self):
        profile = load("embodiedgen.skipbg.json")
        command = command_of(profile, "mesh_postprocess")
        self.assertIn("embodiedgen_assets.py", " ".join(command))
        self.assertIn("--embodiedgen-root", command)
        self.assertEqual(sorted(profile["providers"]), sorted(set(profile["pipeline"])))

    def test_a_profile_declares_every_provider_it_binds(self):
        for path in sorted(PROFILES.glob("*.json")):
            with self.subTest(profile=path.name):
                profile = json.loads(path.read_text())
                if "pipeline" not in profile or "providers" not in profile:
                    continue
                self.assertEqual(sorted(profile["providers"]), sorted(set(profile["pipeline"])),
                                 f"{path.name}: providers and pipeline disagree")


if __name__ == "__main__":
    unittest.main()
