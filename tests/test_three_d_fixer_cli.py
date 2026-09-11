from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import three_d_fixer_cli as bootstrap
from three_d_fixer_module import SPEC
import world_modeling.engine as engine
import world_modeling.services as services
from world_modeling.registry import ModuleRegistry
from world_modeling.modules.geometry_completion import SPEC as STREAM_SPEC


class OptInCLITests(unittest.TestCase):
    def cli(self, *argv):
        process = subprocess.run([sys.executable, str(ROOT / "scripts/three_d_fixer_cli.py"), *argv],
                                 cwd=ROOT, text=True, capture_output=True, timeout=30,
                                 env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(process.returncode, 0, process.stderr)
        return json.loads(process.stdout)

    def test_explicit_activation_keeps_existing_mapping_references_live(self):
        registry = ModuleRegistry([STREAM_SPEC])
        previous_mapping = registry.specs
        self.assertIs(bootstrap.activate(registry), registry)
        self.assertEqual(previous_mapping[SPEC.id], SPEC)
        self.assertNotIn(STREAM_SPEC.id, previous_mapping)
        bootstrap.activate(registry)
        self.assertEqual(len(registry.modules), 1)

    def test_opt_in_cli_discovery_and_serve_parser_accept_new_contract(self):
        result = self.cli("discover")
        modules = {item["id"]: item for item in result["modules"]}
        self.assertEqual(modules[SPEC.id]["inputs"], list(SPEC.inputs))
        self.assertNotIn(STREAM_SPEC.id, modules)
        help_result = subprocess.run([sys.executable, str(ROOT / "scripts/three_d_fixer_cli.py"), "serve", "--help"],
                                     cwd=ROOT, text=True, capture_output=True, timeout=30,
                                     env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(help_result.returncode, 0)
        self.assertIn(SPEC.id, help_result.stdout)

    def test_default_cli_still_discovers_original_twelve_modules(self):
        process = subprocess.run([sys.executable, "-m", "world_modeling.cli", "discover"], cwd=ROOT,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                                 text=True, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        names = {item["id"] for item in json.loads(process.stdout)["modules"]}
        self.assertIn(STREAM_SPEC.id, names)
        self.assertNotIn(SPEC.id, names)

    def test_standalone_profile_parses_only_after_explicit_registration(self):
        registry = ModuleRegistry([STREAM_SPEC])
        with patch.object(engine, "SPECS", registry.specs):
            with self.assertRaisesRegex(engine.PipelineError, "unknown modules"):
                engine.load_profile(ROOT / "profiles/three-d-fixer.example.json")
            bootstrap.activate(registry)
            profile = engine.load_profile(ROOT / "profiles/three-d-fixer.example.json")
            self.assertEqual(profile["pipeline"], [SPEC.id])
            self.assertNotIn("--prepare-only", profile["providers"][SPEC.id]["command"])
            registry.ordered(profile["pipeline"], initial_roles=SPEC.inputs)

    def test_live_http_discovery_cli_registration_and_service_profile_resolution(self):
        provider = {"kind": "command", "command": [sys.executable, str(ROOT / "scripts/three_d_fixer.py"),
                    "--task-dir", "{task_dir}", "--inputs", "{inputs_json}", "--outputs", "{outputs_json}"]}
        service = services.ModuleService(ROOT, SPEC, provider)
        server = services.make_server(service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder)
                registry_path = directory / "services.json"
                endpoint = f"http://127.0.0.1:{server.server_port}"
                registered = self.cli("register", "--registry", str(registry_path), "--endpoint", endpoint)
                self.assertEqual(registered["registered"], [SPEC.id])
                discovered = self.cli("discover", "--registry", str(registry_path))
                self.assertEqual(discovered[SPEC.id]["modules"][0]["inputs"], list(SPEC.inputs))
                profile_path = directory / "service-profile.json"
                profile_path.write_text(json.dumps({"schema_version": "1.0", "registry": "services.json",
                    "providers": {SPEC.id: {"kind": "service"}}}))
                registry = ModuleRegistry([SPEC])
                with patch.object(engine, "SPECS", registry.specs), patch.object(services, "REGISTRY", registry):
                    profile = engine.load_profile(profile_path)
                self.assertEqual(profile["providers"][SPEC.id]["provider_revision"], service.revision)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            service.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
