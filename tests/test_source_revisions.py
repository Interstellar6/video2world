from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from world_modeling.source_revision import (
    SourceRevisionError,
    check_command_revision,
    resolve_command_provider,
    source_snapshot,
)


class SourceRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.entry = self.write("adapter.py", "VALUE = 1\n")
        self.provider = {
            "kind": "command",
            "command": [sys.executable, str(self.entry), "{outputs_json}"],
        }

    def write(self, relative, contents):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def resolve(self, provider=None, **kwargs):
        return resolve_command_provider(self.root, provider or self.provider, **kwargs)

    def test_snapshot_records_absolute_paths_and_content_hashes(self):
        snapshot = source_snapshot(self.root, self.provider)
        self.assertEqual(snapshot["schema_version"], "1.0")
        self.assertEqual(snapshot["files"][str(self.entry)], hashlib.sha256(self.entry.read_bytes()).hexdigest())
        self.assertTrue(all(Path(path).is_absolute() for path in snapshot["files"]))

    def test_resolution_is_deterministic_and_does_not_mutate_configuration(self):
        original = copy.deepcopy(self.provider)
        first = self.resolve()
        second = self.resolve()
        self.assertEqual(self.provider, original)
        self.assertEqual(first, second)
        self.assertRegex(first["provider_revision"], r"^[0-9a-f]{64}$")
        self.assertIn(str(self.entry), first["source_revision"]["files"])
        first["command"].append("--changed-in-resolved-copy")
        self.assertEqual(self.provider, original)

    def test_unchanged_source_passes_revision_check(self):
        check_command_revision(self.root, self.resolve())

    def test_empty_argument_is_valid_and_part_of_provider_revision(self):
        provider = {
            "kind": "command",
            "command": [sys.executable, str(self.entry), "--clean-frame-indices", ""],
        }
        empty = self.resolve(provider)
        populated = self.resolve({**provider, "command": provider["command"][:-1] + ["0,60"]})
        self.assertEqual(empty["command"][-1], "")
        self.assertEqual(empty["source_revision"], populated["source_revision"])
        self.assertNotEqual(empty["provider_revision"], populated["provider_revision"])
        check_command_revision(self.root, empty)

    def test_in_place_entry_edit_changes_revision_and_rejects_old_snapshot(self):
        before = self.resolve()
        self.entry.write_text("VALUE = 2\n", encoding="utf-8")
        after = self.resolve()
        self.assertNotEqual(before["provider_revision"], after["provider_revision"])
        with self.assertRaises(SourceRevisionError):
            check_command_revision(self.root, before)
        check_command_revision(self.root, after)

    def test_deleted_entry_fails_closed(self):
        before = self.resolve()
        self.entry.unlink()
        with self.assertRaises(SourceRevisionError):
            check_command_revision(self.root, before)
        with self.assertRaises(SourceRevisionError):
            self.resolve()

    def test_local_import_dependency_is_followed_transitively(self):
        self.entry.write_text("import helper\n", encoding="utf-8")
        helper = self.write("helper.py", "from leaf import VALUE\n")
        leaf = self.write("leaf.py", "VALUE = 1\n")
        before = self.resolve()
        self.assertTrue({str(self.entry), str(helper), str(leaf)} <= set(before["source_revision"]["files"]))
        leaf.write_text("VALUE = 2\n", encoding="utf-8")
        self.assertNotEqual(before["provider_revision"], self.resolve()["provider_revision"])
        with self.assertRaises(SourceRevisionError):
            check_command_revision(self.root, before)

    def test_deleted_import_dependency_rejects_old_snapshot(self):
        self.entry.write_text("import helper\n", encoding="utf-8")
        helper = self.write("helper.py", "VALUE = 1\n")
        before = self.resolve()
        helper.unlink()
        with self.assertRaises(SourceRevisionError):
            check_command_revision(self.root, before)

    def test_unrelated_script_changes_do_not_change_revision(self):
        unrelated = self.write("scripts/unrelated.py", "VALUE = 1\n")
        before = self.resolve()
        self.assertNotIn(str(unrelated), before["source_revision"]["files"])
        unrelated.write_text("VALUE = 2\n", encoding="utf-8")
        self.assertEqual(before["provider_revision"], self.resolve()["provider_revision"])
        check_command_revision(self.root, before)

    def test_relative_imports_include_package_initializers_and_parent_module(self):
        self.entry.write_text("from package.nested import worker\n", encoding="utf-8")
        package_init = self.write("package/__init__.py", "NAME = 'fixture'\n")
        nested_init = self.write("package/nested/__init__.py", "\n")
        worker = self.write("package/nested/worker.py", "from ..settings import VALUE\n")
        settings = self.write("package/settings.py", "VALUE = 1\n")
        before = self.resolve()
        expected = {str(path) for path in (package_init, nested_init, worker, settings)}
        self.assertTrue(expected <= set(before["source_revision"]["files"]))
        package_init.write_text("NAME = 'changed'\n", encoding="utf-8")
        self.assertNotEqual(before["provider_revision"], self.resolve()["provider_revision"])

    def test_from_package_import_submodule_is_included(self):
        self.entry.write_text("from package import helper\n", encoding="utf-8")
        self.write("package/__init__.py", "\n")
        helper = self.write("package/helper.py", "VALUE = 1\n")
        snapshot = source_snapshot(self.root, self.provider)
        self.assertIn(str(helper), snapshot["files"])

    def test_entry_parent_is_searched_for_sibling_imports(self):
        entry = self.write("scripts/adapter.py", "import sibling\n")
        sibling = self.write("scripts/sibling.py", "VALUE = 1\n")
        provider = {"kind": "command", "command": [sys.executable, str(entry)]}
        snapshot = source_snapshot(self.root, provider)
        self.assertIn(str(sibling), snapshot["files"])

    def test_literal_with_name_helpers_are_followed_without_importing_them(self):
        entry = self.write(
            "scripts/adapter.py",
            "import importlib.util\nfrom pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location('fixture_helper', Path(__file__).with_name('helper.py'))\n"
            "module = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n",
        )
        helper = self.write("scripts/helper.py", "from pathlib import Path\nCHILD = Path(__file__).with_name('leaf.py')\n")
        leaf = self.write("scripts/leaf.py", "raise RuntimeError('leaf must not execute')\n")
        orbit = self.write("scripts/orbit.py", "VALUE = 1\n")
        provider = {"kind": "command", "command": [sys.executable, str(entry)]}
        before = self.resolve(provider)
        self.assertTrue({str(helper), str(leaf)} <= set(before["source_revision"]["files"]))
        self.assertNotIn(str(orbit), before["source_revision"]["files"])
        orbit.write_text("VALUE = 2\n", encoding="utf-8")
        self.assertEqual(before["provider_revision"], self.resolve(provider)["provider_revision"])
        leaf.write_text("raise RuntimeError('edited leaf must not execute')\n", encoding="utf-8")
        self.assertNotEqual(before["provider_revision"], self.resolve(provider)["provider_revision"])
        with self.assertRaises(SourceRevisionError):
            check_command_revision(self.root, before)

    def test_with_name_subprocess_helper_is_a_source_dependency(self):
        self.entry.write_text(
            "from pathlib import Path\n"
            "def build_command(python):\n"
            "    return [python, str(Path(__file__).with_name('child.py'))]\n",
            encoding="utf-8",
        )
        child = self.write("child.py", "VALUE = 1\n")
        snapshot = source_snapshot(self.root, self.provider)
        self.assertIn(str(child), snapshot["files"])

    def test_provider_io_abi_is_included_without_hashing_all_core_modules(self):
        import world_modeling.provider_io as provider_io

        path = Path(provider_io.__file__).resolve()
        snapshot = source_snapshot(self.root, self.provider)
        self.assertIn(str(path), snapshot["files"])
        self.assertNotIn(str(path.with_name("services.py")), snapshot["files"])

    def test_explicit_non_python_dependency_changes_revision(self):
        dependency = self.write("config/provider.json", '{"threshold": 1}\n')
        provider = {**self.provider, "revision_files": ["config/provider.json"]}
        before = self.resolve(provider)
        self.assertIn(str(dependency), before["source_revision"]["files"])
        dependency.write_text('{"threshold": 2}\n', encoding="utf-8")
        self.assertNotEqual(before["provider_revision"], self.resolve(provider)["provider_revision"])
        with self.assertRaises(SourceRevisionError):
            check_command_revision(self.root, before)

    def test_explicit_external_dependency_is_hashed(self):
        with tempfile.TemporaryDirectory() as external:
            dependency = Path(external).resolve() / "adapter.conf"
            dependency.write_text("mode=first\n", encoding="utf-8")
            provider = {**self.provider, "revision_files": [str(dependency)]}
            before = self.resolve(provider)
            self.assertIn(str(dependency), before["source_revision"]["files"])
            dependency.write_text("mode=second\n", encoding="utf-8")
            self.assertNotEqual(before["provider_revision"], self.resolve(provider)["provider_revision"])

    def test_explicit_binary_dependency_is_hashed_in_bounded_chunks(self):
        dependency = self.root / "weights.bin"
        contents = b"\x00\xffsmall binary fixture for streaming checks\n" * 3
        dependency.write_bytes(contents)
        original_open = Path.open
        reads = []

        class BoundedReader:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                return self.stream.__exit__(*args)

            def read(self, size=-1):
                if not isinstance(size, int) or not 0 < size <= 1024 * 1024:
                    raise AssertionError(f"binary source must use bounded reads, received {size}")
                reads.append(size)
                # Short reads exercise the loop without creating a large fixture.
                return self.stream.read(min(size, 11))

        def bounded_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return BoundedReader(stream) if path == dependency else stream

        with patch.object(Path, "open", new=bounded_open):
            snapshot = source_snapshot(self.root, {**self.provider, "revision_files": [str(dependency)]})
        self.assertGreater(len(reads), 1)
        self.assertEqual(snapshot["files"][str(dependency)], hashlib.sha256(contents).hexdigest())

    def test_missing_or_malformed_explicit_dependencies_fail_closed(self):
        for explicit in (["missing.py"], "adapter.py", [123]):
            with self.subTest(explicit=explicit), self.assertRaises(SourceRevisionError):
                self.resolve({**self.provider, "revision_files": explicit})

    def test_dynamic_entries_require_explicit_dependency_declarations(self):
        commands = (
            [sys.executable, "-c", "print('not executed')"],
            [sys.executable, "-m", "package.worker"],
            ["/bin/sh", "-c", "python adapter.py"],
        )
        for command in commands:
            with self.subTest(command=command), self.assertRaises(SourceRevisionError):
                self.resolve({"kind": "command", "command": command})
            with self.subTest(command=command, empty=True), self.assertRaises(SourceRevisionError):
                self.resolve({"kind": "command", "command": command, "revision_files": []})

    def test_declared_dynamic_entry_can_be_resolved_without_execution(self):
        for command in ([sys.executable, "-m", "adapter"], ["/bin/sh", "-c", "exit 99"]):
            with self.subTest(command=command):
                provider = {"kind": "command", "command": command, "revision_files": [str(self.entry)]}
                resolved = self.resolve(provider)
                self.assertIn(str(self.entry), resolved["source_revision"]["files"])
                check_command_revision(self.root, resolved)

    def test_env_wrapped_python_with_flags_is_a_static_entry(self):
        provider = {
            "kind": "command",
            "command": ["/usr/bin/env", "FIXTURE_MODE=testing", sys.executable, "-u", str(self.entry)],
        }
        resolved = self.resolve(provider)
        self.assertIn(str(self.entry), resolved["source_revision"]["files"])
        check_command_revision(self.root, resolved)

    def test_relative_entry_is_resolved_against_execution_cwd(self):
        entry = self.write("scripts/worker.py", "VALUE = 1\n")
        provider = {"kind": "command", "command": [sys.executable, "worker.py"]}
        resolved = self.resolve(provider, cwd=entry.parent)
        self.assertIn(str(entry), resolved["source_revision"]["files"])
        check_command_revision(self.root, resolved, cwd=entry.parent)

    def test_source_validation_does_not_import_or_execute_modules(self):
        marker = self.root / "MUST_NOT_EXIST"
        self.entry.write_text(
            "from pathlib import Path\nimport exploding_helper\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            "raise RuntimeError('adapter must not run during revision validation')\n",
            encoding="utf-8",
        )
        helper = self.write("exploding_helper.py", "raise RuntimeError('helper must not be imported')\n")
        resolved = self.resolve()
        self.assertIn(str(helper), resolved["source_revision"]["files"])
        check_command_revision(self.root, resolved)
        self.assertFalse(marker.exists())
        self.assertFalse(any(self.root.rglob("__pycache__")))

    def test_python_data_argument_is_not_treated_as_a_source_entry(self):
        data = self.write("input_data.py", "arbitrary input bytes, not valid Python\n")
        provider = {"kind": "command", "command": [sys.executable, str(self.entry), "--input", str(data)]}
        before = self.resolve(provider)
        self.assertNotIn(str(data), before["source_revision"]["files"])
        data.write_text("updated data\n", encoding="utf-8")
        self.assertEqual(before["provider_revision"], self.resolve(provider)["provider_revision"])


if __name__ == "__main__":
    unittest.main()
