"""Read-only source identities for isolated command providers, without importing them."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable


class SourceRevisionError(ValueError):
    pass


_DERIVED_FIELDS = {"source_revision", "provider_revision"}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}


def _binding(provider: dict) -> dict:
    if not isinstance(provider, dict) or provider.get("kind") != "command":
        raise SourceRevisionError("source revisions require a command provider")
    return json.loads(json.dumps({key: value for key, value in provider.items() if key not in _DERIVED_FIELDS}))


def _path(value: str, base: Path, context: dict | None) -> Path:
    try:
        rendered = value.format(**context) if context is not None else value
    except (KeyError, ValueError, IndexError) as error:
        raise SourceRevisionError(f"cannot resolve provider source path: {value}") from error
    if "{" in rendered or "}" in rendered:
        raise SourceRevisionError(f"provider source path needs an execution context: {value}")
    path = Path(rendered)
    return (path if path.is_absolute() else base / path).absolute()


def _entrypoint(command: list[str], cwd: Path, context: dict | None, explicit: list[str]) -> tuple[Path, str]:
    index = 0
    search_path = os.environ.get("PATH", "")
    if Path(command[0]).name == "env":
        index = 1
        while index < len(command):
            value = command[index]
            if re.match(r"[A-Za-z_][A-Za-z_0-9]*=", value):
                if value.startswith("PATH="):
                    search_path = value.partition("=")[2]
                index += 1
            elif value in {"-i", "--ignore-environment"}:
                index += 1
            elif value in {"-u", "--unset"}:
                index += 2
            elif value == "--":
                index += 1
                break
            elif value.startswith("-"):
                raise SourceRevisionError("unsupported env wrapper; use an explicit argv entrypoint")
            else:
                break
    if index >= len(command):
        raise SourceRevisionError("command provider has no executable entrypoint")
    executable = command[index]
    if "/" not in executable and "{" not in executable:
        executable = shutil.which(executable, path=search_path) or executable
    binary = _path(executable, cwd, context)
    name = binary.name
    if re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", name):
        index += 1
        while index < len(command):
            option = command[index]
            if option in {"-c", "-m", "-"}:
                break
            if option in {"-W", "-X"}:
                index += 2
            elif option == "--":
                index += 1
                break
            elif option.startswith("-"):
                index += 1
            else:
                break
        if index < len(command) and command[index] not in {"-c", "-m", "-"}:
            return _path(command[index], cwd, context), "python"
        if not explicit:
            raise SourceRevisionError("dynamic Python entrypoint requires nonempty revision_files")
        return binary, "dynamic"
    if name in _SHELLS:
        if not explicit:
            raise SourceRevisionError("shell entrypoint requires nonempty revision_files")
        return binary, "dynamic"
    return binary, "python" if binary.suffix == ".py" else "executable"


def _module_files(name: str, roots: list[Path]) -> list[Path]:
    parts = name.split(".")
    if not all(part.isidentifier() for part in parts):
        return []
    for root in roots:
        candidate = root.joinpath(*parts)
        package = candidate / "__init__.py"
        module = candidate.with_suffix(".py")
        target = package if package.is_file() else module
        if not target.is_file():
            continue
        parents = [root.joinpath(*parts[:index], "__init__.py") for index in range(1, len(parts))]
        return [path for path in parents if path.is_file()] + [target]
    return []


def _literal_path(node: ast.AST, source: Path) -> Path | None:
    if isinstance(node, ast.Name) and node.id == "__file__":
        return source
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = _literal_path(node.left, source), _literal_path(node.right, source)
        return left / right if left is not None and right is not None else None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        parent = _literal_path(node.value, source)
        return parent.parent if parent is not None else None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        parent = _literal_path(node.value.value, source)
        if parent is not None and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            try:
                return parent.parents[node.slice.value]
            except IndexError:
                return None
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1:
            return _literal_path(node.args[0], source)
        if isinstance(node.func, ast.Attribute):
            parent = _literal_path(node.func.value, source)
            if parent is None:
                return None
            if node.func.attr in {"resolve", "absolute"} and not node.args:
                return parent.absolute()
            if node.func.attr in {"with_name", "with_suffix"} and len(node.args) == 1:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    try:
                        return getattr(parent, node.func.attr)(value.value)
                    except ValueError:
                        return None
    return None


def _dependencies(source: Path, contents: bytes, roots: list[Path]) -> set[Path]:
    try:
        tree = ast.parse(contents, filename=str(source))
    except (SyntaxError, ValueError) as error:
        raise SourceRevisionError(f"cannot parse provider source {source}: {error}") from error
    dependencies: set[Path] = set()
    package = []
    parent = source.parent
    while (parent / "__init__.py").is_file():
        package.insert(0, parent.name)
        parent = parent.parent
    search_roots = list(dict.fromkeys([*roots, parent]))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1] if node.level <= len(package) else []
                base += node.module.split(".") if node.module else []
                module = ".".join(base)
            else:
                module = node.module or ""
            if module:
                names = [module] + [f"{module}.{item.name}" for item in node.names if item.name != "*"]
        elif isinstance(node, ast.Call):
            function = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if function in {"import_module", "__import__"} and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    names = [value.value]
            # Existing adapters load sibling helpers through importlib or subprocess.
            literal = _literal_path(node, source)
            if literal is not None and literal.is_absolute() and literal.suffix == ".py" and literal != source:
                dependencies.add(literal)
        for name in names:
            dependencies.update(_module_files(name, search_roots))
    return dependencies


def source_snapshot(root: Path, provider: dict, *, cwd: Path | None = None, context: dict | None = None,
                    extra_files: Iterable[Path] = ()) -> dict:
    binding = _binding(provider)
    command = binding.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command) or not command[0]:
        raise SourceRevisionError("command provider requires a nonempty string argv list")
    explicit = binding.get("revision_files", [])
    if not isinstance(explicit, list) or not all(isinstance(item, str) and item for item in explicit):
        raise SourceRevisionError("provider revision_files must be a string path array")
    root = Path(root).resolve()
    cwd = Path(cwd).resolve() if cwd is not None else root
    entry, kind = _entrypoint(command, cwd, context, explicit)
    core = Path(__file__).resolve().parent
    roots = list(dict.fromkeys([entry.parent, cwd, root / "src", root, core.parent]))
    pending = {entry, core / "provider_io.py", Path(__file__).resolve()}
    pending.update(_path(item, root, context) for item in explicit)
    pending.update(Path(item).absolute() for item in extra_files)
    files: dict[str, str] = {}
    while pending:
        path = sorted(pending)[0]
        pending.remove(path)
        logical = str(path.absolute())
        if logical in files:
            continue
        try:
            if not path.is_file():
                raise SourceRevisionError(f"provider revision file is missing: {path}")
            digest = hashlib.sha256()
            blocks = []
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    if path.suffix == ".py":
                        blocks.append(block)
        except OSError as error:
            raise SourceRevisionError(f"cannot read provider revision file {path}: {error}") from error
        files[logical] = digest.hexdigest()
        if path.suffix == ".py":
            pending.update(_dependencies(path, b"".join(blocks), roots))
    return {"schema_version": "1.0", "entrypoint": {"path": str(entry), "kind": kind}, "files": dict(sorted(files.items()))}


def resolve_command_provider(root: Path, provider: dict, *, cwd: Path | None = None, context: dict | None = None,
                             extra_files: Iterable[Path] = ()) -> dict:
    binding = _binding(provider)
    snapshot = source_snapshot(root, binding, cwd=cwd, context=context, extra_files=extra_files)
    payload = {"provider": binding, "source_revision": snapshot}
    revision = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**binding, "source_revision": snapshot, "provider_revision": revision}


def check_command_revision(root: Path, provider: dict, *, cwd: Path | None = None, context: dict | None = None) -> None:
    if not isinstance(provider.get("source_revision"), dict) or not isinstance(provider.get("provider_revision"), str):
        raise SourceRevisionError("command provider has no resolved source revision")
    try:
        current = resolve_command_provider(root, provider, cwd=cwd, context=context)
    except SourceRevisionError as error:
        raise SourceRevisionError(f"provider source revision changed: {error}") from error
    if current["provider_revision"] != provider["provider_revision"] or current["source_revision"] != provider["source_revision"]:
        raise SourceRevisionError("provider source revision changed during execution")
