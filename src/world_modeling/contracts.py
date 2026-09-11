from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Evidence = Literal["observed", "generated", "derived", "contract_only"]


@dataclass(frozen=True)
class ArtifactRole:
    name: str
    evidence: Evidence
    media_hint: str
    collision_eligible: bool = False


@dataclass(frozen=True)
class ModuleSpec:
    id: str
    title: str
    inputs: tuple[str, ...]
    outputs: tuple[ArtifactRole, ...]
    provider_family: str
    notes: str

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.outputs)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "inputs": list(self.inputs),
            "outputs": [asdict(role) for role in self.outputs],
            "provider_family": self.provider_family,
            "notes": self.notes,
        }


def role(
    name: str,
    evidence: Evidence,
    media_hint: str,
    *,
    collision_eligible: bool = False,
) -> ArtifactRole:
    return ArtifactRole(name, evidence, media_hint, collision_eligible)


def task_relative(task_dir: Path, candidate: Path) -> str:
    return str(candidate.resolve().relative_to(task_dir.resolve()))

