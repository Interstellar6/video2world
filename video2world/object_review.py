"""Canonical object review receipt helpers."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS = ("front", "right", "back", "left", "top", "bottom")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def visual_gate_views_from_canonical_review(
    receipt: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Extract manifest-ready visual gate evidence from a canonical six-view receipt."""

    if receipt.get("kind") != "video2world.canonical_object_six_view_review":
        raise ValueError("canonical object review receipt has unexpected kind")
    if receipt.get("canonicalViews") != list(REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS):
        raise ValueError("canonical object review receipt does not declare the six views")
    if receipt.get("horizontalOrbitAcceptedAsSixViewEvidence") is not False:
        raise ValueError("horizontal orbit frames cannot substitute for canonical six views")
    views = receipt.get("views")
    if not isinstance(views, dict):
        raise ValueError("canonical object review receipt views must be a JSON object")

    gate_views: dict[str, dict[str, str]] = {}
    for view in REQUIRED_CANONICAL_OBJECT_REVIEW_VIEWS:
        evidence = views.get(view)
        if not isinstance(evidence, dict):
            raise ValueError(f"canonical object review receipt is missing {view!r} evidence")
        uri = evidence.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"canonical object review {view!r} evidence requires uri")
        sha256 = evidence.get("sha256")
        if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
            raise ValueError(f"canonical object review {view!r} evidence requires SHA-256")
        gate_views[view] = {"uri": uri, "sha256": sha256}
    return gate_views
