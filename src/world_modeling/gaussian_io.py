"""Shared reading of PGSR/3DGS scene Gaussian tables.

PGSR leaves a minority of splats non-finite when a few view-inconsistent points
never converge. A NaN position cannot be rasterised, so those rows carry no
geometry and must not fail an entire stage -- but they must not disappear
silently either, because how many there are is real evidence about the
reconstruction. Every reader drops them through this one helper and records the
count, so the degeneracy stays visible in QA instead of hiding in a log line.
"""

from __future__ import annotations

from collections import namedtuple


class GaussianError(RuntimeError):
    """Raised when a Gaussian PLY cannot yield usable geometry."""


GaussianCloud = namedtuple("GaussianCloud", "rows points dropped payload")


def read_gaussian_rows(path, *, required=("x", "y", "z"), error=GaussianError):
    """Read a single-table Gaussian PLY and drop every non-finite vertex row.

    Returns a :class:`GaussianCloud` whose ``points`` is an ``(n, 3)`` float array
    of positions and whose ``dropped`` is how many rows were removed; ``payload``
    is the decoded PLY so callers can copy the original header when rewriting.
    Raises ``error`` when the table is malformed or nothing finite remains -- an
    empty result is a genuine failure, not a tolerable defect.
    """
    import numpy as np
    from plyfile import PlyData

    payload = PlyData.read(str(path))
    if len(payload.elements) != 1 or payload.elements[0].name != "vertex":
        raise error(f"expected one PLY vertex table: {path}")
    rows = payload["vertex"].data
    names = list(rows.dtype.names or ())
    missing = [name for name in required if name not in names]
    if missing:
        raise error(f"{path}: missing PLY vertex attributes {missing}")
    total = len(rows)
    finite = np.ones(total, dtype=bool)
    for name in names:
        if rows[name].dtype.kind == "f":
            finite &= np.isfinite(rows[name])
    dropped = int((~finite).sum())
    if dropped:
        if dropped == total:
            raise error(f"{path}: every one of the {total} Gaussian vertices is non-finite")
        rows = rows[finite]
    points = np.column_stack([rows[name] for name in ("x", "y", "z")]).astype(np.float64)
    if not len(points):
        raise error(f"{path}: no Gaussian vertices remain")
    return GaussianCloud(rows=rows, points=points, dropped=dropped, payload=payload)
