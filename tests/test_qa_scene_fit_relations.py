from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "qa_scene_fit_relations.py"
SPEC = importlib.util.spec_from_file_location("qa_scene_fit_relations", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_disjoint_full_projections_pass() -> None:
    left = np.zeros((8, 8), dtype=bool)
    right = np.zeros_like(left)
    left[1:4, 1:3] = True
    right[2:5, 5:7] = True

    result = MODULE.evaluate_disjoint(left, right)

    assert result["status"] == "passed"
    assert result["metrics"]["full_projection_overlap_pixels"] == 0


def test_depth_order_requires_front_to_be_nearer_in_every_shared_bin() -> None:
    front = np.asarray([[np.inf, 2.0], [2.2, 2.1]])
    rear = np.asarray([[np.inf, 2.5], [2.7, 2.6]])
    mask = np.ones((2, 2), dtype=bool)

    passed = MODULE.evaluate_depth_order(
        front_depth=front,
        rear_depth=rear,
        front_mask_bins=mask,
        rear_mask_bins=mask,
        minimum_shared_bins=3,
        minimum_depth_margin=0.1,
    )
    rear[1, 1] = 1.9
    failed = MODULE.evaluate_depth_order(
        front_depth=front,
        rear_depth=rear,
        front_mask_bins=mask,
        rear_mask_bins=mask,
        minimum_shared_bins=3,
        minimum_depth_margin=0.1,
    )

    assert passed["status"] == "passed"
    assert passed["metrics"]["minimum_rear_minus_front_depth_m"] == 0.5
    assert failed["status"] == "rejected"
    assert failed["gates"]["no_depth_order_inversion"] is False


def test_mask_bins_preserves_any_occupied_pixel() -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[4, 6] = True

    binned = MODULE.mask_bins(mask, 4)

    assert binned.shape == (2, 2)
    assert binned.sum() == 1
    assert binned[1, 1]


def test_missing_shared_depth_bins_fail_without_nan() -> None:
    depth = np.full((2, 2), np.inf)
    mask = np.ones((2, 2), dtype=bool)

    result = MODULE.evaluate_depth_order(
        front_depth=depth,
        rear_depth=depth,
        front_mask_bins=mask,
        rear_mask_bins=mask,
        minimum_shared_bins=1,
        minimum_depth_margin=0.1,
    )

    assert result["status"] == "rejected"
    assert result["metrics"]["minimum_rear_minus_front_depth_m"] is None
    assert result["metrics"]["rear_minus_front_depth_quantiles_m"] is None
