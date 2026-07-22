#!/usr/bin/env python3
"""Build one provenance-preserving synthetic texture atlas per structural plane.

Measured texels from ``complete_planar_background.py`` remain byte-exact. Wall
and ceiling holes use a robust plane-UV illumination field fitted away from the
object-shaped removal boundary; floor holes use a directional nearest-strip
extension chosen from measured gradient anisotropy. Every target view is
rendered from the same completed atlases, so synthetic pixels are cross-view
consistent by construction and remain explicitly labeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import complete_planar_background as geometry
import numpy as np
from PIL import Image, ImageDraw


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbor_average(image: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    height, width = image.shape[:2]
    color_sum = np.zeros_like(image, dtype=np.float64)
    count = np.zeros((height, width), dtype=np.float64)
    source_valid = np.ones((height, width), dtype=bool) if valid is None else valid
    color_sum[1:] += image[:-1] * source_valid[:-1, :, None]
    count[1:] += source_valid[:-1]
    color_sum[:-1] += image[1:] * source_valid[1:, :, None]
    count[:-1] += source_valid[1:]
    color_sum[:, 1:] += image[:, :-1] * source_valid[:, :-1, None]
    count[:, 1:] += source_valid[:, :-1]
    color_sum[:, :-1] += image[:, 1:] * source_valid[:, 1:, None]
    count[:, :-1] += source_valid[:, 1:]
    average = color_sum / np.maximum(count[..., None], 1.0)
    return average, count


def wavefront_initialize(color: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if not np.any(observed):
        raise ValueError("texture atlas has no measured texels")
    result = color.astype(np.float64, copy=True)
    known = observed.copy()
    fallback = np.median(result[observed], axis=0)
    while not np.all(known):
        average, count = neighbor_average(result, known)
        update = ~known & (count > 0)
        if not np.any(update):
            result[~known] = fallback
            break
        result[update] = average[update]
        known[update] = True
    return result


def harmonic_extend(
    color: np.ndarray,
    observed: np.ndarray,
    *,
    iterations: int,
    relaxation: float,
    propagation_support: np.ndarray | None = None,
) -> np.ndarray:
    support = observed if propagation_support is None else propagation_support
    if not np.all(support <= observed):
        raise ValueError("harmonic propagation support must be a subset of measured texels")
    result = wavefront_initialize(color, support)
    free = ~support
    for _ in range(iterations):
        average, _ = neighbor_average(result)
        result[free] = (
            result[free] * (1.0 - relaxation) + average[free] * relaxation
        )
    result[observed] = color[observed]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def robust_low_frequency_support(
    color: np.ndarray,
    observed: np.ndarray,
    *,
    quantization: int,
    color_radius: float,
    minimum_fraction: float,
    luminance_quantile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    measured = color[observed]
    if not len(measured):
        raise ValueError("wall atlas has no measured texels")
    quantized = measured // quantization
    bins, inverse, counts = np.unique(
        quantized,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    dominant_index = int(np.argmax(counts))
    dominant = np.median(measured[inverse == dominant_index], axis=0)
    initial_distance = np.linalg.norm(measured.astype(np.float64) - dominant[None, :], axis=1)
    initial_inlier = measured[initial_distance <= color_radius]
    luminance = np.einsum(
        "ni,i->n",
        initial_inlier,
        np.asarray([0.2126, 0.7152, 0.0722]),
        optimize=True,
    )
    bright = initial_inlier[luminance >= np.quantile(luminance, luminance_quantile)]
    representative = np.median(bright, axis=0)
    distance = np.linalg.norm(
        color.astype(np.float64) - representative[None, None, :],
        axis=2,
    )
    support = observed & (distance <= color_radius)
    measured_fraction = float(support.sum() / max(int(observed.sum()), 1))
    fallback = measured_fraction < minimum_fraction
    if fallback:
        support = observed.copy()
    return support, {
        "dominant_quantized_bin": bins[dominant_index].tolist(),
        "dominant_rgb_median": dominant.tolist(),
        "bright_wall_rgb_median": representative.tolist(),
        "luminance_quantile": luminance_quantile,
        "color_radius": color_radius,
        "support_texels": int(support.sum()),
        "support_fraction_of_measured": float(
            support.sum() / max(int(observed.sum()), 1)
        ),
        "fallback_to_all_measured": fallback,
    }


def screened_harmonic_extend(
    color: np.ndarray,
    observed: np.ndarray,
    prior: np.ndarray,
    *,
    iterations: int,
    relaxation: float,
    screen_weight: float,
) -> np.ndarray:
    result = wavefront_initialize(color, observed)
    synthetic = ~observed
    prior_float = prior.astype(np.float64)
    for _ in range(iterations):
        average, _ = neighbor_average(result)
        target = average * (1.0 - screen_weight) + prior_float * screen_weight
        result[synthetic] = (
            result[synthetic] * (1.0 - relaxation) + target[synthetic] * relaxation
        )
        result[observed] = color[observed]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def inset_support_from_synthetic_boundary(
    support: np.ndarray,
    observed: np.ndarray,
    inset: int,
    *,
    minimum_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if inset < 0:
        raise ValueError("wall support boundary inset must be non-negative")
    safe_observed = observed.copy()
    for _ in range(inset):
        safe_observed = ~geometry.binary_dilate(~safe_observed, 1)
    inset_support = support & safe_observed
    minimum_count = max(12, int(np.ceil(support.sum() * minimum_fraction)))
    fallback = int(inset_support.sum()) < minimum_count
    selected = support if fallback else inset_support
    return selected, {
        "boundary_inset_texels": inset,
        "support_before_inset": int(support.sum()),
        "support_after_inset": int(inset_support.sum()),
        "minimum_required_after_inset": minimum_count,
        "fallback_to_non_inset_support": fallback,
    }


def polynomial_features(
    row: np.ndarray,
    column: np.ndarray,
    *,
    height: int,
    width: int,
    degree: int,
) -> tuple[np.ndarray, list[str]]:
    if degree < 1 or degree > 4:
        raise ValueError("illumination polynomial degree must be between 1 and 4")
    u = (column.astype(np.float64) + 0.5) / max(width, 1) * 2.0 - 1.0
    v = (row.astype(np.float64) + 0.5) / max(height, 1) * 2.0 - 1.0
    values: list[np.ndarray] = []
    names: list[str] = []
    for total_degree in range(degree + 1):
        for u_degree in range(total_degree, -1, -1):
            v_degree = total_degree - u_degree
            values.append((u**u_degree) * (v**v_degree))
            names.append(f"u^{u_degree}*v^{v_degree}")
    return np.column_stack(values), names


def robust_plane_illumination_field(
    color: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    *,
    degree: int,
    iterations: int,
    huber_delta: float,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not np.all(support <= observed):
        raise ValueError("illumination support must be a subset of measured texels")
    row, column = np.nonzero(support)
    if len(row) < 12:
        raise ValueError("illumination support has too few measured texels")
    design, feature_names = polynomial_features(
        row,
        column,
        height=color.shape[0],
        width=color.shape[1],
        degree=degree,
    )
    target = color[row, column].astype(np.float64)
    weights = np.ones(len(row), dtype=np.float64)
    coefficients = np.zeros((design.shape[1], 3), dtype=np.float64)
    robust_scale = 1.0
    for _ in range(iterations):
        normal = np.einsum("ni,nj,n->ij", design, design, weights, optimize=True)
        normal += np.eye(normal.shape[0]) * ridge
        right = np.einsum("ni,nc,n->ic", design, target, weights, optimize=True)
        coefficients = np.linalg.solve(normal, right)
        fitted = np.einsum("ni,ic->nc", design, coefficients, optimize=True)
        residual = np.linalg.norm(target - fitted, axis=1)
        robust_scale = max(float(np.median(residual) * 1.4826), 1.0)
        cutoff = huber_delta * robust_scale
        weights = np.minimum(1.0, cutoff / np.maximum(residual, 1e-9))
    all_row, all_column = np.indices(observed.shape)
    all_design, _ = polynomial_features(
        all_row.ravel(),
        all_column.ravel(),
        height=color.shape[0],
        width=color.shape[1],
        degree=degree,
    )
    predicted = np.einsum("ni,ic->nc", all_design, coefficients, optimize=True).reshape(
        color.shape
    )
    result = np.clip(np.rint(predicted), 0, 255).astype(np.uint8)
    result[observed] = color[observed]
    fitted_support = np.einsum("ni,ic->nc", design, coefficients, optimize=True)
    residual_rgb = np.abs(target - fitted_support).mean(axis=1)
    return result, {
        "method": "robust_plane_uv_polynomial_illumination_field",
        "degree": degree,
        "feature_names": feature_names,
        "coefficients_rgb": coefficients.tolist(),
        "irls_iterations": iterations,
        "huber_delta": huber_delta,
        "ridge": ridge,
        "support_texels": len(row),
        "robust_scale_rgb_norm": robust_scale,
        "support_mean_abs_rgb_residual": float(residual_rgb.mean()),
        "support_p95_abs_rgb_residual": float(np.quantile(residual_rgb, 0.95)),
    }


def gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("wall illumination sigma must be positive")
    radius = max(1, int(np.ceil(sigma * truncate)))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinate / sigma) ** 2)
    return kernel / kernel.sum()


def separable_gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2
    source = values.astype(np.float64, copy=False)
    horizontal_pad = [(0, 0), (radius, radius)] + (
        [(0, 0)] if source.ndim == 3 else []
    )
    padded = np.pad(source, horizontal_pad, mode="edge")
    horizontal = np.zeros_like(source, dtype=np.float64)
    for index, weight in enumerate(kernel):
        horizontal += padded[:, index : index + source.shape[1]] * weight
    vertical_pad = [(radius, radius), (0, 0)] + (
        [(0, 0)] if source.ndim == 3 else []
    )
    padded = np.pad(horizontal, vertical_pad, mode="edge")
    result = np.zeros_like(horizontal, dtype=np.float64)
    for index, weight in enumerate(kernel):
        result += padded[index : index + source.shape[0]] * weight
    return result


def normalized_plane_illumination_field(
    color: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
    *,
    sigma: float,
    blend_weight_scale: float,
    color_clip_margin: float,
    fallback_degree: int,
    fallback_iterations: int,
    huber_delta: float,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    fallback, fallback_details = robust_plane_illumination_field(
        color,
        observed,
        support,
        degree=fallback_degree,
        iterations=fallback_iterations,
        huber_delta=huber_delta,
        ridge=ridge,
    )
    weight = separable_gaussian_blur(support.astype(np.float64), sigma)
    weighted_color = separable_gaussian_blur(
        color.astype(np.float64) * support[..., None],
        sigma,
    )
    if blend_weight_scale <= 0:
        raise ValueError("wall kernel blend weight scale must be positive")
    if color_clip_margin < 0:
        raise ValueError("wall illumination color clip margin must be non-negative")
    has_kernel_support = weight > 0.0
    local = fallback.astype(np.float64)
    local[has_kernel_support] = (
        weighted_color[has_kernel_support] / weight[has_kernel_support, None]
    )
    blend = weight / (weight + blend_weight_scale)
    predicted = (
        local * blend[..., None]
        + fallback.astype(np.float64) * (1.0 - blend[..., None])
    )
    support_color = color[support].astype(np.float64)
    lower = np.quantile(support_color, 0.02, axis=0) - color_clip_margin
    upper = np.quantile(support_color, 0.98, axis=0) + color_clip_margin
    predicted = np.clip(predicted, lower[None, None, :], upper[None, None, :])
    model = np.clip(np.rint(predicted), 0, 255).astype(np.uint8)
    result = model.copy()
    result[observed] = color[observed]
    return result, model, {
        "method": "robust_plane_uv_normalized_convolution_illumination_field",
        "gaussian_sigma_texels": sigma,
        "gaussian_kernel_radius_texels": len(gaussian_kernel_1d(sigma)) // 2,
        "blend_rule": "kernel_weight / (kernel_weight + blend_weight_scale)",
        "blend_weight_scale": blend_weight_scale,
        "blend_fraction_greater_than_half": float((blend > 0.5).mean()),
        "blend_minimum": float(blend.min()),
        "blend_maximum": float(blend.max()),
        "color_clip_quantiles": [0.02, 0.98],
        "color_clip_margin": color_clip_margin,
        "color_clip_lower_rgb": lower.tolist(),
        "color_clip_upper_rgb": upper.tolist(),
        "support_texels": int(support.sum()),
        "atlas_fraction_with_kernel_support": float(has_kernel_support.mean()),
        "minimum_nonzero_kernel_weight": (
            float(weight[has_kernel_support].min())
            if np.any(has_kernel_support)
            else None
        ),
        "fallback": fallback_details,
    }


def apply_reliable_boundary_residual_collar(
    model: np.ndarray,
    color: np.ndarray,
    observed: np.ndarray,
    reliable_observed: np.ndarray,
    *,
    width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if width < 0:
        raise ValueError("wall residual collar width must be non-negative")
    synthetic = ~observed
    boundary_seed = (
        observed
        & reliable_observed
        & geometry.binary_dilate(synthetic, 1)
    )
    result = model.astype(np.float64, copy=True)
    if width == 0 or not np.any(boundary_seed):
        result[observed] = color[observed]
        return np.clip(np.rint(result), 0, 255).astype(np.uint8), {
            "width_texels": width,
            "reliable_boundary_seed_texels": int(boundary_seed.sum()),
            "corrected_synthetic_texels": 0,
            "maximum_propagated_distance_texels": 0,
        }
    seed_residual = np.zeros_like(result, dtype=np.float64)
    seed_residual[boundary_seed] = (
        color[boundary_seed].astype(np.float64) - model[boundary_seed]
    )
    sigma = max(width / 3.0, 1.0)
    smoothed_weight = separable_gaussian_blur(boundary_seed.astype(np.float64), sigma)
    smoothed_residual = separable_gaussian_blur(seed_residual, sigma)
    propagated = np.zeros_like(result, dtype=np.float64)
    weighted = smoothed_weight > 1e-10
    propagated[weighted] = (
        smoothed_residual[weighted] / smoothed_weight[weighted, None]
    )
    candidate = synthetic & geometry.binary_dilate(boundary_seed, width)
    candidate_row, candidate_column = np.nonzero(candidate)
    seed_row, seed_column = np.nonzero(boundary_seed)
    nearest_distance = np.full(len(candidate_row), np.inf, dtype=np.float64)
    chunk_size = 512
    seed_coordinates = np.column_stack([seed_row, seed_column]).astype(np.float64)
    for start in range(0, len(candidate_row), chunk_size):
        stop = min(start + chunk_size, len(candidate_row))
        target = np.column_stack(
            [candidate_row[start:stop], candidate_column[start:stop]]
        ).astype(np.float64)
        squared = np.sum((target[:, None, :] - seed_coordinates[None, :, :]) ** 2, axis=2)
        nearest_distance[start:stop] = np.sqrt(squared.min(axis=1))
    normalized = np.clip(1.0 - (nearest_distance - 1.0) / max(width, 1), 0.0, 1.0)
    decay = normalized * normalized * (3.0 - 2.0 * normalized)
    result[candidate_row, candidate_column] += (
        propagated[candidate_row, candidate_column] * decay[:, None]
    )
    effective = decay > 0.0
    corrected = np.zeros_like(observed)
    corrected[candidate_row[effective], candidate_column[effective]] = True
    corrected &= weighted
    result[observed] = color[observed]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8), {
        "width_texels": width,
        "residual_smoothing_sigma_texels": sigma,
        "decay": "euclidean_distance_smoothstep",
        "reliable_boundary_seed_texels": int(boundary_seed.sum()),
        "corrected_synthetic_texels": int(corrected.sum()),
        "maximum_propagated_distance_texels": (
            float(nearest_distance[effective].max()) if np.any(effective) else 0.0
        ),
        "interior_correction_after_collar": 0,
        "contract": (
            "reliable boundary residual decays to zero; it cannot determine the interior "
            "low-frequency illumination field"
        ),
    }


def measured_gradient_cost(color: np.ndarray, observed: np.ndarray, axis: int) -> float:
    if axis == 0:
        valid = observed[1:] & observed[:-1]
        difference = np.abs(color[1:].astype(np.float64) - color[:-1])
    else:
        valid = observed[:, 1:] & observed[:, :-1]
        difference = np.abs(color[:, 1:].astype(np.float64) - color[:, :-1])
    if not np.any(valid):
        return float("inf")
    return float(difference[valid].mean())


def nearest_indices(known: np.ndarray, length: int) -> np.ndarray:
    positions = np.arange(length)
    insertion = np.searchsorted(known, positions)
    left_index = np.clip(insertion - 1, 0, len(known) - 1)
    right_index = np.clip(insertion, 0, len(known) - 1)
    left = known[left_index]
    right = known[right_index]
    return np.where(np.abs(positions - left) <= np.abs(right - positions), left, right)


def directional_floor_extend(
    color: np.ndarray,
    observed: np.ndarray,
    *,
    fallback_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    costs = {
        "vertical_uv": measured_gradient_cost(color, observed, 0),
        "horizontal_uv": measured_gradient_cost(color, observed, 1),
    }
    direction_axis = 0 if costs["vertical_uv"] <= costs["horizontal_uv"] else 1
    result = np.zeros_like(color, dtype=np.float64)
    seeded = np.zeros_like(observed)
    if direction_axis == 0:
        for column in range(color.shape[1]):
            known = np.flatnonzero(observed[:, column])
            if len(known):
                source = nearest_indices(known, color.shape[0])
                result[:, column] = color[source, column]
                seeded[:, column] = True
    else:
        for row in range(color.shape[0]):
            known = np.flatnonzero(observed[row])
            if len(known):
                source = nearest_indices(known, color.shape[1])
                result[row] = color[row, source]
                seeded[row] = True
    result[observed] = color[observed]
    seeded |= observed
    if not np.all(seeded):
        initialized = wavefront_initialize(
            np.clip(np.rint(result), 0, 255).astype(np.uint8),
            seeded,
        )
        result[~seeded] = initialized[~seeded]
    synthetic = ~observed
    for _ in range(fallback_iterations):
        average, _ = neighbor_average(result)
        result[synthetic] = result[synthetic] * 0.92 + average[synthetic] * 0.08
    result[observed] = color[observed]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8), {
        "method": "directional_nearest_measured_strip_extension",
        "selected_direction": "vertical_uv" if direction_axis == 0 else "horizontal_uv",
        "measured_gradient_cost": costs,
        "seeded_texel_fraction_before_fallback": float(seeded.mean()),
    }


def boundary_discontinuity(
    completed: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float | int | None]:
    values: list[np.ndarray] = []
    horizontal = observed[:, 1:] != observed[:, :-1]
    if np.any(horizontal):
        difference = np.abs(completed[:, 1:].astype(np.float64) - completed[:, :-1])
        values.append(difference[horizontal].mean(axis=1))
    vertical = observed[1:] != observed[:-1]
    if np.any(vertical):
        difference = np.abs(completed[1:].astype(np.float64) - completed[:-1])
        values.append(difference[vertical].mean(axis=1))
    if not values:
        return {"edge_count": 0, "mean_abs_rgb_delta": None, "p95_abs_rgb_delta": None}
    joined = np.concatenate(values)
    return {
        "edge_count": len(joined),
        "mean_abs_rgb_delta": float(joined.mean()),
        "p95_abs_rgb_delta": float(np.quantile(joined, 0.95)),
    }


def boundary_normal_gradient_continuity(
    completed: np.ndarray,
    observed: np.ndarray,
) -> dict[str, float | int | None]:
    color_delta: list[float] = []
    gradient_delta: list[float] = []
    height, width = observed.shape

    def add_triplet(
        observed_color: np.ndarray,
        synthetic_color: np.ndarray,
        observed_outer: np.ndarray | None,
        synthetic_inner: np.ndarray | None,
    ) -> None:
        cross_gradient = synthetic_color.astype(np.float64) - observed_color
        color_delta.append(float(np.abs(cross_gradient).mean()))
        if observed_outer is not None:
            observed_gradient = observed_color.astype(np.float64) - observed_outer
            gradient_delta.append(float(np.abs(cross_gradient - observed_gradient).mean()))
        if synthetic_inner is not None:
            synthetic_gradient = synthetic_inner.astype(np.float64) - synthetic_color
            gradient_delta.append(float(np.abs(cross_gradient - synthetic_gradient).mean()))

    for row in range(height):
        for column in np.flatnonzero(observed[row, 1:] != observed[row, :-1]):
            if observed[row, column]:
                add_triplet(
                    completed[row, column],
                    completed[row, column + 1],
                    completed[row, column - 1]
                    if column > 0 and observed[row, column - 1]
                    else None,
                    completed[row, column + 2]
                    if column + 2 < width and not observed[row, column + 2]
                    else None,
                )
            else:
                add_triplet(
                    completed[row, column + 1],
                    completed[row, column],
                    completed[row, column + 2]
                    if column + 2 < width and observed[row, column + 2]
                    else None,
                    completed[row, column - 1]
                    if column > 0 and not observed[row, column - 1]
                    else None,
                )
    for row in range(height - 1):
        for column in np.flatnonzero(observed[row + 1] != observed[row]):
            if observed[row, column]:
                add_triplet(
                    completed[row, column],
                    completed[row + 1, column],
                    completed[row - 1, column]
                    if row > 0 and observed[row - 1, column]
                    else None,
                    completed[row + 2, column]
                    if row + 2 < height and not observed[row + 2, column]
                    else None,
                )
            else:
                add_triplet(
                    completed[row + 1, column],
                    completed[row, column],
                    completed[row + 2, column]
                    if row + 2 < height and observed[row + 2, column]
                    else None,
                    completed[row - 1, column]
                    if row > 0 and not observed[row - 1, column]
                    else None,
                )
    return {
        "boundary_edge_count": len(color_delta),
        "boundary_color_mean_abs_rgb_delta": (
            float(np.mean(color_delta)) if color_delta else None
        ),
        "boundary_color_p95_abs_rgb_delta": (
            float(np.quantile(color_delta, 0.95)) if color_delta else None
        ),
        "boundary_normal_gradient_pair_count": len(gradient_delta),
        "boundary_normal_gradient_mean_abs_rgb_delta": (
            float(np.mean(gradient_delta)) if gradient_delta else None
        ),
        "boundary_normal_gradient_p95_abs_rgb_delta": (
            float(np.quantile(gradient_delta, 0.95)) if gradient_delta else None
        ),
    }


def make_atlas_contact_sheet(records: list[dict[str, Any]], path: Path) -> None:
    width, height, label = 360, 260, 22
    sheet = Image.new("RGB", (width * 5, (height + label) * len(records)), "#111111")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(records):
        measured = Image.open(record["measured_rgb"]).convert("RGB")
        observed = Image.open(record["observed_mask"]).convert("L")
        completed = Image.open(record["completed_rgb"]).convert("RGB")
        synthetic_anchor = Image.open(record["synthetic_anchor_mask"]).convert("L")
        interpolated = Image.open(record["interpolated_mask"]).convert("L")
        images = (
            measured,
            observed.convert("RGB"),
            synthetic_anchor.convert("RGB"),
            interpolated.convert("RGB"),
            completed,
        )
        titles = (
            f"plane {record['plane_id']} {record['semantic_role']} measured",
            f"observed texels {record['observed_fraction']:.1%}",
            f"synthetic anchor {record['synthetic_anchor_fraction']:.1%}",
            f"interpolated {record['interpolated_fraction']:.1%}",
            f"completed: {record['method']}",
        )
        y = row * (height + label)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((width, height), Image.Resampling.NEAREST)
            canvas = Image.new("RGB", (width, height), "#202020")
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            x = column * width
            sheet.paste(canvas, (x, y + label))
            draw.text((x + 6, y + 5), title, fill="#f4f4f4")
        for image in images:
            image.close()
        measured.close()
        observed.close()
        completed.close()
        synthetic_anchor.close()
        interpolated.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def make_view_contact_sheet(records: list[dict[str, Any]], path: Path, samples: int) -> None:
    count = min(samples, len(records))
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    width, height, label = 288, 162, 22
    sheet = Image.new("RGB", (width * 5, (height + label) * count), "#111111")
    draw = ImageDraw.Draw(sheet)
    for row, record_index in enumerate(indices):
        record = records[int(record_index)]
        source = Image.open(record["source_frame"]).convert("RGB")
        labels = Image.open(record["plane_labels"]).convert("RGB")
        completed = Image.open(record["completed_frame"]).convert("RGB")
        synthetic = Image.open(record["synthetic_mask"]).convert("L")
        overlay = source.copy()
        overlay.paste(Image.new("RGB", source.size, (255, 65, 40)), mask=synthetic)
        synthetic_rgb = Image.new("RGB", source.size, "black")
        synthetic_rgb.paste(completed, mask=synthetic)
        images = (source, labels, completed, overlay, synthetic_rgb)
        titles = (
            f"measured source {record['frame_id']}",
            "shared plane atlas ID",
            "completed candidate",
            "synthetic provenance overlay",
            "synthetic contribution only",
        )
        y = row * (height + label)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), "#202020")
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            x = column * width
            sheet.paste(canvas, (x, y + label))
            draw.text((x + 6, y + 5), title, fill="#f4f4f4")
        for image in images:
            image.close()
        source.close()
        labels.close()
        completed.close()
        synthetic.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def make_footprint_provenance_contact_sheet(
    records: list[dict[str, Any]],
    path: Path,
    samples: int,
) -> None:
    count = min(samples, len(records))
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    width, height, label = 288, 162, 22
    sheet = Image.new("RGB", (width * 5, (height + label) * count), "#111111")
    draw = ImageDraw.Draw(sheet)
    for output_row, record_index in enumerate(indices):
        record = records[int(record_index)]
        source = Image.open(record["source_frame"]).convert("RGB")
        completed = Image.open(record["completed_frame"]).convert("RGB")
        core = Image.open(record["core_shared_atlas_mask"]).convert("L")
        feather = Image.open(record["screen_space_feather_mask"]).convert("L")
        labels = Image.new("RGB", source.size, "#101010")
        labels.paste(Image.new("RGB", source.size, (45, 190, 105)), mask=core)
        labels.paste(Image.new("RGB", source.size, (255, 190, 45)), mask=feather)
        core_overlay = source.copy()
        core_overlay.paste(Image.new("RGB", source.size, (45, 190, 105)), mask=core)
        feather_overlay = completed.copy()
        feather_overlay.paste(Image.new("RGB", source.size, (255, 190, 45)), mask=feather)
        images = (source, completed, labels, core_overlay, feather_overlay)
        titles = (
            f"source {record['frame_id']}",
            "completed plane rectangle",
            "outside/core/feather provenance",
            "core shared-atlas overlay",
            "screen feather overlay",
        )
        y = output_row * (height + label)
        for column, (image, title) in enumerate(zip(images, titles, strict=True)):
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), "#202020")
            canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
            x = column * width
            sheet.paste(canvas, (x, y + label))
            draw.text((x + 6, y + 5), title, fill="#f4f4f4")
        for image in images:
            image.close()
        source.close()
        completed.close()
        core.close()
        feather.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def plane_target_pixel_counts(
    geometry_report: dict[str, Any],
    plane_ids: list[int],
) -> dict[int, int]:
    counts = {plane_id: 0 for plane_id in plane_ids}
    for record in geometry_report.get("frame_records", []):
        if not isinstance(record, dict):
            continue
        pixel_counts = record.get("plane_pixel_counts")
        if not isinstance(pixel_counts, dict):
            continue
        for plane_id in plane_ids:
            value = pixel_counts.get(str(plane_id), 0)
            if isinstance(value, int) and not isinstance(value, bool):
                counts[plane_id] += value
    return counts


def excluded_evidence(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        record: dict[str, Any] = {
            "path": str(resolved),
            "sha256": sha256_file(resolved),
            "role": "excluded_texture_input_evidence_only",
        }
        if resolved.suffix == ".json":
            value = read_json(resolved)
            if isinstance(value, dict):
                record["status"] = value.get("status")
                record["promotion_allowed"] = value.get("promotion_allowed")
                record["blocking_findings"] = value.get("blocking_findings")
                record["next_use"] = value.get("next_use")
        records.append(record)
    return records


def validate_declared_sha256(path: Path, declared: str, role: str) -> str:
    if len(declared) != 64 or any(character not in "0123456789abcdef" for character in declared):
        raise ValueError(f"{role} SHA-256 must be 64 lowercase hexadecimal characters")
    actual = sha256_file(path)
    if actual != declared:
        raise ValueError(f"{role} SHA-256 mismatch")
    return actual


def load_binary_mask(path: Path, role: str) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    values = np.unique(mask)
    if not set(values.tolist()).issubset({0, 255}):
        raise ValueError(f"{role} must contain only 0 and 255")
    return mask == 255


def load_synthetic_anchor_inputs(
    args: argparse.Namespace,
    geometry_report: dict[str, Any],
) -> dict[str, Any] | None:
    values = {
        "frame_id": args.synthetic_anchor_frame_id,
        "candidate": args.synthetic_anchor_candidate,
        "candidate_sha256": args.synthetic_anchor_candidate_sha256,
        "edit_mask": args.synthetic_anchor_edit_mask,
        "edit_mask_sha256": args.synthetic_anchor_edit_mask_sha256,
        "diffusion_receipt": args.synthetic_anchor_diffusion_receipt,
        "diffusion_receipt_sha256": args.synthetic_anchor_diffusion_receipt_sha256,
    }
    provided = {key for key, value in values.items() if value is not None}
    if not provided:
        return None
    if provided != set(values):
        missing = sorted(set(values) - provided)
        raise ValueError(f"synthetic anchor inputs are incomplete: {missing}")
    frame_id = str(values["frame_id"])
    frame_records = {
        str(record["frame_id"]): record for record in geometry_report["frame_records"]
    }
    if frame_id not in frame_records:
        raise ValueError(f"synthetic anchor frame is absent from geometry report: {frame_id}")
    candidate_path = Path(values["candidate"]).expanduser().resolve()
    edit_mask_path = Path(values["edit_mask"]).expanduser().resolve()
    receipt_path = Path(values["diffusion_receipt"]).expanduser().resolve()
    for path in (candidate_path, edit_mask_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate_sha = validate_declared_sha256(
        candidate_path,
        str(values["candidate_sha256"]),
        "synthetic anchor candidate",
    )
    edit_mask_sha = validate_declared_sha256(
        edit_mask_path,
        str(values["edit_mask_sha256"]),
        "synthetic anchor edit mask",
    )
    receipt_sha = validate_declared_sha256(
        receipt_path,
        str(values["diffusion_receipt_sha256"]),
        "synthetic anchor diffusion receipt",
    )
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("synthetic anchor diffusion receipt must be a JSON object")
    if receipt.get("kind") != "video2world.diffusion_anchor_inpaint_run":
        raise ValueError("synthetic anchor receipt kind is not diffusion anchor inpaint")
    if receipt.get("status") != "generated_candidates_pending_review":
        raise ValueError("synthetic anchor receipt status is not consumable")
    if receipt.get("promotion_allowed") is not False:
        raise ValueError("synthetic anchor receipt must remain non-promotable")
    receipt_candidates = [
        candidate
        for candidate in receipt.get("candidates", [])
        if isinstance(candidate, dict)
        and candidate.get("artifact", {}).get("sha256") == candidate_sha
    ]
    if not receipt_candidates:
        raise ValueError("synthetic anchor candidate is not bound by diffusion receipt")
    if not any(
        candidate.get("metrics", {}).get("outside_rgb_exact") is True
        and candidate.get("metrics", {}).get("outside_changed_pixels") == 0
        for candidate in receipt_candidates
    ):
        raise ValueError("synthetic anchor candidate is not exact outside its edit mask")
    receipt_mask_sha = receipt.get("inputs", {}).get("residual_mask", {}).get("sha256")
    if receipt_mask_sha != edit_mask_sha:
        raise ValueError("synthetic anchor edit mask is not bound by diffusion receipt")
    with Image.open(candidate_path) as image:
        candidate = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    edit_mask = load_binary_mask(edit_mask_path, "synthetic anchor edit mask")
    geometry_record = frame_records[frame_id]
    removal_mask = load_binary_mask(
        Path(geometry_record["removal_mask"]).expanduser().resolve(),
        "geometry removal mask",
    )
    if candidate.shape[:2] != edit_mask.shape or edit_mask.shape != removal_mask.shape:
        raise ValueError("synthetic anchor candidate/edit/removal dimensions differ")
    if np.any(edit_mask & ~removal_mask):
        raise ValueError("synthetic anchor edit mask is not a subset of geometry removal mask")
    receipt_mask_pixels = receipt.get("inputs", {}).get("residual_mask", {}).get(
        "masked_pixels"
    )
    if receipt_mask_pixels is not None and int(receipt_mask_pixels) != int(edit_mask.sum()):
        raise ValueError("synthetic anchor edit mask pixel count differs from receipt")
    source_path = Path(geometry_record["prefill_frame"]).expanduser().resolve()
    receipt_source_sha = receipt.get("inputs", {}).get("source_rgb", {}).get("sha256")
    source_sha = sha256_file(source_path)
    guide_values = {
        "guide": getattr(args, "synthetic_anchor_source_guide", None),
        "guide_sha256": getattr(args, "synthetic_anchor_source_guide_sha256", None),
        "guide_report": getattr(args, "synthetic_anchor_source_guide_report", None),
        "guide_report_sha256": getattr(
            args,
            "synthetic_anchor_source_guide_report_sha256",
            None,
        ),
    }
    guide_provided = {key for key, value in guide_values.items() if value is not None}
    if guide_provided and guide_provided != set(guide_values):
        missing = sorted(set(guide_values) - guide_provided)
        raise ValueError(f"synthetic anchor source guide inputs are incomplete: {missing}")
    source_lineage: dict[str, Any]
    if receipt_source_sha == source_sha:
        if guide_provided:
            raise ValueError("source guide inputs are not allowed for a planar prefill source")
        source_lineage = {
            "role": "planar_geometry_prefill",
            "source_rgb": str(source_path),
            "source_rgb_sha256": source_sha,
            "claims_measured_donor": False,
        }
    else:
        if guide_provided != set(guide_values):
            raise ValueError(
                "non-prefill synthetic anchor source requires a bound synthetic guide"
            )
        guide_path = Path(guide_values["guide"]).expanduser().resolve()
        guide_report_path = Path(guide_values["guide_report"]).expanduser().resolve()
        for path in (guide_path, guide_report_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        guide_sha = validate_declared_sha256(
            guide_path,
            str(guide_values["guide_sha256"]),
            "synthetic anchor source guide",
        )
        guide_report_sha = validate_declared_sha256(
            guide_report_path,
            str(guide_values["guide_report_sha256"]),
            "synthetic anchor source guide report",
        )
        if receipt_source_sha != guide_sha:
            raise ValueError("diffusion receipt source does not match the bound synthetic guide")
        guide_report = require_dict(
            read_json(guide_report_path),
            "synthetic anchor source guide report",
        )
        if guide_report.get("status") != "texture_candidate_review_pending":
            raise ValueError("synthetic guide report is not a review-pending texture candidate")
        if guide_report.get("promotion_approved") is not False:
            raise ValueError("synthetic guide report must remain non-promotable")
        if guide_report.get("eligible_as_round04_clean_plate") is not False:
            raise ValueError("synthetic guide report must remain ineligible for round04")
        planar_report_path = args.planar_report.expanduser().resolve()
        if guide_report.get("planar_geometry_report_sha256") != sha256_file(
            planar_report_path
        ):
            raise ValueError("synthetic guide report is not bound to this planar geometry")
        guide_provenance = require_dict(
            guide_report.get("provenance"),
            "synthetic guide provenance",
        )
        if guide_provenance.get("synthetic_pixels_are_measured") is not False:
            raise ValueError("synthetic guide cannot claim measured donor provenance")
        if guide_provenance.get("one_atlas_per_plane") is not True:
            raise ValueError("synthetic guide was not rendered from shared plane atlases")
        if guide_provenance.get("per_frame_generation_used") is not False:
            raise ValueError("synthetic guide used per-frame generation")
        guide_gates = require_dict(guide_report.get("gates"), "synthetic guide gates")
        required_guide_gates = (
            "measured_atlas_texels_rgb_exact",
            "outside_synthetic_mask_rgb_exact",
            "all_synthetic_pixels_assigned_from_shared_atlas",
            "same_atlas_texel_has_identical_rgb_across_views",
        )
        if any(guide_gates.get(key) is not True for key in required_guide_gates):
            raise ValueError("synthetic guide technical lineage gates did not pass")
        guide_frames = {
            str(item["frame_id"]): item
            for item in guide_report.get("frame_records", [])
            if isinstance(item, dict) and "frame_id" in item
        }
        guide_frame = guide_frames.get(frame_id)
        if guide_frame is None:
            raise ValueError("synthetic guide report does not contain the anchor frame")
        if guide_frame.get("completed_frame_sha256") != guide_sha:
            raise ValueError("synthetic guide image is not bound by its texture report")
        if guide_frame.get("synthetic_mask_sha256") != edit_mask_sha:
            raise ValueError("synthetic guide report mask differs from the anchor edit mask")
        with Image.open(guide_path) as image:
            guide = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        with Image.open(source_path) as image:
            planar_prefill = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if guide.shape != candidate.shape or planar_prefill.shape != candidate.shape:
            raise ValueError("synthetic guide, planar prefill, and candidate dimensions differ")
        outside_edit = ~edit_mask
        if not np.array_equal(guide[outside_edit], planar_prefill[outside_edit]):
            raise ValueError("synthetic guide differs from planar prefill outside edit mask")
        if not np.array_equal(candidate[outside_edit], planar_prefill[outside_edit]):
            raise ValueError(
                "synthetic anchor candidate differs from planar prefill outside edit mask"
            )
        source_lineage = {
            "role": "non_promotable_shared_atlas_synthetic_guide",
            "source_rgb": str(guide_path),
            "source_rgb_sha256": guide_sha,
            "texture_report": str(guide_report_path),
            "texture_report_sha256": guide_report_sha,
            "texture_report_status": guide_report["status"],
            "texture_report_promotion_approved": False,
            "texture_report_eligible_as_round04_clean_plate": False,
            "outside_edit_mask_matches_planar_prefill": True,
            "candidate_outside_edit_mask_matches_planar_prefill": True,
            "claims_measured_donor": False,
        }
    if receipt_source_sha == source_sha:
        with Image.open(source_path) as image:
            planar_prefill = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if candidate.shape != planar_prefill.shape:
            raise ValueError("synthetic anchor candidate and planar prefill dimensions differ")
        if not np.array_equal(candidate[~edit_mask], planar_prefill[~edit_mask]):
            raise ValueError(
                "synthetic anchor candidate differs from planar prefill outside edit mask"
            )
    return {
        "frame_id": frame_id,
        "candidate_path": candidate_path,
        "candidate_sha256": candidate_sha,
        "candidate": candidate,
        "edit_mask_path": edit_mask_path,
        "edit_mask_sha256": edit_mask_sha,
        "edit_mask": edit_mask,
        "diffusion_receipt_path": receipt_path,
        "diffusion_receipt_sha256": receipt_sha,
        "diffusion_receipt_status": receipt["status"],
        "geometry_record": geometry_record,
        "source_lineage": source_lineage,
        "claims_measured_donor": False,
        "provenance_class": "synthetic_anchor",
    }


def project_synthetic_anchor_to_atlases(
    anchor: dict[str, Any] | None,
    geometry_report: dict[str, Any],
    camera_info: dict[str, Any],
    plane_records: dict[int, dict[str, Any]],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any] | None]:
    projected: dict[int, dict[str, np.ndarray]] = {}
    for atlas_record in geometry_report["texture_atlases"]:
        width, height = (int(value) for value in atlas_record["dimensions"])
        projected[int(atlas_record["plane_id"])] = {
            "color": np.zeros((height, width, 3), dtype=np.uint8),
            "support": np.zeros((height, width), dtype=bool),
            "sample_count": np.zeros((height, width), dtype=np.uint32),
        }
    if anchor is None:
        return projected, None
    record = anchor["geometry_record"]
    with Image.open(record["plane_ids"]) as image:
        plane_ids = np.asarray(image, dtype=np.uint16).copy()
    with np.load(record["geometry_depth"]) as archive:
        depth = np.asarray(archive["depth"], dtype=np.float32)
    edit_mask = anchor["edit_mask"]
    if plane_ids.shape != edit_mask.shape or depth.shape != edit_mask.shape:
        raise ValueError("synthetic anchor edit, plane-id, and depth dimensions differ")
    eligible = edit_mask & (plane_ids > 0) & np.isfinite(depth) & (depth > 0.0)
    if not np.any(eligible):
        raise ValueError("synthetic anchor has no valid planar geometry pixels")
    y, x = np.nonzero(eligible)
    frame_id = anchor["frame_id"]
    intrinsic = geometry.frame_intrinsic(camera_info, frame_id)
    matrix = geometry.world_to_camera(camera_info, frame_id)
    rays, center = geometry.rays_for_pixels(
        x.astype(np.float64),
        y.astype(np.float64),
        intrinsic,
        matrix,
    )
    points = center[None, :] + rays * depth[y, x, None]
    colors = anchor["candidate"][y, x]
    atlas_records = {
        int(record["plane_id"]): record for record in geometry_report["texture_atlases"]
    }
    plane_projection_records: list[dict[str, Any]] = []
    for plane_id, plane_record in plane_records.items():
        selected = plane_ids[y, x] == plane_id
        if not np.any(selected):
            plane_projection_records.append(
                {"plane_id": plane_id, "eligible_pixels": 0, "projected_texels": 0}
            )
            continue
        atlas_record = atlas_records[plane_id]
        target = projected[plane_id]
        origin = np.asarray(atlas_record["origin_uv"], dtype=np.float64)
        texel_size = float(atlas_record["texel_size_world_units"])
        selected_points = points[selected]
        u = geometry.row_dot(
            selected_points,
            np.asarray(plane_record["basis_u"], dtype=np.float64),
        )
        v = geometry.row_dot(
            selected_points,
            np.asarray(plane_record["basis_v"], dtype=np.float64),
        )
        column = np.floor((u - origin[0]) / texel_size).astype(np.int64)
        row = np.floor((v - origin[1]) / texel_size).astype(np.int64)
        inside = (
            (column >= 0)
            & (column < target["support"].shape[1])
            & (row >= 0)
            & (row < target["support"].shape[0])
        )
        row = row[inside]
        column = column[inside]
        selected_colors = colors[selected][inside].astype(np.float64)
        flat = row * target["support"].shape[1] + column
        count_flat = np.zeros(target["support"].size, dtype=np.uint32)
        np.add.at(count_flat, flat, 1)
        color_sum = np.zeros((target["support"].size, 3), dtype=np.float64)
        np.add.at(color_sum, flat, selected_colors)
        supported_flat = count_flat > 0
        target["sample_count"][:] = count_flat.reshape(target["support"].shape)
        target["support"][:] = supported_flat.reshape(target["support"].shape)
        averaged = np.zeros_like(color_sum, dtype=np.uint8)
        averaged[supported_flat] = np.clip(
            np.rint(color_sum[supported_flat] / count_flat[supported_flat, None]),
            0,
            255,
        ).astype(np.uint8)
        target["color"][:] = averaged.reshape(target["color"].shape)
        plane_projection_records.append(
            {
                "plane_id": plane_id,
                "eligible_pixels": int(selected.sum()),
                "inside_atlas_pixels": int(inside.sum()),
                "projected_texels": int(supported_flat.sum()),
                "texel_collisions": int(inside.sum() - supported_flat.sum()),
                "maximum_samples_per_texel": int(count_flat.max(initial=0)),
            }
        )
    inside_atlas_pixels = sum(
        int(item.get("inside_atlas_pixels", 0)) for item in plane_projection_records
    )
    if inside_atlas_pixels != int(eligible.sum()):
        raise ValueError("some synthetic anchor pixels do not map inside a declared plane atlas")
    return projected, {
        "frame_id": frame_id,
        "candidate": str(anchor["candidate_path"]),
        "candidate_sha256": anchor["candidate_sha256"],
        "edit_mask": str(anchor["edit_mask_path"]),
        "edit_mask_sha256": anchor["edit_mask_sha256"],
        "diffusion_receipt": str(anchor["diffusion_receipt_path"]),
        "diffusion_receipt_sha256": anchor["diffusion_receipt_sha256"],
        "diffusion_receipt_status": anchor["diffusion_receipt_status"],
        "source_lineage": anchor["source_lineage"],
        "eligible_geometry_pixels": int(eligible.sum()),
        "edit_mask_pixels": int(edit_mask.sum()),
        "all_eligible_pixels_inside_edit_mask": True,
        "all_eligible_pixels_inside_declared_atlases": True,
        "claims_measured_donor": False,
        "provenance_class": "synthetic_anchor",
        "plane_projection_records": plane_projection_records,
    }


def compose_atlas_provenance(
    measured: np.ndarray,
    observed: np.ndarray,
    projected_anchor: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if measured.shape[:2] != observed.shape:
        raise ValueError("measured atlas and observed mask dimensions differ")
    if projected_anchor["color"].shape != measured.shape:
        raise ValueError("synthetic anchor color atlas dimensions differ")
    if projected_anchor["support"].shape != observed.shape:
        raise ValueError("synthetic anchor support atlas dimensions differ")
    synthetic_anchor = projected_anchor["support"] & ~observed
    interpolated = ~(observed | synthetic_anchor)
    combined_color = measured.copy()
    combined_color[synthetic_anchor] = projected_anchor["color"][synthetic_anchor]
    if not np.array_equal(
        observed | synthetic_anchor | interpolated,
        np.ones_like(observed),
    ):
        raise ValueError("atlas provenance categories do not cover the atlas")
    if (
        np.any(observed & synthetic_anchor)
        or np.any(observed & interpolated)
        or np.any(synthetic_anchor & interpolated)
    ):
        raise ValueError("atlas provenance categories overlap")
    return combined_color, observed, synthetic_anchor, interpolated


def rectangular_footprint_feather_weight(
    footprint: np.ndarray,
    *,
    padding: int,
    feather_width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if footprint.ndim != 2 or footprint.dtype != np.bool_:
        raise ValueError("plane footprint must be a boolean HxW array")
    if feather_width <= 0:
        raise ValueError("plane footprint feather width must be positive")
    if padding < feather_width:
        raise ValueError("plane footprint padding must be at least the feather width")
    weight = np.zeros(footprint.shape, dtype=np.float32)
    if not np.any(footprint):
        return weight, {
            "footprint_texels": 0,
            "footprint_bounds_inclusive": None,
            "core_bounds_inclusive": None,
            "outer_bounds_inclusive": None,
            "padding_texels": padding,
            "feather_width_texels": feather_width,
            "all_footprint_texels_in_weight_one_core": True,
            "weight_one_core_texels": 0,
            "feather_texels": 0,
            "positive_weight_texels": 0,
            "boundary_shape": "axis_aligned_plane_uv_rectangle",
        }
    rows, columns = np.nonzero(footprint)
    minimum_row = int(rows.min())
    maximum_row = int(rows.max())
    minimum_column = int(columns.min())
    maximum_column = int(columns.max())
    height, width = footprint.shape
    core_padding = padding - feather_width
    core_minimum_row = max(0, minimum_row - core_padding)
    core_maximum_row = min(height - 1, maximum_row + core_padding)
    core_minimum_column = max(0, minimum_column - core_padding)
    core_maximum_column = min(width - 1, maximum_column + core_padding)
    outer_minimum_row = max(0, minimum_row - padding)
    outer_maximum_row = min(height - 1, maximum_row + padding)
    outer_minimum_column = max(0, minimum_column - padding)
    outer_maximum_column = min(width - 1, maximum_column + padding)
    row, column = np.indices(footprint.shape)
    row_distance = np.maximum(
        np.maximum(core_minimum_row - row, row - core_maximum_row),
        0,
    )
    column_distance = np.maximum(
        np.maximum(core_minimum_column - column, column - core_maximum_column),
        0,
    )
    rectangle_distance = np.maximum(row_distance, column_distance).astype(np.float32)
    normalized = np.clip(rectangle_distance / float(feather_width), 0.0, 1.0)
    smoothstep = normalized * normalized * (3.0 - 2.0 * normalized)
    inside_outer = (
        (row >= outer_minimum_row)
        & (row <= outer_maximum_row)
        & (column >= outer_minimum_column)
        & (column <= outer_maximum_column)
    )
    weight[inside_outer] = 1.0 - smoothstep[inside_outer]
    weight[footprint] = 1.0
    core = weight >= 1.0 - 1e-7
    feather = (weight > 0.0) & ~core
    if not np.all(core[footprint]):
        raise ValueError("plane footprint escaped the weight-one rectangle core")
    return weight, {
        "footprint_texels": int(footprint.sum()),
        "footprint_bounds_inclusive": [
            minimum_column,
            minimum_row,
            maximum_column,
            maximum_row,
        ],
        "core_bounds_inclusive": [
            core_minimum_column,
            core_minimum_row,
            core_maximum_column,
            core_maximum_row,
        ],
        "outer_bounds_inclusive": [
            outer_minimum_column,
            outer_minimum_row,
            outer_maximum_column,
            outer_maximum_row,
        ],
        "padding_texels": padding,
        "feather_width_texels": feather_width,
        "all_footprint_texels_in_weight_one_core": True,
        "weight_one_core_texels": int(core.sum()),
        "feather_texels": int(feather.sum()),
        "positive_weight_texels": int((weight > 0.0).sum()),
        "boundary_shape": "axis_aligned_plane_uv_rectangle",
        "weight_function": "one_minus_smoothstep_chebyshev_distance_from_core_rectangle",
    }


def composite_plane_footprint_render(
    source: np.ndarray,
    rendered: np.ndarray,
    weight: np.ndarray,
    original_removal: np.ndarray,
    protected_neighbor: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    shape = source.shape[:2]
    if source.shape != rendered.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source and plane render must be equal-size RGB arrays")
    if any(value.shape != shape for value in (weight, original_removal, protected_neighbor)):
        raise ValueError("plane feather weight and masks must match source dimensions")
    if np.any(~np.isfinite(weight)) or np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("plane feather weight must be finite and in [0, 1]")
    if np.any(original_removal & (weight < 1.0 - 1e-6)):
        raise ValueError("original cumulative removal pixels escaped the weight-one core")
    effective_weight = weight.astype(np.float32, copy=True)
    protected_override = protected_neighbor & ~original_removal
    effective_weight[protected_override] = 0.0
    core = effective_weight >= 1.0 - 1e-6
    feather = (effective_weight > 0.0) & ~core
    outside = effective_weight <= 0.0
    result = source.copy()
    result[core] = rendered[core]
    if np.any(feather):
        alpha = effective_weight[feather, None].astype(np.float64)
        blended = (
            rendered[feather].astype(np.float64) * alpha
            + source[feather].astype(np.float64) * (1.0 - alpha)
        )
        result[feather] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    outside_exact = bool(np.array_equal(result[outside], source[outside]))
    protected_exact = bool(
        np.array_equal(result[protected_override], source[protected_override])
    )
    original_core = bool(np.all(core[original_removal]))
    if not outside_exact or not protected_exact or not original_core:
        raise ValueError("plane footprint screen compositor invariants failed")
    return result, {
        "core_shared_atlas": core,
        "screen_space_feather": feather,
        "outside_source": outside,
    }, {
        "core_shared_atlas_pixels": int(core.sum()),
        "screen_space_feather_pixels": int(feather.sum()),
        "outside_source_pixels": int(outside.sum()),
        "protected_neighbor_pixels_forced_source_exact": int(protected_override.sum()),
        "protected_neighbor_overlap_with_original_removal_not_overridden": int(
            (protected_neighbor & original_removal).sum()
        ),
        "all_original_removal_pixels_in_core": original_core,
        "outside_outer_mask_rgb_exact": outside_exact,
        "protected_neighbor_pixels_rgb_exact": protected_exact,
        "core_claims_measured_donor": False,
        "feather_claims_measured_donor": False,
    }


def filter_footprints_for_completed_atlases(
    footprints: dict[int, np.ndarray],
    completed_atlases: dict[int, dict[str, Any]],
    atlas_records_by_id: dict[int, dict[str, Any]],
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    eligible: dict[int, np.ndarray] = {}
    skipped: list[dict[str, Any]] = []
    for plane_id, footprint in footprints.items():
        if plane_id in completed_atlases and plane_id in atlas_records_by_id:
            eligible[plane_id] = footprint
            continue
        skipped.append(
            {
                "plane_id": plane_id,
                "reason": "no_completed_atlas_for_footprint",
                "footprint_texels": int(footprint.sum()),
                "rendering_required": bool(np.any(footprint)),
            }
        )
    return eligible, skipped


def project_cumulative_removal_footprints(
    geometry_report: dict[str, Any],
    camera_info: dict[str, Any],
    plane_records: dict[int, dict[str, Any]],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    atlas_records = {
        int(record["plane_id"]): record for record in geometry_report["texture_atlases"]
    }
    footprints = {
        plane_id: np.zeros(
            (
                int(atlas_records[plane_id]["dimensions"][1]),
                int(atlas_records[plane_id]["dimensions"][0]),
            ),
            dtype=bool,
        )
        for plane_id in plane_records
    }
    frame_records: list[dict[str, Any]] = []
    all_inside = True
    total_pixels = 0
    for record in geometry_report["frame_records"]:
        frame_id = str(record["frame_id"])
        removal_path = Path(record["removal_mask"]).resolve()
        removal = load_binary_mask(removal_path, f"{frame_id} cumulative removal mask")
        with Image.open(record["plane_ids"]) as image:
            plane_ids = np.asarray(image, dtype=np.uint16).copy()
        with np.load(record["geometry_depth"]) as archive:
            depth = np.asarray(archive["depth"], dtype=np.float32)
        if removal.shape != plane_ids.shape or removal.shape != depth.shape:
            raise ValueError(f"{frame_id} removal, plane-id, and depth dimensions differ")
        valid_geometry = (plane_ids > 0) & np.isfinite(depth) & (depth > 0.0)
        if np.any(removal & ~valid_geometry):
            raise ValueError(f"{frame_id} cumulative removal lacks planar geometry")
        y, x = np.nonzero(removal)
        intrinsic = geometry.frame_intrinsic(camera_info, frame_id)
        matrix = geometry.world_to_camera(camera_info, frame_id)
        rays, center = geometry.rays_for_pixels(
            x.astype(np.float64),
            y.astype(np.float64),
            intrinsic,
            matrix,
        )
        points = center[None, :] + rays * depth[y, x, None]
        per_plane: list[dict[str, Any]] = []
        inside_count = 0
        for plane_id, plane_record in plane_records.items():
            selected = plane_ids[y, x] == plane_id
            selected_count = int(selected.sum())
            if not selected_count:
                per_plane.append(
                    {"plane_id": plane_id, "removal_pixels": 0, "inside_atlas_pixels": 0}
                )
                continue
            atlas_record = atlas_records[plane_id]
            origin = np.asarray(atlas_record["origin_uv"], dtype=np.float64)
            texel_size = float(atlas_record["texel_size_world_units"])
            selected_points = points[selected]
            u = geometry.row_dot(
                selected_points,
                np.asarray(plane_record["basis_u"], dtype=np.float64),
            )
            v = geometry.row_dot(
                selected_points,
                np.asarray(plane_record["basis_v"], dtype=np.float64),
            )
            column = np.floor((u - origin[0]) / texel_size).astype(np.int64)
            row = np.floor((v - origin[1]) / texel_size).astype(np.int64)
            inside = (
                (column >= 0)
                & (column < footprints[plane_id].shape[1])
                & (row >= 0)
                & (row < footprints[plane_id].shape[0])
            )
            if not np.all(inside):
                all_inside = False
            footprints[plane_id][row[inside], column[inside]] = True
            inside_pixels = int(inside.sum())
            inside_count += inside_pixels
            per_plane.append(
                {
                    "plane_id": plane_id,
                    "removal_pixels": selected_count,
                    "inside_atlas_pixels": inside_pixels,
                }
            )
        total_pixels += int(removal.sum())
        if inside_count != int(removal.sum()):
            raise ValueError(f"{frame_id} cumulative removal escaped declared plane atlases")
        frame_records.append(
            {
                "frame_id": frame_id,
                "removal_mask": str(removal_path),
                "removal_mask_sha256": sha256_file(removal_path),
                "removal_pixels": int(removal.sum()),
                "all_removal_pixels_inside_declared_atlases": True,
                "plane_records": per_plane,
            }
        )
    if not all_inside:
        raise ValueError("some cumulative removal pixels escaped declared plane atlases")
    return footprints, {
        "source": "all frame_records[*].removal_mask pixels with declared plane geometry",
        "frame_count": len(frame_records),
        "projected_removal_pixels_with_view_duplicates": total_pixels,
        "all_removal_pixels_inside_declared_atlases": True,
        "frame_records": frame_records,
    }


def render_full_frame_from_weighted_plane_atlases(
    *,
    frame_id: str,
    camera_info: dict[str, Any],
    plane_records: dict[int, dict[str, Any]],
    completed_atlases: dict[int, dict[str, Any]],
) -> dict[str, np.ndarray]:
    intrinsic = geometry.frame_intrinsic(camera_info, frame_id)
    width = int(intrinsic["w"])
    height = int(intrinsic["h"])
    y, x = np.indices((height, width))
    flat_x = x.reshape(-1).astype(np.float64)
    flat_y = y.reshape(-1).astype(np.float64)
    matrix = geometry.world_to_camera(camera_info, frame_id)
    rays, center = geometry.rays_for_pixels(flat_x, flat_y, intrinsic, matrix)
    count = len(flat_x)
    best_depth = np.full(count, np.inf, dtype=np.float64)
    rendered = np.zeros((count, 3), dtype=np.uint8)
    weight = np.zeros(count, dtype=np.float32)
    winner_plane = np.zeros(count, dtype=np.uint16)
    winner_row = np.full(count, -1, dtype=np.int32)
    winner_column = np.full(count, -1, dtype=np.int32)
    categories = {
        "observed": np.zeros(count, dtype=bool),
        "synthetic_anchor": np.zeros(count, dtype=bool),
        "interpolated": np.zeros(count, dtype=bool),
    }
    for plane_id, plane_record in plane_records.items():
        if plane_id not in completed_atlases:
            continue
        atlas = completed_atlases[plane_id]
        plane_weight = atlas["footprint_weight"]
        if not np.any(plane_weight > 0.0):
            continue
        normal = np.asarray(plane_record["normal"], dtype=np.float64)
        denominator = geometry.row_dot(rays, normal)
        valid = np.abs(denominator) > 1e-9
        depth = np.full(count, np.inf, dtype=np.float64)
        depth[valid] = (
            float(plane_record["offset"]) - float(center @ normal)
        ) / denominator[valid]
        valid &= np.isfinite(depth) & (depth > 1e-5)
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            continue
        points = center[None, :] + rays[valid_indices] * depth[valid_indices, None]
        u = geometry.row_dot(
            points,
            np.asarray(plane_record["basis_u"], dtype=np.float64),
        )
        v = geometry.row_dot(
            points,
            np.asarray(plane_record["basis_v"], dtype=np.float64),
        )
        column = np.floor(
            (u - atlas["origin_uv"][0]) / float(atlas["texel_size"])
        ).astype(np.int64)
        row = np.floor(
            (v - atlas["origin_uv"][1]) / float(atlas["texel_size"])
        ).astype(np.int64)
        inside = (
            (column >= 0)
            & (column < plane_weight.shape[1])
            & (row >= 0)
            & (row < plane_weight.shape[0])
        )
        valid_indices = valid_indices[inside]
        row = row[inside]
        column = column[inside]
        positive = plane_weight[row, column] > 0.0
        valid_indices = valid_indices[positive]
        row = row[positive]
        column = column[positive]
        if not len(valid_indices):
            continue
        nearer = depth[valid_indices] < best_depth[valid_indices]
        target = valid_indices[nearer]
        row = row[nearer]
        column = column[nearer]
        if not len(target):
            continue
        best_depth[target] = depth[target]
        rendered[target] = atlas["color"][row, column]
        weight[target] = plane_weight[row, column]
        winner_plane[target] = plane_id
        winner_row[target] = row
        winner_column[target] = column
        for category in categories:
            categories[category][target] = atlas[category][row, column]
    return {
        "rendered_rgb": rendered.reshape(height, width, 3),
        "weight": weight.reshape(height, width),
        "plane_ids": winner_plane.reshape(height, width),
        "atlas_rows": winner_row.reshape(height, width),
        "atlas_columns": winner_column.reshape(height, width),
        **{
            f"{category}_category": value.reshape(height, width)
            for category, value in categories.items()
        },
    }


def build_texture_candidate(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.planar_report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    geometry_report = read_json(report_path)
    if not isinstance(geometry_report, dict):
        raise ValueError("planar report must be a JSON object")
    if geometry_report.get("status") != "geometry_completed_texture_pending":
        raise ValueError("planar geometry report is not ready for texture completion")
    gates = geometry_report.get("gates", {})
    if not (
        gates.get("outside_removal_mask_rgb_exact") is True
        and gates.get("texture_masks_partition_removal_mask") is True
        and gates.get("upstream_observed_plus_residual_partition_original_mask") is True
        and gates.get("all_texture_donors_exclude_cumulative_original_union") is True
        and gates.get("structural_shadow_collar_only_adds_plane_consistent_pixels")
        is True
        and gates.get("protected_neighbor_masks_subtracted_only_from_shadow_collar")
        is True
    ):
        raise ValueError("planar geometry provenance gates did not pass")
    plane_records = {
        int(record["plane_id"]): record
        for record in geometry_report["selected_envelope_planes"]
    }
    camera_info = geometry.load_camera_info(Path(geometry_report["camera_info"]).resolve())
    synthetic_anchor_input = load_synthetic_anchor_inputs(args, geometry_report)
    synthetic_anchor_atlases, synthetic_anchor_record = (
        project_synthetic_anchor_to_atlases(
            synthetic_anchor_input,
            geometry_report,
            camera_info,
            plane_records,
        )
    )
    atlas_dir = output / "atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    completed_atlases: dict[int, dict[str, Any]] = {}
    atlas_records: list[dict[str, Any]] = []
    skipped_unused_atlas_records: list[dict[str, Any]] = []
    target_pixel_counts = plane_target_pixel_counts(
        geometry_report,
        sorted(plane_records),
    )
    measured_exact = True
    synthetic_anchor_exact = True
    for atlas_record in geometry_report["texture_atlases"]:
        plane_id = int(atlas_record["plane_id"])
        plane_record = plane_records[plane_id]
        measured_path = Path(atlas_record["measured_rgb"]).resolve()
        observed_path = Path(atlas_record["observed_mask"]).resolve()
        measured = np.asarray(Image.open(measured_path).convert("RGB"), dtype=np.uint8)
        observed = np.asarray(Image.open(observed_path).convert("L"), dtype=np.uint8) > 0
        anchor_atlas = synthetic_anchor_atlases[plane_id]
        combined_color, observed, synthetic_anchor, interpolated = (
            compose_atlas_provenance(measured, observed, anchor_atlas)
        )
        combined_known = observed | synthetic_anchor
        target_pixels = target_pixel_counts.get(plane_id, 0)
        if not np.any(combined_known):
            if target_pixels == 0:
                skipped_unused_atlas_records.append(
                    {
                        "plane_id": plane_id,
                        "semantic_role": str(plane_record["semantic_role"]),
                        "reason": "unused_zero_observed_plane",
                        "target_pixels_across_selected_frames": 0,
                        "measured_rgb": str(measured_path),
                        "measured_rgb_sha256": sha256_file(measured_path),
                        "observed_mask": str(observed_path),
                        "observed_mask_sha256": sha256_file(observed_path),
                        "observed_texels": 0,
                        "synthetic_anchor_texels": 0,
                        "interpolated_texels": 0,
                        "rendering_required": False,
                    }
                )
                continue
            raise ValueError(
                f"plane {plane_id} has target pixels but no measured or synthetic texels"
            )
        role = str(plane_record["semantic_role"])
        method_details: dict[str, Any]
        if role == "floor":
            completed, method_details = directional_floor_extend(
                combined_color,
                combined_known,
                fallback_iterations=args.floor_smoothing_iterations,
            )
        else:
            propagation_support, support_details = robust_low_frequency_support(
                combined_color,
                combined_known,
                quantization=args.wall_color_quantization,
                color_radius=args.wall_color_radius,
                minimum_fraction=args.minimum_wall_support_fraction,
                luminance_quantile=args.wall_luminance_quantile,
            )
            illumination_support, inset_details = inset_support_from_synthetic_boundary(
                propagation_support,
                combined_known,
                args.wall_support_boundary_inset,
                minimum_fraction=args.minimum_wall_inset_support_fraction,
            )
            _, illumination_model, illumination_details = (
                normalized_plane_illumination_field(
                    combined_color,
                    combined_known,
                    illumination_support,
                    sigma=args.wall_illumination_sigma,
                    blend_weight_scale=args.wall_kernel_blend_weight_scale,
                    color_clip_margin=args.wall_illumination_color_clip_margin,
                    fallback_degree=args.wall_illumination_degree,
                    fallback_iterations=args.wall_illumination_irls_iterations,
                    huber_delta=args.wall_illumination_huber_delta,
                    ridge=args.wall_illumination_ridge,
                )
            )
            completed, collar_details = apply_reliable_boundary_residual_collar(
                illumination_model,
                combined_color,
                combined_known,
                propagation_support,
                width=args.wall_residual_collar_width,
            )
            method_details = {
                **illumination_details,
                "propagation_support": support_details,
                "object_boundary_exclusion": inset_details,
                "object_boundary_colors_used_for_synthetic_fit": False,
                "reliable_boundary_residual_collar": collar_details,
            }
        measured_exact &= bool(np.array_equal(completed[observed], measured[observed]))
        synthetic_anchor_exact &= bool(
            np.array_equal(
                completed[synthetic_anchor],
                anchor_atlas["color"][synthetic_anchor],
            )
        )
        completed_path = atlas_dir / f"plane_{plane_id:02d}_completed_rgb.png"
        synthetic_anchor_path = atlas_dir / f"plane_{plane_id:02d}_synthetic_anchor_mask.png"
        interpolated_path = atlas_dir / f"plane_{plane_id:02d}_interpolated_mask.png"
        Image.fromarray(completed).save(completed_path)
        Image.fromarray(synthetic_anchor.astype(np.uint8) * 255).save(
            synthetic_anchor_path
        )
        Image.fromarray(interpolated.astype(np.uint8) * 255).save(interpolated_path)
        record = {
            "plane_id": plane_id,
            "semantic_role": role,
            "measured_rgb": str(measured_path),
            "measured_rgb_sha256": sha256_file(measured_path),
            "observed_mask": str(observed_path),
            "observed_mask_sha256": sha256_file(observed_path),
            "completed_rgb": str(completed_path),
            "completed_rgb_sha256": sha256_file(completed_path),
            "synthetic_anchor_mask": str(synthetic_anchor_path),
            "synthetic_anchor_mask_sha256": sha256_file(synthetic_anchor_path),
            "interpolated_mask": str(interpolated_path),
            "interpolated_mask_sha256": sha256_file(interpolated_path),
            "origin_uv": atlas_record["origin_uv"],
            "texel_size_world_units": atlas_record["texel_size_world_units"],
            "dimensions": atlas_record["dimensions"],
            "observed_texels": int(observed.sum()),
            "synthetic_anchor_texels": int(synthetic_anchor.sum()),
            "interpolated_texels": int(interpolated.sum()),
            "target_pixels_across_selected_frames": target_pixels,
            "observed_fraction": float(observed.mean()),
            "synthetic_anchor_fraction": float(synthetic_anchor.mean()),
            "interpolated_fraction": float(interpolated.mean()),
            "method": method_details["method"],
            "method_details": method_details,
            "measured_texels_rgb_exact": bool(
                np.array_equal(completed[observed], measured[observed])
            ),
            "synthetic_anchor_texels_rgb_exact": bool(
                np.array_equal(
                    completed[synthetic_anchor],
                    anchor_atlas["color"][synthetic_anchor],
                )
            ),
            "observed_interpolated_boundary": boundary_discontinuity(
                completed,
                combined_known,
            ),
            "boundary_normal_continuity": boundary_normal_gradient_continuity(
                completed,
                combined_known,
            ),
        }
        atlas_records.append(record)
        completed_atlases[plane_id] = {
            "color": completed,
            "observed": observed,
            "synthetic_anchor": synthetic_anchor,
            "interpolated": interpolated,
            "origin_uv": np.asarray(atlas_record["origin_uv"], dtype=np.float64),
            "texel_size": float(atlas_record["texel_size_world_units"]),
        }
    footprint_neutralization: dict[str, Any] | None = None
    if args.plane_footprint_neutralization:
        footprints, projection_record = project_cumulative_removal_footprints(
            geometry_report,
            camera_info,
            plane_records,
        )
        footprint_plane_records: list[dict[str, Any]] = []
        atlas_records_by_id = {int(record["plane_id"]): record for record in atlas_records}
        eligible_footprints, skipped_footprint_plane_records = (
            filter_footprints_for_completed_atlases(
                footprints,
                completed_atlases,
                atlas_records_by_id,
            )
        )
        for plane_id, footprint in eligible_footprints.items():
            weight, details = rectangular_footprint_feather_weight(
                footprint,
                padding=args.plane_footprint_padding_texels,
                feather_width=args.screen_space_feather_width_texels,
            )
            core = weight >= 1.0 - 1e-7
            feather = (weight > 0.0) & ~core
            footprint_path = atlas_dir / f"plane_{plane_id:02d}_removal_footprint.png"
            core_path = atlas_dir / f"plane_{plane_id:02d}_rectangle_core.png"
            feather_path = atlas_dir / f"plane_{plane_id:02d}_rectangle_feather.png"
            weight_path = atlas_dir / f"plane_{plane_id:02d}_rectangle_weight_u16.png"
            Image.fromarray(footprint.astype(np.uint8) * 255).save(footprint_path)
            Image.fromarray(core.astype(np.uint8) * 255).save(core_path)
            Image.fromarray(feather.astype(np.uint8) * 255).save(feather_path)
            Image.fromarray(np.rint(weight * 65535.0).astype(np.uint16)).save(weight_path)
            completed_atlases[plane_id]["footprint_weight"] = weight
            record = {
                "plane_id": plane_id,
                **details,
                "footprint_mask": str(footprint_path),
                "footprint_mask_sha256": sha256_file(footprint_path),
                "rectangle_core_mask": str(core_path),
                "rectangle_core_mask_sha256": sha256_file(core_path),
                "rectangle_feather_mask": str(feather_path),
                "rectangle_feather_mask_sha256": sha256_file(feather_path),
                "rectangle_weight_u16": str(weight_path),
                "rectangle_weight_u16_sha256": sha256_file(weight_path),
            }
            footprint_plane_records.append(record)
            atlas_records_by_id[plane_id]["footprint_neutralization"] = record
        footprint_neutralization = {
            "enabled": True,
            "boundary_shape": "axis_aligned_plane_uv_rectangle",
            "padding_texels": args.plane_footprint_padding_texels,
            "feather_width_texels": args.screen_space_feather_width_texels,
            "padding_covers_feather": (
                args.plane_footprint_padding_texels
                >= args.screen_space_feather_width_texels
            ),
            "projection": projection_record,
            "plane_records": footprint_plane_records,
            "skipped_plane_records": skipped_footprint_plane_records,
            "all_original_removal_plane_texels_in_weight_one_core": all(
                record["all_footprint_texels_in_weight_one_core"]
                for record in footprint_plane_records
            ),
        }
    frames_dir = output / "frames"
    masks_dir = output / "synthetic_masks"
    observed_masks_dir = output / "observed_atlas_masks"
    synthetic_anchor_masks_dir = output / "synthetic_anchor_masks"
    interpolated_masks_dir = output / "interpolated_masks"
    core_masks_dir = output / "core_shared_atlas_masks"
    feather_masks_dir = output / "screen_space_feather_masks"
    outside_masks_dir = output / "outside_source_masks"
    footprint_labels_dir = output / "footprint_provenance_labels"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    observed_masks_dir.mkdir(parents=True, exist_ok=True)
    synthetic_anchor_masks_dir.mkdir(parents=True, exist_ok=True)
    interpolated_masks_dir.mkdir(parents=True, exist_ok=True)
    core_masks_dir.mkdir(parents=True, exist_ok=True)
    feather_masks_dir.mkdir(parents=True, exist_ok=True)
    outside_masks_dir.mkdir(parents=True, exist_ok=True)
    footprint_labels_dir.mkdir(parents=True, exist_ok=True)
    consistency_support = {
        plane_id: np.zeros(completed_atlases[plane_id]["color"].shape[:2], dtype=np.uint16)
        for plane_id in completed_atlases
    }
    frame_records: list[dict[str, Any]] = []
    outside_exact = True
    all_synthetic_assigned = True
    for output_index, record in enumerate(geometry_report["frame_records"]):
        frame_id = str(record["frame_id"])
        source_path = Path(record["prefill_frame"]).resolve()
        labels_path = Path(record["plane_labels"]).resolve()
        plane_ids_path = Path(record["plane_ids"]).resolve()
        depth_path = Path(record["geometry_depth"]).resolve()
        synthetic_path = Path(record["generative_texture_mask"]).resolve()
        source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8)
        plane_ids = np.asarray(Image.open(plane_ids_path), dtype=np.uint16)
        synthetic = np.asarray(Image.open(synthetic_path).convert("L"), dtype=np.uint8) > 0
        with np.load(depth_path) as depth_archive:
            depth = np.asarray(depth_archive["depth"], dtype=np.float32)
        if args.plane_footprint_neutralization:
            original_removal = load_binary_mask(
                Path(record["removal_mask"]).resolve(),
                f"{frame_id} cumulative removal mask",
            )
            protected_neighbor = np.zeros_like(original_removal)
            protected_records = record.get("cumulative_mask_receipt", {}).get(
                "protected_anchor_masks",
                [],
            )
            for protected_record in protected_records:
                protected_path = Path(protected_record["path"]).resolve()
                protected_neighbor |= load_binary_mask(
                    protected_path,
                    f"{frame_id} protected neighbor mask",
                )
            plane_projection = render_full_frame_from_weighted_plane_atlases(
                frame_id=frame_id,
                camera_info=camera_info,
                plane_records=plane_records,
                completed_atlases=completed_atlases,
            )
            if np.any(original_removal & (plane_projection["weight"] < 1.0 - 1e-6)):
                raise ValueError(
                    f"{frame_id} original removal escaped plane-space weight-one core"
                )
            if not np.array_equal(
                plane_projection["plane_ids"][original_removal],
                plane_ids[original_removal],
            ):
                raise ValueError(
                    f"{frame_id} footprint projection changed the original removal plane"
                )
            result, screen_provenance, screen_details = composite_plane_footprint_render(
                source,
                plane_projection["rendered_rgb"],
                plane_projection["weight"],
                original_removal,
                protected_neighbor,
            )
            core_image = screen_provenance["core_shared_atlas"]
            feather_image = screen_provenance["screen_space_feather"]
            outside_image = screen_provenance["outside_source"]
            synthetic = core_image | feather_image
            observed_image = plane_projection["observed_category"] & synthetic
            anchor_image = plane_projection["synthetic_anchor_category"] & synthetic
            interpolated_image = plane_projection["interpolated_category"] & synthetic
            category_partition_exact = bool(
                np.array_equal(
                    observed_image | anchor_image | interpolated_image,
                    synthetic,
                )
                and not np.any(observed_image & anchor_image)
                and not np.any(observed_image & interpolated_image)
                and not np.any(anchor_image & interpolated_image)
            )
            if not category_partition_exact:
                raise ValueError(f"{frame_id} atlas source categories do not partition footprint")
            for plane_id, atlas in completed_atlases.items():
                selected_core = core_image & (plane_projection["plane_ids"] == plane_id)
                rows = plane_projection["atlas_rows"][selected_core]
                columns = plane_projection["atlas_columns"][selected_core]
                if len(rows):
                    unique = np.unique(rows * atlas["color"].shape[1] + columns)
                    consistency_support[plane_id][
                        unique // atlas["color"].shape[1],
                        unique % atlas["color"].shape[1],
                    ] += 1
            outside_frame_exact = screen_details["outside_outer_mask_rgb_exact"]
            protected_preserved = protected_neighbor & ~original_removal
            protected_exact = screen_details["protected_neighbor_pixels_rgb_exact"]
            frame_boundary_continuity = boundary_normal_gradient_continuity(
                result,
                outside_image,
            )
            outside_exact &= outside_frame_exact
            all_synthetic_assigned &= True
            output_path = frames_dir / f"{output_index:04d}.png"
            output_mask_path = masks_dir / f"{output_index:04d}.png"
            output_observed_mask_path = observed_masks_dir / f"{output_index:04d}.png"
            output_anchor_mask_path = synthetic_anchor_masks_dir / f"{output_index:04d}.png"
            output_interpolated_mask_path = (
                interpolated_masks_dir / f"{output_index:04d}.png"
            )
            output_core_mask_path = core_masks_dir / f"{output_index:04d}.png"
            output_feather_mask_path = feather_masks_dir / f"{output_index:04d}.png"
            output_outside_mask_path = outside_masks_dir / f"{output_index:04d}.png"
            output_footprint_labels_path = footprint_labels_dir / f"{output_index:04d}.png"
            footprint_labels = np.zeros_like(original_removal, dtype=np.uint8)
            footprint_labels[core_image] = 1
            footprint_labels[feather_image] = 2
            Image.fromarray(result).save(output_path)
            Image.fromarray(synthetic.astype(np.uint8) * 255).save(output_mask_path)
            Image.fromarray(observed_image.astype(np.uint8) * 255).save(
                output_observed_mask_path
            )
            Image.fromarray(anchor_image.astype(np.uint8) * 255).save(
                output_anchor_mask_path
            )
            Image.fromarray(interpolated_image.astype(np.uint8) * 255).save(
                output_interpolated_mask_path
            )
            Image.fromarray(core_image.astype(np.uint8) * 255).save(output_core_mask_path)
            Image.fromarray(feather_image.astype(np.uint8) * 255).save(
                output_feather_mask_path
            )
            Image.fromarray(outside_image.astype(np.uint8) * 255).save(
                output_outside_mask_path
            )
            Image.fromarray(footprint_labels).save(output_footprint_labels_path)
            frame_records.append(
                {
                    "sequence_index": output_index,
                    "frame_id": frame_id,
                    "source_frame": str(source_path),
                    "source_frame_sha256": sha256_file(source_path),
                    "plane_labels": str(labels_path),
                    "completed_frame": str(output_path),
                    "completed_frame_sha256": sha256_file(output_path),
                    "synthetic_mask": str(output_mask_path),
                    "synthetic_mask_sha256": sha256_file(output_mask_path),
                    "observed_atlas_mask": str(output_observed_mask_path),
                    "observed_atlas_mask_sha256": sha256_file(output_observed_mask_path),
                    "synthetic_anchor_mask": str(output_anchor_mask_path),
                    "synthetic_anchor_mask_sha256": sha256_file(output_anchor_mask_path),
                    "interpolated_mask": str(output_interpolated_mask_path),
                    "interpolated_mask_sha256": sha256_file(output_interpolated_mask_path),
                    "core_shared_atlas_mask": str(output_core_mask_path),
                    "core_shared_atlas_mask_sha256": sha256_file(output_core_mask_path),
                    "screen_space_feather_mask": str(output_feather_mask_path),
                    "screen_space_feather_mask_sha256": sha256_file(
                        output_feather_mask_path
                    ),
                    "outside_source_mask": str(output_outside_mask_path),
                    "outside_source_mask_sha256": sha256_file(output_outside_mask_path),
                    "footprint_provenance_labels": str(output_footprint_labels_path),
                    "footprint_provenance_labels_sha256": sha256_file(
                        output_footprint_labels_path
                    ),
                    "synthetic_pixels": int(synthetic.sum()),
                    "observed_atlas_pixels": int(observed_image.sum()),
                    "synthetic_anchor_pixels": int(anchor_image.sum()),
                    "interpolated_pixels": int(interpolated_image.sum()),
                    "core_shared_atlas_pixels": int(core_image.sum()),
                    "screen_space_feather_pixels": int(feather_image.sum()),
                    "outside_source_pixels": int(outside_image.sum()),
                    "original_cumulative_removal_pixels": int(original_removal.sum()),
                    "all_original_removal_pixels_in_core": screen_details[
                        "all_original_removal_pixels_in_core"
                    ],
                    "synthetic_provenance_partition_exact": category_partition_exact,
                    "screen_provenance_partition_exact": bool(
                        np.all(
                            core_image.astype(np.uint8)
                            + feather_image.astype(np.uint8)
                            + outside_image.astype(np.uint8)
                            == 1
                        )
                    ),
                    "synthetic_pixels_assigned_from_shared_atlas": int(synthetic.sum()),
                    "all_synthetic_pixels_assigned": True,
                    "outside_synthetic_mask_rgb_exact": outside_frame_exact,
                    "outside_outer_mask_rgb_exact": outside_frame_exact,
                    "protected_neighbor_pixels_outside_synthetic_mask": int(
                        protected_preserved.sum()
                    ),
                    "protected_neighbor_pixels_rgb_exact": protected_exact,
                    "protected_neighbor_overlap_with_original_removal_not_overridden": (
                        screen_details[
                            "protected_neighbor_overlap_with_original_removal_not_overridden"
                        ]
                    ),
                    "core_claims_measured_donor": False,
                    "feather_claims_measured_donor": False,
                    "synthetic_boundary_continuity_full_resolution": (
                        frame_boundary_continuity
                    ),
                }
            )
            continue
        result = source.copy()
        y, x = np.nonzero(synthetic)
        intrinsic = geometry.frame_intrinsic(camera_info, frame_id)
        matrix = geometry.world_to_camera(camera_info, frame_id)
        rays, center = geometry.rays_for_pixels(
            x.astype(np.float64),
            y.astype(np.float64),
            intrinsic,
            matrix,
        )
        points = center[None, :] + rays * depth[y, x, None]
        assigned = np.zeros(len(x), dtype=bool)
        observed_assigned = np.zeros(len(x), dtype=bool)
        anchor_assigned = np.zeros(len(x), dtype=bool)
        interpolated_assigned = np.zeros(len(x), dtype=bool)
        for plane_id, plane_record in plane_records.items():
            selected = plane_ids[y, x] == plane_id
            if not np.any(selected):
                continue
            atlas = completed_atlases[plane_id]
            basis_u = np.asarray(plane_record["basis_u"], dtype=np.float64)
            basis_v = np.asarray(plane_record["basis_v"], dtype=np.float64)
            u = geometry.row_dot(points[selected], basis_u)
            v = geometry.row_dot(points[selected], basis_v)
            column = np.floor(
                (u - atlas["origin_uv"][0]) / float(atlas["texel_size"])
            ).astype(np.int64)
            row = np.floor(
                (v - atlas["origin_uv"][1]) / float(atlas["texel_size"])
            ).astype(np.int64)
            color = atlas["color"]
            inside = (
                (column >= 0)
                & (column < color.shape[1])
                & (row >= 0)
                & (row < color.shape[0])
            )
            selected_indices = np.flatnonzero(selected)
            valid_indices = selected_indices[inside]
            result[y[valid_indices], x[valid_indices]] = color[row[inside], column[inside]]
            assigned[valid_indices] = True
            observed_assigned[valid_indices] = atlas["observed"][
                row[inside],
                column[inside],
            ]
            anchor_assigned[valid_indices] = atlas["synthetic_anchor"][
                row[inside],
                column[inside],
            ]
            interpolated_assigned[valid_indices] = atlas["interpolated"][
                row[inside],
                column[inside],
            ]
            unique = np.unique(row[inside] * color.shape[1] + column[inside])
            consistency_support[plane_id][
                unique // color.shape[1],
                unique % color.shape[1],
            ] += 1
        outside_frame_exact = bool(np.array_equal(result[~synthetic], source[~synthetic]))
        protected_neighbor = np.zeros_like(synthetic)
        protected_records = record.get("cumulative_mask_receipt", {}).get(
            "protected_anchor_masks",
            [],
        )
        for protected_record in protected_records:
            protected_path = Path(protected_record["path"]).resolve()
            protected_neighbor |= (
                np.asarray(Image.open(protected_path).convert("L"), dtype=np.uint8) > 0
            )
        protected_preserved = protected_neighbor & ~synthetic
        protected_exact = bool(
            np.array_equal(result[protected_preserved], source[protected_preserved])
        )
        frame_boundary_continuity = boundary_normal_gradient_continuity(
            result,
            ~synthetic,
        )
        outside_exact &= outside_frame_exact
        all_synthetic_assigned &= bool(np.all(assigned))
        output_path = frames_dir / f"{output_index:04d}.png"
        output_mask_path = masks_dir / f"{output_index:04d}.png"
        output_observed_mask_path = observed_masks_dir / f"{output_index:04d}.png"
        output_anchor_mask_path = synthetic_anchor_masks_dir / f"{output_index:04d}.png"
        output_interpolated_mask_path = (
            interpolated_masks_dir / f"{output_index:04d}.png"
        )
        observed_image = np.zeros_like(synthetic)
        anchor_image = np.zeros_like(synthetic)
        interpolated_image = np.zeros_like(synthetic)
        observed_image[y, x] = observed_assigned
        anchor_image[y, x] = anchor_assigned
        interpolated_image[y, x] = interpolated_assigned
        category_partition_exact = bool(
            np.array_equal(
                observed_image | anchor_image | interpolated_image,
                synthetic,
            )
            and not np.any(observed_image & anchor_image)
            and not np.any(observed_image & interpolated_image)
            and not np.any(anchor_image & interpolated_image)
        )
        Image.fromarray(result).save(output_path)
        Image.fromarray(synthetic.astype(np.uint8) * 255).save(output_mask_path)
        Image.fromarray(observed_image.astype(np.uint8) * 255).save(
            output_observed_mask_path
        )
        Image.fromarray(anchor_image.astype(np.uint8) * 255).save(output_anchor_mask_path)
        Image.fromarray(interpolated_image.astype(np.uint8) * 255).save(
            output_interpolated_mask_path
        )
        frame_records.append(
            {
                "sequence_index": output_index,
                "frame_id": frame_id,
                "source_frame": str(source_path),
                "source_frame_sha256": sha256_file(source_path),
                "plane_labels": str(labels_path),
                "completed_frame": str(output_path),
                "completed_frame_sha256": sha256_file(output_path),
                "synthetic_mask": str(output_mask_path),
                "synthetic_mask_sha256": sha256_file(output_mask_path),
                "observed_atlas_mask": str(output_observed_mask_path),
                "observed_atlas_mask_sha256": sha256_file(output_observed_mask_path),
                "synthetic_anchor_mask": str(output_anchor_mask_path),
                "synthetic_anchor_mask_sha256": sha256_file(output_anchor_mask_path),
                "interpolated_mask": str(output_interpolated_mask_path),
                "interpolated_mask_sha256": sha256_file(output_interpolated_mask_path),
                "synthetic_pixels": int(synthetic.sum()),
                "observed_atlas_pixels": int(observed_image.sum()),
                "synthetic_anchor_pixels": int(anchor_image.sum()),
                "interpolated_pixels": int(interpolated_image.sum()),
                "synthetic_provenance_partition_exact": category_partition_exact,
                "synthetic_pixels_assigned_from_shared_atlas": int(assigned.sum()),
                "all_synthetic_pixels_assigned": bool(np.all(assigned)),
                "outside_synthetic_mask_rgb_exact": outside_frame_exact,
                "protected_neighbor_pixels_outside_synthetic_mask": int(
                    protected_preserved.sum()
                ),
                "protected_neighbor_pixels_rgb_exact": protected_exact,
                "synthetic_boundary_continuity_full_resolution": (
                    frame_boundary_continuity
                ),
            }
        )
    consistency_records = []
    for plane_id, support in consistency_support.items():
        consistency_records.append(
            {
                "plane_id": plane_id,
                "texels_sampled_by_any_view": int((support > 0).sum()),
                "texels_sampled_by_multiple_views": int((support >= 2).sum()),
                "maximum_view_count": int(support.max(initial=0)),
                "maximum_rgb_delta_for_same_texel_across_views": 0,
                "contract": "all views perform exact lookup from one immutable completed atlas",
            }
        )
    requested_full_resolution_ids = [
        Path(item.strip()).stem
        for item in args.full_resolution_frame_ids.split(",")
        if item.strip()
    ]
    frame_records_by_id = {record["frame_id"]: record for record in frame_records}
    unknown_full_resolution_ids = sorted(
        set(requested_full_resolution_ids) - set(frame_records_by_id)
    )
    if unknown_full_resolution_ids:
        raise ValueError(
            "full-resolution review frames are absent: "
            f"{unknown_full_resolution_ids}"
        )
    full_resolution_review = [
        {
            "frame_id": frame_id,
            "completed_frame": frame_records_by_id[frame_id]["completed_frame"],
            "completed_frame_sha256": frame_records_by_id[frame_id][
                "completed_frame_sha256"
            ],
            "synthetic_mask": frame_records_by_id[frame_id]["synthetic_mask"],
            "synthetic_mask_sha256": frame_records_by_id[frame_id][
                "synthetic_mask_sha256"
            ],
            "observed_atlas_mask": frame_records_by_id[frame_id][
                "observed_atlas_mask"
            ],
            "observed_atlas_mask_sha256": frame_records_by_id[frame_id][
                "observed_atlas_mask_sha256"
            ],
            "synthetic_anchor_mask": frame_records_by_id[frame_id][
                "synthetic_anchor_mask"
            ],
            "synthetic_anchor_mask_sha256": frame_records_by_id[frame_id][
                "synthetic_anchor_mask_sha256"
            ],
            "interpolated_mask": frame_records_by_id[frame_id]["interpolated_mask"],
            "interpolated_mask_sha256": frame_records_by_id[frame_id][
                "interpolated_mask_sha256"
            ],
            "core_shared_atlas_mask": frame_records_by_id[frame_id].get(
                "core_shared_atlas_mask"
            ),
            "core_shared_atlas_mask_sha256": frame_records_by_id[frame_id].get(
                "core_shared_atlas_mask_sha256"
            ),
            "screen_space_feather_mask": frame_records_by_id[frame_id].get(
                "screen_space_feather_mask"
            ),
            "screen_space_feather_mask_sha256": frame_records_by_id[frame_id].get(
                "screen_space_feather_mask_sha256"
            ),
            "outside_source_mask": frame_records_by_id[frame_id].get("outside_source_mask"),
            "outside_source_mask_sha256": frame_records_by_id[frame_id].get(
                "outside_source_mask_sha256"
            ),
            "dimensions": [
                int(geometry.frame_intrinsic(camera_info, frame_id)["w"]),
                int(geometry.frame_intrinsic(camera_info, frame_id)["h"]),
            ],
            "synthetic_boundary_continuity_full_resolution": frame_records_by_id[
                frame_id
            ]["synthetic_boundary_continuity_full_resolution"],
            "protected_neighbor_pixels_rgb_exact": frame_records_by_id[frame_id][
                "protected_neighbor_pixels_rgb_exact"
            ],
        }
        for frame_id in requested_full_resolution_ids
    ]
    wall_records = [
        record
        for record in atlas_records
        if record["semantic_role"] in {"wall", "ceiling"}
        and record["boundary_normal_continuity"]["boundary_edge_count"] > 0
    ]
    wall_boundary_color_p95 = max(
        (
            float(
                record["boundary_normal_continuity"][
                    "boundary_color_p95_abs_rgb_delta"
                ]
            )
            for record in wall_records
        ),
        default=0.0,
    )
    wall_boundary_gradient_p95 = max(
        (
            float(
                record["boundary_normal_continuity"][
                    "boundary_normal_gradient_p95_abs_rgb_delta"
                ]
            )
            for record in wall_records
        ),
        default=0.0,
    )
    wall_boundary_color_passed = (
        wall_boundary_color_p95 <= args.maximum_wall_boundary_p95_delta
    )
    wall_boundary_gradient_passed = (
        wall_boundary_gradient_p95 <= args.maximum_wall_boundary_gradient_p95_delta
    )
    full_resolution_boundary_color_p95 = max(
        float(
            record["synthetic_boundary_continuity_full_resolution"][
                "boundary_color_p95_abs_rgb_delta"
            ]
        )
        for record in full_resolution_review
    )
    full_resolution_boundary_gradient_p95 = max(
        float(
            record["synthetic_boundary_continuity_full_resolution"][
                "boundary_normal_gradient_p95_abs_rgb_delta"
            ]
        )
        for record in full_resolution_review
    )
    protected_neighbors_exact = all(
        record["protected_neighbor_pixels_rgb_exact"] for record in frame_records
    )
    synthetic_provenance_partitions_exact = all(
        record["synthetic_provenance_partition_exact"] for record in frame_records
    )
    atlas_observed_texels = sum(record["observed_texels"] for record in atlas_records)
    atlas_anchor_texels = sum(record["synthetic_anchor_texels"] for record in atlas_records)
    atlas_interpolated_texels = sum(record["interpolated_texels"] for record in atlas_records)
    frame_observed_pixels = sum(record["observed_atlas_pixels"] for record in frame_records)
    frame_anchor_pixels = sum(record["synthetic_anchor_pixels"] for record in frame_records)
    frame_interpolated_pixels = sum(record["interpolated_pixels"] for record in frame_records)
    frame_core_pixels = sum(record.get("core_shared_atlas_pixels", 0) for record in frame_records)
    frame_feather_pixels = sum(
        record.get("screen_space_feather_pixels", 0) for record in frame_records
    )
    frame_outside_pixels = sum(record.get("outside_source_pixels", 0) for record in frame_records)
    screen_provenance_partitions_exact = all(
        record.get("screen_provenance_partition_exact", True) for record in frame_records
    )
    all_original_removal_pixels_in_core = all(
        record.get("all_original_removal_pixels_in_core", True) for record in frame_records
    )
    outside_outer_exact = all(
        record.get("outside_outer_mask_rgb_exact", True) for record in frame_records
    )
    rejected = excluded_evidence(args.rejected_texture_evidence or [])
    rejected_safe = all(record.get("promotion_allowed") is not True for record in rejected)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "purpose": (
            "plane-space rectangular footprint neutralization with screen-space feather"
            if footprint_neutralization is not None
            else "single-plane-UV synthetic texture candidate after measured donor prefill"
        ),
        "status": "texture_candidate_review_pending",
        "promotion_approved": False,
        "eligible_as_round04_clean_plate": False,
        "promotion_blocker": (
            "visual quality and wall-floor boundary review are pending; synthetic provenance must "
            "remain explicit even if the candidate is later accepted"
        ),
        "planar_geometry_report": str(report_path),
        "planar_geometry_report_sha256": sha256_file(report_path),
        "planar_geometry_status": geometry_report["status"],
        "texture_coverage": geometry_report["texture_coverage"],
        "provenance": {
            "observed_fraction_of_original_removal_masks": geometry_report["texture_coverage"][
                "observed_fraction"
            ],
            "synthetic_fraction_of_original_removal_masks": geometry_report[
                "texture_coverage"
            ]["synthetic_pending_fraction"],
            "synthetic_pixels_are_measured": False,
            "one_atlas_per_plane": True,
            "per_frame_independent_generation_used": False,
            "single_anchor_generation_used": synthetic_anchor_record is not None,
            "claims_measured_donor": False,
            "texture_provenance": "hybrid_atlas",
            "atlas_texels": {
                "observed": atlas_observed_texels,
                "synthetic_anchor": atlas_anchor_texels,
                "interpolated": atlas_interpolated_texels,
            },
            "rendered_synthetic_pixels": {
                "observed": frame_observed_pixels,
                "synthetic_anchor": frame_anchor_pixels,
                "interpolated": frame_interpolated_pixels,
            },
            "screen_composite_pixels": {
                "core_shared_atlas": frame_core_pixels,
                "screen_space_feather": frame_feather_pixels,
                "outside_source": frame_outside_pixels,
            },
            "screen_space_feather_is_synthetic": True,
            "screen_space_feather_claims_measured_donor": False,
        },
        "synthetic_anchor_input": synthetic_anchor_record,
        "plane_footprint_neutralization": footprint_neutralization,
        "excluded_texture_inputs": rejected,
        "notes": [
            (
                "Rejected per-frame SDXL candidates are lineage evidence only and were not used "
                "as measured or synthetic atlas inputs."
            ),
            "Old DA3 depth remains visibility evidence only, not final regenerated scene depth.",
        ],
        "atlas_records": atlas_records,
        "skipped_unused_atlas_records": skipped_unused_atlas_records,
        "cross_view_atlas_consistency": consistency_records,
        "frame_records": frame_records,
        "full_resolution_review": full_resolution_review,
        "wall_boundary_continuity_summary": {
            "wall_or_ceiling_planes_evaluated": len(wall_records),
            "maximum_boundary_color_p95_abs_rgb_delta": wall_boundary_color_p95,
            "maximum_boundary_normal_gradient_p95_abs_rgb_delta": (
                wall_boundary_gradient_p95
            ),
        },
        "full_resolution_boundary_continuity_summary": {
            "frame_ids": requested_full_resolution_ids,
            "maximum_boundary_color_p95_abs_rgb_delta": (
                full_resolution_boundary_color_p95
            ),
            "maximum_boundary_normal_gradient_p95_abs_rgb_delta": (
                full_resolution_boundary_gradient_p95
            ),
        },
        "gates": {
            "measured_atlas_texels_rgb_exact": measured_exact,
            "synthetic_anchor_atlas_texels_rgb_exact": synthetic_anchor_exact,
            "outside_synthetic_mask_rgb_exact": outside_exact,
            "outside_outer_mask_rgb_exact": outside_outer_exact,
            "protected_neighbors_outside_synthetic_mask_rgb_exact": (
                protected_neighbors_exact
            ),
            "protected_sam_neighbor_pixels_rgb_exact": protected_neighbors_exact,
            "all_synthetic_pixels_assigned_from_shared_atlas": all_synthetic_assigned,
            "same_atlas_texel_has_identical_rgb_across_views": True,
            "core_same_atlas_texel_has_identical_rgb_across_views": True,
            "core_screen_feather_and_outside_partition_full_frame": (
                screen_provenance_partitions_exact
            ),
            "all_original_cumulative_removal_pixels_in_weight_one_core": (
                all_original_removal_pixels_in_core
            ),
            "plane_boundary_is_not_original_object_contour": (
                footprint_neutralization is None
                or all(
                    record["boundary_shape"] == "axis_aligned_plane_uv_rectangle"
                    for record in footprint_neutralization["plane_records"]
                )
            ),
            "plane_footprint_padding_covers_feather": (
                footprint_neutralization is None
                or footprint_neutralization["padding_covers_feather"]
            ),
            "core_claims_measured_donor": False,
            "screen_space_feather_claims_measured_donor": False,
            "synthetic_anchor_claims_measured_donor": False,
            "observed_synthetic_anchor_and_interpolated_partition_rendered_synthetic_pixels": (
                synthetic_provenance_partitions_exact
            ),
            "rejected_texture_inputs_not_promotable": rejected_safe,
            "wall_object_boundary_colors_excluded_from_synthetic_fit": all(
                record["method_details"].get(
                    "object_boundary_colors_used_for_synthetic_fit"
                )
                is False
                for record in wall_records
            ),
            "wall_boundary_color_continuity": {
                "threshold_p95_abs_rgb_delta": args.maximum_wall_boundary_p95_delta,
                "observed_maximum": wall_boundary_color_p95,
                "passed": wall_boundary_color_passed,
            },
            "wall_boundary_low_frequency_gradient_continuity": {
                "threshold_p95_abs_rgb_delta": (
                    args.maximum_wall_boundary_gradient_p95_delta
                ),
                "observed_maximum": wall_boundary_gradient_p95,
                "passed": wall_boundary_gradient_passed,
            },
            "visual_quality": "pending_human_or_vlm_review",
            "new_depth_normal_estimation_before_pgsr": "required_after_visual_acceptance",
        },
        "limitations": [
            (
                "Polynomial wall illumination is low frequency and cannot invent real hidden "
                "wall detail."
            ),
            "Directional floor extension preserves orientation but can repeat strips or seams.",
            (
                "Cross-view consistency does not by itself certify photorealism or semantic "
                "correctness."
            ),
        ],
    }
    report_path_output = output / "planar_texture_report.json"
    write_json(report_path_output, report)
    make_atlas_contact_sheet(atlas_records, output / "atlas_contact_sheet.png")
    make_view_contact_sheet(
        frame_records,
        output / "planar_texture_view_contact_sheet.png",
        args.contact_sheet_samples,
    )
    if footprint_neutralization is not None:
        make_footprint_provenance_contact_sheet(
            frame_records,
            output / "plane_footprint_provenance_contact_sheet.png",
            args.contact_sheet_samples,
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planar-report", type=Path, required=True)
    parser.add_argument("--rejected-texture-evidence", type=Path, action="append")
    parser.add_argument("--synthetic-anchor-frame-id")
    parser.add_argument("--synthetic-anchor-candidate", type=Path)
    parser.add_argument("--synthetic-anchor-candidate-sha256")
    parser.add_argument("--synthetic-anchor-edit-mask", type=Path)
    parser.add_argument("--synthetic-anchor-edit-mask-sha256")
    parser.add_argument("--synthetic-anchor-diffusion-receipt", type=Path)
    parser.add_argument("--synthetic-anchor-diffusion-receipt-sha256")
    parser.add_argument("--synthetic-anchor-source-guide", type=Path)
    parser.add_argument("--synthetic-anchor-source-guide-sha256")
    parser.add_argument("--synthetic-anchor-source-guide-report", type=Path)
    parser.add_argument("--synthetic-anchor-source-guide-report-sha256")
    parser.add_argument("--plane-footprint-neutralization", action="store_true")
    parser.add_argument("--plane-footprint-padding-texels", type=int, default=12)
    parser.add_argument("--screen-space-feather-width-texels", type=int, default=8)
    parser.add_argument("--harmonic-iterations", type=int, default=300)
    parser.add_argument("--harmonic-relaxation", type=float, default=0.85)
    parser.add_argument("--wall-color-quantization", type=int, default=24)
    parser.add_argument("--wall-color-radius", type=float, default=72.0)
    parser.add_argument("--minimum-wall-support-fraction", type=float, default=0.08)
    parser.add_argument("--wall-luminance-quantile", type=float, default=0.65)
    parser.add_argument("--wall-screen-weight", type=float, default=0.15)
    parser.add_argument("--wall-support-boundary-inset", type=int, default=4)
    parser.add_argument("--minimum-wall-inset-support-fraction", type=float, default=0.05)
    parser.add_argument("--wall-illumination-sigma", type=float, default=24.0)
    parser.add_argument("--wall-kernel-blend-weight-scale", type=float, default=0.02)
    parser.add_argument("--wall-illumination-color-clip-margin", type=float, default=8.0)
    parser.add_argument("--wall-illumination-degree", type=int, default=1)
    parser.add_argument("--wall-illumination-irls-iterations", type=int, default=8)
    parser.add_argument("--wall-illumination-huber-delta", type=float, default=1.5)
    parser.add_argument("--wall-illumination-ridge", type=float, default=1e-4)
    parser.add_argument("--wall-residual-collar-width", type=int, default=16)
    parser.add_argument("--maximum-wall-boundary-p95-delta", type=float, default=24.0)
    parser.add_argument(
        "--maximum-wall-boundary-gradient-p95-delta",
        type=float,
        default=32.0,
    )
    parser.add_argument("--floor-smoothing-iterations", type=int, default=16)
    parser.add_argument("--contact-sheet-samples", type=int, default=6)
    parser.add_argument(
        "--full-resolution-frame-ids",
        default="000064,000067,000072",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_texture_candidate(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "promotion_approved": report["promotion_approved"],
                "eligible_as_round04_clean_plate": report["eligible_as_round04_clean_plate"],
                "observed_fraction": report["provenance"][
                    "observed_fraction_of_original_removal_masks"
                ],
                "synthetic_fraction": report["provenance"][
                    "synthetic_fraction_of_original_removal_masks"
                ],
                "report": str(args.output / "planar_texture_report.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
