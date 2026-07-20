#!/usr/bin/env python3
"""ROS-free trajectory evaluation for Meridian.

The external Newer College comparison protocol is equivalent to::

    evo_ape tum GT_BASE.tum EST_BASE.tum \
        --align --pose_relation trans_part --t_max_diff 0.01

That is a one-to-one timestamp association followed by an SE(3) Umeyama
alignment (rotation and translation, never scale).  Select it with
``--association nearest --max-dt 0.01``.

The evaluator is configured directly from the command line. It never
extrapolates poses or silently trims a result to improve its score.

TUM input rows are::

    timestamp tx ty tz qx qy qz qw

The ground truth and estimate must express the same body frame.  Both frame
names are required inputs and must match exactly; evaluation stops before
timestamp association or alignment when they differ.  Use NCD's base-frame
GT for ``trajectory_base.tum``; ``tum_asimu`` is only valid for an
AlphaSense-IMU-frame estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


FULL_REFERENCE_PROTOCOL = "meridian.full_reference/v1"
FIXED_COLD_START_PROTOCOL = "meridian.fixed_cold_start/v1"


class EvaluationError(ValueError):
    """An input or protocol error that makes a score invalid."""


@dataclass(frozen=True)
class Trajectory:
    """A strictly time-ordered sequence of world-from-body poses."""

    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1:
            raise EvaluationError("timestamps must have shape (N,)")
        count = self.timestamps.shape[0]
        if self.positions.shape != (count, 3):
            raise EvaluationError("positions must have shape (N, 3)")
        if self.quaternions_xyzw.shape != (count, 4):
            raise EvaluationError("quaternions must have shape (N, 4)")
        if count == 0:
            raise EvaluationError("trajectory is empty")
        if not (
            np.all(np.isfinite(self.timestamps))
            and np.all(np.isfinite(self.positions))
            and np.all(np.isfinite(self.quaternions_xyzw))
        ):
            raise EvaluationError("trajectory contains NaN or infinity")
        if count > 1 and np.any(np.diff(self.timestamps) <= 0.0):
            raise EvaluationError("timestamps must be strictly increasing")
        quaternion_norms = np.linalg.norm(self.quaternions_xyzw, axis=1)
        if np.any(quaternion_norms < 1.0e-12):
            raise EvaluationError("trajectory contains a zero quaternion")
        if np.any(np.abs(quaternion_norms - 1.0) > 1.0e-6):
            raise EvaluationError("Trajectory quaternions must be normalized")

    @property
    def count(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def duration(self) -> float:
        if self.count < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


@dataclass(frozen=True)
class EvaluationConfig:
    """All choices that can change a reported score."""

    reference_body_frame: str
    estimate_body_frame: str
    association: str = "interpolate"
    max_dt: float = 0.01
    exact_tolerance: float = 1.0e-9
    max_interpolation_gap: float = 0.20
    min_coverage: float = 0.90
    max_output_gap: float = 0.50
    max_linear_speed: float = 30.0
    max_angular_speed_deg: float = 720.0
    rpe_distance_m: float = 10.0
    track_label: str = "unspecified"
    coverage_protocol: str = FULL_REFERENCE_PROTOCOL
    startup_allowance_s: float = 0.0
    maximum_startup_delay_s: Optional[float] = None
    maximum_end_loss_s: Optional[float] = None

    def validate(self) -> None:
        for name in ("reference_body_frame", "estimate_body_frame"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise EvaluationError(f"{name} must be a non-empty frame name")
            if value != value.strip():
                raise EvaluationError(f"{name} must not contain surrounding whitespace")
        if self.reference_body_frame != self.estimate_body_frame:
            raise EvaluationError(
                "body-frame mismatch: "
                f"reference='{self.reference_body_frame}', "
                f"estimate='{self.estimate_body_frame}'"
            )
        if self.association not in ("interpolate", "nearest", "exact"):
            raise EvaluationError(f"unsupported association: {self.association}")
        for name in (
            "max_dt",
            "exact_tolerance",
            "max_interpolation_gap",
            "max_output_gap",
            "max_linear_speed",
            "max_angular_speed_deg",
            "rpe_distance_m",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise EvaluationError(f"{name} must be finite and positive")
        if not math.isfinite(self.min_coverage) or not 0.0 <= self.min_coverage <= 1.0:
            raise EvaluationError("min_coverage must be in [0, 1]")
        if self.coverage_protocol not in (FULL_REFERENCE_PROTOCOL, FIXED_COLD_START_PROTOCOL):
            raise EvaluationError(f"unsupported coverage protocol: {self.coverage_protocol}")
        if not math.isfinite(self.startup_allowance_s) or self.startup_allowance_s < 0.0:
            raise EvaluationError("startup_allowance_s must be finite and non-negative")
        if self.coverage_protocol == FULL_REFERENCE_PROTOCOL:
            if self.startup_allowance_s != 0.0:
                raise EvaluationError("full-reference protocol cannot declare a startup allowance")
            if self.maximum_startup_delay_s is not None or self.maximum_end_loss_s is not None:
                raise EvaluationError("full-reference protocol cannot declare cold-start limits")
        else:
            if self.maximum_startup_delay_s is None or self.maximum_end_loss_s is None:
                raise EvaluationError(
                    "fixed cold-start protocol requires startup and end-loss limits"
                )
            if (
                not math.isfinite(self.maximum_startup_delay_s)
                or self.maximum_startup_delay_s < 0.0
            ):
                raise EvaluationError(
                    "maximum_startup_delay_s must be finite and non-negative"
                )
            if (
                not math.isfinite(self.maximum_end_loss_s)
                or self.maximum_end_loss_s < 0.0
            ):
                raise EvaluationError("maximum_end_loss_s must be finite and non-negative")
            if self.startup_allowance_s < self.maximum_startup_delay_s:
                raise EvaluationError(
                    "fixed eligible window must not begin before the allowed startup delay"
                )


@dataclass(frozen=True)
class AssociatedTrajectories:
    """Reference poses and estimates evaluated at the same logical stamps."""

    reference: Trajectory
    estimate: Trajectory
    source_time_error_s: np.ndarray

    def __post_init__(self) -> None:
        if self.reference.count != self.estimate.count:
            raise EvaluationError("associated trajectory sizes differ")
        if self.source_time_error_s.shape != (self.reference.count,):
            raise EvaluationError("invalid association time-error array")
        if not np.array_equal(self.reference.timestamps, self.estimate.timestamps):
            raise EvaluationError("associated timestamps are not identical")


def load_tum(path: Path | str, quaternion_norm_tolerance: float = 1.0e-2) -> Trajectory:
    """Parse a TUM trajectory without sorting, dropping, or repairing bad rows.

    Silently sorting timestamp resets or dropping duplicate/NaN rows can turn a
    broken run into a deceptively good partial score, so every non-comment row
    is validated and errors include its source line.
    """

    source = Path(path)
    rows: List[List[float]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            content = raw_line.split("#", 1)[0].strip()
            if not content:
                continue
            fields = content.replace(",", " ").split()
            if len(fields) != 8:
                raise EvaluationError(
                    f"{source}:{line_number}: expected 8 TUM fields, got {len(fields)}"
                )
            try:
                values = [float(field) for field in fields]
            except ValueError as exc:
                raise EvaluationError(
                    f"{source}:{line_number}: non-numeric TUM field"
                ) from exc
            if not np.all(np.isfinite(values)):
                raise EvaluationError(f"{source}:{line_number}: NaN or infinity")
            if rows and values[0] <= rows[-1][0]:
                failure = "duplicate timestamp" if values[0] == rows[-1][0] else "timestamp reset"
                raise EvaluationError(f"{source}:{line_number}: {failure}")
            rows.append(values)

    if not rows:
        raise EvaluationError(f"{source}: no TUM poses")

    matrix = np.asarray(rows, dtype=np.float64)
    quaternions = matrix[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1.0e-12):
        bad = int(np.flatnonzero(norms < 1.0e-12)[0]) + 1
        raise EvaluationError(f"{source}: row {bad} has a zero quaternion")
    norm_error = np.abs(norms - 1.0)
    if np.any(norm_error > quaternion_norm_tolerance):
        bad = int(np.argmax(norm_error)) + 1
        raise EvaluationError(
            f"{source}: row {bad} quaternion norm error {norm_error[bad - 1]:.6g} "
            f"exceeds {quaternion_norm_tolerance:.6g}"
        )
    quaternions = quaternions / norms[:, None]
    return Trajectory(matrix[:, 0], matrix[:, 1:4], quaternions)


def quaternions_to_matrices(quaternions_xyzw: np.ndarray) -> np.ndarray:
    """Convert normalized xyzw quaternions to rotation matrices."""

    q = np.asarray(quaternions_xyzw, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1)[:, None]
    x, y, z, w = q.T
    matrices = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def slerp_xyzw(first: np.ndarray, second: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Vectorized shortest-arc quaternion interpolation."""

    q0 = np.asarray(first, dtype=np.float64).copy()
    q1 = np.asarray(second, dtype=np.float64).copy()
    alpha = np.asarray(fraction, dtype=np.float64).reshape(-1)
    dots = np.sum(q0 * q1, axis=1)
    negative = dots < 0.0
    q1[negative] *= -1.0
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    result = np.empty_like(q0)
    nearly_equal = dots > 0.9995
    if np.any(nearly_equal):
        a = alpha[nearly_equal, None]
        linear = (1.0 - a) * q0[nearly_equal] + a * q1[nearly_equal]
        result[nearly_equal] = linear / np.linalg.norm(linear, axis=1)[:, None]

    curved = ~nearly_equal
    if np.any(curved):
        theta = np.arccos(dots[curved])
        sin_theta = np.sin(theta)
        a = alpha[curved]
        weight0 = np.sin((1.0 - a) * theta) / sin_theta
        weight1 = np.sin(a * theta) / sin_theta
        result[curved] = weight0[:, None] * q0[curved] + weight1[:, None] * q1[curved]
        result[curved] /= np.linalg.norm(result[curved], axis=1)[:, None]
    return result


def _trajectory_subset(trajectory: Trajectory, indices: np.ndarray) -> Trajectory:
    return Trajectory(
        trajectory.timestamps[indices].copy(),
        trajectory.positions[indices].copy(),
        trajectory.quaternions_xyzw[indices].copy(),
    )


def _one_to_one_pairs(
    reference_times: np.ndarray,
    estimate_times: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily select unique closest pairs, matching the TUM/evo convention."""

    candidates: List[Tuple[float, int, int]] = []
    for reference_index, stamp in enumerate(reference_times):
        begin = int(np.searchsorted(estimate_times, stamp - tolerance, side="left"))
        end = int(np.searchsorted(estimate_times, stamp + tolerance, side="right"))
        for estimate_index in range(begin, end):
            candidates.append(
                (
                    abs(float(estimate_times[estimate_index] - stamp)),
                    reference_index,
                    estimate_index,
                )
            )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_reference: set[int] = set()
    used_estimate: set[int] = set()
    accepted: List[Tuple[int, int, float]] = []
    for delta, reference_index, estimate_index in candidates:
        if reference_index in used_reference or estimate_index in used_estimate:
            continue
        used_reference.add(reference_index)
        used_estimate.add(estimate_index)
        accepted.append((reference_index, estimate_index, delta))
    accepted.sort(key=lambda item: item[0])
    if not accepted:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.asarray([item[0] for item in accepted], dtype=np.int64),
        np.asarray([item[1] for item in accepted], dtype=np.int64),
        np.asarray([item[2] for item in accepted], dtype=np.float64),
    )


def associate_trajectories(
    reference: Trajectory,
    estimate: Trajectory,
    config: EvaluationConfig,
) -> AssociatedTrajectories:
    """Associate poses without changing either trajectory's clock origin."""

    # Keep this guard here as well as in evaluate_trajectories: association is
    # a public helper and a frame mismatch must never reach timestamp pairing.
    config.validate()

    if config.association in ("nearest", "exact"):
        tolerance = config.max_dt if config.association == "nearest" else config.exact_tolerance
        reference_indices, estimate_indices, time_errors = _one_to_one_pairs(
            reference.timestamps, estimate.timestamps, tolerance
        )
        reference_associated = _trajectory_subset(reference, reference_indices)
        estimate_source = _trajectory_subset(estimate, estimate_indices)
        # Logical stamps are the GT stamps; source_time_error preserves the actual offset.
        estimate_associated = Trajectory(
            reference_associated.timestamps.copy(),
            estimate_source.positions,
            estimate_source.quaternions_xyzw,
        )
        return AssociatedTrajectories(reference_associated, estimate_associated, time_errors)

    targets = reference.timestamps
    right = np.searchsorted(estimate.timestamps, targets, side="left")
    clipped_right = np.minimum(right, estimate.count - 1)
    exact = (right < estimate.count) & (
        np.abs(estimate.timestamps[clipped_right] - targets) <= config.exact_tolerance
    )
    left = right - 1
    bracketed = (~exact) & (left >= 0) & (right < estimate.count)
    span = np.full(reference.count, np.inf, dtype=np.float64)
    span[bracketed] = estimate.timestamps[right[bracketed]] - estimate.timestamps[left[bracketed]]
    eligible = exact | (bracketed & (span <= config.max_interpolation_gap))
    reference_indices = np.flatnonzero(eligible)

    positions = np.empty((reference_indices.shape[0], 3), dtype=np.float64)
    quaternions = np.empty((reference_indices.shape[0], 4), dtype=np.float64)
    source_error = np.empty(reference_indices.shape[0], dtype=np.float64)
    output_exact = exact[reference_indices]

    if np.any(output_exact):
        selected = reference_indices[output_exact]
        estimate_indices = right[selected]
        positions[output_exact] = estimate.positions[estimate_indices]
        quaternions[output_exact] = estimate.quaternions_xyzw[estimate_indices]
        source_error[output_exact] = np.abs(
            estimate.timestamps[estimate_indices] - targets[selected]
        )

    interpolated = ~output_exact
    if np.any(interpolated):
        selected = reference_indices[interpolated]
        lower = left[selected]
        upper = right[selected]
        interval = estimate.timestamps[upper] - estimate.timestamps[lower]
        fraction = (targets[selected] - estimate.timestamps[lower]) / interval
        positions[interpolated] = (
            (1.0 - fraction[:, None]) * estimate.positions[lower]
            + fraction[:, None] * estimate.positions[upper]
        )
        quaternions[interpolated] = slerp_xyzw(
            estimate.quaternions_xyzw[lower], estimate.quaternions_xyzw[upper], fraction
        )
        source_error[interpolated] = np.minimum(
            targets[selected] - estimate.timestamps[lower],
            estimate.timestamps[upper] - targets[selected],
        )

    reference_associated = _trajectory_subset(reference, reference_indices)
    estimate_associated = Trajectory(
        reference_associated.timestamps.copy(), positions, quaternions
    )
    return AssociatedTrajectories(reference_associated, estimate_associated, source_error)


def umeyama_se3(
    source: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return R,t minimizing ||R*source+t-target||, with scale fixed to one."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise EvaluationError("Umeyama inputs must both have shape (N, 3)")
    if source.shape[0] < 3:
        raise EvaluationError("at least three associated poses are required")
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    covariance = (target - target_mean).T @ (source - source_mean) / source.shape[0]
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0.0:
        correction[2, 2] = -1.0
    rotation = left @ correction @ right_t
    translation = target_mean - rotation @ source_mean
    return rotation, translation, singular_values


def _rotation_angles(rotation_errors: np.ndarray) -> np.ndarray:
    cosine = (np.trace(rotation_errors, axis1=1, axis2=2) - 1.0) * 0.5
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    values = np.asarray(values, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _so3_exp(tangent: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(tangent))
    if angle < 1.0e-12:
        skew = np.array(
            [[0.0, -tangent[2], tangent[1]],
             [tangent[2], 0.0, -tangent[0]],
             [-tangent[1], tangent[0], 0.0]]
        )
        return np.eye(3) + skew
    axis = tangent / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]],
         [axis[2], 0.0, -axis[0]],
         [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array(
        [rotation[2, 1] - rotation[1, 2],
         rotation[0, 2] - rotation[2, 0],
         rotation[1, 0] - rotation[0, 1]]
    )
    if angle < 1.0e-10:
        return 0.5 * vee
    if math.pi - angle < 1.0e-6:
        # The eigenvector with eigenvalue one is stable at the pi branch.
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= np.linalg.norm(axis)
        if np.dot(axis, vee) < 0.0:
            axis *= -1.0
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * vee


def _interpolate_rotation(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    return first @ _so3_exp(fraction * _so3_log(first.T @ second))


def _distance_rpe(
    reference_positions: np.ndarray,
    reference_rotations: np.ndarray,
    estimate_positions: np.ndarray,
    estimate_rotations: np.ndarray,
    distance_m: float,
) -> Dict[str, Any]:
    """Compute overlapping relative errors at exactly ``distance_m`` of GT path."""

    increments = np.linalg.norm(np.diff(reference_positions, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))
    translation_errors: List[float] = []
    rotation_errors_deg: List[float] = []

    for start in range(reference_positions.shape[0] - 1):
        target_distance = cumulative[start] + distance_m
        if target_distance > cumulative[-1]:
            break
        upper = int(np.searchsorted(cumulative, target_distance, side="left"))
        if upper <= start:
            continue
        if cumulative[upper] == target_distance:
            lower = upper
            fraction = 0.0
        else:
            lower = upper - 1
            segment = cumulative[upper] - cumulative[lower]
            if segment <= 1.0e-12:
                continue
            fraction = (target_distance - cumulative[lower]) / segment

        if lower == upper:
            reference_end_position = reference_positions[upper]
            estimate_end_position = estimate_positions[upper]
            reference_end_rotation = reference_rotations[upper]
            estimate_end_rotation = estimate_rotations[upper]
        else:
            reference_end_position = (
                (1.0 - fraction) * reference_positions[lower]
                + fraction * reference_positions[upper]
            )
            estimate_end_position = (
                (1.0 - fraction) * estimate_positions[lower]
                + fraction * estimate_positions[upper]
            )
            reference_end_rotation = _interpolate_rotation(
                reference_rotations[lower], reference_rotations[upper], fraction
            )
            estimate_end_rotation = _interpolate_rotation(
                estimate_rotations[lower], estimate_rotations[upper], fraction
            )

        reference_relative_rotation = reference_rotations[start].T @ reference_end_rotation
        estimate_relative_rotation = estimate_rotations[start].T @ estimate_end_rotation
        reference_relative_translation = reference_rotations[start].T @ (
            reference_end_position - reference_positions[start]
        )
        estimate_relative_translation = estimate_rotations[start].T @ (
            estimate_end_position - estimate_positions[start]
        )
        translation_error = reference_relative_rotation.T @ (
            estimate_relative_translation - reference_relative_translation
        )
        rotation_error = reference_relative_rotation.T @ estimate_relative_rotation
        translation_errors.append(float(np.linalg.norm(translation_error)))
        rotation_errors_deg.append(float(math.degrees(_rotation_angles(rotation_error[None])[0])))

    translation = np.asarray(translation_errors, dtype=np.float64)
    rotation = np.asarray(rotation_errors_deg, dtype=np.float64)
    result: Dict[str, Any] = {
        "distance_m": float(distance_m),
        "pair_count": int(translation.size),
        "translation_m": _stats(translation),
        "rotation_deg": _stats(rotation),
    }
    if translation.size:
        result["translation_drift_percent"] = _stats(100.0 * translation / distance_m)
    return result


def _trajectory_health(
    trajectory: Trajectory,
    max_gap: float,
    max_linear_speed: float,
    max_angular_speed_deg: float,
) -> Dict[str, Any]:
    if trajectory.count < 2:
        return {
            "duration_s": trajectory.duration,
            "largest_gap_s": 0.0,
            "gap_count": 0,
            "reset_count": 0,
            "reset_candidates": [],
        }
    dt = np.diff(trajectory.timestamps)
    gaps = np.flatnonzero(dt > max_gap)
    linear_speed = np.linalg.norm(np.diff(trajectory.positions, axis=0), axis=1) / dt
    rotations = quaternions_to_matrices(trajectory.quaternions_xyzw)
    relative = np.einsum("nij,njk->nik", np.transpose(rotations[:-1], (0, 2, 1)), rotations[1:])
    angular_speed_deg = np.degrees(_rotation_angles(relative)) / dt
    reset_indices = np.flatnonzero(
        (linear_speed > max_linear_speed) | (angular_speed_deg > max_angular_speed_deg)
    )
    reset_candidates = [
        {
            "before_index": int(index),
            "after_timestamp": float(trajectory.timestamps[index + 1]),
            "dt_s": float(dt[index]),
            "translation_step_m": float(
                np.linalg.norm(trajectory.positions[index + 1] - trajectory.positions[index])
            ),
            "linear_speed_mps": float(linear_speed[index]),
            "angular_speed_degps": float(angular_speed_deg[index]),
        }
        for index in reset_indices[:20]
    ]
    return {
        "duration_s": trajectory.duration,
        "largest_gap_s": float(np.max(dt)),
        "gap_count": int(gaps.size),
        "max_linear_speed_mps": float(np.max(linear_speed)),
        "max_angular_speed_degps": float(np.max(angular_speed_deg)),
        "reset_count": int(reset_indices.size),
        "reset_candidates": reset_candidates,
    }


def evaluate_trajectories(
    reference: Trajectory,
    estimate: Trajectory,
    config: EvaluationConfig,
) -> Dict[str, Any]:
    """Evaluate one complete track and return a JSON-serializable report."""

    settings = config
    settings.validate()
    estimate_health = _trajectory_health(
        estimate,
        settings.max_output_gap,
        settings.max_linear_speed,
        settings.max_angular_speed_deg,
    )
    whole_associated = associate_trajectories(reference, estimate, settings)
    if settings.coverage_protocol == FIXED_COLD_START_PROTOCOL:
        window_start = float(reference.timestamps[0] + settings.startup_allowance_s)
        eligible_indices = np.flatnonzero(reference.timestamps >= window_start)
        if eligible_indices.size < 3:
            raise EvaluationError(
                "fixed cold-start window contains fewer than three ground-truth poses"
            )
        eligible_reference = _trajectory_subset(reference, eligible_indices)
    else:
        window_start = float(reference.timestamps[0])
        eligible_reference = reference
    window_end = float(reference.timestamps[-1])
    associated = associate_trajectories(eligible_reference, estimate, settings)
    if associated.reference.count < 3:
        raise EvaluationError(
            f"only {associated.reference.count} associated poses; at least 3 are required"
        )

    alignment_rotation, alignment_translation, singular_values = umeyama_se3(
        associated.estimate.positions, associated.reference.positions
    )
    estimate_positions_aligned = (
        associated.estimate.positions @ alignment_rotation.T + alignment_translation
    )
    reference_rotations = quaternions_to_matrices(associated.reference.quaternions_xyzw)
    estimate_rotations = quaternions_to_matrices(associated.estimate.quaternions_xyzw)
    estimate_rotations_aligned = np.einsum(
        "ij,njk->nik", alignment_rotation, estimate_rotations
    )

    translation_residuals = np.linalg.norm(
        estimate_positions_aligned - associated.reference.positions, axis=1
    )
    orientation_residuals_deg = np.degrees(
        _rotation_angles(
            np.einsum(
                "nij,njk->nik",
                np.transpose(reference_rotations, (0, 2, 1)),
                estimate_rotations_aligned,
            )
        )
    )

    whole_matched_duration = whole_associated.reference.duration
    whole_time_coverage = (
        1.0 if reference.duration == 0.0 else whole_matched_duration / reference.duration
    )
    whole_pose_coverage = whole_associated.reference.count / reference.count
    window_duration = window_end - window_start
    matched_window_start = max(window_start, float(associated.reference.timestamps[0]))
    matched_window_end = min(window_end, float(associated.reference.timestamps[-1]))
    matched_window_duration = max(0.0, matched_window_end - matched_window_start)
    time_coverage = 1.0 if window_duration == 0.0 else matched_window_duration / window_duration
    pose_coverage = associated.reference.count / eligible_reference.count
    startup_delay_s = max(0.0, float(estimate.timestamps[0] - reference.timestamps[0]))
    end_loss_s = max(0.0, float(reference.timestamps[-1] - estimate.timestamps[-1]))
    coverage_label = (
        "fixed eligible-window"
        if settings.coverage_protocol == FIXED_COLD_START_PROTOCOL
        else "whole-reference"
    )
    quality_failures: List[str] = []
    if time_coverage < settings.min_coverage:
        quality_failures.append(
            f"{coverage_label} time coverage {time_coverage:.3%} is below "
            f"{settings.min_coverage:.3%}"
        )
    if pose_coverage < settings.min_coverage:
        quality_failures.append(
            f"{coverage_label} GT pose coverage {pose_coverage:.3%} is below "
            f"{settings.min_coverage:.3%}"
        )
    if (
        settings.maximum_startup_delay_s is not None
        and startup_delay_s > settings.maximum_startup_delay_s
    ):
        quality_failures.append(
            f"startup delay {startup_delay_s:.6f} s exceeds "
            f"{settings.maximum_startup_delay_s:.6f} s"
        )
    if settings.maximum_end_loss_s is not None and end_loss_s > settings.maximum_end_loss_s:
        quality_failures.append(
            f"end loss {end_loss_s:.6f} s exceeds {settings.maximum_end_loss_s:.6f} s"
        )
    if estimate_health["gap_count"]:
        quality_failures.append(
            f"estimate has {estimate_health['gap_count']} output gap(s) over "
            f"{settings.max_output_gap:.3f} s"
        )
    if estimate_health["reset_count"]:
        quality_failures.append(
            f"estimate has {estimate_health['reset_count']} reset/discontinuity candidate(s)"
        )

    warnings: List[str] = []
    if settings.track_label == "unspecified":
        warnings.append("track label is unspecified; report local and final-global separately")
    if settings.association == "nearest" and abs(settings.max_dt - 0.01) > 1.0e-12:
        warnings.append("nearest max_dt differs from the official NCD/evo 0.01 s protocol")
    if settings.association != "nearest":
        warnings.append(
            "interpolated/exact association is an internal Meridian metric; use nearest/0.01 s "
            "for published NCD comparisons"
        )

    report: Dict[str, Any] = {
        "schema": "meridian.trajectory_evaluation.v3",
        "protocol": {
            "track_label": settings.track_label,
            "reference_body_frame": settings.reference_body_frame,
            "estimate_body_frame": settings.estimate_body_frame,
            "association": settings.association,
            "unique_one_to_one": settings.association in ("nearest", "exact"),
            "coverage_protocol": settings.coverage_protocol,
            "max_dt_s": settings.max_dt if settings.association == "nearest" else None,
            "exact_tolerance_s": settings.exact_tolerance,
            "max_interpolation_gap_s": (
                settings.max_interpolation_gap if settings.association == "interpolate" else None
            ),
            "alignment": "umeyama_se3_rotation_translation_no_scale",
            "alignment_scale": 1.0,
            "estimate_dependent_trimming": False,
            "extrapolation": False,
            "full_reference_no_trimming": (
                settings.coverage_protocol == FULL_REFERENCE_PROTOCOL
            ),
            "translation_pose_relation": "trans_part",
            "reference_window": {
                "kind": (
                    "fixed_startup_allowance"
                    if settings.coverage_protocol == FIXED_COLD_START_PROTOCOL
                    else "full_reference"
                ),
                "start_timestamp": window_start,
                "end_timestamp": window_end,
                "startup_allowance_s": settings.startup_allowance_s,
                "start_inclusive": True,
                "selection": (
                    "fixed_before_estimate_association"
                    if settings.coverage_protocol == FIXED_COLD_START_PROTOCOL
                    else "full_reference"
                ),
                "ground_truth_stamps_modified": False,
            },
        },
        "input": {
            "reference_pose_count": reference.count,
            "eligible_reference_pose_count": eligible_reference.count,
            "estimate_pose_count": estimate.count,
            "reference_first_timestamp": float(reference.timestamps[0]),
            "reference_last_timestamp": float(reference.timestamps[-1]),
            "estimate_first_timestamp": float(estimate.timestamps[0]),
            "estimate_last_timestamp": float(estimate.timestamps[-1]),
        },
        "availability": {
            "startup_delay_s": startup_delay_s,
            "maximum_startup_delay_s": settings.maximum_startup_delay_s,
            "startup_delay_passed": (
                None
                if settings.maximum_startup_delay_s is None
                else startup_delay_s <= settings.maximum_startup_delay_s
            ),
            "end_loss_s": end_loss_s,
            "maximum_end_loss_s": settings.maximum_end_loss_s,
            "end_loss_passed": (
                None
                if settings.maximum_end_loss_s is None
                else end_loss_s <= settings.maximum_end_loss_s
            ),
        },
        "whole_reference_association": {
            "pose_count": whole_associated.reference.count,
            "first_matched_timestamp": float(whole_associated.reference.timestamps[0]),
            "last_matched_timestamp": float(whole_associated.reference.timestamps[-1]),
            "time_coverage_ratio": float(whole_time_coverage),
            "reference_pose_coverage_ratio": float(whole_pose_coverage),
        },
        "association": {
            "pose_count": associated.reference.count,
            "first_matched_timestamp": float(associated.reference.timestamps[0]),
            "last_matched_timestamp": float(associated.reference.timestamps[-1]),
            "time_coverage_ratio": float(time_coverage),
            "reference_pose_coverage_ratio": float(pose_coverage),
            "coverage_denominator": (
                "fixed_eligible_reference_window"
                if settings.coverage_protocol == FIXED_COLD_START_PROTOCOL
                else "full_reference"
            ),
            "minimum_coverage_ratio": settings.min_coverage,
            "source_time_error_s": _stats(associated.source_time_error_s),
        },
        "estimate_health": estimate_health,
        "alignment": {
            "rotation": alignment_rotation.tolist(),
            "translation_m": alignment_translation.tolist(),
            "cross_covariance_singular_values": singular_values.tolist(),
        },
        "ate_translation_m": _stats(translation_residuals),
        "absolute_rotation_deg": _stats(orientation_residuals_deg),
        "rpe": _distance_rpe(
            associated.reference.positions,
            reference_rotations,
            estimate_positions_aligned,
            estimate_rotations_aligned,
            settings.rpe_distance_m,
        ),
        "quality": {
            "passed": not quality_failures,
            "failures": quality_failures,
            "warnings": warnings,
        },
    }
    return report


def _human_report(report: Mapping[str, Any]) -> str:
    protocol = report["protocol"]
    association = report["association"]
    whole = report["whole_reference_association"]
    availability = report["availability"]
    health = report["estimate_health"]
    ate = report["ate_translation_m"]
    rotation = report["absolute_rotation_deg"]
    rpe = report["rpe"]
    quality = report["quality"]
    active_coverage_name = (
        "eligible"
        if protocol["coverage_protocol"] == FIXED_COLD_START_PROTOCOL
        else "whole-reference"
    )
    lines = [
        f"track: {protocol['track_label']}  "
        f"reference body: {protocol['reference_body_frame']}  "
        f"estimate body: {protocol['estimate_body_frame']}",
        f"protocol: association={protocol['association']}, "
        f"coverage={protocol['coverage_protocol']}, "
        "alignment=SE(3) Umeyama (scale=1), estimate-dependent trimming=off",
        f"associated: {association['pose_count']}  "
        f"{active_coverage_name} GT pose coverage="
        f"{100.0 * association['reference_pose_coverage_ratio']:.2f}%  "
        f"{active_coverage_name} time coverage="
        f"{100.0 * association['time_coverage_ratio']:.2f}%",
        f"whole-reference: poses={100.0 * whole['reference_pose_coverage_ratio']:.2f}%  "
        f"time={100.0 * whole['time_coverage_ratio']:.2f}%",
        f"availability: startup={availability['startup_delay_s']:.6f} s  "
        f"end_loss={availability['end_loss_s']:.6f} s",
        f"matched time: {association['first_matched_timestamp']:.6f} .. "
        f"{association['last_matched_timestamp']:.6f}",
        f"output health: largest_gap={health['largest_gap_s']:.6f} s, "
        f"gaps={health['gap_count']}, resets={health['reset_count']}",
        "",
        f"ATE translation: RMSE={ate['rmse']:.6f} m  median={ate['median']:.6f} m  "
        f"p95={ate['p95']:.6f} m  max={ate['max']:.6f} m",
        f"absolute rotation: RMSE={rotation['rmse']:.6f} deg  "
        f"median={rotation['median']:.6f} deg  p95={rotation['p95']:.6f} deg  "
        f"max={rotation['max']:.6f} deg",
    ]
    if rpe["pair_count"]:
        rpe_translation = rpe["translation_m"]
        rpe_rotation = rpe["rotation_deg"]
        lines.append(
            f"RPE {rpe['distance_m']:.3f} m: pairs={rpe['pair_count']}  "
            f"translation RMSE={rpe_translation['rmse']:.6f} m  "
            f"rotation RMSE={rpe_rotation['rmse']:.6f} deg  "
            f"drift={rpe['translation_drift_percent']['rmse']:.4f}%"
        )
    else:
        lines.append(f"RPE {rpe['distance_m']:.3f} m: unavailable (track is too short)")
    lines.append("")
    lines.append("QUALITY: PASS" if quality["passed"] else "QUALITY: FAIL")
    lines.extend(f"  failure: {failure}" for failure in quality["failures"])
    lines.extend(f"  warning: {warning}" for warning in quality["warnings"])
    return "\n".join(lines)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ground_truth", type=Path, help="TUM ground truth in the evaluated body frame"
    )
    parser.add_argument("estimate", type=Path, help="TUM estimate in the same body frame")
    parser.add_argument(
        "--association",
        choices=("interpolate", "nearest", "exact"),
        default=None,
        help=(
            "interpolate estimate at GT stamps (default); nearest is unique evo/TUM matching; "
            "exact requires equal stamps"
        ),
    )
    parser.add_argument(
        "--max-dt",
        type=float,
        default=None,
        help="nearest association tolerance in seconds (official NCD/evo: 0.01)",
    )
    parser.add_argument("--exact-tolerance", type=float, default=None)
    parser.add_argument(
        "--max-interpolation-gap",
        type=float,
        default=None,
        help="never interpolate across a larger estimate gap (seconds)",
    )
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument(
        "--max-output-gap",
        type=float,
        default=None,
        help="a larger estimate interval fails the run quality gate (seconds)",
    )
    parser.add_argument("--max-linear-speed", type=float, default=None)
    parser.add_argument("--max-angular-speed-deg", type=float, default=None)
    parser.add_argument("--rpe-distance", type=float, default=None)
    parser.add_argument(
        "--track-label",
        choices=("local", "final-global", "unspecified"),
        default="unspecified",
        help="keep local and final-global benchmark reports separate",
    )
    parser.add_argument(
        "--reference-body-frame",
        default=None,
        help="body frame expressed by the ground-truth poses (for example, base_link)",
    )
    parser.add_argument(
        "--estimate-body-frame",
        default=None,
        help="body frame expressed by the estimate poses; must exactly match the reference",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--output-json", type=Path, help="also write the full report to this path")
    parser.add_argument(
        "--allow-quality-failures",
        action="store_true",
        help="return success despite coverage, output-gap, or reset gate failures",
    )
    return parser


def _config_from_arguments(args: argparse.Namespace) -> EvaluationConfig:
    if args.reference_body_frame is None or args.estimate_body_frame is None:
        raise EvaluationError(
            "--reference-body-frame and --estimate-body-frame are required"
        )
    config = EvaluationConfig(
        reference_body_frame=args.reference_body_frame,
        estimate_body_frame=args.estimate_body_frame,
        association=args.association or "interpolate",
        max_dt=0.01 if args.max_dt is None else args.max_dt,
        exact_tolerance=1.0e-9 if args.exact_tolerance is None else args.exact_tolerance,
        max_interpolation_gap=(
            0.20 if args.max_interpolation_gap is None else args.max_interpolation_gap
        ),
        min_coverage=0.90 if args.min_coverage is None else args.min_coverage,
        max_output_gap=0.50 if args.max_output_gap is None else args.max_output_gap,
        max_linear_speed=30.0 if args.max_linear_speed is None else args.max_linear_speed,
        max_angular_speed_deg=(
            720.0 if args.max_angular_speed_deg is None else args.max_angular_speed_deg
        ),
        rpe_distance_m=10.0 if args.rpe_distance is None else args.rpe_distance,
        track_label=args.track_label,
    )
    config.validate()
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_arguments(args)
        reference = load_tum(args.ground_truth)
        estimate = load_tum(args.estimate)
        report = evaluate_trajectories(reference, estimate, config)
    except (OSError, EvaluationError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized if args.json else _human_report(report))
    if not report["quality"]["passed"] and not args.allow_quality_failures:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
