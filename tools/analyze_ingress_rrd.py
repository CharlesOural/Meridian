#!/usr/bin/env python3
"""Post-run integrity and size report for a Meridian ingress RRD.

The recording is inspected through Rerun's supported local catalog API. When
``--bag`` is supplied, ``--config`` is required: the ROS parameter file names
the IMU and LiDAR topics whose rosbag message counts are compared with the
accepted rows in the RRD. No dataset-specific topic or count is built in.
Optional local-RT initialization, bootstrap, preintegration, estimator quality,
registration-map snapshots, stage timing, and trajectory records are summarized
when present; their absence is valid for ingress-only recordings. Supplying
``--ground-truth-state`` evaluates every valid recorded base-frame trajectory
with the official Newer College nearest 10 ms association and the existing
Meridian ATE/RPE implementation.

Examples::

    python3 tools/analyze_ingress_rrd.py out/ingress.rrd

    python3 tools/analyze_ingress_rrd.py out/ingress.rrd \
        --bag path/to/bag \
        --config path/to/ingress.yaml

    python3 tools/analyze_ingress_rrd.py out/run.rrd \
        --ground-truth-state bags/newer-college/gt/state/gt-nc-quad-easy.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


IMU_PATH = "/sensors/imu/sample"
LIDAR_PATH = "/sensors/lidar/metadata"
PREVIEW_PATH = "/sensors/lidar/preview"
REGISTRATION_MAP_PATH = "/local_rt/map/registration"
FAILURE_PATH = "/runtime/ingress_failure"
DEBUG_DROPS_PATH = "/run/debug_events_dropped"
DEBUG_ERRORS_PATH = "/run/debug_log_errors"

INITIALIZATION_STATUS_PATH = "/local_rt/initialization/status"
INITIALIZATION_COUNTS_PATH = "/local_rt/initialization/counts"
ACCEPTED_SEED_PATH = "/local_rt/initialization/accepted_seed/velocity_odom_m_s"
BOOTSTRAP_QUALITY_PATH = "/local_rt/bootstrap/quality"
PREINTEGRATION_QUALITY_PATH = "/local_rt/preintegration/quality"
PREINTEGRATION_BACKEND_PATH = "/local_rt/preintegration/backend"
ESTIMATOR_QUALITY_PATH = "/local_rt/estimator/quality"

TIMING_PREFIX = "/local_rt/timing/"
TRAJECTORY_PATHS = {
    "causal": "/local_rt/trajectory/causal/pose_odom_base",
    "final": "/local_rt/trajectory/final/pose_odom_base",
}
TRAJECTORY_FIELDS = (
    "tx_m",
    "ty_m",
    "tz_m",
    "qx",
    "qy",
    "qz",
    "qw",
)

SCALAR_COMPONENT = "rerun.components.Scalar"
POSITION_COMPONENT = "rerun.components.Position3D"
TEXT_COMPONENT = "rerun.components.Text"

IMU_FIELDS = (
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "arrival_steady_ns",
    "conversion_ns",
)
LIDAR_FIELDS = (
    "source_points",
    "accepted_points",
    "nonfinite_xyz_points",
    "zero_xyz_points",
    "flattened_time_regressions",
    "arrival_steady_ns",
    "conversion_ns",
    "queue_depth",
    "acquisition_duration_ns",
)
INITIALIZATION_COUNT_FIELDS = (
    "imu_sample_count",
    "lidar_sweep_count",
    "fitted_transition_count",
    "rejected_transition_count",
)
ESTIMATOR_QUALITY_FIELDS = (
    "active_state_count",
    "imu_factor_count",
    "lidar_batch_count",
    "active_lidar_rows",
    "finalized_lidar_rows",
    "finalized_map_points",
    "selected_active_owners",
    "registration_correspondences",
    "marginal_prior_rank",
    "registration_rmse_m",
    "initial_cost",
    "final_cost",
    "pose_correction_translation_m",
    "pose_correction_rotation_rad",
    "accepted",
    "prepared_target_points",
    "prepared_source_points",
    "association_pass_count",
    "association_input_points",
    "association_rows_before_cap",
    "registration_iterations",
    "live_query_voxel_probes",
    "finalized_query_voxel_probes",
    "reassociated_rows",
    "rejected_stale_rows",
)
ESTIMATOR_QUALITY_BASE_FIELD_COUNT = 15
ESTIMATOR_QUALITY_APPENDED_FIELDS = ESTIMATOR_QUALITY_FIELDS[
    ESTIMATOR_QUALITY_BASE_FIELD_COUNT:
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    required: bool = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rrd", type=Path, help="completed Rerun .rrd recording")
    parser.add_argument(
        "--bag",
        type=Path,
        help="optional rosbag directory or file for post-run count and size comparison",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="ROS parameter YAML containing imu_topic and lidar_topic",
    )
    parser.add_argument(
        "--ground-truth-state",
        type=Path,
        help=(
            "optional exact state CSV (#sec,nsec,x,y,z,qx,qy,qz,qw); evaluates any "
            "recorded base-frame causal/final trajectories with nearest 10 ms SE(3) alignment"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise ValueError(f"path does not exist or is not a file/directory: {path}")


def _verify_rrd(path: Path) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["rerun", "rrd", "verify", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except FileNotFoundError:
        return False, "rerun CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "rerun rrd verify timed out after 60 seconds"
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    detail = output.splitlines()[-1] if output else f"exit code {result.returncode}"
    return result.returncode == 0, detail


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read ROS and rosbag YAML") from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read YAML {path}: {exc}") from exc


def _config_report(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    document = _load_yaml(resolved)
    candidates: set[tuple[str, str, float, int]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            parameters = value.get("ros__parameters")
            if isinstance(parameters, Mapping):
                imu = parameters.get("imu_topic")
                lidar = parameters.get("lidar_topic")
                if isinstance(imu, str) and imu and isinstance(lidar, str) and lidar:
                    rate = parameters.get("preview_rate_hz", 1.0)
                    point_cap = parameters.get("preview_max_points", 4096)
                    if (
                        not isinstance(rate, (int, float))
                        or isinstance(rate, bool)
                        or not math.isfinite(float(rate))
                        or float(rate) <= 0.0
                    ):
                        raise ValueError("preview_rate_hz must be finite and positive")
                    if (
                        not isinstance(point_cap, int)
                        or isinstance(point_cap, bool)
                        or point_cap <= 0
                    ):
                        raise ValueError("preview_max_points must be a positive integer")
                    candidates.add((imu, lidar, float(rate), point_cap))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    if not candidates:
        raise ValueError(
            f"{resolved} has no ros__parameters mapping with imu_topic and lidar_topic"
        )
    if len(candidates) != 1:
        raise ValueError(f"{resolved} contains conflicting IMU/LiDAR topic pairs")
    imu_topic, lidar_topic, preview_rate_hz, preview_max_points = next(iter(candidates))
    return {
        "path": str(resolved),
        "imu_topic": imu_topic,
        "lidar_topic": lidar_topic,
        "preview_rate_hz": preview_rate_hz,
        "preview_max_points": preview_max_points,
    }


def _metadata_path(bag: Path) -> Optional[Path]:
    if bag.is_dir():
        candidate = bag / "metadata.yaml"
    elif bag.name == "metadata.yaml":
        candidate = bag
    else:
        candidate = bag.parent / "metadata.yaml"
    return candidate if candidate.is_file() else None


def _counts_from_metadata(path: Path) -> dict[str, int]:
    document = _load_yaml(path)
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} must contain a mapping")
    information = document.get("rosbag2_bagfile_information")
    if not isinstance(information, Mapping):
        raise ValueError(f"{path} has no rosbag2_bagfile_information mapping")
    entries = information.get("topics_with_message_count")
    if not isinstance(entries, list):
        raise ValueError(f"{path} has no topics_with_message_count list")
    counts: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        metadata = entry.get("topic_metadata")
        name = metadata.get("name") if isinstance(metadata, Mapping) else None
        count = entry.get("message_count")
        if isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool):
            counts[name] = count
    if not counts:
        raise ValueError(f"{path} contains no readable topic counts")
    return counts


def _counts_from_ros2_info(bag: Path) -> dict[str, int]:
    try:
        result = subprocess.run(
            ["ros2", "bag", "info", str(bag)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    except FileNotFoundError as exc:
        raise ValueError("metadata.yaml is unavailable and ros2 is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ros2 bag info timed out after 60 seconds") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise ValueError(
            "ros2 bag info failed: " + (detail[-1] if detail else str(result.returncode))
        )
    pattern = re.compile(r"Topic:\s*(\S+)\s*\|.*?\bCount:\s*(\d+)")
    counts = {match.group(1): int(match.group(2)) for match in pattern.finditer(result.stdout)}
    if not counts:
        raise ValueError("ros2 bag info output contained no parseable topic counts")
    return counts


def _bag_report(path: Path, topics: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.resolve()
    size = _path_size(resolved)
    metadata = _metadata_path(resolved)
    metadata_error: Optional[str] = None
    if metadata is not None:
        try:
            counts = _counts_from_metadata(metadata)
            source = str(metadata)
        except ValueError as exc:
            metadata_error = str(exc)
            counts = _counts_from_ros2_info(resolved)
            source = "ros2 bag info"
    else:
        counts = _counts_from_ros2_info(resolved)
        source = "ros2 bag info"
    selected = {
        "imu": {"topic": topics["imu_topic"], "messages": counts.get(topics["imu_topic"])},
        "lidar": {
            "topic": topics["lidar_topic"],
            "messages": counts.get(topics["lidar_topic"]),
        },
    }
    return {
        "path": str(resolved),
        "size_bytes": size,
        "count_source": source,
        "metadata_error": metadata_error,
        "selected_topics": selected,
    }


def _component_names(descriptors: Iterable[Any], path: str, component_type: str) -> list[str]:
    return [
        descriptor.name
        for descriptor in descriptors
        if descriptor.entity_path == path
        and not descriptor.is_property
        and descriptor.component_type == component_type
    ]


def _validity(table: Any, column: str) -> list[bool]:
    if column not in table.column_names:
        return [False] * table.num_rows
    return table[column].is_valid().to_pylist()


def _event_mask(table: Any, columns: Sequence[str]) -> list[bool]:
    validity = [_validity(table, column) for column in columns]
    if not validity:
        return [False] * table.num_rows
    return [any(column[row] for column in validity) for row in range(table.num_rows)]


def _read_entity(
    dataset: Any,
    path: str,
    columns: Sequence[str],
    index_names: set[str],
    *,
    temporal: bool,
    preferred_indices: Sequence[str] = (
        "measurement_id",
        "sensor_time",
        "log_time",
    ),
) -> tuple[Optional[str], Any]:
    view = dataset.filter_contents(path)
    if temporal:
        last: Optional[tuple[str, Any]] = None
        for index in preferred_indices:
            if index not in index_names:
                continue
            table = view.reader(index=index).to_arrow_table()
            last = (index, table)
            if any(_event_mask(table, columns)):
                return index, table
        if last is not None:
            return last
    return None, view.reader(index=None).to_arrow_table()


def _as_list(value: Any) -> Optional[list[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _stats(values: Iterable[Any]) -> Optional[dict[str, Any]]:
    finite = sorted(
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    if not finite:
        return None
    return {
        "count": len(finite),
        "min": finite[0],
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": _quantile(finite, 0.95),
        "p99": _quantile(finite, 0.99),
        "max": finite[-1],
    }


def _timeline_values(table: Any, name: str, mask: Sequence[bool]) -> list[int]:
    if name not in table.column_names:
        return []
    try:
        values = table[name].cast("int64").to_pylist()
    except (TypeError, ValueError):
        values = table[name].to_pylist()
    return [
        int(value)
        for value, keep in zip(values, mask)
        if keep and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def _timeline_report(table: Any, mask: Sequence[bool]) -> dict[str, Any]:
    sensor_time = _timeline_values(table, "sensor_time", mask)
    measurement_ids = _timeline_values(table, "measurement_id", mask)
    state_ids = _timeline_values(table, "state_id", mask)
    estimator_revisions = _timeline_values(table, "estimator_revision", mask)
    gaps_ns = [current - previous for previous, current in zip(sensor_time, sensor_time[1:])]
    duration_s: Optional[float] = None
    rate_hz: Optional[float] = None
    if len(sensor_time) >= 2:
        duration_s = (sensor_time[-1] - sensor_time[0]) / 1.0e9
        if duration_s > 0.0:
            rate_hz = (len(sensor_time) - 1) / duration_s
    return {
        "sensor_time_rows": len(sensor_time),
        "sensor_time_first_ns": sensor_time[0] if sensor_time else None,
        "sensor_time_last_ns": sensor_time[-1] if sensor_time else None,
        "duration_s": duration_s,
        "mean_rate_hz": rate_hz,
        "timestamp_gap_s": _stats(gap / 1.0e9 for gap in gaps_ns),
        "nonpositive_timestamp_gaps": sum(gap <= 0 for gap in gaps_ns),
        "measurement_id_rows": len(measurement_ids),
        "measurement_id_first": measurement_ids[0] if measurement_ids else None,
        "measurement_id_last": measurement_ids[-1] if measurement_ids else None,
        "measurement_id_unique": len(set(measurement_ids)),
        "measurement_ids": measurement_ids,
        "state_id_rows": len(state_ids),
        "state_id_first": state_ids[0] if state_ids else None,
        "state_id_last": state_ids[-1] if state_ids else None,
        "state_id_unique": len(set(state_ids)),
        "estimator_revision_rows": len(estimator_revisions),
        "estimator_revision_first": estimator_revisions[0]
        if estimator_revisions
        else None,
        "estimator_revision_last": estimator_revisions[-1]
        if estimator_revisions
        else None,
        "estimator_revision_unique": len(set(estimator_revisions)),
    }


def _scalar_report(
    dataset: Any,
    descriptors: Sequence[Any],
    index_names: set[str],
    path: str,
    fields: Sequence[str],
    *,
    preferred_indices: Sequence[str] = (
        "measurement_id",
        "sensor_time",
        "log_time",
    ),
) -> tuple[dict[str, Any], set[int]]:
    columns = _component_names(descriptors, path, SCALAR_COMPONENT)
    query_index, table = _read_entity(
        dataset,
        path,
        columns,
        index_names,
        temporal=True,
        preferred_indices=preferred_indices,
    )
    mask = _event_mask(table, columns)
    cells = table[columns[0]].to_pylist() if len(columns) == 1 else []
    rows = [
        normalized
        for normalized, keep in zip((_as_list(cell) for cell in cells), mask)
        if keep and normalized is not None
    ]
    widths = [len(row) for row in rows]
    invalid_values = sum(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for row in rows
        for value in row
    )
    field_stats = {
        field: _stats(row[index] for row in rows if index < len(row))
        for index, field in enumerate(fields)
    }
    timeline = _timeline_report(table, mask)
    measurement_ids = set(timeline.pop("measurement_ids"))
    return (
        {
            "path": path,
            "present": bool(columns),
            "query_index": query_index,
            "primary_component_columns": columns,
            "rows": len(rows),
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "invalid_values": invalid_values,
            "timeline": timeline,
            "field_stats": field_stats,
        },
        measurement_ids,
    )


def _point_report(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str]
) -> tuple[dict[str, Any], set[int]]:
    columns = _component_names(descriptors, PREVIEW_PATH, POSITION_COMPONENT)
    query_index, table = _read_entity(
        dataset, PREVIEW_PATH, columns, index_names, temporal=True
    )
    mask = _event_mask(table, columns)
    counts = _point_counts(table, columns, mask)
    timeline = _timeline_report(table, mask)
    ids = set(timeline.pop("measurement_ids"))
    return (
        {
            "path": PREVIEW_PATH,
            "present": bool(columns),
            "query_index": query_index,
            "primary_component_columns": columns,
            "frames": len(counts),
            "total_points": sum(counts),
            "points_per_frame": _stats(counts),
            "max_points": max(counts) if counts else None,
            "timeline": timeline,
        },
        ids,
    )


def _point_counts(table: Any, columns: Sequence[str], mask: Sequence[bool]) -> list[int]:
    if len(columns) != 1:
        return []
    try:
        import pyarrow.compute as pc

        lengths = pc.list_value_length(table[columns[0]]).to_pylist()
    except (ImportError, TypeError, ValueError):
        lengths = [
            None if cell is None else len(cell)
            for cell in table[columns[0]].to_pylist()
        ]
    return [
        int(value)
        for value, keep in zip(lengths, mask)
        if keep and value is not None
    ]


def _registration_map_report(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str]
) -> dict[str, Any]:
    columns = _component_names(descriptors, REGISTRATION_MAP_PATH, POSITION_COMPONENT)
    query_index, table = _read_entity(
        dataset,
        REGISTRATION_MAP_PATH,
        columns,
        index_names,
        temporal=True,
        # State IDs and revisions are provenance. Sensor time is the native
        # snapshot cadence and must remain the primary dataframe index.
        preferred_indices=("sensor_time", "state_id", "log_time"),
    )
    mask = _event_mask(table, columns)
    counts = _point_counts(table, columns, mask)
    timeline = _timeline_report(table, mask)
    timeline.pop("measurement_ids")
    return {
        "path": REGISTRATION_MAP_PATH,
        "present": bool(columns),
        "query_index": query_index,
        "primary_component_columns": columns,
        "frames": len(counts),
        "total_points": sum(counts),
        "points_per_frame": _stats(counts),
        "achieved_rate_hz": timeline["mean_rate_hz"],
        "timeline": timeline,
    }


def _estimator_quality_report(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str]
) -> dict[str, Any]:
    report, _ = _scalar_report(
        dataset,
        descriptors,
        index_names,
        ESTIMATOR_QUALITY_PATH,
        ESTIMATOR_QUALITY_FIELDS,
        # Rejected candidates can reuse a would-be state ID. Sensor time keeps
        # every estimator attempt and leaves state/revision as provenance.
        preferred_indices=("sensor_time", "state_id", "log_time"),
    )
    return report


def _failure_report(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str]
) -> dict[str, Any]:
    columns = _component_names(descriptors, FAILURE_PATH, TEXT_COMPONENT)
    query_index, table = _read_entity(
        dataset, FAILURE_PATH, columns, index_names, temporal=True
    )
    mask = _event_mask(table, columns)
    messages: list[str] = []
    if len(columns) == 1:
        for cell, keep in zip(table[columns[0]].to_pylist(), mask):
            if not keep:
                continue
            values = _as_list(cell) or []
            messages.extend(value for value in values if isinstance(value, str))
    timeline = _timeline_report(table, mask)
    timeline.pop("measurement_ids")
    return {
        "path": FAILURE_PATH,
        "present": bool(columns),
        "query_index": query_index,
        "primary_component_columns": columns,
        "rows": sum(mask),
        "text_values": len(messages),
        "latest_messages": messages[-20:],
        "timeline": timeline,
    }


def _shutdown_scalar(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str], path: str
) -> dict[str, Any]:
    columns = _component_names(descriptors, path, SCALAR_COMPONENT)
    query_index, table = _read_entity(dataset, path, columns, index_names, temporal=True)
    mask = _event_mask(table, columns)
    rows: list[list[Any]] = []
    if len(columns) == 1:
        rows = [
            normalized
            for normalized, keep in zip(
                (_as_list(cell) for cell in table[columns[0]].to_pylist()), mask
            )
            if keep and normalized is not None
        ]
    value: Optional[int] = None
    if len(rows) == 1 and len(rows[0]) == 1:
        raw = rows[0][0]
        if (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
            and float(raw) >= 0.0
            and float(raw).is_integer()
        ):
            value = int(raw)
    timeline = _timeline_report(table, mask)
    timeline.pop("measurement_ids")
    return {
        "path": path,
        "present": bool(columns),
        "query_index": query_index,
        "primary_component_columns": columns,
        "rows": len(rows),
        "widths": [len(row) for row in rows],
        "value": value,
        "timeline": timeline,
    }


def _component_records(
    dataset: Any,
    descriptors: Sequence[Any],
    index_names: set[str],
    path: str,
    component_type: str,
    preferred_indices: Sequence[str],
) -> tuple[bool, Any, list[bool], list[list[Any]]]:
    columns = _component_names(descriptors, path, component_type)
    _, table = _read_entity(
        dataset,
        path,
        columns,
        index_names,
        temporal=True,
        preferred_indices=preferred_indices,
    )
    mask = _event_mask(table, columns)
    rows: list[list[Any]] = []
    if len(columns) == 1:
        rows = [
            normalized
            for normalized, keep in zip(
                (_as_list(cell) for cell in table[columns[0]].to_pylist()), mask
            )
            if keep and normalized is not None
        ]
    return bool(columns), table, mask, rows


def _normalized_number(value: Any) -> Optional[int | float]:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _final_fields(row: Optional[Sequence[Any]], fields: Sequence[str]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return {
        field: _normalized_number(row[index]) if index < len(row) else None
        for index, field in enumerate(fields)
    }


def _parse_initialization_status(value: str) -> dict[str, Optional[str]]:
    match = re.fullmatch(
        r"mode=(?P<mode>\S+)\s+status=(?P<status>\S+)(?:\s+reason=(?P<reason>.*))?",
        value,
    )
    return {
        "raw": value,
        "mode": match.group("mode") if match else None,
        "status": match.group("status") if match else None,
        "reason": match.group("reason") if match else None,
    }


def _local_rt_report(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str]
) -> dict[str, Any]:
    status_present, _, _, status_rows = _component_records(
        dataset,
        descriptors,
        index_names,
        INITIALIZATION_STATUS_PATH,
        TEXT_COMPONENT,
        ("sensor_time", "log_time"),
    )
    status_values = [value for row in status_rows for value in row if isinstance(value, str)]
    counts_present, _, _, count_rows = _component_records(
        dataset,
        descriptors,
        index_names,
        INITIALIZATION_COUNTS_PATH,
        SCALAR_COMPONENT,
        ("sensor_time", "log_time"),
    )
    seed_path_present, _, _, seed_rows = _component_records(
        dataset,
        descriptors,
        index_names,
        ACCEPTED_SEED_PATH,
        SCALAR_COMPONENT,
        ("state_id", "sensor_time", "log_time"),
    )
    seed_present = seed_path_present and bool(seed_rows)
    final_status = _parse_initialization_status(status_values[-1]) if status_values else None
    initialization = {
        "present": status_present or counts_present or seed_present,
        "records": len(status_rows),
        "final": final_status,
        "accepted_seed_present": seed_present,
        "final_counts": _final_fields(
            count_rows[-1] if count_rows else None, INITIALIZATION_COUNT_FIELDS
        ),
    }

    bootstrap_present, _, _, bootstrap_rows = _component_records(
        dataset,
        descriptors,
        index_names,
        BOOTSTRAP_QUALITY_PATH,
        SCALAR_COMPONENT,
        ("measurement_id", "sensor_time", "log_time"),
    )
    accepted_bootstrap = sum(
        len(row) > 4
        and isinstance(row[4], (int, float))
        and not isinstance(row[4], bool)
        and math.isfinite(float(row[4]))
        and float(row[4]) > 0.5
        for row in bootstrap_rows
    )
    bootstrap_summary = {
        "present": bootstrap_present,
        "records": len(bootstrap_rows),
        "accepted_records": accepted_bootstrap,
        "rejected_records": len(bootstrap_rows) - accepted_bootstrap,
    }

    preintegration_present, preintegration_table, preintegration_mask, preintegration_rows = (
        _component_records(
            dataset,
            descriptors,
            index_names,
            PREINTEGRATION_QUALITY_PATH,
            SCALAR_COMPONENT,
            ("state_id", "sensor_time", "log_time"),
        )
    )
    backend_present, _, _, backend_rows = _component_records(
        dataset,
        descriptors,
        index_names,
        PREINTEGRATION_BACKEND_PATH,
        TEXT_COMPONENT,
        ("state_id", "sensor_time", "log_time"),
    )
    backend_values = [value for row in backend_rows for value in row if isinstance(value, str)]
    durations_ns = [
        float(row[2])
        for row in preintegration_rows
        if len(row) > 2
        and isinstance(row[2], (int, float))
        and not isinstance(row[2], bool)
        and math.isfinite(float(row[2]))
    ]
    end_times_ns = _timeline_values(preintegration_table, "sensor_time", preintegration_mask)
    first_begin_ns: Optional[int] = None
    if preintegration_rows and end_times_ns and len(preintegration_rows[0]) > 2:
        first_duration = _normalized_number(preintegration_rows[0][2])
        if isinstance(first_duration, int):
            first_begin_ns = end_times_ns[0] - first_duration
    state_ids = _timeline_values(preintegration_table, "state_id", preintegration_mask)
    preintegration_summary = {
        "present": preintegration_present or backend_present,
        "records": len(preintegration_rows),
        "state_id_first": state_ids[0] if state_ids else None,
        "state_id_last": state_ids[-1] if state_ids else None,
        "state_id_unique": len(set(state_ids)),
        "backend": backend_values[-1] if backend_values else None,
        "backends": list(dict.fromkeys(backend_values)),
        "support": {
            "first_begin_ns": first_begin_ns,
            "last_end_ns": end_times_ns[-1] if end_times_ns else None,
            "total_duration_s": sum(durations_ns) / 1.0e9 if durations_ns else None,
            "duration_ns": _stats(
                row[2] for row in preintegration_rows if len(row) > 2
            ),
        },
        "source_sample_count": _stats(
            row[3] for row in preintegration_rows if len(row) > 3
        ),
        "integration_segment_count": _stats(
            row[4] for row in preintegration_rows if len(row) > 4
        ),
        "maximum_source_gap_ns": _stats(
            row[5] for row in preintegration_rows if len(row) > 5
        ),
    }

    return {
        "present": initialization["present"]
        or bootstrap_summary["present"]
        or preintegration_summary["present"],
        "initialization": initialization,
        "bootstrap": bootstrap_summary,
        "preintegration": preintegration_summary,
    }


def _timing_report(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str]
) -> dict[str, Any]:
    paths = sorted(
        {
            descriptor.entity_path
            for descriptor in descriptors
            if not descriptor.is_property
            and descriptor.component_type == SCALAR_COMPONENT
            and descriptor.entity_path.startswith(TIMING_PREFIX)
            and descriptor.entity_path != TIMING_PREFIX
        }
    )
    stages: dict[str, Any] = {}
    for path in paths:
        stage = path[len(TIMING_PREFIX) :]
        columns = _component_names(descriptors, path, SCALAR_COMPONENT)
        query_index, table = _read_entity(
            dataset,
            path,
            columns,
            index_names,
            temporal=True,
            # A rejected candidate deliberately reuses its would-be StateId on
            # the next sweep. Rerun's dataframe query keeps only the latest row
            # for duplicate values of the selected index, so state_id would
            # silently discard timing attempts. Sensor time is per-sweep and
            # preserves every attempt; state_id remains available as provenance.
            preferred_indices=("sensor_time", "state_id", "measurement_id", "log_time"),
        )
        mask = _event_mask(table, columns)
        rows: list[list[Any]] = []
        if len(columns) == 1:
            rows = [
                normalized
                for normalized, keep in zip(
                    (_as_list(cell) for cell in table[columns[0]].to_pylist()), mask
                )
                if keep and normalized is not None
            ]
        durations: list[float] = []
        invalid_rows = 0
        for row in rows:
            value = row[0] if len(row) == 1 else None
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                invalid_rows += 1
                continue
            durations.append(float(value))
        state_ids = _timeline_values(table, "state_id", mask)
        measurement_ids = _timeline_values(table, "measurement_id", mask)
        sensor_times = _timeline_values(table, "sensor_time", mask)
        stages[stage] = {
            "path": path,
            "query_index": query_index,
            "primary_component_columns": columns,
            "records": len(rows),
            "valid_duration_rows": len(durations),
            "invalid_rows": invalid_rows,
            "duration_ns": _stats(durations),
            "timeline": {
                "state_id_rows": len(state_ids),
                "state_id_unique": len(set(state_ids)),
                "measurement_id_rows": len(measurement_ids),
                "measurement_id_unique": len(set(measurement_ids)),
                "sensor_time_rows": len(sensor_times),
                "sensor_time_first_ns": sensor_times[0] if sensor_times else None,
                "sensor_time_last_ns": sensor_times[-1] if sensor_times else None,
            },
        }
    return {"present": bool(stages), "stage_count": len(stages), "stages": stages}


def _trajectory_eval_module() -> Any:
    try:
        from tools import trajectory_eval
    except ModuleNotFoundError as exc:
        if exc.name != "tools":
            raise
        import trajectory_eval  # type: ignore[no-redef]

    return trajectory_eval


def _trajectory_track_report(
    dataset: Any,
    descriptors: Sequence[Any],
    index_names: set[str],
    label: str,
    path: str,
) -> tuple[dict[str, Any], Optional[Any]]:
    columns = _component_names(descriptors, path, SCALAR_COMPONENT)
    query_index, table = _read_entity(
        dataset,
        path,
        columns,
        index_names,
        temporal=True,
        preferred_indices=("state_id", "sensor_time", "log_time"),
    )
    mask = _event_mask(table, columns)
    rows: list[list[Any]] = []
    if len(columns) == 1:
        rows = [
            normalized
            for normalized, keep in zip(
                (_as_list(cell) for cell in table[columns[0]].to_pylist()), mask
            )
            if keep and normalized is not None
        ]
    sensor_times = _timeline_values(table, "sensor_time", mask)
    state_ids = _timeline_values(table, "state_id", mask)
    revisions = _timeline_values(table, "estimator_revision", mask)
    widths = [len(row) for row in rows]
    invalid_values = sum(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for row in rows
        for value in row
    )
    errors: list[str] = []
    if columns and len(columns) != 1:
        errors.append(f"expected one Scalar component column, found {len(columns)}")
    if columns and not rows:
        errors.append("trajectory entity has no pose rows")
    if rows and any(width != len(TRAJECTORY_FIELDS) for width in widths):
        errors.append(
            f"pose rows must have width {len(TRAJECTORY_FIELDS)}, observed "
            f"{min(widths)}..{max(widths)}"
        )
    if invalid_values:
        errors.append(f"pose rows contain {invalid_values} non-finite/non-numeric value(s)")
    if rows and len(sensor_times) != len(rows):
        errors.append(f"sensor_time rows {len(sensor_times)} != pose rows {len(rows)}")
    if rows and len(state_ids) != len(rows):
        errors.append(f"state_id rows {len(state_ids)} != pose rows {len(rows)}")
    if rows and len(revisions) != len(rows):
        errors.append(f"estimator_revision rows {len(revisions)} != pose rows {len(rows)}")
    time_steps = [current - previous for previous, current in zip(sensor_times, sensor_times[1:])]
    if any(step <= 0 for step in time_steps):
        errors.append("sensor_time is not strictly increasing")
    state_steps = [current - previous for previous, current in zip(state_ids, state_ids[1:])]
    if any(step <= 0 for step in state_steps):
        errors.append("state_id is not strictly increasing")
    if len(set(state_ids)) != len(state_ids):
        errors.append("state_id contains duplicates")

    gap_threshold_ns = 500_000_000
    gap_indices = [index for index, step in enumerate(time_steps) if step > gap_threshold_ns]
    gap_candidates = [
        {
            "before_state_id": state_ids[index] if index < len(state_ids) else None,
            "after_state_id": state_ids[index + 1] if index + 1 < len(state_ids) else None,
            "before_timestamp_ns": sensor_times[index],
            "after_timestamp_ns": sensor_times[index + 1],
            "gap_ns": time_steps[index],
            "gap_s": time_steps[index] / 1.0e9,
        }
        for index in gap_indices[:20]
    ]
    timeline = {
        "sensor_time_rows": len(sensor_times),
        "sensor_time_first_ns": sensor_times[0] if sensor_times else None,
        "sensor_time_last_ns": sensor_times[-1] if sensor_times else None,
        "duration_s": (
            (sensor_times[-1] - sensor_times[0]) / 1.0e9
            if len(sensor_times) >= 2
            else 0.0 if sensor_times else None
        ),
        "timestamp_gap_s": _stats(step / 1.0e9 for step in time_steps),
        "nonpositive_timestamp_gaps": sum(step <= 0 for step in time_steps),
        "gap_threshold_s": gap_threshold_ns / 1.0e9,
        "gap_count": len(gap_indices),
        "gap_candidates": gap_candidates,
        "state_id_rows": len(state_ids),
        "state_id_first": state_ids[0] if state_ids else None,
        "state_id_last": state_ids[-1] if state_ids else None,
        "state_id_unique": len(set(state_ids)),
        "estimator_revision_rows": len(revisions),
        "estimator_revision_first": revisions[0] if revisions else None,
        "estimator_revision_last": revisions[-1] if revisions else None,
    }

    trajectory = None
    if columns and rows and not errors:
        trajectory_eval = _trajectory_eval_module()
        import numpy as np

        matrix = np.asarray(rows, dtype=np.float64)
        quaternion_norms = np.linalg.norm(matrix[:, 3:7], axis=1)
        maximum_norm_error = float(np.max(np.abs(quaternion_norms - 1.0)))
        if maximum_norm_error > 1.0e-6:
            errors.append(
                f"quaternion norm error {maximum_norm_error:.6g} exceeds 1e-6"
            )
        else:
            try:
                timestamps = np.asarray(sensor_times, dtype=np.longdouble) / np.longdouble(1.0e9)
                trajectory = trajectory_eval.Trajectory(
                    timestamps=timestamps,
                    positions=matrix[:, 0:3],
                    quaternions_xyzw=matrix[:, 3:7],
                )
            except trajectory_eval.EvaluationError as exc:
                errors.append(str(exc))

    return (
        {
            "label": label,
            "path": path,
            "present": bool(columns),
            "valid": bool(columns) and bool(rows) and not errors,
            "query_index": query_index,
            "primary_component_columns": columns,
            "rows": len(rows),
            "expected_width": len(TRAJECTORY_FIELDS),
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "invalid_values": invalid_values,
            "fields": list(TRAJECTORY_FIELDS),
            "timeline": timeline,
            "errors": errors,
        },
        trajectory,
    )


def _trajectory_report(
    dataset: Any, descriptors: Sequence[Any], index_names: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    tracks: dict[str, Any] = {}
    trajectories: dict[str, Any] = {}
    for label, path in TRAJECTORY_PATHS.items():
        track_report, trajectory = _trajectory_track_report(
            dataset, descriptors, index_names, label, path
        )
        tracks[label] = track_report
        if trajectory is not None:
            trajectories[label] = trajectory
    return (
        {"present": any(track["present"] for track in tracks.values()), "tracks": tracks},
        trajectories,
    )


def _load_ground_truth_state(path: Path) -> tuple[Any, dict[str, Any]]:
    trajectory_eval = _trajectory_eval_module()
    import numpy as np

    source = path.resolve()
    timestamps_ns: list[int] = []
    poses: list[list[float]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            content = raw_line.split("#", 1)[0].strip()
            if not content:
                continue
            fields = content.replace(",", " ").split()
            if len(fields) != 9:
                raise ValueError(
                    f"{source}:{line_number}: expected 9 state fields, got {len(fields)}"
                )
            try:
                seconds = int(fields[0])
                nanoseconds = int(fields[1])
                pose = [float(field) for field in fields[2:]]
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: non-numeric state field") from exc
            if not 0 <= nanoseconds < 1_000_000_000:
                raise ValueError(f"{source}:{line_number}: nsec is outside [0, 1e9)")
            timestamp_ns = seconds * 1_000_000_000 + nanoseconds
            if timestamps_ns and timestamp_ns <= timestamps_ns[-1]:
                failure = (
                    "duplicate timestamp"
                    if timestamp_ns == timestamps_ns[-1]
                    else "timestamp reset"
                )
                raise ValueError(f"{source}:{line_number}: {failure}")
            if not all(math.isfinite(value) for value in pose):
                raise ValueError(f"{source}:{line_number}: NaN or infinity")
            timestamps_ns.append(timestamp_ns)
            poses.append(pose)
    if not poses:
        raise ValueError(f"{source}: no ground-truth poses")

    matrix = np.asarray(poses, dtype=np.float64)
    quaternions = matrix[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1.0e-12):
        row = int(np.flatnonzero(norms < 1.0e-12)[0]) + 1
        raise ValueError(f"{source}: pose row {row} has a zero quaternion")
    norm_error = np.abs(norms - 1.0)
    if np.any(norm_error > 1.0e-2):
        row = int(np.argmax(norm_error)) + 1
        raise ValueError(
            f"{source}: pose row {row} quaternion norm error {norm_error[row - 1]:.6g} "
            "exceeds 0.01"
        )
    quaternions = quaternions / norms[:, None]
    timestamps = np.asarray(timestamps_ns, dtype=np.longdouble) / np.longdouble(1.0e9)
    trajectory = trajectory_eval.Trajectory(timestamps, matrix[:, 0:3], quaternions)
    return trajectory, {
        "path": str(source),
        "format": "sec_nsec_xyz_quaternion_xyzw",
        "pose_semantics": "T_world_base",
        "body_frame": "base_link",
        "pose_count": len(poses),
        "first_timestamp_ns": timestamps_ns[0],
        "last_timestamp_ns": timestamps_ns[-1],
    }


def _evaluate_trajectory_tracks(
    ground_truth_state: Path, trajectories: Mapping[str, Any]
) -> dict[str, Any]:
    trajectory_eval = _trajectory_eval_module()
    reference, ground_truth_report = _load_ground_truth_state(ground_truth_state)
    tracks: dict[str, Any] = {}
    for label, estimate in trajectories.items():
        config = trajectory_eval.EvaluationConfig(
            reference_body_frame="base_link",
            estimate_body_frame="base_link",
            association="nearest",
            max_dt=0.01,
            exact_tolerance=1.0e-9,
            max_interpolation_gap=0.20,
            min_coverage=0.90,
            max_output_gap=0.50,
            max_linear_speed=30.0,
            max_angular_speed_deg=720.0,
            rpe_distance_m=10.0,
            track_label=f"local-{label}",
        )
        try:
            tracks[label] = {
                "evaluated": True,
                "error": None,
                "report": trajectory_eval.evaluate_trajectories(reference, estimate, config),
            }
        except trajectory_eval.EvaluationError as exc:
            tracks[label] = {"evaluated": False, "error": str(exc), "report": None}
    return {
        "requested": True,
        "protocol": "newer_college_nearest_10ms_umeyama_se3_no_scale",
        "ground_truth": ground_truth_report,
        "evaluated_track_count": sum(track["evaluated"] for track in tracks.values()),
        "tracks": tracks,
    }


def _add_check(
    checks: list[Check],
    name: str,
    passed: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    checks.append(Check(name=name, passed=passed, detail=detail, required=required))


def _build_checks(report: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    verification = report["rrd"]["verification"]
    _add_check(checks, "RRD footer and manifest", verification["passed"], verification["detail"])

    for name, expected_width in (("imu", len(IMU_FIELDS)), ("lidar", len(LIDAR_FIELDS))):
        sensor = report[name]
        scalar_valid = (
            sensor["present"]
            and len(sensor["primary_component_columns"]) == 1
            and sensor["rows"] > 0
            and sensor["width_min"] == expected_width
            and sensor["width_max"] == expected_width
            and sensor["invalid_values"] == 0
        )
        _add_check(
            checks,
            f"{name.upper()} Scalars structure",
            scalar_valid,
            f"rows={sensor['rows']}, columns={len(sensor['primary_component_columns'])}, "
            f"width={sensor['width_min']}..{sensor['width_max']}, invalid={sensor['invalid_values']}",
        )
        timeline = sensor["timeline"]
        timeline_valid = (
            sensor["rows"] > 0
            and timeline["sensor_time_rows"] == sensor["rows"]
            and timeline["measurement_id_rows"] == sensor["rows"]
            and timeline["measurement_id_unique"] == sensor["rows"]
        )
        _add_check(
            checks,
            f"{name.upper()} measurement timelines",
            timeline_valid,
            f"rows={sensor['rows']}, sensor_time={timeline['sensor_time_rows']}, "
            f"measurement_id={timeline['measurement_id_rows']}, "
            f"unique_ids={timeline['measurement_id_unique']}",
        )

    preview = report["preview"]
    preview_valid = (
        preview["present"]
        and len(preview["primary_component_columns"]) == 1
        and preview["frames"] > 0
    )
    _add_check(
        checks,
        "LiDAR preview Points3D structure",
        preview_valid,
        f"frames={preview['frames']}, columns={len(preview['primary_component_columns'])}, "
        f"max_points={preview['max_points']}",
    )
    preview_timeline = preview["timeline"]
    _add_check(
        checks,
        "LiDAR preview measurement timelines",
        preview["frames"] > 0
        and preview_timeline["sensor_time_rows"] == preview["frames"]
        and preview_timeline["measurement_id_rows"] == preview["frames"]
        and preview_timeline["measurement_id_unique"] == preview["frames"],
        f"frames={preview['frames']}, sensor_time={preview_timeline['sensor_time_rows']}, "
        f"measurement_id={preview_timeline['measurement_id_rows']}, "
        f"unique_ids={preview_timeline['measurement_id_unique']}",
    )
    config = report.get("config")
    if config is not None:
        configured_rate = config["preview_rate_hz"]
        configured_cap = config["preview_max_points"]
        observed_rate = preview_timeline["mean_rate_hz"]
        _add_check(
            checks,
            "LiDAR preview configuration",
            preview["max_points"] is not None
            and preview["max_points"] <= configured_cap
            and (observed_rate is None or observed_rate <= configured_rate * 1.01),
            f"configured_rate={configured_rate} Hz, observed_rate={observed_rate}, "
            f"configured_cap={configured_cap}, observed_max={preview['max_points']}",
        )

    registration_map = report["registration_map"]
    if registration_map["present"]:
        map_timeline = registration_map["timeline"]
        _add_check(
            checks,
            "registration map Points3D structure",
            len(registration_map["primary_component_columns"]) == 1
            and registration_map["frames"] > 0
            and registration_map["points_per_frame"] is not None,
            f"frames={registration_map['frames']}, "
            f"columns={len(registration_map['primary_component_columns'])}, "
            f"points={registration_map['total_points']}",
        )
        _add_check(
            checks,
            "registration map native snapshot timelines",
            registration_map["query_index"] == "sensor_time"
            and map_timeline["sensor_time_rows"] == registration_map["frames"]
            and map_timeline["state_id_rows"] == registration_map["frames"]
            and map_timeline["estimator_revision_rows"] == registration_map["frames"]
            and map_timeline["nonpositive_timestamp_gaps"] == 0,
            f"index={registration_map['query_index']}, frames={registration_map['frames']}, "
            f"sensor_time={map_timeline['sensor_time_rows']}, "
            f"state_id={map_timeline['state_id_rows']}, "
            f"revision={map_timeline['estimator_revision_rows']}",
        )

    estimator_quality = report["estimator_quality"]
    if estimator_quality["present"]:
        _add_check(
            checks,
            "local estimator quality structure",
            len(estimator_quality["primary_component_columns"]) == 1
            and estimator_quality["rows"] > 0
            and estimator_quality["width_min"] is not None
            and estimator_quality["width_min"] >= ESTIMATOR_QUALITY_BASE_FIELD_COUNT
            and estimator_quality["width_max"] is not None
            and estimator_quality["invalid_values"] == 0,
            f"rows={estimator_quality['rows']}, "
            f"width={estimator_quality['width_min']}..{estimator_quality['width_max']}, "
            f"invalid={estimator_quality['invalid_values']}",
        )

    failures = report["ingress_failures"]
    failure_valid = (
        not failures["present"]
        or (
            len(failures["primary_component_columns"]) == 1
            and failures["rows"] > 0
            and failures["text_values"] == failures["rows"]
            and failures["timeline"]["measurement_id_rows"] == failures["rows"]
        )
    )
    _add_check(
        checks,
        "ingress failure TextLog structure",
        failure_valid,
        f"rows={failures['rows']}, text={failures['text_values']}, "
        f"measurement_id={failures['timeline']['measurement_id_rows']}",
    )

    shutdown = report["shutdown"]
    for name in ("debug_events_dropped", "debug_log_errors"):
        counter = shutdown[name]
        structural = (
            counter["present"]
            and len(counter["primary_component_columns"]) == 1
            and counter["rows"] == 1
            and counter["widths"] == [1]
            and counter["value"] is not None
            and counter["timeline"]["sensor_time_rows"] == 0
            and counter["timeline"]["measurement_id_rows"] == 0
        )
        _add_check(
            checks,
            f"shutdown {name} structure",
            structural,
            f"rows={counter['rows']}, widths={counter['widths']}, value={counter['value']}, "
            f"sensor_time={counter['timeline']['sensor_time_rows']}, "
            f"measurement_id={counter['timeline']['measurement_id_rows']}",
        )
        _add_check(
            checks,
            f"shutdown {name} is zero",
            counter["value"] == 0,
            f"value={counter['value']}",
        )

    bag = report.get("bag")
    if bag is not None:
        for name in ("imu", "lidar"):
            expected = bag["selected_topics"][name]["messages"]
            actual = report[name]["rows"]
            topic = bag["selected_topics"][name]["topic"]
            _add_check(
                checks,
                f"bag/{name.upper()} accepted row count (advisory)",
                expected is not None and actual == expected,
                f"topic={topic}, bag={expected}, rrd={actual}",
                required=False,
            )

    timing = report["timing"]
    for stage, stage_report in timing["stages"].items():
        timing_valid = (
            len(stage_report["primary_component_columns"]) == 1
            and stage_report["records"] > 0
            and stage_report["valid_duration_rows"] == stage_report["records"]
            and stage_report["invalid_rows"] == 0
            and stage_report["timeline"]["sensor_time_rows"] == stage_report["records"]
            and (
                stage_report["timeline"]["state_id_rows"] == stage_report["records"]
                or stage_report["timeline"]["measurement_id_rows"]
                == stage_report["records"]
            )
        )
        _add_check(
            checks,
            f"timing/{stage} duration structure",
            timing_valid,
            f"rows={stage_report['records']}, valid={stage_report['valid_duration_rows']}, "
            f"invalid={stage_report['invalid_rows']}, "
            f"columns={len(stage_report['primary_component_columns'])}",
        )

    trajectories = report["trajectories"]
    for label, track in trajectories["tracks"].items():
        if not track["present"]:
            continue
        _add_check(
            checks,
            f"trajectory/{label} pose structure",
            track["valid"],
            f"rows={track['rows']}, width={track['width_min']}..{track['width_max']}, "
            f"errors={track['errors']}",
        )

    evaluation = report.get("trajectory_evaluation")
    if evaluation is not None:
        _add_check(
            checks,
            "ground-truth trajectory evaluation produced a track",
            evaluation["evaluated_track_count"] > 0,
            f"evaluated_tracks={evaluation['evaluated_track_count']}",
        )
        for label, evaluated in evaluation["tracks"].items():
            evaluation_valid = evaluated["evaluated"] and evaluated["report"] is not None
            _add_check(
                checks,
                f"trajectory/{label} metric evaluation",
                evaluation_valid,
                "ok" if evaluation_valid else str(evaluated["error"]),
            )
            if evaluation_valid:
                quality = evaluated["report"]["quality"]
                _add_check(
                    checks,
                    f"trajectory/{label} metric quality",
                    quality["passed"],
                    "passed" if quality["passed"] else "; ".join(quality["failures"]),
                )
    return checks


def analyze(
    rrd: Path,
    bag: Optional[Path],
    config: Optional[Path],
    ground_truth_state: Optional[Path] = None,
) -> dict[str, Any]:
    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError("Rerun with data-platform support is required") from exc

    rrd_path = rrd.resolve()
    if not rrd_path.is_file():
        raise ValueError(f"RRD does not exist or is not a regular file: {rrd_path}")
    if rrd_path.stat().st_size == 0:
        raise ValueError(f"RRD is empty: {rrd_path}")
    verified, verify_detail = _verify_rrd(rrd_path)

    config_report = _config_report(config) if config is not None else None
    bag_report = _bag_report(bag, config_report) if bag is not None else None

    with rr.server.Server(datasets={"meridian_run": [rrd_path]}) as server:
        dataset = server.client().get_dataset("meridian_run")
        schema = dataset.schema()
        descriptors = list(schema.component_columns())
        index_names = {descriptor.name for descriptor in schema.index_columns()}
        imu, _ = _scalar_report(dataset, descriptors, index_names, IMU_PATH, IMU_FIELDS)
        lidar, lidar_ids = _scalar_report(
            dataset, descriptors, index_names, LIDAR_PATH, LIDAR_FIELDS
        )
        preview, preview_ids = _point_report(dataset, descriptors, index_names)
        registration_map = _registration_map_report(dataset, descriptors, index_names)
        estimator_quality = _estimator_quality_report(
            dataset, descriptors, index_names
        )
        failures = _failure_report(dataset, descriptors, index_names)
        debug_drops = _shutdown_scalar(
            dataset, descriptors, index_names, DEBUG_DROPS_PATH
        )
        debug_errors = _shutdown_scalar(
            dataset, descriptors, index_names, DEBUG_ERRORS_PATH
        )
        local_rt = _local_rt_report(dataset, descriptors, index_names)
        timing = _timing_report(dataset, descriptors, index_names)
        trajectories, trajectory_values = _trajectory_report(
            dataset, descriptors, index_names
        )
    trajectory_evaluation = (
        _evaluate_trajectory_tracks(ground_truth_state, trajectory_values)
        if ground_truth_state is not None
        else None
    )
    rrd_size = rrd_path.stat().st_size
    report: dict[str, Any] = {
        "schema": "meridian.ingress_rrd_analysis/v1",
        "rrd": {
            "path": str(rrd_path),
            "size_bytes": rrd_size,
            "size_mib": rrd_size / (1024.0 * 1024.0),
            "verification": {"passed": verified, "detail": verify_detail},
        },
        "config": config_report,
        "bag": bag_report,
        "imu": imu,
        "lidar": lidar,
        "preview": preview,
        "registration_map": registration_map,
        "estimator_quality": estimator_quality,
        "ingress_failures": failures,
        "preview_ids_missing_from_lidar": len(preview_ids - lidar_ids),
        "shutdown": {
            "debug_events_dropped": debug_drops,
            "debug_log_errors": debug_errors,
        },
        "local_rt": local_rt,
        "timing": timing,
        "trajectories": trajectories,
        "trajectory_evaluation": trajectory_evaluation,
    }
    if bag_report is not None:
        bag_size = bag_report["size_bytes"]
        report["rrd"]["to_bag_size_ratio"] = rrd_size / bag_size if bag_size else None
    checks = _build_checks(report)
    _add_check(
        checks,
        "preview measurement IDs refer to accepted LiDAR rows",
        report["preview_ids_missing_from_lidar"] == 0,
        f"missing={report['preview_ids_missing_from_lidar']}",
    )
    report["checks"] = [asdict(check) for check in checks]
    report["passed"] = all(check.passed or not check.required for check in checks)
    return report


def _format(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}{suffix}"
    return f"{value}{suffix}"


def _stat_value(stats: Optional[Mapping[str, Any]], key: str, suffix: str = "") -> str:
    return "n/a" if stats is None else _format(stats.get(key), suffix)


def _print_report(report: Mapping[str, Any]) -> None:
    rrd = report["rrd"]
    print("Result:", "PASS" if report["passed"] else "FAIL")
    print(f"RRD: {rrd['path']}")
    print(f"Size: {rrd['size_bytes']} bytes ({rrd['size_mib']:.3f} MiB)")
    print(f"Verify: {'PASS' if rrd['verification']['passed'] else 'FAIL'} - "
          f"{rrd['verification']['detail']}")
    bag = report.get("bag")
    if bag is not None:
        ratio = rrd.get("to_bag_size_ratio")
        print(f"Bag: {bag['path']} ({bag['size_bytes']} bytes; counts: {bag['count_source']})")
        print(f"RRD/bag size: {_format(None if ratio is None else 100.0 * ratio, '%')}")

    for name in ("imu", "lidar"):
        sensor = report[name]
        timeline = sensor["timeline"]
        gaps = timeline["timestamp_gap_s"]
        conversion = sensor["field_stats"].get("conversion_ns")
        print()
        print(f"{name.upper()}: {sensor['rows']} accepted rows")
        print(
            "  sensor time: "
            f"{timeline['sensor_time_first_ns']} .. {timeline['sensor_time_last_ns']} ns; "
            f"duration={_format(timeline['duration_s'], ' s')}; "
            f"rate={_format(timeline['mean_rate_hz'], ' Hz')}"
        )
        print(
            "  timestamp gaps: "
            f"median={_stat_value(gaps, 'median', ' s')}; "
            f"p95={_stat_value(gaps, 'p95', ' s')}; "
            f"max={_stat_value(gaps, 'max', ' s')}; "
            f"nonpositive={timeline['nonpositive_timestamp_gaps']}"
        )
        print(
            "  conversion: "
            f"mean={_stat_value(conversion, 'mean', ' ns')}; "
            f"p95={_stat_value(conversion, 'p95', ' ns')}; "
            f"max={_stat_value(conversion, 'max', ' ns')}"
        )

    lidar_fields = report["lidar"]["field_stats"]
    print("  point stats:")
    for field in (
        "source_points",
        "accepted_points",
        "nonfinite_xyz_points",
        "zero_xyz_points",
        "flattened_time_regressions",
    ):
        stats = lidar_fields.get(field)
        print(
            f"    {field}: mean={_stat_value(stats, 'mean')}; "
            f"p95={_stat_value(stats, 'p95')}; max={_stat_value(stats, 'max')}"
        )

    preview = report["preview"]
    preview_timeline = preview["timeline"]
    print()
    print(
        f"PREVIEW: {preview['frames']} frames; total_points={preview['total_points']}; "
        f"max_points={preview['max_points']}; "
        f"rate={_format(preview_timeline['mean_rate_hz'], ' Hz')}"
    )
    print(
        "  points/frame: "
        f"mean={_stat_value(preview['points_per_frame'], 'mean')}; "
        f"p95={_stat_value(preview['points_per_frame'], 'p95')}"
    )

    registration_map = report["registration_map"]
    map_timeline = registration_map["timeline"]
    map_points = registration_map["points_per_frame"]
    print()
    if not registration_map["present"]:
        print("REGISTRATION MAP: not recorded")
    else:
        print(
            f"REGISTRATION MAP: {registration_map['frames']} snapshots; "
            f"time={_format(map_timeline['sensor_time_first_ns'])} .. "
            f"{_format(map_timeline['sensor_time_last_ns'])} ns; "
            f"rate={_format(registration_map['achieved_rate_hz'], ' Hz')}"
        )
        print(
            "  points/snapshot: "
            f"min={_stat_value(map_points, 'min')}; "
            f"mean={_stat_value(map_points, 'mean')}; "
            f"max={_stat_value(map_points, 'max')}"
        )

    estimator_quality = report["estimator_quality"]
    print()
    if not estimator_quality["present"]:
        print("ESTIMATOR QUALITY: not recorded")
    else:
        print(
            f"ESTIMATOR QUALITY: {estimator_quality['rows']} records; "
            f"width={estimator_quality['width_min']}..{estimator_quality['width_max']}"
        )
        for field in ESTIMATOR_QUALITY_APPENDED_FIELDS:
            stats = estimator_quality["field_stats"][field]
            print(
                f"  {field}: mean={_stat_value(stats, 'mean')}; "
                f"p95={_stat_value(stats, 'p95')}; "
                f"max={_stat_value(stats, 'max')}"
            )

    failures = report["ingress_failures"]
    print()
    print(f"INGRESS FAILURES: {failures['rows']}")
    for message in failures["latest_messages"]:
        print(f"  {message}")
    shutdown = report["shutdown"]
    print(
        "Shutdown: "
        f"debug_events_dropped={shutdown['debug_events_dropped']['value']}, "
        f"debug_log_errors={shutdown['debug_log_errors']['value']}"
    )

    local_rt = report["local_rt"]
    print()
    if not local_rt["present"]:
        print("LOCAL RT: not recorded")
    else:
        initialization = local_rt["initialization"]
        final = initialization["final"] or {}
        print(
            "LOCAL RT INITIALIZATION: "
            f"{initialization['records']} records; "
            f"mode={_format(final.get('mode'))}; "
            f"status={_format(final.get('status'))}; "
            f"accepted_seed={'yes' if initialization['accepted_seed_present'] else 'no'}"
        )
        print(f"  final reason: {_format(final.get('reason'))}")
        final_counts = initialization["final_counts"] or {}
        print(
            "  final counts: "
            f"imu={_format(final_counts.get('imu_sample_count'))}; "
            f"lidar={_format(final_counts.get('lidar_sweep_count'))}; "
            f"fitted_transitions={_format(final_counts.get('fitted_transition_count'))}; "
            f"rejected_transitions={_format(final_counts.get('rejected_transition_count'))}"
        )

        bootstrap = local_rt["bootstrap"]
        print(
            "LOCAL RT BOOTSTRAP: "
            f"{bootstrap['records']} records; accepted={bootstrap['accepted_records']}; "
            f"rejected={bootstrap['rejected_records']}"
        )

        preintegration = local_rt["preintegration"]
        support = preintegration["support"]
        print(
            "LOCAL RT PREINTEGRATION: "
            f"{preintegration['records']} records; "
            f"backend={_format(preintegration['backend'])}"
        )
        print(
            "  support: "
            f"{_format(support['first_begin_ns'])} .. {_format(support['last_end_ns'])} ns; "
            f"total={_format(support['total_duration_s'], ' s')}; "
            f"mean={_stat_value(support['duration_ns'], 'mean', ' ns')}; "
            f"p95={_stat_value(support['duration_ns'], 'p95', ' ns')}"
        )
        print(
            "  source samples: "
            f"mean={_stat_value(preintegration['source_sample_count'], 'mean')}; "
            f"p95={_stat_value(preintegration['source_sample_count'], 'p95')}; "
            f"max={_stat_value(preintegration['source_sample_count'], 'max')}"
        )
        print(
            "  maximum source gap: "
            f"mean={_stat_value(preintegration['maximum_source_gap_ns'], 'mean', ' ns')}; "
            f"p95={_stat_value(preintegration['maximum_source_gap_ns'], 'p95', ' ns')}; "
            f"max={_stat_value(preintegration['maximum_source_gap_ns'], 'max', ' ns')}"
        )

    timing = report["timing"]
    print()
    if not timing["present"]:
        print("TIMING: not recorded")
    else:
        print(f"TIMING: {timing['stage_count']} stage(s)")
        for stage, stage_report in timing["stages"].items():
            duration = stage_report["duration_ns"]
            print(
                f"  {stage}: count={_stat_value(duration, 'count')}; "
                f"mean={_stat_value(duration, 'mean', ' ns')}; "
                f"median={_stat_value(duration, 'median', ' ns')}; "
                f"p95={_stat_value(duration, 'p95', ' ns')}; "
                f"p99={_stat_value(duration, 'p99', ' ns')}; "
                f"max={_stat_value(duration, 'max', ' ns')}"
            )

    trajectories = report["trajectories"]
    print()
    if not trajectories["present"]:
        print("TRAJECTORIES: not recorded")
    else:
        print("TRAJECTORIES")
        for label, track in trajectories["tracks"].items():
            if not track["present"]:
                continue
            timeline = track["timeline"]
            print(
                f"  {label}: rows={track['rows']}; valid={'yes' if track['valid'] else 'no'}; "
                f"time={_format(timeline['sensor_time_first_ns'])} .. "
                f"{_format(timeline['sensor_time_last_ns'])} ns; "
                f"gaps>{timeline['gap_threshold_s']:.3f}s={timeline['gap_count']}"
            )
            for error in track["errors"]:
                print(f"    error: {error}")

    evaluation = report.get("trajectory_evaluation")
    if evaluation is not None:
        print()
        ground_truth = evaluation["ground_truth"]
        print(
            "TRAJECTORY EVALUATION: "
            f"GT={ground_truth['path']} ({ground_truth['pose_count']} base poses)"
        )
        for label, evaluated in evaluation["tracks"].items():
            if not evaluated["evaluated"]:
                print(f"  {label}: ERROR - {evaluated['error']}")
                continue
            metrics = evaluated["report"]
            ate = metrics["ate_translation_m"]
            rpe = metrics["rpe"]
            association = metrics["association"]
            health = metrics["estimate_health"]
            print(
                f"  {label}: ATE RMSE={ate['rmse']:.6f} m; "
                f"coverage={association['reference_pose_coverage_ratio']:.3%}; "
                f"gaps={health['gap_count']}; quality="
                f"{'PASS' if metrics['quality']['passed'] else 'FAIL'}"
            )
            if rpe["pair_count"]:
                print(
                    f"    RPE {rpe['distance_m']:.3f} m: "
                    f"translation RMSE={rpe['translation_m']['rmse']:.6f} m; "
                    f"rotation RMSE={rpe['rotation_deg']['rmse']:.6f} deg"
                )

    print()
    print("Checks")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else ("INFO" if not check["required"] else "FAIL")
        print(f"  [{status}] {check['name']}: {check['detail']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.bag is not None and args.config is None:
        parser.error("--config is required when --bag is supplied")
    try:
        report = analyze(
            args.rrd,
            args.bag,
            args.config,
            ground_truth_state=args.ground_truth_state,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
