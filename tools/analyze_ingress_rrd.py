#!/usr/bin/env python3
"""Post-run integrity and size report for a Meridian ingress RRD.

The recording is inspected through Rerun's supported local catalog API. When
``--bag`` is supplied, ``--config`` is required: the ROS parameter file names
the IMU and LiDAR topics whose rosbag message counts are compared with the
accepted rows in the RRD. No dataset-specific topic or count is built in.

Examples::

    python3 tools/analyze_ingress_rrd.py out/ingress.rrd

    python3 tools/analyze_ingress_rrd.py out/ingress.rrd \
        --bag path/to/bag \
        --config path/to/ingress.yaml
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
FAILURE_PATH = "/runtime/ingress_failure"
DEBUG_DROPS_PATH = "/run/debug_events_dropped"
DEBUG_ERRORS_PATH = "/run/debug_log_errors"

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
) -> tuple[Optional[str], Any]:
    view = dataset.filter_contents(path)
    if temporal:
        last: Optional[tuple[str, Any]] = None
        for index in ("measurement_id", "sensor_time", "log_time"):
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
    }


def _scalar_report(
    dataset: Any,
    descriptors: Sequence[Any],
    index_names: set[str],
    path: str,
    fields: Sequence[str],
) -> tuple[dict[str, Any], set[int]]:
    columns = _component_names(descriptors, path, SCALAR_COMPONENT)
    query_index, table = _read_entity(dataset, path, columns, index_names, temporal=True)
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
    counts: list[int] = []
    if len(columns) == 1:
        try:
            import pyarrow.compute as pc

            lengths = pc.list_value_length(table[columns[0]]).to_pylist()
        except (ImportError, TypeError, ValueError):
            lengths = [
                None if cell is None else len(cell)
                for cell in table[columns[0]].to_pylist()
            ]
        counts = [int(value) for value, keep in zip(lengths, mask) if keep and value is not None]
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
    return checks


def analyze(rrd: Path, bag: Optional[Path], config: Optional[Path]) -> dict[str, Any]:
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
        failures = _failure_report(dataset, descriptors, index_names)
        debug_drops = _shutdown_scalar(
            dataset, descriptors, index_names, DEBUG_DROPS_PATH
        )
        debug_errors = _shutdown_scalar(
            dataset, descriptors, index_names, DEBUG_ERRORS_PATH
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
        "ingress_failures": failures,
        "preview_ids_missing_from_lidar": len(preview_ids - lidar_ids),
        "shutdown": {
            "debug_events_dropped": debug_drops,
            "debug_log_errors": debug_errors,
        },
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
        report = analyze(args.rrd, args.bag, args.config)
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
