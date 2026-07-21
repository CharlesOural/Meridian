#!/usr/bin/env python3
"""Extract a Barakuda RTK/SBG reference trajectory from a ROS 2 MCAP bag.

The recorded Barakuda bags contain position-only TF samples
``rtk_enu -> rtk_antenna`` and full SBG attitude samples on ``/imu/data``.
This tool interpolates the SBG attitude at each RTK timestamp and writes the
pose of the SBG IMU origin after removing the configured IMU-to-antenna lever
arm::

    p_world_imu = p_world_antenna - R_world_imu * r_imu_antenna

Inputs may be an extracted ``.mcap`` file, a rosbag directory containing MCAP
files, or one of the original uncompressed ``.tar`` archives.  The parser uses
only the Python standard library and reads the standard ROS messages directly;
the SBG custom message package is not required.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import math
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, Iterator, Sequence


MCAP_MAGIC = b"\x89MCAP0\r\n"
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06


class ExtractionError(RuntimeError):
    """The input bag cannot produce the requested reference trajectory."""


@dataclass(frozen=True)
class OrientationSample:
    stamp_ns: int
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class PositionSample:
    stamp_ns: int
    position_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class PoseSample:
    stamp_ns: int
    position_xyz_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass
class ExtractionStats:
    mcap_messages: int = 0
    orientation_messages: int = 0
    position_messages: int = 0
    rejected_orientations: int = 0
    unmatched_positions: int = 0


class ByteReader:
    def __init__(self, data: bytes, label: str = "record") -> None:
        self.data = data
        self.offset = 0
        self.label = label

    def _take(self, size: int) -> bytes:
        end = self.offset + size
        if size < 0 or end > len(self.data):
            raise ExtractionError(f"truncated {self.label}")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def uint8(self) -> int:
        return self._take(1)[0]

    def uint16(self) -> int:
        return struct.unpack("<H", self._take(2))[0]

    def uint32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def uint64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def string(self) -> str:
        size = self.uint32()
        try:
            return self._take(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExtractionError(f"invalid UTF-8 in {self.label}") from exc

    def byte_array64(self) -> bytes:
        return self._take(self.uint64())

    def remainder(self) -> bytes:
        return self._take(len(self.data) - self.offset)


class CdrReader:
    """Small CDR1 little-endian reader for Header, Imu, and TFMessage."""

    def __init__(self, data: bytes, label: str) -> None:
        if len(data) < 4 or data[0:2] != b"\x00\x01":
            identifier = data[0:2].hex() if len(data) >= 2 else "missing"
            raise ExtractionError(
                f"{label}: unsupported CDR encapsulation identifier {identifier}"
            )
        self.data = data
        self.origin = 4
        self.offset = 4
        self.label = label

    def _align(self, size: int) -> None:
        self.offset += (-(self.offset - self.origin)) % size

    def _unpack(self, code: str, alignment: int):
        self._align(alignment)
        size = struct.calcsize(code)
        end = self.offset + size
        if end > len(self.data):
            raise ExtractionError(f"{self.label}: truncated CDR value")
        value = struct.unpack_from("<" + code, self.data, self.offset)[0]
        self.offset = end
        return value

    def uint32(self) -> int:
        return int(self._unpack("I", 4))

    def int32(self) -> int:
        return int(self._unpack("i", 4))

    def float64(self) -> float:
        return float(self._unpack("d", 8))

    def string(self) -> str:
        size = self.uint32()
        if size == 0:
            raise ExtractionError(f"{self.label}: zero-length CDR string")
        end = self.offset + size
        if end > len(self.data):
            raise ExtractionError(f"{self.label}: truncated CDR string")
        raw = self.data[self.offset:end]
        self.offset = end
        if raw[-1] != 0:
            raise ExtractionError(f"{self.label}: CDR string is not NUL-terminated")
        try:
            return raw[:-1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExtractionError(f"{self.label}: invalid UTF-8 string") from exc


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ExtractionError(f"{label}: unexpected end of file")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _iter_records(data: bytes, label: str) -> Iterator[tuple[int, bytes]]:
    offset = 0
    while offset < len(data):
        if len(data) - offset < 9:
            raise ExtractionError(f"{label}: truncated MCAP record header")
        opcode = data[offset]
        size = struct.unpack_from("<Q", data, offset + 1)[0]
        begin = offset + 9
        end = begin + size
        if end > len(data):
            raise ExtractionError(f"{label}: truncated MCAP record payload")
        yield opcode, data[begin:end]
        offset = end


def _chunk_records(payload: bytes, label: str) -> Iterator[tuple[int, bytes]]:
    reader = ByteReader(payload, f"{label} chunk")
    reader.uint64()  # message_start_time
    reader.uint64()  # message_end_time
    expected_size = reader.uint64()
    reader.uint32()  # uncompressed CRC; the recorder leaves it disabled (zero)
    compression = reader.string()
    records = reader.byte_array64()
    if compression:
        raise ExtractionError(
            f"{label}: MCAP chunk compression '{compression}' is unsupported; "
            "these Barakuda recordings are expected to use uncompressed chunks"
        )
    if len(records) != expected_size:
        raise ExtractionError(
            f"{label}: MCAP chunk size mismatch ({len(records)} != {expected_size})"
        )
    yield from _iter_records(records, f"{label} chunk records")


@contextlib.contextmanager
def _open_inputs(path: Path) -> Iterator[list[tuple[str, BinaryIO]]]:
    streams: list[tuple[str, BinaryIO]] = []
    opened: list[BinaryIO] = []
    archive: tarfile.TarFile | None = None
    try:
        if path.is_dir():
            files = sorted(path.glob("*.mcap"))
            if not files:
                raise ExtractionError(f"{path}: directory contains no .mcap files")
            for file_path in files:
                stream = file_path.open("rb")
                opened.append(stream)
                streams.append((str(file_path), stream))
        elif path.suffix.lower() == ".tar":
            archive = tarfile.open(path, mode="r:")
            members = sorted(
                (member for member in archive.getmembers() if member.isfile() and member.name.endswith(".mcap")),
                key=lambda member: member.name,
            )
            if not members:
                raise ExtractionError(f"{path}: archive contains no .mcap files")
            for member in members:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ExtractionError(f"{path}: cannot open {member.name}")
                opened.append(stream)
                streams.append((f"{path}:{member.name}", stream))
        elif path.is_file():
            stream = path.open("rb")
            opened.append(stream)
            streams.append((str(path), stream))
        else:
            raise ExtractionError(f"{path}: input does not exist")
        yield streams
    finally:
        for stream in opened:
            stream.close()
        if archive is not None:
            archive.close()


def _parse_imu_orientation(data: bytes) -> OrientationSample:
    reader = CdrReader(data, "sensor_msgs/msg/Imu")
    seconds = reader.int32()
    nanoseconds = reader.uint32()
    reader.string()  # frame_id
    quaternion = tuple(reader.float64() for _ in range(4))
    if not 0 <= nanoseconds < 1_000_000_000:
        raise ExtractionError("sensor_msgs/msg/Imu: nanosecond stamp is out of range")
    return OrientationSample(
        seconds * 1_000_000_000 + nanoseconds,
        _normalized_quaternion(quaternion),
    )


def _parse_tf_positions(
    data: bytes, parent_frame: str, child_frame: str
) -> list[PositionSample]:
    reader = CdrReader(data, "tf2_msgs/msg/TFMessage")
    count = reader.uint32()
    if count > 10_000:
        raise ExtractionError(f"TFMessage has implausible transform count {count}")
    result: list[PositionSample] = []
    for _ in range(count):
        seconds = reader.int32()
        nanoseconds = reader.uint32()
        parent = reader.string()
        child = reader.string()
        translation = tuple(reader.float64() for _ in range(3))
        tuple(reader.float64() for _ in range(4))  # position-only TF rotation
        if parent == parent_frame and child == child_frame:
            if not 0 <= nanoseconds < 1_000_000_000:
                raise ExtractionError("TFMessage nanosecond stamp is out of range")
            if not all(math.isfinite(value) for value in translation):
                raise ExtractionError("TFMessage position contains NaN or infinity")
            result.append(
                PositionSample(
                    seconds * 1_000_000_000 + nanoseconds,
                    translation,
                )
            )
    return result


def _normalized_quaternion(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ExtractionError("orientation quaternion contains NaN or infinity")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1.0e-12:
        raise ExtractionError("orientation quaternion has zero norm")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _parse_mcap_stream(
    stream: BinaryIO,
    label: str,
    orientation_topic: str,
    position_topic: str,
    parent_frame: str,
    child_frame: str,
    orientations: list[OrientationSample],
    positions: list[PositionSample],
    stats: ExtractionStats,
) -> None:
    if _read_exact(stream, len(MCAP_MAGIC), label) != MCAP_MAGIC:
        raise ExtractionError(f"{label}: invalid MCAP magic")

    channels: dict[int, str] = {}

    def consume(opcode: int, payload: bytes) -> None:
        if opcode == OP_SCHEMA:
            return
        if opcode == OP_CHANNEL:
            reader = ByteReader(payload, f"{label} channel")
            channel_id = reader.uint16()
            reader.uint16()  # schema_id
            topic = reader.string()
            reader.string()  # message_encoding
            channels[channel_id] = topic
            return
        if opcode != OP_MESSAGE:
            return

        reader = ByteReader(payload, f"{label} message")
        channel_id = reader.uint16()
        reader.uint32()  # sequence
        reader.uint64()  # log_time
        reader.uint64()  # publish_time
        message = reader.remainder()
        topic = channels.get(channel_id)
        if topic is None:
            raise ExtractionError(f"{label}: message references unknown channel {channel_id}")
        stats.mcap_messages += 1
        if topic == orientation_topic:
            try:
                orientations.append(_parse_imu_orientation(message))
                stats.orientation_messages += 1
            except ExtractionError:
                stats.rejected_orientations += 1
        elif topic == position_topic:
            samples = _parse_tf_positions(message, parent_frame, child_frame)
            positions.extend(samples)
            stats.position_messages += len(samples)

    while True:
        first = stream.read(1)
        if not first:
            raise ExtractionError(f"{label}: missing trailing MCAP magic")
        if first == MCAP_MAGIC[:1]:
            trailer = first + _read_exact(stream, len(MCAP_MAGIC) - 1, label)
            if trailer != MCAP_MAGIC:
                raise ExtractionError(f"{label}: invalid trailing MCAP magic")
            break
        header = first + _read_exact(stream, 8, label)
        opcode = header[0]
        size = struct.unpack_from("<Q", header, 1)[0]
        payload = _read_exact(stream, size, label)
        if opcode == OP_CHUNK:
            for nested_opcode, nested_payload in _chunk_records(payload, label):
                consume(nested_opcode, nested_payload)
        else:
            consume(opcode, payload)


def _deduplicate_orientations(
    samples: Sequence[OrientationSample],
) -> list[OrientationSample]:
    ordered = sorted(samples, key=lambda sample: sample.stamp_ns)
    result: list[OrientationSample] = []
    for sample in ordered:
        if result and sample.stamp_ns == result[-1].stamp_ns:
            result[-1] = sample
        else:
            result.append(sample)
    return result


def _deduplicate_positions(samples: Sequence[PositionSample]) -> list[PositionSample]:
    ordered = sorted(samples, key=lambda sample: sample.stamp_ns)
    result: list[PositionSample] = []
    for sample in ordered:
        if result and sample.stamp_ns == result[-1].stamp_ns:
            delta = max(
                abs(a - b)
                for a, b in zip(sample.position_xyz_m, result[-1].position_xyz_m)
            )
            if delta > 1.0e-9:
                raise ExtractionError(
                    f"conflicting RTK positions at timestamp {sample.stamp_ns}"
                )
            continue
        result.append(sample)
    return result


def _slerp(
    first: Sequence[float], second: Sequence[float], fraction: float
) -> tuple[float, float, float, float]:
    if not 0.0 <= fraction <= 1.0:
        raise ExtractionError("quaternion interpolation fraction is outside [0, 1]")
    left = _normalized_quaternion(first)
    right = _normalized_quaternion(second)
    dot = sum(a * b for a, b in zip(left, right))
    if dot < 0.0:
        right = tuple(-value for value in right)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalized_quaternion(
            tuple(a + fraction * (b - a) for a, b in zip(left, right))
        )
    angle = math.acos(dot)
    denominator = math.sin(angle)
    first_weight = math.sin((1.0 - fraction) * angle) / denominator
    second_weight = math.sin(fraction * angle) / denominator
    return _normalized_quaternion(
        tuple(
            first_weight * a + second_weight * b for a, b in zip(left, right)
        )
    )


def _rotate(
    quaternion_xyzw: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    x, y, z, w = _normalized_quaternion(quaternion_xyzw)
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _associate(
    orientations: Sequence[OrientationSample],
    positions: Sequence[PositionSample],
    max_gap_ns: int,
    imu_to_antenna_m: Sequence[float],
    stats: ExtractionStats,
) -> list[PoseSample]:
    if len(orientations) < 2:
        raise ExtractionError("fewer than two valid /imu/data orientation samples")
    stamps = [sample.stamp_ns for sample in orientations]
    result: list[PoseSample] = []
    for position in positions:
        upper_index = bisect.bisect_left(stamps, position.stamp_ns)
        if upper_index < len(stamps) and stamps[upper_index] == position.stamp_ns:
            quaternion = orientations[upper_index].quaternion_xyzw
        elif upper_index == 0 or upper_index == len(stamps):
            stats.unmatched_positions += 1
            continue
        else:
            lower = orientations[upper_index - 1]
            upper = orientations[upper_index]
            before = position.stamp_ns - lower.stamp_ns
            after = upper.stamp_ns - position.stamp_ns
            if before > max_gap_ns or after > max_gap_ns:
                stats.unmatched_positions += 1
                continue
            fraction = before / (upper.stamp_ns - lower.stamp_ns)
            quaternion = _slerp(
                lower.quaternion_xyzw, upper.quaternion_xyzw, fraction
            )
        rotated_lever = _rotate(quaternion, imu_to_antenna_m)
        imu_position = tuple(
            antenna - lever
            for antenna, lever in zip(position.position_xyz_m, rotated_lever)
        )
        result.append(PoseSample(position.stamp_ns, imu_position, quaternion))
    if not result:
        raise ExtractionError("no RTK position had a usable SBG orientation bracket")
    return result


def extract_trajectory(
    input_path: Path,
    orientation_topic: str,
    position_topic: str,
    parent_frame: str,
    child_frame: str,
    max_gap_ns: int,
    imu_to_antenna_m: Sequence[float],
) -> tuple[list[PoseSample], ExtractionStats, list[OrientationSample]]:
    orientations: list[OrientationSample] = []
    positions: list[PositionSample] = []
    stats = ExtractionStats()
    with _open_inputs(input_path) as streams:
        for label, stream in streams:
            _parse_mcap_stream(
                stream,
                label,
                orientation_topic,
                position_topic,
                parent_frame,
                child_frame,
                orientations,
                positions,
                stats,
            )
    orientations = _deduplicate_orientations(orientations)
    positions = _deduplicate_positions(positions)
    if not positions:
        raise ExtractionError(
            f"no {parent_frame} -> {child_frame} transforms found on {position_topic}"
        )
    poses = _associate(
        orientations, positions, max_gap_ns, imu_to_antenna_m, stats
    )
    return poses, stats, orientations


def _atomic_text_output(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )

    @contextlib.contextmanager
    def manager():
        try:
            yield temporary
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary.close()
            os.replace(temporary.name, path)
        except BaseException:
            temporary.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary.name)
            raise

    return manager()


def _write_tum(
    path: Path,
    poses: Sequence[PoseSample],
    source: Path,
    parent_frame: str,
    imu_to_antenna_m: Sequence[float],
) -> None:
    lever = ", ".join(f"{value:.9g}" for value in imu_to_antenna_m)
    with _atomic_text_output(path) as stream:
        stream.write(f"# source: {source}\n")
        stream.write(f"# pose_semantics: T_{parent_frame}_sbg_imu\n")
        stream.write(f"# imu_to_rtk_antenna_m: [{lever}]\n")
        stream.write("# timestamp x y z qx qy qz qw\n")
        for pose in poses:
            seconds, nanoseconds = divmod(pose.stamp_ns, 1_000_000_000)
            timestamp = f"{seconds}.{nanoseconds:09d}"
            values = (*pose.position_xyz_m, *pose.quaternion_xyzw)
            stream.write(timestamp + " " + " ".join(f"{value:.12g}" for value in values) + "\n")


def _write_state(path: Path, poses: Sequence[PoseSample]) -> None:
    with _atomic_text_output(path) as stream:
        stream.write("# sec,nsec,x,y,z,qx,qy,qz,qw\n")
        for pose in poses:
            seconds, nanoseconds = divmod(pose.stamp_ns, 1_000_000_000)
            values = (*pose.position_xyz_m, *pose.quaternion_xyzw)
            stream.write(
                f"{seconds},{nanoseconds},"
                + ",".join(f"{value:.12g}" for value in values)
                + "\n"
            )


def _maximum_gap_ns(samples: Sequence[OrientationSample]) -> int:
    return max(
        (right.stamp_ns - left.stamp_ns for left, right in zip(samples, samples[1:])),
        default=0,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract RTK positions and interpolated SBG attitudes from a Barakuda "
            "MCAP bag into a TUM reference trajectory."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="MCAP file, rosbag directory, or original uncompressed tar archive",
    )
    parser.add_argument("tum_output", type=Path, help="output TUM trajectory")
    parser.add_argument(
        "--state-output",
        type=Path,
        help="also write sec,nsec,x,y,z,qx,qy,qz,qw CSV for analyze_ingress_rrd.py",
    )
    parser.add_argument("--orientation-topic", default="/imu/data")
    parser.add_argument("--position-topic", default="/tf")
    parser.add_argument("--parent-frame", default="rtk_enu")
    parser.add_argument("--position-child-frame", default="rtk_antenna")
    parser.add_argument(
        "--imu-to-antenna-m",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
        help=(
            "vector from sbg_imu_link to the RTK antenna phase centre, expressed "
            "in sbg_imu_link, in metres (default: zero; provisional only)"
        ),
    )
    parser.add_argument(
        "--max-orientation-gap-ms",
        type=float,
        default=50.0,
        help="maximum allowed RTK-to-each-orientation-bracket gap (default: 50 ms)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        not math.isfinite(args.max_orientation_gap_ms)
        or args.max_orientation_gap_ms <= 0.0
    ):
        print("error: --max-orientation-gap-ms must be finite and positive", file=sys.stderr)
        return 2
    if not all(math.isfinite(value) for value in args.imu_to_antenna_m):
        print("error: --imu-to-antenna-m must contain finite values", file=sys.stderr)
        return 2
    if all(abs(value) < 1.0e-12 for value in args.imu_to_antenna_m):
        print(
            "warning: using a zero IMU-to-RTK-antenna lever arm; ATE is provisional",
            file=sys.stderr,
        )
    try:
        poses, stats, orientations = extract_trajectory(
            args.input,
            args.orientation_topic,
            args.position_topic,
            args.parent_frame,
            args.position_child_frame,
            int(round(args.max_orientation_gap_ms * 1.0e6)),
            args.imu_to_antenna_m,
        )
        _write_tum(
            args.tum_output,
            poses,
            args.input,
            args.parent_frame,
            args.imu_to_antenna_m,
        )
        if args.state_output is not None:
            _write_state(args.state_output, poses)
    except (ExtractionError, OSError, tarfile.TarError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    duration_s = (poses[-1].stamp_ns - poses[0].stamp_ns) / 1.0e9
    print(f"TUM: {args.tum_output}")
    if args.state_output is not None:
        print(f"state: {args.state_output}")
    print(
        f"poses={len(poses)}, duration_s={duration_s:.3f}, "
        f"RTK_unmatched={stats.unmatched_positions}, "
        f"IMU_orientations={len(orientations)}, "
        f"IMU_max_header_gap_ms={_maximum_gap_ns(orientations) / 1.0e6:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
