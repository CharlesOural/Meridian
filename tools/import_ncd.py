#!/usr/bin/env python3
"""Import a Newer College (multi-cam, 2021 OS0-128 + Alphasense) calibration into a
Meridian config, in place.

Sources:
  - Kalibr camchain-imucam YAML (per collection): cam0 intrinsics, equidistant
    distortion, resolution, T_cam_imu against the Alphasense IMU, and the
    cam->imu timeshift.
  - os_imu_lidar_transforms.yaml (shared by all collections): the rigid chain
    between the Ouster frames and the Alphasense IMU. The bags' PointCloud2 is
    stamped in the `os_sensor` frame, so the LiDAR extrinsic the config wants is
    exactly the `os_sensor_to_as_imu` entry (T_as-imu_os-sensor: maps an
    os_sensor-frame point into the Alphasense-IMU/estimation frame).
  - IMU noise: the camchain has none; the dataset's Kalibr IMU model is passed
    via flags (defaults below are the dataset's own values). Meridian's `cov_*`
    keys hold squared continuous-time densities, so the tool writes density^2 /
    random_walk^2.

Conventions:
  - os_imu_lidar_transforms.yaml quaternions are (qx, qy, qz, qw) — qx first.
    The os->Alphasense entries are a 180-degree rotation about X, not identity;
    importing them as identity silently flips Z and breaks gravity alignment.
  - camchain T_cam_imu maps an IMU-frame point into the camera frame; Meridian's
    `camera.extrinsic` is T_imu_cam as [tx, ty, tz, qx, qy, qz, qw], so the
    stored matrix is inverted.
  - timeshift_cam_imu is `t_imu = t_cam + shift`, the same direction as the
    config's `t_corrected = t + time_offset_ms`, so it is written as-is in ms.
    Both sensors expose a `time_offset_ms` key at the same indent; the camera's
    is edited inside the `camera:` section only.

The config edit is text-level and idempotent: every byte outside the owned value
tokens is preserved.

Usage (per collection -> per config):
    python3 tools/import_ncd.py \
        --camchain bags/newer-college/calib/collection1/cam0-1/*camchain-imucam*.yaml \
        --config src/meridian_ros/config/newer-college-quad.yaml [--dry-run]
"""
import argparse
import math
import re
import sys
from pathlib import Path

import yaml


def quat_to_rotation(qw, qx, qy, qz):
    """Rotation matrix (row-major 3x3 nested lists) for a (qw,qx,qy,qz) quaternion."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        raise SystemExit('error: zero-norm quaternion in calibration')
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ]


def invert_rt(R, t):
    """Inverse of a rigid transform: returns (R^T, -R^T t) as (3x3 lists, 3-list)."""
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    ti = [-sum(Rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    return Rt, ti


def rotation_to_quat(R):
    """(qx, qy, qz, qw) for a row-major 3x3 rotation, branching on the largest
    diagonal term so the divisor stays well away from zero for any rotation."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0.0:
        sq = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * sq
        qx = (R[2][1] - R[1][2]) / sq
        qy = (R[0][2] - R[2][0]) / sq
        qz = (R[1][0] - R[0][1]) / sq
    elif R[0][0] >= R[1][1] and R[0][0] >= R[2][2]:
        sq = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        qw = (R[2][1] - R[1][2]) / sq
        qx = 0.25 * sq
        qy = (R[0][1] + R[1][0]) / sq
        qz = (R[0][2] + R[2][0]) / sq
    elif R[1][1] >= R[2][2]:
        sq = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        qw = (R[0][2] - R[2][0]) / sq
        qx = (R[0][1] + R[1][0]) / sq
        qy = 0.25 * sq
        qz = (R[1][2] + R[2][1]) / sq
    else:
        sq = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        qw = (R[1][0] - R[0][1]) / sq
        qx = (R[0][2] + R[2][0]) / sq
        qy = (R[1][2] + R[2][1]) / sq
        qz = 0.25 * sq
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return qx / n, qy / n, qz / n, qw / n


def fmt_num(x: float) -> str:
    """Round-trippable repr that stays compact for the values this tool writes."""
    return repr(float(x))


def fmt_seq(values) -> str:
    return '[' + ', '.join(fmt_num(v) for v in values) + ']'


# An owned scalar/sequence key: `<indent><name>: <value>` with an optional inline
# comment. The value may span continuation lines (the row-major matrix), so a
# separate multiline form handles the bracketed sequence case.
def _scalar_pattern(indent: str, name: str):
    return re.compile(
        rf'(?P<lead>^{re.escape(indent)}{re.escape(name)}:[ \t]*)'
        rf'(?P<value>[^\n#]*?)'
        rf'(?P<trail>[ \t]*(?:#[^\n]*)?)$',
        re.MULTILINE,
    )


def _seq_pattern(indent: str, name: str):
    # `name: [ ... ]` where the bracket body may contain newlines (row-major R).
    return re.compile(
        rf'(?P<lead>^{re.escape(indent)}{re.escape(name)}:[ \t]*)'
        rf'(?P<value>\[.*?\])',
        re.MULTILINE | re.DOTALL,
    )


def replace_scalar(text: str, indent: str, name: str, new_value: str, owner: str):
    pat = _scalar_pattern(indent, name)
    m = pat.search(text)
    if m is None:
        raise SystemExit(f'error: config is missing expected key {indent}{name} '
                         f'(needed for {owner})')
    if m.group('value').strip() == new_value.strip():
        return text, False
    start, end = m.start('value'), m.end('value')
    return text[:start] + new_value + text[end:], True


def replace_seq(text: str, indent: str, name: str, new_value: str, owner: str):
    pat = _seq_pattern(indent, name)
    m = pat.search(text)
    if m is None:
        raise SystemExit(f'error: config is missing expected key {indent}{name} '
                         f'(needed for {owner})')
    if m.group('value') == new_value:
        return text, False
    start, end = m.start('value'), m.end('value')
    return text[:start] + new_value + text[end:], True


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def lidar_extrinsic(transforms: dict, src: Path):
    """T_imu_lidar (R flat 9-list, t 3-list) from the os_sensor_to_as_imu entry."""
    entry = transforms.get('os_sensor_to_as_imu')
    if entry is None:
        raise SystemExit(f'error: {src} is missing os_sensor_to_as_imu')
    t = list(entry['translation'])
    qx, qy, qz, qw = entry['rotation']  # file order is (qx, qy, qz, qw)
    R = quat_to_rotation(qw, qx, qy, qz)
    return R[0] + R[1] + R[2], t


def camera_block(camchain: dict, src: Path):
    """cam0 values: intrinsics, distortion, size, T_imu_cam pose, timeshift [ms]."""
    cam = camchain.get('cam0')
    if cam is None:
        raise SystemExit(f'error: {src} has no cam0 entry')
    if cam.get('camera_model') != 'pinhole':
        raise SystemExit(f"error: cam0 model {cam.get('camera_model')!r} != pinhole")
    model = cam.get('distortion_model')
    if model not in ('equidistant', 'radtan', 'plumb_bob'):
        raise SystemExit(f'error: unsupported distortion model {model!r}')
    model = 'radtan' if model == 'plumb_bob' else model
    T = cam['T_cam_imu']
    R_cam_imu = [row[:3] for row in T[:3]]
    t_cam_imu = [row[3] for row in T[:3]]
    R_imu_cam, t_imu_cam = invert_rt(R_cam_imu, t_cam_imu)
    qx, qy, qz, qw = rotation_to_quat(R_imu_cam)
    pose = list(t_imu_cam) + [qx, qy, qz, qw]
    width, height = cam['resolution']
    return (list(cam['intrinsics']), model, list(cam['distortion_coeffs']),
            int(width), int(height), pose, float(cam.get('timeshift_cam_imu', 0.0)) * 1e3)


def section_span(text: str, name: str):
    """[start, end) byte span of a 4-space-indented `name:` section's body."""
    m = re.search(rf'^    {re.escape(name)}:[ \t]*$', text, re.MULTILINE)
    if m is None:
        raise SystemExit(f'error: config has no `    {name}:` section')
    nxt = re.compile(r'^    \S', re.MULTILINE).search(text, m.end())
    return m.end(), nxt.start() if nxt else len(text)


def replace_scalar_in_section(text, section, indent, name, value, owner):
    start, end = section_span(text, section)
    body, did = replace_scalar(text[start:end], indent, name, value, owner)
    return text[:start] + body + text[end:], did


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    nc = repo / 'bags' / 'newer-college'
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camchain', type=Path, required=True,
                    help='Kalibr camchain-imucam YAML for the matching collection')
    ap.add_argument('--transforms', type=Path,
                    default=nc / 'calib' / 'os_imu_lidar_transforms.yaml')
    ap.add_argument('--config', type=Path, required=True,
                    help='Meridian config to update in place')
    # Alphasense BMI085 continuous-time densities. The dataset's Kalibr file puts the
    # accelerometer density (0.019) in the gyro field too; a real BMI085 gyro is an
    # order of magnitude quieter (~2e-3 rad/s/sqrt(Hz)), and importing the duplicated
    # value de-weights the gyro residual ~90x in variance — enough to let window
    # rotation float and diverge the whole estimator. The dataset's gyro random walk
    # (2.66e-4 rad/s^2/sqrt(Hz)) is unusable the same way: it lets the bias absorb
    # LiDAR misfit within a single window (the bias pegs its box during brisk motion
    # and heading diverges at meters scale); 4e-6 holds the bias near its static-init
    # value, which still outruns real BMI085 bias drift on a minutes-long run.
    # Defaults below keep the dataset's accel numbers and use the realistic gyro values.
    ap.add_argument('--acc-nd', type=float, default=0.019, help='[m/s^2/sqrt(Hz)]')
    ap.add_argument('--acc-rw', type=float, default=0.0043, help='[m/s^3/sqrt(Hz)]')
    ap.add_argument('--gyr-nd', type=float, default=0.002, help='[rad/s/sqrt(Hz)]')
    ap.add_argument('--gyr-rw', type=float, default=4e-6, help='[rad/s^2/sqrt(Hz)]')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    for p in (args.camchain, args.transforms, args.config):
        if not p.exists():
            raise SystemExit(f'error: {p} not found')

    R_flat, t_vec = lidar_extrinsic(load_yaml(args.transforms), args.transforms)
    (intr, dist_model, dist_coeffs, cam_w, cam_h,
     cam_pose, cam_shift_ms) = camera_block(load_yaml(args.camchain), args.camchain)

    text = args.config.read_text()
    changed = {}
    seq_edits = [
        ('extrinsic_T', fmt_seq(t_vec), 'lidar.extrinsic_T'),
        ('extrinsic_R', fmt_seq(R_flat), 'lidar.extrinsic_R'),
        ('intrinsics', fmt_seq(intr), 'camera.intrinsics'),
        ('distortion_coeffs', fmt_seq(dist_coeffs), 'camera.distortion_coeffs'),
        ('extrinsic', fmt_seq(cam_pose), 'camera.extrinsic'),
    ]
    for name, value, owner in seq_edits:
        text, did = replace_seq(text, '      ', name, value, owner)
        changed[owner] = did
    scalar_edits = [
        ('distortion_model', dist_model, 'camera.distortion_model'),
        ('width', str(cam_w), 'camera.width'),
        ('height', str(cam_h), 'camera.height'),
        ('cov_acc', fmt_num(args.acc_nd ** 2), 'imu.cov_acc'),
        ('cov_gyr', fmt_num(args.gyr_nd ** 2), 'imu.cov_gyr'),
        ('b_acc_cov', fmt_num(args.acc_rw ** 2), 'imu.b_acc_cov'),
        ('b_gyr_cov', fmt_num(args.gyr_rw ** 2), 'imu.b_gyr_cov'),
    ]
    for name, value, owner in scalar_edits:
        text, did = replace_scalar(text, '      ', name, value, owner)
        changed[owner] = did
    text, did = replace_scalar_in_section(
        text, 'camera', '      ', 'time_offset_ms', fmt_num(cam_shift_ms),
        'camera.time_offset_ms')
    changed['camera.time_offset_ms'] = did

    w = max(len(k) for k in changed)
    print('summary (key | written):')
    for (name, value, owner) in seq_edits + scalar_edits:
        print(f'  {owner:<{w}}  {value}{"" if changed[owner] else "  (unchanged)"}')
    print(f'  {"camera.time_offset_ms":<{w}}  {fmt_num(cam_shift_ms)}'
          f'{"" if changed["camera.time_offset_ms"] else "  (unchanged)"}')

    if args.dry_run:
        print('\n--dry-run: no file written.')
        return 0
    if not any(changed.values()):
        print(f'\n{args.config}: already up to date; no write needed.')
        return 0
    args.config.write_text(text)

    # Round-trip: the owned values must parse back to what was computed.
    doc = yaml.safe_load(args.config.read_text())
    s = doc['meridian']['sensors']
    def close(a, b):
        return abs(float(a) - float(b)) <= 1e-12 + 1e-9 * abs(float(b))
    for name, want, have in (
        ('lidar.extrinsic_R', R_flat, s['lidar']['extrinsic_R']),
        ('lidar.extrinsic_T', t_vec, s['lidar']['extrinsic_T']),
        ('camera.intrinsics', intr, s['camera']['intrinsics']),
        ('camera.distortion_coeffs', dist_coeffs, s['camera']['distortion_coeffs']),
        ('camera.extrinsic', cam_pose, s['camera']['extrinsic']),
    ):
        if len(have) != len(want) or not all(close(h, x) for h, x in zip(have, want)):
            raise SystemExit(f'error: round-trip mismatch on {name}')
    for name, want, have in (
        ('camera.time_offset_ms', cam_shift_ms, s['camera']['time_offset_ms']),
        ('imu.cov_acc', args.acc_nd ** 2, s['imu']['cov_acc']),
        ('imu.cov_gyr', args.gyr_nd ** 2, s['imu']['cov_gyr']),
        ('imu.b_acc_cov', args.acc_rw ** 2, s['imu']['b_acc_cov']),
        ('imu.b_gyr_cov', args.gyr_rw ** 2, s['imu']['b_gyr_cov']),
    ):
        if not close(have, want):
            raise SystemExit(f'error: round-trip mismatch on {name}')
    if s['camera']['distortion_model'] != dist_model:
        raise SystemExit('error: round-trip mismatch on distortion_model')
    print(f'\n{args.config}: written and round-trip verified.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
