#!/usr/bin/env python3
"""Import a FusionPortable sensor calibration into a Meridian config, in place.

The calibration ships as a directory of OpenCV-matrix YAML files (despite the
"kalibr" name, these are FusionPortable's own `!!opencv-matrix` layout, not a
Kalibr camchain). This tool reads the LiDAR/IMU/camera files it is told to and
rewrites exactly six value tokens in the Meridian config:

    sensors.lidar.extrinsic_R   sensors.lidar.extrinsic_T
    sensors.imu.cov_acc         sensors.imu.cov_gyr
    sensors.imu.b_acc_cov       sensors.imu.b_gyr_cov
    sensors.camera.intrinsics   sensors.camera.distortion_model
    sensors.camera.distortion_coeffs
    sensors.camera.width        sensors.camera.height

Conventions:

LiDAR extrinsic direction. Meridian's `extrinsic_R`/`extrinsic_T` are
T_imu_lidar: they map a point in the LiDAR frame into the IMU (estimation)
frame, p_imu = R*p_lidar + t.
`ouster00.yaml` instead stores `quaternion_sensor_bodyimu`/
`translation_sensor_bodyimu`, which is T_lidar_bodyimu: the companion
`quaternion_sensor_osimu00` block is annotated `rosrun tf tf_echo os_sensor
os_imu`, and tf_echo A B reports T_A_B, so `*_sensor_X` here means T_lidar_X
(maps an X-frame point into the LiDAR frame). That is the inverse of what the
config wants, so this tool inverts it: R_out = R_stored^T, t_out = -R_stored^T t.
The active calibration is `ouster00.yaml` option 1 (uncommented); option 2 is a
commented-out alternative and is ignored.

Quaternion order is (qw, qx, qy, qz) as the file comments state.

IMU noise units. The source files give continuous-time noise *densities*
(accelerometer m/s^2/sqrt(Hz), gyroscope rad/s/sqrt(Hz)) and bias random walks
(m/s^3/sqrt(Hz), rad/s^2/sqrt(Hz)). Meridian's `cov_*` keys hold the squared
densities sigma^2 (the continuous-time power spectral density / variance density
that feeds the process-noise Q directly). Squaring preserves the continuous-time
form, so this tool writes density^2 and random_walk^2 with no rate factor
applied.

Camera distortion. The calib's distortion model name is normalised to the
config's closed set (OpenCV/ROS `plumb_bob` -> `radtan`; `equidistant`/`equi`
-> `equidistant`; absent/`none` -> `none`) and written to
`sensors.camera.distortion_model`; the coefficients (radtan k1,k2,p1,p2[,k3])
and the image `width`/`height` are written verbatim. This is dataset-agnostic:
the model name and coefficient count come straight from the source file.

The config edit is text-level: every line, comment, and ordering byte outside the
owned value tokens is preserved exactly. A missing expected key is a hard error.
Running the tool twice is a no-op the second time (the writer compares the
rendered value against what is already present).

Usage (defaults target the FusionPortable calib + the davis-left/ouster/stim300
chain):
    python3 tools/import_kalibr.py [--calib-dir bags/calib] \
        [--camera event_cam00] [--lidar ouster00] [--imu body_imu] \
        [--config src/meridian_ros/config/fusionportable.yaml] [--dry-run]
"""
import argparse
import math
import re
import sys
from pathlib import Path

import yaml


# The OpenCV-matrix tag carries `rows`/`cols`/`dt`/`data`; collapse each block to
# its flat `data` list so the rest of the tool sees plain numbers.
def _opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    return mapping['data']


class CalibLoader(yaml.SafeLoader):
    pass


CalibLoader.add_constructor('tag:yaml.org,2002:opencv-matrix', _opencv_matrix)


def load_calib(path: Path) -> dict:
    with path.open() as f:
        return yaml.load(f, Loader=CalibLoader)


def require(doc: dict, key: str, src: Path):
    """Fetch `key` from a parsed calib file or fail naming the file."""
    if key not in doc:
        raise SystemExit(f'error: {src} is missing required key {key!r}')
    return doc[key]


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


def lidar_extrinsic(doc: dict, src: Path):
    """T_imu_lidar (R row-major 9-list, t 3-list) from the stored T_lidar_bodyimu."""
    q = require(doc, 'quaternion_sensor_bodyimu', src)
    t = require(doc, 'translation_sensor_bodyimu', src)
    if len(q) != 4 or len(t) != 3:
        raise SystemExit(f'error: {src} body-imu extrinsic has wrong shape')
    R_lidar_imu = quat_to_rotation(*q)
    R_imu_lidar, t_imu_lidar = invert_rt(R_lidar_imu, list(t))
    flat = R_imu_lidar[0] + R_imu_lidar[1] + R_imu_lidar[2]
    return flat, t_imu_lidar


# Map a source distortion-model name onto the config's closed set. The radial-
# tangential model travels under several names across toolchains (OpenCV/ROS
# `plumb_bob`, Kalibr `radtan`); the equidistant/fisheye model likewise. An
# unrecognised name is a hard error rather than a silent `none`, so a fisheye
# calibration cannot be quietly imported as if it had no distortion.
def normalize_distortion_model(name: str) -> str:
    key = (name or 'none').strip().lower()
    table = {
        'none': 'none', '': 'none',
        'radtan': 'radtan', 'plumb_bob': 'radtan', 'pinhole-radtan': 'radtan',
        'equidistant': 'equidistant', 'equi': 'equidistant',
        'pinhole-equi': 'equidistant', 'fisheye': 'equidistant',
        'kannala-brandt': 'equidistant',
    }
    if key not in table:
        raise SystemExit(f'error: unknown distortion model {name!r}; '
                         'expected plumb_bob/radtan/equidistant/none')
    return table[key]


def camera_intrinsics(doc: dict, src: Path):
    """([fx,fy,cx,cy], model, coeffs, width, height) from a camera calib file."""
    K = require(doc, 'camera_matrix', src)
    if len(K) != 9:
        raise SystemExit(f'error: {src} camera_matrix is not 3x3')
    fx, cx, fy, cy = K[0], K[2], K[4], K[5]
    model = normalize_distortion_model(doc.get('distortion_model', 'none'))
    dist = list(doc.get('distortion_coefficients', []))
    width = int(doc.get('image_width', 0))
    height = int(doc.get('image_height', 0))
    return [fx, fy, cx, cy], model, dist, width, height


def imu_noise(doc: dict, src: Path):
    """(cov_acc, cov_gyr, b_acc_cov, b_gyr_cov) as squared continuous-time densities."""
    acc_nd = require(doc, 'accelerometer_noise_density', src)
    acc_rw = require(doc, 'accelerometer_random_walk', src)
    gyr_nd = require(doc, 'gyroscope_noise_density', src)
    gyr_rw = require(doc, 'gyroscope_random_walk', src)
    return acc_nd ** 2, gyr_nd ** 2, acc_rw ** 2, gyr_rw ** 2


def fmt_num(x: float) -> str:
    """Round-trippable repr that stays compact for the values this tool writes."""
    return repr(float(x))


def fmt_seq(values) -> str:
    return '[' + ', '.join(fmt_num(v) for v in values) + ']'


# An owned scalar/sequence key: `<indent><name>: <value>` with an optional inline
# comment. The value may span continuation lines (the row-major matrix), so the
# value capture is greedy up to the closing bracket or end of line, and a separate
# multiline form handles the bracketed case.
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


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calib-dir', type=Path, default=repo / 'bags' / 'calib',
                    help='directory of FusionPortable calib YAML files')
    ap.add_argument('--camera', default='event_cam00',
                    help='camera calib basename (default event_cam00 = davis_left)')
    ap.add_argument('--lidar', default='ouster00',
                    help='lidar calib basename')
    ap.add_argument('--imu', default='body_imu',
                    help='imu noise calib basename')
    ap.add_argument('--config', type=Path,
                    default=repo / 'src' / 'meridian_ros' / 'config' / 'fusionportable.yaml',
                    help='Meridian config to update in place')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the would-be changes without writing')
    args = ap.parse_args()

    lidar_src = args.calib_dir / f'{args.lidar}.yaml'
    cam_src = args.calib_dir / f'{args.camera}.yaml'
    imu_src = args.calib_dir / f'{args.imu}.yaml'
    for src in (lidar_src, cam_src, imu_src, args.config):
        if not src.exists():
            raise SystemExit(f'error: {src} not found')

    R_flat, t_vec = lidar_extrinsic(load_calib(lidar_src), lidar_src)
    intr, dist_model, dist_coeffs, cam_w, cam_h = camera_intrinsics(
        load_calib(cam_src), cam_src)
    cov_acc, cov_gyr, b_acc_cov, b_gyr_cov = imu_noise(load_calib(imu_src), imu_src)

    # Render the matrix on one line; the config's existing 3-row layout is replaced
    # wholesale, which keeps the surgery a single bracketed-value swap.
    R_str = fmt_seq(R_flat)
    t_str = fmt_seq(t_vec)
    intr_str = fmt_seq(intr)
    coeffs_str = fmt_seq(dist_coeffs)

    text = args.config.read_text()
    edits = [
        ('seq',    '      ', 'extrinsic_T',       t_str,             'lidar.extrinsic_T'),
        ('seq',    '      ', 'extrinsic_R',       R_str,             'lidar.extrinsic_R'),
        ('seq',    '      ', 'intrinsics',        intr_str,          'camera.intrinsics'),
        ('scalar', '      ', 'distortion_model',  dist_model,        'camera.distortion_model'),
        ('seq',    '      ', 'distortion_coeffs', coeffs_str,        'camera.distortion_coeffs'),
        ('scalar', '      ', 'width',             str(cam_w),        'camera.width'),
        ('scalar', '      ', 'height',            str(cam_h),        'camera.height'),
        ('scalar', '      ', 'cov_acc',           fmt_num(cov_acc),  'imu.cov_acc'),
        ('scalar', '      ', 'cov_gyr',           fmt_num(cov_gyr),  'imu.cov_gyr'),
        ('scalar', '      ', 'b_acc_cov',         fmt_num(b_acc_cov),'imu.b_acc_cov'),
        ('scalar', '      ', 'b_gyr_cov',         fmt_num(b_gyr_cov),'imu.b_gyr_cov'),
    ]

    changed = {}
    new_text = text
    for kind, indent, name, value, owner in edits:
        if kind == 'seq':
            new_text, did = replace_seq(new_text, indent, name, value, owner)
        else:
            new_text, did = replace_scalar(new_text, indent, name, value, owner)
        changed[owner] = did

    rows = [
        ('lidar.extrinsic_R',         lidar_src.name, R_str),
        ('lidar.extrinsic_T',         lidar_src.name, t_str),
        ('camera.intrinsics',         cam_src.name,   intr_str),
        ('camera.distortion_model',   cam_src.name,   dist_model),
        ('camera.distortion_coeffs',  cam_src.name,   coeffs_str),
        ('camera.width',              cam_src.name,   str(cam_w)),
        ('camera.height',             cam_src.name,   str(cam_h)),
        ('imu.cov_acc',               imu_src.name,   fmt_num(cov_acc)),
        ('imu.cov_gyr',               imu_src.name,   fmt_num(cov_gyr)),
        ('imu.b_acc_cov',             imu_src.name,   fmt_num(b_acc_cov)),
        ('imu.b_gyr_cov',             imu_src.name,   fmt_num(b_gyr_cov)),
    ]
    w_key = max(len(r[0]) for r in rows)
    w_src = max(len(r[1]) for r in rows)
    print('summary (sensor key | source file | value written):')
    for key, src_name, value in rows:
        flag = '' if changed.get(key) else '  (unchanged)'
        print(f'  {key:<{w_key}}  {src_name:<{w_src}}  {value}{flag}')

    any_change = any(changed.values())
    if args.dry_run:
        print()
        print('--dry-run: no file written.' if any_change
              else '--dry-run: config already up to date; nothing would change.')
        return 0

    if not any_change:
        print()
        print(f'{args.config}: already up to date; no write needed.')
        return 0

    args.config.write_text(new_text)

    # Round-trip: re-read and assert every owned value parses back to what we wrote.
    check = load_config_values(args.config)
    assert_roundtrip(check, R_flat, t_vec, intr, dist_model, dist_coeffs, cam_w, cam_h,
                     cov_acc, cov_gyr, b_acc_cov, b_gyr_cov)
    print()
    print(f'{args.config}: written and round-trip verified.')
    return 0


def load_config_values(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text())
    s = doc['meridian']['sensors']
    return {
        'extrinsic_R': s['lidar']['extrinsic_R'],
        'extrinsic_T': s['lidar']['extrinsic_T'],
        'intrinsics': s['camera']['intrinsics'],
        'distortion_model': s['camera']['distortion_model'],
        'distortion_coeffs': s['camera']['distortion_coeffs'],
        'width': s['camera']['width'],
        'height': s['camera']['height'],
        'cov_acc': s['imu']['cov_acc'],
        'cov_gyr': s['imu']['cov_gyr'],
        'b_acc_cov': s['imu']['b_acc_cov'],
        'b_gyr_cov': s['imu']['b_gyr_cov'],
    }


def assert_roundtrip(got, R_flat, t_vec, intr, dist_model, dist_coeffs, cam_w, cam_h,
                     cov_acc, cov_gyr, b_acc_cov, b_gyr_cov):
    def close(a, b):
        return abs(float(a) - float(b)) <= 1e-12 + 1e-9 * abs(float(b))

    for name, want, have in (
        ('extrinsic_R', R_flat, got['extrinsic_R']),
        ('extrinsic_T', t_vec, got['extrinsic_T']),
        ('intrinsics', intr, got['intrinsics']),
        ('distortion_coeffs', dist_coeffs, got['distortion_coeffs']),
    ):
        if len(have) != len(want) or not all(close(h, w) for h, w in zip(have, want)):
            raise SystemExit(f'error: round-trip mismatch on {name}: '
                             f'wrote {want}, read {have}')
    if got['distortion_model'] != dist_model:
        raise SystemExit(f"error: round-trip mismatch on distortion_model: "
                         f"wrote {dist_model!r}, read {got['distortion_model']!r}")
    for name, want, have in (
        ('width', cam_w, got['width']),
        ('height', cam_h, got['height']),
    ):
        if int(have) != int(want):
            raise SystemExit(f'error: round-trip mismatch on {name}: '
                             f'wrote {want}, read {have}')
    for name, want, have in (
        ('cov_acc', cov_acc, got['cov_acc']),
        ('cov_gyr', cov_gyr, got['cov_gyr']),
        ('b_acc_cov', b_acc_cov, got['b_acc_cov']),
        ('b_gyr_cov', b_gyr_cov, got['b_gyr_cov']),
    ):
        if not close(have, want):
            raise SystemExit(f'error: round-trip mismatch on {name}: '
                             f'wrote {want}, read {have}')


if __name__ == '__main__':
    sys.exit(main())
