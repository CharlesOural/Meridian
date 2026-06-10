#!/ usr / bin / env python3
"""Generate a truth-anchored loop-closure injection file from a packet-log index.

Reads the ``.index.txt`` sidecar written by PacketLogWriter, finds co-located
keyframe pairs in the drifted estimate, and emits a YAML loop list consumed by
``backend_runner --inject-loops``. When a ground-truth TUM trajectory is given,
the loop translation is taken from GT while the rotation is taken from the
estimate, so each loop pulls the drifted estimate back toward truth. A handful
of deliberately-corrupted ("outlier") loops are also emitted to exercise the
back-end's pairwise-consistency rejection.

Usage:
  make_loops.py <index.txt> <out.yaml> [--gt FILE] [--num N] [--outliers N]
    [--min-id-gap N] [--min-time-gap-s S] [--max-dist M] [--cooldown N]
    [--noise-t S] [--noise-r S] [--seed N]
"""
import argparse
import sys

import numpy as np

#-- -- quaternion helpers(xyzw order, Hamilton convention) -- -- -- -- -- -- -- -- -- -

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_to_mat(q):
    """Rotation matrix from a quaternion given as [x, y, z, w]."""
    x, y, z, w = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])


def mat_to_quat(R):
    """Quaternion [x, y, z, w] from a rotation matrix (Shepperd's method)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w]))


def so3_exp(omega):
    """Rotation matrix from an axis-angle so(3) vector via Rodrigues' formula."""
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

#-- -- parsing -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

def parse_index(path):
    """Read keyframe rows from a packet-log index.

    Each 'kf' line begins:
      kf <id> <stamp_ns> <tx> <ty> <tz> <qx> <qy> <qz> <qw> ...
    Only the first nine fields after the tag are used; trailing columns
    (rel_to_id, constraint kind) are ignored.
    """
    ids, stamps, pos, quat = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            v = line.split()
            if v[0] != 'kf' or len(v) < 10:
                continue
            ids.append(int(v[1]))
            stamps.append(int(v[2]))
            pos.append([float(v[3]), float(v[4]), float(v[5])])
            quat.append([float(v[6]), float(v[7]), float(v[8]), float(v[9])])
    if not ids:
        raise ValueError("no 'kf' lines found in index: " + path)
    order = np.argsort(np.array(ids))
    ids = np.array(ids)[order]
    stamps = np.array(stamps, dtype=np.int64)[order]
    pos = np.array(pos)[order]
    quat = np.array(quat)[order]
    return ids, stamps, pos, quat


def load_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            v = line.split()
            if len(v) < 8:
                continue
            rows.append([float(x) for x in v[:8]])
    a = np.array(rows)
    a = a[np.argsort(a[:, 0])]
    return a[:, 0], a[:, 1:4], a[:, 4:8]  # t_seconds, xyz, qxyzw


def gt_at_stamps(gt_t, gt_pos, gt_quat, stamps_ns):
    """Nearest-stamp lookup of GT pose for each keyframe stamp.

    Keyframe stamps are nanoseconds; the TUM time column is seconds.
    """
    stamps_s = stamps_ns.astype(np.float64) * 1e-9
    out_pos = np.empty((len(stamps_s), 3))
    out_quat = np.empty((len(stamps_s), 4))
    for k, t in enumerate(stamps_s):
        j = int(np.argmin(np.abs(gt_t - t)))
        out_pos[k] = gt_pos[j]
        out_quat[k] = gt_quat[j]
    return out_pos, out_quat

#-- -- loop construction -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

def relative_pose(R_from, p_from, R_to, p_to):
    """T_from_to: pose of 'to' expressed in the 'from' frame."""
    R_rel = R_from.T @ R_to
    t_rel = R_from.T @ (p_to - p_from)
    return R_rel, t_rel


def select_inliers(ids, stamps, pos, args):
    """Deterministically pick co-located pairs spread out by a cooldown.

    Candidates are all i<j with a sufficient id gap, time gap, and small
    estimate-position distance. The 'to' (newer, larger-id) keyframe of each
    accepted loop must be at least --cooldown keyframe ids away from every
    previously accepted 'to', so loops do not bunch up.
    """
    if args.num <= 0:
        return []
    n = len(ids)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (ids[j] - ids[i]) < args.min_id_gap:
                continue
            dt_s = (stamps[j] - stamps[i]) * 1e-9
            if dt_s < args.min_time_gap_s:
                continue
            if np.linalg.norm(pos[j] - pos[i]) > args.max_dist:
                continue
#store(from_idx, to_idx); j is newer / larger id
            candidates.append((ids[j], ids[i], i, j))
#sort by to_id then from_id so selection is order - independent
    candidates.sort(key=lambda c: (c[0], c[1]))
    chosen = []
    chosen_to = []
    for to_id, _from_id, i, j in candidates:
        if any(abs(int(to_id) - c) < args.cooldown for c in chosen_to):
            continue
        chosen.append((i, j))
        chosen_to.append(int(to_id))
        if len(chosen) >= args.num:
            break
    return chosen


def select_outlier_pairs(ids, stamps, args, n_out, used_to):
    """Pick far-apart pairs (large id gap) for corrupted loops.

    Endpoints must be individually plausible: a large id and time gap, but the
    estimate distance is irrelevant since the translation will be corrupted.
    Avoids reusing a 'to' id already taken by an inlier.
    """
    if n_out <= 0:
        return []
    n = len(ids)
    pairs = []
    big_gap = max(args.min_id_gap, n // 3 if n >= 3 else args.min_id_gap)
    for i in range(n):
        for j in range(i + 1, n):
            if (ids[j] - ids[i]) < big_gap:
                continue
            dt_s = (stamps[j] - stamps[i]) * 1e-9
            if dt_s < args.min_time_gap_s:
                continue
            pairs.append((ids[j], ids[i], i, j))
    pairs.sort(key=lambda c: (-(c[0] - c[1]), c[0]))
    out = []
    taken_to = list(used_to)
    for to_id, _from_id, i, j in pairs:
        if any(abs(int(to_id) - t) < args.cooldown for t in taken_to):
            continue
        out.append((i, j))
        taken_to.append(int(to_id))
        if len(out) >= n_out:
            break
    return out


def build_loop(i, j, ids, est_pos, est_quat, gt_pos, gt_quat, args, rng,
               outlier):
    """Assemble one loop dict for keyframes at index i (from/older) and j (to/newer)."""
    from_id = int(ids[i])
    to_id = int(ids[j])
    R_from = quat_to_mat(est_quat[i])
    R_to = quat_to_mat(est_quat[j])
#Rotation always from the estimate.
    R_rel = R_from.T @ R_to
#Translation from GT when available(truth - anchored), else from estimate.
    if gt_pos is not None:
        R_from_gt = quat_to_mat(gt_quat[i])
        t_rel = R_from_gt.T @ (gt_pos[j] - gt_pos[i])
    else:
        t_rel = R_from.T @ (est_pos[j] - est_pos[i])

    if outlier:
#Offset by a large wrong vector so the loop is geometrically wrong
#while still reporting a passing fitness.
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        t_rel = t_rel + direction * rng.uniform(3.0, 8.0)
        fitness = 0.85
    else:
        fitness = 0.9

#Small Gaussian perturbation so no loop is numerically perfect.
    t_rel = t_rel + rng.normal(scale=args.noise_t, size=3)
    R_rel = so3_exp(rng.normal(scale=args.noise_r, size=3)) @ R_rel

    q_rel = mat_to_quat(R_rel)
    vt = max(args.noise_t, 1e-3) ** 2
    vr = max(args.noise_r, 1e-3) ** 2
    cov_diag = [vt, vt, vt, vr, vr, vr]
    at_kf = max(from_id, to_id)
    return {
        'from_id': from_id,
        'to_id': to_id,
        't': [float(t_rel[0]), float(t_rel[1]), float(t_rel[2])],
        'q': [float(q_rel[0]), float(q_rel[1]), float(q_rel[2]), float(q_rel[3])],
        'cov_diag': [float(c) for c in cov_diag],
        'fitness': float(fitness),
        'at_kf': int(at_kf),
        'outlier': bool(outlier),
    }

#-- -- YAML emission(hand - written, no pyyaml dependency) -- -- -- -- -- -- -- -- -- -- -

def fmt_list(values):
    return '[' + ', '.join(repr(float(v)) for v in values) + ']'


def write_yaml(path, loops):
    lines = ['loops:']
    for lp in loops:
        lines.append('  - from_id: {}'.format(lp['from_id']))
        lines.append('    to_id: {}'.format(lp['to_id']))
        lines.append('    t: ' + fmt_list(lp['t']))
        lines.append('    q: ' + fmt_list(lp['q']))
        lines.append('    cov_diag: ' + fmt_list(lp['cov_diag']))
        lines.append('    fitness: {!r}'.format(float(lp['fitness'])))
        lines.append('    at_kf: {}'.format(lp['at_kf']))
        lines.append('    outlier: {}'.format('true' if lp['outlier'] else 'false'))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

#-- -- main -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Generate a truth-anchored loop-closure injection YAML '
                    'from a packet-log index.')
    p.add_argument('index', help='packet-log .index.txt sidecar')
    p.add_argument('out', help='output loop YAML path')
    p.add_argument('--gt', default=None, help='ground-truth trajectory (TUM)')
    p.add_argument('--num', type=int, default=8, help='max inlier loops')
    p.add_argument('--outliers', type=int, default=2,
                   help='number of corrupted loops to emit')
    p.add_argument('--min-id-gap', type=int, default=50,
                   help='minimum keyframe-id gap between endpoints')
    p.add_argument('--min-time-gap-s', type=float, default=20.0,
                   help='minimum wall-time gap between endpoints (s)')
    p.add_argument('--max-dist', type=float, default=5.0,
                   help='max estimate-position distance for a candidate (m)')
    p.add_argument('--cooldown', type=int, default=30,
                   help='min id separation between selected loop endpoints')
    p.add_argument('--noise-t', type=float, default=0.02,
                   help='translation noise sigma (m)')
    p.add_argument('--noise-r', type=float, default=0.005,
                   help='rotation noise sigma (rad)')
    p.add_argument('--seed', type=int, default=0, help='numpy RNG seed')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    ids, stamps, est_pos, est_quat = parse_index(args.index)

    gt_pos = gt_quat = None
    if args.gt:
        gt_t, gp, gq = load_tum(args.gt)
        gt_pos, gt_quat = gt_at_stamps(gt_t, gp, gq, stamps)

    inlier_idx = select_inliers(ids, stamps, est_pos, args)
    loops = [
        build_loop(i, j, ids, est_pos, est_quat, gt_pos, gt_quat, args, rng,
                   outlier=False)
        for (i, j) in inlier_idx
    ]

    used_to = [lp['to_id'] for lp in loops]
    outlier_idx = select_outlier_pairs(ids, stamps, args, args.outliers, used_to)
    out_loops = [
        build_loop(i, j, ids, est_pos, est_quat, gt_pos, gt_quat, args, rng,
                   outlier=True)
        for (i, j) in outlier_idx
    ]
    loops.extend(out_loops)

    write_yaml(args.out, loops)
    sys.stderr.write('{} inliers + {} outliers -> {}\n'.format(
        len(inlier_idx), len(out_loops), args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
