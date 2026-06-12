#!/usr/bin/env python3
"""Identity-property check: corrected back-end trajectory vs front-end keyframe poses.

On an odometry-only graph (between-factors + gauge anchor, no loops/GNSS) the optimum is
exactly the composed odometry chain, so the corrected trajectory must reproduce the
front-end keyframe poses to solver/print precision. Any deviation means a bug in edge
construction, lifting, or tangent ordering.

Inputs: the packet-dump index (`kf <id> <stamp_ns> <pose>` lines = front-end poses) and a
back-end TUM file (one line per keyframe). Rows are matched in order; stamps must agree.
"""
import argparse
import math
import sys

import numpy as np


def load_index_kf(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "kf":
                continue
            kf_id = int(parts[1])
            stamp = int(parts[2]) * 1e-9
            t = np.array([float(x) for x in parts[3:6]])
            q = np.array([float(x) for x in parts[6:10]])  # x y z w
            rows.append((kf_id, stamp, t, q))
    return rows


def load_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 8 or parts[0].startswith("#"):
                continue
            rows.append((float(parts[0]), np.array([float(x) for x in parts[1:4]]),
                         np.array([float(x) for x in parts[4:8]])))
    return rows


def quat_angle(qa, qb):
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return 2.0 * math.acos(min(1.0, d))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, help="<packets.bin>.index.txt from the dump")
    ap.add_argument("--backend", required=True, help="corrected back-end TUM")
    # 2e-6 m / 1e-7 rad absorb the %.6f / %.9f print quantization of both files.
    ap.add_argument("--tol-trans", type=float, default=2e-6)
    ap.add_argument("--tol-rot", type=float, default=1e-7)
    ap.add_argument("--tol-stamp", type=float, default=1e-6)
    args = ap.parse_args()

    kf = load_index_kf(args.index)
    be = load_tum(args.backend)
    if not kf:
        print(f"FAIL: no kf lines in {args.index}")
        return 1
    if len(kf) != len(be):
        print(f"FAIL: keyframe count mismatch: index has {len(kf)}, backend TUM has {len(be)}")
        return 1

    worst_t, worst_r, worst_id = 0.0, 0.0, -1
    for (kf_id, stamp, t, q), (bt, bt_t, bt_q) in zip(kf, be):
        if abs(stamp - bt) > args.tol_stamp:
            print(f"FAIL: stamp mismatch at kf {kf_id}: index {stamp:.9f} vs backend {bt:.9f}")
            return 1
        dt = float(np.linalg.norm(t - bt_t))
        dr = quat_angle(q, bt_q)
        if dt > worst_t:
            worst_t, worst_id = dt, kf_id
        worst_r = max(worst_r, dr)

    print(f"keyframes={len(kf)} max|dt|={worst_t:.3e} m (kf {worst_id}) max|dr|={worst_r:.3e} rad")
    if worst_t > args.tol_trans or worst_r > args.tol_rot:
        print(f"FAIL: exceeds tolerance ({args.tol_trans} m / {args.tol_rot} rad)")
        return 1
    print("identity property holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
