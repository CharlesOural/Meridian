#!/usr/bin/env python3
"""Re-express Newer College TUM ground truth from the base frame into the
Alphasense-IMU frame (Meridian's estimation frame), so eval_ate.py compares
like with like.

The dataset's GT poses are T_world_base. Meridian's /meridian/odom (and the
record_tum.py output) is T_world_imu with imu = Alphasense IMU. The two differ
by the constant rig transform T_base_imu (from os_imu_lidar_transforms.yaml,
`as_imu_to_base`): a 180-degree flip about X plus a ~7.6 cm offset. Left-side
Umeyama alignment cannot absorb a *body-side* constant — under rotation it
injects up to ~2x the lever arm into ATE — so the GT itself must be moved into
the IMU frame:

    T_world_imu(t) = T_world_base(t) * T_base_imu

Reads/writes TUM rows (stamp x y z qx qy qz qw). Usage:
    python3 tools/ncd_gt_to_imu_frame.py IN.csv OUT.csv
"""
import sys

import numpy as np


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def R_to_quat(R):
    # Branch on the largest diagonal term to keep the divisor well-conditioned.
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.empty(4)
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q / np.linalg.norm(q)


# T_base_imu: as_imu_to_base from os_imu_lidar_transforms.yaml (qx-first quaternion
# [1,0,0,0] = 180 deg about X; same values for all three collections).
R_BI = quat_to_R(1.0, 0.0, 0.0, 0.0)
T_BI = np.array([0.038, -0.008, 0.065])


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 1
    n = 0
    with open(sys.argv[1]) as src, open(sys.argv[2], 'w') as dst:
        for line in src:
            s = line.split()
            if not s or line.startswith('#') or len(s) < 8:
                continue
            stamp = s[0]
            x, y, z, qx, qy, qz, qw = (float(v) for v in s[1:8])
            R_wb = quat_to_R(qx, qy, qz, qw)
            R_wi = R_wb @ R_BI
            t_wi = R_wb @ T_BI + np.array([x, y, z])
            q = R_to_quat(R_wi)
            dst.write(f'{stamp} {t_wi[0]:.9f} {t_wi[1]:.9f} {t_wi[2]:.9f} '
                      f'{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')
            n += 1
    print(f'{sys.argv[2]}: {n} poses re-expressed base -> alphasense-imu')
    return 0


if __name__ == '__main__':
    sys.exit(main())
