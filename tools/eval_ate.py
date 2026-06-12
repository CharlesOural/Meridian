#!/usr/bin/env python3
"""ATE evaluation of a TUM trajectory against ground truth.

Associates poses by nearest timestamp (max 50 ms), aligns with Umeyama SE(3)
(rotation+translation, no scale), reports RMSE/median/max ATE and the per-axis
error envelope. Usage: eval_ate.py <gt_tum> <est_tum> [t_max_offset_s]
"""
import sys
import numpy as np


def load_tum(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        v = line.split()
        if len(v) < 8:
            continue
        rows.append([float(x) for x in v[:8]])
    a = np.array(rows)
    a = a[np.argsort(a[:, 0])]
    # drop duplicate stamps (recorder can echo a pose twice)
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 1e-6])
    a = a[keep]
    return a[:, 0], a[:, 1:4], a[:, 4:8]  # t, xyz, qxyzw


def associate(t_gt, t_est, max_dt=0.05):
    pairs = []
    j = 0
    for i, t in enumerate(t_est):
        # advance gt pointer to the closest stamp
        while j + 1 < len(t_gt) and abs(t_gt[j + 1] - t) < abs(t_gt[j] - t):
            j += 1
        if abs(t_gt[j] - t) <= max_dt:
            pairs.append((j, i))
    return pairs


def umeyama(src, dst):
    """R, t minimizing ||R src + t - dst||^2 (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    return R, t


def main():
    gt_path, est_path = sys.argv[1], sys.argv[2]
    t_gt, p_gt, _ = load_tum(gt_path)
    t_est, p_est, _ = load_tum(est_path)
    print(f"gt: {len(t_gt)} poses [{t_gt[0]:.3f} .. {t_gt[-1]:.3f}]")
    print(f"est: {len(t_est)} poses [{t_est[0]:.3f} .. {t_est[-1]:.3f}]")
    pairs = associate(t_gt, t_est)
    if len(pairs) < 3:
        print("FATAL: <3 associated pairs (timestamp overlap problem?)")
        sys.exit(1)
    gi = np.array([p[0] for p in pairs])
    ei = np.array([p[1] for p in pairs])
    A, B = p_est[ei], p_gt[gi]   # align est -> gt
    R, t = umeyama(A, B)
    A_al = A @ R.T + t
    err = np.linalg.norm(A_al - B, axis=1)
    print(f"\nassociated pairs: {len(pairs)}")
    print(f"ATE  rmse:   {np.sqrt((err**2).mean()):8.3f} m")
    print(f"ATE  mean:   {err.mean():8.3f} m")
    print(f"ATE  median: {np.median(err):8.3f} m")
    print(f"ATE  max:    {err.max():8.3f} m")
    d = A_al - B
    print(f"per-axis |err| p95: x={np.percentile(abs(d[:,0]),95):.3f} "
          f"y={np.percentile(abs(d[:,1]),95):.3f} z={np.percentile(abs(d[:,2]),95):.3f}")
    # error growth over time: print a sparse profile
    print("\n  t_rel[s]  err[m]")
    for k in np.linspace(0, len(pairs) - 1, 12).astype(int):
        print(f"  {t_est[ei[k]] - t_est[ei[0]]:8.1f}  {err[k]:8.3f}")


if __name__ == '__main__':
    main()
