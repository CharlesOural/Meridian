#!/usr/bin/env python3
"""GNSS-referenced ATE of a front-end trajectory from a packet-log index.

The GNSS fixes are an independent absolute reference, so this measures front-end
drift without a separately-aligned ground-truth file. It reads the keyframe map
poses and the GNSS lat/lon/alt from a packet-log `.index.txt`, converts the fixes
to a local ENU frame (WGS84), interpolates the map-frame antenna position at each
fix instant, fits the 4-DoF datum (yaw about +z plus 3-translation, scale = 1) on
an early trusted window exactly as the back-end does, then reports the residual of
every fix against that datum. A clean front-end keeps the residual flat; drift
shows up as residual growth, and the first keyframe past --onset-thresh is the
front-end's divergence onset as seen by GNSS.

usage: gnss_ate.py <index.txt> [--lever x y z] [--fit-window-s S] [--onset-thresh M]
                    [--max-kf N] [--csv out.csv]
"""
import argparse
import sys

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    x = (n + alt_m) * cl * co
    y = (n + alt_m) * cl * so
    z = (n * (1.0 - WGS84_E2) + alt_m) * sl
    return np.stack([x, y, z], axis=-1)


def ecef_to_enu(ecef, lat0_deg, lon0_deg, ecef0):
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)
    sl, cl = np.sin(lat0), np.cos(lat0)
    so, co = np.sin(lon0), np.cos(lon0)
    # Rows are the east/north/up basis vectors expressed in ECEF.
    rot = np.array([[-so, co, 0.0],
                    [-sl * co, -sl * so, cl],
                    [cl * co, cl * so, sl]])
    return (ecef - ecef0) @ rot.T


def quat_slerp(q0, q1, u):
    # q* are [x,y,z,w]; nearest-arc slerp at fraction u in [0,1].
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th0 = np.arccos(np.clip(d, -1.0, 1.0))
    th = th0 * u
    q2 = q1 - q0 * d
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(th) + q2 * np.sin(th)


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def parse_index(path):
    kf_t, kf_p, kf_q, kf_id = [], [], [], []
    g_t, g_lla, g_kf = [], [], []
    for line in open(path):
        v = line.split()
        if not v or v[0].startswith('#'):
            continue
        if v[0] == 'kf' and len(v) >= 10:
            # kf id stamp tx ty tz qx qy qz qw rel_to kind  (a killed dump may truncate the last line)
            kf_id.append(int(v[1]))
            kf_t.append(int(v[2]) * 1e-9)
            kf_p.append([float(v[3]), float(v[4]), float(v[5])])
            kf_q.append([float(v[6]), float(v[7]), float(v[8]), float(v[9])])
        elif v[0] == 'gnss' and len(v) >= 8:
            # gnss stamp lat lon alt rank nsats nearest_kf
            g_t.append(int(v[1]) * 1e-9)
            g_lla.append([float(v[2]), float(v[3]), float(v[4])])
            g_kf.append(int(v[7]))
    order = np.argsort(kf_t)
    return (np.array(kf_id)[order], np.array(kf_t)[order], np.array(kf_p)[order],
            np.array(kf_q)[order], np.array(g_t), np.array(g_lla), np.array(g_kf))


def antenna_at(stamp, kf_t, kf_p, kf_q, lever):
    """Map-frame antenna position at a fix stamp by interpolating the bracketing keyframes."""
    j = int(np.searchsorted(kf_t, stamp))
    if j <= 0:
        p, q = kf_p[0], kf_q[0]
    elif j >= len(kf_t):
        p, q = kf_p[-1], kf_q[-1]
    else:
        span = kf_t[j] - kf_t[j - 1]
        u = 0.0 if span <= 0 else (stamp - kf_t[j - 1]) / span
        p = kf_p[j - 1] + u * (kf_p[j] - kf_p[j - 1])
        q = quat_slerp(kf_q[j - 1], kf_q[j], u)
    return p + quat_to_rot(q) @ lever


def fit_yaw_trans(enu, mp):
    """p_map ~ Rz(yaw) p_enu + t. Yaw from the horizontal cross-covariance; t in full 3-D."""
    me, mm = enu.mean(0), mp.mean(0)
    de, dm = enu - me, mp - mm
    s_xx = float((de[:, 0] * dm[:, 0] + de[:, 1] * dm[:, 1]).sum())
    s_xy = float((de[:, 0] * dm[:, 1] - de[:, 1] * dm[:, 0]).sum())
    yaw = np.arctan2(s_xy, s_xx)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = mm - rot @ me
    return yaw, rot, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('index')
    ap.add_argument('--lever', nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    help='antenna lever arm in body [m] (default 0 0 0)')
    ap.add_argument('--fit-window-s', type=float, default=90.0,
                    help='fit the datum on fixes within this many seconds of the first (default 90)')
    ap.add_argument('--onset-thresh', type=float, default=5.0,
                    help='residual [m] above which a keyframe is called the drift onset (default 5)')
    ap.add_argument('--max-kf', type=int, default=0, help='ignore fixes anchored past this kf id')
    ap.add_argument('--csv', default='', help='write per-fix (t_rel, residual) to this file')
    args = ap.parse_args()

    kf_id, kf_t, kf_p, kf_q, g_t, g_lla, g_kf = parse_index(args.index)
    if len(kf_t) < 2 or len(g_t) < 6:
        print('FATAL: need >=2 keyframes and >=6 gnss fixes')
        sys.exit(1)
    lever = np.array(args.lever)

    if args.max_kf > 0:
        keep = g_kf <= args.max_kf
        g_t, g_lla, g_kf = g_t[keep], g_lla[keep], g_kf[keep]

    # GNSS fixes -> ENU (origin at the first fix), and the matching map-frame antenna track.
    ecef0 = lla_to_ecef(*g_lla[0])
    enu = ecef_to_enu(lla_to_ecef(g_lla[:, 0], g_lla[:, 1], g_lla[:, 2]),
                      g_lla[0, 0], g_lla[0, 1], ecef0)
    ant = np.array([antenna_at(t, kf_t, kf_p, kf_q, lever) for t in g_t])

    # Fit the datum on the early trusted window, then score every fix against it.
    fit = g_t - g_t[0] <= args.fit_window_s
    if fit.sum() < 6:
        fit[:] = True
    yaw, rot, t = fit_yaw_trans(enu[fit], ant[fit])
    pred = enu @ rot.T + t
    res = np.linalg.norm(ant - pred, axis=1)
    res_h = np.linalg.norm((ant - pred)[:, :2], axis=1)

    onset_kf = -1
    over = np.where(res > args.onset_thresh)[0]
    if len(over):
        onset_kf = int(g_kf[over[0]])

    rmse = float(np.sqrt((res ** 2).mean()))
    rmse_fit = float(np.sqrt((res[fit] ** 2).mean()))
    print(f'index: {len(kf_t)} keyframes, {len(g_t)} gnss fixes')
    print(f'datum fit: yaw={np.degrees(yaw):.2f} deg on {int(fit.sum())} fixes '
          f'(first {args.fit_window_s:.0f} s)')
    print(f'ATE rmse(all)={rmse:.3f} m  rmse(fit-window)={rmse_fit:.3f} m  '
          f'median={np.median(res):.3f}  max={res.max():.3f}  horiz_rmse={np.sqrt((res_h**2).mean()):.3f}')
    print(f'drift onset: kf {onset_kf} (first residual > {args.onset_thresh:.1f} m)'
          if onset_kf >= 0 else f'drift onset: none (max residual {res.max():.2f} m)')
    print('\n  t_rel[s]  resid[m]')
    for k in np.linspace(0, len(g_t) - 1, 12).astype(int):
        print(f'  {g_t[k] - g_t[0]:8.1f}  {res[k]:8.3f}')

    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('t_rel_s,nearest_kf,residual_m,residual_h_m\n')
            for k in range(len(g_t)):
                f.write(f'{g_t[k]-g_t[0]:.3f},{g_kf[k]},{res[k]:.4f},{res_h[k]:.4f}\n')

    # One machine-parseable line for the eval wrapper / A-B sweeps.
    print(f'\nGNSS-ATE rmse={rmse:.3f} rmse_fit={rmse_fit:.3f} '
          f'yaw_deg={np.degrees(yaw):.2f} onset_kf={onset_kf} n_fix={len(g_t)}')


if __name__ == '__main__':
    main()
