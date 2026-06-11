#!/usr/bin/env python3
"""One-shot diagnosis of a Meridian odometry run against ground truth.

Consumes the artifacts a headless run produces (a TUM trajectory, optionally a
ground-truth TUM, the `/meridian/events` echo, and the `/meridian/stage_timing`
echo) and emits a text report plus a diagnostic figure. The point is to answer,
in order and without hand-written analysis:

  1. Is the reference trustworthy?  (GT orientation present, timestamps sane,
     time overlap with the estimate) -- a guard against drawing conclusions from
     an identity-quaternion or misaligned GT.
  2. What did the platform physically do?  (stationary / in-place-turn / forward
     segmentation from the GT position, before any estimator interpretation.)
  3. How does the estimate compare?  (ATE via SE3 Umeyama, divergence onset, and
     phantom-motion: the estimate translating while the platform is still.)
  4. Is the pipeline healthy?  (drops / restarts / deadline / degeneracy from the
     event stream; per-stage wall time from stage_timing.)

Self-contained: numpy + matplotlib only (no evo dependency).

Usage (inside the dev container):
    python3 tools/diagnose_run.py --est traj.tum [--gt GT.tum] \
        [--events events.txt] [--stage-timing stage_timing.txt] \
        [--name my_run] [--out OUTDIR]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Thresholds (tune per platform; defaults suit a ~10 Hz walking dataset).
V_STILL = 0.12      # [m/s] below this the platform is treated as stationary
V_PHANTOM = 0.30    # [m/s] estimate speed while GT still that counts as phantom
TURN_DEG = 20.0     # heading change across a pause to call it an in-place turn
DIVERGE_M = 5.0     # |EST_disp - GT_disp| that marks divergence onset


def load_tum(path):
    a = np.loadtxt(path)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    # Drop pre-init rows (zero/garbage stamps); they poison every t0-relative metric.
    a = a[a[:, 0] > 1.0]
    return a[:, 0], a[:, 1:4], a[:, 4:8]   # t, xyz, quat(xyzw)


def speed_of(t, xyz, smooth_s=0.0):
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    sp = np.linalg.norm(np.gradient(xyz, axis=0), axis=1) / max(dt, 1e-6)
    if smooth_s > 0 and len(sp) > 3:
        w = max(1, int(round(smooth_s / max(dt, 1e-6))))
        sp = np.convolve(sp, np.ones(w) / w, mode="same")
    return sp


def heading_of(xyz):
    v = np.gradient(xyz[:, :2], axis=0)
    return np.degrees(np.arctan2(v[:, 1], v[:, 0]))


def umeyama_se3(src, dst):
    """Rigid (R,t) least-squares aligning src onto dst (no scale). Nx3 each."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    H = S.T @ D / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = mu_d - R @ mu_s
    return R, t


def associate(t_est, t_gt, max_dt=0.05):
    """Index pairs (i_est, j_gt) matched by nearest timestamp within max_dt."""
    pairs = []
    j = 0
    for i, te in enumerate(t_est):
        k = np.searchsorted(t_gt, te)
        best, bj = max_dt, -1
        for cand in (k - 1, k):
            if 0 <= cand < len(t_gt) and abs(t_gt[cand] - te) < best:
                best, bj = abs(t_gt[cand] - te), cand
        if bj >= 0:
            pairs.append((i, bj))
    return np.array(pairs, dtype=int) if pairs else np.empty((0, 2), int)


def parse_observability(path):
    """Time-series of the per-axis observability score from a telemetry echo.

    The front-end publishes one `frontend/obs` vector per solve (values ordered
    tx,ty,tz,rx,ry,rz) plus a `frontend/obs_min` scalar. Returns (tmin, {axis:
    [(t,score)]}) or None. A score near 0 on an axis = an unconstrained DOF."""
    txt = Path(path).read_text()
    order = ["tx", "ty", "tz", "rx", "ry", "rz"]
    axes = {a: [] for a in order}
    tmin = []
    for b in txt.split("---"):
        k = re.search(r"key:\s*(frontend/obs\w*)", b)
        if not k:
            continue
        st = re.search(r"sec:\s*(\d+)", b)
        ns = re.search(r"nanosec:\s*(\d+)", b)
        if not st:
            continue
        t = int(st.group(1)) + (int(ns.group(1)) * 1e-9 if ns else 0)
        vals = [float(x) for x in re.findall(r"-\s*([0-9.eE+-]+)", b.split("values:")[-1].split("axis_order")[0])] \
            if "values:" in b else []
        if k.group(1) == "frontend/obs_min" and vals:
            tmin.append((t, vals[0]))
        elif k.group(1) == "frontend/obs" and len(vals) >= 6:
            for i, a in enumerate(order):
                axes[a].append((t, vals[i]))
    axes = {a: v for a, v in axes.items() if v}
    if not tmin and not axes:
        return None
    return tmin, axes


def validate_gt(t_gt, q_gt, t_est):
    out = []
    ident = np.mean(np.all(np.round(q_gt, 4) == [0, 0, 0, 1], axis=1))
    orient_ok = ident < 0.99
    out.append(("GT poses", f"{len(t_gt)}"))
    out.append(("GT time span", f"{t_gt[-1]-t_gt[0]:.1f} s"))
    out.append(("orientation", "VALID" if orient_ok
                else f"IDENTITY-ONLY ({ident*100:.0f}%) -> heading metrics DISABLED"))
    mono = bool(np.all(np.diff(t_gt) > 0))
    out.append(("timestamps monotonic", str(mono)))
    overlap = (max(t_gt[0], t_est[0]), min(t_gt[-1], t_est[-1]))
    span = overlap[1] - overlap[0]
    out.append(("EST/GT time overlap", f"{span:.1f} s"
                + ("" if span > 1 else "  *** NO OVERLAP -- check clocks ***")))
    return out, orient_ok


MIN_SEG_S = 0.8     # merge motion segments shorter than this into their neighbour


def segment_motion(t, xyz):
    """Return a list of (kind, t_start, t_end, heading_deg) over the GT path.

    kind in {stationary, turn, forward}. A stationary span flanked by forward
    spans whose mean headings differ by > TURN_DEG is reported as an in-place turn.
    Speed is smoothed and short segments are merged so gait-induced speed ripple
    does not shred a steady walk into dozens of fragments.
    """
    sp = speed_of(t, xyz, smooth_s=0.7)
    moving = sp > V_STILL
    segs = []
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and moving[j] == moving[i]:
            j += 1
        segs.append([moving[i], i, j - 1])
        i = j
    # Merge any segment shorter than MIN_SEG_S into the previous one (absorb ripple).
    merged = []
    for mv, a, b in segs:
        if merged and (t[b] - t[a]) < MIN_SEG_S:
            merged[-1][2] = b
        elif merged and merged[-1][0] == mv:
            merged[-1][2] = b
        else:
            merged.append([mv, a, b])
    segs = merged
    out = []
    for idx, (mv, a, b) in enumerate(segs):
        if mv:
            net = xyz[b, :2] - xyz[a, :2]
            hd = np.degrees(np.arctan2(net[1], net[0])) if np.linalg.norm(net) > 0.05 else float("nan")
            out.append(("forward", t[a], t[b], hd))
        else:
            prev_h = next_h = float("nan")
            for k in range(idx - 1, -1, -1):
                if out and out[-1][0] == "forward":
                    prev_h = out[-1][3]
                    break
            for k in range(idx + 1, len(segs)):
                if segs[k][0]:
                    na, nb = segs[k][1], segs[k][2]
                    net = xyz[nb, :2] - xyz[na, :2]
                    if np.linalg.norm(net) > 0.05:
                        next_h = np.degrees(np.arctan2(net[1], net[0]))
                    break
            turned = (not np.isnan(prev_h) and not np.isnan(next_h)
                      and abs((next_h - prev_h + 180) % 360 - 180) > TURN_DEG)
            out.append(("turn" if turned else "stationary", t[a], t[b], next_h - prev_h
                        if turned else float("nan")))
    return out


def diagnose(args):
    t_e, p_e, q_e = load_tum(args.est)
    rep = [f"=== DIAGNOSIS: {args.name} ===",
           f"estimate: {args.est}  ({len(t_e)} poses, {t_e[-1]-t_e[0]:.1f} s)"]

    gt = None
    orient_ok = False
    if args.gt and Path(args.gt).exists():
        t_g, p_g, q_g = load_tum(args.gt)
        gt = (t_g, p_g, q_g)
        rep.append("\n[1] GROUND-TRUTH VALIDATION")
        rows, orient_ok = validate_gt(t_g, q_g, t_e)
        for k, v in rows:
            rep.append(f"    {k:28s}: {v}")

        rep.append("\n[2] MOTION PROFILE (from GT position, clipped to run window)")
        # Clip GT to the estimate's time window so the profile describes this run.
        win = (t_g >= t_e[0]) & (t_g <= t_e[-1])
        t_gw, p_gw = (t_g[win], p_g[win]) if win.sum() > 3 else (t_g, p_g)
        path = np.sum(np.linalg.norm(np.diff(p_gw, axis=0), axis=1))
        disp = np.linalg.norm(p_gw[-1] - p_gw[0])
        rep.append(f"    total path={path:.1f} m  displacement={disp:.1f} m")
        for kind, ta, tb, hd in segment_motion(t_gw - t_g[0], p_gw):
            extra = (f"  heading={hd:+.0f} deg" if kind == "forward" and not np.isnan(hd)
                     else f"  turn={hd:+.0f} deg" if kind == "turn" else "")
            rep.append(f"    {ta:6.1f}-{tb:6.1f}s  {kind:10s}{extra}")

        # [3] estimate vs GT
        rep.append("\n[3] ESTIMATE vs GROUND TRUTH")
        pr = associate(t_e, t_g, args.max_dt)
        if len(pr) < 3:
            rep.append("    too few time-associated pairs for ATE")
        else:
            se, sg = p_e[pr[:, 0]], p_g[pr[:, 1]]
            R, tt = umeyama_se3(se, sg)
            agn = (R @ se.T).T + tt
            err = np.linalg.norm(agn - sg, axis=1)
            rep.append(f"    ATE (SE3-aligned): rmse={np.sqrt((err**2).mean()):.2f}  "
                       f"mean={err.mean():.2f}  max={err.max():.2f} m  (n={len(pr)})")
            # divergence onset
            d_e = np.linalg.norm(p_e - p_e[0], axis=1)
            d_g = np.linalg.norm(p_g - p_g[0], axis=1)
            onset = None
            for i, j in pr:
                if d_e[i] - d_g[j] > DIVERGE_M:
                    onset = t_e[i] - t_e[0]
                    break
            rep.append(f"    divergence onset (|EST-GT disp|>{DIVERGE_M:.0f} m): "
                       + (f"t={onset:.1f} s" if onset else "none in window"))
            # phantom motion while GT still
            sp_e = speed_of(t_e, p_e)
            sp_g = speed_of(t_g, p_g)
            phantom_t = phantom_d = 0.0
            dt_e = np.median(np.diff(t_e))
            for i, j in pr:
                if sp_g[j] < V_STILL and sp_e[i] > V_PHANTOM:
                    phantom_t += dt_e
                    phantom_d += sp_e[i] * dt_e
            rep.append(f"    PHANTOM MOTION (EST moving while GT still): "
                       f"{phantom_t:.1f} s, ~{phantom_d:.1f} m of bogus travel")
            fin = agn[-1] - sg[-1]
            rep.append(f"    final aligned error: x={fin[0]:+.1f} y={fin[1]:+.1f} z={fin[2]:+.1f} m")

    # [4] pipeline health
    rep.append("\n[4] PIPELINE HEALTH")
    if args.events and Path(args.events).exists():
        txt = Path(args.events).read_text()
        for label, pat in [("dropped whole sweep", r"dropped a whole sweep"),
                           ("window restart/reseed", r"window_restart|reseed|exceeds IMU horizon"),
                           ("deadline hit", r"deadline"),
                           ("degeneracy", r"degener"),
                           ("health dropout", r"\bdropout\b")]:
            n = len(re.findall(pat, txt))
            if n:
                rep.append(f"    {label:24s}: {n}")
    else:
        rep.append("    (no events file)")
    if args.stage_timing and Path(args.stage_timing).exists():
        last = {}
        for b in Path(args.stage_timing).read_text().split("---"):
            s = re.search(r"stage:\s*(\S+)", b)
            a = re.search(r"ms_avg:\s*([0-9.]+)", b)
            if s and a:
                last[s.group(1)] = float(a.group(1))
        rep.append("    stage timing (ms_avg, top 16 incl. sub-ops):")
        for k, v in sorted(last.items(), key=lambda kv: -kv[1])[:16]:
            rep.append(f"        {k:36s} {v:7.2f} ms")

    # [5] observability (degeneracy detector output over time)
    obs = None
    if args.telemetry and Path(args.telemetry).exists():
        obs = parse_observability(args.telemetry)
        rep.append("\n[5] OBSERVABILITY (per-axis score, ~0 = unconstrained DOF)")
        if obs and obs[0]:
            tmin = np.array(obs[0])
            t0 = tmin[0, 0]
            rep.append(f"    min-axis score: median={np.median(tmin[:,1]):.2f}  "
                       f"floor={tmin[:,1].min():.2f}  "
                       f"frac<0.1(degenerate)={np.mean(tmin[:,1]<0.1)*100:.0f}%")
            for name, ser in (obs[1] or {}).items():
                a = np.array(ser)
                rep.append(f"    axis {name}: median={np.median(a[:,1]):.2f}  min={a[:,1].min():.2f}")
        else:
            rep.append("    (no frontend/obs/* telemetry found)")

    report = "\n".join(rep)
    print(report)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.name}_report.txt").write_text(report + "\n")
    _plot(args, t_e, p_e, gt, obs, outdir)
    print(f"\nwrote {outdir/(args.name+'_report.txt')} and {outdir/(args.name+'_diag.png')}")
    return 0


def _plot(args, t_e, p_e, gt, obs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 4 if (obs and obs[0]) else 3
    fig, ax = plt.subplots(1, ncol, figsize=(6 * ncol, 6))
    series = [("EST", t_e - t_e[0], p_e, "tab:red")]
    if gt:
        series.insert(0, ("GT", gt[0] - gt[0][0], gt[1], "k"))
    for name, t, p, c in series:
        p0 = p - p[0]
        ax[0].plot(p0[:, 0], p0[:, 1], c, label=name, lw=2 if name == "GT" else 1.2)
        sp = np.clip(speed_of(t, p), 0, 12)
        ax[1].plot(t, sp, c, label=name, lw=2 if name == "GT" else 1)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        ax[2].plot(t, cum, c, label=name, lw=2 if name == "GT" else 1)
    ax[0].scatter([0], [0], c="green", s=40, zorder=5)
    ax[0].set_title(f"{args.name}: top-down path"); ax[0].axis("equal")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[1].set_title("speed vs time"); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("m/s")
    ax[2].set_title("cumulative distance"); ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("m")
    if obs and obs[0]:
        for name, ser in (obs[1] or {}).items():
            a = np.array(ser)
            ax[3].plot(a[:, 0] - a[0, 0], a[:, 1], lw=1, label=name)
        ax[3].axhline(0.1, color="red", ls="--", lw=1, label="degenerate<0.1")
        ax[3].set_title("observability per axis (0=unconstrained)")
        ax[3].set_xlabel("t [s]"); ax[3].set_ylabel("score"); ax[3].set_ylim(-0.05, 1.05)
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{args.name}_diag.png", dpi=110)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--events", default=None)
    ap.add_argument("--stage-timing", default=None)
    ap.add_argument("--telemetry", default=None)
    ap.add_argument("--name", default="run")
    ap.add_argument("--out", default="/tmp/meridian_diag")
    ap.add_argument("--max-dt", type=float, default=0.05)
    return diagnose(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
