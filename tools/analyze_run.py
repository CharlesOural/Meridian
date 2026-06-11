#!/usr/bin/env python3
"""Correlate a run's trajectory error vs ground truth with the logged telemetry,
to pinpoint WHEN and WHY the live path degrades.

Parses the FileSink sidecars (`run_telemetry.txt` / `run_events.txt`) -- the
complete, unthrottled record -- not the rate-limited ROS topics. Emits:
  <out>_traj.png      top-down XY estimate-vs-GT, colored by per-pose error
  <out>_timeline.png  stacked: error(t) | solve_ms | outer_iters | obs | queues
                      | vel_norm, with drop (red) / loop (green) event marks
and a one-line verdict: divergence onset + what coincided with it.

Usage:
  analyze_run.py <run_dir> <gt.tum> [--backend] [--out <prefix>]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-sweep scalar keys the FileSink already logs (no new telemetry needed).
SCALARS = [
    "frontend/solve_ms",
    "frontend/outer_iters",
    "frontend/observability",
    "pipeline/q_sensors_depth",
    "backend/queue_depth",
    "frontend/state/vel_norm",
    "frontend/map/size",
]
DROP_TAGS = ["dropped", "overload"]
LOOP_TAGS = ["loop_accepted", "loop_rejected"]


def read_tum(path):
    a = np.loadtxt(path)
    return a[:, 0], a[:, 1:4]  # timestamps, xyz


def associate(t_ref, t_est, max_diff=0.02):
    """For each est stamp, the nearest ref stamp within max_diff (TUM convention)."""
    j = np.clip(np.searchsorted(t_ref, t_est), 1, len(t_ref) - 1)
    pick_left = (t_est - t_ref[j - 1]) < (t_ref[j] - t_est)
    near = np.where(pick_left, j - 1, j)
    keep = np.abs(t_ref[near] - t_est) < max_diff
    return np.arange(len(t_est))[keep], near[keep]


def umeyama(src, dst):
    """Rigid (rotation+translation, no scale) fit src->dst, the evo -a alignment."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d) / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_d - R @ mu_s


def traj_error(est_path, gt_path):
    te, pe = read_tum(est_path)
    tr, pr = read_tum(gt_path)
    ie, ir = associate(tr, te)
    pe_m, pr_m, te_m = pe[ie], pr[ir], te[ie]
    R, t = umeyama(pe_m, pr_m)
    pe_al = (R @ pe_m.T).T + t
    err = np.linalg.norm(pe_al - pr_m, axis=1)
    return te_m, err, pe_al, pr_m


def parse_scalars(path, keys):
    data = {k: ([], []) for k in keys}
    sec = ns = key = val = None
    inv = False
    with open(path) as f:
        for line in f:
            if line.startswith("  sec:"):
                sec = int(line[6:])
            elif line.startswith("  nanosec:"):
                ns = int(line[10:])
            elif line.startswith("key:"):
                key = line[4:].strip()
            elif line.startswith("values:"):
                inv, val = True, None
            elif inv and line.startswith("- "):
                try:
                    val = float(line[2:])
                except ValueError:
                    val = None
                inv = False
            elif line.startswith("---"):
                if key in data and sec is not None and val is not None:
                    data[key][0].append(sec + ns * 1e-9)
                    data[key][1].append(val)
                sec = ns = key = val = None
                inv = False
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in data.items()}


def parse_events(path, substrs):
    out = []
    sec = ns = tag = None
    with open(path) as f:
        for line in f:
            if line.startswith("  sec:"):
                sec = int(line[6:])
            elif line.startswith("  nanosec:"):
                ns = int(line[10:])
            elif line.startswith("tag:"):
                tag = line[4:].strip()
            elif line.startswith("---"):
                if tag and any(s in tag for s in substrs):
                    out.append((sec + ns * 1e-9, tag))
                sec = ns = tag = None
    return out


def detect_onset(t, err):
    """First stamp where the error crosses a robust threshold and STAYS above it
    (sustained divergence), vs a single excursion. Threshold is relative to the
    healthy early-run baseline so a clean run never trips."""
    n = len(err)
    baseline = float(np.median(err[: max(1, n // 4)]))
    thr = max(1.0, 8.0 * baseline)
    for i in np.where(err > thr)[0]:
        if float(np.mean(err[i:])) > thr:  # the rest stays bad => real divergence
            return float(t[i]), thr
    return None, thr


def near(times, t0, window=2.0):
    times = np.asarray(times)
    return times[np.abs(times - t0) <= window]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("gt")
    ap.add_argument("--backend", action="store_true", help="use run_backend.tum")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tum = os.path.join(args.run_dir, "run_backend.tum" if args.backend else "run.tum")
    tele = os.path.join(args.run_dir, "run_telemetry.txt")
    evp = os.path.join(args.run_dir, "run_events.txt")
    out = args.out or os.path.join(args.run_dir, "analysis")

    t, err, est_xy, gt_xy = traj_error(tum, args.gt)
    t0 = t[0]
    tr = t - t0
    onset, thr = detect_onset(t, err)
    sc = parse_scalars(tele, SCALARS) if os.path.exists(tele) else {}
    drops = parse_events(evp, DROP_TAGS) if os.path.exists(evp) else []
    loops = parse_events(evp, LOOP_TAGS) if os.path.exists(evp) else []

    # --- trajectory overlay ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color="0.6", lw=1, label="GT")
    s = ax.scatter(est_xy[:, 0], est_xy[:, 1], c=err, s=6, cmap="inferno", label="est")
    fig.colorbar(s, ax=ax, label="error [m]")
    if onset is not None:
        oi = int(np.argmin(np.abs(t - onset)))
        ax.plot(est_xy[oi, 0], est_xy[oi, 1], "*", ms=18, mec="k", mfc="cyan",
                label=f"onset t+{onset - t0:.0f}s")
    ax.set_aspect("equal")
    ax.set_title(f"{os.path.basename(args.run_dir)} — RMSE {np.sqrt((err**2).mean()):.3f} m")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out + "_traj.png", dpi=110)
    plt.close(fig)

    # --- stacked diagnostic timeline ---
    panels = [("error [m]", t, err)]
    for k in SCALARS:
        if k in sc and len(sc[k][0]):
            panels.append((k.split("/")[-1], sc[k][0], sc[k][1]))
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 1.7 * len(panels)),
                             sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, tt, vv) in zip(axes, panels):
        ax.plot(np.asarray(tt) - t0, vv, lw=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        for dt, _ in drops:
            ax.axvline(dt - t0, color="red", alpha=0.25, lw=0.6)
        for lt, _ in loops:
            ax.axvline(lt - t0, color="green", alpha=0.4, lw=0.8)
        if onset is not None:
            ax.axvline(onset - t0, color="blue", ls="--", lw=1)
    axes[-1].set_xlabel("time since start [s]  (red=drop, green=loop, blue=onset)")
    axes[0].set_title(os.path.basename(args.run_dir))
    fig.tight_layout()
    fig.savefig(out + "_timeline.png", dpi=110)
    plt.close(fig)

    # --- verdict ---
    rmse = float(np.sqrt((err**2).mean()))
    parts = [f"{os.path.basename(args.run_dir)}: RMSE={rmse:.3f}m max={err.max():.2f}m"]
    if onset is None:
        parts.append("CLEAN" if rmse < 0.30
                     else f"UNIFORMLY ELEVATED ~{rmse:.2f}m (degraded solve quality, "
                          "no divergence event)")
    else:
        coinc = []
        if "frontend/solve_ms" in sc and len(sc["frontend/solve_ms"][0]):
            st, sv = sc["frontend/solve_ms"]
            w = sv[np.abs(st - onset) <= 2.0]
            if len(w) and w.max() > 3 * np.median(sv):
                coinc.append(f"solve_ms spike {w.max():.0f}ms (med {np.median(sv):.0f})")
        if "frontend/outer_iters" in sc and len(sc["frontend/outer_iters"][0]):
            it, iv = sc["frontend/outer_iters"]
            w = iv[np.abs(it - onset) <= 2.0]
            if len(w) and w.min() < 0.5 * np.median(iv):
                coinc.append(f"iters drop {w.min():.0f} (med {np.median(iv):.0f})")
        if len(near([d[0] for d in drops], onset)):
            coinc.append("SWEEP/IMU DROP")
        if len(near([l[0] for l in loops], onset)):
            coinc.append("loop fold")
        parts.append(f"onset t+{onset - t0:.0f}s; coincident: "
                     + (", ".join(coinc) if coinc else "NONE (→ ikd-Tree race / silent)"))
    print(" | ".join(parts))


if __name__ == "__main__":
    main()
