#!/usr/bin/env python3
"""Compare several Meridian runs against one ground truth, at a glance.

Prints a table of ATE / divergence-onset / phantom-motion per run and draws a
single overlay figure (top-down paths + cumulative distance). Built for A/B
sweeps -- config variants, estimator kinds, time-offset probes -- where the
question is "which one tracked, and where did the others go wrong".

Usage (inside the dev container):
    python3 tools/compare_runs.py --gt GT.tum --out OUTDIR \
        name1=traj1.tum name2=traj2.tum name3=traj3.tum
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_run import (load_tum, umeyama_se3, associate, speed_of,  # noqa: E402
                          V_STILL, V_PHANTOM, DIVERGE_M)


def metrics(est_path, gt):
    t_e, p_e, _ = load_tum(est_path)
    t_g, p_g, _ = gt
    pr = associate(t_e, t_g)
    if len(pr) < 3:
        return None
    se, sg = p_e[pr[:, 0]], p_g[pr[:, 1]]
    R, tt = umeyama_se3(se, sg)
    err = np.linalg.norm((R @ se.T).T + tt - sg, axis=1)
    d_e = np.linalg.norm(p_e - p_e[0], axis=1)
    d_g = np.linalg.norm(p_g - p_g[0], axis=1)
    onset = next((t_e[i] - t_e[0] for i, j in pr if d_e[i] - d_g[j] > DIVERGE_M), None)
    sp_e, sp_g = speed_of(t_e, p_e), speed_of(t_g, p_g)
    dt_e = np.median(np.diff(t_e))
    phantom = sum(sp_e[i] * dt_e for i, j in pr if sp_g[j] < V_STILL and sp_e[i] > V_PHANTOM)
    return dict(ate=float(np.sqrt((err ** 2).mean())), onset=onset,
                phantom=phantom, t=t_e, p=p_e, n=len(pr))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="/tmp/meridian_diag")
    ap.add_argument("runs", nargs="+", help="name=traj.tum ...")
    args = ap.parse_args()

    gt = load_tum(args.gt)
    results = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path or not Path(path).exists():
            print(f"skip {spec}: missing", file=sys.stderr); continue
        results[name] = metrics(path, gt)

    print(f"{'run':16s} {'ATE rmse':>10s} {'onset':>8s} {'phantom':>10s}  poses")
    print("-" * 52)
    for name, m in results.items():
        if not m:
            print(f"{name:16s}   (too few matched poses)"); continue
        print(f"{name:16s} {m['ate']:9.1f}m {('%.1fs'%m['onset']) if m['onset'] else '   --':>8s}"
              f" {m['phantom']:8.1f}m  {m['n']}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    gp = gt[1] - gt[1][0]
    ax[0].plot(gp[:, 0], gp[:, 1], "k", lw=2.5, label="GT")
    gcum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt[1], axis=0), axis=1))])
    ax[1].plot(gt[0] - gt[0][0], gcum, "k", lw=2.5, label="GT")
    for name, m in results.items():
        if not m:
            continue
        p0 = m["p"] - m["p"][0]
        ax[0].plot(p0[:, 0], p0[:, 1], lw=1.2, label=name)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(m["p"], axis=0), axis=1))])
        ax[1].plot(m["t"] - m["t"][0], cum, lw=1.2, label=name)
    ax[0].set_title("top-down paths (start at origin)"); ax[0].axis("equal")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[1].set_title("cumulative distance"); ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("m")
    for a in ax:
        a.legend(); a.grid(alpha=.3)
    plt.tight_layout()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "compare.png", dpi=110)
    print(f"\nwrote {out/'compare.png'}")


if __name__ == "__main__":
    sys.exit(main())
