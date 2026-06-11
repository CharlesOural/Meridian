#!/usr/bin/env python3
"""One standard multi-panel front-end debug figure from a Meridian run's artifacts.

Consumes the `ros2 topic echo` block captures a headless run produces (telemetry,
events, stage timing — the same files diagnose_run.py reads, or the replay_runner
sidecars, which share the format) and renders the full front-end debug surface on
one 3x4 grid:

  (1) association funnel        (2) per-family residual RMS     (3) residual histogram heatmap
  (4) solver health             (5) IMU innovation              (6) deskew / CT correction
  (7) spline kinematics         (8) IMU consistency + biases    (9) per-axis observability
 (10) queue depths / drops     (11) stage timing (top 8)       (12) visual / GNSS funnel

Panels for keys the run did not record (debug group off) render as "group off",
so the figure layout is stable across postures and runs are visually comparable.

Self-contained: numpy + matplotlib only.

Usage:
    python3 tools/plot_frontend.py --telemetry RUN_telemetry.txt \
        [--events RUN_events.txt] [--stage-timing RUN_stage.txt] \
        [--name myrun] [--out OUTDIR]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_telemetry(path):
    """Generic loader for the `ros2 topic echo` block format.

    Returns {key: (t [N], values [N, M])} with M the (maximum) vector width of the
    key; scalar keys have M == 1. Rows shorter than M are right-padded with NaN so
    a key whose width varies (e.g. per-iteration traces) still loads.
    """
    txt = Path(path).read_text(errors="replace")
    series = {}
    for block in txt.split("\n---"):
        k = re.search(r"key:\s*(\S+)", block)
        sec = re.search(r"sec:\s*(\d+)", block)
        if not (k and sec) or "values:" not in block:
            continue
        ns = re.search(r"nanosec:\s*(\d+)", block)
        t = int(sec.group(1)) + (int(ns.group(1)) * 1e-9 if ns else 0.0)
        tail = block.split("values:", 1)[1].split("axis_order")[0]
        vals = [float(x) for x in re.findall(r"-\s*([0-9.eE+\-nainf]+)", tail)]
        if not vals:
            continue
        series.setdefault(k.group(1), []).append((t, vals))
    out = {}
    for key, rows in series.items():
        rows.sort(key=lambda r: r[0])
        width = max(len(v) for _, v in rows)
        arr = np.full((len(rows), width), np.nan)
        for i, (_, v) in enumerate(rows):
            arr[i, : len(v)] = v
        out[key] = (np.array([t for t, _ in rows]), arr)
    return out


def parse_stage_timing(path):
    """{stage: (ms_avg, ms_max, count)} from a stage_timing echo or replay sidecar."""
    txt = Path(path).read_text(errors="replace")
    stages = {}
    for block in txt.split("---"):
        st = re.search(r"stage:\s*(\S+)", block)
        avg = re.search(r"ms_avg:\s*([0-9.eE+-]+)", block)
        mx = re.search(r"ms_max:\s*([0-9.eE+-]+)", block)
        cnt = re.search(r"count:\s*(\d+)", block)
        if st and avg:
            stages[st.group(1)] = (
                float(avg.group(1)),
                float(mx.group(1)) if mx else 0.0,
                int(cnt.group(1)) if cnt else 0,
            )
    return stages


def parse_events(path):
    """[(t, tag)] from an events echo."""
    txt = Path(path).read_text(errors="replace")
    out = []
    for block in txt.split("---"):
        tag = re.search(r"tag:\s*(\S+)", block)
        sec = re.search(r"sec:\s*(\d+)", block)
        if tag and sec:
            out.append((int(sec.group(1)), tag.group(1)))
    return out


def _t0_of(tele):
    t0 = None
    for t, _ in tele.values():
        if len(t):
            t0 = t[0] if t0 is None else min(t0, t[0])
    return t0 or 0.0


def _off(ax, label):
    ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes,
            fontsize=9, color="0.5")
    ax.set_xticks([])
    ax.set_yticks([])


def _line(ax, tele, key, t0, label=None, col=0, **kw):
    if key not in tele:
        return False
    t, v = tele[key]
    ax.plot(t - t0, v[:, col], label=label or key.split("/")[-1], lw=0.8, **kw)
    return True


def plot(tele, events, stages, name, outdir):
    fig, axes = plt.subplots(3, 4, figsize=(22, 11))
    fig.suptitle(f"{name}: front-end debug surface", fontsize=13)
    t0 = _t0_of(tele)

    # (1) association funnel
    ax = axes[0, 0]
    any_funnel = False
    for key, lab in [("frontend/lidar/n_input", "input"),
                     ("frontend/assoc/n_attempted", "attempted"),
                     ("frontend/assoc/n_matched", "matched"),
                     ("frontend/lidar/n_inlier", "accepted"),
                     ("frontend/lidar/n_factors_kept", "kept(cap)")]:
        any_funnel |= _line(ax, tele, key, t0, lab)
    if any_funnel:
        ax.set_title("association funnel [count]")
        ax.legend(fontsize=6)
        if "frontend/lidar/inlier_ratio" in tele:
            ax2 = ax.twinx()
            t, v = tele["frontend/lidar/inlier_ratio"]
            ax2.plot(t - t0, v[:, 0], color="k", alpha=0.4, lw=0.7)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("inlier ratio", fontsize=7)
    else:
        _off(ax, "association funnel: no data")

    # (2) per-family residual RMS
    ax = axes[0, 1]
    families = ["lidar", "imu", "bias", "anchor", "motionreg", "visual", "gnss", "prior"]
    got = False
    for fam in families:
        got |= _line(ax, tele, f"frontend/resid/{fam}/rms", t0, fam)
    if got:
        ax.set_yscale("log")
        ax.set_title("per-family residual RMS (whitened)")
        ax.legend(fontsize=6, ncol=2)
    else:
        _off(ax, "resid probe: no data")

    # (3) residual histogram heatmap
    ax = axes[0, 2]
    if "frontend/assoc/res_hist" in tele:
        t, v = tele["frontend/assoc/res_hist"]
        row_sum = v.sum(axis=1, keepdims=True)
        norm = np.divide(v, row_sum, out=np.zeros_like(v), where=row_sum > 0)
        ax.imshow(norm.T, aspect="auto", origin="lower", cmap="inferno",
                  extent=[t[0] - t0, t[-1] - t0, 0, v.shape[1]])
        ax.set_title("|r| histogram vs time (row-normalised)")
        ax.set_ylabel("bin (0 .. 2*plane_thresh)")
    else:
        _off(ax, "assoc group off:\nfrontend/assoc/res_hist")

    # (4) solver health
    ax = axes[0, 3]
    got = _line(ax, tele, "frontend/cost_total", t0, "cost_total")
    got |= _line(ax, tele, "frontend/dx_norm", t0, "dx_norm")
    if got:
        ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper left")
    ax2 = ax.twinx()
    _line(ax2, tele, "frontend/iter_count", t0, "iters", color="g", alpha=0.5)
    _line(ax2, tele, "frontend/deadline_hit", t0, "deadline", color="r", alpha=0.5)
    ax2.legend(fontsize=6, loc="upper right")
    ax.set_title("solver: cost / step / iters / deadline")

    # (5) IMU innovation
    ax = axes[1, 0]
    got = _line(ax, tele, "frontend/innov/trans_m", t0, "trans [m]")
    ax2 = ax.twinx()
    got |= _line(ax2, tele, "frontend/innov/rot_deg", t0, "rot [deg]", color="orange")
    if got:
        ax.set_title("innovation vs IMU seed")
        ax.legend(fontsize=6, loc="upper left")
        ax2.legend(fontsize=6, loc="upper right")
    else:
        _off(ax, "innovation: no data")

    # (6) deskew / CT correction
    ax = axes[1, 1]
    got = False
    for key, lab in [("frontend/deskew/corr_mean_m", "corr mean"),
                     ("frontend/deskew/corr_p95_m", "corr p95"),
                     ("frontend/deskew/corr_max_m", "corr max"),
                     ("frontend/deskew/sweep_trans_m", "sweep trans")]:
        got |= _line(ax, tele, key, t0, lab)
    if got:
        ax.set_title("deskew correction [m]")
        ax.legend(fontsize=6)
        ax2 = ax.twinx()
        _line(ax2, tele, "frontend/deskew/sweep_rot_deg", t0, "sweep rot", color="purple",
              alpha=0.5)
        ax2.set_ylabel("deg", fontsize=7)
    else:
        _off(ax, "deskew group off:\nfrontend/deskew/*")

    # (7) spline kinematics
    ax = axes[1, 2]
    got = False
    if "frontend/spline/vel" in tele:
        t, v = tele["frontend/spline/vel"]
        ax.plot(t - t0, np.linalg.norm(v[:, :3], axis=1), label="|v| [m/s]", lw=0.8)
        got = True
    for key, lab in [("frontend/spline/acc_rms_span", "|a| rms [m/s^2]"),
                     ("frontend/spline/omega_rms_span", "|w| rms [rad/s]"),
                     ("frontend/spline/jerk_rms", "jerk rms [m/s^3]")]:
        got |= _line(ax, tele, key, t0, lab)
    if got:
        ax.set_yscale("log")
        ax.set_title("spline kinematics (span RMS)")
        ax.legend(fontsize=6)
    else:
        _off(ax, "spline group off:\nfrontend/spline/*")

    # (8) IMU consistency + biases
    ax = axes[1, 3]
    got = _line(ax, tele, "frontend/imu/res_acc", t0, "res_acc [m/s^2]")
    got |= _line(ax, tele, "frontend/imu/res_gyr", t0, "res_gyr [rad/s]")
    if "frontend/bias_acc" in tele:
        t, v = tele["frontend/bias_acc"]
        for i, axis in enumerate("xyz"):
            ax.plot(t - t0, np.abs(v[:, i]), ls="--", lw=0.6, alpha=0.6,
                    label=f"|ba_{axis}|")
        got = True
    if "frontend/bias_gyr" in tele:
        t, v = tele["frontend/bias_gyr"]
        ax.plot(t - t0, np.linalg.norm(v[:, :3], axis=1), ls=":", lw=0.8, label="|bg|")
        got = True
    if got:
        ax.set_yscale("log")
        ax.set_title("IMU consistency + biases")
        ax.legend(fontsize=6, ncol=2)
    else:
        _off(ax, "IMU consistency: no data")

    # (9) per-axis observability
    ax = axes[2, 0]
    if "frontend/obs" in tele:
        t, v = tele["frontend/obs"]
        for i, axis in enumerate(["tx", "ty", "tz", "rx", "ry", "rz"][: v.shape[1]]):
            ax.plot(t - t0, v[:, i], lw=0.8, label=axis)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("per-axis observability score")
        ax.legend(fontsize=6, ncol=3)
    else:
        _off(ax, "observability: no data")

    # (10) queue depths / drops
    ax = axes[2, 1]
    got = False
    for key in ["pipeline/q_sensors_depth", "pipeline/q_meas_depth", "pipeline/q_kf_depth",
                "pipeline/q_map_depth", "pipeline/q_meas_dropped",
                "pipeline/q_meas_dropped_sweep", "pipeline/group_imu_count"]:
        got |= _line(ax, tele, key, t0)
    if got:
        ax.set_title("queue depths / drops")
        ax.legend(fontsize=6)
    else:
        _off(ax, "queues: no data")
    for t_ev, tag in events:
        if tag in ("frontend/window_restart", "frontend/sweep_gap_bridged"):
            ax.axvline(t_ev - t0, color="r" if "restart" in tag else "y", alpha=0.4, lw=0.7)

    # (11) stage timing (top 8 by avg)
    ax = axes[2, 2]
    if stages:
        top = sorted(stages.items(), key=lambda kv: -kv[1][0])[:8]
        labels = [k for k, _ in top]
        avgs = [v[0] for _, v in top]
        maxs = [v[1] for _, v in top]
        y = np.arange(len(top))
        ax.barh(y, maxs, color="0.85", label="max")
        ax.barh(y, avgs, color="tab:blue", label="avg")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_title("stage timing [ms]")
        ax.legend(fontsize=6)
    else:
        _off(ax, "stage timing: no data")

    # (12) visual / GNSS funnel
    ax = axes[2, 3]
    got = False
    for key, lab in [("frontend/visual/map_points", "map pts"),
                     ("frontend/visual/n_candidates", "candidates"),
                     ("frontend/visual/n_tracked", "tracked"),
                     ("frontend/visual/n_converged", "converged"),
                     ("frontend/gnss/innovation_m", "gnss innov [m]"),
                     ("frontend/gnss/accept_rate", "gnss accept")]:
        got |= _line(ax, tele, key, t0, lab)
    if got:
        ax.set_title("visual / GNSS funnel")
        ax.legend(fontsize=6)
    else:
        _off(ax, "visual & GNSS: inactive")

    for ax in axes.flat:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(outdir) / f"{name}_frontend.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--events", default=None)
    ap.add_argument("--stage-timing", default=None)
    ap.add_argument("--name", default="run")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    tele = parse_telemetry(args.telemetry)
    if not tele:
        print("error: no telemetry parsed", file=sys.stderr)
        return 1
    events = parse_events(args.events) if args.events else []
    stages = parse_stage_timing(args.stage_timing) if args.stage_timing else {}
    plot(tele, events, stages, args.name, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
