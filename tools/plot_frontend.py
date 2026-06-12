#!/usr/bin/env python3
"""One standard multi-panel front-end debug figure from a Meridian run's artifacts.

Consumes the `ros2 topic echo` block captures a headless run produces (telemetry,
events, stage timing — the same files diagnose_run.py reads, or the replay_runner
sidecars, which share the format) and renders the full front-end debug surface on
one 3x4 grid:

  (1) association funnel        (2) solver GN trace             (3) gravity regularizer
  (4) state norms               (5) per-axis observability      (6) map growth
  (7) sweep span                (8) group composition           (9) queue depths / drops
 (10) stage timing (top 8)     (11) event stream               (12) keyframe cadence

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
    for key, lab in [("frontend/assoc/n_attempted", "keypoints"),
                     ("frontend/assoc/n_matched", "matched")]:
        any_funnel |= _line(ax, tele, key, t0, lab)
    if any_funnel:
        ax.set_title("association funnel [count]")
        ax.legend(fontsize=6)
        att = tele.get("frontend/assoc/n_attempted")
        mat = tele.get("frontend/assoc/n_matched")
        # Per-key rate limiting can desync the two streams; the ratio only makes
        # sense when the captured samples line up one-to-one.
        if att and mat and len(att[0]) == len(mat[0]):
            ax2 = ax.twinx()
            ratio = np.divide(mat[1][:, 0], att[1][:, 0],
                              out=np.zeros_like(mat[1][:, 0]), where=att[1][:, 0] > 0)
            ax2.plot(att[0] - t0, ratio, color="k", alpha=0.4, lw=0.7)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("match ratio", fontsize=7)
    else:
        _off(ax, "assoc group off:\nfrontend/assoc/*")

    # (2) solver GN trace
    ax = axes[0, 1]
    got = _line(ax, tele, "frontend/solver/chi", t0, "chi")
    got |= _line(ax, tele, "frontend/solver/dx_norm", t0, "dx_norm")
    if got:
        ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper left")
        ax2 = ax.twinx()
        _line(ax2, tele, "frontend/solver/gn_iters", t0, "iters", color="g", alpha=0.5)
        ax2.legend(fontsize=6, loc="upper right")
        ax.set_title("solver: chi / step / iters")
    else:
        _off(ax, "solver group off:\nfrontend/solver/*")

    # (3) gravity regularizer
    ax = axes[0, 2]
    if "frontend/lio/beta" in tele:
        t, v = tele["frontend/lio/beta"]
        beta = np.where(v[:, 0] < 0, np.nan, v[:, 0])  # negative = regularizer off
        ax.plot(t - t0, beta, label="beta", lw=0.8)
        ax.set_title("gravity regularizer")
        ax.legend(fontsize=6, loc="upper left")
        ax2 = ax.twinx()
        _line(ax2, tele, "frontend/lio/accel_var", t0, "accel var", color="orange",
              alpha=0.6)
        ax2.legend(fontsize=6, loc="upper right")
        ax2.set_ylabel("(m/s^2)^2", fontsize=7)
    else:
        _off(ax, "lio group off:\nfrontend/lio/*")

    # (4) state norms
    ax = axes[0, 3]
    got = _line(ax, tele, "frontend/state/vel_norm", t0, "|v| [m/s]")
    if got:
        ax.set_title("state: velocity + bias norms")
        ax.legend(fontsize=6, loc="upper left")
        ax2 = ax.twinx()
        got2 = _line(ax2, tele, "frontend/state/bias_gyr_norm", t0, "|bg| [rad/s]",
                     color="tab:red", alpha=0.6)
        got2 |= _line(ax2, tele, "frontend/state/bias_acc_norm", t0, "|ba| [m/s^2]",
                      color="tab:purple", alpha=0.6)
        if got2:
            ax2.legend(fontsize=6, loc="upper right")
    else:
        _off(ax, "state norms: no data")

    # (5) per-axis observability
    ax = axes[1, 0]
    if "frontend/observability" in tele:
        t, v = tele["frontend/observability"]
        for i, axis in enumerate(["tx", "ty", "tz", "rx", "ry", "rz"][: v.shape[1]]):
            ax.plot(t - t0, v[:, i], lw=0.8, label=axis)
        _line(ax, tele, "frontend/obs_min", t0, "min", color="k", ls="--")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("per-axis observability score")
        ax.legend(fontsize=6, ncol=3)
    else:
        _off(ax, "observability: no data")

    # (6) map growth
    ax = axes[1, 1]
    got = _line(ax, tele, "frontend/map/points", t0, "points")
    got |= _line(ax, tele, "frontend/map/voxels", t0, "voxels")
    if got:
        ax.set_yscale("log")
        ax.set_title("map growth [count]")
        ax.legend(fontsize=6)
    else:
        _off(ax, "map_health group off:\nfrontend/map/*")

    # (7) sweep span: the deskew window vs the driver-stamped sweep duration
    ax = axes[1, 2]
    got = _line(ax, tele, "frontend/lio/deskew_span_t_ms", t0, "deskew span")
    got |= _line(ax, tele, "sensors/lidar/sweep_duration_ms", t0, "sweep duration")
    if got:
        ax.set_title("sweep span [ms]")
        ax.legend(fontsize=6)
    else:
        _off(ax, "sweep span: no data")

    # (8) group composition
    ax = axes[1, 3]
    got = _line(ax, tele, "pipeline/group_imu_count", t0, "imu/group")
    if got:
        ax.legend(fontsize=6, loc="upper left")
    if "pipeline/group_points" in tele:
        ax2 = ax.twinx()
        _line(ax2, tele, "pipeline/group_points", t0, "points/group", color="orange",
              alpha=0.6)
        ax2.legend(fontsize=6, loc="upper right")
        got = True
    if got:
        ax.set_title("group composition")
    else:
        _off(ax, "group composition: no data")

    # (9) queue depths / drops
    ax = axes[2, 0]
    got = False
    for key in ["pipeline/q_sensors_depth", "pipeline/q_sensors_dropped",
                "pipeline/q_meas_dropped", "pipeline/q_meas_dropped_sweep",
                "pipeline/q_meas_dropped_imu"]:
        got |= _line(ax, tele, key, t0)
    if got:
        ax.set_title("queue depths / drops")
        ax.legend(fontsize=6)
    else:
        _off(ax, "queues: no data")
    for t_ev, tag in events:
        if tag in ("frontend/lio/gap", "frontend/lio/reseed"):
            ax.axvline(t_ev - t0, color="r" if "reseed" in tag else "y", alpha=0.4, lw=0.7)

    # (10) stage timing (top 8 by avg)
    ax = axes[2, 1]
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

    # (11) event stream
    ax = axes[2, 2]
    if events:
        tags = sorted({tag for _, tag in events})
        y_of = {tag: i for i, tag in enumerate(tags)}
        warn = ("gap", "reseed", "reject", "dropped", "error", "backlog")
        for t_ev, tag in events:
            col = "tab:red" if any(w in tag for w in warn) else "0.5"
            ax.plot(t_ev - t0, y_of[tag], "|", color=col, ms=8)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, fontsize=6)
        ax.set_ylim(-0.5, len(tags) - 0.5)
        ax.set_title("event stream")
    else:
        _off(ax, "events: no data")

    # (12) keyframe cadence
    ax = axes[2, 3]
    if "frontend/keyframe_count" in tele:
        t, v = tele["frontend/keyframe_count"]
        ax.step(t - t0, v[:, 0], where="post", lw=0.8, label="count")
        ax.set_title("keyframe cadence")
        ax.legend(fontsize=6, loc="upper left")
        if len(t) > 1:
            ax2 = ax.twinx()
            ax2.plot((t[1:] + t[:-1]) / 2 - t0, np.diff(t), ".", color="orange",
                     ms=2, alpha=0.6)
            ax2.set_ylabel("interval [s]", fontsize=7)
    else:
        _off(ax, "keyframes: no data")

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
