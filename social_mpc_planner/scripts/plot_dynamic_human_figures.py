#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate thesis-ready figures for dynamic-human interaction scenarios.

Outputs:
  human_crossing.png/pdf
  human_head_on.png/pdf
  human_overtaking.png/pdf

Expected run directory structure:
  <root>/
    human_crossing_01_.../
      trajectory.csv
      planner.csv
      humans.csv
      predicted_humans.csv          optional
    human_head_on_01_.../
      ...
    human_overtaking_01_.../
      ...

The script:
- plots robot trajectory
- plots start and goal
- plots replanning points using planner.csv event == replan_cycle_time
- plots desired final orientation using an annotate arrow
- plots human trajectory from humans.csv
- plots human start/end positions
- optionally plots the last predicted human trajectory from predicted_humans.csv
- does NOT plot ideal/reference path
- does NOT plot footsteps
- saves one clean figure per scenario
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


FIG_DPI = 300
FIG_SIZE = (6.5, 4.5)

# Fixed axis limits for all dynamic-human scenario figures.
# Requested fixed limits: x from 0 to 6 and y from -2 to 2.
GLOBAL_XLIM = (-1.0, 6.0)
GLOBAL_YLIM = (-2.0, 2.0)

SCENARIOS = [
    ("human_crossing", "Crossing Human", "human_crossing"),
    ("human_head_on", "Head-on Interaction", "human_head_on"),
    ("human_overtake", "Overtaking Interaction", "human_overtaking"),
]

GOAL_YAWS = {
    "human_crossing": 0.0,
    "human_head_on": 0.0,
    "human_overtake": 0.0,
}

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_sort_key(path):
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_first_run(root, scenario):
    root = Path(root)
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(scenario + "_")],
        key=natural_sort_key,
    )
    if not candidates:
        raise FileNotFoundError("No run directory found for scenario: {}".format(scenario))
    return candidates[0]


def load_trajectory(run_dir):
    path = Path(run_dir) / "trajectory.csv"
    if not path.exists():
        raise FileNotFoundError("Missing trajectory.csv in {}".format(run_dir))
    df = pd.read_csv(path)
    required = ["t", "x", "y", "theta", "goal_x", "goal_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("trajectory.csv missing columns {} in {}".format(missing, run_dir))
    return df


def load_humans(run_dir):
    path = Path(run_dir) / "humans.csv"
    if not path.exists():
        return pd.DataFrame(columns=["t", "person_id", "x", "y", "radius"])
    df = pd.read_csv(path)
    required = ["t", "person_id", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("humans.csv missing columns {} in {}".format(missing, run_dir))
    if "radius" not in df.columns:
        df["radius"] = 0.20
    return df.sort_values(["person_id", "t"]).reset_index(drop=True)


def load_last_predictions(run_dir):
    path = Path(run_dir) / "predicted_humans.csv"
    if not path.exists():
        return pd.DataFrame(columns=["t", "person_id", "pred_step", "x", "y", "radius"])
    df = pd.read_csv(path)
    required = ["t", "person_id", "pred_step", "x", "y"]
    missing = [c for c in required if c not in df.columns]
    if missing or df.empty:
        return pd.DataFrame(columns=["t", "person_id", "pred_step", "x", "y", "radius"])
    if "radius" not in df.columns:
        df["radius"] = 0.20
    parts = []
    for pid, g in df.groupby("person_id"):
        last_t = g["t"].max()
        parts.append(g[g["t"] == last_t].sort_values("pred_step"))
    if not parts:
        return pd.DataFrame(columns=["t", "person_id", "pred_step", "x", "y", "radius"])
    return pd.concat(parts, ignore_index=True)


def load_replanning_points(run_dir, traj_df):
    planner_path = Path(run_dir) / "planner.csv"
    if not planner_path.exists():
        return np.array([]), np.array([])
    planner = pd.read_csv(planner_path)
    if "event" not in planner.columns or "t" not in planner.columns:
        return np.array([]), np.array([])
    replans = planner[planner["event"].astype(str) == "replan_cycle_time"].copy()
    if replans.empty:
        return np.array([]), np.array([])
    tx = traj_df["t"].to_numpy()
    xs = traj_df["x"].to_numpy()
    ys = traj_df["y"].to_numpy()
    points = []
    for t in replans["t"].to_numpy():
        idx = int(np.argmin(np.abs(tx - t)))
        x, y = xs[idx], ys[idx]
        if not points or math.hypot(x - points[-1][0], y - points[-1][1]) > 0.08:
            points.append((x, y))
    if not points:
        return np.array([]), np.array([])
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def break_large_jumps(x, y, threshold=0.75):
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    if len(x) < 2:
        return x, y
    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    jump_idx = np.where(dist > threshold)[0]
    for idx in jump_idx:
        if idx + 1 < len(x):
            x[idx + 1] = np.nan
            y[idx + 1] = np.nan
    return x, y


def draw_goal_orientation(ax, gx, gy, yaw, length=None):
    if length is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        length = 0.07 * max(x1 - x0, y1 - y0)
    dx = length * math.cos(yaw)
    dy = length * math.sin(yaw)
    start = (gx - 0.02 * math.cos(yaw), gy - 0.02 * math.sin(yaw))
    end = (gx + dx, gy + dy)
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=1.8,
            mutation_scale=13,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=7,
    )


def draw_human_trajectories(ax, humans):

    if humans.empty:
        return

    for _, g in humans.groupby("person_id"):

        g = g.sort_values("t")

        hx = g["x"].to_numpy(dtype=float)
        hy = g["y"].to_numpy(dtype=float)

        radius = 0.30

        step = max(1, len(hx)//8)

        for i in range(0, len(hx), step):

            alpha = 0.3 + 0.7 * i / max(len(hx)-1, 1)

            circle = Circle(
                (hx[i], hy[i]),
                radius=radius,
                facecolor="C3",
                edgecolor="none",
                alpha=alpha,
                zorder=4
            )

            ax.add_patch(circle)

        # شروع
        ax.scatter(
            [hx[0]],
            [hy[0]],
            marker="s",
            s=80,
            color="C3",
            zorder=6
        )

        # پایان
        ax.scatter(
            [hx[-1]],
            [hy[-1]],
            marker="D",
            s=80,
            color="C3",
            zorder=6
        )

def draw_predicted_human_trajectories(ax, pred):
    if pred.empty:
        return
    for _, g in pred.groupby("person_id"):
        g = g.sort_values("pred_step")
        ax.plot(
            g["x"].to_numpy(dtype=float),
            g["y"].to_numpy(dtype=float),
            linewidth=1.8,
            linestyle="--",
            color="C4",
            alpha=0.9,
            zorder=2,
        )


def legend_handles(show_prediction=True):
    handles = [
        Line2D([0], [0], color="C0", lw=2.0, label="Robot trajectory"),
        Line2D([0], [0], marker="*", color="C0", linestyle="None", markersize=11, label="Robot start"),
        Line2D([0], [0], marker="X", color="C1", linestyle="None", markersize=9, label="Goal"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="black",
               linestyle="None", markersize=7, label="Replanning point"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="black", linestyle="None",
               markersize=16, label="Desired final orientation"),
        Line2D([0], [0], color="C3", lw=2.0, label="Human trajectory"),
        Line2D([0], [0], marker="s", color="C3", linestyle="None", markersize=8, label="Human start"),
        Line2D([0], [0], marker="D", color="C3", linestyle="None", markersize=7, label="Human end"),
    ]
    if show_prediction:
        handles.append(Line2D([0], [0], color="C4", lw=1.8, linestyle="--",
                              label="Predicted human trajectory"))
    return handles


def set_clean_limits(ax, xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    pad = 0.12 * max(dx, dy)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")




def set_fixed_limits(ax):
    """Use identical axis limits and aspect ratio for all scenarios."""
    ax.set_xlim(*GLOBAL_XLIM)
    ax.set_ylim(*GLOBAL_YLIM)
    ax.set_aspect("equal", adjustable="box")

def plot_scenario(root, out_dir, scenario, title, output_name):
    run_dir = find_first_run(root, scenario)
    traj = load_trajectory(run_dir)
    humans = load_humans(run_dir)
    pred = load_last_predictions(run_dir)
    replan_x, replan_y = load_replanning_points(run_dir, traj)

    x_raw = traj["x"].to_numpy(dtype=float)
    y_raw = traj["y"].to_numpy(dtype=float)
    x_plot, y_plot = break_large_jumps(x_raw, y_raw)

    sx = float(traj["x"].iloc[0])
    sy = float(traj["y"].iloc[0])
    gx = float(traj["goal_x"].iloc[-1])
    gy = float(traj["goal_y"].iloc[-1])

    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=False)

    ax.plot(x_plot, y_plot, linewidth=2.0, color="C0", zorder=4)
    ax.scatter([sx], [sy], marker="*", s=120, color="C0", zorder=6)
    ax.scatter([gx], [gy], marker="X", s=100, color="C1", zorder=6)

    if len(replan_x) > 0:
        ax.scatter(
            replan_x,
            replan_y,
            marker="o",
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )

    draw_predicted_human_trajectories(ax, pred)
    draw_human_trajectories(ax, humans)

    xs_all = [x_raw, np.array([sx, gx])]
    ys_all = [y_raw, np.array([sy, gy])]
    if len(replan_x) > 0:
        xs_all.append(replan_x)
        ys_all.append(replan_y)
    if not humans.empty:
        xs_all.append(humans["x"].to_numpy(dtype=float))
        ys_all.append(humans["y"].to_numpy(dtype=float))
    if not pred.empty:
        xs_all.append(pred["x"].to_numpy(dtype=float))
        ys_all.append(pred["y"].to_numpy(dtype=float))

    set_fixed_limits(ax)

    draw_goal_orientation(ax, gx, gy, GOAL_YAWS.get(scenario, 0.0))

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    handles = legend_handles(show_prediction=not pred.empty)
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.20, 1, 1])
    fig.savefig(out_dir / f"{output_name}.png", dpi=FIG_DPI)
    fig.savefig(out_dir / f"{output_name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root experiment directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--scenario", default=None,
                        help="Optional scenario prefix, e.g. human_crossing")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    scenarios = SCENARIOS
    if args.scenario is not None:
        scenarios = [s for s in SCENARIOS if s[0] == args.scenario]
        if not scenarios:
            raise ValueError("Unknown scenario: {}".format(args.scenario))

    for scenario, title, output_name in scenarios:
        plot_scenario(args.root, out_dir, scenario, title, output_name)

    print("Saved dynamic-human figures to:", out_dir.resolve())


if __name__ == "__main__":
    main()
