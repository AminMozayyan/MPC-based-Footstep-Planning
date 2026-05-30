#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate thesis-ready separate no-obstacle navigation figures.

Outputs:
  no_obstacle_straight.png/pdf
  no_obstacle_straight_yaw45.png/pdf
  no_obstacle_turning.png/pdf
  no_obstacle_long.png/pdf

Expected run directory structure:
  <root>/
    no_obstacle_straight_01_.../
      trajectory.csv
      planner.csv
    no_obstacle_straightyaw45_01_.../
      trajectory.csv
      planner.csv
    no_obstacle_turning_01_.../
      trajectory.csv
      planner.csv
    no_obstacle_long_01_.../
      trajectory.csv
      planner.csv

The script:
- plots robot trajectory
- plots start and goal
- plots replanning points using planner.csv event == replan_cycle_time
- plots desired final orientation using an annotate arrow
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


FIG_DPI = 300

SCENARIOS = [
    ("no_obstacle_straight", "Straight Motion", "no_obstacle_straight"),
    ("no_obstacle_straightyaw45", "Straight Motion with Final Orientation", "no_obstacle_straight_yaw45"),
    ("no_obstacle_turning", "Turning Motion", "no_obstacle_turning"),
    ("no_obstacle_long", "Long Path Scenario", "no_obstacle_long"),
]

# Desired final yaws in radians. Adjust if your tests use different final orientations.
GOAL_YAWS = {
    "no_obstacle_straight": 0.0,
    "no_obstacle_straightyaw45": math.radians(45.0),
    "no_obstacle_turning": math.radians(45.0),
    "no_obstacle_long": math.radians(90.0),
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
        key=natural_sort_key
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

    rp_x, rp_y = [], []
    for t in replans["t"].to_numpy():
        idx = int(np.argmin(np.abs(tx - t)))
        rp_x.append(xs[idx])
        rp_y.append(ys[idx])

    # Remove duplicate/near-duplicate points for cleaner thesis figures.
    points = []
    for x, y in zip(rp_x, rp_y):
        if not points:
            points.append((x, y))
        else:
            px, py = points[-1]
            if math.hypot(x - px, y - py) > 0.08:
                points.append((x, y))

    if not points:
        return np.array([]), np.array([])

    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def break_large_jumps(x, y, threshold=0.75):
    """Avoid connecting discontinuities in trajectory."""
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
    """Draw desired final orientation as a clean arrow next to the goal."""
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
        zorder=5,
    )


def legend_handles():
    return [
        Line2D([0], [0], color="C0", lw=2.0, label="Robot trajectory"),
        Line2D([0], [0], marker="*", color="C0", linestyle="None",
               markersize=11, label="Start"),
        Line2D([0], [0], marker="X", color="C1", linestyle="None",
               markersize=9, label="Goal"),
        Line2D([0], [0], marker="o", markerfacecolor="none",
               markeredgecolor="black", linestyle="None",
               markersize=7, label="Replanning point"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="black",
               linestyle="None", markersize=16,
               label="Desired final orientation"),
    ]


def set_clean_limits(ax, xs, ys, scenario):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]

    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    dx = xmax - xmin
    dy = ymax - ymin
    if dx < 1e-6:
        dx = 1.0
    if dy < 1e-6:
        dy = 1.0

    pad_x = 0.12 * dx
    pad_y = 0.20 * dy

    # For straight paths, keep enough y-span so the figure does not look flat.
    if scenario in ("no_obstacle_straight", "no_obstacle_straightyaw45"):
        min_y_span = 1.1
        y_center = 0.5 * (ymin + ymax)
        if (dy + 2 * pad_y) < min_y_span:
            ymin = y_center - min_y_span / 2
            ymax = y_center + min_y_span / 2
            pad_y = 0.0

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")


def plot_scenario(root, out_dir, scenario, title, output_name):
    run_dir = find_first_run(root, scenario)
    traj = load_trajectory(run_dir)
    replan_x, replan_y = load_replanning_points(run_dir, traj)

    x_raw = traj["x"].to_numpy(dtype=float)
    y_raw = traj["y"].to_numpy(dtype=float)
    x_plot, y_plot = break_large_jumps(x_raw, y_raw)

    sx = float(traj["x"].iloc[0])
    sy = float(traj["y"].iloc[0])
    gx = float(traj["goal_x"].iloc[-1])
    gy = float(traj["goal_y"].iloc[-1])

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=False)

    ax.plot(x_plot, y_plot, linewidth=2.0, label="Robot trajectory", zorder=2)
    ax.scatter([sx], [sy], marker="*", s=120, label="Start", zorder=4)
    ax.scatter([gx], [gy], marker="X", s=100, label="Goal", zorder=4)

    if len(replan_x) > 0:
        ax.scatter(
            replan_x,
            replan_y,
            marker="o",
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Replanning point",
            zorder=3,
        )

    all_x = np.concatenate([x_raw, np.array([sx, gx]), replan_x if len(replan_x) else np.array([])])
    all_y = np.concatenate([y_raw, np.array([sy, gy]), replan_y if len(replan_y) else np.array([])])
    set_clean_limits(ax, all_x, all_y, scenario)

    draw_goal_orientation(ax, gx, gy, GOAL_YAWS.get(scenario, 0.0))

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.legend(
        legend_handles(),
        [h.get_label() for h in legend_handles()],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.15, 1, 1])

    fig.savefig(out_dir / f"{output_name}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{output_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root experiment directory")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)

    for scenario, title, output_name in SCENARIOS:
        plot_scenario(args.root, out_dir, scenario, title, output_name)

    print("Saved separate figures to:", out_dir.resolve())


if __name__ == "__main__":
    main()
