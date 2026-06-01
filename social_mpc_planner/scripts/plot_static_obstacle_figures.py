#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate thesis-ready static-obstacle navigation figures with identical axes.

Important:
- The trajectory/obstacles are NOT translated or re-centered.
- First, the script computes the normal limits of each scenario.
- Then, it uses the union of those limits for ALL scenarios:
      xmin = min(xmin_i), xmax = max(xmax_i)
      ymin = min(ymin_i), ymax = max(ymax_i)
- This means every plot has exactly the same x/y limits and the same world scale.
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

SCENARIOS = [
    ("single_obstacle", "Single Obstacle", "single_obstacle"),
    ("multiple_obstacle", "Multiple Obstacles", "multiple_obstacle"),
    ("narrow_obstacle", "Narrow Passage", "narrow_obstacle"),
]

GOAL_YAWS = {
    "single_obstacle": 0.0,
    "multiple_obstacle": 0.0,
    "narrow_obstacle": 0.0,
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
        raise FileNotFoundError(f"No run directory found for scenario: {scenario}")
    return candidates[0]


def load_trajectory(run_dir):
    path = Path(run_dir) / "trajectory.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory.csv in {run_dir}")

    df = pd.read_csv(path)
    required = ["t", "x", "y", "theta", "goal_x", "goal_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trajectory.csv missing columns {missing} in {run_dir}")
    return df


def load_static_obstacles(run_dir):
    path = Path(run_dir) / "static_obstacles.csv"
    if not path.exists():
        return pd.DataFrame(columns=["obstacle_id", "x", "y", "radius"])

    df = pd.read_csv(path)
    required = ["x", "y", "radius"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"static_obstacles.csv missing columns {missing} in {run_dir}")

    if "obstacle_id" not in df.columns:
        df["obstacle_id"] = np.arange(len(df))

    tmp = df[["obstacle_id", "x", "y", "radius"]].copy()
    tmp["x_r"] = tmp["x"].round(4)
    tmp["y_r"] = tmp["y"].round(4)
    tmp["r_r"] = tmp["radius"].round(4)
    tmp = tmp.drop_duplicates(subset=["obstacle_id", "x_r", "y_r", "r_r"])
    return tmp[["obstacle_id", "x", "y", "radius"]].reset_index(drop=True)


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
        zorder=6,
    )


def draw_static_obstacles(ax, obstacles):
    for _, row in obstacles.iterrows():
        ox = float(row["x"])
        oy = float(row["y"])
        r = float(row["radius"])
        circle = Circle(
            (ox, oy),
            r,
            facecolor="0.82",
            edgecolor="0.25",
            linewidth=1.4,
            alpha=0.75,
            zorder=1,
        )
        ax.add_patch(circle)
        ax.scatter([ox], [oy], marker=".", s=18, color="0.25", zorder=2)


def legend_handles():
    return [
        Line2D([0], [0], color="C0", lw=2.0, label="Robot trajectory"),
        Line2D([0], [0], marker="*", color="C0", linestyle="None", markersize=11, label="Start"),
        Line2D([0], [0], marker="X", color="C1", linestyle="None", markersize=9, label="Goal"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="black", linestyle="None", markersize=7, label="Replanning point"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="black", linestyle="None", markersize=16, label="Desired final orientation"),
        Line2D([0], [0], marker="o", markerfacecolor="0.82", markeredgecolor="0.25", linestyle="None", markersize=10, label="Static obstacle"),
    ]


def collect_plot_points(root, scenario):
    run_dir = find_first_run(root, scenario)
    traj = load_trajectory(run_dir)
    obstacles = load_static_obstacles(run_dir)
    replan_x, replan_y = load_replanning_points(run_dir, traj)

    x_raw = traj["x"].to_numpy(dtype=float)
    y_raw = traj["y"].to_numpy(dtype=float)
    sx = float(traj["x"].iloc[0])
    sy = float(traj["y"].iloc[0])
    gx = float(traj["goal_x"].iloc[-1])
    gy = float(traj["goal_y"].iloc[-1])

    obstacle_x, obstacle_y = [], []
    for _, row in obstacles.iterrows():
        ox = float(row["x"])
        oy = float(row["y"])
        r = float(row["radius"])
        obstacle_x.extend([ox - r, ox + r])
        obstacle_y.extend([oy - r, oy + r])

    all_x = np.concatenate([
        x_raw,
        np.array([sx, gx]),
        replan_x if len(replan_x) else np.array([]),
        np.array(obstacle_x) if obstacle_x else np.array([]),
    ])
    all_y = np.concatenate([
        y_raw,
        np.array([sy, gy]),
        replan_y if len(replan_y) else np.array([]),
        np.array(obstacle_y) if obstacle_y else np.array([]),
    ])
    return all_x, all_y


def scenario_limits_from_points(xs, ys, scenario):
    """Same padding logic as the original script, but returns limits instead of setting them."""
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

    pad_x = 0.14 * dx
    pad_y = 0.24 * dy

    min_y_span = 1.2
    if scenario == "narrow_obstacle":
        min_y_span = 1.6
    elif scenario == "multiple_obstacle":
        min_y_span = 1.4

    y_center = 0.5 * (ymin + ymax)
    if (dy + 2 * pad_y) < min_y_span:
        ymin = y_center - min_y_span / 2
        ymax = y_center + min_y_span / 2
        pad_y = 0.0

    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def compute_common_union_limits(root):
    """Union of each scenario's own limits. No centering, no coordinate shifting."""
    limits = []
    for scenario, _, _ in SCENARIOS:
        xs, ys = collect_plot_points(root, scenario)
        limits.append(scenario_limits_from_points(xs, ys, scenario))

    xmin = min(lim[0] for lim in limits)
    xmax = max(lim[1] for lim in limits)
    ymin = min(lim[2] for lim in limits)
    ymax = max(lim[3] for lim in limits)
    return xmin, xmax, ymin, ymax


def plot_scenario(root, out_dir, scenario, title, output_name, common_limits):
    run_dir = find_first_run(root, scenario)
    traj = load_trajectory(run_dir)
    obstacles = load_static_obstacles(run_dir)
    replan_x, replan_y = load_replanning_points(run_dir, traj)

    x_raw = traj["x"].to_numpy(dtype=float)
    y_raw = traj["y"].to_numpy(dtype=float)
    x_plot, y_plot = break_large_jumps(x_raw, y_raw)

    sx = float(traj["x"].iloc[0])
    sy = float(traj["y"].iloc[0])
    gx = float(traj["goal_x"].iloc[-1])
    gy = float(traj["goal_y"].iloc[-1])

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=False)

    draw_static_obstacles(ax, obstacles)
    ax.plot(x_plot, y_plot, linewidth=2.0, label="Robot trajectory", zorder=3)
    ax.scatter([sx], [sy], marker="*", s=120, label="Start", zorder=5)
    ax.scatter([gx], [gy], marker="X", s=100, label="Goal", zorder=5)

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
            zorder=4,
        )

    xmin, xmax, ymin, ymax = common_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    draw_goal_orientation(ax, gx, gy, GOAL_YAWS.get(scenario, 0.0))

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    handles = legend_handles()
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.17, 1, 1])

    # Do not use bbox_inches="tight"; it can make final image boxes inconsistent.
    fig.savefig(out_dir / f"{output_name}.png", dpi=FIG_DPI)
    fig.savefig(out_dir / f"{output_name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root experiment directory")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    common_limits = compute_common_union_limits(args.root)

    print("Common union limits:")
    print(f"  xlim = ({common_limits[0]:.4f}, {common_limits[1]:.4f})")
    print(f"  ylim = ({common_limits[2]:.4f}, {common_limits[3]:.4f})")

    for scenario, title, output_name in SCENARIOS:
        plot_scenario(args.root, out_dir, scenario, title, output_name, common_limits)

    print("Saved static-obstacle figures to:", out_dir.resolve())


if __name__ == "__main__":
    main()
