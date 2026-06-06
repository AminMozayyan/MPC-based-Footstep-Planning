#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate thesis-ready figures for dense multi-human navigation scenarios.

Outputs:
  dense_3human.png/pdf
  dense_5human.png/pdf
  dense_7human.png/pdf   optional

Expected run directory structure:
  <root>/
    dense_3human_01_.../
      trajectory.csv
      planner.csv
      humans.csv
      predicted_humans.csv          optional
    dense_5human_01_.../
      ...
    dense_7human_01_.../
      ...

Main features:
- fixed and identical axis limits for all scenarios
- larger figure size for thesis use
- robot trajectory, start, goal, replanning points, final orientation
- multiple human trajectories shown as semi-transparent circles along the path
- human start/end markers
- optional latest predicted human trajectories
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

# Bigger than the previous version so the title and legend fit properly.
FIG_SIZE = (8.2, 5.6)

# Fixed axis limits for all dense-human figures.
# Change these if your dense scenarios use a larger map.
GLOBAL_XLIM = (-1.0, 6.0)
GLOBAL_YLIM = (-3.5, 2.0)

SCENARIOS = [
    ("dense_3human", "Dense Scenario: 3 Humans", "dense_3human"),
    ("dense_5human", "Dense Scenario: 5 Humans", "dense_5human"),
    ("dense_1human", "Dense Scenario: 1 Humans", "dense_1human"),
]

GOAL_YAWS = {
    "dense_3human": 0.0,
    "dense_5human": 0.0,
    "dense_1human": 0.0,
}

# Distinct colors for different humans.
HUMAN_COLORS = ["C3", "C4", "C5", "C6", "C7", "C8", "C9"]


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
        df["radius"] = 0.25

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
        df["radius"] = 0.30

    parts = []
    for _, g in df.groupby("person_id"):
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
        zorder=8,
    )


def draw_human_trajectories(ax, humans, radius_override=0.30, use_distinct_colors=True):
    """
    Draw human motion similar to the previous dynamic-human plotting code:
    semi-transparent disks along the observed trajectory, plus start/end markers.

    For dense scenarios, each person_id can use a distinct color.
    """
    if humans.empty:
        return

    for idx, (pid, g) in enumerate(humans.groupby("person_id")):
        g = g.sort_values("t")

        hx = g["x"].to_numpy(dtype=float)
        hy = g["y"].to_numpy(dtype=float)

        if len(hx) == 0:
            continue

        color = HUMAN_COLORS[idx % len(HUMAN_COLORS)] if use_distinct_colors else "C3"

        radius = radius_override
        if "radius" in g.columns and np.isfinite(g["radius"].iloc[0]):
            r_value = float(g["radius"].iloc[0])
            if r_value > 0.0:
                radius = r_value

        # Same visual logic as your previous code:
        # sparse semi-transparent circles along the path, with increasing alpha.
        step = max(1, len(hx) // 20)

        for i in range(0, len(hx), step):
            alpha = 0.25 + 0.60 * i / max(len(hx) - 1, 1)
            circle = Circle(
                (hx[i], hy[i]),
                radius=radius,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=3,
            )
            ax.add_patch(circle)

        # A thin centerline helps the path remain readable when there are many people.
        ax.plot(
            hx,
            hy,
            linewidth=1.2,
            color=color,
            alpha=0.75,
            zorder=4,
        )

        # Human start and end markers.
        ax.scatter([hx[0]], [hy[0]], marker="s", s=75, color=color, zorder=6)
        ax.scatter([hx[-1]], [hy[-1]], marker="D", s=75, color=color, zorder=6)


def draw_predicted_human_trajectories(ax, pred, use_distinct_colors=True):
    if pred.empty:
        return

    for idx, (pid, g) in enumerate(pred.groupby("person_id")):
        g = g.sort_values("pred_step")
        color = HUMAN_COLORS[idx % len(HUMAN_COLORS)] if use_distinct_colors else "C4"

        ax.plot(
            g["x"].to_numpy(dtype=float),
            g["y"].to_numpy(dtype=float),
            linewidth=1.4,
            linestyle="--",
            color=color,
            alpha=0.75,
            zorder=2,
        )


def set_fixed_limits(ax):
    ax.set_xlim(*GLOBAL_XLIM)
    ax.set_ylim(*GLOBAL_YLIM)
    ax.set_aspect("equal", adjustable="box")


def legend_handles(show_prediction=False):
    handles = [
        Line2D([0], [0], color="C0", lw=2.0, label="Robot trajectory"),
        Line2D([0], [0], marker="*", color="C0", linestyle="None",
               markersize=11, label="Robot start"),
        Line2D([0], [0], marker="X", color="C1", linestyle="None",
               markersize=9, label="Goal"),
        Line2D([0], [0], marker="o", markerfacecolor="none",
               markeredgecolor="black", linestyle="None",
               markersize=7, label="Replanning point"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="black",
               linestyle="None", markersize=16,
               label="Desired final orientation"),
        Line2D([0], [0], marker="o", color="C3", linestyle="None",
               markersize=9, label="Human trajectory"),
        Line2D([0], [0], marker="s", color="C3", linestyle="None",
               markersize=7, label="Human start"),
        Line2D([0], [0], marker="D", color="C3", linestyle="None",
               markersize=7, label="Human end"),
    ]

    if show_prediction:
        handles.append(
            Line2D([0], [0], color="0.35", lw=1.4, linestyle="--",
                   label="Predicted human trajectory")
        )

    return handles


def plot_scenario(
    root,
    out_dir,
    scenario,
    title,
    output_name,
    show_predictions=False,
    use_distinct_colors=True,
    no_title=False,
):
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

    ax.plot(x_plot, y_plot, linewidth=2.1, color="C0", zorder=5)
    ax.scatter([sx], [sy], marker="*", s=125, color="C0", zorder=8)
    ax.scatter([gx], [gy], marker="X", s=105, color="C1", zorder=8)

    if len(replan_x) > 0:
        ax.scatter(
            replan_x,
            replan_y,
            marker="o",
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=1.15,
            zorder=7,
        )

    if show_predictions:
        draw_predicted_human_trajectories(
            ax,
            pred,
            use_distinct_colors=use_distinct_colors,
        )

    draw_human_trajectories(
        ax,
        humans,
        radius_override=0.30,
        use_distinct_colors=use_distinct_colors,
    )

    set_fixed_limits(ax)
    draw_goal_orientation(ax, gx, gy, GOAL_YAWS.get(scenario, 0.0))

    if not no_title:
        ax.set_title(title, fontsize=13, pad=10)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    handles = legend_handles(show_prediction=show_predictions and not pred.empty)
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8.8,
    )

    # More bottom space for legend and more top space for title.
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.24)

    fig.savefig(out_dir / f"{output_name}.png", dpi=FIG_DPI)
    fig.savefig(out_dir / f"{output_name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root experiment directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario prefix, e.g. dense_3human",
    )
    parser.add_argument(
        "--show-predictions",
        action="store_true",
        help="Overlay latest predicted human trajectories.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing scenarios instead of raising an error.",
    )
    parser.add_argument(
        "--same-human-color",
        action="store_true",
        help="Use the same color for all humans, exactly like the previous dynamic-human figure style.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not draw the plot title. Useful when the LaTeX subfigure caption already gives the title.",
    )

    args = parser.parse_args()
    out_dir = ensure_dir(args.out)

    scenarios = SCENARIOS
    if args.scenario is not None:
        scenarios = [s for s in SCENARIOS if s[0] == args.scenario]
        if not scenarios:
            raise ValueError("Unknown scenario: {}".format(args.scenario))

    for scenario, title, output_name in scenarios:
        try:
            plot_scenario(
                args.root,
                out_dir,
                scenario,
                title,
                output_name,
                show_predictions=args.show_predictions,
                use_distinct_colors=not args.same_human_color,
                no_title=args.no_title,
            )
        except FileNotFoundError as e:
            if args.skip_missing:
                print("Skipping:", e)
                continue
            raise

    print("Saved dense-human figures to:", out_dir.resolve())


if __name__ == "__main__":
    main()
