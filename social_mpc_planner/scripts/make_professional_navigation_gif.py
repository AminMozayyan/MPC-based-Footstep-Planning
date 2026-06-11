#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a professional GIF/MP4 animation from one run directory saved by
navigation_experiment_logger_node.py.

Expected files in --run-dir:
  trajectory.csv
  humans.csv
  predicted_humans.csv
  static_obstacles.csv

Optional:
  gait/replan_*_gait_traj.csv   # if available, footsteps/ZMP/CoM/DCM may be extended later

Example:
  python3 make_professional_navigation_gif.py \
      --run-dir ~/Surena_experiments/prediction_01_20260606_194030 \
      --out scenario_professional.gif \
      --fps 12 \
      --max-frames 220

Dependencies:
  pip3 install pandas numpy matplotlib pillow imageio
"""

import argparse
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter


# ----------------------------- Helpers -----------------------------

def load_csv(path, required=True):
    if os.path.exists(path):
        return pd.read_csv(path)
    if required:
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.DataFrame()


def nearest_time_rows(df, t, time_col="t", max_dt=None):
    """Return all rows at the logged timestamp closest to t."""
    if df.empty or time_col not in df.columns:
        return df.iloc[0:0]
    times = df[time_col].to_numpy()
    nearest_t = times[np.argmin(np.abs(times - t))]
    if max_dt is not None and abs(nearest_t - t) > max_dt:
        return df.iloc[0:0]
    return df[np.isclose(df[time_col], nearest_t)]


def make_equal_limits(traj, humans, pred, obs, padding=0.8):
    xs, ys = [], []

    for df in [traj, humans, pred, obs]:
        if df is not None and not df.empty:
            if "x" in df.columns:
                xs.extend(df["x"].dropna().tolist())
            if "y" in df.columns:
                ys.extend(df["y"].dropna().tolist())
            if "goal_x" in df.columns:
                xs.extend(df["goal_x"].dropna().tolist())
            if "goal_y" in df.columns:
                ys.extend(df["goal_y"].dropna().tolist())

    if len(xs) == 0:
        xs, ys = [0, 1], [0, 1]

    xmin, xmax = min(xs) - padding, max(xs) + padding
    ymin, ymax = min(ys) - padding, max(ys) + padding

    # keep square visual scale
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    return cx - half, cx + half, cy - half, cy + half


def draw_robot(ax, x, y, theta, radius, color):
    robot = Circle((x, y), radius, facecolor=color, edgecolor="#08306b", lw=2.0, alpha=0.85, zorder=8)
    ax.add_patch(robot)

    # heading arrow
    L = radius * 1.5
    ax.arrow(
        x, y,
        L * math.cos(theta), L * math.sin(theta),
        width=0.018,
        head_width=0.11,
        head_length=0.13,
        length_includes_head=True,
        color="#08306b",
        zorder=9,
    )


def draw_goal(ax, gx, gy, gtheta=None):
    ax.scatter([gx], [gy], marker="*", s=260, c="#f1c40f", edgecolors="#7f6000", linewidths=1.4, zorder=10)
    if gtheta is not None and not np.isnan(gtheta):
        L = 0.45
        ax.arrow(
            gx, gy,
            L * math.cos(gtheta), L * math.sin(gtheta),
            width=0.012,
            head_width=0.09,
            head_length=0.12,
            length_includes_head=True,
            color="#7f6000",
            zorder=11,
        )


# ----------------------------- Main animation -----------------------------

def create_animation(
    run_dir,
    out_path,
    fps=12,
    max_frames=220,
    robot_radius=0.40,
    default_human_radius=0.30,
    show_predictions=True,
    show_human_trails=True,
    prediction_history_window=0.25,
    human_trail_seconds=8.0,
):
    run_dir = Path(run_dir).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    traj = load_csv(run_dir / "trajectory.csv")
    humans = load_csv(run_dir / "humans.csv", required=False)
    pred = load_csv(run_dir / "predicted_humans.csv", required=False)
    obs = load_csv(run_dir / "static_obstacles.csv", required=False)

    if "radius" in humans.columns:
        humans["radius"] = humans["radius"].fillna(default_human_radius)
    if "radius" in pred.columns:
        pred["radius"] = pred["radius"].fillna(default_human_radius)

    # Static obstacles should be drawn once, using unique IDs.
    if not obs.empty:
        obs = obs.drop_duplicates(subset=["obstacle_id"], keep="last")

    # Frame times. We sample trajectory to keep the GIF size reasonable.
    traj = traj.sort_values("t").reset_index(drop=True)
    n_frames = min(max_frames, len(traj))
    frame_indices = np.linspace(0, len(traj) - 1, n_frames).astype(int)
    frame_times = traj.loc[frame_indices, "t"].to_numpy()

    xmin, xmax, ymin, ymax = make_equal_limits(traj, humans, pred, obs)

    # Professional but readable palette.
    colors = {
        "robot": "#2E86DE",          # blue
        "robot_path": "#1B4F72",     # dark blue
        "human": "#E74C3C",          # red
        "human_path": "#922B21",     # dark red
        "prediction": "#8E44AD",     # purple
        "obstacle": "#6C757D",       # gray
        "obstacle_fill": "#BDC3C7",  # light gray
        "safe_zone": "#F1948A",      # soft red
        "grid": "#D5DBDB",
    }

    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=140)
    fig.patch.set_facecolor("white")

    def update(frame_no):
        ax.clear()
        t = frame_times[frame_no]
        row = traj.iloc[frame_indices[frame_no]]

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, color=colors["grid"], linewidth=0.7, alpha=0.55)
        ax.set_xlabel("x [m]", fontsize=11)
        ax.set_ylabel("y [m]", fontsize=11)
        ax.set_title("Humanoid Social Navigation Scenario", fontsize=14, fontweight="bold", pad=12)

        # Static obstacles.
        if not obs.empty:
            for _, o in obs.iterrows():
                ox, oy, rr = float(o.x), float(o.y), float(o.radius)
                ax.add_patch(Circle((ox, oy), rr, facecolor=colors["obstacle_fill"], edgecolor=colors["obstacle"],
                                    lw=2.0, alpha=0.85, zorder=2))
                # optional inflated visual zone around obstacles
                ax.add_patch(Circle((ox, oy), rr + robot_radius, facecolor="none", edgecolor=colors["obstacle"],
                                    lw=0.9, ls="--", alpha=0.30, zorder=1))

        # Robot path until current time.
        past = traj[traj["t"] <= t]
        ax.plot(past["x"], past["y"], color=colors["robot_path"], lw=2.6, alpha=0.95, zorder=5)

        # Human trails.
        if show_human_trails and not humans.empty:
            htrail = humans[(humans["t"] >= t - human_trail_seconds) & (humans["t"] <= t)]
            for pid, g in htrail.groupby("person_id"):
                g = g.sort_values("t")
                ax.plot(g["x"], g["y"], color=colors["human_path"], lw=1.8, alpha=0.45, zorder=4)

        # Predictions closest to current time.
        if show_predictions and not pred.empty:
            pnow = nearest_time_rows(pred, t, max_dt=prediction_history_window)
            for pid, g in pnow.groupby("person_id"):
                g = g.sort_values("pred_step")
                ax.plot(g["x"], g["y"], color=colors["prediction"], lw=1.7, ls="--", alpha=0.78, zorder=6)
                ax.scatter(g["x"], g["y"], color=colors["prediction"], s=12, alpha=0.55, zorder=6)

        # Current humans.
        hnow = nearest_time_rows(humans, t, max_dt=prediction_history_window)
        if not hnow.empty:
            for _, h in hnow.iterrows():
                hx, hy = float(h.x), float(h.y)
                hr = float(h.radius) if "radius" in h and not pd.isna(h.radius) else default_human_radius
                pid = int(h.person_id)
                ax.add_patch(Circle((hx, hy), hr, facecolor=colors["human"], edgecolor="#641E16",
                                    lw=1.8, alpha=0.88, zorder=8))
                # combined robot-human clearance visualization
                ax.add_patch(Circle((hx, hy), hr + robot_radius, facecolor="none", edgecolor=colors["safe_zone"],
                                    lw=1.0, ls=":", alpha=0.55, zorder=3))
                ax.text(hx, hy + hr + 0.12, f"H{pid}", fontsize=9, ha="center", va="bottom",
                        color="#641E16", fontweight="bold", zorder=12)

        # Goal and robot.
        gx = float(row.goal_x) if "goal_x" in row else float(traj.goal_x.iloc[-1])
        gy = float(row.goal_y) if "goal_y" in row else float(traj.goal_y.iloc[-1])
        gtheta = float(row.goal_theta) if "goal_theta" in row and not pd.isna(row.goal_theta) else None
        draw_goal(ax, gx, gy, gtheta)
        draw_robot(ax, float(row.x), float(row.y), float(row.theta), robot_radius, colors["robot"])

        # Time and distance box.
        dist = float(row.distance_to_goal) if "distance_to_goal" in row else math.hypot(gx - row.x, gy - row.y)
        info = f"time = {t:5.1f} s\nrobot-goal distance = {dist:4.2f} m"
        ax.text(
            0.02, 0.98, info,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#7F8C8D", alpha=0.88),
            zorder=20,
        )

        # Legend.
        legend_items = [
            Line2D([0], [0], color=colors["robot_path"], lw=2.6, label="Robot path"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["robot"],
                   markeredgecolor="#08306b", markersize=10, label="Robot"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["human"],
                   markeredgecolor="#641E16", markersize=10, label="Moving humans"),
            Line2D([0], [0], color=colors["prediction"], lw=1.7, ls="--", label="Predicted human trajectory"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["obstacle_fill"],
                   markeredgecolor=colors["obstacle"], markersize=10, label="Static obstacles"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#f1c40f",
                   markeredgecolor="#7f6000", markersize=14, label="Goal"),
        ]
        ax.legend(handles=legend_items, loc="lower right", fontsize=8.5, framealpha=0.90)

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        anim.save(str(out_path), writer=PillowWriter(fps=fps))
    else:
        # Requires ffmpeg for mp4.
        anim.save(str(out_path), fps=fps)

    plt.close(fig)
    print(f"Saved animation: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to one experiment run directory")
    parser.add_argument("--out", default="scenario_professional.gif", help="Output .gif or .mp4 path")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=220)
    parser.add_argument("--robot-radius", type=float, default=0.40)
    parser.add_argument("--human-radius", type=float, default=0.30)
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--no-human-trails", action="store_true")
    args = parser.parse_args()

    create_animation(
        run_dir=args.run_dir,
        out_path=args.out,
        fps=args.fps,
        max_frames=args.max_frames,
        robot_radius=args.robot_radius,
        default_human_radius=args.human_radius,
        show_predictions=not args.no_predictions,
        show_human_trails=not args.no_human_trails,
    )
