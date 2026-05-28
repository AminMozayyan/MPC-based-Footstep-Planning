#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot gait stability figures from gait_traj_debug.csv
Compatible with Python 3.8

Required CSV columns:

i,
com_x,com_y,com_z,
lx,ly,lz,l_yaw,
rx,ry,rz,r_yaw,
zmp_x,zmp_y,zmp_z,
dcm_x,dcm_y,dcm_z,
phase

Optional columns:

poly_x0,poly_y0,...,poly_x7,poly_y7,
zmp_margin,
zmp_inside
"""

import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================= USER PARAMETERS =========================

DEFAULT_DT = 0.005
FIG_DPI = 300
DOWNSAMPLE_TOP_VIEW = 5

# ================================================================


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(csv_path):

    df = pd.read_csv(csv_path)

    required = [
        "i",

        "com_x", "com_y", "com_z",

        "lx", "ly", "lz", "l_yaw",
        "rx", "ry", "rz", "r_yaw",

        "zmp_x", "zmp_y", "zmp_z",

        "dcm_x", "dcm_y", "dcm_z",

        "phase"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required CSV columns: {}".format(missing)
        )

    return df


def unique_footsteps(df, foot, decimals=4):

    if foot == "l":
        raw_cols = ["lx", "ly", "lz", "l_yaw"]
    else:
        raw_cols = ["rx", "ry", "rz", "r_yaw"]

    tmp = df[raw_cols].copy()

    rounded = tmp.round(decimals)

    mask = rounded.ne(rounded.shift()).any(axis=1)

    steps = tmp[mask].copy()

    steps["sample"] = df.loc[mask, "i"].values
    steps["time"] = df.loc[mask, "time"].values

    return steps


# ================================================================
# TOP VIEW
# ================================================================

def plot_top_view(df, out_dir):

    fig, ax = plt.subplots(figsize=(8, 7))

    valid_zmp = ~(
        (df["zmp_x"].abs() < 1e-9) &
        (df["zmp_y"].abs() < 1e-9)
    )

    left_steps = unique_footsteps(df, "l")
    right_steps = unique_footsteps(df, "r")

    left_steps = left_steps[left_steps["lz"].abs() < 1e-3]
    right_steps = right_steps[right_steps["rz"].abs() < 1e-3]

    sl = slice(None, None, DOWNSAMPLE_TOP_VIEW)

    # CoM
    ax.plot(
        df["com_x"].iloc[sl],
        df["com_y"].iloc[sl],
        linewidth=1.5,
        label="CoM"
    )

    # ZMP
    zmp_x = df.loc[valid_zmp, "zmp_x"].iloc[sl].to_numpy()
    zmp_y = df.loc[valid_zmp, "zmp_y"].iloc[sl].to_numpy()

    dist = np.sqrt(np.diff(zmp_x)**2 + np.diff(zmp_y)**2)

    jump_idx = np.where(dist > 0.25)[0]

    for idx in jump_idx:
        zmp_x[idx + 1] = np.nan
        zmp_y[idx + 1] = np.nan

    ax.plot(
        zmp_x,
        zmp_y,
        linewidth=1.5,
        color="green",
        label="ZMP"
)

    # DCM
    dcm_x = df["dcm_x"].iloc[sl].to_numpy()
    dcm_y = df["dcm_y"].iloc[sl].to_numpy()

    # remove invalid huge jumps
    dist = np.sqrt(np.diff(dcm_x)**2 + np.diff(dcm_y)**2)

    jump_idx = np.where(dist > 0.25)[0]

    for idx in jump_idx:
        dcm_x[idx + 1] = np.nan
        dcm_y[idx + 1] = np.nan

    ax.plot(
        dcm_x,
        dcm_y,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label="DCM"
    )

    # Left footsteps
    ax.scatter(
        left_steps["lx"],
        left_steps["ly"],
        marker="o",
        s=20,
        color="tab:pink",
        label="Left footsteps"
    )

    # Right footsteps
    ax.scatter(
        right_steps["rx"],
        right_steps["ry"],
        marker="o",
        s=20,
        color="tab:purple",
        label="Right footsteps"
    )

    ax.set_title("Top-view gait trajectory")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.axis("equal")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.legend(loc="best")

    fig.tight_layout()

    fig.savefig(
        out_dir / "01_top_view.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "01_top_view.pdf"
    )

    plt.close(fig)


# ================================================================
# X/Y TRAJECTORIES
# ================================================================

def plot_xy_trajectories(df, out_dir):

    # x trajectory
    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(df["time"], df["com_x"],
            linewidth=2.0,
            label="CoM x")

    ax.plot(df["time"], df["zmp_x"],
            linewidth=1.8,
            label="ZMP x")

    ax.plot(df["time"], df["dcm_x"],
            linestyle="--",
            linewidth=1.8,
            label="DCM x")

    ax.set_title("x trajectory")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("x [m]")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir / "02_x_trajectory.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "02_x_trajectory.pdf"
    )

    plt.close(fig)

    # y trajectory
    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(df["time"], df["com_y"],
            linewidth=2.0,
            label="CoM y")

    ax.plot(df["time"], df["zmp_y"],
            linewidth=1.8,
            label="ZMP y")

    ax.plot(df["time"], df["dcm_y"],
            linestyle="--",
            linewidth=1.8,
            label="DCM y")

    ax.set_title("y trajectory")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("y [m]")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir / "03_y_trajectory.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "03_y_trajectory.pdf"
    )

    plt.close(fig)


# ================================================================
# ZMP MARGIN
# ================================================================

def plot_zmp_margin(df, out_dir):

    if "zmp_margin" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(
        df["time"],
        df["zmp_margin"],
        linewidth=2.0,
        label="ZMP margin"
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0
    )

    ax.set_title(
        "ZMP distance to support polygon boundary"
    )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("margin [m]")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir / "04_zmp_margin.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "04_zmp_margin.pdf"
    )

    plt.close(fig)


# ================================================================
# PHASE TIMELINE
# ================================================================

def plot_phase_timeline(df, out_dir):

    fig, ax = plt.subplots(figsize=(9, 3.5))

    ax.step(
        df["time"],
        df["phase"],
        where="post",
        linewidth=1.8
    )

    ax.set_title("Walking phase timeline")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("phase")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()

    fig.savefig(
        out_dir / "05_phase_timeline.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "05_phase_timeline.pdf"
    )

    plt.close(fig)


# ================================================================
# VERTICAL MOTION
# ================================================================

def plot_vertical_motion(df, out_dir):

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(
        df["time"],
        df["com_z"],
        linewidth=2.0,
        label="CoM z"
    )

    ax.plot(
        df["time"],
        df["lz"],
        linewidth=1.5,
        label="Left ankle z"
    )

    ax.plot(
        df["time"],
        df["rz"],
        linewidth=1.5,
        label="Right ankle z"
    )

    ax.set_title("Vertical motion")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [m]")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir / "06_vertical_motion.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "06_vertical_motion.pdf"
    )

    plt.close(fig)


# ================================================================
# DCM-COM DISTANCE
# ================================================================

def plot_dcm_com_distance(df, out_dir):

    err = np.sqrt(
        (df["dcm_x"] - df["com_x"])**2 +
        (df["dcm_y"] - df["com_y"])**2
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(
        df["time"],
        err,
        linewidth=2.0
    )

    ax.set_title("DCM-CoM separation")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()

    fig.savefig(
        out_dir / "07_dcm_com_distance.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "07_dcm_com_distance.pdf"
    )

    plt.close(fig)


# ================================================================
# COM VELOCITY
# ================================================================

def plot_com_velocity(df, out_dir):

    vx = np.gradient(df["com_x"], DEFAULT_DT)
    vy = np.gradient(df["com_y"], DEFAULT_DT)

    speed = np.sqrt(vx**2 + vy**2)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.plot(
        df["time"],
        vx,
        linewidth=1.5,
        label="CoM vx"
    )

    ax.plot(
        df["time"],
        vy,
        linewidth=1.5,
        label="CoM vy"
    )

    ax.plot(
        df["time"],
        speed,
        linewidth=2.0,
        label="CoM speed"
    )

    ax.set_title("CoM velocity")

    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")

    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir / "08_com_velocity.png",
        dpi=FIG_DPI
    )

    fig.savefig(
        out_dir / "08_com_velocity.pdf"
    )

    plt.close(fig)


# ================================================================
# SUMMARY
# ================================================================

def save_summary(df, out_dir):

    rows = []

    def add(metric, value, unit=""):
        rows.append({
            "metric": metric,
            "value": value,
            "unit": unit
        })

    add(
        "duration",
        float(df["time"].iloc[-1] - df["time"].iloc[0]),
        "s"
    )

    add(
        "num_samples",
        int(len(df)),
        ""
    )

    if "zmp_margin" in df.columns:

        add(
            "min_zmp_margin",
            float(df["zmp_margin"].min()),
            "m"
        )

        add(
            "mean_zmp_margin",
            float(df["zmp_margin"].mean()),
            "m"
        )

    if "zmp_inside" in df.columns:

        add(
            "zmp_outside_ratio",
            float((df["zmp_inside"] == 0).mean()),
            "-"
        )

    add(
        "path_length_com_xy",
        float(np.sum(
            np.sqrt(
                np.diff(df["com_x"])**2 +
                np.diff(df["com_y"])**2
            )
        )),
        "m"
    )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        out_dir / "stability_summary.csv",
        index=False
    )


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="gait_traj_debug.csv",
        help="Path to CSV"
    )

    parser.add_argument(
        "--out",
        default="gait_stability_figures",
        help="Output directory"
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Sampling time"
    )

    args = parser.parse_args()

    out_dir = ensure_dir(args.out)

    df = load_csv(args.csv)

    df["time"] = df["i"].astype(float) * args.dt

    plot_top_view(df, out_dir)

    plot_xy_trajectories(df, out_dir)

    plot_zmp_margin(df, out_dir)

    plot_phase_timeline(df, out_dir)

    plot_vertical_motion(df, out_dir)

    plot_dcm_com_distance(df, out_dir)

    plot_com_velocity(df, out_dir)

    save_summary(df, out_dir)

    print(
        "Saved figures to:",
        out_dir.resolve()
    )


if __name__ == "__main__":
    main()