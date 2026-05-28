#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute gait stability metrics from gait_traj_debug.csv and generate
publication-quality plots for thesis/report.

Compatible with Python 3.8.

Input CSV should include:
i,
com_x, com_y, com_z,
lx, ly, lz, l_yaw,
rx, ry, rz, r_yaw,
zmp_x, zmp_y, zmp_z,
dcm_x, dcm_y, dcm_z,
phase,
poly_x0, poly_y0, ..., poly_x7, poly_y7

Outputs:
- gait_stability_metrics.xlsx
- PNG/PDF plots
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_DT = 0.005
FIG_DPI = 300


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(csv_path, dt):
    df = pd.read_csv(csv_path)

    required = [
        "i",
        "com_x", "com_y", "com_z",
        "lx", "ly", "lz", "l_yaw",
        "rx", "ry", "rz", "r_yaw",
        "zmp_x", "zmp_y", "zmp_z",
        "dcm_x", "dcm_y", "dcm_z",
        "phase",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required CSV columns: {}".format(missing))

    df = df.copy()
    df["time"] = df["i"].astype(float) * dt
    return df


def get_polygon(row):
    pts = []
    for k in range(8):
        x_col = "poly_x{}".format(k)
        y_col = "poly_y{}".format(k)

        if x_col not in row.index or y_col not in row.index:
            continue

        x = row[x_col]
        y = row[y_col]

        if np.isfinite(x) and np.isfinite(y):
            pts.append([float(x), float(y)])

    return np.asarray(pts, dtype=float)


def point_segment_distance(p, a, b):
    ab = b - a
    denom = float(np.dot(ab, ab))

    if denom < 1e-12:
        return float(np.linalg.norm(p - a))

    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))

    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def point_in_polygon(p, poly):
    """
    Ray casting point-in-polygon test.
    Works independent of clockwise/counter-clockwise vertex order.
    """
    if len(poly) < 3:
        return False

    x = float(p[0])
    y = float(p[1])
    inside = False

    n = len(poly)
    j = n - 1

    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def signed_margin_to_polygon(point_xy, poly):
    """
    Returns signed minimum distance to polygon boundary:
    positive: inside polygon
    zero: on boundary
    negative: outside polygon.
    """
    if len(poly) < 3:
        return np.nan, 0

    p = np.asarray(point_xy, dtype=float)

    distances = []
    for k in range(len(poly)):
        a = poly[k]
        b = poly[(k + 1) % len(poly)]
        distances.append(point_segment_distance(p, a, b))

    min_dist = float(np.min(distances))
    inside = point_in_polygon(p, poly)

    return (min_dist if inside else -min_dist), int(inside)


def compute_metrics(df):
    zmp_margin = []
    zmp_inside = []
    dcm_margin = []
    dcm_inside = []

    for _, row in df.iterrows():
        poly = get_polygon(row)

        zmp_m, zmp_in = signed_margin_to_polygon(
            np.array([row["zmp_x"], row["zmp_y"]], dtype=float), poly
        )
        dcm_m, dcm_in = signed_margin_to_polygon(
            np.array([row["dcm_x"], row["dcm_y"]], dtype=float), poly
        )

        zmp_margin.append(zmp_m)
        zmp_inside.append(zmp_in)
        dcm_margin.append(dcm_m)
        dcm_inside.append(dcm_in)

    out = df.copy()
    out["zmp_margin"] = zmp_margin
    out["zmp_inside"] = zmp_inside
    out["dcm_margin"] = dcm_margin
    out["dcm_inside"] = dcm_inside

    out["dcm_com_distance"] = np.sqrt(
        (out["dcm_x"] - out["com_x"]) ** 2 +
        (out["dcm_y"] - out["com_y"]) ** 2
    )

    out["zmp_com_distance"] = np.sqrt(
        (out["zmp_x"] - out["com_x"]) ** 2 +
        (out["zmp_y"] - out["com_y"]) ** 2
    )

    return out


def compute_footstep_feasibility(df,
                                 max_step_length=0.35,
                                 min_step_width=0.08,
                                 max_step_width=0.35,
                                 max_yaw_change=0.35):
    """
    Offline approximate footstep feasibility from logged foot poses.
    This is geometric feasibility, not full IK feasibility.
    """
    contacts = []

    left = df[["lx", "ly", "lz", "l_yaw", "time"]].copy()
    left.columns = ["x", "y", "z", "yaw", "time"]
    left["foot"] = "left"

    right = df[["rx", "ry", "rz", "r_yaw", "time"]].copy()
    right.columns = ["x", "y", "z", "yaw", "time"]
    right["foot"] = "right"

    left = left[left["z"].abs() < 1e-3]
    right = right[right["z"].abs() < 1e-3]

    for foot_df in [left, right]:
        rounded = foot_df[["x", "y", "yaw"]].round(4)
        mask = rounded.ne(rounded.shift()).any(axis=1)
        contacts.append(foot_df[mask])

    steps = pd.concat(contacts, ignore_index=True)
    steps = steps.sort_values("time").reset_index(drop=True)

    if len(steps) == 0:
        return np.nan, steps

    feasible_list = []
    step_length_list = []
    yaw_change_list = []

    last_by_foot = {}

    for _, row in steps.iterrows():
        foot = row["foot"]
        feasible = True
        step_length = 0.0
        yaw_change = 0.0

        if foot in last_by_foot:
            prev = last_by_foot[foot]

            step_length = float(np.linalg.norm(
                np.array([row["x"] - prev["x"], row["y"] - prev["y"]])
            ))

            yaw_change = float(abs(math.atan2(
                math.sin(row["yaw"] - prev["yaw"]),
                math.cos(row["yaw"] - prev["yaw"])
            )))

            if step_length > max_step_length:
                feasible = False

            if yaw_change > max_yaw_change:
                feasible = False

        last_by_foot[foot] = row

        nearest_idx = int(np.argmin(np.abs(df["time"].to_numpy() - row["time"])))
        original_row = df.iloc[nearest_idx]
        foot_width = abs(float(original_row["ly"] - original_row["ry"]))

        if foot_width < min_step_width or foot_width > max_step_width:
            feasible = False

        feasible_list.append(int(feasible))
        step_length_list.append(step_length)
        yaw_change_list.append(yaw_change)

    steps["step_length"] = step_length_list
    steps["yaw_change"] = yaw_change_list
    steps["geometric_feasible"] = feasible_list

    feasibility_ratio = float(np.mean(feasible_list)) if len(feasible_list) else np.nan
    return feasibility_ratio, steps


def make_summary(df, feasibility_ratio):
    rows = []

    def add(metric, value, unit, description):
        rows.append({
            "Metric": metric,
            "Value": value,
            "Unit": unit,
            "Description": description
        })

    add("Minimum ZMP Margin", float(np.nanmin(df["zmp_margin"])), "m",
        "Minimum signed distance from ZMP to support polygon boundary")
    add("Mean ZMP Margin", float(np.nanmean(df["zmp_margin"])), "m",
        "Average signed ZMP stability margin")
    add("ZMP Outside Ratio", float(np.mean(df["zmp_inside"] == 0)), "-",
        "Fraction of samples where ZMP is outside the support polygon")

    add("Minimum DCM Margin", float(np.nanmin(df["dcm_margin"])), "m",
        "Minimum signed distance from DCM to support polygon boundary")
    add("Mean DCM Margin", float(np.nanmean(df["dcm_margin"])), "m",
        "Average signed DCM stability margin")
    add("DCM Outside Ratio", float(np.mean(df["dcm_inside"] == 0)), "-",
        "Fraction of samples where DCM is outside the support polygon")

    add("Maximum DCM-CoM Separation", float(np.nanmax(df["dcm_com_distance"])), "m",
        "Maximum horizontal distance between DCM and CoM")
    add("Mean DCM-CoM Separation", float(np.nanmean(df["dcm_com_distance"])), "m",
        "Average horizontal distance between DCM and CoM")

    add("Footstep Feasibility Ratio", feasibility_ratio, "-",
        "Approximate geometric feasibility ratio of generated footsteps")
    add("Duration", float(df["time"].iloc[-1] - df["time"].iloc[0]), "s",
        "Trajectory duration")

    return pd.DataFrame(rows)


def save_excel(df, summary, steps, out_path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="time_series_metrics", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        steps.to_excel(writer, sheet_name="footstep_feasibility", index=False)


def plot_margin(df, out_dir, column, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.plot(df["time"], df[column], linewidth=2.2, label=title)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, label="Support boundary")

    ax.fill_between(df["time"], df[column], 0,
                    where=(df[column] >= 0), alpha=0.18, interpolate=True)
    ax.fill_between(df["time"], df[column], 0,
                    where=(df[column] < 0), alpha=0.18, interpolate=True)

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "{}.png".format(filename), dpi=FIG_DPI)
    fig.savefig(out_dir / "{}.pdf".format(filename))
    plt.close(fig)


def plot_dcm_com_distance(df, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(df["time"], df["dcm_com_distance"], linewidth=2.3, color="tab:purple")

    ax.set_title("DCM-CoM Separation")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance [m]")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "03_dcm_com_separation.png", dpi=FIG_DPI)
    fig.savefig(out_dir / "03_dcm_com_separation.pdf")
    plt.close(fig)


def plot_top_view(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))

    valid_zmp = ~((df["zmp_x"].abs() < 1e-9) & (df["zmp_y"].abs() < 1e-9))

    ax.plot(df["com_x"], df["com_y"], linewidth=2.8, color="black", label="CoM")
    ax.plot(df.loc[valid_zmp, "zmp_x"], df.loc[valid_zmp, "zmp_y"],
            linewidth=2.3, color="tab:blue", label="ZMP")
    ax.plot(df["dcm_x"], df["dcm_y"], linestyle="--",
            linewidth=2.6, color="tab:purple", label="DCM")

    left = df[df["lz"].abs() < 1e-3][["lx", "ly"]].round(4).drop_duplicates()
    right = df[df["rz"].abs() < 1e-3][["rx", "ry"]].round(4).drop_duplicates()

    ax.scatter(left["lx"], left["ly"], marker="o", s=70, color="tab:red",
               edgecolors="black", linewidths=0.8, label="Left footsteps")
    ax.scatter(right["rx"], right["ry"], marker="s", s=70, color="tab:red",
               edgecolors="black", linewidths=0.8, label="Right footsteps")

    ax.set_title("Top-view Gait Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "04_top_view.png", dpi=FIG_DPI)
    fig.savefig(out_dir / "04_top_view.pdf")
    plt.close(fig)


def plot_summary_bar(summary, out_dir):
    selected = summary[
        summary["Metric"].isin([
            "Minimum ZMP Margin",
            "Minimum DCM Margin",
            "Maximum DCM-CoM Separation",
            "Footstep Feasibility Ratio"
        ])
    ].copy()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = selected["Metric"].tolist()
    values = selected["Value"].astype(float).to_numpy()

    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Main Gait Stability Metrics")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_dir / "05_summary_metrics.png", dpi=FIG_DPI)
    fig.savefig(out_dir / "05_summary_metrics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="gait_traj_debug.csv", help="Input CSV file")
    parser.add_argument("--out", default="gait_metric_results", help="Output directory")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Sampling time [s]")
    parser.add_argument("--max-step-length", type=float, default=0.35)
    parser.add_argument("--min-step-width", type=float, default=0.08)
    parser.add_argument("--max-step-width", type=float, default=0.35)
    parser.add_argument("--max-yaw-change", type=float, default=0.35)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)

    df = load_csv(args.csv, args.dt)
    df = compute_metrics(df)

    feasibility_ratio, steps = compute_footstep_feasibility(
        df,
        max_step_length=args.max_step_length,
        min_step_width=args.min_step_width,
        max_step_width=args.max_step_width,
        max_yaw_change=args.max_yaw_change
    )

    summary = make_summary(df, feasibility_ratio)

    print("\n========== Stability Metrics ==========")

    print("Mean ZMP Margin : {:.4f} m".format(
        np.nanmean(df["zmp_margin"])
    ))

    print("Min ZMP Margin  : {:.4f} m".format(
        np.nanmin(df["zmp_margin"])
    ))

    print("Mean DCM Margin : {:.4f} m".format(
        np.nanmean(df["dcm_margin"])
    ))

    print("Min DCM Margin  : {:.4f} m".format(
        np.nanmin(df["dcm_margin"])
    ))

    print("=======================================\n")

    excel_path = out_dir / "gait_stability_metrics.xlsx"
    save_excel(df, summary, steps, excel_path)

    plot_margin(df, out_dir, "zmp_margin",
                "ZMP Stability Margin", "Signed distance [m]", "01_zmp_margin")
    plot_margin(df, out_dir, "dcm_margin",
                "DCM Stability Margin", "Signed distance [m]", "02_dcm_margin")
    plot_dcm_com_distance(df, out_dir)
    plot_top_view(df, out_dir)
    plot_summary_bar(summary, out_dir)

    print("Saved Excel file: {}".format(excel_path.resolve()))
    print("Saved plots to: {}".format(out_dir.resolve()))


if __name__ == "__main__":
    main()
