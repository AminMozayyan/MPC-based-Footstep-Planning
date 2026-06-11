#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot one humanoid navigation run with optional dynamic humans, static obstacles,
learned human predictions, and Social-STGCNN uncertainty ellipses.

By default, when --show-uncertainty is enabled, the dashed prediction path is
clipped to the prediction steps that have logged uncertainty. Use
--show-full-prediction to draw the full logged prediction path, including any
extrapolated tail without uncertainty.

Expected files inside run_dir:
  trajectory.csv
  humans.csv
  static_obstacles.csv                 optional
  predicted_humans.csv                 optional
  predicted_human_uncertainty.csv      optional
  planner.csv                          optional

Example:
  python3 plot_navigation_uncertainty_scenario.py \
    --run-dir ~/Surena_experiments/uncertainty_04 \
    --out ~/Surena_experiments/figures \
    --show-predictions \
    --show-uncertainty \
    --prediction-time latest

  python3 plot_navigation_uncertainty_scenario.py \
    --root ~/Surena_experiments \
    --scenario uncertainty_04 \
    --out ~/Surena_experiments/figures \
    --show-static-obstacles \
    --show-predictions \
    --show-uncertainty
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

FIG_DPI = 300
FIG_SIZE = (8.6, 5.8)
HUMAN_COLORS = ["C3", "C4", "C5", "C6", "C7", "C8", "C9"]
UNCERTAINTY_CMAPS = ["Reds", "Purples", "Greens", "Blues", "Oranges", "Greys", "YlOrBr"]


def ensure_dir(path):
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_sort_key(path):
    s = str(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_run_dir(root, scenario):
    root = Path(root).expanduser()
    direct = root / scenario
    if direct.is_dir():
        return direct

    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(scenario)],
        key=natural_sort_key,
    )
    if not candidates:
        raise FileNotFoundError("No run directory found for scenario '{}' under {}".format(scenario, root))
    return candidates[0]


def read_csv_if_exists(run_dir, filename, columns):
    path = Path(run_dir) / filename
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df


def load_trajectory(run_dir):
    path = Path(run_dir) / "trajectory.csv"
    if not path.exists():
        raise FileNotFoundError("Missing trajectory.csv in {}".format(run_dir))
    df = pd.read_csv(path)
    required = ["t", "x", "y", "theta", "goal_x", "goal_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("trajectory.csv missing columns {} in {}".format(missing, run_dir))
    if "goal_theta" not in df.columns:
        df["goal_theta"] = 0.0
    return df


def load_humans(run_dir):
    df = read_csv_if_exists(run_dir, "humans.csv", ["t", "person_id", "x", "y", "radius"])
    if df.empty:
        return df
    if "radius" not in df.columns or df["radius"].isna().all():
        df["radius"] = 0.25
    return df.sort_values(["person_id", "t"]).reset_index(drop=True)


def load_static_obstacles(run_dir):
    df = read_csv_if_exists(run_dir, "static_obstacles.csv", ["t", "obstacle_id", "x", "y", "radius"])
    if df.empty:
        return df
    # Static obstacles may be logged once or multiple times. Keep the latest geometry per id.
    return df.sort_values("t").groupby("obstacle_id", as_index=False).tail(1).reset_index(drop=True)


def load_planner(run_dir):
    return read_csv_if_exists(run_dir, "planner.csv", ["t", "event", "value"])


def load_predictions(run_dir):
    df = read_csv_if_exists(run_dir, "predicted_humans.csv", ["t", "person_id", "pred_step", "x", "y", "radius"])
    if df.empty:
        return df
    if "radius" not in df.columns or df["radius"].isna().all():
        df["radius"] = 0.25
    return df.sort_values(["t", "person_id", "pred_step"]).reset_index(drop=True)


def load_uncertainty(run_dir):
    df = read_csv_if_exists(run_dir, "predicted_human_uncertainty.csv", ["t", "person_id", "pred_step", "sigma_x", "sigma_y", "rho"])
    if df.empty:
        return df
    return df.sort_values(["t", "person_id", "pred_step"]).reset_index(drop=True)


def select_time(df, requested):
    if df.empty or "t" not in df.columns:
        return None
    if requested == "latest":
        return float(df["t"].max())
    if requested == "first":
        return float(df["t"].min())
    return float(requested)


def select_snapshot_by_time(df, requested_time):
    if df.empty or requested_time is None:
        return df.iloc[0:0].copy()
    times = np.asarray(sorted(df["t"].dropna().unique()), dtype=float)
    if len(times) == 0:
        return df.iloc[0:0].copy()
    chosen = float(times[int(np.argmin(np.abs(times - requested_time)))])
    return df[np.isclose(df["t"], chosen)].copy()



def clip_prediction_to_uncertainty_horizon(pred_snapshot, unc_snapshot):
    """
    Keep only prediction steps for which uncertainty exists for the same person.

    This avoids drawing a long dashed trajectory produced by extrapolation while
    uncertainty ellipses are available only for the learned model horizon.
    """
    if pred_snapshot.empty or unc_snapshot.empty:
        return pred_snapshot

    valid_parts = []
    for pid, gp in pred_snapshot.groupby("person_id"):
        gu = unc_snapshot[unc_snapshot["person_id"] == pid]
        if gu.empty:
            valid_parts.append(gp)
            continue
        valid_steps = set(gu["pred_step"].astype(int).tolist())
        keep = gp[gp["pred_step"].astype(int).isin(valid_steps)].copy()
        if not keep.empty:
            valid_parts.append(keep)
    if not valid_parts:
        return pred_snapshot.iloc[0:0].copy()
    return pd.concat(valid_parts, ignore_index=True)


def select_latest_per_person(df):
    if df.empty:
        return df
    parts = []
    for _, g in df.groupby("person_id"):
        last_t = g["t"].max()
        parts.append(g[np.isclose(g["t"], last_t)])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def break_large_jumps(x, y, threshold=0.75):
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    if len(x) < 2:
        return x, y
    dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    for idx in np.where(dist > threshold)[0]:
        if idx + 1 < len(x):
            x[idx + 1] = np.nan
            y[idx + 1] = np.nan
    return x, y


def load_replanning_points(run_dir, traj_df):
    planner = load_planner(run_dir)
    if planner.empty or "event" not in planner.columns:
        return np.array([]), np.array([])
    replans = planner[planner["event"].astype(str) == "replan_cycle_time"]
    if replans.empty:
        return np.array([]), np.array([])

    tx = traj_df["t"].to_numpy(dtype=float)
    xs = traj_df["x"].to_numpy(dtype=float)
    ys = traj_df["y"].to_numpy(dtype=float)
    points = []
    for t in replans["t"].to_numpy(dtype=float):
        idx = int(np.argmin(np.abs(tx - t)))
        x, y = xs[idx], ys[idx]
        if not points or math.hypot(x - points[-1][0], y - points[-1][1]) > 0.08:
            points.append((x, y))
    if not points:
        return np.array([]), np.array([])
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def draw_goal_orientation(ax, gx, gy, yaw, length=None):
    if length is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        length = 0.07 * max(x1 - x0, y1 - y0)
    dx = length * math.cos(yaw)
    dy = length * math.sin(yaw)
    ax.annotate(
        "",
        xy=(gx + dx, gy + dy),
        xytext=(gx, gy),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, mutation_scale=13, shrinkA=0, shrinkB=0),
        zorder=9,
    )


def draw_static_obstacles(ax, obstacles):
    if obstacles.empty:
        return
    for _, o in obstacles.iterrows():
        r = float(o.get("radius", 0.2))
        circ = Circle((float(o["x"]), float(o["y"])), radius=r, facecolor="0.45", edgecolor="0.10", alpha=0.45, lw=1.0, zorder=2)
        ax.add_patch(circ)


def draw_human_trajectories(ax, humans, use_distinct_colors=True, draw_circles=True, radius_override=None, max_circles=24):
    if humans.empty:
        return
    for idx, (pid, g) in enumerate(humans.groupby("person_id")):
        g = g.sort_values("t")
        hx = g["x"].to_numpy(dtype=float)
        hy = g["y"].to_numpy(dtype=float)
        if len(hx) == 0:
            continue
        color = HUMAN_COLORS[idx % len(HUMAN_COLORS)] if use_distinct_colors else "C3"
        radius = float(g["radius"].dropna().iloc[0]) if "radius" in g.columns and not g["radius"].dropna().empty else 0.25
        if radius_override is not None:
            radius = radius_override

        if draw_circles:
            step = max(1, int(math.ceil(len(hx) / float(max_circles))))
            for i in range(0, len(hx), step):
                alpha = 0.18 + 0.55 * i / max(len(hx) - 1, 1)
                ax.add_patch(Circle((hx[i], hy[i]), radius=radius, facecolor=color, edgecolor="none", alpha=alpha, zorder=4))

        ax.plot(hx, hy, linewidth=1.25, color=color, alpha=0.85, zorder=5)
        ax.scatter([hx[0]], [hy[0]], marker="s", s=70, color=color, zorder=7)
        ax.scatter([hx[-1]], [hy[-1]], marker="D", s=70, color=color, zorder=7)


def draw_predictions(ax, pred_snapshot, use_distinct_colors=True):
    if pred_snapshot.empty:
        return
    for idx, (pid, g) in enumerate(pred_snapshot.groupby("person_id")):
        g = g.sort_values("pred_step")
        color = HUMAN_COLORS[idx % len(HUMAN_COLORS)] if use_distinct_colors else "C4"
        ax.plot(g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float), linestyle="--", linewidth=1.55, color=color, alpha=0.85, zorder=3)
        ax.scatter(g["x"].iloc[0], g["y"].iloc[0], marker="o", s=32, color=color, zorder=4)


def ellipse_params(sx, sy, rho, n_std=2.0, sigma_scale=1.0):
    """
    Convert Social-STGCNN Gaussian parameters to a rotated uncertainty ellipse.

    The ellipse orientation is computed from the principal axis of the covariance
    matrix, not from the human heading. This is the statistically correct
    rotation for parameters (sigma_x, sigma_y, rho) in the global frame.
    """
    sx = max(float(sx) * float(sigma_scale), 0.0)
    sy = max(float(sy) * float(sigma_scale), 0.0)
    rho = float(np.clip(rho, -0.999, 0.999))

    a = sx * sx
    b = rho * sx * sy
    d = sy * sy

    # Closed-form eigenvalues of [[a,b],[b,d]]; stable and gives a clear angle.
    trace = a + d
    delta = math.sqrt(max((a - d) * (a - d) + 4.0 * b * b, 0.0))
    lambda1 = max(0.5 * (trace + delta), 0.0)
    lambda2 = max(0.5 * (trace - delta), 0.0)

    # Principal-axis angle of the largest eigenvalue in the global x-y frame.
    # For b=0, this naturally becomes 0 deg when sx>=sy and 90 deg when sy>sx.
    if abs(b) < 1e-12:
        angle = 0.0 if a >= d else 90.0
    else:
        angle = 0.5 * math.degrees(math.atan2(2.0 * b, a - d))

    width = 2.0 * n_std * math.sqrt(lambda1)
    height = 2.0 * n_std * math.sqrt(lambda2)
    return width, height, angle



def gaussian_pdf_grid(X, Y, mean, cov):
    """Evaluate a 2-D Gaussian PDF on a grid."""
    pos = np.dstack((X, Y))
    diff = pos - np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float) + np.eye(2) * 1e-8
    inv_cov = np.linalg.inv(cov)
    det_cov = max(float(np.linalg.det(cov)), 1e-12)
    expo = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    pdf = np.exp(-0.5 * expo) / (2.0 * np.pi * math.sqrt(det_cov))
    return pdf


def draw_gaussian_density_map(
    ax,
    g,
    color_cmap,
    xlim=None,
    ylim=None,
    grid_size=300,
    threshold=0.025,
    alpha=0.88,
    zorder=7,
    sigma_scale=4.0,
):
    """
    Draw the notebook-style uncertainty cloud by summing Gaussian PDFs over all
    predicted future steps for one person.
    """
    if g.empty:
        return

    xs = g["x"].to_numpy(dtype=float)
    ys = g["y"].to_numpy(dtype=float)
    if len(xs) == 0:
        return

    if xlim is None or ylim is None:
        sx_max = float(np.nanmax(g["sigma_x"].to_numpy(dtype=float))) if "sigma_x" in g else 0.2
        sy_max = float(np.nanmax(g["sigma_y"].to_numpy(dtype=float))) if "sigma_y" in g else 0.2
        margin = max(0.5, 3.5 * max(sx_max, sy_max))
        xlim = (float(np.nanmin(xs)) - margin, float(np.nanmax(xs)) + margin)
        ylim = (float(np.nanmin(ys)) - margin, float(np.nanmax(ys)) + margin)

    xx = np.linspace(xlim[0], xlim[1], int(grid_size))
    yy = np.linspace(ylim[0], ylim[1], int(grid_size))
    X, Y = np.meshgrid(xx, yy)
    Z_total = np.zeros_like(X, dtype=float)

    for _, row in g.iterrows():
        # The raw model sigmas are often small for simple linear scripted humans.
        # sigma_scale only affects visualization of the density cloud; it does not
        # change the logged values or the prediction itself.
        sx = max(float(row["sigma_x"]) * float(sigma_scale), 1e-6)
        sy = max(float(row["sigma_y"]) * float(sigma_scale), 1e-6)
        rho = float(np.clip(row["rho"], -0.999, 0.999))
        cov = np.array(
            [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]],
            dtype=float,
        )
        mean = np.array([float(row["x"]), float(row["y"])], dtype=float)
        Z_total += gaussian_pdf_grid(X, Y, mean, cov)

    max_val = float(np.nanmax(Z_total)) if np.isfinite(Z_total).any() else 0.0
    if max_val <= 1e-12:
        return

    Z_total = Z_total / max_val
    Z_total[Z_total < float(threshold)] = np.nan

    ax.contourf(
        X,
        Y,
        Z_total,
        levels=20,
        cmap=color_cmap,
        alpha=alpha,
        zorder=zorder,
        antialiased=True,
    )

def draw_gaussian_cloud(ax, cx, cy, sx, sy, rho, color, angle_override=None, levels=(3.0, 2.4, 1.8, 1.2, 0.6), base_alpha=0.035, zorder=6):
    """Draw nested filled ellipses to mimic the soft uncertainty clouds used in notebooks."""
    for j, ns in enumerate(levels):
        w, h, angle = ellipse_params(sx, sy, rho, n_std=ns)
        if angle_override is not None:
            angle = float(angle_override)
        alpha = base_alpha * (1.0 + 0.30 * j)
        ell = Ellipse(
            (cx, cy),
            width=w,
            height=h,
            angle=angle,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(ell)


def draw_uncertainty_ellipses(
    ax,
    pred_snapshot,
    unc_snapshot,
    n_std=2.0,
    every=1,
    max_ellipses_per_person=12,
    use_distinct_colors=True,
    face_alpha=0.10,
    edge_alpha=0.85,
    draw_centers=False,
    uncertainty_style="ellipse",
    align_with_motion=False,
    density_sigma_scale=4.0,
    density_threshold=0.025,
    density_alpha=0.88,
    density_grid_size=300,
    ellipse_sigma_scale=1.0,
):
    """
    Draw Gaussian uncertainty ellipses at the predicted future positions.

    Important:
    - Ellipse centers come from predicted_humans.csv: x,y at each pred_step.
    - Ellipse sizes/orientations come from predicted_human_uncertainty.csv:
      sigma_x, sigma_y, rho at the same pred_step.
    - This is intentionally drawn after human trajectories so it is not hidden
      under the semi-transparent human history circles.
    """
    if pred_snapshot.empty or unc_snapshot.empty:
        return

    merged = pd.merge(
        pred_snapshot[["person_id", "pred_step", "x", "y"]],
        unc_snapshot[["person_id", "pred_step", "sigma_x", "sigma_y", "rho"]],
        on=["person_id", "pred_step"],
        how="inner",
    )
    if merged.empty:
        return

    for idx, (pid, g) in enumerate(merged.groupby("person_id")):
        g = g.sort_values("pred_step")

        # Step 0 is the current measured point and its uncertainty is normally zero.
        g = g[g["pred_step"].astype(int) > 0]

        if every > 1:
            g = g[g["pred_step"].astype(int) % every == 0]

        if len(g) > max_ellipses_per_person:
            take = np.linspace(0, len(g) - 1, max_ellipses_per_person).round().astype(int)
            g = g.iloc[np.unique(take)]

        color = HUMAN_COLORS[idx % len(HUMAN_COLORS)] if use_distinct_colors else "C4"

        if uncertainty_style == "density":
            # Notebook-style density: sum one Gaussian PDF per prediction step.
            # This uses the covariance exactly as logged, without forcing alignment
            # with the motion direction.
            cmap = UNCERTAINTY_CMAPS[idx % len(UNCERTAINTY_CMAPS)]
            draw_gaussian_density_map(
                ax,
                g,
                cmap,
                xlim=ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else None,
                ylim=ax.get_ylim() if ax.get_ylim() != (0.0, 1.0) else None,
                grid_size=density_grid_size,
                threshold=density_threshold,
                alpha=density_alpha,
                zorder=7,
                sigma_scale=density_sigma_scale,
            )
            if draw_centers:
                ax.scatter(g["x"], g["y"], marker=".", s=12, color=color, alpha=0.8, zorder=9)
            continue

        # Build a map from prediction step to local motion heading. This is used only
        # when --align-uncertainty-with-motion is requested.
        heading_by_step = {}
        if align_with_motion:
            gg = g.sort_values("pred_step").reset_index(drop=True)
            xs = gg["x"].to_numpy(dtype=float)
            ys = gg["y"].to_numpy(dtype=float)
            steps = gg["pred_step"].astype(int).to_numpy()
            for ii, step_i in enumerate(steps):
                if len(xs) < 2:
                    heading_by_step[int(step_i)] = None
                elif ii < len(xs) - 1:
                    dx = xs[ii + 1] - xs[ii]
                    dy = ys[ii + 1] - ys[ii]
                    heading_by_step[int(step_i)] = math.degrees(math.atan2(dy, dx))
                else:
                    dx = xs[ii] - xs[ii - 1]
                    dy = ys[ii] - ys[ii - 1]
                    heading_by_step[int(step_i)] = math.degrees(math.atan2(dy, dx))

        for _, row in g.iterrows():
            sx = float(row["sigma_x"])
            sy = float(row["sigma_y"])
            if sx <= 0.0 or sy <= 0.0:
                continue

            cx = float(row["x"])
            cy = float(row["y"])
            pred_step = int(row["pred_step"])
            angle_override = heading_by_step.get(pred_step, None) if align_with_motion else None

            if uncertainty_style == "cloud":
                draw_gaussian_cloud(
                    ax, cx, cy, sx, sy, row["rho"], color,
                    angle_override=angle_override,
                    levels=(3.0, 2.5, 2.0, 1.5, 1.0, 0.5),
                    base_alpha=0.026,
                    zorder=6,
                )
            else:
                w, h, angle = ellipse_params(sx, sy, row["rho"], n_std=n_std, sigma_scale=ellipse_sigma_scale)
                if angle_override is not None:
                    angle = float(angle_override)

                # Filled transparent ellipse.
                ell_fill = Ellipse(
                    (cx, cy),
                    width=w,
                    height=h,
                    angle=angle,
                    facecolor=color,
                    edgecolor="none",
                    alpha=face_alpha,
                    zorder=7,
                )
                ax.add_patch(ell_fill)

                # Clear outline so it is visible even on top of human path circles.
                ell_edge = Ellipse(
                    (cx, cy),
                    width=w,
                    height=h,
                    angle=angle,
                    facecolor="none",
                    edgecolor=color,
                    alpha=edge_alpha,
                    lw=1.2,
                    zorder=8,
                )
                ax.add_patch(ell_edge)

            if draw_centers:
                ax.scatter([cx], [cy], marker=".", s=12, color=color, alpha=0.8, zorder=9)


def compute_limits(traj, humans, obstacles, margin=0.65):
    xs = list(traj["x"].dropna().astype(float)) + list(traj["goal_x"].dropna().astype(float))
    ys = list(traj["y"].dropna().astype(float)) + list(traj["goal_y"].dropna().astype(float))
    if not humans.empty:
        xs += list(humans["x"].dropna().astype(float))
        ys += list(humans["y"].dropna().astype(float))
    if not obstacles.empty:
        for _, o in obstacles.iterrows():
            r = float(o.get("radius", 0.2))
            xs += [float(o["x"]) - r, float(o["x"]) + r]
            ys += [float(o["y"]) - r, float(o["y"]) + r]
    if not xs or not ys:
        return (-1, 6), (-3, 3)
    return (min(xs) - margin, max(xs) + margin), (min(ys) - margin, max(ys) + margin)


def build_legend(show_predictions=False, show_uncertainty=False, show_static_obstacles=True):
    handles = [
        Line2D([0], [0], color="C0", lw=2.2, label="Robot trajectory"),
        Line2D([0], [0], marker="*", color="C0", linestyle="None", markersize=11, label="Robot start"),
        Line2D([0], [0], marker="X", color="C1", linestyle="None", markersize=9, label="Goal"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="black", linestyle="None", markersize=7, label="Replanning point"),
        Line2D([0], [0], marker="o", color="C3", linestyle="None", markersize=9, label="Human path"),
        Line2D([0], [0], marker="s", color="C3", linestyle="None", markersize=7, label="Human start"),
        Line2D([0], [0], marker="D", color="C3", linestyle="None", markersize=7, label="Human end"),
    ]
    if show_static_obstacles:
        handles.append(Line2D([0], [0], marker="o", color="0.35", markerfacecolor="0.55", linestyle="None", markersize=8, label="Static obstacle"))
    if show_predictions:
        handles.append(Line2D([0], [0], color="0.25", lw=1.5, linestyle="--", label="Predicted human path"))
    if show_uncertainty:
        handles.append(Line2D([0], [0], marker="o", color="0.45", markerfacecolor="none", linestyle="None", markersize=9, label="Prediction uncertainty"))
    return handles


def plot_run(
    run_dir,
    out_dir,
    output_name=None,
    title=None,
    show_static_obstacles=True,
    show_predictions=False,
    show_uncertainty=False,
    prediction_time="latest",
    uncertainty_std=2.0,
    uncertainty_every=1,
    max_uncertainty_ellipses=12,
    uncertainty_style="ellipse",
    align_uncertainty_with_motion=False,
    density_sigma_scale=4.0,
    density_threshold=0.025,
    density_alpha=0.88,
    density_grid_size=300,
    ellipse_sigma_scale=1.0,
    clip_prediction_to_uncertainty=True,
    show_human_circles=True,
    same_human_color=False,
    fixed_xlim=None,
    fixed_ylim=None,
    no_title=False,
):
    run_dir = Path(run_dir).expanduser()
    out_dir = ensure_dir(out_dir)
    output_name = output_name or run_dir.name
    title = title or run_dir.name.replace("_", " ")

    traj = load_trajectory(run_dir)
    humans = load_humans(run_dir)
    obstacles = load_static_obstacles(run_dir)
    pred = load_predictions(run_dir)
    unc = load_uncertainty(run_dir)

    if show_predictions or show_uncertainty:
        if prediction_time == "latest-per-person":
            pred_snapshot = select_latest_per_person(pred)
            unc_snapshot = select_latest_per_person(unc)
        else:
            base_time = select_time(pred if not pred.empty else unc, prediction_time)
            pred_snapshot = select_snapshot_by_time(pred, base_time)
            unc_snapshot = select_snapshot_by_time(unc, base_time)
    else:
        pred_snapshot = pred.iloc[0:0].copy()
        unc_snapshot = unc.iloc[0:0].copy()

    if show_predictions and show_uncertainty and clip_prediction_to_uncertainty:
        pred_snapshot = clip_prediction_to_uncertainty_horizon(pred_snapshot, unc_snapshot)

    x_raw = traj["x"].to_numpy(dtype=float)
    y_raw = traj["y"].to_numpy(dtype=float)
    x_plot, y_plot = break_large_jumps(x_raw, y_raw)
    sx, sy = float(traj["x"].iloc[0]), float(traj["y"].iloc[0])
    gx, gy = float(traj["goal_x"].iloc[-1]), float(traj["goal_y"].iloc[-1])
    gtheta = float(traj["goal_theta"].iloc[-1]) if "goal_theta" in traj.columns else 0.0
    replan_x, replan_y = load_replanning_points(run_dir, traj)

    fig, ax = plt.subplots(figsize=FIG_SIZE, constrained_layout=False)

    if show_static_obstacles:
        draw_static_obstacles(ax, obstacles)

    # Draw observed human motion first. Prediction and uncertainty are drawn on top
    # so uncertainty ellipses are located visibly on the predicted future path,
    # not hidden under the observed human-history circles.
    draw_human_trajectories(
        ax,
        humans,
        use_distinct_colors=not same_human_color,
        draw_circles=show_human_circles,
    )

    if show_predictions:
        draw_predictions(ax, pred_snapshot, use_distinct_colors=not same_human_color)

    if show_uncertainty:
        draw_uncertainty_ellipses(
            ax,
            pred_snapshot,
            unc_snapshot,
            n_std=uncertainty_std,
            every=uncertainty_every,
            max_ellipses_per_person=max_uncertainty_ellipses,
            use_distinct_colors=not same_human_color,
            uncertainty_style=uncertainty_style,
            align_with_motion=align_uncertainty_with_motion,
            density_sigma_scale=density_sigma_scale,
            density_threshold=density_threshold,
            density_alpha=density_alpha,
            density_grid_size=density_grid_size,
            ellipse_sigma_scale=ellipse_sigma_scale,
        )

    ax.plot(x_plot, y_plot, linewidth=2.2, color="C0", zorder=8)
    ax.scatter([sx], [sy], marker="*", s=125, color="C0", zorder=10)
    ax.scatter([gx], [gy], marker="X", s=110, color="C1", zorder=10)

    if len(replan_x) > 0:
        ax.scatter(replan_x, replan_y, marker="o", s=42, facecolors="none", edgecolors="black", linewidths=1.1, zorder=9)

    if fixed_xlim is None or fixed_ylim is None:
        xlim, ylim = compute_limits(traj, humans, obstacles)
    else:
        xlim, ylim = fixed_xlim, fixed_ylim
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    draw_goal_orientation(ax, gx, gy, gtheta)

    if not no_title:
        ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    handles = build_legend(
        show_predictions=show_predictions and not pred_snapshot.empty,
        show_uncertainty=show_uncertainty and not unc_snapshot.empty,
        show_static_obstacles=show_static_obstacles and not obstacles.empty,
    )
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.25)

    png_path = out_dir / f"{output_name}.png"
    pdf_path = out_dir / f"{output_name}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)
    print("Saved:", png_path)
    print("Saved:", pdf_path)


def parse_range(text):
    if text is None:
        return None
    parts = [float(v.strip()) for v in text.split(",")]
    if len(parts) != 2:
        raise ValueError("Axis range must be 'min,max'")
    return (parts[0], parts[1])


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", help="Direct path to one experiment run directory")
    group.add_argument("--root", help="Root experiment directory; use with --scenario")
    parser.add_argument("--scenario", help="Scenario/run prefix when using --root")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--output-name", default=None, help="Output file basename without extension")
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--no-title", action="store_true")

    parser.add_argument("--show-static-obstacles", action="store_true", default=False)
    parser.add_argument("--hide-human-circles", action="store_true")
    parser.add_argument("--show-predictions", action="store_true")
    parser.add_argument("--show-uncertainty", action="store_true")
    parser.add_argument("--prediction-time", default="latest", help="latest, latest-per-person, first, or numeric time in seconds")
    parser.add_argument("--uncertainty-std", type=float, default=2.0, help="Ellipse confidence scale, e.g. 1, 2, or 3 sigma")
    parser.add_argument("--ellipse-sigma-scale", type=float, default=1.0, help="Visualization-only multiplier for sigma_x and sigma_y when drawing ellipses")
    parser.add_argument("--uncertainty-every", type=int, default=1, help="Draw every Nth prediction-step ellipse")
    parser.add_argument("--max-uncertainty-ellipses", type=int, default=12, help="Maximum ellipses per person")
    parser.add_argument("--uncertainty-style", choices=["ellipse", "cloud", "density"], default="ellipse", help="Draw single ellipses, nested ellipse clouds, or summed Gaussian density maps")
    parser.add_argument("--density-sigma-scale", type=float, default=4.0, help="Visualization-only multiplier for sigma_x and sigma_y in density mode")
    parser.add_argument("--density-threshold", type=float, default=0.025, help="Normalized density cutoff in density mode; lower values show a larger cloud")
    parser.add_argument("--density-alpha", type=float, default=0.88, help="Density cloud opacity")
    parser.add_argument("--density-grid-size", type=int, default=300, help="Grid resolution for density mode")
    parser.add_argument("--align-uncertainty-with-motion", action="store_true", help="Rotate uncertainty major axis to the local predicted motion direction; visualization only")
    parser.add_argument("--show-full-prediction", action="store_true", help="Do not clip the dashed prediction path to the available uncertainty horizon")
    parser.add_argument("--same-human-color", action="store_true")
    parser.add_argument("--xlim", default=None, help="Fixed x-axis range: min,max")
    parser.add_argument("--ylim", default=None, help="Fixed y-axis range: min,max")

    args = parser.parse_args()
    if args.root:
        if not args.scenario:
            raise ValueError("--scenario is required when using --root")
        run_dir = find_run_dir(args.root, args.scenario)
    else:
        run_dir = Path(args.run_dir).expanduser()

    plot_run(
        run_dir=run_dir,
        out_dir=args.out,
        output_name=args.output_name,
        title=args.title,
        show_static_obstacles=args.show_static_obstacles,
        show_predictions=args.show_predictions,
        show_uncertainty=args.show_uncertainty,
        prediction_time=args.prediction_time,
        uncertainty_std=args.uncertainty_std,
        uncertainty_every=max(1, args.uncertainty_every),
        max_uncertainty_ellipses=max(1, args.max_uncertainty_ellipses),
        uncertainty_style=args.uncertainty_style,
        align_uncertainty_with_motion=args.align_uncertainty_with_motion,
        density_sigma_scale=max(0.1, args.density_sigma_scale),
        density_threshold=max(0.0, min(0.95, args.density_threshold)),
        density_alpha=max(0.0, min(1.0, args.density_alpha)),
        density_grid_size=max(80, args.density_grid_size),
        ellipse_sigma_scale=max(0.1, args.ellipse_sigma_scale),
        clip_prediction_to_uncertainty=not args.show_full_prediction,
        show_human_circles=not args.hide_human_circles,
        same_human_color=args.same_human_color,
        fixed_xlim=parse_range(args.xlim),
        fixed_ylim=parse_range(args.ylim),
        no_title=args.no_title,
    )


if __name__ == "__main__":
    main()
