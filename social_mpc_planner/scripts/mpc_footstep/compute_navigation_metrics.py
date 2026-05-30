#!/usr/bin/env python3
"""
Compute navigation + gait stability metrics for one humanoid navigation run.

Expected run directory:
  run_dir/
    trajectory.csv
    distances.csv
    planner.csv
    events.csv
    summary_live.json
    gait/
      replan_001_gait_traj.csv
      replan_002_gait_traj.csv
      ...

Usage:
  python3 compute_navigation_metrics.py --run-dir ~/Surena_experiments/no_obstacle_straight_01_YYYYMMDD_HHMMSS
  python3 compute_navigation_metrics.py --results-root ~/Surena_experiments --aggregate

Outputs:
  metrics.json inside each run directory
  gait_stability_summary.csv inside each run directory, if gait CSV files exist
"""

import argparse
import csv
import json
import math
import os
from glob import glob
from typing import Dict, List, Optional, Tuple


# ----------------------------- basic utilities -----------------------------

def to_float(value):
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values):
    return [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]


def mean(values):
    values = finite_values(values)
    return sum(values) / len(values) if values else None


def std(values):
    values = finite_values(values)
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def maximum(values):
    values = finite_values(values)
    return max(values) if values else None


def minimum(values):
    values = finite_values(values)
    return min(values) if values else None


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_live_summary(run_dir):
    path = os.path.join(run_dir, "summary_live.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ----------------------------- navigation metrics -----------------------------

def compute_path_length(trajectory_rows):
    if len(trajectory_rows) < 2:
        return 0.0
    length = 0.0
    prev = None
    for row in trajectory_rows:
        x = to_float(row.get("x"))
        y = to_float(row.get("y"))
        if math.isnan(x) or math.isnan(y):
            continue
        if prev is not None:
            ds = math.hypot(x - prev[0], y - prev[1])
            # Ignore very large jumps caused by reset/localization glitches.
            if ds < 2.0:
                length += ds
        prev = (x, y)
    return length


def compute_final_goal_error(trajectory_rows):
    if not trajectory_rows:
        return None
    last = trajectory_rows[-1]
    x = to_float(last.get("x"))
    y = to_float(last.get("y"))
    gx = to_float(last.get("goal_x"))
    gy = to_float(last.get("goal_y"))
    if any(math.isnan(v) for v in [x, y, gx, gy]):
        return None
    return math.hypot(gx - x, gy - y)

def compute_final_heading_error(trajectory_rows):

    if not trajectory_rows:
        return None

    last = trajectory_rows[-1]

    theta = to_float(last.get("theta"))
    goal_theta = to_float(last.get("goal_theta"))

    if math.isnan(theta) or math.isnan(goal_theta):
        return None

    error = abs(
        wrap_angle(theta - goal_theta)
    )

    return math.degrees(error)


def compute_navigation_time(trajectory_rows, live_summary):
    if trajectory_rows:
        t0 = to_float(trajectory_rows[0].get("t"))
        t1 = to_float(trajectory_rows[-1].get("t"))
        if not math.isnan(t0) and not math.isnan(t1):
            return max(0.0, t1 - t0)
    return live_summary.get("navigation_time")


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_smoothness(trajectory_rows):
    thetas = []
    for row in trajectory_rows:
        theta = to_float(row.get("theta"))
        if not math.isnan(theta):
            thetas.append(theta)
    if len(thetas) < 3:
        return None
    increments = [wrap_angle(thetas[i] - thetas[i - 1]) for i in range(1, len(thetas))]
    changes = [abs(wrap_angle(increments[i] - increments[i - 1])) for i in range(1, len(increments))]
    return mean(changes)


def compute_mean_abs_yaw_rate(trajectory_rows):
    values = []
    prev_t = None
    prev_theta = None
    for row in trajectory_rows:
        t = to_float(row.get("t"))
        theta = to_float(row.get("theta"))
        if math.isnan(t) or math.isnan(theta):
            continue
        if prev_t is not None and t > prev_t:
            values.append(abs(wrap_angle(theta - prev_theta)) / (t - prev_t))
        prev_t, prev_theta = t, theta
    return mean(values)


def compute_planner_metric(planner_rows, event_name):
    return [to_float(r.get("value")) for r in planner_rows if r.get("event") == event_name]


def event_exists(events, name):
    return any(r.get("event") == name for r in events)


# ----------------------------- gait stability metrics -----------------------------

def get_gait_csv_files(run_dir: str) -> List[str]:
    gait_dir = os.path.join(run_dir, "gait")
    files = []
    if os.path.isdir(gait_dir):
        files.extend(glob(os.path.join(gait_dir, "*.csv")))
    # Backward-compatible fallback for old single-file logging.
    old_file = os.path.join(run_dir, "gait_traj_debug.csv")
    if os.path.exists(old_file):
        files.append(old_file)
    return sorted(set(files))


def get_polygon(row: Dict[str, str]) -> List[Tuple[float, float]]:
    pts = []
    for k in range(8):
        x = to_float(row.get(f"poly_x{k}"))
        y = to_float(row.get(f"poly_y{k}"))
        if not math.isnan(x) and not math.isnan(y):
            pts.append((x, y))
    return pts


def point_segment_distance(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    u = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    cx, cy = ax + u * vx, ay + u * vy
    return math.hypot(px - cx, py - cy)


def point_in_polygon(px, py, poly):
    if len(poly) < 3:
        return False
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)):
            x_intersect = (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
            if px < x_intersect:
                inside = not inside
        j = i
    return inside


def signed_polygon_margin(px, py, poly):
    if len(poly) < 3 or math.isnan(px) or math.isnan(py):
        return float("nan"), None
    d = min(
        point_segment_distance(px, py, poly[i][0], poly[i][1], poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1])
        for i in range(len(poly))
    )
    inside = point_in_polygon(px, py, poly)
    return (d if inside else -d), inside


def rounded_pose_key(x, y, yaw, decimals=4):
    return (round(x, decimals), round(y, decimals), round(yaw, decimals))


def compute_footstep_feasibility_from_rows(
    rows,
    max_step_length=0.35,
    min_step_width=0.08,
    max_step_width=0.35,
    max_yaw_change=0.35,
):
    """Approximate geometric feasibility ratio based on logged foot poses."""
    contacts = []

    for foot in ["left", "right"]:
        prefix = "l" if foot == "left" else "r"
        last_key = None
        for row in rows:
            x = to_float(row.get(f"{prefix}x"))
            y = to_float(row.get(f"{prefix}y"))
            z = to_float(row.get(f"{prefix}z"))
            yaw = to_float(row.get(f"{prefix}_yaw"))
            sample = to_float(row.get("i"))
            if any(math.isnan(v) for v in [x, y, z, yaw]) or abs(z) > 1e-3:
                continue
            key = rounded_pose_key(x, y, yaw)
            if key != last_key:
                contacts.append({"foot": foot, "x": x, "y": y, "yaw": yaw, "sample": sample, "row": row})
                last_key = key

    contacts.sort(key=lambda c: c["sample"] if not math.isnan(c["sample"]) else 0.0)
    if not contacts:
        return None

    feasible_values = []
    last_by_foot = {}

    for c in contacts:
        feasible = True
        foot = c["foot"]

        if foot in last_by_foot:
            prev = last_by_foot[foot]
            step_length = math.hypot(c["x"] - prev["x"], c["y"] - prev["y"])
            yaw_change = abs(wrap_angle(c["yaw"] - prev["yaw"]))
            if step_length > max_step_length or yaw_change > max_yaw_change:
                feasible = False

        row = c["row"]
        ly = to_float(row.get("ly"))
        ry = to_float(row.get("ry"))
        if not math.isnan(ly) and not math.isnan(ry):
            width = abs(ly - ry)
            if width < min_step_width or width > max_step_width:
                feasible = False

        last_by_foot[foot] = c
        feasible_values.append(1.0 if feasible else 0.0)

    return mean(feasible_values)


def compute_gait_stability_metrics(run_dir: str, dt: float = 0.005) -> Dict[str, Optional[float]]:
    files = get_gait_csv_files(run_dir)

    zmp_margins = []
    dcm_margins = []
    dcm_com_distances = []
    zmp_inside_flags = []
    dcm_inside_flags = []
    feasibility_ratios = []
    all_summary_rows = []

    for idx, csv_path in enumerate(files, start=1):
        rows = read_csv(csv_path)
        if not rows:
            continue

        file_zmp_margins = []
        file_dcm_margins = []
        file_dcm_com = []
        file_zmp_inside = []
        file_dcm_inside = []

        for row in rows:
            poly = get_polygon(row)
            zmp_x = to_float(row.get("zmp_x"))
            zmp_y = to_float(row.get("zmp_y"))
            dcm_x = to_float(row.get("dcm_x"))
            dcm_y = to_float(row.get("dcm_y"))
            com_x = to_float(row.get("com_x"))
            com_y = to_float(row.get("com_y"))

            zmp_margin, zmp_inside = signed_polygon_margin(zmp_x, zmp_y, poly)
            dcm_margin, dcm_inside = signed_polygon_margin(dcm_x, dcm_y, poly)
            if not math.isnan(zmp_margin):
                zmp_margins.append(zmp_margin)
                file_zmp_margins.append(zmp_margin)
                zmp_inside_flags.append(1.0 if zmp_inside else 0.0)
                file_zmp_inside.append(1.0 if zmp_inside else 0.0)
            if not math.isnan(dcm_margin):
                dcm_margins.append(dcm_margin)
                file_dcm_margins.append(dcm_margin)
                dcm_inside_flags.append(1.0 if dcm_inside else 0.0)
                file_dcm_inside.append(1.0 if dcm_inside else 0.0)
            if not any(math.isnan(v) for v in [dcm_x, dcm_y, com_x, com_y]):
                val = math.hypot(dcm_x - com_x, dcm_y - com_y)
                dcm_com_distances.append(val)
                file_dcm_com.append(val)

        feas = compute_footstep_feasibility_from_rows(rows)
        if feas is not None:
            feasibility_ratios.append(feas)

        all_summary_rows.append({
            "file": os.path.basename(csv_path),
            "min_zmp_margin": minimum(file_zmp_margins),
            "mean_zmp_margin": mean(file_zmp_margins),
            "zmp_outside_ratio": None if not file_zmp_inside else 1.0 - mean(file_zmp_inside),
            "min_dcm_margin": minimum(file_dcm_margins),
            "mean_dcm_margin": mean(file_dcm_margins),
            "dcm_outside_ratio": None if not file_dcm_inside else 1.0 - mean(file_dcm_inside),
            "max_dcm_com_distance": maximum(file_dcm_com),
            "mean_dcm_com_distance": mean(file_dcm_com),
            "footstep_feasibility_ratio": feas,
        })

    # Save per-replan gait summary for debugging/thesis traceability.
    if all_summary_rows:
        out_path = os.path.join(run_dir, "gait_stability_summary.csv")
        fieldnames = list(all_summary_rows[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_summary_rows)

    return {
        "num_gait_replans_logged": len(files),
        "zmp_stability_margin_min": minimum(zmp_margins),
        "zmp_stability_margin_mean": mean(zmp_margins),
        "zmp_outside_ratio": None if not zmp_inside_flags else 1.0 - mean(zmp_inside_flags),
        "dcm_stability_margin_min": minimum(dcm_margins),
        "dcm_stability_margin_mean": mean(dcm_margins),
        "dcm_outside_ratio": None if not dcm_inside_flags else 1.0 - mean(dcm_inside_flags),
        "dcm_com_distance_max": maximum(dcm_com_distances),
        "dcm_com_distance_mean": mean(dcm_com_distances),
        "footstep_feasibility_ratio": mean(feasibility_ratios),
    }


# ----------------------------- run metrics -----------------------------

def compute_run_metrics(run_dir, success_time_limit=None, collision_margin=0.0, gait_dt=0.005):
    trajectory = read_csv(os.path.join(run_dir, "trajectory.csv"))
    distances = read_csv(os.path.join(run_dir, "distances.csv"))
    planner = read_csv(os.path.join(run_dir, "planner.csv"))
    zmp = read_csv(os.path.join(run_dir, "zmp.csv"))
    dcm = read_csv(os.path.join(run_dir, "dcm.csv"))
    events = read_csv(os.path.join(run_dir, "events.csv"))
    live = load_live_summary(run_dir)

    final_goal_error = compute_final_goal_error(trajectory)
    final_heading_error = compute_final_heading_error(trajectory)
    path_length = compute_path_length(trajectory)
    navigation_time = compute_navigation_time(trajectory, live)
    smoothness = compute_smoothness(trajectory)
    mean_abs_yaw_rate = compute_mean_abs_yaw_rate(trajectory)

    human_clearances = [to_float(r.get("min_human_clearance")) for r in distances]
    obstacle_clearances = [to_float(r.get("min_obstacle_clearance")) for r in distances]
    min_human_distance = minimum(human_clearances)
    min_obstacle_distance = minimum(obstacle_clearances)

    # Online logger stability topics, if available.
    zmp_margins = [to_float(r.get("stability_margin")) for r in zmp]
    dcm_margins = [to_float(r.get("dcm_margin")) for r in dcm]
    dcm_com_distances = [to_float(r.get("dcm_com_distance")) for r in dcm]

    online_stability = {
        "zmp_stability_margin_min": minimum(zmp_margins),
        "zmp_stability_margin_mean": mean(zmp_margins),
        "dcm_stability_margin_min": minimum(dcm_margins),
        "dcm_stability_margin_mean": mean(dcm_margins),
        "dcm_com_distance_max": maximum(dcm_com_distances),
        "dcm_com_distance_mean": mean(dcm_com_distances),
    }

    # Offline gait CSV stability has priority because it evaluates every gait sample
    # of every replanning segment produced by robot.cpp.
    gait_stability = compute_gait_stability_metrics(run_dir, dt=gait_dt)
    for key, value in online_stability.items():
        if gait_stability.get(key) is None:
            gait_stability[key] = value

    mpc_solve_times = compute_planner_metric(planner, "mpc_solve_time")
    replan_cycle_times = compute_planner_metric(planner, "replan_cycle_time")
    executed_batches = compute_planner_metric(planner, "executed_footsteps_count")
    generated_batches = compute_planner_metric(planner, "generated_footsteps_count")

    collision_human = any(int(to_float(r.get("collision_human")) or 0) == 1 for r in distances)
    collision_obstacle = any(int(to_float(r.get("collision_obstacle")) or 0) == 1 for r in distances)

    if min_human_distance is not None and min_human_distance <= collision_margin:
        collision_human = True
    if min_obstacle_distance is not None and min_obstacle_distance <= collision_margin:
        collision_obstacle = True

    collision = collision_human or collision_obstacle
    timeout = bool(live.get("timeout", False) or event_exists(events, "timeout"))

    reached_by_event = any(
        r.get("event") == "finish" and ("goal" in r.get("details", "") or "tolerance" in r.get("details", ""))
        for r in events
    )
    reached_by_error = final_goal_error is not None and final_goal_error <= live.get("goal_tolerance", 0.15)
    goal_reached = bool(live.get("goal_reached", False) or reached_by_event or reached_by_error)

    if success_time_limit is None:
        success_time_limit = live.get("max_navigation_time")

    time_ok = True
    if success_time_limit is not None and navigation_time is not None:
        time_ok = navigation_time <= float(success_time_limit)

    success = bool(goal_reached and not collision and time_ok and not timeout)

    number_executed = int(sum(v for v in executed_batches if not math.isnan(v)))
    number_generated = int(sum(v for v in generated_batches if not math.isnan(v)))

    metrics = {
        "run_dir": os.path.abspath(run_dir),

        "success": success,
        "goal_reached": bool(goal_reached),
        "timeout": bool(timeout),
        "collision": bool(collision),
        "collision_human": bool(collision_human),
        "collision_obstacle": bool(collision_obstacle),

        "final_goal_error": final_goal_error,
        "final_heading_error": final_heading_error,
        "path_length": path_length,
        "navigation_time": navigation_time,
        "smoothness": smoothness,
        "mean_abs_yaw_rate": mean_abs_yaw_rate,

        "minimum_human_distance": min_human_distance,
        "minimum_obstacle_distance": min_obstacle_distance,

        "number_of_footsteps": number_executed,
        "number_of_footsteps_executed": number_executed,
        "number_of_footsteps_generated": number_generated,

        "mpc_solve_time_mean": mean(mpc_solve_times),
        "mpc_solve_time_std": std(mpc_solve_times),
        "mpc_solve_time_max": maximum(mpc_solve_times),
        "mpc_solve_time_min": minimum(mpc_solve_times),
        "replanning_cycle_time_mean": mean(replan_cycle_times),
        "replanning_cycle_time_std": std(replan_cycle_times),
        "replanning_cycle_time_max": maximum(replan_cycle_times),
        "replanning_cycle_time_min": minimum(replan_cycle_times),
        "num_replanning_cycles": len(finite_values(replan_cycle_times)),
    }

    metrics.update(gait_stability)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics


# ----------------------------- aggregate mode -----------------------------

def find_run_dirs(results_root):
    candidates = []
    for path in glob(os.path.join(os.path.expanduser(results_root), "*")):
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "trajectory.csv")):
            candidates.append(path)
    return sorted(candidates)


def aggregate_metrics(metrics_list, output_root):
    n = len(metrics_list)
    if n == 0:
        raise RuntimeError("No valid runs found for aggregation.")

    aggregate = {
        "num_runs": n,
        "success_rate": sum(1 for m in metrics_list if m["success"]) / n,
        "collision_rate": sum(1 for m in metrics_list if m["collision"]) / n,
        "collision_human_rate": sum(1 for m in metrics_list if m["collision_human"]) / n,
        "collision_obstacle_rate": sum(1 for m in metrics_list if m["collision_obstacle"]) / n,
        "timeout_rate": sum(1 for m in metrics_list if m.get("timeout", False)) / n,
    }

    scalar_keys = [
        "final_goal_error", "final_heading_error", "path_length", "navigation_time", "smoothness", "mean_abs_yaw_rate",
        "minimum_human_distance", "minimum_obstacle_distance",
        "number_of_footsteps", "number_of_footsteps_executed", "number_of_footsteps_generated",
        "zmp_stability_margin_min", "zmp_stability_margin_mean", "zmp_outside_ratio",
        "dcm_stability_margin_min", "dcm_stability_margin_mean", "dcm_outside_ratio",
        "dcm_com_distance_max", "dcm_com_distance_mean", "footstep_feasibility_ratio",
        "mpc_solve_time_mean", "mpc_solve_time_max",
        "replanning_cycle_time_mean", "replanning_cycle_time_max", "num_replanning_cycles",
        "num_gait_replans_logged",
    ]
    for key in scalar_keys:
        vals = [m.get(key) for m in metrics_list if m.get(key) is not None]
        vals = [float(v) for v in vals if not math.isnan(float(v)) and not math.isinf(float(v))]
        aggregate[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        aggregate[f"{key}_std"] = std(vals) if vals else None
        aggregate[f"{key}_min"] = min(vals) if vals else None
        aggregate[f"{key}_max"] = max(vals) if vals else None

    out_json = os.path.join(output_root, "aggregate_metrics.json")
    with open(out_json, "w") as f:
        json.dump(aggregate, f, indent=4)

    out_csv = os.path.join(output_root, "aggregate_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in aggregate.items():
            writer.writerow([k, v])

    return aggregate


# ----------------------------- command line -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="One run directory produced by experiment_logger_node.py")
    parser.add_argument("--results-root", default="~/navigation_experiments", help="Root directory containing multiple run folders")
    parser.add_argument("--aggregate", action="store_true", help="Compute aggregate metrics over all run folders in results-root")
    parser.add_argument("--success-time-limit", type=float, default=None, help="Maximum allowed navigation time for success")
    parser.add_argument("--collision-margin", type=float, default=0.0, help="Clearance threshold for collision")
    parser.add_argument("--gait-dt", type=float, default=0.005, help="Sampling time of gait_traj CSV files")
    args = parser.parse_args()

    if args.aggregate:
        root = os.path.expanduser(args.results_root)
        run_dirs = find_run_dirs(root)
        metrics = [compute_run_metrics(d, args.success_time_limit, args.collision_margin, args.gait_dt) for d in run_dirs]
        aggregate = aggregate_metrics(metrics, root)
        print(json.dumps(aggregate, indent=4))
        return

    if not args.run_dir:
        raise SystemExit("Provide --run-dir or use --aggregate with --results-root")

    metrics = compute_run_metrics(
        os.path.expanduser(args.run_dir),
        args.success_time_limit,
        args.collision_margin,
        args.gait_dt,
    )
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
