#!/usr/bin/env python3
"""
Compute navigation metrics from experiment_logger_node.py output.

Usage:
  python3 compute_navigation_metrics.py --run-dir ~/navigation_experiments/mpc_run_YYYYMMDD_HHMMSS
  python3 compute_navigation_metrics.py --results-root ~/navigation_experiments --aggregate

Outputs:
  metrics.json inside each run directory
  aggregate_metrics.json and aggregate_metrics.csv when --aggregate is used
"""

import argparse
import csv
import json
import math
import os
from glob import glob


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
            dx = x - prev[0]
            dy = y - prev[1]
            ds = math.hypot(dx, dy)
            # Ignore extremely large jumps caused by reset/localization glitches.
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
    """Mean absolute change in consecutive yaw increments.

    S = mean(|dtheta_k - dtheta_{k-1}|).
    Lower values mean smoother heading evolution.
    """
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
    """Optional auxiliary smoothness-related metric."""
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


def compute_run_metrics(run_dir, success_time_limit=None, collision_margin=0.0):
    trajectory = read_csv(os.path.join(run_dir, "trajectory.csv"))
    distances = read_csv(os.path.join(run_dir, "distances.csv"))
    planner = read_csv(os.path.join(run_dir, "planner.csv"))
    zmp = read_csv(os.path.join(run_dir, "zmp.csv"))
    dcm = read_csv(os.path.join(run_dir, "dcm.csv"))
    events = read_csv(os.path.join(run_dir, "events.csv"))
    live = load_live_summary(run_dir)

    final_goal_error = compute_final_goal_error(trajectory)
    path_length = compute_path_length(trajectory)
    navigation_time = compute_navigation_time(trajectory, live)
    smoothness = compute_smoothness(trajectory)
    mean_abs_yaw_rate = compute_mean_abs_yaw_rate(trajectory)

    human_clearances = [to_float(r.get("min_human_clearance")) for r in distances]
    obstacle_clearances = [to_float(r.get("min_obstacle_clearance")) for r in distances]
    min_human_distance = minimum(human_clearances)
    min_obstacle_distance = minimum(obstacle_clearances)

    zmp_margins = [to_float(r.get("stability_margin")) for r in zmp]
    min_zmp_margin = minimum(zmp_margins)
    mean_zmp_margin = mean(zmp_margins)

    dcm_margins = [to_float(r.get("dcm_margin")) for r in dcm]
    dcm_com_distances = [to_float(r.get("dcm_com_distance")) for r in dcm]
    min_dcm_margin = minimum(dcm_margins)
    mean_dcm_margin = mean(dcm_margins)
    max_dcm_com_distance = maximum(dcm_com_distances)
    mean_dcm_com_distance = mean(dcm_com_distances)

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

        # Binary outcomes
        "success": success,
        "goal_reached": bool(goal_reached),
        "timeout": bool(timeout),
        "collision": bool(collision),
        "collision_human": bool(collision_human),
        "collision_obstacle": bool(collision_obstacle),

        # Navigation quality
        "final_goal_error": final_goal_error,
        "path_length": path_length,
        "navigation_time": navigation_time,
        "smoothness": smoothness,
        "mean_abs_yaw_rate": mean_abs_yaw_rate,

        # Safety
        "minimum_human_distance": min_human_distance,
        "minimum_obstacle_distance": min_obstacle_distance,

        # Footstep planning
        "number_of_footsteps": number_executed,
        "number_of_footsteps_executed": number_executed,
        "number_of_footsteps_generated": number_generated,

        # Stability
        "zmp_stability_margin_min": min_zmp_margin,
        "zmp_stability_margin_mean": mean_zmp_margin,
        "dcm_stability_margin_min": min_dcm_margin,
        "dcm_stability_margin_mean": mean_dcm_margin,
        "dcm_com_distance_max": max_dcm_com_distance,
        "dcm_com_distance_mean": mean_dcm_com_distance,

        # Runtime
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

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics


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
        "final_goal_error",
        "path_length",
        "navigation_time",
        "smoothness",
        "mean_abs_yaw_rate",
        "minimum_human_distance",
        "minimum_obstacle_distance",
        "number_of_footsteps",
        "number_of_footsteps_executed",
        "number_of_footsteps_generated",
        "zmp_stability_margin_min",
        "zmp_stability_margin_mean",
        "dcm_stability_margin_min",
        "dcm_stability_margin_mean",
        "dcm_com_distance_max",
        "dcm_com_distance_mean",
        "mpc_solve_time_mean",
        "mpc_solve_time_max",
        "replanning_cycle_time_mean",
        "replanning_cycle_time_max",
        "num_replanning_cycles",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="One run directory produced by experiment_logger_node.py")
    parser.add_argument("--results-root", default="~/navigation_experiments", help="Root directory containing multiple run folders")
    parser.add_argument("--aggregate", action="store_true", help="Compute aggregate metrics over all run folders in results-root")
    parser.add_argument("--success-time-limit", type=float, default=None, help="Maximum allowed navigation time for success")
    parser.add_argument("--collision-margin", type=float, default=0.0, help="Clearance threshold for collision")
    args = parser.parse_args()

    if args.aggregate:
        root = os.path.expanduser(args.results_root)
        run_dirs = find_run_dirs(root)
        metrics = [compute_run_metrics(d, args.success_time_limit, args.collision_margin) for d in run_dirs]
        aggregate = aggregate_metrics(metrics, root)
        print(json.dumps(aggregate, indent=4))
        return

    if not args.run_dir:
        raise SystemExit("Provide --run-dir or use --aggregate with --results-root")

    metrics = compute_run_metrics(os.path.expanduser(args.run_dir), args.success_time_limit, args.collision_margin)
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
