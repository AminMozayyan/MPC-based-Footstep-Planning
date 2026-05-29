#!/usr/bin/env python3
"""
Aggregate navigation evaluation results across multiple experiment runs.

Expected input structure:

~/Surena_experiments/
    proposed_run_01/
        metrics.json
    proposed_run_02/
        metrics.json
    dwa_run_01/
        metrics.json

Outputs:
    aggregate_metrics.json
    aggregate_metrics.csv
    aggregate_metrics_latex_table.tex
    all_runs_metrics.csv
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


NUMERIC_METRICS = [
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

BOOLEAN_METRICS = [
    "success",
    "goal_reached",
    "collision",
    "collision_human",
    "collision_obstacle",
    "timeout",
]

DISPLAY_NAMES_FA = {
    "success_rate": "نرخ موفقیت (\\%)",
    "collision_rate": "نرخ برخورد (\\%)",
    "timeout_rate": "نرخ اتمام زمان (\\%)",
    "final_goal_error": "خطای نهایی هدف (m)",
    "path_length": "طول مسیر (m)",
    "navigation_time": "زمان ناوبری (s)",
    "smoothness": "نرمی مسیر (rad)",
    "minimum_human_distance": "حداقل فاصله تا انسان (m)",
    "minimum_obstacle_distance": "حداقل فاصله تا مانع (m)",
    "number_of_footsteps": "تعداد گام‌ها",
    "zmp_stability_margin_min": "حاشیه پایداری ZMP (m)",
    "dcm_stability_margin_min": "حاشیه پایداری DCM (m)",
    "dcm_com_distance_max": "بیشینه فاصله DCM-CoM (m)",
    "mpc_solve_time_mean": "زمان حل MPC (s)",
    "replanning_cycle_time_mean": "زمان بازبرنامه‌ریزی (s)",
    "num_replanning_cycles": "تعداد بازبرنامه‌ریزی",
}


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def std(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def infer_method_name(run_name: str) -> str:
    """
    Infer the scenario/method name from a run folder.

    Examples:
      no_obstacle_straight_01_20260529_142913 -> no_obstacle_straight
      no_obstacle_straight_01_20260529        -> no_obstacle_straight
      no_obstacle_straight_01                 -> no_obstacle_straight
      proposed_run_01_20260529_142913         -> proposed
      dwa_run_03                              -> dwa

    The goal is to group repeated runs of the same scenario together so that
    the final table reports mean ± std over all repetitions.
    """
    name = run_name

    # Remove timestamp suffixes created by the logger.
    # Matches _YYYYMMDD_HHMMSS or _YYYYMMDD at the end.
    name = re.sub(r"_[0-9]{8}_[0-9]{6}$", "", name)
    name = re.sub(r"_[0-9]{8}$", "", name)

    # Common naming convention: method_run_01 or scenario_run_01.
    if "_run_" in name:
        return name.split("_run_")[0]

    # Remove final run index: _01, _02, -01, etc.
    name = re.sub(r"[_-][0-9]+$", "", name)

    return name


def load_method_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("method-map must be a JSON object: {run_folder_name: method_name}")
    return {str(k): str(v) for k, v in data.items()}


def discover_runs(root_dir: str, method_map: Dict[str, str]) -> List[Dict[str, Any]]:
    runs = []
    for current_root, _dirs, files in os.walk(root_dir):
        if "metrics.json" not in files:
            continue
        metrics_path = os.path.join(current_root, "metrics.json")
        run_name = os.path.basename(current_root)
        method = method_map.get(run_name, infer_method_name(run_name))
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        runs.append({
            "method": method,
            "run_name": run_name,
            "run_dir": current_root,
            "metrics_path": metrics_path,
            "metrics": metrics,
        })
    runs.sort(key=lambda r: (r["method"], r["run_name"]))
    return runs


def bool_value(metrics: Dict[str, Any], key: str) -> Optional[bool]:
    if key not in metrics:
        return None
    value = metrics[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "y"]
    return None


def numeric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    """
    Read a numeric metric with a few compatibility fallbacks.
    This keeps the aggregator robust to older metrics.json files.
    """
    if key == "number_of_footsteps":
        value = metrics.get("number_of_footsteps")
        if value is None:
            value = metrics.get("number_of_footsteps_executed")
        return safe_float(value)

    if key == "mpc_solve_time_mean":
        value = metrics.get("mpc_solve_time_mean")
        if value is None:
            value = metrics.get("mean_mpc_solve_time")
        return safe_float(value)

    if key == "mpc_solve_time_max":
        value = metrics.get("mpc_solve_time_max")
        if value is None:
            value = metrics.get("max_mpc_solve_time")
        return safe_float(value)

    if key == "replanning_cycle_time_mean":
        value = metrics.get("replanning_cycle_time_mean")
        if value is None:
            value = metrics.get("mean_replanning_cycle_time")
        return safe_float(value)

    if key == "replanning_cycle_time_max":
        value = metrics.get("replanning_cycle_time_max")
        if value is None:
            value = metrics.get("max_replanning_cycle_time")
        return safe_float(value)

    return safe_float(metrics.get(key))


def aggregate_by_method(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["method"]].append(run)

    aggregate = {}
    for method, method_runs in grouped.items():
        result = {
            "method": method,
            "num_runs": len(method_runs),
            "runs": [r["run_name"] for r in method_runs],
        }

        for bool_key in ["success", "collision", "collision_human", "collision_obstacle", "timeout"]:
            vals = [bool_value(r["metrics"], bool_key) for r in method_runs]
            vals = [v for v in vals if v is not None]
            rate_key = f"{bool_key}_rate"
            result[rate_key] = mean([1.0 if v else 0.0 for v in vals]) if vals else None

        for key in NUMERIC_METRICS:
            vals = [numeric_value(r["metrics"], key) for r in method_runs]
            vals = [v for v in vals if v is not None]
            result[f"{key}_mean"] = mean(vals)
            result[f"{key}_std"] = std(vals)

        aggregate[method] = result
    return aggregate


def write_all_runs_csv(runs: List[Dict[str, Any]], path: str) -> None:
    fields = ["method", "run_name", "run_dir"] + BOOLEAN_METRICS + NUMERIC_METRICS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {"method": run["method"], "run_name": run["run_name"], "run_dir": run["run_dir"]}
            metrics = run["metrics"]
            for key in BOOLEAN_METRICS + NUMERIC_METRICS:
                row[key] = metrics.get(key, "")
            writer.writerow(row)


def write_aggregate_csv(aggregate: Dict[str, Dict[str, Any]], path: str) -> None:
    fields = ["method", "num_runs", "success_rate", "collision_rate", "collision_human_rate", "collision_obstacle_rate", "timeout_rate"]
    for key in NUMERIC_METRICS:
        fields.extend([f"{key}_mean", f"{key}_std"])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method in sorted(aggregate.keys()):
            row = {field: aggregate[method].get(field, "") for field in fields}
            writer.writerow(row)


def fmt(value: Optional[float], decimals: int = 3, percent: bool = False) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{100.0 * value:.1f}"
    return f"{value:.{decimals}f}"


def mean_pm_std(row: Dict[str, Any], key: str, decimals: int = 3) -> str:
    m = row.get(f"{key}_mean")
    s = row.get(f"{key}_std")
    if m is None:
        return "--"
    if s is None:
        return fmt(m, decimals)
    return f"{fmt(m, decimals)} $\\pm$ {fmt(s, decimals)}"


def write_latex_table(aggregate: Dict[str, Dict[str, Any]], path: str) -> None:
    metrics_order = [
        "success_rate",
        "collision_rate",
        "final_goal_error",
        "path_length",
        "navigation_time",
        "minimum_human_distance",
        "minimum_obstacle_distance",
        "smoothness",
        "number_of_footsteps",
        "zmp_stability_margin_min",
        "dcm_stability_margin_min",
        "dcm_com_distance_max",
        "mpc_solve_time_mean",
        "replanning_cycle_time_mean",
        "num_replanning_cycles",
    ]

    methods = sorted(aggregate.keys())

    with open(path, "w") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{خلاصه آماری نتایج کمی ارزیابی سامانه ناوبری در سناریوهای آزمایشی}\n")
        f.write("\\label{tab:aggregate_navigation_metrics}\n")
        f.write("\\resizebox{0.95\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{|p{4cm}" + "|c" * len(methods) + "|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{معیار}")
        for method in methods:
            f.write(f" & \\textbf{{{method}}}")
        f.write(" \\\\ \n\\hline\n")

        for metric in metrics_order:
            name = DISPLAY_NAMES_FA.get(metric, metric)
            f.write(name)
            for method in methods:
                row = aggregate[method]
                if metric in ["success_rate", "collision_rate", "timeout_rate"]:
                    value = fmt(row.get(metric), decimals=1, percent=True)
                else:
                    value = mean_pm_std(row, metric, decimals=3)
                f.write(f" & {value}")
            f.write(" \\\\ \n\\hline\n")

        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics.json files from multiple navigation experiment runs.")
    parser.add_argument("root_dir", help="Root directory containing experiment run folders.")
    parser.add_argument("--output-dir", default=None, help="Directory for aggregate outputs. Defaults to root_dir.")
    parser.add_argument("--method-map", default=None, help="Optional JSON file mapping run folder names to method names.")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.expanduser(args.root_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir or root_dir))
    os.makedirs(output_dir, exist_ok=True)

    method_map = load_method_map(args.method_map)
    runs = discover_runs(root_dir, method_map)
    if not runs:
        raise RuntimeError(f"No metrics.json files found under: {root_dir}")

    aggregate = aggregate_by_method(runs)

    aggregate_json_path = os.path.join(output_dir, "aggregate_metrics.json")
    aggregate_csv_path = os.path.join(output_dir, "aggregate_metrics.csv")
    all_runs_csv_path = os.path.join(output_dir, "all_runs_metrics.csv")
    latex_path = os.path.join(output_dir, "aggregate_metrics_latex_table.tex")

    with open(aggregate_json_path, "w") as f:
        json.dump(aggregate, f, indent=4)

    write_aggregate_csv(aggregate, aggregate_csv_path)
    write_all_runs_csv(runs, all_runs_csv_path)
    write_latex_table(aggregate, latex_path)

    print("Found runs:", len(runs))
    for method in sorted(aggregate.keys()):
        row = aggregate[method]
        print(
            f"{method}: runs={row['num_runs']} | "
            f"success={fmt(row.get('success_rate'), percent=True)}% | "
            f"collision={fmt(row.get('collision_rate'), percent=True)}% | "
            f"time={mean_pm_std(row, 'navigation_time')} s | "
            f"path={mean_pm_std(row, 'path_length')} m"
        )

    print("\nSaved:")
    print("-", aggregate_json_path)
    print("-", aggregate_csv_path)
    print("-", all_runs_csv_path)
    print("-", latex_path)


if __name__ == "__main__":
    main()
