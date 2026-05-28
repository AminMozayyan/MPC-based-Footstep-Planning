#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = "/home/mozayyan/Surena_ws/human_validation.csv"

df = pd.read_csv(csv_path)

MAX_TRAJECTORY_GAP = 3.0
MAX_TIME_GAP = 3.0


def break_line_on_gap(x_series, y_series, time_series, max_gap_dist, max_gap_time):

    x_out = []
    y_out = []

    prev_x = None
    prev_y = None
    prev_t = None

    for x, y, t in zip(x_series, y_series, time_series):

        if prev_x is not None:

            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            dt = t - prev_t

            if dist > max_gap_dist or dt > max_gap_time:
                x_out.append(np.nan)
                y_out.append(np.nan)

        x_out.append(x)
        y_out.append(y)

        prev_x = x
        prev_y = y
        prev_t = t

    return x_out, y_out


def break_line_on_distance_only(x_series, y_series, max_gap_dist):

    x_out = []
    y_out = []

    prev_x = None
    prev_y = None

    for x, y in zip(x_series, y_series):

        if prev_x is not None:
            dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)

            if dist > max_gap_dist:
                x_out.append(np.nan)
                y_out.append(np.nan)

        x_out.append(x)
        y_out.append(y)

        prev_x = x
        prev_y = y

    return x_out, y_out


# -------------------------------------------------
# Split CSV rows
# -------------------------------------------------

if "row_type" in df.columns:
    df_det = df[df["row_type"] == "detection"].copy()
    df_robot = df[df["row_type"] == "robot"].copy()
else:
    df_det = df.copy()
    df_robot = df[["time", "robot_x", "robot_y"]].copy() if "robot_x" in df.columns else pd.DataFrame()


# -------------------------------------------------
# Metrics
# -------------------------------------------------

print("Detection Samples:", len(df_det))

if len(df_det) > 0:
    print("Mean Error:", df_det["distance_error"].mean())
    print("RMSE:", (df_det["distance_error"] ** 2).mean() ** 0.5)
    print("X RMSE:", (df_det["dx"] ** 2).mean() ** 0.5)
    print("Y RMSE:", (df_det["dy"] ** 2).mean() ** 0.5)
    print("X Offset:", df_det["dx"].mean())
    print("Y Offset:", df_det["dy"].mean())

    if "overall_detection_rate" in df_det.columns:
        valid = df_det["overall_detection_rate"].dropna()
        if len(valid) > 0:
            print("Final Detection Rate:", valid.iloc[-1])

    if "overall_tracking_success_rate" in df_det.columns:
        valid = df_det["overall_tracking_success_rate"].dropna()
        if len(valid) > 0:
            print("Final Tracking Success Rate:", valid.iloc[-1])

    if "id_switch_count" in df_det.columns:
        valid = df_det["id_switch_count"].dropna()
        if len(valid) > 0:
            print("Final ID Switch Count:", int(valid.iloc[-1]))

print("Robot Samples:", len(df_robot))


# -------------------------------------------------
# Trajectory plot: GT + Detected + Robot
# -------------------------------------------------

plt.figure(figsize=(8, 8))

if len(df_det) > 0:

    unique_ids = sorted(df_det["gt_id"].dropna().unique())
    id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_ids)}

    for gt_id, group in df_det.groupby("gt_id"):

        group = group.sort_values("time")
        display_id = id_map[gt_id]

        gt_x = group["gt_x"].values
        gt_y = group["gt_y"].values

        det_x, det_y = break_line_on_gap(
            group["det_x"].values,
            group["det_y"].values,
            group["time"].values,
            MAX_TRAJECTORY_GAP,
            MAX_TIME_GAP
        )

        plt.plot(
            gt_x,
            gt_y,
            label=f"GT {display_id}",
            linewidth=2.5
        )

        plt.plot(
            det_x,
            det_y,
            linestyle="--",
            label=f"Detected {display_id}",
            linewidth=2.5,
        )


if len(df_robot) > 1 and "robot_x" in df_robot.columns and "robot_y" in df_robot.columns:

    robot_df = df_robot[["time", "robot_x", "robot_y"]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    robot_df = robot_df.sort_values("time").drop_duplicates()

    if len(robot_df) > 1:

        robot_x, robot_y = break_line_on_distance_only(
            robot_df["robot_x"].values,
            robot_df["robot_y"].values,
            max_gap_dist=1.5
        )

        plt.plot(
            robot_x,
            robot_y,
            linewidth=3,
            label="Robot trajectory"
        )

        plt.scatter(
            robot_df["robot_x"].iloc[0],
            robot_df["robot_y"].iloc[0],
            marker="o",
            label="Robot start"
        )

        plt.scatter(
            robot_df["robot_x"].iloc[-1],
            robot_df["robot_y"].iloc[-1],
            marker="x",
            label="Robot end"
        )


plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Trajectory Comparison with Robot Path")
plt.legend()
plt.grid()
plt.axis("equal")


# # -------------------------------------------------
# # Error over time
# # -------------------------------------------------

# if len(df_det) > 0:

#     plt.figure(figsize=(10, 4))
#     plt.plot(df_det["distance_error"].values)
#     plt.xlabel("Sample")
#     plt.ylabel("Error [m]")
#     plt.title("Position Error Over Time")
#     plt.grid()

#     plt.figure(figsize=(10, 4))
#     plt.plot(df_det["dx"].values, label="dx = det_x - gt_x")
#     plt.plot(df_det["dy"].values, label="dy = det_y - gt_y")
#     plt.axhline(0.0, linestyle="--")
#     plt.xlabel("Sample")
#     plt.ylabel("Signed Error [m]")
#     plt.title("Signed X/Y Error Over Time")
#     plt.legend()
#     plt.grid()

#     plt.figure(figsize=(10, 4))
#     plt.plot(df_det["dx"].abs().values, label="|X Error|")
#     plt.plot(df_det["dy"].abs().values, label="|Y Error|")
#     plt.xlabel("Sample")
#     plt.ylabel("Absolute Error [m]")
#     plt.title("Absolute X/Y Error Over Time")
#     plt.legend()
#     plt.grid()


# # -------------------------------------------------
# # Detection / tracking rates
# # -------------------------------------------------

# if len(df_det) > 0 and "overall_detection_rate" in df_det.columns:

#     plt.figure(figsize=(10, 4))
#     plt.plot(df_det["overall_detection_rate"].values, label="Detection Rate")
#     plt.plot(df_det["overall_tracking_success_rate"].values, label="Tracking Success Rate")
#     plt.xlabel("Sample")
#     plt.ylabel("Rate")
#     plt.ylim(-0.05, 1.05)
#     plt.title("Detection and Tracking Success Rates")
#     plt.legend()
#     plt.grid()


# if len(df_det) > 0 and "id_switch_count" in df_det.columns:

#     plt.figure(figsize=(10, 4))
#     plt.plot(df_det["id_switch_count"].values)
#     plt.xlabel("Sample")
#     plt.ylabel("ID Switch Count")
#     plt.title("Cumulative ID Switches")
#     plt.grid()


# # -------------------------------------------------
# # Histograms
# # -------------------------------------------------

# if len(df_det) > 0:

#     plt.figure(figsize=(6, 4))
#     plt.hist(df_det["distance_error"].dropna(), bins=30)
#     plt.xlabel("Error [m]")
#     plt.ylabel("Count")
#     plt.title("Error Histogram")
#     plt.grid()

#     plt.figure(figsize=(6, 4))
#     plt.hist(df_det["dx"].dropna(), bins=40)
#     plt.title("X Error Distribution")
#     plt.xlabel("dx [m]")
#     plt.grid()

#     plt.figure(figsize=(6, 4))
#     plt.hist(df_det["dy"].dropna(), bins=40)
#     plt.title("Y Error Distribution")
#     plt.xlabel("dy [m]")
#     plt.grid()


# # -------------------------------------------------
# # X trajectory over samples
# # -------------------------------------------------

# if len(df_det) > 0:

#     plt.figure(figsize=(10, 4))
#     plt.plot(df_det["gt_x"].values, label="GT X")
#     plt.plot(df_det["det_x"].values, label="Detected X")

#     if len(df_robot) > 0 and "robot_x" in df_robot.columns:
#         robot_df = df_robot[["robot_x"]].replace([np.inf, -np.inf], np.nan).dropna()
#         plt.plot(robot_df["robot_x"].values, label="Robot X", linewidth=2)

#     plt.xlabel("Sample")
#     plt.ylabel("X [m]")
#     plt.title("X Trajectory Comparison")
#     plt.legend()
#     plt.grid()


# # -------------------------------------------------
# # Y trajectory over samples
# # -------------------------------------------------

# if len(df_det) > 0:

#     plt.figure(figsize=(10, 4))
#     plt.plot(df_det["gt_y"].values, label="GT Y")
#     plt.plot(df_det["det_y"].values, label="Detected Y")

#     if len(df_robot) > 0 and "robot_y" in df_robot.columns:
#         robot_df = df_robot[["robot_y"]].replace([np.inf, -np.inf], np.nan).dropna()
#         plt.plot(robot_df["robot_y"].values, label="Robot Y", linewidth=2)

#     plt.xlabel("Sample")
#     plt.ylabel("Y [m]")
#     plt.title("Y Trajectory Comparison")
#     plt.legend()
#     plt.grid()


plt.show()