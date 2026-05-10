#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/mozayyan/Surena_ws/human_validation.csv"

df = pd.read_csv(csv_path)

# -----------------------------------
# trajectory plot grouped by GT id
# -----------------------------------

plt.figure(figsize=(8,8))

for gt_id, group in df.groupby("gt_id"):

    group = group.sort_values("time")

    plt.plot(
        group["gt_x"],
        group["gt_y"],
        label=f"GT {gt_id}"
    )

    plt.plot(
        group["det_x"],
        group["det_y"],
        linestyle="--",
        label=f"Detected matched to GT {gt_id}"
    )

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Trajectory Comparison Grouped by Ground Truth ID")
plt.legend()
plt.grid()
plt.axis("equal")

# -----------------------------------
# error over time
# -----------------------------------

plt.figure(figsize=(10,4))

plt.plot(df["distance_error"])

plt.xlabel("Sample")
plt.ylabel("Error [m]")

plt.title("Position Error Over Time")

plt.grid()



# -----------------------------------
# x error over time
# -----------------------------------

plt.figure(figsize=(10,4))

plt.plot(df["dx"])

plt.xlabel("Sample")
plt.ylabel("X Error [m]")

plt.title("X Error Over Time")

plt.grid()


# -----------------------------------
# y error over time
# -----------------------------------

plt.figure(figsize=(10,4))

plt.plot(df["dy"])

plt.xlabel("Sample")
plt.ylabel("Y Error [m]")

plt.title("Y Error Over Time")

plt.grid()


# -----------------------------------
# absolute x/y error comparison
# -----------------------------------

plt.figure(figsize=(10,4))

plt.plot(df["dx"].abs(), label="|X Error|")
plt.plot(df["dy"].abs(), label="|Y Error|")

plt.xlabel("Sample")
plt.ylabel("Absolute Error [m]")

plt.title("Absolute X/Y Error Over Time")

plt.legend()
plt.grid()


# -----------------------------------
# histogram
# -----------------------------------

plt.figure(figsize=(6,4))

plt.hist(df["distance_error"], bins=30)

plt.xlabel("Error [m]")
plt.ylabel("Count")

plt.title("Error Histogram")

plt.grid()

# -----------------------------------
# print metrics
# -----------------------------------

print("Mean Error:", df["distance_error"].mean())
print("RMSE:", (df["distance_error"]**2).mean()**0.5)

print("X RMSE:", (df["dx"]**2).mean()**0.5)
print("Y RMSE:", (df["dy"]**2).mean()**0.5)

plt.show()


# -----------------------------------
# x trajectory comparison
# -----------------------------------

plt.figure(figsize=(10,4))

plt.plot(df["gt_x"], label="GT X")
plt.plot(df["det_x"], label="Detected X")

plt.xlabel("Sample")
plt.ylabel("X [m]")

plt.title("X Trajectory Comparison")

plt.legend()
plt.grid()


# -----------------------------------
# y trajectory comparison
# -----------------------------------

plt.figure(figsize=(10,4))

plt.plot(df["gt_y"], label="GT Y")
plt.plot(df["det_y"], label="Detected Y")

plt.xlabel("Sample")
plt.ylabel("Y [m]")

plt.title("Y Trajectory Comparison")

plt.legend()
plt.grid()


plt.figure(figsize=(6,4))
plt.hist(df["dx"], bins=40)
plt.title("X Error Distribution")
plt.xlabel("dx [m]")
plt.grid()

plt.figure(figsize=(6,4))
plt.hist(df["dy"], bins=40)
plt.title("Y Error Distribution")
plt.xlabel("dy [m]")
plt.grid()