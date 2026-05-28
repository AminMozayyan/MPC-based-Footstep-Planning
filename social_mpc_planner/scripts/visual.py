import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

# مسیر فایل کانفیگ
config_path = "/home/mozayyan/Surena_ws/src/surenaV_core/gait_planner/config/walk_config.json"

with open(config_path, "r") as f:
    config = json.load(f)

footsteps = config["footsteps"]

# ابعاد ثابت پا
foot_length = 0.22   # متر
foot_width = 0.10    # متر

fig, ax = plt.subplots(figsize=(10, 7))

for i, step in enumerate(footsteps):

    x = step[0]
    y = step[1]
    theta = step[3]   # زاویه (رادیان)

    foot = "Left" if i % 2 == 0 else "Right"
    color = "blue" if foot == "Left" else "red"

    # -----------------------------
    # رسم مستطیل پا
    # -----------------------------
    
    # مستطیل حول مرکز پا ساخته می‌شود
    rect = Rectangle(
        (-foot_length / 2, -foot_width / 2),
        foot_length,
        foot_width,
        linewidth=2,
        edgecolor=color,
        facecolor='none'
    )

    # چرخش + انتقال
    transform = (
        Affine2D()
        .rotate(theta)
        .translate(x, y)
        + ax.transData
    )

    rect.set_transform(transform)

    ax.add_patch(rect)

    # -----------------------------
    # رسم فلش جهت پا
    # -----------------------------
    
    arrow_len = 0.12

    dx = arrow_len * np.cos(theta)
    dy = arrow_len * np.sin(theta)

    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=0.025,
        head_length=0.03,
        fc=color,
        ec=color,
        length_includes_head=True
    )

    # مرکز پا
    ax.scatter(x, y, color=color, s=30)

    # شماره قدم
    ax.text(
        x + 0.03,
        y + 0.03,
        f"{i}",
        fontsize=12,
        color=color
    )

# تنظیمات نمودار
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Footstep Rectangles")

ax.axis("equal")
ax.grid(True)

plt.show()