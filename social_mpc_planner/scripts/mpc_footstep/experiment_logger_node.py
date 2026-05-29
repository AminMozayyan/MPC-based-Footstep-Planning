#!/usr/bin/env python3
"""
Experiment logger for humanoid navigation evaluation.

This node records raw data during one navigation run and writes CSV files that
can later be processed by compute_navigation_metrics.py.

Recommended topics:
  /odom                                      nav_msgs/Odometry
  /move_base_simple/goal                     geometry_msgs/PoseStamped
  /static_obstacles                          social_mpc_planner/CircularObstacleArray
  /predicted_people                          social_mpc_planner/PredictedPeople
  /evaluation/mpc_solve_time                 std_msgs/Float32
  /evaluation/replan_cycle_time              std_msgs/Float32
  /evaluation/executed_footsteps_count       std_msgs/Int32
  /evaluation/generated_footsteps_count      std_msgs/Int32
  /evaluation/goal_reached                   std_msgs/Bool

Optional stability topics:
  /zmp                                       geometry_msgs/PointStamped
  /dcm                                       geometry_msgs/PointStamped
  /com                                       geometry_msgs/PointStamped
  /support_polygon                           geometry_msgs/PolygonStamped
"""

import csv
import json
import math
import os
from datetime import datetime

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped, PolygonStamped
from std_msgs.msg import Float32, Int32, Bool, String
from tf.transformations import euler_from_quaternion

from social_mpc_planner.msg import CircularObstacleArray, PredictedPeople


class ExperimentLogger:
    def __init__(self):
        rospy.init_node("navigation_experiment_logger_node")

        self.output_root = os.path.expanduser(rospy.get_param("~output_root", "~/navigation_experiments"))
        self.experiment_name = rospy.get_param("~experiment_name", "mpc_run")
        self.robot_radius = float(rospy.get_param("~robot_radius", 0.25))
        self.default_human_radius = float(rospy.get_param("~default_human_radius", 0.20))
        self.collision_margin = float(rospy.get_param("~collision_margin", 0.0))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.15))
        self.max_navigation_time = float(rospy.get_param("~max_navigation_time", 120.0))
        self.log_rate_hz = float(rospy.get_param("~log_rate_hz", 20.0))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_root, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.active = False
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.goal = None

        self.latest_odom = None
        self.latest_people = []
        self.latest_obstacles = []

        self.latest_zmp = None
        self.latest_dcm = None
        self.latest_com = None
        self.latest_support_polygon = []

        self.total_executed_footsteps = 0
        self.total_generated_footsteps = 0
        self.mpc_solve_times = []
        self.replan_cycle_times = []

        self.min_human_clearance = float("inf")
        self.min_obstacle_clearance = float("inf")
        self.min_zmp_margin = float("inf")
        self.min_dcm_margin = float("inf")
        self.max_dcm_com_distance = float("-inf")

        self.collision_human = False
        self.collision_obstacle = False
        self.goal_reached = False
        self.last_status = "idle"

        self.files = {}
        self.writers = {}
        self._open_csv_files()

        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        rospy.Subscriber("/static_obstacles", CircularObstacleArray, self.static_obstacles_callback)
        rospy.Subscriber("/predicted_people", PredictedPeople, self.predicted_people_callback)

        rospy.Subscriber("/evaluation/mpc_solve_time", Float32, self.mpc_solve_time_callback)
        rospy.Subscriber("/evaluation/replan_cycle_time", Float32, self.replan_cycle_time_callback)
        rospy.Subscriber("/evaluation/executed_footsteps_count", Int32, self.executed_footsteps_callback)
        rospy.Subscriber("/evaluation/generated_footsteps_count", Int32, self.generated_footsteps_callback)
        rospy.Subscriber("/evaluation/goal_reached", Bool, self.goal_reached_callback)
        rospy.Subscriber("/evaluation/planner_status", String, self.planner_status_callback)

        rospy.Subscriber(rospy.get_param("~zmp_topic", "/zmp"), PointStamped, self.zmp_callback)
        rospy.Subscriber(rospy.get_param("~dcm_topic", "/dcm"), PointStamped, self.dcm_callback)
        rospy.Subscriber(rospy.get_param("~com_topic", "/com"), PointStamped, self.com_callback)
        rospy.Subscriber(rospy.get_param("~support_polygon_topic", "/support_polygon"), PolygonStamped, self.support_polygon_callback)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.log_rate_hz), self.timer_callback)
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("Experiment logger started. Output: %s", self.run_dir)

    def _open_csv_files(self):
        specs = {
            "trajectory": ["t", "x", "y", "theta", "goal_x", "goal_y", "distance_to_goal"],
            "distances": ["t", "min_human_clearance", "min_obstacle_clearance", "collision_human", "collision_obstacle"],
            "planner": ["t", "event", "value"],
            "zmp": ["t", "zmp_x", "zmp_y", "stability_margin"],
            "dcm": ["t", "dcm_x", "dcm_y", "dcm_margin", "com_x", "com_y", "dcm_com_distance"],
            "events": ["t", "event", "details"],
        }
        for name, header in specs.items():
            f = open(os.path.join(self.run_dir, f"{name}.csv"), "w", newline="")
            w = csv.writer(f)
            w.writerow(header)
            self.files[name] = f
            self.writers[name] = w

    def now_rel(self):
        if self.start_time is None:
            return 0.0
        return (rospy.Time.now() - self.start_time).to_sec()

    @staticmethod
    def yaw_from_odom(msg):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    def write_event(self, event, details=""):
        self.writers["events"].writerow([self.now_rel(), event, details])
        self.files["events"].flush()

    def goal_callback(self, msg):
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.active = True
        self.finished = False
        self.goal_reached = False
        self.start_time = rospy.Time.now()
        self.end_time = None

        self.min_human_clearance = float("inf")
        self.min_obstacle_clearance = float("inf")
        self.min_zmp_margin = float("inf")
        self.min_dcm_margin = float("inf")
        self.max_dcm_com_distance = float("-inf")

        self.collision_human = False
        self.collision_obstacle = False
        self.total_executed_footsteps = 0
        self.total_generated_footsteps = 0
        self.mpc_solve_times = []
        self.replan_cycle_times = []

        self.write_event("new_goal", json.dumps({"goal_x": self.goal[0], "goal_y": self.goal[1]}))
        rospy.loginfo("Experiment logging started for goal x=%.3f y=%.3f", self.goal[0], self.goal[1])

    def odom_callback(self, msg):
        self.latest_odom = msg

    def static_obstacles_callback(self, msg):
        self.latest_obstacles = [(o.x, o.y, o.radius) for o in msg.obstacles]

    def predicted_people_callback(self, msg):
        people = []
        for p in msg.people:
            if len(p.trajectory) == 0:
                continue
            pose = p.trajectory[0]
            radius = p.radius if p.radius > 0.0 else self.default_human_radius
            people.append((p.id, pose.x, pose.y, radius))
        self.latest_people = people

    def mpc_solve_time_callback(self, msg):
        if not self.active:
            return
        value = float(msg.data)
        self.mpc_solve_times.append(value)
        self.writers["planner"].writerow([self.now_rel(), "mpc_solve_time", value])
        self.files["planner"].flush()

    def replan_cycle_time_callback(self, msg):
        if not self.active:
            return
        value = float(msg.data)
        self.replan_cycle_times.append(value)
        self.writers["planner"].writerow([self.now_rel(), "replan_cycle_time", value])
        self.files["planner"].flush()

    def executed_footsteps_callback(self, msg):
        if not self.active:
            return
        value = int(msg.data)
        self.total_executed_footsteps += value
        self.writers["planner"].writerow([self.now_rel(), "executed_footsteps_count", value])
        self.files["planner"].flush()

    def generated_footsteps_callback(self, msg):
        if not self.active:
            return
        value = int(msg.data)
        self.total_generated_footsteps += value
        self.writers["planner"].writerow([self.now_rel(), "generated_footsteps_count", value])
        self.files["planner"].flush()

    def planner_status_callback(self, msg):
        if not self.active:
            return
        self.last_status = msg.data
        self.write_event("planner_status", msg.data)

    def goal_reached_callback(self, msg):
        if not self.active:
            return
        if bool(msg.data):
            self.goal_reached = True
            self.finish("goal_reached")

    def zmp_callback(self, msg):
        self.latest_zmp = (msg.point.x, msg.point.y)

    def dcm_callback(self, msg):
        self.latest_dcm = (msg.point.x, msg.point.y)

    def com_callback(self, msg):
        self.latest_com = (msg.point.x, msg.point.y)

    def support_polygon_callback(self, msg):
        self.latest_support_polygon = [(p.x, p.y) for p in msg.polygon.points]

    @staticmethod
    def point_to_segment_distance(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        denom = vx * vx + vy * vy
        if denom <= 1e-12:
            return math.hypot(px - ax, py - ay)
        u = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        cx, cy = ax + u * vx, ay + u * vy
        return math.hypot(px - cx, py - cy)

    @staticmethod
    def point_in_polygon(px, py, poly):
        inside = False
        n = len(poly)
        if n < 3:
            return False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersect = ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
            )
            if intersect:
                inside = not inside
            j = i
        return inside

    def signed_polygon_margin(self, px, py, poly):
        if len(poly) < 2:
            return float("nan")
        d = min(
            self.point_to_segment_distance(
                px, py,
                poly[i][0], poly[i][1],
                poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1]
            )
            for i in range(len(poly))
        )
        return d if self.point_in_polygon(px, py, poly) else -d

    def timer_callback(self, _event):
        if not self.active or self.finished or self.latest_odom is None or self.goal is None:
            return

        t = self.now_rel()
        x = self.latest_odom.pose.pose.position.x
        y = self.latest_odom.pose.pose.position.y
        theta = self.yaw_from_odom(self.latest_odom)
        gx, gy = self.goal
        dist_to_goal = math.hypot(gx - x, gy - y)

        self.writers["trajectory"].writerow([t, x, y, theta, gx, gy, dist_to_goal])

        human_clearance = float("inf")
        for _pid, hx, hy, hr in self.latest_people:
            clearance = math.hypot(x - hx, y - hy) - self.robot_radius - hr
            human_clearance = min(human_clearance, clearance)

        obstacle_clearance = float("inf")
        for ox, oy, obstacle_radius in self.latest_obstacles:
            clearance = math.hypot(x - ox, y - oy) - self.robot_radius - obstacle_radius
            obstacle_clearance = min(obstacle_clearance, clearance)

        if human_clearance < self.min_human_clearance:
            self.min_human_clearance = human_clearance
        if obstacle_clearance < self.min_obstacle_clearance:
            self.min_obstacle_clearance = obstacle_clearance

        if human_clearance <= self.collision_margin:
            self.collision_human = True
        if obstacle_clearance <= self.collision_margin:
            self.collision_obstacle = True

        self.writers["distances"].writerow([
            t, human_clearance, obstacle_clearance,
            int(self.collision_human), int(self.collision_obstacle)
        ])

        if self.latest_zmp is not None and len(self.latest_support_polygon) >= 3:
            zmp_x, zmp_y = self.latest_zmp
            margin = self.signed_polygon_margin(zmp_x, zmp_y, self.latest_support_polygon)
            if not math.isnan(margin) and margin < self.min_zmp_margin:
                self.min_zmp_margin = margin
            self.writers["zmp"].writerow([t, zmp_x, zmp_y, margin])

        if self.latest_dcm is not None:
            dcm_x, dcm_y = self.latest_dcm
            dcm_margin = float("nan")
            if len(self.latest_support_polygon) >= 3:
                dcm_margin = self.signed_polygon_margin(dcm_x, dcm_y, self.latest_support_polygon)
                if not math.isnan(dcm_margin) and dcm_margin < self.min_dcm_margin:
                    self.min_dcm_margin = dcm_margin

            com_x, com_y, dcm_com = float("nan"), float("nan"), float("nan")
            if self.latest_com is not None:
                com_x, com_y = self.latest_com
                dcm_com = math.hypot(dcm_x - com_x, dcm_y - com_y)
                if dcm_com > self.max_dcm_com_distance:
                    self.max_dcm_com_distance = dcm_com

            self.writers["dcm"].writerow([t, dcm_x, dcm_y, dcm_margin, com_x, com_y, dcm_com])

        for f in self.files.values():
            f.flush()

        if dist_to_goal <= self.goal_tolerance:
            self.goal_reached = True
            self.finish("goal_tolerance_reached")
        elif t >= self.max_navigation_time:
            self.finish("timeout")

    def finish(self, reason):
        if self.finished:
            return
        self.finished = True
        self.active = False
        self.end_time = rospy.Time.now()
        self.write_event("finish", reason)
        self.write_summary(reason)
        rospy.loginfo("Experiment finished: %s. Summary written to %s", reason, self.run_dir)

    @staticmethod
    def finite_or_none(value):
        if value == float("inf") or value == float("-inf") or math.isnan(value):
            return None
        return value

    def write_summary(self, reason="running"):
        duration = None
        if self.start_time is not None:
            end = self.end_time if self.end_time is not None else rospy.Time.now()
            duration = (end - self.start_time).to_sec()

        summary = {
            "reason": reason,
            "goal_reached": bool(self.goal_reached),
            "timeout": bool(reason == "timeout"),
            "collision_human": bool(self.collision_human),
            "collision_obstacle": bool(self.collision_obstacle),
            "collision": bool(self.collision_human or self.collision_obstacle),
            "navigation_time": duration,
            "goal_tolerance": self.goal_tolerance,
            "max_navigation_time": self.max_navigation_time,
            "robot_radius": self.robot_radius,
            "collision_margin": self.collision_margin,
            "minimum_human_distance": self.finite_or_none(self.min_human_clearance),
            "minimum_obstacle_distance": self.finite_or_none(self.min_obstacle_clearance),
            "minimum_zmp_stability_margin": self.finite_or_none(self.min_zmp_margin),
            "minimum_dcm_stability_margin": self.finite_or_none(self.min_dcm_margin),
            "maximum_dcm_com_distance": self.finite_or_none(self.max_dcm_com_distance),
            "total_executed_footsteps": self.total_executed_footsteps,
            "total_generated_footsteps": self.total_generated_footsteps,
            "mpc_solve_time_mean": sum(self.mpc_solve_times) / len(self.mpc_solve_times) if self.mpc_solve_times else None,
            "mpc_solve_time_max": max(self.mpc_solve_times) if self.mpc_solve_times else None,
            "replan_cycle_time_mean": sum(self.replan_cycle_times) / len(self.replan_cycle_times) if self.replan_cycle_times else None,
            "replan_cycle_time_max": max(self.replan_cycle_times) if self.replan_cycle_times else None,
        }
        with open(os.path.join(self.run_dir, "summary_live.json"), "w") as f:
            json.dump(summary, f, indent=4)

    def shutdown(self):
        if self.active and not self.finished:
            self.finish("ros_shutdown")
        else:
            self.write_summary("shutdown")
        for f in self.files.values():
            f.close()


if __name__ == "__main__":
    node = ExperimentLogger()
    rospy.spin()
