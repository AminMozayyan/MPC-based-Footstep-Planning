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

from social_mpc_planner.msg import (
    CircularObstacleArray,
    PredictedPeople,
    PredictedPeopleUncertainty,
    HumanArray,
)


class ExperimentLogger:
    def __init__(self):
        rospy.init_node("navigation_experiment_logger_node")

        self.output_root = os.path.expanduser(rospy.get_param("~output_root", "~/navigation_experiments"))
        self.experiment_name = rospy.get_param("~experiment_name", "mpc_run")
        self.robot_radius = float(rospy.get_param("~robot_radius", 0.25))
        self.default_human_radius = float(rospy.get_param("~default_human_radius", 0.20))
        self.collision_margin = float(rospy.get_param("~collision_margin", 0.0))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.10))
        self.goal_yaw_tolerance = math.radians(float(rospy.get_param("~goal_yaw_tolerance_deg", 10.0)))
        self.max_navigation_time = float(rospy.get_param("~max_navigation_time", 120.0))
        self.log_rate_hz = float(rospy.get_param("~log_rate_hz", 20.0))

        # Human source topics for independent logging.
        # These are logged separately from /predicted_people so that GT and
        # camera detections can be plotted and compared per human ID.
        self.gt_humans_topic = rospy.get_param("~gt_humans_topic", "/gazebo_humans_world")
        self.detected_humans_topic = rospy.get_param("~detected_humans_topic", "/detected_humans_world")
        self.predicted_people_topic = rospy.get_param("~predicted_people_topic", "/predicted_people")
        self.planner_human_source = rospy.get_param("~planner_human_source", "selected")
        self.log_human_sources_when_inactive = bool(rospy.get_param("~log_human_sources_when_inactive", False))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.output_root, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.gait_dir = os.path.join(self.run_dir, "gait")
        os.makedirs(self.gait_dir, exist_ok=True)

        rospy.set_param("/evaluation/run_dir", self.run_dir)
        rospy.set_param("/evaluation/gait_log_dir", self.gait_dir)

        rospy.loginfo("Gait logs directory: %s", self.gait_dir)

        self.active = False
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.goal = None
        self.goal_theta = None

        self.latest_odom = None
        self.latest_people = []
        self.latest_gt_humans = []
        self.latest_detected_humans = []
        self.latest_predicted_people = []
        self.latest_predicted_people_uncertainty = []
        self.latest_obstacles = []
        self.last_people_signature = None
        self.last_gt_humans_signature = None
        self.last_detected_humans_signature = None
        self.last_predicted_people_signature = None
        self.last_predicted_people_uncertainty_signature = None
        self.last_logged_obstacles_signature = None

        self.latest_zmp = None
        self.latest_dcm = None
        self.latest_com = None
        self.latest_support_polygon = []

        self.total_executed_footsteps = 0
        self.total_generated_footsteps = 0
        self.mpc_solve_times = []
        self.replan_cycle_times = []
        self.timing_stats = {}

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
        rospy.Subscriber(self.gt_humans_topic, HumanArray, self.gt_humans_callback)
        rospy.Subscriber(self.detected_humans_topic, HumanArray, self.detected_humans_callback)
        rospy.Subscriber(self.predicted_people_topic, PredictedPeople, self.predicted_people_callback)
        rospy.Subscriber(
            "/predicted_people_uncertainty",
            PredictedPeopleUncertainty,
            self.predicted_people_uncertainty_callback,
        )

        rospy.Subscriber("/evaluation/mpc_solve_time", Float32, self.mpc_solve_time_callback)
        rospy.Subscriber("/evaluation/replan_cycle_time", Float32, self.replan_cycle_time_callback)
        rospy.Subscriber("/evaluation/executed_footsteps_count", Int32, self.executed_footsteps_callback)
        rospy.Subscriber("/evaluation/generated_footsteps_count", Int32, self.generated_footsteps_callback)
        rospy.Subscriber("/evaluation/goal_reached", Bool, self.goal_reached_callback)
        rospy.Subscriber("/evaluation/planner_status", String, self.planner_status_callback)

        # Timing and frequency topics from perception, prediction and planner integration.
        timing_topics = {
            "/evaluation/camera_rgb_rate_hz": "camera_rgb_rate_hz",
            "/evaluation/camera_depth_rate_hz": "camera_depth_rate_hz",
            "/evaluation/detection_processing_time": "detection_processing_time",
            "/evaluation/detection_total_cycle_time": "detection_total_cycle_time",
            "/evaluation/detection_publish_rate_hz": "detection_publish_rate_hz",
            "/evaluation/gazebo_humans_publish_rate_hz": "gazebo_humans_publish_rate_hz",
            "/evaluation/gt_humans_publish_rate_hz": "gt_humans_publish_rate_hz",
            "/evaluation/prediction_input_rate_hz": "prediction_input_rate_hz",
            "/evaluation/prediction_processing_time": "prediction_processing_time",
            "/evaluation/prediction_total_cycle_time": "prediction_total_cycle_time",
            "/evaluation/prediction_publish_rate_hz": "prediction_publish_rate_hz",
        }
        for topic, metric_name in timing_topics.items():
            rospy.Subscriber(topic, Float32, self.timing_callback, callback_args=metric_name)

        count_topics = {
            "/evaluation/detection_num_humans": "detection_num_humans",
            "/evaluation/prediction_history_size": "prediction_history_size",
            "/evaluation/prediction_num_people": "prediction_num_people",
        }
        for topic, metric_name in count_topics.items():
            rospy.Subscriber(topic, Int32, self.timing_callback, callback_args=metric_name)

        rospy.Subscriber(rospy.get_param("~zmp_topic", "/zmp"), PointStamped, self.zmp_callback)
        rospy.Subscriber(rospy.get_param("~dcm_topic", "/dcm"), PointStamped, self.dcm_callback)
        rospy.Subscriber(rospy.get_param("~com_topic", "/com"), PointStamped, self.com_callback)
        rospy.Subscriber(rospy.get_param("~support_polygon_topic", "/support_polygon"), PolygonStamped, self.support_polygon_callback)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.log_rate_hz), self.timer_callback)
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("Experiment logger started. Output: %s", self.run_dir)

    def _open_csv_files(self):
        specs = {
            "trajectory": ["t", "x", "y", "theta", "goal_x", "goal_y", "goal_theta", "distance_to_goal"],
            "distances": ["t", "min_human_clearance", "min_obstacle_clearance", "collision_human", "collision_obstacle"],
            "planner": ["t", "event", "value"],
            "zmp": ["t", "zmp_x", "zmp_y", "stability_margin"],
            "dcm": ["t", "dcm_x", "dcm_y", "dcm_margin", "com_x", "com_y", "dcm_com_distance"],
            # selected_humans/humans: the first point of /predicted_people,
            # i.e., the human positions actually used by the predictor/MPC.
            "humans": ["t", "person_id", "x", "y", "radius"],

            # Independent raw sources for plotting and post-processing.
            "gt_humans": ["t", "person_id", "x", "y", "radius"],
            "detected_humans": ["t", "person_id", "x", "y", "radius"],

            "predicted_humans": ["t", "person_id", "pred_step", "x", "y", "radius"],
            "predicted_human_uncertainty": ["t", "person_id", "pred_step", "sigma_x", "sigma_y", "rho"],
            "events": ["t", "event", "details"],
            "static_obstacles": ["t", "obstacle_id", "x", "y", "radius"],
            "timing": ["t", "metric", "value"],
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
        
        q = msg.pose.orientation

        _, _, self.goal_theta = euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        self.goal = (
            msg.pose.position.x,
            msg.pose.position.y
        )

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
        self.timing_stats = {}

        self.last_people_signature = None
        self.last_gt_humans_signature = None
        self.last_detected_humans_signature = None
        self.last_predicted_people_signature = None
        self.last_predicted_people_uncertainty_signature = None

        self.write_event("new_goal", json.dumps({"goal_x": self.goal[0], "goal_y": self.goal[1]}))
        rospy.loginfo("Experiment logging started for goal x=%.3f y=%.3f", self.goal[0], self.goal[1])

    def odom_callback(self, msg):
        self.latest_odom = msg

    def static_obstacles_callback(self, msg):
        self.latest_obstacles = [(o.x, o.y, o.radius) for o in msg.obstacles]

        # Store obstacle geometry for post-processing and plotting.
        # Obstacles are usually static, so avoid writing the same list repeatedly.
        signature = tuple(
            (round(o.x, 4), round(o.y, 4), round(o.radius, 4))
            for o in msg.obstacles
        )

        if signature == self.last_logged_obstacles_signature:
            return

        self.last_logged_obstacles_signature = signature

        t = self.now_rel()
        for i, o in enumerate(msg.obstacles):
            self.writers["static_obstacles"].writerow([
                t,
                i,
                float(o.x),
                float(o.y),
                float(o.radius)
            ])

        self.files["static_obstacles"].flush()
        self.write_event(
            "static_obstacles_updated",
            json.dumps({
                "num_obstacles": len(msg.obstacles),
                "obstacles": [
                    {
                        "id": i,
                        "x": float(o.x),
                        "y": float(o.y),
                        "radius": float(o.radius)
                    }
                    for i, o in enumerate(msg.obstacles)
                ]
            })
        )

    def _humanarray_to_list(self, msg):
        humans = []
        for h in msg.humans:
            # Human messages in this project normally contain id, x and y.
            # If a radius field exists in a future message definition, use it;
            # otherwise fall back to the logger default.
            radius = getattr(h, "radius", self.default_human_radius)
            if radius is None or float(radius) <= 0.0:
                radius = self.default_human_radius

            humans.append((
                int(h.id),
                float(h.x),
                float(h.y),
                float(radius),
            ))
        return humans

    def _log_human_source_rows(self, file_key, humans, signature_attr):
        # Keep latest values even before a goal starts, but write CSV rows only
        # during an active run unless explicitly requested.
        should_log = (self.active and not self.finished) or self.log_human_sources_when_inactive
        if not should_log:
            return

        signature = tuple(
            (pid, round(x, 4), round(y, 4), round(r, 4))
            for pid, x, y, r in humans
        )

        if signature == getattr(self, signature_attr):
            return

        t = self.now_rel()
        for pid, x, y, r in humans:
            self.writers[file_key].writerow([t, pid, x, y, r])

        self.files[file_key].flush()
        setattr(self, signature_attr, signature)

    def gt_humans_callback(self, msg):
        self.latest_gt_humans = self._humanarray_to_list(msg)
        self._log_human_source_rows(
            "gt_humans",
            self.latest_gt_humans,
            "last_gt_humans_signature"
        )

    def detected_humans_callback(self, msg):
        self.latest_detected_humans = self._humanarray_to_list(msg)
        self._log_human_source_rows(
            "detected_humans",
            self.latest_detected_humans,
            "last_detected_humans_signature"
        )

    def predicted_people_callback(self, msg):
        """
        Store both the current/observed human position and the predicted
        trajectory for later visualization.

        Output files:
          humans.csv:
            t, person_id, x, y, radius

          predicted_humans.csv:
            t, person_id, pred_step, x, y, radius
        """
        people = []
        predicted_people = []

        for p in msg.people:
            radius = p.radius if p.radius > 0.0 else self.default_human_radius

            if len(p.trajectory) == 0:
                continue

            # First point is treated as the current/observed human position.
            pose0 = p.trajectory[0]
            people.append((p.id, pose0.x, pose0.y, radius))

            traj = []
            for k, pose in enumerate(p.trajectory):
                traj.append((k, pose.x, pose.y, radius))

            predicted_people.append({
                "id": p.id,
                "radius": radius,
                "trajectory": traj
            })

        self.latest_people = people
        self.latest_predicted_people = predicted_people

        if not self.active or self.finished:
            return

        # Log current human positions. Avoid writing exact duplicates at the same state.
        people_signature = tuple(
            (pid, round(x, 4), round(y, 4), round(r, 4))
            for pid, x, y, r in people
        )

        if people_signature != self.last_people_signature:
            t = self.now_rel()
            for pid, x, y, r in people:
                self.writers["humans"].writerow([t, pid, x, y, r])
            self.files["humans"].flush()
            self.last_people_signature = people_signature

        # Log predicted trajectories. These are time-stamped by the planning/logger
        # receive time and indexed by pred_step.
        pred_signature_items = []
        for person in predicted_people:
            pid = person["id"]
            for k, x, y, r in person["trajectory"]:
                pred_signature_items.append((pid, k, round(x, 4), round(y, 4), round(r, 4)))

        pred_signature = tuple(pred_signature_items)

        if pred_signature != self.last_predicted_people_signature:
            t = self.now_rel()
            for person in predicted_people:
                pid = person["id"]
                for k, x, y, r in person["trajectory"]:
                    self.writers["predicted_humans"].writerow([t, pid, k, x, y, r])
            self.files["predicted_humans"].flush()
            self.last_predicted_people_signature = pred_signature

    def predicted_people_uncertainty_callback(self, msg):
        """
        Store learned prediction uncertainty for later plotting/video rendering.

        Input topic:
          /predicted_people_uncertainty

        Output file:
          predicted_human_uncertainty.csv:
            t, person_id, pred_step, sigma_x, sigma_y, rho
        """
        predicted_uncertainty = []

        for p in msg.people:
            max_len = max(len(p.sigma_x), len(p.sigma_y), len(p.rho))
            trajectory_unc = []

            for k in range(max_len):
                sx = float(p.sigma_x[k]) if k < len(p.sigma_x) else float("nan")
                sy = float(p.sigma_y[k]) if k < len(p.sigma_y) else float("nan")
                corr = float(p.rho[k]) if k < len(p.rho) else float("nan")
                trajectory_unc.append((k, sx, sy, corr))

            predicted_uncertainty.append({
                "id": p.id,
                "trajectory": trajectory_unc,
            })

        self.latest_predicted_people_uncertainty = predicted_uncertainty

        if not self.active or self.finished:
            return

        signature_items = []
        for person in predicted_uncertainty:
            pid = person["id"]
            for k, sx, sy, corr in person["trajectory"]:
                signature_items.append((
                    pid,
                    k,
                    round(sx, 5) if math.isfinite(sx) else sx,
                    round(sy, 5) if math.isfinite(sy) else sy,
                    round(corr, 5) if math.isfinite(corr) else corr,
                ))

        signature = tuple(signature_items)
        if signature == self.last_predicted_people_uncertainty_signature:
            return

        t = self.now_rel()
        for person in predicted_uncertainty:
            pid = person["id"]
            for k, sx, sy, corr in person["trajectory"]:
                self.writers["predicted_human_uncertainty"].writerow([
                    t,
                    pid,
                    k,
                    sx,
                    sy,
                    corr,
                ])

        self.files["predicted_human_uncertainty"].flush()
        self.last_predicted_people_uncertainty_signature = signature


    def timing_callback(self, msg, metric_name):
        if not self.active:
            return
        value = float(msg.data)
        if metric_name not in self.timing_stats:
            self.timing_stats[metric_name] = []
        self.timing_stats[metric_name].append(value)
        self.writers["timing"].writerow([self.now_rel(), metric_name, value])
        self.files["timing"].flush()

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

        self.writers["trajectory"].writerow([t, x, y, theta, gx, gy, self.goal_theta, dist_to_goal])

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

        yaw_error = abs(
            math.atan2(
                math.sin(theta - self.goal_theta),
                math.cos(theta - self.goal_theta)
            )
        )

        if dist_to_goal <= self.goal_tolerance and yaw_error <= self.goal_yaw_tolerance:
            self.goal_reached = True
            self.finish("goal_pose_tolerance_reached")
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


    def build_timing_summary(self):
        summary = {}
        for metric, values in self.timing_stats.items():
            if not values:
                continue
            vals = [float(v) for v in values if math.isfinite(float(v))]
            if not vals:
                continue
            summary[metric] = {
                "count": len(vals),
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
        return summary

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
            "goal_yaw_tolerance_deg": math.degrees(self.goal_yaw_tolerance),
            "max_navigation_time": self.max_navigation_time,
            "robot_radius": self.robot_radius,
            "collision_margin": self.collision_margin,
            "num_static_obstacles": len(self.latest_obstacles),
            "planner_human_source": self.planner_human_source,
            "gt_humans_topic": self.gt_humans_topic,
            "detected_humans_topic": self.detected_humans_topic,
            "predicted_people_topic": self.predicted_people_topic,
            "num_gt_humans_latest": len(self.latest_gt_humans),
            "num_detected_humans_latest": len(self.latest_detected_humans),
            "static_obstacles": [
                {"id": i, "x": ox, "y": oy, "radius": r}
                for i, (ox, oy, r) in enumerate(self.latest_obstacles)
            ],
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
            "timing_summary": self.build_timing_summary(),
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
