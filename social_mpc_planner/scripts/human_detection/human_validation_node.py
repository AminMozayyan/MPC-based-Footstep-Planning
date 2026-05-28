#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import csv

from social_mpc_planner.msg import HumanArray, HumanValidation
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class HumanValidationNode:

    def __init__(self):

        rospy.init_node("human_validation_node")
        rospy.on_shutdown(self.shutdown_hook)

        self.latest_gazebo_msg = None
        self.latest_robot_msg = None

        self.match_threshold = rospy.get_param("~match_threshold", 1.0)
        self.max_match_distance = rospy.get_param("~max_match_distance", 1.0)

        self.camera_fov_deg = rospy.get_param("~camera_fov_deg", 70.0)
        self.camera_fov = math.radians(self.camera_fov_deg)
        self.max_visible_range = rospy.get_param("~max_visible_range", 8.0)
        self.min_visible_range = rospy.get_param("~min_visible_range", 0.05)

        self.gt_tracks = {}
        self.det_tracks = {}
        self.robot_track = []

        self.max_track_len = rospy.get_param("~max_track_len", 300)

        self.match_memory = {}

        self.errors = []
        self.x_errors = []
        self.y_errors = []

        self.frame_count = 0
        self.total_visible_gt_count = 0
        self.total_detected_count = 0
        self.total_valid_matches = 0
        self.total_visible_tracked_count = 0

        self.id_switch_count = 0
        self.gt_to_det_assignment = {}

        self.csv_path = rospy.get_param(
            "~csv_path",
            "/home/mozayyan/Surena_ws/human_validation.csv"
        )

        self.csv_file = open(self.csv_path, "w")
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "row_type",
            "time",
            "det_id",
            "gt_id",
            "det_x",
            "det_y",
            "gt_x",
            "gt_y",
            "dx",
            "dy",
            "distance_error",
            "robot_x",
            "robot_y",
            "robot_yaw",
            "is_gt_visible",
            "visible_gt_count",
            "frame_detection_rate",
            "frame_tracking_success_rate",
            "overall_detection_rate",
            "overall_tracking_success_rate",
            "id_switch_count"
        ])

        self.csv_file.flush()

        rospy.loginfo("Saving CSV to: %s", self.csv_path)
        rospy.loginfo(
            "FOV validation: fov=%.1f deg | max_range=%.2f m",
            self.camera_fov_deg,
            self.max_visible_range
        )

        rospy.Subscriber(
            "/gazebo_humans_world",
            HumanArray,
            self.gazebo_callback,
            queue_size=10
        )

        rospy.Subscriber(
            "/detected_humans_world",
            HumanArray,
            self.detected_callback,
            queue_size=10
        )

        rospy.Subscriber(
            "/robot_pose_world",
            HumanArray,
            self.robot_callback,
            queue_size=10
        )

        self.validation_pub = rospy.Publisher(
            "/human_validation",
            HumanValidation,
            queue_size=10
        )

        self.marker_pub = rospy.Publisher(
            "/human_validation_markers",
            MarkerArray,
            queue_size=10
        )

        rospy.loginfo("Human Validation Node Started")

    def gazebo_callback(self, msg):
        self.latest_gazebo_msg = msg

        # Log all Gazebo ground-truth humans independently of the detection callback.
        # This guarantees that GT trajectories are saved even when YOLO/tracker
        # publishes no detection for a frame, for example during occlusion.
        robot_pose = self.get_robot_pose()

        if robot_pose is not None:
            robot_x, robot_y = robot_pose
        else:
            robot_x = float("nan")
            robot_y = float("nan")

        robot_yaw = self.estimate_robot_yaw_from_track()
        if robot_yaw is None:
            robot_yaw = float("nan")

        visible_gts = self.get_visible_gt_humans(msg.humans)
        visible_gt_ids = set(gt.id for gt in visible_gts)
        frame_visible_gt_count = len(visible_gts)

        overall_detection_rate = (
            float(self.total_valid_matches) / float(self.total_visible_gt_count)
            if self.total_visible_gt_count > 0 else float("nan")
        )

        overall_tracking_success_rate = overall_detection_rate

        t = msg.header.stamp.to_sec()

        for gt in msg.humans:
            self.update_track(self.gt_tracks, gt.id, gt.x, gt.y)

            self.csv_writer.writerow([
                "gt",
                t,
                -1,
                gt.id,
                float("nan"),
                float("nan"),
                gt.x,
                gt.y,
                float("nan"),
                float("nan"),
                float("nan"),
                robot_x,
                robot_y,
                robot_yaw,
                int(gt.id in visible_gt_ids),
                frame_visible_gt_count,
                float("nan"),
                float("nan"),
                overall_detection_rate,
                overall_tracking_success_rate,
                self.id_switch_count
            ])

        self.csv_file.flush()

    def robot_callback(self, msg):

        if len(msg.humans) == 0:
            return

        self.latest_robot_msg = msg

        robot = msg.humans[0]
        self.update_robot_track(robot.x, robot.y)

        robot_yaw = self.estimate_robot_yaw_from_track()
        if robot_yaw is None:
            robot_yaw = float("nan")

        t = msg.header.stamp.to_sec()

        self.csv_writer.writerow([
            "robot",
            t,
            -1,
            -1,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            robot.x,
            robot.y,
            robot_yaw,
            0,
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            self.id_switch_count
        ])

        self.csv_file.flush()

    def get_robot_pose(self):

        if self.latest_robot_msg is None or len(self.latest_robot_msg.humans) == 0:
            return None

        robot = self.latest_robot_msg.humans[0]
        return robot.x, robot.y

    def estimate_robot_yaw_from_track(self):

        if len(self.robot_track) < 2:
            return None

        x1, y1 = self.robot_track[-2]
        x2, y2 = self.robot_track[-1]

        dx = x2 - x1
        dy = y2 - y1

        if math.sqrt(dx * dx + dy * dy) < 1e-4:
            return None

        return math.atan2(dy, dx)

    def normalize_angle(self, angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def is_gt_in_fov(self, gt):

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            return True

        robot_x, robot_y = robot_pose

        robot_yaw = self.estimate_robot_yaw_from_track()

        if robot_yaw is None:
            return True

        dx = gt.x - robot_x
        dy = gt.y - robot_y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.min_visible_range or dist > self.max_visible_range:
            return False

        angle_to_gt = math.atan2(dy, dx)
        rel_angle = self.normalize_angle(angle_to_gt - robot_yaw)

        return abs(rel_angle) <= 0.5 * self.camera_fov

    def get_visible_gt_humans(self, gt_humans):
        return [gt for gt in gt_humans if self.is_gt_in_fov(gt)]

    def distance(self, h1, h2):

        dx = h1.x - h2.x
        dy = h1.y - h2.y

        return math.sqrt(dx * dx + dy * dy)

    def detected_callback(self, det_msg):

        if self.latest_gazebo_msg is None:
            return

        gt_msg = self.latest_gazebo_msg

        self.frame_count += 1

        robot_pose = self.get_robot_pose()

        if robot_pose is not None:
            robot_x, robot_y = robot_pose
        else:
            robot_x = float("nan")
            robot_y = float("nan")

        robot_yaw = self.estimate_robot_yaw_from_track()

        if robot_yaw is None:
            robot_yaw = float("nan")

        visible_gts = self.get_visible_gt_humans(gt_msg.humans)
        visible_gt_ids = set(gt.id for gt in visible_gts)

        frame_visible_gt_count = len(visible_gts)
        frame_det_count = len(det_msg.humans)

        self.total_visible_gt_count += frame_visible_gt_count
        self.total_detected_count += frame_det_count

        matched_gt_ids_in_frame = set()
        valid_matches_in_frame = 0

        # GT tracks are updated and logged in gazebo_callback so that they are
        # preserved even when no detection is available in the current frame.
        marker_array = MarkerArray()
        marker_id = 0

        for gt in gt_msg.humans:

            is_visible = gt.id in visible_gt_ids

            if is_visible:
                r, g, b = 0.0, 1.0, 0.0
            else:
                r, g, b = 0.4, 0.4, 0.4

            marker_array.markers.append(
                self.make_sphere_marker(
                    marker_id,
                    gt.x,
                    gt.y,
                    0.05,
                    "world",
                    r,
                    g,
                    b,
                    1.0,
                    0.25
                )
            )
            marker_id += 1

            label = "GT_" + str(gt.id)

            if not is_visible:
                label += "_outFOV"

            marker_array.markers.append(
                self.make_text_marker(
                    marker_id,
                    gt.x,
                    gt.y,
                    0.45,
                    "world",
                    label,
                    r,
                    g,
                    b,
                    1.0
                )
            )
            marker_id += 1

        for det in det_msg.humans:

            self.update_track(self.det_tracks, det.id, det.x, det.y)

            best_gt, best_error = self.stable_match(det, visible_gts)

            marker_array.markers.append(
                self.make_sphere_marker(
                    marker_id,
                    det.x,
                    det.y,
                    0.08,
                    "world",
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.25
                )
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(
                    marker_id,
                    det.x,
                    det.y,
                    0.65,
                    "world",
                    "DET_" + str(det.id),
                    1.0,
                    0.0,
                    0.0,
                    1.0
                )
            )
            marker_id += 1

            if best_gt is None:
                continue

            if best_error > self.match_threshold:
                rospy.logwarn(
                    "Detection id=%d has no valid visible-GT match. nearest error=%.3f",
                    det.id,
                    best_error
                )
                continue

            if best_gt.id in self.gt_to_det_assignment:

                previous_det_id = self.gt_to_det_assignment[best_gt.id]

                if previous_det_id != det.id:
                    self.id_switch_count += 1
                    rospy.logwarn(
                        "ID SWITCH: GT_%d changed from DET_%d to DET_%d",
                        best_gt.id,
                        previous_det_id,
                        det.id
                    )

            self.gt_to_det_assignment[best_gt.id] = det.id

            matched_gt_ids_in_frame.add(best_gt.id)

            valid_matches_in_frame += 1
            self.total_valid_matches += 1

            self.errors.append(best_error)

            dx = det.x - best_gt.x
            dy = det.y - best_gt.y

            self.x_errors.append(dx)
            self.y_errors.append(dy)

            frame_detection_rate = (
                float(len(matched_gt_ids_in_frame)) / float(frame_visible_gt_count)
                if frame_visible_gt_count > 0 else 0.0
            )

            frame_tracking_success_rate = frame_detection_rate

            overall_detection_rate = (
                float(self.total_valid_matches) / float(self.total_visible_gt_count)
                if self.total_visible_gt_count > 0 else 0.0
            )

            overall_tracking_success_rate = overall_detection_rate

            t = det_msg.header.stamp.to_sec()

            self.csv_writer.writerow([
                "detection",
                t,
                det.id,
                best_gt.id,
                det.x,
                det.y,
                best_gt.x,
                best_gt.y,
                dx,
                dy,
                best_error,
                robot_x,
                robot_y,
                robot_yaw,
                int(best_gt.id in visible_gt_ids),
                frame_visible_gt_count,
                frame_detection_rate,
                frame_tracking_success_rate,
                overall_detection_rate,
                overall_tracking_success_rate,
                self.id_switch_count
            ])

            self.csv_file.flush()

            val_msg = HumanValidation()
            val_msg.header.stamp = det_msg.header.stamp
            val_msg.header.frame_id = "world"

            val_msg.det_id = det.id
            val_msg.gt_id = best_gt.id
            val_msg.error = best_error

            val_msg.det_x = det.x
            val_msg.det_y = det.y

            val_msg.gt_x = best_gt.x
            val_msg.gt_y = best_gt.y

            self.validation_pub.publish(val_msg)

            rospy.loginfo_throttle(
                1.0,
                "N=%d | RMSE=%.3f | Xoff=%.3f | Yoff=%.3f | VisibleGT=%d | DetRate=%.2f | TrackSucc=%.2f | IDSW=%d",
                len(self.errors),
                self.rmse(),
                self.mean_x_offset(),
                self.mean_y_offset(),
                frame_visible_gt_count,
                overall_detection_rate,
                overall_tracking_success_rate,
                self.id_switch_count
            )

            marker_array.markers.append(
                self.make_line_marker(
                    marker_id,
                    det.x,
                    det.y,
                    best_gt.x,
                    best_gt.y,
                    "world",
                    1.0,
                    1.0,
                    0.0,
                    1.0
                )
            )
            marker_id += 1

        for gt_id, track in self.gt_tracks.items():

            if len(track) < 2:
                continue

            marker_array.markers.append(
                self.make_path_marker(
                    marker_id,
                    track,
                    "world",
                    "gt_trajectories",
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.035
                )
            )
            marker_id += 1

        for det_id, track in self.det_tracks.items():

            if len(track) < 2:
                continue

            marker_array.markers.append(
                self.make_path_marker(
                    marker_id,
                    track,
                    "world",
                    "det_trajectories",
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.035
                )
            )
            marker_id += 1

        if len(self.robot_track) >= 2:

            marker_array.markers.append(
                self.make_path_marker(
                    marker_id,
                    self.robot_track,
                    "world",
                    "robot_trajectory",
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.07
                )
            )
            marker_id += 1

        if robot_pose is not None:

            marker_array.markers.append(
                self.make_sphere_marker(
                    marker_id,
                    robot_x,
                    robot_y,
                    0.08,
                    "world",
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.35
                )
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(
                    marker_id,
                    robot_x,
                    robot_y,
                    0.55,
                    "world",
                    "ROBOT",
                    0.0,
                    0.0,
                    1.0,
                    1.0
                )
            )
            marker_id += 1

        self.marker_pub.publish(marker_array)

    def stable_match(self, det, gt_humans):

        if len(gt_humans) == 0:
            return None, None

        if det.id in self.match_memory:

            previous_gt_id = self.match_memory[det.id]

            for gt in gt_humans:

                if gt.id == previous_gt_id:

                    err = self.distance(det, gt)

                    if err < self.max_match_distance:
                        return gt, err

        best_gt = None
        best_error = float("inf")

        for gt in gt_humans:

            err = self.distance(det, gt)

            if err < best_error:
                best_error = err
                best_gt = gt

        if best_gt is None:
            return None, None

        if best_error > self.max_match_distance:
            return None, best_error

        self.match_memory[det.id] = best_gt.id

        return best_gt, best_error

    def update_robot_track(self, x, y):

        self.robot_track.append((x, y))

        if len(self.robot_track) > self.max_track_len:
            self.robot_track.pop(0)

    def update_track(self, tracks, human_id, x, y):

        if human_id not in tracks:
            tracks[human_id] = []

        tracks[human_id].append((x, y))

        if len(tracks[human_id]) > self.max_track_len:
            tracks[human_id].pop(0)

    def rmse(self):

        if len(self.errors) == 0:
            return 0.0

        return math.sqrt(sum(e * e for e in self.errors) / len(self.errors))

    def rmse_x(self):

        if len(self.x_errors) == 0:
            return 0.0

        return math.sqrt(sum(e * e for e in self.x_errors) / len(self.x_errors))

    def rmse_y(self):

        if len(self.y_errors) == 0:
            return 0.0

        return math.sqrt(sum(e * e for e in self.y_errors) / len(self.y_errors))

    def mean_error(self):

        if len(self.errors) == 0:
            return 0.0

        return sum(self.errors) / len(self.errors)

    def mean_x_offset(self):

        if len(self.x_errors) == 0:
            return 0.0

        return sum(self.x_errors) / len(self.x_errors)

    def mean_y_offset(self):

        if len(self.y_errors) == 0:
            return 0.0

        return sum(self.y_errors) / len(self.y_errors)

    def shutdown_hook(self):

        if hasattr(self, "csv_file"):
            self.csv_file.flush()
            self.csv_file.close()

        rospy.loginfo("CSV saved: %s", self.csv_path)

    def make_sphere_marker(self, marker_id, x, y, z, frame_id, r, g, b, a, scale):

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = frame_id

        m.ns = "human_points"
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0

        m.scale.x = scale
        m.scale.y = scale
        m.scale.z = scale

        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a

        m.lifetime = rospy.Duration(0.2)

        return m

    def make_text_marker(self, marker_id, x, y, z, frame_id, text, r, g, b, a):

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = frame_id

        m.ns = "human_labels"
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0

        m.scale.z = 0.3

        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a

        m.text = text
        m.lifetime = rospy.Duration(0.2)

        return m

    def make_path_marker(self, marker_id, track, frame_id, ns, r, g, b, a, width):

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = frame_id

        m.ns = ns
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.pose.orientation.w = 1.0

        m.scale.x = width

        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a

        for x, y in track:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.03
            m.points.append(p)

        m.lifetime = rospy.Duration(0.0)

        return m

    def make_line_marker(self, marker_id, x1, y1, x2, y2, frame_id, r, g, b, a):

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = frame_id

        m.ns = "human_error_lines"
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        p1 = Point()
        p1.x = x1
        p1.y = y1
        p1.z = 0.15

        p2 = Point()
        p2.x = x2
        p2.y = y2
        p2.z = 0.15

        m.points.append(p1)
        m.points.append(p2)

        m.scale.x = 0.04

        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a

        m.lifetime = rospy.Duration(0.2)

        return m


if __name__ == "__main__":
    node = HumanValidationNode()
    rospy.spin()