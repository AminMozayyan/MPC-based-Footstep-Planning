#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import csv
import os

from social_mpc_planner.msg import HumanArray, HumanValidation
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class HumanValidationNode:

    def __init__(self):

        rospy.init_node("human_validation_node")
        rospy.on_shutdown(self.shutdown_hook)

        self.latest_gazebo_msg = None

        self.match_threshold = rospy.get_param("~match_threshold", 1.0)

        self.gt_tracks = {}
        self.det_tracks = {}

        self.max_track_len = 200

        self.match_memory = {}
        self.max_match_distance = 1.0

        self.errors = []
        self.x_errors = []
        self.y_errors = []

        # CSV path
        self.csv_path = "/home/mozayyan/Surena_ws/human_validation.csv"

        # open CSV
        self.csv_file = open(self.csv_path, "w")

        # CSV writer
        self.csv_writer = csv.writer(self.csv_file)

        # header
        self.csv_writer.writerow([
            "time",
            "det_id",
            "gt_id",
            "det_x",
            "det_y",
            "gt_x",
            "gt_y",
            "dx",
            "dy",
            "distance_error"
        ])

        self.csv_file.flush()

        rospy.loginfo("Saving CSV to: %s", self.csv_path)

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

    def distance(self, h1, h2):
        dx = h1.x - h2.x
        dy = h1.y - h2.y
        return math.sqrt(dx * dx + dy * dy)

    def detected_callback(self, det_msg):

        if self.latest_gazebo_msg is None:
            return

        gt_msg = self.latest_gazebo_msg

        for gt in gt_msg.humans:
            self.update_track(
                self.gt_tracks,
                gt.id,
                gt.x,
                gt.y
            )

        marker_array = MarkerArray()
        marker_id = 0

        # ground truth markers
        for gt in gt_msg.humans:
            marker_array.markers.append(
                self.make_sphere_marker(
                    marker_id,
                    gt.x,
                    gt.y,
                    0.05,
                    "world",
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.25
                )
            )
            marker_id += 1

            marker_array.markers.append(
                self.make_text_marker(
                    marker_id,
                    gt.x,
                    gt.y,
                    0.45,
                    "world",
                    "GT_" + str(gt.id),
                    0.0,
                    1.0,
                    0.0,
                    1.0
                )
            )
            marker_id += 1

        # detected markers + matching
        for det in det_msg.humans:

            self.update_track(
                self.det_tracks,
                det.id,
                det.x,
                det.y
            )
                        
            best_gt, best_error = self.stable_match(det, gt_msg.humans)

            # detected marker
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
                    "Detection id=%d has no valid GT match. nearest error=%.3f",
                    det.id,
                    best_error
                )
                continue
            

            self.errors.append(best_error)
            dx = det.x - best_gt.x
            dy = det.y - best_gt.y

            # self.x_errors.append(abs(dx))
            # self.y_errors.append(abs(dy))

            self.x_errors.append(dx)
            self.y_errors.append(dy)

            t = det_msg.header.stamp.to_sec()

            self.csv_writer.writerow([
                t,
                det.id,
                best_gt.id,
                det.x,
                det.y,
                best_gt.x,
                best_gt.y,
                dx,
                dy,
                best_error
            ])

            self.csv_file.flush()
                        
            # publish validation result
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
                #"N=%d | RMSE=%.3f | X_RMSE=%.3f | Y_RMSE=%.3f | X_offset=%.3f | Y_offset=%.3f",
                "N=%d | RMSE=%.3f | X_offset=%.3f | Y_offset=%.3f",
                len(self.errors),
                self.rmse(),
                # self.rmse_x(),
                # self.rmse_y(),
                self.mean_x_offset(),
                self.mean_y_offset()
            )

            # line between detected and ground truth
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


        # GT trajectories
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
                    1.0
                )
            )
            marker_id += 1


        # DET trajectories
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
                    1.0
                )
            )
            marker_id += 1

        self.marker_pub.publish(marker_array)

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

        return math.sqrt(
            sum(e * e for e in self.x_errors) / len(self.x_errors)
        )
    
    def rmse_y(self):

        if len(self.y_errors) == 0:
            return 0.0

        return math.sqrt(
            sum(e * e for e in self.y_errors) / len(self.y_errors)
        )

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


    
    def make_path_marker(self, marker_id, track, frame_id, ns, r, g, b, a):

        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = frame_id

        m.ns = ns
        m.id = marker_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.pose.orientation.w = 1.0

        m.scale.x = 0.035

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
    

    def stable_match(self, det, gt_humans):

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


    def shutdown_hook(self):

        self.csv_file.flush()
        self.csv_file.close()

        rospy.loginfo("CSV saved.")


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