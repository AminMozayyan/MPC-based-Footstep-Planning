#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import math

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from ultralytics import YOLO
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
from social_mpc_planner.msg import Human, HumanArray

class HumanTrackerDepthNode:

    def __init__(self):

        rospy.init_node("human_tracker_depth_node")

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        self.rgb_frame = None
        self.depth_frame = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.camera_frame = "camera_depth_optical_frame"   
        self.base_frame = "base_link"

        self.robot_name = "surenaV"

        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.last_valid_detection = {}
        self.max_jump_distance = 0.7  

        self.smooth_detection = {}
        self.alpha = 0.5

        self.human_model_keywords = ["person", "human", "actor"]

        self.detected_human_pub = rospy.Publisher(
            "/detected_humans_world",
            HumanArray,
            queue_size=10
        )

        self.gazebo_human_pub = rospy.Publisher(
            "/gazebo_humans_world",
            HumanArray,
            queue_size=10
        )

        rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.gazebo_callback,
            queue_size=1
        )

        # -------------------------------
        # Camera intrinsics (from camera_info)
        # -------------------------------
        self.fx = None
        self.fy = None
        self.cx0 = None
        self.cy0 = None

        # -------------------------------
        # Subscribers
        # -------------------------------
        rospy.Subscriber("/camera/image_raw", Image, self.rgb_callback, queue_size=1)
        rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)

        rospy.Subscriber("/camera/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)

        # -------------------------------
        # Publisher
        # -------------------------------

        rospy.loginfo("Human BEV Tracker Started")


            
    def filter_detection(self, track_id, x_world, y_world):

        if track_id not in self.last_valid_detection:
            self.last_valid_detection[track_id] = (x_world, y_world)
            return x_world, y_world

        last_x, last_y = self.last_valid_detection[track_id]

        dx = x_world - last_x
        dy = y_world - last_y

        jump = math.sqrt(dx * dx + dy * dy)

        if jump > self.max_jump_distance:
            rospy.logwarn(
                "Outlier detected for id=%d | jump=%.3f m -> using previous value",
                track_id,
                jump
            )
            return last_x, last_y

        self.last_valid_detection[track_id] = (x_world, y_world)
        return x_world, y_world




    # =====================================================
    # camera info callback
    # =====================================================
    def camera_info_callback(self, msg):

        K = msg.K

        self.fx = K[0]
        self.fy = K[4]
        self.cx0 = K[2]
        self.cy0 = K[5]


    # =====================================================
    # RGB
    # =====================================================
    def rgb_callback(self, msg):
        self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")


    # =====================================================
    # Depth
    # =====================================================
    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, "passthrough")


    # =====================================================
    # depth getter (robust)
    # =====================================================
    def get_depth(self, cx, cy):

        if self.depth_frame is None:
            return -1

        h, w = self.depth_frame.shape[:2]

        values = []

        for dx in range(-2, 3):
            for dy in range(-2, 3):

                x = cx + dx
                y = cy + dy

                if 0 <= x < w and 0 <= y < h:

                    d = self.depth_frame[y, x]

                    if np.isfinite(d) and d > 0:
                        values.append(float(d))

        if len(values) == 0:
            return -1

        depth = np.median(values)

        if self.depth_frame.dtype == np.uint16:
            depth /= 1000.0

        return float(depth)


    # =====================================================
    # pixel -> 3D camera frame
    # =====================================================
    def pixel_to_3d(self, u, v, z):

        if self.fx is None:
            return None

        x = (u - self.cx0) * z / self.fx
        y = (v - self.cy0) * z / self.fy

        return x, y, z
    
    def camera_to_base(self, X, Y, Z):

        p_cam = PointStamped()
        p_cam.header.stamp = rospy.Time(0)
        p_cam.header.frame_id = self.camera_frame

        p_cam.point.x = X
        p_cam.point.y = Y
        p_cam.point.z = Z

        try:
            p_base = self.tf_buffer.transform(
                p_cam,
                self.base_frame,
                rospy.Duration(0.1)
            )

            return p_base.point.x, p_base.point.y, p_base.point.z

        except Exception as e:
            rospy.logwarn("camera_to_base TF failed: %s", str(e))
            return None
        
    
    def smooth_position(self, track_id, x, y):

            if track_id not in self.smooth_detection:
                self.smooth_detection[track_id] = (x, y)
                return x, y

            last_x, last_y = self.smooth_detection[track_id]

            x_s = self.alpha * x + (1.0 - self.alpha) * last_x
            y_s = self.alpha * y + (1.0 - self.alpha) * last_y

            self.smooth_detection[track_id] = (x_s, y_s)

            return x_s, y_s
        
    
    def gazebo_callback(self, msg):

        # -----------------------------
        # Robot pose from Gazebo
        # -----------------------------
        if self.robot_name in msg.name:

            idx = msg.name.index(self.robot_name)
            pose = msg.pose[idx]

            self.robot_x = pose.position.x
            self.robot_y = pose.position.y

            q = pose.orientation
            quat = [q.x, q.y, q.z, q.w]

            _, _, self.robot_yaw = euler_from_quaternion(quat)

        # -----------------------------
        # Humans ground truth from Gazebo
        # -----------------------------
        gazebo_msg = HumanArray()
        gazebo_msg.header.stamp = rospy.Time.now()
        gazebo_msg.header.frame_id = "world"

        for i, name in enumerate(msg.name):

            is_human = any(
                keyword in name.lower()
                for keyword in self.human_model_keywords
            )

            if not is_human:
                continue

            pose = msg.pose[i]
            

            human = Human()
            human.id = int(i)
            human.x = float(pose.position.x)
            human.y = float(pose.position.y)

            gazebo_msg.humans.append(human)

        self.gazebo_human_pub.publish(gazebo_msg)


    def base_to_world(self, x_base, y_base):

        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            return None

        c = math.cos(self.robot_yaw)
        s = math.sin(self.robot_yaw)

        x_world = self.robot_x + c * x_base - s * y_base
        y_world = self.robot_y + s * x_base + c * y_base

        return x_world, y_world

    # =====================================================
    # 3D camera -> 2D ground map
    # =====================================================
    def to_map(self, x, z):

    # ground plane assumption
        x_map = x
        y_map = z

        return x_map, y_map


    # =====================================================
    # main loop
    # =====================================================
    def run(self):

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            if self.rgb_frame is None or self.depth_frame is None:
                rate.sleep()
                continue

            frame = self.rgb_frame.copy()

            results = self.model.track(
                frame,
                persist=True,
                classes=[0],
                verbose=False
            )[0]

            detected_msg = HumanArray()
            detected_msg.header.stamp = rospy.Time.now()
            detected_msg.header.frame_id = "world"

            if results.boxes is not None:

                for box in results.boxes:

                    conf = float(box.conf[0])
                    if conf < 0.5:
                        continue

                    track_id = int(box.id[0]) if box.id is not None else -1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    # depth
                    z = self.get_depth(cx, cy)

                    if z <= 0:
                        continue

                    # -----------------------------
                    # BEV / 3D conversion
                    # -----------------------------
                    xyz = self.pixel_to_3d(cx, cy, z)

                    if xyz is None:
                        continue

                    X, Y, Z = xyz

                    base_point = self.camera_to_base(X, Y, Z)

                    if base_point is None:
                        continue

                    x_base, y_base, z_base = base_point

                    world_point = self.base_to_world(x_base, y_base)

                    if world_point is None:
                        continue

                    x_world, y_world = world_point

                    X_world, Y_world = self.filter_detection(track_id,x_world,y_world)
                    x_world, y_world = self.smooth_position(track_id, x_world, y_world)

                    human = Human()
                    human.id = int(track_id)
                    human.x = float(x_world)
                    human.y = float(y_world)

                    detected_msg.humans.append(human)

                    # draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    label = f"human_{track_id}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            self.detected_human_pub.publish(detected_msg)

            cv2.imshow("Human BEV Tracker", frame)
            cv2.waitKey(1)

            rate.sleep()


if __name__ == "__main__":
    node = HumanTrackerDepthNode()
    node.run()