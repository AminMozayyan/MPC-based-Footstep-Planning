#!/usr/bin/env python3

import rospy
from collections import defaultdict, deque

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from social_mpc_planner.msg import (
    HumanArray,
    SocialSTGCNNPredictionArray
)


class SocialSTGCNNVisualizer:

    def __init__(self):
        rospy.init_node("social_stgcnn_visualizer_node")

        self.obs_len = rospy.get_param("~obs_len", 8)
        self.pred_topic = rospy.get_param("~pred_topic", "/social_stgcnn/predictions")
        self.human_topic = rospy.get_param("~human_topic", "/gazebo_humans_world")
        self.marker_topic = rospy.get_param("~marker_topic", "/social_stgcnn/markers")

        self.fixed_frame = rospy.get_param("~fixed_frame", "")
        self.line_width = rospy.get_param("~line_width", 0.10)
        self.obs_point_radius = rospy.get_param("~obs_point_radius", 0.11)
        self.current_point_radius = rospy.get_param("~current_point_radius", 0.22)

        self.show_pred_points = rospy.get_param("~show_pred_points", False)
        self.pred_point_radius = rospy.get_param("~pred_point_radius", 0.05)

        self.history = defaultdict(lambda: deque(maxlen=self.obs_len))

        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        rospy.Subscriber(self.human_topic, HumanArray, self.human_callback, queue_size=1)
        rospy.Subscriber(self.pred_topic, SocialSTGCNNPredictionArray, self.prediction_callback, queue_size=1)

        rospy.loginfo("Social-STGCNN visualizer started.")
        rospy.loginfo(f"Subscribed human topic: {self.human_topic}")
        rospy.loginfo(f"Subscribed prediction topic: {self.pred_topic}")
        rospy.loginfo(f"Publishing markers: {self.marker_topic}")

    def human_callback(self, msg):
        for human in msg.humans:
            self.history[int(human.id)].append((float(human.x), float(human.y)))

    def get_frame_id(self, msg_frame):
        if self.fixed_frame:
            return self.fixed_frame
        if msg_frame:
            return msg_frame
        return "world"

    def make_delete_all_marker(self, frame_id):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        marker.action = Marker.DELETEALL
        return marker

    def make_line_marker(self, frame_id, ns, marker_id, points, color, width, z):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        for x, y in points:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = z
            marker.points.append(p)

        return marker

    def make_sphere_marker(self, frame_id, ns, marker_id, x, y, color, radius, z):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0

        marker.scale.x = radius
        marker.scale.y = radius
        marker.scale.z = radius

        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        return marker

    def make_text_marker(self, frame_id, ns, marker_id, x, y, text):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.85
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.35

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = text
        return marker

    def prediction_callback(self, msg):
        frame_id = self.get_frame_id(msg.header.frame_id)

        markers = MarkerArray()
        markers.markers.append(self.make_delete_all_marker(frame_id))

        marker_id = 0

        for person in msg.people:
            human_id = int(person.id)

            observed_points = list(self.history[human_id])
            predicted_points = [(pose.x, pose.y) for pose in person.trajectory]

            if len(observed_points) == 0 and len(predicted_points) == 0:
                continue

            current_point = None
            if len(observed_points) > 0:
                current_point = observed_points[-1]

            # 1) Observed trajectory: 8 previous samples, blue
            if len(observed_points) >= 2:
                markers.markers.append(
                    self.make_line_marker(
                        frame_id=frame_id,
                        ns=f"observed_line_{human_id}",
                        marker_id=marker_id,
                        points=observed_points,
                        color=(0.0, 0.2, 1.0, 1.0),
                        width=self.line_width,
                        z=0.18
                    )
                )
                marker_id += 1

            # Observed points: blue spheres
            for t, (x, y) in enumerate(observed_points):
                markers.markers.append(
                    self.make_sphere_marker(
                        frame_id=frame_id,
                        ns=f"observed_points_{human_id}",
                        marker_id=marker_id,
                        x=x,
                        y=y,
                        color=(0.0, 0.55, 1.0, 1.0),
                        radius=self.obs_point_radius,
                        z=0.24
                    )
                )
                marker_id += 1

            # 2) Current human position: green
            if current_point is not None:
                cx, cy = current_point
                markers.markers.append(
                    self.make_sphere_marker(
                        frame_id=frame_id,
                        ns=f"current_{human_id}",
                        marker_id=marker_id,
                        x=cx,
                        y=cy,
                        color=(0.0, 1.0, 0.0, 1.0),
                        radius=self.current_point_radius,
                        z=0.32
                    )
                )
                marker_id += 1

            # 3) Predicted trajectory: start from current position then future points, red
            if len(predicted_points) >= 1:
                if current_point is not None:
                    predicted_line = [current_point] + predicted_points
                else:
                    predicted_line = predicted_points

                if len(predicted_line) >= 2:
                    markers.markers.append(
                        self.make_line_marker(
                            frame_id=frame_id,
                            ns=f"predicted_line_{human_id}",
                            marker_id=marker_id,
                            points=predicted_line,
                            color=(1.0, 0.0, 0.0, 1.0),
                            width=self.line_width,
                            z=0.38
                        )
                    )
                    marker_id += 1

            # Optional predicted points, disabled by default
            if self.show_pred_points:
                for pose in person.trajectory:
                    markers.markers.append(
                        self.make_sphere_marker(
                            frame_id=frame_id,
                            ns=f"predicted_points_{human_id}",
                            marker_id=marker_id,
                            x=pose.x,
                            y=pose.y,
                            color=(1.0, 0.45, 0.0, 1.0),
                            radius=self.pred_point_radius,
                            z=0.44
                        )
                    )
                    marker_id += 1

            # ID label
            label_x, label_y = current_point if current_point is not None else predicted_points[0]
            label = f"id={human_id}"
            if person.is_stale:
                label += f" stale {person.age:.2f}s"

            markers.markers.append(
                self.make_text_marker(
                    frame_id=frame_id,
                    ns=f"text_{human_id}",
                    marker_id=marker_id,
                    x=label_x,
                    y=label_y,
                    text=label
                )
            )
            marker_id += 1

        self.marker_pub.publish(markers)

        rospy.loginfo_throttle(
            1.0,
            f"Visualized people={len(msg.people)}, markers={len(markers.markers)}"
        )


if __name__ == "__main__":
    node = SocialSTGCNNVisualizer()
    rospy.spin()