#!/usr/bin/env python3

import rospy
import math
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose2D
from social_mpc_planner.msg import PredictedPerson, PredictedPeople
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose2D, Point


class GazeboHumanPredictionNode:
    def __init__(self):
        rospy.init_node("gazebo_human_prediction_node")

        self.person_prefix = rospy.get_param("~person_prefix", "person_")
        self.person_radius = rospy.get_param("~person_radius", 0.25)

        self.dt = rospy.get_param("~prediction_dt", 0.2)
        self.N = rospy.get_param("~prediction_horizon", 30*5)

        self.prev = {}
        self.prev_time = None

        self.pub = rospy.Publisher(
            "/predicted_people",
            PredictedPeople,
            queue_size=10
        )

        self.sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback
        )

        self.marker_pub = rospy.Publisher(
            "/human_prediction_markers",
            MarkerArray,
            queue_size=150
        )

        rospy.loginfo("gazebo_human_prediction_node started.")

    def model_states_callback(self, msg):
        now = rospy.Time.now()

        out = PredictedPeople()
        out.header.stamp = now
        out.header.frame_id = "odom"
        out.dt = self.dt
        out.horizon = self.N

        for i, name in enumerate(msg.name):
            if not name.startswith(self.person_prefix):
                continue

            x = msg.pose[i].position.x
            y = msg.pose[i].position.y

            vx = 0.0
            vy = 0.0

            if name in self.prev and self.prev_time is not None:
                px, py = self.prev[name]
                dtt = (now - self.prev_time).to_sec()

                if dtt > 1e-4:
                    vx = (x - px) / dtt
                    vy = (y - py) / dtt

            self.prev[name] = (x, y)

            person = PredictedPerson()
            person.id = self.extract_id(name)
            person.radius = self.person_radius

            for k in range(self.N + 1):
                pose = Pose2D()
                pose.x = x + vx * k * self.dt
                pose.y = y + vy * k * self.dt
                pose.theta = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-4 else 0.0
                person.trajectory.append(pose)

            out.people.append(person)

        self.prev_time = now
        self.pub.publish(out)
        self.publish_human_markers(out)

        # rospy.loginfo_throttle(
        #     2.0,
        #     "Published predictions for %d people",
        #     len(out.people)
        # )

    def extract_id(self, name):
        try:
            return int(name.split("_")[-1])
        except Exception:
            return 0

    def publish_human_markers(self, predicted_people):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        now = rospy.Time.now()
        marker_id = 0

        for person in predicted_people.people:
            if len(person.trajectory) == 0:
                continue

            # current human position
            p0 = person.trajectory[0]

            body_marker = Marker()
            body_marker.header.frame_id = "odom"
            body_marker.header.stamp = now
            body_marker.ns = "humans"
            body_marker.id = marker_id
            marker_id += 1

            body_marker.type = Marker.CYLINDER
            body_marker.action = Marker.ADD

            body_marker.pose.position.x = p0.x
            body_marker.pose.position.y = p0.y
            body_marker.pose.position.z = 0.5
            body_marker.pose.orientation.w = 1.0

            body_marker.scale.x = person.radius * 2.0
            body_marker.scale.y = person.radius * 2.0
            body_marker.scale.z = 1.0

            body_marker.color.r = 0.0
            body_marker.color.g = 1.0
            body_marker.color.b = 1.0
            body_marker.color.a = 0.8

            marker_array.markers.append(body_marker)

            # predicted trajectory line
            line_marker = Marker()
            line_marker.header.frame_id = "odom"
            line_marker.header.stamp = now
            line_marker.ns = "human_predictions"
            line_marker.id = marker_id
            marker_id += 1

            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD

            line_marker.scale.x = 0.03

            line_marker.color.r = 1.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 1.0

            for pose in person.trajectory:
                pt = Point()
                pt.x = pose.x
                pt.y = pose.y
                pt.z = 0.1
                line_marker.points.append(pt)

            marker_array.markers.append(line_marker)

            # predicted future points
            for k, pose in enumerate(person.trajectory):
                point_marker = Marker()
                point_marker.header.frame_id = "odom"
                point_marker.header.stamp = now
                point_marker.ns = "human_prediction_points"
                point_marker.id = marker_id
                marker_id += 1

                point_marker.type = Marker.SPHERE
                point_marker.action = Marker.ADD

                point_marker.pose.position.x = pose.x
                point_marker.pose.position.y = pose.y
                point_marker.pose.position.z = 0.08
                point_marker.pose.orientation.w = 1.0

                point_marker.scale.x = 0.08
                point_marker.scale.y = 0.08
                point_marker.scale.z = 0.08

                point_marker.color.r = 1.0
                point_marker.color.g = 0.5
                point_marker.color.b = 0.0
                point_marker.color.a = 0.8

                marker_array.markers.append(point_marker)

        self.marker_pub.publish(marker_array)


if __name__ == "__main__":
    node = GazeboHumanPredictionNode()
    rospy.spin()