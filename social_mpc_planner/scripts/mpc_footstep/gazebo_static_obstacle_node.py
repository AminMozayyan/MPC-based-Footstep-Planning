#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from social_mpc_planner.msg import CircularObstacle, CircularObstacleArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GazeboStaticObstacleNode:
    def __init__(self):
        rospy.init_node("gazebo_static_obstacle_node")

        self.obstacle_prefix = rospy.get_param("~obstacle_prefix", "obstacle_")
        self.obstacle_radius = rospy.get_param("~obstacle_radius", 0.15)

        self.pub = rospy.Publisher(
            "/static_obstacles",
            CircularObstacleArray,
            queue_size=10
        )

        self.sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.model_states_callback
        )

        self.marker_pub = rospy.Publisher(
            "/obstacle_markers",
            MarkerArray,
            queue_size=10
        )

        self.path_pub = rospy.Publisher("/mpc_path", Path, queue_size=10)

        rospy.loginfo("gazebo_static_obstacle_node started.")

    def model_states_callback(self, msg):
        out = CircularObstacleArray()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "odom"

        for i, name in enumerate(msg.name):
            if name.startswith(self.obstacle_prefix):
                obs = CircularObstacle()
                obs.x = msg.pose[i].position.x
                obs.y = msg.pose[i].position.y
                obs.radius = self.obstacle_radius
                out.obstacles.append(obs)

        self.pub.publish(out)

        # rospy.loginfo_throttle(
        #     2.0,
        #     "Published %d static obstacles",
        #     len(out.obstacles)
        # )

        marker_array = MarkerArray()

        for i, obs in enumerate(out.obstacles):
            marker = Marker()

            marker.header.frame_id = "odom"
            marker.header.stamp = rospy.Time.now()

            marker.ns = "obstacles"
            marker.id = i

            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = obs.x
            marker.pose.position.y = obs.y
            marker.pose.position.z = 0.5

            marker.pose.orientation.w = 1.0

            marker.scale.x = obs.radius * 2
            marker.scale.y = obs.radius * 2
            marker.scale.z = 1.0

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
    
    def publish_mpc_path(self, X_sol):
            path = Path()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = "odom"

            for k in range(self.N + 1):
                pose = PoseStamped()
                pose.header = path.header

                pose.pose.position.x = float(X_sol[0, k])
                pose.pose.position.y = float(X_sol[1, k])
                pose.pose.position.z = 0.02
                pose.pose.orientation.w = 1.0

                path.poses.append(pose)

            self.path_pub.publish(path)


if __name__ == "__main__":
    node = GazeboStaticObstacleNode()
    rospy.spin()