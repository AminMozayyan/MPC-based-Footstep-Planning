#!/usr/bin/env python3

import rospy
import yaml

from gazebo_msgs.msg import ModelStates
from social_mpc_planner.msg import CircularObstacle, CircularObstacleArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class GazeboStaticObstacleNode:
    def __init__(self):
        rospy.init_node("gazebo_static_obstacle_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        # If use_yaml=True, obstacles are read from a YAML file and published periodically.
        # If use_yaml=False, obstacles are read from /gazebo/model_states using obstacle_prefix.
        self.use_yaml = rospy.get_param("~use_yaml", False)

        self.obstacle_prefix = rospy.get_param("~obstacle_prefix", "obstacle_")
        self.obstacle_radius = rospy.get_param("~obstacle_radius", 0.2)

        self.yaml_file = rospy.get_param("~yaml_file", "")
        self.publish_rate = rospy.get_param("~publish_rate", 10.0)
        self.frame_id = rospy.get_param("~frame_id", "odom")

        self.marker_z = rospy.get_param("~marker_z", 0.5)
        self.marker_height = rospy.get_param("~marker_height", 1.0)

        self.yaml_obstacles = []

        # -----------------------------
        # Publishers
        # -----------------------------
        self.pub = rospy.Publisher(
            "/static_obstacles",
            CircularObstacleArray,
            queue_size=10
        )

        self.marker_pub = rospy.Publisher(
            "/obstacle_markers",
            MarkerArray,
            queue_size=10
        )

        self.path_pub = rospy.Publisher("/mpc_path", Path, queue_size=10)

        # -----------------------------
        # Input mode
        # -----------------------------
        if self.use_yaml:
            self.load_yaml_obstacles()

            period = 1.0 / max(float(self.publish_rate), 1e-6)
            self.timer = rospy.Timer(
                rospy.Duration(period),
                self.publish_yaml_obstacles
            )

            rospy.loginfo(
                "gazebo_static_obstacle_node started in YAML mode. Loaded %d obstacles from: %s",
                len(self.yaml_obstacles),
                self.yaml_file
            )
        else:
            self.sub = rospy.Subscriber(
                "/gazebo/model_states",
                ModelStates,
                self.model_states_callback
            )

            rospy.loginfo(
                "gazebo_static_obstacle_node started in Gazebo model_states mode. prefix='%s', radius=%.3f",
                self.obstacle_prefix,
                self.obstacle_radius
            )

    # -------------------------------------------------------------------------
    # YAML mode
    # -------------------------------------------------------------------------
    def load_yaml_obstacles(self):
        if not self.yaml_file:
            rospy.logerr("Parameter ~yaml_file is empty, but ~use_yaml is True.")
            return

        try:
            with open(self.yaml_file, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            rospy.logerr("Failed to read obstacle YAML file '%s': %s", self.yaml_file, str(e))
            return

        if data is None:
            rospy.logerr("Obstacle YAML file '%s' is empty.", self.yaml_file)
            return

        # Optional frame_id inside YAML overrides the default only if ~frame_id was not set explicitly.
        # Your generated YAML currently uses: frame_id: odom
        if "frame_id" in data and not rospy.has_param("~frame_id"):
            self.frame_id = str(data["frame_id"])

        raw_obstacles = data.get("static_obstacles", [])
        valid_obstacles = []

        for idx, item in enumerate(raw_obstacles):
            try:
                x = float(item["x"])
                y = float(item["y"])

                # Supports both keys:
                #   r: 0.35
                #   radius: 0.35
                if "r" in item:
                    radius = float(item["r"])
                else:
                    radius = float(item["radius"])

                if radius <= 0.0:
                    rospy.logwarn(
                        "Skipping obstacle %d from YAML because radius <= 0: %s",
                        idx,
                        str(item)
                    )
                    continue

                valid_obstacles.append({
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "source": item.get("source", "yaml_obstacle_%d" % idx)
                })

            except Exception as e:
                rospy.logwarn(
                    "Skipping invalid obstacle %d in YAML: %s | error: %s",
                    idx,
                    str(item),
                    str(e)
                )

        self.yaml_obstacles = valid_obstacles

    def publish_yaml_obstacles(self, event=None):
        out = CircularObstacleArray()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.frame_id

        for item in self.yaml_obstacles:
            obs = CircularObstacle()
            obs.x = item["x"]
            obs.y = item["y"]
            obs.radius = item["radius"]
            out.obstacles.append(obs)

        self.publish_obstacles_and_markers(out)

        rospy.loginfo_throttle(
            5.0,
            "Published %d YAML static obstacles on /static_obstacles",
            len(out.obstacles)
        )

    # -------------------------------------------------------------------------
    # Gazebo model_states mode, kept for old tests with obstacle_ models
    # -------------------------------------------------------------------------
    def model_states_callback(self, msg):
        out = CircularObstacleArray()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.frame_id

        for i, name in enumerate(msg.name):
            if name.startswith(self.obstacle_prefix):
                obs = CircularObstacle()
                obs.x = msg.pose[i].position.x
                obs.y = msg.pose[i].position.y
                obs.radius = self.obstacle_radius
                out.obstacles.append(obs)

        self.publish_obstacles_and_markers(out)

        rospy.loginfo_throttle(
            5.0,
            "Published %d Gazebo static obstacles on /static_obstacles",
            len(out.obstacles)
        )

    # -------------------------------------------------------------------------
    # Common publishing utilities
    # -------------------------------------------------------------------------
    def publish_obstacles_and_markers(self, out):
        self.pub.publish(out)
        self.publish_markers(out)

    def publish_markers(self, out):
        marker_array = MarkerArray()
        now = rospy.Time.now()

        for i, obs in enumerate(out.obstacles):
            marker = Marker()

            marker.header.frame_id = out.header.frame_id
            marker.header.stamp = now

            marker.ns = "obstacles"
            marker.id = i

            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = obs.x
            marker.pose.position.y = obs.y
            marker.pose.position.z = self.marker_z
            marker.pose.orientation.w = 1.0

            # IMPORTANT:
            # Marker diameter is exactly 2 * radius from YAML / Gazebo mode.
            marker.scale.x = obs.radius * 2.0
            marker.scale.y = obs.radius * 2.0
            marker.scale.z = self.marker_height

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def publish_mpc_path(self, X_sol):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

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
