#!/usr/bin/env python3

import math
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D
from social_mpc_planner.msg import PredictedPerson, PredictedPeople


class ScriptedHuman:
    def __init__(self, cfg, default_radius, default_z):
        self.id = int(cfg.get("id", 1))
        self.model_name = str(cfg.get("model_name", "person_%d" % self.id))

        self.ax = float(cfg.get("ax", 0.0))
        self.ay = float(cfg.get("ay", 0.0))
        self.bx = float(cfg.get("bx", 1.0))
        self.by = float(cfg.get("by", 0.0))

        self.speed = float(cfg.get("speed", 0.03))
        self.z = float(cfg.get("z", default_z))
        self.radius = float(cfg.get("radius", default_radius))

        self.x = self.ax
        self.y = self.ay
        self.direction = int(cfg.get("direction", 1))

        self.switch_tolerance = float(cfg.get("switch_tolerance", 0.1))

        # Optional yaw offset for models whose visual front is not aligned with +X.
        # Use values like 1.5708, -1.5708, or 3.14159 if your human mesh faces
        # sideways/backward relative to its velocity direction.
        self.yaw_offset = float(cfg.get("yaw_offset", 0.0))

    def _unit_direction(self):
        dx = self.bx - self.ax
        dy = self.by - self.ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            rospy.logwarn_throttle(
                2.0,
                "Human %s has identical start and goal points. Keeping it static.",
                self.model_name,
            )
            return 0.0, 0.0
        return dx / length, dy / length

    def step(self, dt):
        ux, uy = self._unit_direction()

        vx = self.direction * self.speed * ux
        vy = self.direction * self.speed * uy

        self.x += vx * dt
        self.y += vy * dt

        dist_to_b = math.hypot(self.x - self.bx, self.y - self.by)
        dist_to_a = math.hypot(self.x - self.ax, self.y - self.ay)

        if self.direction == 1 and dist_to_b < self.switch_tolerance:
            self.direction = -1
        elif self.direction == -1 and dist_to_a < self.switch_tolerance:
            self.direction = 1

        return vx, vy

    def _yaw_from_velocity(self, vx, vy):
        # Keep the current path heading when the model is effectively static.
        # This avoids unstable orientation changes caused by tiny numerical velocities.
        if math.hypot(vx, vy) > 1e-6:
            return math.atan2(vy, vx) + self.yaw_offset

        ux, uy = self._unit_direction()
        return math.atan2(self.direction * uy, self.direction * ux) + self.yaw_offset

    def _quaternion_from_yaw(self, yaw):
        # Roll = pitch = 0, yaw around Z axis.
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def build_model_state(self, vx, vy):
        state = ModelState()
        state.model_name = self.model_name
        state.reference_frame = "world"

        state.pose.position.x = self.x
        state.pose.position.y = self.y
        state.pose.position.z = self.z

        yaw = self._yaw_from_velocity(vx, vy)
        qz, qw = self._quaternion_from_yaw(yaw)
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw

        state.twist.linear.x = vx
        state.twist.linear.y = vy
        state.twist.angular.z = 0.0
        return state

    def build_marker(self, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()

        marker.ns = "scripted_humans"
        marker.id = self.id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.radius * 2.0
        marker.scale.y = self.radius * 2.0
        marker.scale.z = 1.0

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        return marker

    def build_prediction(self, dt_pred, horizon):
        person = PredictedPerson()
        person.id = self.id
        person.radius = self.radius

        px = self.x
        py = self.y
        direction_now = self.direction

        ux, uy = self._unit_direction()

        for _ in range(horizon + 1):
            pose = Pose2D()
            pose.x = px
            pose.y = py
            pose.theta = math.atan2(direction_now * uy, direction_now * ux)
            person.trajectory.append(pose)

            px += direction_now * self.speed * ux * dt_pred
            py += direction_now * self.speed * uy * dt_pred

            dist_to_b = math.hypot(px - self.bx, py - self.by)
            dist_to_a = math.hypot(px - self.ax, py - self.ay)

            if direction_now == 1 and dist_to_b < self.switch_tolerance:
                direction_now = -1
            elif direction_now == -1 and dist_to_a < self.switch_tolerance:
                direction_now = 1

        return person


class ScriptedMultiHumanMotionNode:
    def __init__(self):
        rospy.init_node("scripted_multi_human_motion_node")

        self.default_radius = rospy.get_param("~person_radius", 0.25)
        self.default_z = rospy.get_param("~z", 0.0)
        self.marker_frame = rospy.get_param("~marker_frame", "odom")
        self.prediction_frame = rospy.get_param("~prediction_frame", "odom")
        self.dt_pred = rospy.get_param("~prediction_dt", 0.2)
        self.n_pred = rospy.get_param("~prediction_horizon", 150)
        self.rate_hz = rospy.get_param("~rate", 20.0)

        humans_cfg = rospy.get_param("~humans", [])

        # Backward-compatible fallback: if no list is provided, behave like the old one-human node.
        if not humans_cfg:
            humans_cfg = [
                {
                    "id": 1,
                    "model_name": rospy.get_param("~model_name", "person_1"),
                    "ax": rospy.get_param("~ax", 1.5),
                    "ay": rospy.get_param("~ay", 0.0),
                    "bx": rospy.get_param("~bx", 15.0),
                    "by": rospy.get_param("~by", 0.0),
                    "speed": rospy.get_param("~speed", 0.03),
                    "z": rospy.get_param("~z", 0.0),
                    "radius": rospy.get_param("~person_radius", 0.25),
                }
            ]

        self.humans = [ScriptedHuman(cfg, self.default_radius, self.default_z) for cfg in humans_cfg]

        self.pred_pub = rospy.Publisher("/predicted_people", PredictedPeople, queue_size=10)
        self.marker_pub = rospy.Publisher("/human_obstacle_marker", Marker, queue_size=10)

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("scripted multi-human motion started with %d humans", len(self.humans))

    def build_future_reference(self):
        out = PredictedPeople()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.prediction_frame
        out.dt = self.dt_pred
        out.horizon = self.n_pred

        for human in self.humans:
            out.people.append(human.build_prediction(self.dt_pred, self.n_pred))

        return out

    def run(self):
        prev_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - prev_time).to_sec()
            prev_time = now

            try:
                for human in self.humans:
                    vx, vy = human.step(dt)
                    self.set_state(human.build_model_state(vx, vy))
                    self.marker_pub.publish(human.build_marker(self.marker_frame))

                self.pred_pub.publish(self.build_future_reference())

            except rospy.ServiceException as e:
                rospy.logwarn_throttle(1.0, "Failed to set model state: %s", str(e))

            self.rate.sleep()


if __name__ == "__main__":
    node = ScriptedMultiHumanMotionNode()
    node.run()
