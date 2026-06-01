#!/usr/bin/env python3

import rospy
import math
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D
from social_mpc_planner.msg import PredictedPerson, PredictedPeople


class ScriptedHumanMotionNode:
    def __init__(self):
        rospy.init_node("scripted_human_motion_node")

        self.model_name = rospy.get_param("~model_name", "person_1")

      
        self.ax = rospy.get_param("~ax", 2.0)
        self.ay = rospy.get_param("~ay", 3.0)

        self.bx = rospy.get_param("~bx", 2.0)
        self.by = rospy.get_param("~by", -3.0)

        self.speed = rospy.get_param("~speed", 0.05)
        self.z = rospy.get_param("~z", 0.)

        
        self.x = self.ax
        self.y = self.ay

        
        self.dir = 1

        self.dt_pred = rospy.get_param("~prediction_dt", 0.2)
        self.N_pred = rospy.get_param("~prediction_horizon", 50)

        self.pred_pub = rospy.Publisher(
            "/predicted_people",
            PredictedPeople,
            queue_size=10
        )

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.marker_pub = rospy.Publisher(
            "/human_obstacle_marker",
            Marker,
            queue_size=10
        )

        self.marker_frame = rospy.get_param("~marker_frame", "odom")
        self.person_radius = rospy.get_param("~person_radius", 0.25)

        self.rate = rospy.Rate(20.0)
        rospy.loginfo("scripted human motion (A<->B) started")

    def step(self, dt):
        
        dx = self.bx - self.ax
        dy = self.by - self.ay
        length = math.sqrt(dx*dx + dy*dy)

        ux = dx / length
        uy = dy / length

        
        vx = self.dir * self.speed * ux
        vy = self.dir * self.speed * uy

        self.x += vx * dt
        self.y += vy * dt

        
        dist_to_B = math.hypot(self.x - self.bx, self.y - self.by)
        dist_to_A = math.hypot(self.x - self.ax, self.y - self.ay)

        
        if self.dir == 1 and dist_to_B < 0.1:
            self.dir = -1
        elif self.dir == -1 and dist_to_A < 0.1:
            self.dir = 1

        return vx, vy

    def run(self):
        prev_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - prev_time).to_sec()
            prev_time = now

            vx, vy = self.step(dt)

            state = ModelState()
            state.model_name = self.model_name
            state.reference_frame = "world"

            state.pose.position.x = self.x
            state.pose.position.y = self.y
            state.pose.position.z = self.z
            state.pose.orientation.w = 1.0

            state.twist.linear.x = vx
            state.twist.linear.y = vy

            try:
                self.set_state(state)
                self.publish_marker()
                future_ref = self.build_future_reference()
                self.pred_pub.publish(future_ref)
            except rospy.ServiceException as e:
                rospy.logwarn_throttle(1.0, "Failed to set model state: %s", str(e))

            self.rate.sleep()

    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = self.marker_frame
        marker.header.stamp = rospy.Time.now()

        marker.ns = "scripted_human"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.person_radius * 2.0
        marker.scale.y = self.person_radius * 2.0
        marker.scale.z = 1.0

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.8

        self.marker_pub.publish(marker)

    def build_future_reference(self):
        out = PredictedPeople()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "odom"
        out.dt = self.dt_pred
        out.horizon = self.N_pred

        person = PredictedPerson()
        person.id = 1
        person.radius = self.person_radius

        px = self.x
        py = self.y
        dir_now = self.dir

        dx = self.bx - self.ax
        dy = self.by - self.ay
        length = math.hypot(dx, dy)

        ux = dx / length
        uy = dy / length

        for k in range(self.N_pred + 1):
            pose = Pose2D()
            pose.x = px
            pose.y = py
            pose.theta = math.atan2(dir_now * uy, dir_now * ux)
            person.trajectory.append(pose)

            px += dir_now * self.speed * ux * self.dt_pred
            py += dir_now * self.speed * uy * self.dt_pred

            dist_to_B = math.hypot(px - self.bx, py - self.by)
            dist_to_A = math.hypot(px - self.ax, py - self.ay)

            if dir_now == 1 and dist_to_B < 0.1:
                dir_now = -1
            elif dir_now == -1 and dist_to_A < 0.1:
                dir_now = 1

        out.people.append(person)
        return out
        

if __name__ == "__main__":
    node = ScriptedHumanMotionNode()
    node.run()