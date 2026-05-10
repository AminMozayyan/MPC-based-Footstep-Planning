#!/usr/bin/env python3

import rospy
import math
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class ScriptedHumanMotionNode:
    def __init__(self):
        rospy.init_node("scripted_human_motion_node")

        self.model_name = rospy.get_param("~model_name", "person_1")

      
        self.ax = rospy.get_param("~ax", 1.0)
        self.ay = rospy.get_param("~ay", -3.0)

        self.bx = rospy.get_param("~bx", 1.0)
        self.by = rospy.get_param("~by", 3.0)

        self.speed = rospy.get_param("~speed", 0.1)
        self.z = rospy.get_param("~z", 0.85)

        
        self.x = self.ax
        self.y = self.ay

        
        self.dir = 1

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

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
            except rospy.ServiceException as e:
                rospy.logwarn_throttle(1.0, "Failed to set model state: %s", str(e))

            self.rate.sleep()


if __name__ == "__main__":
    node = ScriptedHumanMotionNode()
    node.run()