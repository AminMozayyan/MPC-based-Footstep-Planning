#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped, Quaternion


class GazeboOdomPublisher:
    def __init__(self):
        self.robot_name = rospy.get_param("~robot_name", "surenaV")

        self.sub = rospy.Subscriber(
            "/gazebo/model_states",
            ModelStates,
            self.callback,
            queue_size=1
        )

        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)

        self.br = tf2_ros.TransformBroadcaster()

        self.last_stamp = rospy.Time(0)

    def callback(self, msg):
        if self.robot_name not in msg.name:
            return

        index = msg.name.index(self.robot_name)

        pose = msg.pose[index]
        twist = msg.twist[index]

        # گرفتن yaw
        q = pose.orientation
        (_, _, yaw) = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        q_yaw = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)

        stamp = rospy.Time.now()
        if self.last_stamp != rospy.Time(0) and stamp <= self.last_stamp:
            stamp = self.last_stamp + rospy.Duration(1e-4)

        self.last_stamp = stamp

        # -------- TF: odom -> base_footprint --------
        t1 = TransformStamped()
        t1.header.stamp = stamp
        t1.header.frame_id = "odom"
        t1.child_frame_id = "base_footprint"

        t1.transform.translation.x = pose.position.x
        t1.transform.translation.y = pose.position.y
        t1.transform.translation.z = 0.0

        t1.transform.rotation = Quaternion(*q_yaw)

        self.br.sendTransform(t1)

        # -------- TF: base_footprint -> base_link --------
        t2 = TransformStamped()
        t2.header.stamp = stamp
        t2.header.frame_id = "base_footprint"
        t2.child_frame_id = "base_link"

        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = pose.position.z

        t2.transform.rotation = Quaternion(0, 0, 0, 1)

        self.br.sendTransform(t2)

        # -------- /odom --------
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"

        odom.pose.pose.position.x = pose.position.x
        odom.pose.pose.position.y = pose.position.y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation = Quaternion(*q_yaw)

        odom.twist.twist.linear.x = twist.linear.x
        odom.twist.twist.linear.y = twist.linear.y
        odom.twist.twist.linear.z = 0.0

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = twist.angular.z

        self.odom_pub.publish(odom)


if __name__ == "__main__":
    rospy.init_node("gazebo_odom_publisher")

    node = GazeboOdomPublisher()

    rospy.spin()