#!/usr/bin/env python3

import rospy
import math
import casadi as ca
import json
import os
import subprocess
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Point
from social_mpc_planner.msg import CircularObstacleArray
from social_mpc_planner.msg import PredictedPeople
from gait_planner.srv import Trajectory, TrajectoryRequest


class FootstepMPCPlanner:
    def __init__(self):
        rospy.init_node("footstep_mpc_planner_node")

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_received = False

        self.gx = 0.0
        self.gy = 0.0
        self.goal_received = False

        self.N = 10

        self.step_width = 0.11
        self.step_width_tol = 0.02

        # lateral distance between opposite feet
        self.min_cross_width = 0.2
        self.max_cross_width = 0.25
        self.nominal_cross_width = 2.0 * self.step_width

        self.min_step_x = 0.03
        self.max_step_x = 0.2

        self.max_yaw_step = 0.08

        self.Q_goal = 80.0
        self.Q_progress = 3.0
        self.Q_nominal = 30.0
        self.Q_yaw = 0.2
        self.R_smooth = 15.0
        self.R_yaw_smooth = 15.0

        self.next_foot_is_left = True

        self.obstacles = []
        self.max_obstacles = 9
        self.foot_clearance = 0.5
        self.W_backward = 10000.0
        self.W_obs_slack = 30000.0
        self.W_human_slack = 40000.0

        self.static_obstacles_sub = rospy.Subscriber(
            "/static_obstacles",
            CircularObstacleArray,
            self.static_obstacles_callback
        )

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

        self.marker_pub = rospy.Publisher("/footstep_markers", MarkerArray, queue_size=10)

        rospy.loginfo("footstep_mpc_planner_node started.")

        self.walk_config_path = os.path.expanduser(
            "~/Surena_ws/src/surenaV_core/gait_planner/config/walk_config.json"
        )

        self.step_width_config = 0.10
        self.default_walk_config = {
            "ankle_height": 0.025,
            "COM_height": 0.68,
            "com_offset": 0.01,
            "t_step": 1.0,
            "t_double_support": 0.1,
            "dt": 0.005,
            "alpha": 0.44,
            "hand_swing_angle": 15.0,
        }

        self.step_time = 1.0
        self.human_prediction_dt = 0.2

        self.max_people = 3
        self.human_safety_margin = 0.2
        self.default_human_radius = 0.20
        self.foot_radius = 0.2

        self.predicted_people = [
            {
                "id": -1,
                "radius": self.default_human_radius,
                "trajectory": [(100.0, 100.0) for _ in range(self.N + 1)]
            }
            for _ in range(self.max_people)
        ]

        self.predicted_people_sub = rospy.Subscriber(
            "/predicted_people",
            PredictedPeople,
            self.predicted_people_callback
        )

        self.execution_steps = 2
        self.use_receding_horizon = True

        self.goal_tolerance = 0.15
        self.max_replan_cycles = 20
        self.replan_count = 0
        self.is_walking = False
        self.active_goal = False

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, self.theta = euler_from_quaternion(quat)

        self.odom_received = True

    def goal_callback(self, msg):
        self.gx = msg.pose.position.x
        self.gy = msg.pose.position.y
        self.goal_received = True
        self.active_goal = True
        self.replan_count = 0

        rospy.loginfo("New footstep goal: x=%.2f y=%.2f", self.gx, self.gy)

        dxx = self.gx - self.x
        dyy = self.gy - self.y
        dist = math.sqrt(dxx*dxx + dyy*dyy)

        nominal_step = 0.15
        N = int(math.ceil(dist / nominal_step))

        N = max(4, min(N, 12))  
        self.N = N


        if self.odom_received:
            self.plan_footsteps()

    def distance_to_goal(self):
        dx = self.gx - self.x
        dy = self.gy - self.y
        return math.sqrt(dx * dx + dy * dy)

    def normalize_angle_expr(self, angle):
        return ca.atan2(ca.sin(angle), ca.cos(angle))

    def world_to_local(self, xw, yw, x0, y0, th0):
        dx = xw - x0
        dy = yw - y0

        xl = math.cos(th0) * dx + math.sin(th0) * dy
        yl = -math.sin(th0) * dx + math.cos(th0) * dy

        return xl, yl


    def local_to_world(self, xl, yl, x0, y0, th0):
        xw = x0 + math.cos(th0) * xl - math.sin(th0) * yl
        yw = y0 + math.sin(th0) * xl + math.cos(th0) * yl

        return xw, yw
        
    def predicted_people_callback(self, msg):
        people = []

        for person in msg.people[:self.max_people]:
            traj = []

            for pose in person.trajectory:
                traj.append((pose.x, pose.y))

            if len(traj) == 0:
                traj = [(100.0, 100.0)]

            people.append({
                "id": person.id,
                "radius": person.radius,
                "trajectory": traj
            })

        while len(people) < self.max_people:
            people.append({
                "id": -1,
                "radius": self.default_human_radius,
                "trajectory": [(100.0, 100.0)]
            })

        self.predicted_people = people

        # rospy.loginfo_throttle(
        #     2.0,
        #     "Footstep MPC received %d predicted people",
        #     len(msg.people)
        # )

    def plan_footsteps(self):
        if self.is_walking:
            rospy.logwarn("Robot is walking; skip planning.")
            return

        if not self.active_goal:
            return
        
        N = self.N
        x_start = self.x
        y_start = self.y
        th_start = self.theta

        gx_l, gy_l = self.world_to_local(
            self.gx, self.gy,
            x_start, y_start, th_start
        )

        local_obstacles = []

        for ox, oy, r in self.obstacles:
            ox_l, oy_l = self.world_to_local(
                ox, oy,
                x_start, y_start, th_start
            )
            local_obstacles.append((ox_l, oy_l, r))

        local_people = []

        for person in self.predicted_people[:self.max_people]:
            traj_l = []

            traj = person["trajectory"]

            for k in range(N):
                t_k = k * self.step_time

                human_index = int(round(t_k / self.human_prediction_dt))
                human_index = min(human_index, len(traj) - 1)

                hx, hy = traj[human_index]

                hx_l, hy_l = self.world_to_local(
                    hx, hy,
                    x_start, y_start, th_start
                )

                traj_l.append((hx_l, hy_l))

            local_people.append({
                "id": person["id"],
                "radius": person["radius"],
                "trajectory": traj_l
            })

        opti = ca.Opti()

        X = opti.variable(N)
        Y = opti.variable(N)
        TH = opti.variable(N)
        num_obs = len(local_obstacles)
        S_obs = opti.variable(num_obs, N) if num_obs > 0 else None
        S_human = opti.variable(self.max_people, N)

        cost = 0.0

        prev_x = 0.0
        prev_y = 0.0
        prev_th = 0.0

        goal_theta = math.atan2(gy_l, gx_l)

        for k in range(N):
            xk = X[k]
            yk = Y[k]
            thk = TH[k]

            dx = xk - prev_x
            dy = yk - prev_y

            c = ca.cos(prev_th)
            s = ca.sin(prev_th)

            dx_body = c * dx + s * dy
            dy_body = -s * dx + c * dy

            # backward_amount = ca.fmax(0.0, -dx_body)
            # cost += self.W_backward * backward_amount * backward_amount

            is_left = self.next_foot_is_left if k % 2 == 0 else not self.next_foot_is_left
            if k == 0:
                # first planned step is relative to body center / odom pose
                desired_y = self.step_width if is_left else -self.step_width
            else:
                # later steps are relative to previous foot
                desired_y = 2.0 * self.step_width if is_left else -2.0 * self.step_width

            # step reachability
            opti.subject_to(dx_body >= self.min_step_x)
            opti.subject_to(dx_body <= self.max_step_x)

            if k == 0:
                desired_y = self.step_width if is_left else -self.step_width

                opti.subject_to(dy_body >= desired_y - self.step_width_tol)
                opti.subject_to(dy_body <= desired_y + self.step_width_tol)

            else:
                desired_y = 2.0 * self.step_width if is_left else -2.0 * self.step_width

                if is_left:
                    opti.subject_to(dy_body >= self.min_cross_width)
                    opti.subject_to(dy_body <= self.max_cross_width)

                else:
                    opti.subject_to(dy_body <= -self.min_cross_width)
                    opti.subject_to(dy_body >= -self.max_cross_width)


            for i, (ox, oy, r) in enumerate(local_obstacles):
                safe_dist = r + self.foot_clearance

                dx_obs = xk - ox
                dy_obs = yk - oy

                dist_sq = dx_obs * dx_obs + dy_obs * dy_obs

                opti.subject_to(S_obs[i, k] >= 0.0)
                opti.subject_to(dist_sq + S_obs[i, k] >= safe_dist * safe_dist)

                cost += self.W_obs_slack * S_obs[i, k] * S_obs[i, k]
            
            human_constraint_horizon = min(N, 5)

            if k < human_constraint_horizon:
                for h, person in enumerate(local_people):
                    hx, hy = person["trajectory"][k]
                    hr = person["radius"]

                    safe_dist = hr + self.human_safety_margin + self.foot_radius

                    dx_h = xk - hx
                    dy_h = yk - hy

                    dist_sq_h = dx_h * dx_h + dy_h * dy_h

                    opti.subject_to(S_human[h, k] >= 0.0)
                    opti.subject_to(dist_sq_h + S_human[h, k] >= safe_dist * safe_dist)

                    cost += self.W_human_slack * S_human[h, k] * S_human[h, k]

            # yaw constraint
            dtheta = self.normalize_angle_expr(thk - prev_th)
            opti.subject_to(dtheta >= -self.max_yaw_step)
            opti.subject_to(dtheta <= self.max_yaw_step)

            # cost
            dist_goal_sq = (xk - gx_l) ** 2 + (yk - gy_l) ** 2

            cost += self.Q_progress * dist_goal_sq
            cost += self.Q_nominal * (dy_body - desired_y) ** 2
            cost += self.Q_yaw * self.normalize_angle_expr(thk - goal_theta) ** 2

            if k > 0:
                dstep_x = X[k] - X[k - 1]
                dstep_y = Y[k] - Y[k - 1]
                cost += self.R_smooth * (dstep_x ** 2 + dstep_y ** 2)

                dyaw_smooth = self.normalize_angle_expr(TH[k] - TH[k - 1])
                cost += self.R_yaw_smooth * dyaw_smooth ** 2

            prev_x = xk
            prev_y = yk
            prev_th = thk

        terminal_cost = (X[N - 1] - gx_l) ** 2 + (Y[N - 1] - gy_l) ** 2
        cost += self.Q_goal * terminal_cost

        opti.minimize(cost)

        for k in range(N):
            alpha = float(k + 1) / float(N)

            is_left = self.next_foot_is_left if k % 2 == 0 else not self.next_foot_is_left
            lateral = self.step_width if is_left else -self.step_width

            x_ref = alpha * gx_l
            y_ref = alpha * gy_l

            x_init = x_ref - lateral * math.sin(goal_theta)
            y_init = y_ref + lateral * math.cos(goal_theta)

            opti.set_initial(X[k], x_init)
            opti.set_initial(Y[k], y_init)
            opti.set_initial(TH[k], goal_theta)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 150,
            "ipopt.tol": 5e-2,
        }

        opti.solver("ipopt", opts)
        opti.set_initial(S_human, 0.1)

        try:
            t0 = rospy.Time.now()
            sol = opti.solve()
            rospy.loginfo("MPC solve time: %.3f", (rospy.Time.now() - t0).to_sec())

            footsteps = []

            for k in range(N):
                is_left = self.next_foot_is_left if k % 2 == 0 else not self.next_foot_is_left

                xk = float(sol.value(X[k]))
                yk = float(sol.value(Y[k]))
                thk = float(sol.value(TH[k]))

                footsteps.append({
                    "x": xk,
                    "y": yk,
                    "theta": thk,
                    "foot": "L" if is_left else "R"
                })


            # 1) add final double support in LOCAL frame
            # if len(footsteps) >= 1:
            #     last = footsteps[-1]
            #     last_theta = last["theta"]

            #     last_is_left = (last["foot"] == "L")
            #     lateral_last = self.step_width if last_is_left else -self.step_width

            #     other_is_left = not last_is_left
            #     lateral_other = self.step_width if other_is_left else -self.step_width

            #     # center between feet
            #     cx = last["x"] + lateral_last * math.sin(last_theta)
            #     cy = last["y"] - lateral_last * math.cos(last_theta)

            #     # opposite foot
            #     x_final = cx - lateral_other * math.sin(last_theta)
            #     y_final = cy + lateral_other * math.cos(last_theta)

            #     footsteps.append({
            #         "x": x_final,
            #         "y": y_final,
            #         "theta": last_theta,
            #         "foot": "L" if other_is_left else "R"
            #     })


            # 2) convert LOCAL footsteps to ODOM only for visualization
            world_footsteps = []

            for fs in footsteps:
                xw, yw = self.local_to_world(
                    fs["x"], fs["y"],
                    x_start, y_start, th_start
                )

                world_footsteps.append({
                    "x": xw,
                    "y": yw,
                    "theta": self.wrap_angle(th_start + fs["theta"]),
                    "foot": fs["foot"]
                })


            # 3) local footsteps go to config, world footsteps go to RViz
            if self.use_receding_horizon:
                exec_footsteps = self.make_execution_batch(footsteps)
            else:
                exec_footsteps = footsteps

            self.print_footsteps(exec_footsteps)
            self.publish_markers(world_footsteps)

            self.write_walk_config(exec_footsteps)
            success = self.call_walk_service()

            if not success:
                return

            rospy.sleep(0.2)  # allow odom to update

            dist = self.distance_to_goal()
            rospy.loginfo("Distance to goal after batch: %.3f", dist)

            if dist <= self.goal_tolerance:
                rospy.loginfo("Goal reached. Stopping receding horizon.")
                self.active_goal = False
                return

            self.replan_count += 1

            if self.replan_count >= self.max_replan_cycles:
                rospy.logwarn("Max replan cycles reached. Stopping.")
                self.active_goal = False
                return

            if self.active_goal and not rospy.is_shutdown():
                rospy.loginfo("Replanning next batch...")
                self.plan_footsteps()

        except RuntimeError as e:
            rospy.logwarn("Footstep MPC failed: %s", str(e))

    def print_footsteps(self, footsteps):
        rospy.loginfo("Generated footsteps:")
        for i, fs in enumerate(footsteps):
            rospy.loginfo(
                "%02d | %s | x=%.2f y=%.2f theta=%.2f",
                i,
                fs["foot"],
                fs["x"],
                fs["y"],
                fs["theta"]
            )
    
    def static_obstacles_callback(self, msg):
        obstacles = []

        for obs in msg.obstacles[:self.max_obstacles]:
            obstacles.append((obs.x, obs.y, obs.radius))

        self.obstacles = obstacles

        # rospy.loginfo_throttle(
        #     2.0,
        #     "Footstep planner received %d static obstacles",
        #     len(self.obstacles)
        # )

    def publish_markers(self, footsteps):
        markers = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)

        now = rospy.Time.now()

        # Start marker
        start_marker = Marker()
        start_marker.header.frame_id = "odom"
        start_marker.header.stamp = now
        start_marker.ns = "footstep_start"
        start_marker.id = 1000
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = self.x
        start_marker.pose.position.y = self.y
        start_marker.pose.position.z = 0.08
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = 0.18
        start_marker.scale.y = 0.18
        start_marker.scale.z = 0.18
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        start_marker.color.a = 1.0
        markers.markers.append(start_marker)

        # Goal marker
        goal_marker = Marker()
        goal_marker.header.frame_id = "odom"
        goal_marker.header.stamp = now
        goal_marker.ns = "footstep_goal"
        goal_marker.id = 1001
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = self.gx
        goal_marker.pose.position.y = self.gy
        goal_marker.pose.position.z = 0.08
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.22
        goal_marker.scale.y = 0.22
        goal_marker.scale.z = 0.22
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 1.0
        goal_marker.color.a = 1.0
        markers.markers.append(goal_marker)

        # Path line
        line_marker = Marker()
        line_marker.header.frame_id = "odom"
        line_marker.header.stamp = now
        line_marker.ns = "footstep_line"
        line_marker.id = 1002
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.03
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0

        from geometry_msgs.msg import Point

        p0 = Point()
        p0.x = self.x
        p0.y = self.y
        p0.z = 0.05
        line_marker.points.append(p0)

        for fs in footsteps:
            p = Point()
            p.x = fs["x"]
            p.y = fs["y"]
            p.z = 0.05
            line_marker.points.append(p)

        markers.markers.append(line_marker)

        # Footstep markers
        for i, fs in enumerate(footsteps):
            marker = Marker()
            marker.header.frame_id = "odom"
            marker.header.stamp = now

            marker.ns = "footsteps"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = fs["x"]
            marker.pose.position.y = fs["y"]
            marker.pose.position.z = 0.02

            q = quaternion_from_euler(0.0, 0.0, fs["theta"])
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]

            marker.scale.x = 0.22
            marker.scale.y = 0.10
            marker.scale.z = 0.02

            if fs["foot"] == "L":
                marker.color.r = 0.0
                marker.color.g = 0.2
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.2
                marker.color.b = 0.0

            marker.color.a = 0.9
            markers.markers.append(marker)

        self.marker_pub.publish(markers)

        if len(footsteps) > 0:
            last = footsteps[-1]
            dist_to_goal = math.sqrt((last["x"] - self.gx)**2 + (last["y"] - self.gy)**2)
            rospy.loginfo(
                "Last footstep to goal distance: %.3f m",
                dist_to_goal
            )

    def wrap_angle(self, a):
        return math.atan2(math.sin(a), math.cos(a))


    def write_walk_config(self, footsteps):
        config_steps = []

        config_steps.append([0.0, self.step_width_config, 0.0, 0.0])
        config_steps.append([0.0, -self.step_width_config, 0.0, 0.0])

        for fs in footsteps:
            config_steps.append([
                float(fs["x"]),
                float(fs["y"]),
                0.0,
                float(fs["theta"])
            ])

        walk_config = dict(self.default_walk_config)
        walk_config["footsteps"] = config_steps
        walk_config["do_start_transition"] = (self.replan_count == 0)

        tmp_path = self.walk_config_path + ".tmp"

        with open(tmp_path, "w") as f:
            json.dump(walk_config, f, indent=4)

        os.replace(tmp_path, self.walk_config_path)

        rospy.loginfo(
            "Wrote walk config with %d footsteps to %s",
            len(config_steps),
            self.walk_config_path
        )

    def make_execution_batch(self, footsteps):

        if len(footsteps) == 0:
            return []

        K = min(self.execution_steps, len(footsteps))
        batch = footsteps[:K]

        last = batch[-1]
        last_theta = last["theta"]
        last_is_left = (last["foot"] == "L")

        other_is_left = not last_is_left
        lateral_other = self.step_width_config if other_is_left else -self.step_width_config

        final_cross_width = self.min_cross_width

        if last["foot"] == "L":
            # place right foot to the right of the last left foot
            final_x = last["x"] + final_cross_width * math.sin(last_theta)
            final_y = last["y"] - final_cross_width * math.cos(last_theta)
            final_foot = "R"
        else:
            # place left foot to the left of the last right foot
            final_x = last["x"] - final_cross_width * math.sin(last_theta)
            final_y = last["y"] + final_cross_width * math.cos(last_theta)
            final_foot = "L"

        final_step = {
            "x": final_x,
            "y": final_y,
            "theta": last_theta,
            "foot": final_foot
        }

        batch.append(final_step)

        return batch

    def call_walk_service(self):
        try:
            self.is_walking = True

            rospy.wait_for_service("/walk_service", timeout=5.0)
            walk_srv = rospy.ServiceProxy("/walk_service", Trajectory)
            req = TrajectoryRequest()

            req.is_config = True
            res = walk_srv(req)

            if not res.result:
                rospy.logwarn("Walk service returned false.")
                self.active_goal = False
                return False

            return True

        except Exception as e:
            rospy.logwarn("Walk service failed: %s", str(e))
            self.active_goal = False
            return False

        finally:
            self.is_walking = False

if __name__ == "__main__":
    node = FootstepMPCPlanner()
    rospy.spin()