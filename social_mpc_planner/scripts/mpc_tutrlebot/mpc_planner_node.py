#!/usr/bin/env python3

import rospy
import math
import casadi as ca
import numpy as np
import csv
import os
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion
from social_mpc_planner.msg import CircularObstacleArray
from social_mpc_planner.msg import PredictedPeople


class MPCPlannerNode:
    def __init__(self):
        rospy.init_node("mpc_planner_node")

        log_dir = os.path.expanduser("~/MPC_ws/logs")
        os.makedirs(log_dir, exist_ok=True)

        self.log_file_path = os.path.join(log_dir, "mpc_log.csv")
        self.log_file = open(self.log_file_path, "w")
        self.csv_writer = csv.writer(self.log_file)

        self.csv_writer.writerow([
            "time",
            "dist_goal",
            "min_obs_clearance",
            "cmd_v",
            "cmd_omega",
            "solver_success"
        ])

        rospy.loginfo("Logging MPC data to %s", self.log_file_path)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_received = False

        self.gx = 0.0
        self.gy = 0.0
        self.goal_received = False

        self.dt = 0.2
        self.N = 12

        self.max_v = 0.4
        self.min_v = 0
        self.max_omega = 0.8
        self.goal_tolerance = 0.2

        self.last_v = 0.0
        self.last_omega = 0.0

        self.prev_X_sol = None
        self.prev_U_sol = None

        self.robot_radius = 0.25
        self.safety_margin = 0.08
        self.max_obstacles = 9
        self.obstacle_timeout = 1.0
        self.last_obstacle_time = rospy.Time(0)
        self.placeholder_obstacle = (100.0, 100.0, 0.15)

        self.max_people = 1
        self.human_safety_margin = 0.2
        self.default_human_radius = 0.2

        rospy.loginfo(
            "People params: max_people=%d human_safety_margin=%.2f default_radius=%.2f",
            self.max_people,
            self.human_safety_margin,
            self.default_human_radius
        )

        self.predicted_people = []

        self.predicted_people = [
            {
                "id": -1,
                "radius": self.default_human_radius,
                "trajectory": [(100.0, 100.0) for _ in range(self.N + 1)]
            }
            for _ in range(self.max_people)
        ]

        self.obstacles = [
            self.placeholder_obstacle
            for _ in range(self.max_obstacles)
        ]

        self.static_obstacles_sub = rospy.Subscriber(
            "/static_obstacles",
            CircularObstacleArray,
            self.static_obstacles_callback
        )

        self.predicted_people_sub = rospy.Subscriber(
            "/predicted_people",
            PredictedPeople,
            self.predicted_people_callback
        )

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.path_pub = rospy.Publisher("/mpc_path", Path, queue_size=10)

        self.build_mpc()

        self.rate = rospy.Rate(10.0)
        rospy.loginfo("mpc_planner_node with CasADi MPC started.")

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
        rospy.loginfo("New goal received: x=%.3f y=%.3f", self.gx, self.gy)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def static_obstacles_callback(self, msg):
        obstacles = []

        for obs in msg.obstacles:
            dx = self.x - obs.x
            dy = self.y - obs.y
            dist = math.sqrt(dx * dx + dy * dy)
            obstacles.append((dist, obs.x, obs.y, obs.radius))

        obstacles.sort(key=lambda item: item[0])

        selected = []
        for item in obstacles[:self.max_obstacles]:
            _, ox, oy, r = item
            selected.append((ox, oy, r))

        while len(selected) < self.max_obstacles:
            selected.append(self.placeholder_obstacle)

        self.obstacles = selected

        rospy.loginfo_throttle(
            2.0,
            "MPC received %d obstacles, using nearest %d",
            len(msg.obstacles),
            self.max_obstacles
        )
    
    def predicted_people_callback(self, msg):
        people = []

        for person in msg.people[:self.max_people]:
            traj = []

            for pose in person.trajectory[:self.N + 1]:
                traj.append((pose.x, pose.y))

            while len(traj) < self.N + 1:
                if len(traj) > 0:
                    traj.append(traj[-1])
                else:
                    traj.append((100.0, 100.0))

            people.append({
                "id": person.id,
                "radius": person.radius,
                "trajectory": traj
            })

        while len(people) < self.max_people:
            people.append({
                "id": -1,
                "radius": self.default_human_radius,
                "trajectory": [(100.0, 100.0) for _ in range(self.N + 1)]
            })

        self.predicted_people = people

        rospy.loginfo_throttle(
            2.0,
            "MPC received %d predicted people",
            len(msg.people)
        )

        if len(self.predicted_people) > 0:
            p = self.predicted_people[0]
            hx, hy = p["trajectory"][0]
            rospy.loginfo_throttle(
                1.0,
                "Human[0]: x=%.2f y=%.2f r=%.2f",
                hx,
                hy,
                p["radius"]
        )

    def build_mpc(self):
        N = self.N
        dt = self.dt

        opti = ca.Opti()

        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)
        S = opti.variable(len(self.obstacles), N + 1)
        SH = opti.variable(self.max_people, N + 1)

        P = opti.parameter(
            5
            + 3 * self.max_obstacles
            + 3 * self.max_people * (N + 1)
        )

        x0 = P[0]
        y0 = P[1]
        theta0 = P[2]
        gx = P[3]
        gy = P[4]

        cost = 0

        Q_pos = 25.0
        Q_theta = 0.4
        R_v = 0.3
        R_omega = 0.25
        R_smooth = 1.0
        R_jerk = 0.2
        W_slack = 10000.0
        W_human_slack = 25000.0

        opti.subject_to(X[0, 0] == x0)
        opti.subject_to(X[1, 0] == y0)
        opti.subject_to(X[2, 0] == theta0)

        for k in range(N):
            px = X[0, k]
            py = X[1, k]
            th = X[2, k]

            v = U[0, k]
            omega = U[1, k]

            px_next = px + v * ca.cos(th) * dt
            py_next = py + v * ca.sin(th) * dt
            th_next = th + omega * dt

            for j in range(self.max_obstacles):
                base = 5 + 3 * j
                human_base = 5 + 3 * self.max_obstacles

                ox = P[base]
                oy = P[base + 1]
                r = P[base + 2]
                safe_dist = self.robot_radius + r + self.safety_margin

                dx_obs = X[0, k] - ox
                dy_obs = X[1, k] - oy

                dist_sq = dx_obs * dx_obs + dy_obs * dy_obs

                opti.subject_to(S[j, k] >= 0.0)
                opti.subject_to(dist_sq + S[j, k] >= safe_dist * safe_dist)

                cost += W_slack * S[j, k] * S[j, k]

            human_base = 5 + 3 * self.max_obstacles

            for h in range(self.max_people):
                base = human_base + h * 3 * (N + 1) + 3 * k

                hx = P[base]
                hy = P[base + 1]
                hr = P[base + 2]

                safe_dist = self.robot_radius + hr + self.human_safety_margin

                dx_h = X[0, k] - hx
                dy_h = X[1, k] - hy

                dist_sq_h = dx_h * dx_h + dy_h * dy_h

                opti.subject_to(SH[h, k] >= 0.0)
                opti.subject_to(dist_sq_h + SH[h, k] >= safe_dist * safe_dist)

                cost += W_human_slack * SH[h, k] * SH[h, k]
                

            opti.subject_to(X[0, k + 1] == px_next)
            opti.subject_to(X[1, k + 1] == py_next)
            opti.subject_to(X[2, k + 1] == th_next)

            dx = X[0, k] - gx
            dy = X[1, k] - gy

            goal_dist_sq = dx * dx + dy * dy

            eps = 0.05
            heading_weight = Q_theta * goal_dist_sq / (goal_dist_sq + eps)

            desired_theta = ca.atan2(gy - X[1, k], gx - X[0, k])

            theta_error = ca.atan2(
                ca.sin(X[2, k] - desired_theta),
                ca.cos(X[2, k] - desired_theta)
            )

            cost += Q_pos * goal_dist_sq
            cost += heading_weight * theta_error * theta_error

            if k > 0:
                dv = U[0, k] - U[0, k - 1]
                domega = U[1, k] - U[1, k - 1]
                cost += R_smooth * (dv * dv + domega * domega)

            if k > 1:
                d2v = U[0, k] - 2.0 * U[0, k - 1] + U[0, k - 2]
                d2omega = U[1, k] - 2.0 * U[1, k - 1] + U[1, k - 2]
                cost += R_jerk * (d2v * d2v + d2omega * d2omega)

            opti.subject_to(opti.bounded(self.min_v, v, self.max_v))
            opti.subject_to(opti.bounded(-self.max_omega, omega, self.max_omega))

        for j in range(self.max_obstacles):
                base = 5 + 3 * j

                ox = P[base]
                oy = P[base + 1]
                r = P[base + 2]

                safe_dist = self.robot_radius + r + self.safety_margin

                dx_obs = X[0, N] - ox
                dy_obs = X[1, N] - oy

                dist_sq = dx_obs * dx_obs + dy_obs * dy_obs

                opti.subject_to(S[j, N] >= 0.0)
                opti.subject_to(dist_sq + S[j, N] >= safe_dist * safe_dist)

                cost += W_slack * S[j, N] * S[j, N]
        
        human_base = 5 + 3 * self.max_obstacles

        for h in range(self.max_people):
            k = N
            base = human_base + h * 3 * (N + 1) + 3 * k

            hx = P[base]
            hy = P[base + 1]
            hr = P[base + 2]

            safe_dist = self.robot_radius + hr + self.human_safety_margin

            dx_h = X[0, N] - hx
            dy_h = X[1, N] - hy

            dist_sq_h = dx_h * dx_h + dy_h * dy_h

            opti.subject_to(SH[h, k] >= 0.0)
            opti.subject_to(dist_sq_h + SH[h, k] >= safe_dist * safe_dist)

            cost += W_human_slack * SH[h, k] * SH[h, k]


        terminal_dx = X[0, N] - gx
        terminal_dy = X[1, N] - gy
        cost += 30.0 * (terminal_dx * terminal_dx + terminal_dy * terminal_dy)

        opti.minimize(cost)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": 150,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2    
        }

        opti.solver("ipopt", opts)

        self.opti = opti
        self.X = X
        self.U = U
        self.P = P
        self.S = S
        self.SH = SH

    def solve_mpc(self):

        p_value = [self.x, self.y, self.theta, self.gx, self.gy]

        for obs in self.obstacles[:self.max_obstacles]:
            ox, oy, r = obs
            p_value += [ox, oy, r]

        while len(p_value) < 5 + 3 * self.max_obstacles:
            p_value += [100.0, 100.0, 0.15]

        for person in self.predicted_people[:self.max_people]:
            r = person["radius"]
            traj = person["trajectory"]

            for k in range(self.N + 1):
                hx, hy = traj[k]
                p_value += [hx, hy, r]

        expected_len = (
            5
            + 3 * self.max_obstacles
            + 3 * self.max_people * (self.N + 1)
        )

        while len(p_value) < expected_len:
            p_value += [100.0, 100.0, self.default_human_radius]

        self.opti.set_value(self.P, p_value)

        X_init = np.zeros((3, self.N + 1))
        U_init = np.zeros((2, self.N))
        SH_init = np.ones((self.max_people, self.N + 1)) * 0.1
        self.opti.set_initial(self.SH, SH_init)

        for k in range(self.N + 1):
            alpha = k / self.N

            X_init[0, k] = self.x + alpha * (self.gx - self.x)
            X_init[1, k] = self.y + alpha * (self.gy - self.y)

  
            X_init[1, k] += 0.25 * math.sin(math.pi * alpha)

            X_init[2, k] = self.theta

        U_init[0, :] = 0.15
        U_init[1, :] = 0.0

        if self.prev_X_sol is not None and self.prev_U_sol is not None:
            X_guess = np.zeros((3, self.N + 1))
            U_guess = np.zeros((2, self.N))

            X_guess[:, :-1] = self.prev_X_sol[:, 1:]
            X_guess[:, -1] = self.prev_X_sol[:, -1]

            U_guess[:, :-1] = self.prev_U_sol[:, 1:]
            U_guess[:, -1] = self.prev_U_sol[:, -1]

            self.opti.set_initial(self.X, X_guess)
            self.opti.set_initial(self.U, U_guess)
        else:
            self.opti.set_initial(self.X, X_init)
            self.opti.set_initial(self.U, U_init)

        S_init = np.ones((len(self.obstacles), self.N + 1)) * 0.1
        self.opti.set_initial(self.S, S_init)

        try:
            sol = self.opti.solve()

            X_sol = sol.value(self.X)
            U_sol = sol.value(self.U)

            self.prev_X_sol = X_sol
            self.prev_U_sol = U_sol

            self.publish_mpc_path(X_sol)

            u0 = sol.value(self.U[:, 0])

            v = float(u0[0])
            omega = float(u0[1])

            self.last_v = v
            self.last_omega = omega

            return True, v, omega

        except RuntimeError as e:
            rospy.logwarn_throttle(1.0, "MPC solver failed. Using fallback.")
            rospy.logwarn_throttle(1.0, str(e))

            dx = self.gx - self.x
            dy = self.gy - self.y

            self.prev_X_sol = None
            self.prev_U_sol = None

            #dist = math.sqrt(dx * dx + dy * dy)

            #if dist < 0.2:
            #    return False, 0.0, 0.0

            #return False, 0.5 * self.last_v, 0.5 * self.last_omega
            return False, 0.0, 0.0

    def control(self):
        cmd = Twist()

        if not self.odom_received or not self.goal_received:
            return cmd

        dx = self.gx - self.x
        dy = self.gy - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.goal_tolerance:
            rospy.loginfo_throttle(1.0, "Goal reached. Stopping.")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
        
        if dist < 0.5:
            self.max_v_runtime = 0.20
        else:
            self.max_v_runtime = self.max_v

        success, v, omega = self.solve_mpc()

        cmd.linear.x = v
        cmd.angular.z = omega

        min_obs_dist = self.compute_min_obstacle_distance()
        self.write_log(dist, min_obs_dist, v, omega, success)

        rospy.loginfo_throttle(
            1.0,
            "MPC success=%s | dist_goal=%.2f | min_obs_clearance=%.2f | cmd_v=%.2f | cmd_omega=%.2f",
            str(success),
            dist,
            min_obs_dist,
            v,
            omega
        )

        return cmd

    def run(self):
        while not rospy.is_shutdown():
            cmd = self.control()
            self.cmd_pub.publish(cmd)
            self.rate.sleep()

    def compute_min_obstacle_distance(self):
        if len(self.obstacles) == 0:
            return 999.0

        min_dist = 999.0

        for ox, oy, r in self.obstacles:
            # obstacleهای placeholder را رد کن
            if abs(ox) > 50.0 or abs(oy) > 50.0:
                continue

            dx = self.x - ox
            dy = self.y - oy

            center_dist = math.sqrt(dx * dx + dy * dy)
            clearance = center_dist - (self.robot_radius + r)

            if clearance < min_dist:
                min_dist = clearance

        return min_dist

    def publish_mpc_path(self, X_sol):
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "odom"

        for k in range(self.N + 1):
            pose = PoseStamped()
            pose.header.stamp = path.header.stamp
            pose.header.frame_id = "odom"

            pose.pose.position.x = float(X_sol[0, k])
            pose.pose.position.y = float(X_sol[1, k])
            pose.pose.position.z = 0.05
            pose.pose.orientation.w = 1.0

            path.poses.append(pose)

        self.path_pub.publish(path)
    
    def write_log(self, dist_goal, min_obs_clearance, cmd_v, cmd_omega, solver_success):
        self.csv_writer.writerow([
            rospy.Time.now().to_sec(),
            dist_goal,
            min_obs_clearance,
            cmd_v,
            cmd_omega,
            int(solver_success)
        ])
        self.log_file.flush()

if __name__ == "__main__":
    node = MPCPlannerNode()
    node.run()