#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import copy
import importlib
from collections import deque

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D
from social_mpc_planner.msg import (
    PredictedPerson,
    PredictedPeople,
    PredictedPersonUncertainty,
    PredictedPeopleUncertainty
)


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
        if math.hypot(vx, vy) > 1e-6:
            return math.atan2(vy, vx) + self.yaw_offset
        ux, uy = self._unit_direction()
        return math.atan2(self.direction * uy, self.direction * ux) + self.yaw_offset

    def _quaternion_from_yaw(self, yaw):
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

    def build_gt_extrapolation_prediction(self, dt_pred, horizon):
        """Fallback only; this is the old GT-based future extrapolation."""
        px = self.x
        py = self.y
        direction_now = self.direction
        ux, uy = self._unit_direction()

        traj = []
        for _ in range(horizon + 1):
            theta = math.atan2(direction_now * uy, direction_now * ux)
            traj.append((px, py, theta))

            px += direction_now * self.speed * ux * dt_pred
            py += direction_now * self.speed * uy * dt_pred

            dist_to_b = math.hypot(px - self.bx, py - self.by)
            dist_to_a = math.hypot(px - self.ax, py - self.ay)
            if direction_now == 1 and dist_to_b < self.switch_tolerance:
                direction_now = -1
            elif direction_now == -1 and dist_to_a < self.switch_tolerance:
                direction_now = 1

        return traj


class SocialSTGCNNPredictor:
    """
    Online wrapper around the same logic used in test_base.ipynb:
      obs_xy: [obs_len, N, 2]
      output: [horizon + 1, N, 2] absolute positions, including current point at index 0.
    """

    def __init__(self):
        self.enabled = bool(rospy.get_param("~use_learned_prediction", True))
        self.obs_len = int(rospy.get_param("~obs_len", 8))
        self.model_pred_len = int(rospy.get_param("~model_pred_len", 12))
        self.n_stgcnn = int(rospy.get_param("~n_stgcnn", 1))
        self.n_txpcnn = int(rospy.get_param("~n_txpcnn", 5))
        self.output_feat = int(rospy.get_param("~output_feat", 5))
        self.kernel_size = int(rospy.get_param("~kernel_size", 3))
        self.norm_lap_matr = bool(rospy.get_param("~norm_lap_matr", True))
        self.device_name = str(rospy.get_param("~device", "cpu"))

        # Path containing model.py and utils.py from Social-STGCNN.
        self.social_stgcnn_path = str(rospy.get_param("~social_stgcnn_path", ""))
        self.checkpoint_path = str(rospy.get_param("~checkpoint_path", ""))

        self.model = None
        self.seq_to_graph = None
        self.torch = None
        self.device = None

        if self.enabled:
            self._load_model()

    def _load_model(self):
        rospy.loginfo(
            "Social-STGCNN params | path='%s' | checkpoint='%s' | device='%s'",
            self.social_stgcnn_path,
            self.checkpoint_path,
            self.device_name
        )

        if self.social_stgcnn_path:
            abs_path = os.path.abspath(os.path.expanduser(self.social_stgcnn_path))
            if abs_path not in sys.path:
                sys.path.insert(0, abs_path)
            rospy.loginfo("Added Social-STGCNN path to sys.path: %s", abs_path)
        else:
            rospy.logwarn("~social_stgcnn_path is empty. Import will only work if model.py/utils.py are already on PYTHONPATH.")

        try:
            self.torch = importlib.import_module("torch")

            nx = importlib.import_module("networkx")
            if not hasattr(nx, "from_numpy_matrix") and hasattr(nx, "from_numpy_array"):
                nx.from_numpy_matrix = nx.from_numpy_array

            utils_mod = importlib.import_module("utils")
            model_mod = importlib.import_module("model")
            self.seq_to_graph = utils_mod.seq_to_graph
            social_stgcnn = model_mod.social_stgcnn
        except Exception as e:
            self.enabled = False
            rospy.logerr("Could not import Social-STGCNN modules. Falling back to constant-velocity prediction. Error: %s", str(e))
            rospy.logerr("Check that ~social_stgcnn_path points to the folder containing model.py and utils.py.")
            return

        if self.device_name == "cuda" and not self.torch.cuda.is_available():
            rospy.logwarn("CUDA requested but unavailable. Using CPU.")
            self.device_name = "cpu"
        self.device = self.torch.device(self.device_name)

        self.model = social_stgcnn(
            n_stgcnn=self.n_stgcnn,
            n_txpcnn=self.n_txpcnn,
            output_feat=self.output_feat,
            seq_len=self.obs_len,
            kernel_size=self.kernel_size,
            pred_seq_len=self.model_pred_len,
        ).to(self.device)

        if not self.checkpoint_path:
            self.enabled = False
            rospy.logerr("~checkpoint_path is empty. Falling back to GT extrapolation.")
            return

        ckpt = os.path.abspath(os.path.expanduser(self.checkpoint_path))
        if not os.path.exists(ckpt):
            self.enabled = False
            rospy.logerr("Checkpoint file does not exist: %s. Falling back to constant-velocity prediction.", ckpt)
            return

        state = self.torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        rospy.loginfo("Loaded Social-STGCNN checkpoint: %s", ckpt)

    def predict(self, obs_xy, publish_horizon):
        """
        obs_xy: [obs_len, N, 2], absolute GT observation window.
        return: [publish_horizon + 1, N, 2], absolute prediction including current point.
        """
        obs_xy = np.asarray(obs_xy, dtype=np.float32)
        if obs_xy.ndim != 3 or obs_xy.shape[0] != self.obs_len or obs_xy.shape[2] != 2:
            raise ValueError("obs_xy must have shape [obs_len, N, 2]")

        num_people = obs_xy.shape[1]

        rospy.logwarn_throttle(
            1.0,
            "PREDICT DEBUG | enabled=%s | num_people=%d | model=%s | seq_to_graph=%s",
            str(self.enabled),
            num_people,
            str(self.model is not None),
            str(self.seq_to_graph is not None)
        )
        
        if (not self.enabled) or num_people < 2:
            out = self._constant_velocity_fallback(obs_xy, publish_horizon)
            # No learned covariance is available in fallback mode.
            # Publish small zero uncertainty arrays with the same temporal length.
            T = out.shape[0]
            sigma_x = np.zeros((T, num_people), dtype=np.float32)
            sigma_y = np.zeros((T, num_people), dtype=np.float32)
            rho = np.zeros((T, num_people), dtype=np.float32)
            return out, sigma_x, sigma_y, rho

        obs_rel = np.zeros_like(obs_xy)
        obs_rel[1:] = obs_xy[1:] - obs_xy[:-1]

        obs_traj = self.torch.from_numpy(obs_xy.transpose(1, 2, 0)).float()
        obs_traj_rel = self.torch.from_numpy(obs_rel.transpose(1, 2, 0)).float()

        V_obs, A_obs = self.seq_to_graph(obs_traj, obs_traj_rel, norm_lap_matr=self.norm_lap_matr)
        V_in = V_obs.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        A_in = A_obs.to(self.device)

        with self.torch.no_grad():
            raw_pred, _ = self.model(V_in, A_in)
            raw_np = raw_pred.detach().cpu().numpy()

            rospy.logwarn_throttle(
                1.0,
                "RAW DEBUG | raw min=%.6f max=%.6f | ch2 min=%.6f max=%.6f | ch3 min=%.6f max=%.6f | ch4 min=%.6f max=%.6f",
                float(np.min(raw_np)),
                float(np.max(raw_np)),
                float(np.min(raw_np[0, 2])),
                float(np.max(raw_np[0, 2])),
                float(np.min(raw_np[0, 3])),
                float(np.max(raw_np[0, 3])),
                float(np.min(raw_np[0, 4])),
                float(np.max(raw_np[0, 4]))
            )


        # raw_pred shape is [1, 5, pred_len, N].
        # Keep uncertainty arrays as [pred_len, N]. Do NOT transpose them.
        sigma_x = np.exp(raw_pred[0, 2].detach().cpu().numpy()).astype(np.float32)
        sigma_y = np.exp(raw_pred[0, 3].detach().cpu().numpy()).astype(np.float32)
        rho = np.tanh(raw_pred[0, 4].detach().cpu().numpy()).astype(np.float32)


        # Same as test_base: raw_pred [1, 5, pred_len, N], first two channels are relative dx, dy.
        pred_rel = (
            raw_pred[0, 0:2]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        pred_abs = np.zeros_like(pred_rel, dtype=np.float32)
        pred_abs[0] = obs_xy[-1] + pred_rel[0]
        for t in range(1, pred_rel.shape[0]):
            pred_abs[t] = pred_abs[t - 1] + pred_rel[t]

        # Insert current measured point as trajectory[0], because your MPC message previously used horizon+1 points.
        out = np.concatenate([obs_xy[-1:, :, :], pred_abs], axis=0)
        out = self._extend_or_trim(out, publish_horizon + 1)

        # Add one uncertainty row for the current measured point.
        # The current point is measured, so its model uncertainty is set to zero.
        zero_unc = np.zeros((1, num_people), dtype=np.float32)
        sigma_x = np.concatenate([zero_unc, sigma_x], axis=0)
        sigma_y = np.concatenate([zero_unc, sigma_y], axis=0)
        rho = np.concatenate([zero_unc, rho], axis=0)

        # Do not extrapolate uncertainty. If prediction_horizon is longer than the model output,
        # uncertainty is published only for the valid learned prediction length.
        max_unc_len = min(sigma_x.shape[0], publish_horizon + 1)
        sigma_x = sigma_x[:max_unc_len]
        sigma_y = sigma_y[:max_unc_len]
        rho = rho[:max_unc_len]

        return out, sigma_x, sigma_y, rho

    def _extend_or_trim(self, traj, target_len):
        if traj.shape[0] == target_len:
            return traj
        if traj.shape[0] > target_len:
            return traj[:target_len]

        extra = target_len - traj.shape[0]
        if traj.shape[0] >= 2:
            v = traj[-1] - traj[-2]
        else:
            v = np.zeros_like(traj[-1])

        tail = []
        last = traj[-1].copy()
        for _ in range(extra):
            last = last + v
            tail.append(last.copy())
        return np.concatenate([traj, np.stack(tail, axis=0)], axis=0)

    def _constant_velocity_fallback(self, obs_xy, publish_horizon):
        if obs_xy.shape[0] >= 2:
            v = obs_xy[-1] - obs_xy[-2]
        else:
            v = np.zeros_like(obs_xy[-1])

        out = np.zeros((publish_horizon + 1, obs_xy.shape[1], 2), dtype=np.float32)
        out[0] = obs_xy[-1]
        for k in range(1, publish_horizon + 1):
            out[k] = out[k - 1] + v
        return out


class ScriptedMultiHumanMotionNode:
    def __init__(self):
        rospy.init_node("scripted_multi_human_motion_node")

        self.default_radius = rospy.get_param("~person_radius", 0.25)
        self.default_z = rospy.get_param("~z", 0.0)
        self.marker_frame = rospy.get_param("~marker_frame", "odom")
        self.prediction_frame = rospy.get_param("~prediction_frame", "odom")
        self.dt_pred = float(rospy.get_param("~prediction_dt", 0.5))
        self.n_pred = int(rospy.get_param("~prediction_horizon", 12))
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.publish_before_history_full = bool(rospy.get_param("~publish_before_history_full", True))

        humans_cfg = rospy.get_param("~humans", [])
        if not humans_cfg:
            humans_cfg = [{
                "id": 1,
                "model_name": rospy.get_param("~model_name", "person_1"),
                "ax": rospy.get_param("~ax", 1.5),
                "ay": rospy.get_param("~ay", 0.0),
                "bx": rospy.get_param("~bx", 15.0),
                "by": rospy.get_param("~by", 0.0),
                "speed": rospy.get_param("~speed", 0.03),
                "z": rospy.get_param("~z", 0.0),
                "radius": rospy.get_param("~person_radius", 0.25),
            }]

        self.humans = [ScriptedHuman(cfg, self.default_radius, self.default_z) for cfg in humans_cfg]
        self.predictor = SocialSTGCNNPredictor()

        self.history = deque(maxlen=self.predictor.obs_len)
        self.last_history_stamp = None

        self.pred_pub = rospy.Publisher("/predicted_people", PredictedPeople, queue_size=10)
        self.unc_pub = rospy.Publisher("/predicted_people_uncertainty", PredictedPeopleUncertainty, queue_size=10)
        self.marker_pub = rospy.Publisher("/human_obstacle_marker", Marker, queue_size=10)

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo("scripted multi-human motion started with %d humans", len(self.humans))

    def _current_positions(self):
        return np.asarray([[h.x, h.y] for h in self.humans], dtype=np.float32)
    
    def build_uncertainty_message(self, sigma_x, sigma_y, rho):
        msg = PredictedPeopleUncertainty()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.prediction_frame
        msg.dt = self.dt_pred
        msg.horizon = sigma_x.shape[0] - 1

        for i, human in enumerate(self.humans):
            p = PredictedPersonUncertainty()
            p.id = human.id
            p.sigma_x = [float(v) for v in sigma_x[:, i]]
            p.sigma_y = [float(v) for v in sigma_y[:, i]]
            p.rho = [float(v) for v in rho[:, i]]
            msg.people.append(p)

        return msg

    def _update_history(self, now):
        if self.last_history_stamp is None:
            self.history.append(self._current_positions())
            self.last_history_stamp = now
            return

        if (now - self.last_history_stamp).to_sec() >= self.dt_pred:
            self.history.append(self._current_positions())
            self.last_history_stamp = now

    def _build_person_msg(self, human_index, pred_xy):
        human = self.humans[human_index]
        person = PredictedPerson()
        person.id = human.id
        person.radius = human.radius

        for k in range(pred_xy.shape[0]):
            pose = Pose2D()
            pose.x = float(pred_xy[k, human_index, 0])
            pose.y = float(pred_xy[k, human_index, 1])

            if k < pred_xy.shape[0] - 1:
                dx = pred_xy[k + 1, human_index, 0] - pred_xy[k, human_index, 0]
                dy = pred_xy[k + 1, human_index, 1] - pred_xy[k, human_index, 1]
            elif k > 0:
                dx = pred_xy[k, human_index, 0] - pred_xy[k - 1, human_index, 0]
                dy = pred_xy[k, human_index, 1] - pred_xy[k - 1, human_index, 1]
            else:
                dx, dy = 0.0, 0.0

            pose.theta = float(math.atan2(dy, dx)) if math.hypot(dx, dy) > 1e-6 else 0.0
            person.trajectory.append(pose)

        return person

    def build_learned_prediction_message(self):
        out = PredictedPeople()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.prediction_frame
        out.dt = self.dt_pred
        out.horizon = self.n_pred

        if len(self.history) < self.predictor.obs_len:
            if not self.publish_before_history_full:
                return out
            # Until the observation window is full, keep the old GT extrapolation so MPC is not starved.
            for human in self.humans:
                traj = human.build_gt_extrapolation_prediction(self.dt_pred, self.n_pred)
                person = PredictedPerson()
                person.id = human.id
                person.radius = human.radius
                for x, y, th in traj:
                    pose = Pose2D()
                    pose.x = float(x)
                    pose.y = float(y)
                    pose.theta = float(th)
                    person.trajectory.append(pose)
                out.people.append(person)
            return out

        obs_xy = np.stack(list(self.history), axis=0)  # [obs_len, N, 2]

        try:
            pred_xy, sigma_x, sigma_y, rho = self.predictor.predict(obs_xy, self.n_pred)

            for i in range(len(self.humans)):
                out.people.append(self._build_person_msg(i, pred_xy))

            # Publish uncertainty once, outside the loop over humans.
            unc_msg = self.build_uncertainty_message(sigma_x, sigma_y, rho)
            self.unc_pub.publish(unc_msg)

        except Exception as e:
            rospy.logwarn_throttle(1.0, "Learned prediction failed; using GT extrapolation fallback: %s", str(e))
            for human in self.humans:
                traj = human.build_gt_extrapolation_prediction(self.dt_pred, self.n_pred)
                person = PredictedPerson()
                person.id = human.id
                person.radius = human.radius
                for x, y, th in traj:
                    pose = Pose2D()
                    pose.x = float(x)
                    pose.y = float(y)
                    pose.theta = float(th)
                    person.trajectory.append(pose)
                out.people.append(person)

        return out

    def run(self):
        prev_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - prev_time).to_sec()
            prev_time = now

            # If simulated time jumps backward or is not advancing yet, avoid corrupting motion.
            if dt < 0.0:
                rospy.logwarn_throttle(1.0, "Negative dt detected. Skipping this cycle.")
                self.rate.sleep()
                continue

            # 1) Move humans. This must never depend on whether prediction/model works.
            try:
                for human in self.humans:
                    vx, vy = human.step(dt)
                    self.set_state(human.build_model_state(vx, vy))
                    self.marker_pub.publish(human.build_marker(self.marker_frame))
            except rospy.ServiceException as e:
                rospy.logwarn_throttle(1.0, "Failed to set model state: %s", str(e))
            except Exception as e:
                rospy.logwarn_throttle(1.0, "Unexpected human motion error: %s", str(e))

            # 2) Prediction/uncertainty publishing. If this fails, humans still keep moving.
            try:
                self._update_history(now)
                pred_msg = self.build_learned_prediction_message()
                self.pred_pub.publish(pred_msg)
            except Exception as e:
                rospy.logwarn_throttle(1.0, "Prediction publishing failed, but human motion continues: %s", str(e))

            self.rate.sleep()


if __name__ == "__main__":
    node = ScriptedMultiHumanMotionNode()
    node.run()
