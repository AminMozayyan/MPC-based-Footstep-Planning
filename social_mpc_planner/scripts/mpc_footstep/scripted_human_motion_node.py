#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import copy
import importlib
import time
from collections import deque
from threading import Lock

import numpy as np
import rospy
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose2D, Point
from social_mpc_planner.msg import (
    PredictedPerson,
    PredictedPeople,
    PredictedPersonUncertainty,
    PredictedPeopleUncertainty,
    HumanArray,
    Human
)
from std_msgs.msg import Float32, Int32


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
            raise RuntimeError("Learned prediction unavailable: disabled or fewer than two people")
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
        self.dt_pred = float(rospy.get_param("~prediction_dt", 0.4))
        self.n_pred = int(rospy.get_param("~prediction_horizon", 12))
        self.rate_hz = float(rospy.get_param("~rate", 20.0))
        self.publish_before_history_full = False

        # Select where the observation history for prediction comes from.
        #   gt              : use the internally scripted Gazebo humans (old behavior)
        #   detection       : use /detected_humans_world published by yolo_node
        #   gazebo_topic    : use /gazebo_humans_world published by yolo_node
        self.human_source = str(rospy.get_param("~human_source", "gazebo_model_states")).lower()
        self.prediction_source = str(rospy.get_param("~prediction_source", "learned")).lower()
        if self.prediction_source not in ("learned", "ground_truth"):
            raise ValueError("~prediction_source must be learned or ground_truth")
        self.gazebo_model_states_topic = rospy.get_param("~gazebo_model_states_topic", "/gazebo/model_states")
        self.person_prefix = rospy.get_param("~person_prefix", "person_")
        self.gazebo_people = {}
        self.gazebo_people_stamp = None
        self.detected_humans_topic = rospy.get_param("~detected_humans_topic", "/detected_humans_world")
        self.gazebo_humans_topic = rospy.get_param("~gazebo_humans_topic", "/gazebo_humans_world")
        self.gt_humans_topic = rospy.get_param("~gt_humans_topic", self.gazebo_humans_topic)
        self.detected_marker_topic = rospy.get_param("~detected_marker_topic", "/human_detected_marker")
        self.gt_marker_topic = rospy.get_param("~gt_marker_topic", "/human_gt_marker")
        self.external_human_timeout = float(rospy.get_param("~external_human_timeout", 0.7))
        self.move_scripted_humans = bool(rospy.get_param("~move_scripted_humans", True))

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
        self.humans_by_id = {int(h.id): h for h in self.humans}
        # The learned predictor is loaded only when selected, avoiding unnecessary checkpoint loads.
        self.predictor = SocialSTGCNNPredictor()
        if not self.predictor.enabled:
            raise RuntimeError("Social-STGCNN checkpoint/model failed to load; refusing CV fallback")
        self.obs_len = int(rospy.get_param("~obs_len", 8))

        self.history = deque(maxlen=self.obs_len)
        self.gazebo_lock = Lock()
        self.gazebo_buffer = deque(maxlen=5000)  # ~5 seconds at 1000 Hz
        self.next_sample_time = None
        self.last_prediction_time = None
        self.last_history_stamp = None
        self.active_people = []
        self.active_ids = []

        # Optional reproducibility log: exact tensors passed to Social-STGCNN.
        self.debug_save_model_io = bool(rospy.get_param("~debug_save_model_io", False))
        self.debug_model_io_dir = os.path.expanduser(str(rospy.get_param(
            "~debug_model_io_dir", "~/social_stgcnn_debug")))
        self.debug_model_io_max_files = int(rospy.get_param("~debug_model_io_max_files", 150))
        self.debug_model_io_every = max(1, int(rospy.get_param("~debug_model_io_every", 1)))
        self.debug_model_io_calls = 0
        self.debug_model_io_saved = 0
        if self.debug_save_model_io:
            os.makedirs(self.debug_model_io_dir, exist_ok=True)
            rospy.loginfo("Social-STGCNN exact input/output logging enabled: %s", self.debug_model_io_dir)

        # External source used only when human_source='gazebo_topic'.
        # Detection data is stored separately because we want to visualize it in RViz
        # even when the predictor/MPC uses internal GT.
        self.external_humans = {}
        self.external_humans_stamp = None
        self.external_sub = None
        self.last_external_input_time = None
        self.last_detected_input_time = None
        self.last_gt_publish_time = None
        self.last_prediction_publish_time = None

        self.detected_humans = {}
        self.detected_humans_stamp = None
        self.detected_sub = rospy.Subscriber(
            self.detected_humans_topic,
            HumanArray,
            self.detected_humans_callback,
            queue_size=1
        )
        rospy.loginfo("Detection marker source subscribed: %s", self.detected_humans_topic)

        if self.human_source == "gazebo_model_states":
            self.gazebo_sub = rospy.Subscriber(
                self.gazebo_model_states_topic, ModelStates,
                self.gazebo_model_states_callback, queue_size=1)
            rospy.loginfo("Human observation source: %s", self.gazebo_model_states_topic)
        elif self.human_source == "detection":
            # Do not create a second subscriber to the same topic.
            # _current_people() will read self.detected_humans for prediction/MPC,
            # while publish_detected_markers() uses the same data for RViz.
            rospy.loginfo("Human prediction source: detection topic %s", self.detected_humans_topic)

        elif self.human_source in ["gazebo_topic", "gt_topic"]:
            self.human_source = "gazebo_topic"
            self.external_sub = rospy.Subscriber(
                self.gazebo_humans_topic,
                HumanArray,
                self.external_humans_callback,
                queue_size=1
            )
            rospy.loginfo("Human prediction source: Gazebo HumanArray topic %s", self.gazebo_humans_topic)

        elif self.human_source == "gt":
            rospy.loginfo("Human prediction source: internal scripted GT")
        else:
            rospy.logwarn("Unknown ~human_source='%s'. Falling back to 'gt'.", self.human_source)
            self.human_source = "gt"

        self.pred_pub = rospy.Publisher("/predicted_people", PredictedPeople, queue_size=10)
        self.learned_pub = rospy.Publisher("/human_social_stgcnn_prediction", PredictedPeople, queue_size=10)
        self.oracle_pub = rospy.Publisher("/human_ground_truth_future", PredictedPeople, queue_size=10)  # scripted oracle, not future measured Gazebo
        self.unc_pub = rospy.Publisher("/predicted_people_uncertainty", PredictedPeopleUncertainty, queue_size=10)

        # Timing / frequency diagnostics for integration analysis.
        self.prediction_processing_time_pub = rospy.Publisher("/evaluation/prediction_processing_time", Float32, queue_size=10)
        self.prediction_total_cycle_time_pub = rospy.Publisher("/evaluation/prediction_total_cycle_time", Float32, queue_size=10)
        self.prediction_publish_rate_pub = rospy.Publisher("/evaluation/prediction_publish_rate_hz", Float32, queue_size=10)
        self.prediction_input_rate_pub = rospy.Publisher("/evaluation/prediction_input_rate_hz", Float32, queue_size=10)
        self.prediction_history_size_pub = rospy.Publisher("/evaluation/prediction_history_size", Int32, queue_size=10)
        self.prediction_num_people_pub = rospy.Publisher("/evaluation/prediction_num_people", Int32, queue_size=10)
        self.gt_humans_publish_rate_pub = rospy.Publisher("/evaluation/gt_humans_publish_rate_hz", Float32, queue_size=10)

        # Always publish raw GT HumanArray so the logger and other nodes can save it,
        # even when the planner is configured to use detections.
        self.gt_humans_pub = rospy.Publisher(
            rospy.get_param("~scripted_gt_output_topic", "/scripted_humans_world"),
            HumanArray, queue_size=10)

        # Source-based markers: these are the human positions actually used for prediction/MPC.
        self.marker_pub = rospy.Publisher("/human_obstacle_marker", Marker, queue_size=10)

        # Always publish internal scripted GT and detected markers separately for RViz comparison.
        self.gt_marker_pub = rospy.Publisher(self.gt_marker_topic, Marker, queue_size=10)
        self.detected_marker_pub = rospy.Publisher(self.detected_marker_topic, Marker, queue_size=10)

        # Predicted future human trajectories for RViz visualization.
        # These are generated from the same PredictedPeople message published to the MPC.
        self.pred_marker_pub = rospy.Publisher("/predicted_people_marker", MarkerArray, queue_size=10)
        self.pred_marker_stride = int(rospy.get_param("~prediction_marker_stride", 1))
        self.pred_marker_z = float(rospy.get_param("~prediction_marker_z", 0.08))

        self.set_state = None
        if self.move_scripted_humans:
            rospy.wait_for_service("/gazebo/set_model_state")
            self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.rate = rospy.Rate(self.rate_hz)
        rospy.loginfo(
            "scripted multi-human prediction node started | scripted_humans=%d | source=%s",
            len(self.humans),
            self.human_source
        )

    def _parse_human_array_msg(self, msg):
        humans = {}
        for h in msg.humans:
            hid = int(h.id)
            if hid < 0:
                continue

            radius = self.default_radius
            if hasattr(h, "radius"):
                try:
                    if float(h.radius) > 0.0:
                        radius = float(h.radius)
                except Exception:
                    radius = self.default_radius

            humans[hid] = {
                "id": hid,
                "x": float(h.x),
                "y": float(h.y),
                "radius": float(radius),
            }

        return humans


    def _publish_rate_from_interval(self, publisher, current_time, last_time):
        if last_time is not None:
            dt = (current_time - last_time).to_sec()
            if dt > 1e-6:
                publisher.publish(Float32(1.0 / dt))
        return current_time

    def gazebo_model_states_callback(self, msg):
        people = {}
        for index, name in enumerate(msg.name):
            if not name.startswith(self.person_prefix):
                continue
            try:
                hid = int(name[len(self.person_prefix):])
            except ValueError:
                continue
            pose = msg.pose[index]
            twist = msg.twist[index]
            person = self.humans_by_id.get(hid)
            people[hid] = {
                "id": hid, "x": float(pose.position.x), "y": float(pose.position.y),
                "vx": float(twist.linear.x), "vy": float(twist.linear.y),
                "radius": float(person.radius) if person else float(self.default_radius),
            }
        stamp = rospy.Time.now().to_sec()  # ModelStates has no Header; ROS sim receive-time proxy
        with self.gazebo_lock:
            self.gazebo_people = people
            self.gazebo_people_stamp = rospy.Time.from_sec(stamp)
            if self.gazebo_buffer:
                previous_stamp = self.gazebo_buffer[-1][0]
                # Multiple ModelStates messages can have the same /clock timestamp.
                # This must NOT reset the observation history.
                if abs(stamp - previous_stamp) < 1e-9:
                    self.gazebo_buffer[-1] = (stamp, people)
                    return
                if stamp < previous_stamp:
                    # Genuine simulated-time rewind (e.g. Gazebo reset).
                    self.gazebo_buffer.clear()
                    self.history.clear()
                    self.active_ids = []
                    self.next_sample_time = None
                    self.last_history_stamp = None
                    rospy.logwarn("Gazebo time moved backwards; restarting observation history")
            self.gazebo_buffer.append((stamp, people))

    def external_humans_callback(self, msg):
        now = rospy.Time.now()
        self.last_external_input_time = self._publish_rate_from_interval(
            self.prediction_input_rate_pub, now, self.last_external_input_time
        )
        self.external_humans = self._parse_human_array_msg(msg)
        self.external_humans_stamp = now

    def detected_humans_callback(self, msg):
        # This callback is always active, independent of human_source.
        # Therefore RViz can show detections even when prediction/MPC uses GT.
        now = rospy.Time.now()
        self.last_detected_input_time = self._publish_rate_from_interval(
            self.prediction_input_rate_pub, now, self.last_detected_input_time
        )
        self.detected_humans = self._parse_human_array_msg(msg)
        self.detected_humans_stamp = now

    def _current_people(self):
        if self.human_source == "gazebo_model_states":
            if self.gazebo_people_stamp is None:
                rospy.logwarn_throttle(1.0, "Waiting for Gazebo ModelStates")
                return None
            if (rospy.Time.now() - self.gazebo_people_stamp).to_sec() > self.external_human_timeout:
                rospy.logwarn_throttle(1.0, "Gazebo ModelStates stale")
                return None
            if not self.gazebo_people:
                return None
            return [self.gazebo_people[k] for k in sorted(self.gazebo_people)]
        if self.human_source == "gt":
            return [
                {"id": int(h.id), "x": float(h.x), "y": float(h.y), "radius": float(h.radius)}
                for h in self.humans
            ]

        if self.human_source == "detection":
            if self.detected_humans_stamp is None:
                rospy.logwarn_throttle(1.0, "Waiting for detected human data from topic '%s'.", self.detected_humans_topic)
                return None

            age = (rospy.Time.now() - self.detected_humans_stamp).to_sec()
            if age > self.external_human_timeout:
                rospy.logwarn_throttle(
                    1.0,
                    "Detected human data timeout. age=%.2f sec",
                    age
                )
                return None

            if len(self.detected_humans) == 0:
                rospy.logwarn_throttle(1.0, "Detected human topic currently has no humans.")
                return None

            return [self.detected_humans[k] for k in sorted(self.detected_humans.keys())]

        if self.external_humans_stamp is None:
            rospy.logwarn_throttle(1.0, "Waiting for external human data from source='%s'.", self.human_source)
            return None

        age = (rospy.Time.now() - self.external_humans_stamp).to_sec()
        if age > self.external_human_timeout:
            rospy.logwarn_throttle(
                1.0,
                "External human data timeout for source='%s'. age=%.2f sec",
                self.human_source,
                age
            )
            return None

        if len(self.external_humans) == 0:
            rospy.logwarn_throttle(1.0, "External human source='%s' currently has no humans.", self.human_source)
            return None

        return [self.external_humans[k] for k in sorted(self.external_humans.keys())]

    def _positions_from_people(self, people):
        return np.asarray([[p["x"], p["y"]] for p in people], dtype=np.float32)

    def _gazebo_at(self, t):
        with self.gazebo_lock:
            b = list(self.gazebo_buffer)
        if not b or t < b[0][0] - 1e-6 or t > b[-1][0] + 1e-6:
            return None
        for j in range(1, len(b)):
            ta, a = b[j - 1]
            tb, c = b[j]
            if ta <= t <= tb and tb > ta:
                if set(a) != set(c):
                    return None
                r = (t - ta) / (tb - ta)
                return [{"id": hid, "x": (1-r)*a[hid]["x"]+r*c[hid]["x"],
                         "y": (1-r)*a[hid]["y"]+r*c[hid]["y"],
                         "radius": c[hid]["radius"]} for hid in sorted(c)]
        return [dict(q) for _, q in sorted(b[-1][1].items())] if abs(t-b[-1][0])<1e-6 else None

    def _update_history(self, now):
        with self.gazebo_lock:
            if not self.gazebo_buffer:
                return False
            oldest, newest = self.gazebo_buffer[0][0], self.gazebo_buffer[-1][0]
        if self.next_sample_time is None:
            self.next_sample_time = oldest + self.dt_pred
        # If Gazebo has advanced faster than this node could process, do not
        # remain stuck requesting timestamps that have fallen out of the buffer.
        if self.next_sample_time < oldest - 1e-6:
            rospy.logwarn_throttle(2.0, "Gazebo buffer overrun: resynchronizing observation window")
            self.history.clear()
            self.active_ids = []
            self.next_sample_time = oldest + self.dt_pred
        updated = False
        while self.next_sample_time <= newest + 1e-9:
            t = self.next_sample_time
            people = self._gazebo_at(t)
            self.next_sample_time += self.dt_pred
            if not people:
                self.history.clear()
                self.active_ids = []
                continue
            ids = [int(p["id"]) for p in people]
            if self.active_ids and ids != self.active_ids:
                self.history.clear()
            self.active_ids = ids
            self.active_people = people
            self.history.append(self._positions_from_people(people))
            self.last_history_stamp = rospy.Time.from_sec(t)
            updated = True
        rospy.loginfo_throttle(
            2.0, "Prediction history: %d/%d | Gazebo span %.3fs | next sample %.3f | updated=%s",
            len(self.history), self.obs_len, max(0.0, newest-oldest),
            self.next_sample_time, str(updated))
        return updated and len(self.history) == self.obs_len

    def build_oracle_message(self, stamp):
        """Scripted-motion oracle; not measured future Gazebo ground truth."""
        arr = np.zeros((self.n_pred + 1, len(self.active_people), 2), dtype=np.float32)
        for j, p in enumerate(self.active_people):
            h = self.humans_by_id.get(p["id"])
            if h is None:
                raise RuntimeError("No scripted motion definition for human %s" % p["id"])
            clone = copy.deepcopy(h)
            clone.x, clone.y = float(p["x"]), float(p["y"])
            arr[0,j] = (clone.x, clone.y)
            # Small steps for accuracy at endpoint direction switches.
            substep = self.dt_pred / 40.0
            for k in range(1, self.n_pred + 1):
                for _ in range(40):
                    clone.step(substep)
                arr[k,j] = (clone.x, clone.y)
        return self._build_prediction_message_from_xy(arr, stamp)

    def _constant_velocity_prediction_from_history(self, people):
        num_people = len(people)
        out = np.zeros((self.n_pred + 1, num_people, 2), dtype=np.float32)

        if len(self.history) > 0:
            out[0] = self.history[-1]
        else:
            out[0] = self._positions_from_people(people)

        if len(self.history) >= 2:
            v = self.history[-1] - self.history[-2]
        else:
            v = np.zeros((num_people, 2), dtype=np.float32)

        for k in range(1, self.n_pred + 1):
            out[k] = out[k - 1] + v

        return out

    def _build_person_msg(self, human_index, pred_xy):
        person_info = self.active_people[human_index]
        person = PredictedPerson()
        person.id = int(person_info["id"])
        person.radius = float(person_info.get("radius", self.default_radius))

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

    def build_uncertainty_message(self, sigma_x, sigma_y, rho, stamp):
        msg = PredictedPeopleUncertainty()
        msg.header.stamp = stamp
        msg.header.frame_id = self.prediction_frame
        msg.dt = self.dt_pred
        msg.horizon = sigma_x.shape[0] - 1

        for i, person_info in enumerate(self.active_people):
            p = PredictedPersonUncertainty()
            p.id = int(person_info["id"])
            p.sigma_x = [float(v) for v in sigma_x[:, i]]
            p.sigma_y = [float(v) for v in sigma_y[:, i]]
            p.rho = [float(v) for v in rho[:, i]]
            msg.people.append(p)

        return msg

    def _build_prediction_message_from_xy(self, pred_xy, stamp):
        out = PredictedPeople()
        out.header.stamp = stamp
        out.header.frame_id = self.prediction_frame
        out.dt = self.dt_pred
        out.horizon = self.n_pred

        for i in range(len(self.active_people)):
            out.people.append(self._build_person_msg(i, pred_xy))

        return out

    def _marker_color_for_source(self):
        """Return RGB color for the currently selected human source."""
        if self.human_source == "gazebo_model_states":
            return 0.0, 1.0, 0.0
        if self.human_source == "gt":
            # Cyan: internal scripted ground truth
            return 0.0, 1.0, 1.0
        if self.human_source == "detection":
            # Blue: camera / YOLO detections
            return 0.0, 0.2, 1.0
        if self.human_source == "gazebo_topic":
            # Green: Gazebo GT received from topic
            return 0.0, 1.0, 0.0
        return 1.0, 1.0, 1.0

    def build_gt_human_array(self):
        msg = HumanArray()
        if hasattr(msg, "header"):
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.prediction_frame

        for human in self.humans:
            hmsg = Human()
            hmsg.id = int(human.id)
            hmsg.x = float(human.x)
            hmsg.y = float(human.y)
            if hasattr(hmsg, "radius"):
                hmsg.radius = float(human.radius)
            msg.humans.append(hmsg)
        return msg

    def publish_gt_human_array(self):
        self.gt_humans_pub.publish(self.build_gt_human_array())
        now_pub = rospy.Time.now()
        self.last_gt_publish_time = self._publish_rate_from_interval(
            self.gt_humans_publish_rate_pub, now_pub, self.last_gt_publish_time
        )

    def publish_detected_markers(self):
        """Publish latest detected HumanArray positions as separate RViz markers."""
        delete_marker = Marker()
        delete_marker.header.frame_id = self.marker_frame
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "detected_humans"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        self.detected_marker_pub.publish(delete_marker)

        if self.detected_humans_stamp is None or len(self.detected_humans) == 0:
            return

        if (rospy.Time.now() - self.detected_humans_stamp).to_sec() > self.external_human_timeout:
            return

        now = rospy.Time.now()
        for i, person in enumerate(self.detected_humans.values()):
            marker = Marker()
            marker.header.frame_id = self.marker_frame
            marker.header.stamp = now
            marker.ns = "detected_humans"
            marker.id = int(person.get("id", i))
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(person["x"])
            marker.pose.position.y = float(person["y"])
            marker.pose.position.z = 0.55
            marker.pose.orientation.w = 1.0
            radius = float(person.get("radius", self.default_radius))
            marker.scale.x = 2.0 * radius
            marker.scale.y = 2.0 * radius
            marker.scale.z = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.2
            marker.color.b = 1.0
            marker.color.a = 0.60
            self.detected_marker_pub.publish(marker)

    def publish_source_markers(self):
        """
        Publish RViz markers from the same source used for prediction.

        This avoids showing internal GT cylinders while the planner is actually
        using camera detections. The topic remains /human_obstacle_marker for
        compatibility with the existing RViz configuration.
        """
        # First clear old markers so disappeared detections do not remain in RViz.
        delete_marker = Marker()
        delete_marker.header.frame_id = self.marker_frame
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "selected_human_source"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        self.marker_pub.publish(delete_marker)

        if len(self.active_people) == 0:
            return

        r_col, g_col, b_col = self._marker_color_for_source()
        now = rospy.Time.now()

        for i, person in enumerate(self.active_people):
            marker = Marker()
            marker.header.frame_id = self.marker_frame
            marker.header.stamp = now
            marker.ns = "selected_human_source"
            marker.id = int(person.get("id", i))
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = float(person["x"])
            marker.pose.position.y = float(person["y"])
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0

            radius = float(person.get("radius", self.default_radius))
            marker.scale.x = 2.0 * radius
            marker.scale.y = 2.0 * radius
            marker.scale.z = 1.0

            marker.color.r = r_col
            marker.color.g = g_col
            marker.color.b = b_col
            marker.color.a = 0.85

            self.marker_pub.publish(marker)

    def publish_gt_markers(self):
        """
        Always publish internal scripted GT human markers on /human_gt_marker.

        These markers are independent of human_source. Therefore, when
        human_source='detection', RViz can show both:
          - /human_obstacle_marker : detections used by MPC
          - /human_gt_marker       : true scripted/Gazebo human positions
        """
        delete_marker = Marker()
        delete_marker.header.frame_id = self.marker_frame
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.ns = "gt_humans"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        self.gt_marker_pub.publish(delete_marker)

        now = rospy.Time.now()

        for human in self.humans:
            marker = Marker()
            marker.header.frame_id = self.marker_frame
            marker.header.stamp = now
            marker.ns = "gt_humans"
            marker.id = int(human.id)
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = float(human.x)
            marker.pose.position.y = float(human.y)
            marker.pose.position.z = 0.5
            marker.pose.orientation.w = 1.0

            marker.scale.x = 2.0 * float(human.radius)
            marker.scale.y = 2.0 * float(human.radius)
            marker.scale.z = 1.0

            # Green GT markers.
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.55

            self.gt_marker_pub.publish(marker)


    def publish_prediction_markers(self, pred_msg):
        """
        Publish RViz markers for the predicted human trajectories.

        The visualization is built directly from the same PredictedPeople
        message that is sent to the MPC. Therefore, it shows exactly the
        prediction used by the planner, independent of whether the observation
        source is GT, detection, or gazebo_topic.
        """
        marker_array = MarkerArray()
        now = rospy.Time.now()

        delete_marker = Marker()
        delete_marker.header.frame_id = pred_msg.header.frame_id or self.prediction_frame
        delete_marker.header.stamp = now
        delete_marker.ns = "predicted_people"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 1
        stride = max(1, int(self.pred_marker_stride))

        for person in pred_msg.people:
            if len(person.trajectory) == 0:
                continue

            # A line strip shows the full predicted path.
            line_marker = Marker()
            line_marker.header.frame_id = pred_msg.header.frame_id or self.prediction_frame
            line_marker.header.stamp = now
            line_marker.ns = "predicted_people"
            line_marker.id = marker_id
            marker_id += 1
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = 0.035
            line_marker.color.r = 1.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 0.9

            for pose in person.trajectory:
                pt = Point()
                pt.x = float(pose.x)
                pt.y = float(pose.y)
                pt.z = self.pred_marker_z
                line_marker.points.append(pt)

            marker_array.markers.append(line_marker)

            # Small spheres show discrete prediction samples along the path.
            radius = float(person.radius) if person.radius > 1e-6 else float(self.default_radius)
            for k, pose in enumerate(person.trajectory):
                if k % stride != 0:
                    continue

                sphere = Marker()
                sphere.header.frame_id = pred_msg.header.frame_id or self.prediction_frame
                sphere.header.stamp = now
                sphere.ns = "predicted_people"
                sphere.id = marker_id
                marker_id += 1
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose.position.x = float(pose.x)
                sphere.pose.position.y = float(pose.y)
                sphere.pose.position.z = self.pred_marker_z
                sphere.pose.orientation.w = 1.0

                # The current point is slightly larger; future samples are smaller
                # to keep the display readable when many people are visible.
                if k == 0:
                    scale = max(0.12, 0.70 * radius)
                    alpha = 0.95
                else:
                    scale = max(0.07, 0.45 * radius)
                    alpha = 0.55

                sphere.scale.x = scale
                sphere.scale.y = scale
                sphere.scale.z = scale
                sphere.color.r = 1.0
                sphere.color.g = 1.0
                sphere.color.b = 0.0
                sphere.color.a = alpha
                marker_array.markers.append(sphere)

        self.pred_marker_pub.publish(marker_array)

    def build_learned_prediction_message(self):
        if len(self.history) < self.obs_len:
            return None, None
        stamp = self.last_history_stamp
        obs_xy = np.stack(list(self.history), axis=0)
        # Do not silently substitute a constant-velocity model.
        if len(self.active_people) < 2:
            rospy.logwarn_throttle(2.0, "Social-STGCNN requires at least two people in this implementation")
            return None, None
        t0 = time.monotonic()
        pred_xy, sx, sy, rho = self.predictor.predict(obs_xy, self.n_pred)
        # Save exact prediction input/output WITHOUT affecting the model, MPC or evaluator.
        self.debug_model_io_calls += 1
        if (self.debug_save_model_io and
                self.debug_model_io_saved < self.debug_model_io_max_files and
                self.debug_model_io_calls % self.debug_model_io_every == 0):
            try:
                idx = self.debug_model_io_saved + 1
                path = os.path.join(self.debug_model_io_dir,
                    "stgcnn_%04d_t%010.3f.npz" % (idx, stamp.to_sec()))
                np.savez_compressed(
                    path,
                    obs_xy=obs_xy.copy(),          # [8, N, 2], exact predictor input
                    pred_xy=pred_xy.copy(),        # [13, N, 2], includes current pose
                    sigma_x=sx.copy(), sigma_y=sy.copy(), rho=rho.copy(),
                    person_ids=np.asarray(self.active_ids, dtype=np.int32),
                    t0_sim=np.float64(stamp.to_sec()),
                    prediction_dt=np.float64(self.dt_pred),
                    prediction_horizon=np.int32(self.n_pred),
                    model_checkpoint=np.asarray(str(getattr(self.predictor, 'checkpoint_path', ''))))
                self.debug_model_io_saved += 1
                rospy.loginfo("Saved exact Social-STGCNN IO [%d/%d]: %s",
                    self.debug_model_io_saved, self.debug_model_io_max_files, path)
            except Exception as err:
                rospy.logerr_throttle(2.0, "Unable to save Social-STGCNN IO: %s", str(err))
        self.prediction_processing_time_pub.publish(Float32(time.monotonic() - t0))
        return (self._build_prediction_message_from_xy(pred_xy, stamp),
                self.build_uncertainty_message(sx, sy, rho, stamp))

    def run(self):
        prev_time = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - prev_time).to_sec()
            prev_time = now

            if dt < 0.0:
                rospy.logwarn_throttle(1.0, "Negative dt detected. Skipping this cycle.")
                self.rate.sleep()
                continue

            # Move scripted Gazebo humans when requested. This is independent from the
            # prediction source: in detection mode, the camera may observe these same moving actors.
            if self.move_scripted_humans:
                try:
                    for human in self.humans:
                        vx, vy = human.step(dt)
                        self.set_state(human.build_model_state(vx, vy))
                        # Do not publish GT markers here. Human markers are published
                        # later from the selected source (GT / detection / gazebo_topic),
                        # so RViz always shows exactly what the predictor and MPC use.
                except rospy.ServiceException as e:
                    rospy.logwarn_throttle(1.0, "Failed to set model state: %s", str(e))
                except Exception as e:
                    rospy.logwarn_throttle(1.0, "Unexpected human motion error: %s", str(e))

            # Publish GT markers on a separate topic regardless of the selected source.
            # This is useful in detection mode to visually compare detection vs. ground truth.
            self.publish_gt_markers()
            self.publish_gt_human_array()
            self.publish_detected_markers()

            try:
                ok = self._update_history(now)
                if ok:
                    self.publish_source_markers()
                    learned_msg, unc_msg = self.build_learned_prediction_message()
                    if learned_msg is not None:
                        stamp = learned_msg.header.stamp
                        oracle_msg = self.build_oracle_message(stamp)
                        self.learned_pub.publish(learned_msg)
                        self.oracle_pub.publish(oracle_msg)
                        self.unc_pub.publish(unc_msg)
                        selected = learned_msg if self.prediction_source == "learned" else oracle_msg
                        self.pred_pub.publish(selected)
                        self.publish_prediction_markers(selected)
                        self.last_prediction_publish_time = self._publish_rate_from_interval(
                            self.prediction_publish_rate_pub, rospy.Time.now(), self.last_prediction_publish_time)
                        self.prediction_history_size_pub.publish(Int32(len(self.history)))
                        self.prediction_num_people_pub.publish(Int32(len(learned_msg.people)))
            except Exception as e:
                rospy.logerr_throttle(1.0, "Prediction invalid (not publishing stale/fallback): %s", str(e))

            self.rate.sleep()


if __name__ == "__main__":
    node = ScriptedMultiHumanMotionNode()
    node.run()
