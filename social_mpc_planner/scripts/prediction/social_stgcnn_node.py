#!/usr/bin/env python3

import os
import sys
import rospy
import numpy as np
import torch
from collections import defaultdict, deque

PKG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_PATH)

from social_stgcnn.model import social_stgcnn
from geometry_msgs.msg import Pose2D

from social_mpc_planner.msg import (
    HumanArray,
    SocialSTGCNNPersonPrediction,
    SocialSTGCNNPredictionArray
)


def anorm(p1, p2):
    n = np.linalg.norm(p1 - p2)
    if n == 0:
        return 0.0
    return 1.0 / n


def normalize_adjacency_matrix(A):
    Dl = np.sum(A, axis=0)
    num_node = A.shape[0]

    Dn = np.zeros((num_node, num_node), dtype=np.float32)

    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)

    AD = np.matmul(A, Dn)
    return np.matmul(Dn, AD)


def seq_to_graph(seq, seq_rel, norm_lap_matr=True):
    seq = seq.numpy()
    seq_rel = seq_rel.numpy()

    seq_len = seq.shape[2]
    max_nodes = seq.shape[0]

    V = np.zeros((seq_len, max_nodes, 2), dtype=np.float32)
    A = np.zeros((seq_len, max_nodes, max_nodes), dtype=np.float32)

    for s in range(seq_len):
        step_ = seq[:, :, s]
        step_rel = seq_rel[:, :, s]

        for h in range(max_nodes):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1.0

            for k in range(h + 1, max_nodes):
                l2_norm = anorm(step_[h], step_[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm

        if norm_lap_matr:
            A[s, :, :] = normalize_adjacency_matrix(A[s, :, :])

    return torch.from_numpy(V).float(), torch.from_numpy(A).float()


class SocialSTGCNNNode:

    def __init__(self):
        rospy.init_node("social_stgcnn_node")

        self.obs_len = rospy.get_param("~obs_len", 8)
        self.pred_len = rospy.get_param("~pred_len", 12)
        self.dt = rospy.get_param("~dt", 0.4)
        self.track_timeout = rospy.get_param("~track_timeout", 0.8)
        self.default_radius = rospy.get_param("~human_radius", 0.35)

        self.last_seen = {}
        self.history = defaultdict(lambda: deque(maxlen=self.obs_len))

        self.human_topic = rospy.get_param(
            "~human_topic",
            "/detected_humans_world"
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        rospy.loginfo(f"Using device: {self.device}")

        pkg_path = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )

        model_path = os.path.join(pkg_path, "models", "val_best.pth")
        rospy.loginfo(f"Loading model from: {model_path}")

        self.model = social_stgcnn(
            n_stgcnn=1,
            n_txpcnn=5,
            output_feat=5,
            seq_len=self.obs_len,
            kernel_size=3,
            pred_seq_len=self.pred_len
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        rospy.loginfo("Model loaded successfully.")

        self.pred_pub = rospy.Publisher(
            "/social_stgcnn/predictions",
            SocialSTGCNNPredictionArray,
            queue_size=1
        )

        rospy.Subscriber(
            self.human_topic,
            HumanArray,
            self.human_array_callback,
            queue_size=1
        )

        rospy.loginfo(f"Subscribed to HumanArray topic: {self.human_topic}")
        rospy.loginfo("Social-STGCNN node started.")

    def human_array_callback(self, msg):
        now = msg.header.stamp
        if now.to_sec() == 0.0:
            now = rospy.Time.now()

        current_ids = set()

        for human in msg.humans:
            human_id = int(human.id)
            current_ids.add(human_id)

            self.history[human_id].append(
                (float(human.x), float(human.y))
            )
            self.last_seen[human_id] = now

        valid_ids = []

        for human_id, hist in list(self.history.items()):
            if len(hist) < self.obs_len:
                continue

            if human_id not in self.last_seen:
                continue

            age = (now - self.last_seen[human_id]).to_sec()

            if age <= self.track_timeout:
                valid_ids.append(human_id)
            else:
                del self.history[human_id]
                del self.last_seen[human_id]

        if len(valid_ids) == 0:
            rospy.loginfo_throttle(
                1.0,
                "Waiting for enough human history..."
            )
            return

        valid_ids = sorted(valid_ids)

        obs_traj = np.zeros(
            (len(valid_ids), 2, self.obs_len),
            dtype=np.float32
        )

        is_stale = []
        ages = []

        for i, human_id in enumerate(valid_ids):
            age = (now - self.last_seen[human_id]).to_sec()

            is_stale.append(human_id not in current_ids)
            ages.append(age)

            for t, (x, y) in enumerate(self.history[human_id]):
                obs_traj[i, 0, t] = x
                obs_traj[i, 1, t] = y

        self.predict_from_observation(
            header=msg.header,
            dt=self.dt,
            agent_ids=valid_ids,
            obs_traj=obs_traj,
            is_stale=is_stale,
            ages=ages
        )

    def predict_from_observation(
        self,
        header,
        dt,
        agent_ids,
        obs_traj,
        is_stale,
        ages
    ):
        num_agents = len(agent_ids)

        if num_agents == 0:
            return

        obs_rel = np.zeros_like(obs_traj)
        obs_rel[:, :, 1:] = obs_traj[:, :, 1:] - obs_traj[:, :, :-1]

        V_obs, A_obs = seq_to_graph(
            torch.from_numpy(obs_traj).float(),
            torch.from_numpy(obs_rel).float(),
            norm_lap_matr=True
        )

        V_in = V_obs.unsqueeze(0).permute(0, 3, 1, 2)
        V_in = V_in.to(self.device)
        A_obs = A_obs.to(self.device)

        with torch.no_grad():
            pred, _ = self.model(V_in, A_obs)

        pred = pred[0].detach().cpu()

        mu_x_rel = pred[0]
        mu_y_rel = pred[1]
        sigma_x = torch.exp(pred[2])
        sigma_y = torch.exp(pred[3])
        rho = torch.tanh(pred[4])

        pred_rel = torch.stack(
            [mu_x_rel, mu_y_rel],
            dim=-1
        ).numpy()

        last_pos = obs_traj[:, :, -1]
        pred_abs = last_pos[None, :, :] + np.cumsum(pred_rel, axis=0)


        for i, agent_id in enumerate(agent_ids):
            obs_start = obs_traj[i, :, 0]
            obs_last = obs_traj[i, :, -1]
            obs_prev = obs_traj[i, :, -2]

            obs_window_disp = obs_last - obs_start
            obs_last_step = obs_last - obs_prev

            pred_first = pred_abs[0, i, :]
            pred_last = pred_abs[-1, i, :]

            pred_first_step = pred_first - obs_last
            pred_window_disp = pred_last - obs_last

            dot_first = np.dot(obs_last_step, pred_first_step)
            dot_window = np.dot(obs_window_disp, pred_window_disp)

            rospy.loginfo(
                f"[CHECK id={agent_id}] "
                f"obs_start=({obs_start[0]:.2f},{obs_start[1]:.2f}) "
                f"obs_last=({obs_last[0]:.2f},{obs_last[1]:.2f}) "
                f"obs_last_step=({obs_last_step[0]:.3f},{obs_last_step[1]:.3f}) "
                f"pred_first_step=({pred_first_step[0]:.3f},{pred_first_step[1]:.3f}) "
                f"pred_window_disp=({pred_window_disp[0]:.3f},{pred_window_disp[1]:.3f}) "
                f"dot_first={dot_first:.4f} "
                f"dot_window={dot_window:.4f}"
            )

        out = SocialSTGCNNPredictionArray()
        out.header = header
        out.dt = dt
        out.obs_len = self.obs_len
        out.pred_len = self.pred_len

        for i, agent_id in enumerate(agent_ids):
            person = SocialSTGCNNPersonPrediction()
            person.id = int(agent_id)
            person.radius = float(self.default_radius)
            person.is_stale = bool(is_stale[i])
            person.age = float(ages[i])

            for t in range(self.pred_len):
                pose = Pose2D()
                pose.x = float(pred_abs[t, i, 0])
                pose.y = float(pred_abs[t, i, 1])
                pose.theta = 0.0

                person.trajectory.append(pose)
                person.sigma_x.append(float(sigma_x[t, i]))
                person.sigma_y.append(float(sigma_y[t, i]))
                person.rho.append(float(rho[t, i]))

            out.people.append(person)

        self.pred_pub.publish(out)

        rospy.loginfo(
            f"Published STGCNN prediction: "
            f"people={len(out.people)}, "
            f"pred_len={self.pred_len}"
        )


if __name__ == "__main__":
    node = SocialSTGCNNNode()
    rospy.spin()