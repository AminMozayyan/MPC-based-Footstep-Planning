#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>

#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace gazebo
{

class SimpleHumanCrowdPlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override
  {
    world = _model->GetWorld();

    human_prefix = "human";

    xmin = -10.5;
    xmax =  10.5;
    ymin = -10.5;
    ymax =  10.5;

    v0 = 0.45;
    v_max = 0.9;
    tau = 0.7;

    A_human = 5.0;
    B_human = 0.6;
    R_human = 1.2;

    A_obs = 7.0;
    B_obs = 0.5;
    R_obs = 1.4;

    A_wall = 10.0;
    B_wall = 0.5;
    wall_margin = 1.2;

    tangential_gain = 0.5;

    wander_heading_noise = 0.35;
    heading_change_rate = 0.9;

    max_acc = 2.5;

    hard_human_radius = 0.75;
    hard_obs_radius = 0.9;

    wall_turn_cooldown_s = 0.8;
    wall_turn_active_only_if_outward = true;

    robot_name = "my_robot";
    A_robot = 6.0;
    B_robot = 0.6;
    R_robot = 1.5;
    hard_robot_radius = 0.9;

    if (_sdf)
    {
      if (_sdf->HasElement("human_prefix")) human_prefix = _sdf->Get<std::string>("human_prefix");

      if (_sdf->HasElement("xmin")) xmin = _sdf->Get<double>("xmin");
      if (_sdf->HasElement("xmax")) xmax = _sdf->Get<double>("xmax");
      if (_sdf->HasElement("ymin")) ymin = _sdf->Get<double>("ymin");
      if (_sdf->HasElement("ymax")) ymax = _sdf->Get<double>("ymax");

      if (_sdf->HasElement("v0")) v0 = _sdf->Get<double>("v0");
      if (_sdf->HasElement("v_max")) v_max = _sdf->Get<double>("v_max");
      if (_sdf->HasElement("tau")) tau = _sdf->Get<double>("tau");

      if (_sdf->HasElement("A_human")) A_human = _sdf->Get<double>("A_human");
      if (_sdf->HasElement("B_human")) B_human = _sdf->Get<double>("B_human");
      if (_sdf->HasElement("R_human")) R_human = _sdf->Get<double>("R_human");

      if (_sdf->HasElement("A_obs")) A_obs = _sdf->Get<double>("A_obs");
      if (_sdf->HasElement("B_obs")) B_obs = _sdf->Get<double>("B_obs");
      if (_sdf->HasElement("R_obs")) R_obs = _sdf->Get<double>("R_obs");

      if (_sdf->HasElement("A_wall")) A_wall = _sdf->Get<double>("A_wall");
      if (_sdf->HasElement("B_wall")) B_wall = _sdf->Get<double>("B_wall");
      if (_sdf->HasElement("wall_margin")) wall_margin = _sdf->Get<double>("wall_margin");

      if (_sdf->HasElement("tangential_gain")) tangential_gain = _sdf->Get<double>("tangential_gain");

      if (_sdf->HasElement("wander_heading_noise"))
        wander_heading_noise = _sdf->Get<double>("wander_heading_noise");

      if (_sdf->HasElement("heading_change_rate"))
        heading_change_rate = _sdf->Get<double>("heading_change_rate");

      if (_sdf->HasElement("max_acc")) max_acc = _sdf->Get<double>("max_acc");

      if (_sdf->HasElement("hard_human_radius"))
        hard_human_radius = _sdf->Get<double>("hard_human_radius");

      if (_sdf->HasElement("hard_obs_radius"))
        hard_obs_radius = _sdf->Get<double>("hard_obs_radius");

      if (_sdf->HasElement("robot_name")) robot_name = _sdf->Get<std::string>("robot_name");

      if (_sdf->HasElement("A_robot")) A_robot = _sdf->Get<double>("A_robot");
      if (_sdf->HasElement("B_robot")) B_robot = _sdf->Get<double>("B_robot");
      if (_sdf->HasElement("R_robot")) R_robot = _sdf->Get<double>("R_robot");

      if (_sdf->HasElement("hard_robot_radius"))
        hard_robot_radius = _sdf->Get<double>("hard_robot_radius");

      if (_sdf->HasElement("wall_turn_cooldown_s"))
        wall_turn_cooldown_s = _sdf->Get<double>("wall_turn_cooldown_s");
    }

    for (auto m : world->Models())
    {
      if (!m) continue;

      if (isHumanName(m->GetName()))
        humans.push_back(m);
      else
        obstacles.push_back(m);
    }

    states.resize(humans.size());

    for (auto &s : states)
    {
      s.desired_heading = uniform(-M_PI, M_PI);
      s.last_wall_turn_time = 0.0;
      s.last_vel = ignition::math::Vector3d::Zero;
    }

    last_time = world->SimTime();
    RefreshHumans();
    RefreshObstacles();

std::cout << "[HumanCrowd] Plugin loaded!" << std::endl;

    updateConn = event::Events::ConnectWorldUpdateBegin(
      std::bind(&SimpleHumanCrowdPlugin::OnUpdate, this)
    );

    std::cout << "[HumanCrowd] Plugin loaded!" << std::endl;
  }

private:
  struct HumanState
  {
    double desired_heading;
    double last_wall_turn_time;
    ignition::math::Vector3d last_vel;
  };

  void OnUpdate()
  {
    const common::Time now = world->SimTime();
    double dt = (now - last_time).Double();

    if (dt <= 0.0) return;

    last_time = now;
    static int refresh_counter = 0;

    if (++refresh_counter % 30 == 0)
    {
      RefreshHumans();
      RefreshObstacles();
    }

    physics::ModelPtr robot = world->ModelByName(robot_name);

    for (size_t i = 0; i < humans.size(); ++i)
    {
      auto me = humans[i];
      if (!me) continue;

      auto &state = states[i];

      ignition::math::Vector3d pos = me->WorldPose().Pos();
      pos.Z() = 0.0;

      ignition::math::Vector3d v = me->WorldLinearVel();
      v.Z() = 0.0;

      // =========================
      // 1) Natural wandering
      // =========================
      state.desired_heading += normal(0.0, wander_heading_noise) * dt;
      state.desired_heading = wrapPi(state.desired_heading);

      double drift = normal(0.0, 1.0) * heading_change_rate * dt * 0.2;
      state.desired_heading = wrapPi(state.desired_heading + drift);

      ignition::math::Vector3d e_des(
        std::cos(state.desired_heading),
        std::sin(state.desired_heading),
        0.0
      );

      ignition::math::Vector3d v_des = v0 * e_des;

      // =========================
      // 2) Social forces
      // =========================
      ignition::math::Vector3d F = ignition::math::Vector3d::Zero;

      // Avoid robot
      if (robot)
      {
        ignition::math::Vector3d rpos = robot->WorldPose().Pos();
        rpos.Z() = 0.0;

        F += repulsiveForce(pos, rpos, A_robot, B_robot, R_robot, tangential_gain);
      }

      // Avoid other humans
      for (size_t j = 0; j < humans.size(); ++j)
      {
        if (i == j) continue;

        auto other = humans[j];
        if (!other) continue;

        ignition::math::Vector3d opos = other->WorldPose().Pos();
        opos.Z() = 0.0;

        F += repulsiveForce(pos, opos, A_human, B_human, R_human, tangential_gain);
      }

      // Avoid static obstacles / walls / objects
      for (auto obs : obstacles)
      {
        if (!obs) continue;

        std::string name = obs->GetName();

        if (name == "ground_plane") continue;
        if (name == robot_name) continue;

        ignition::math::Vector3d opos = obs->WorldPose().Pos();
        opos.Z() = 0.0;

        F += repulsiveForce(pos, opos, A_obs, B_obs, R_obs, tangential_gain);
      }

      // =========================
      // 3) Rectangular boundary handling
      // =========================
      handleWallForce(pos, v, state, now.Double(), F);

      // =========================
      // 4) Relaxation model
      // =========================
      ignition::math::Vector3d a =
        (v_des - v) / std::max(1e-3, tau) + F;

      if (a.Length() > max_acc)
        a = a.Normalized() * max_acc;

      ignition::math::Vector3d v_cmd = v + a * dt;
      v_cmd.Z() = 0.0;

      if (v_cmd.Length() > v_max)
        v_cmd = v_cmd.Normalized() * v_max;

      // =========================
      // 5) Hard collision prevention
      // =========================
      applyHardSafety(i, pos, robot, v_cmd);

      // =========================
      // 6) Apply velocity and orientation
      // =========================
      double yaw = me->WorldPose().Rot().Yaw();

      if (v_cmd.Length() > 1e-3)
        yaw = std::atan2(v_cmd.Y(), v_cmd.X()) + M_PI_2;

      ignition::math::Pose3d new_pose = me->WorldPose();
      new_pose.Pos().Z() = me->WorldPose().Pos().Z();
      new_pose.Rot() = ignition::math::Quaterniond(0, 0, yaw);

      me->SetWorldPose(new_pose);
      me->SetLinearVel(v_cmd);
      me->SetAngularVel(ignition::math::Vector3d::Zero);

      state.last_vel = v_cmd;
    }
  }

  void handleWallForce(
    const ignition::math::Vector3d &pos,
    const ignition::math::Vector3d &v,
    HumanState &state,
    double tnow,
    ignition::math::Vector3d &F)
  {
    auto addWall = [&](double edge_dist,
                       const ignition::math::Vector3d &n_in,
                       const ignition::math::Vector3d &n_out)
    {
      if (edge_dist < wall_margin)
      {
        double d = std::max(1e-3, edge_dist);
        double mag = A_wall * std::exp((wall_margin - d) / std::max(1e-3, B_wall));

        F += mag * n_in;

        bool moving_outward = v.Dot(n_out) > 0.05;
        bool allowed = (tnow - state.last_wall_turn_time) > wall_turn_cooldown_s;

        bool should_turn =
          allowed && (!wall_turn_active_only_if_outward || moving_outward);

        if (should_turn)
        {
          double base_heading = state.desired_heading;

          if (v.Length() > 0.05)
            base_heading = std::atan2(v.Y(), v.X());

          double off = uniform(2.0 * M_PI / 3.0, 4.0 * M_PI / 3.0);

          state.desired_heading = wrapPi(base_heading + off);
          state.last_wall_turn_time = tnow;
        }
      }
    };

    addWall(pos.X() - xmin, ignition::math::Vector3d(1, 0, 0),
                             ignition::math::Vector3d(-1, 0, 0));

    addWall(xmax - pos.X(), ignition::math::Vector3d(-1, 0, 0),
                             ignition::math::Vector3d(1, 0, 0));

    addWall(pos.Y() - ymin, ignition::math::Vector3d(0, 1, 0),
                             ignition::math::Vector3d(0, -1, 0));

    addWall(ymax - pos.Y(), ignition::math::Vector3d(0, -1, 0),
                             ignition::math::Vector3d(0, 1, 0));
  }

  void applyHardSafety(
    size_t i,
    const ignition::math::Vector3d &pos,
    const physics::ModelPtr &robot,
    ignition::math::Vector3d &v_cmd)
  {
    if (robot)
    {
      ignition::math::Vector3d rpos = robot->WorldPose().Pos();
      rpos.Z() = 0.0;

      double d = (pos - rpos).Length();

      if (d < hard_robot_radius && d > 1e-6)
      {
        ignition::math::Vector3d n = (pos - rpos) / d;
        v_cmd = std::min(v_max, v0) * n;
        return;
      }
    }

    for (size_t j = 0; j < humans.size(); ++j)
    {
      if (i == j) continue;
      if (!humans[j]) continue;

      ignition::math::Vector3d opos = humans[j]->WorldPose().Pos();
      opos.Z() = 0.0;

      double d = (pos - opos).Length();

      if (d < hard_human_radius && d > 1e-6)
      {
        ignition::math::Vector3d n = (pos - opos) / d;
        v_cmd = std::min(v_max, v0) * n;
        return;
      }
    }

    for (auto obs : obstacles)
    {
      if (!obs) continue;

      std::string name = obs->GetName();

      if (name == "ground_plane") continue;
      if (name == robot_name) continue;

      ignition::math::Vector3d opos = obs->WorldPose().Pos();
      opos.Z() = 0.0;

      double d = (pos - opos).Length();

      if (d < hard_obs_radius && d > 1e-6)
      {
        ignition::math::Vector3d n = (pos - opos) / d;
        v_cmd = std::min(v_max, v0) * n;
        return;
      }
    }
  }

  ignition::math::Vector3d repulsiveForce(
    const ignition::math::Vector3d &self,
    const ignition::math::Vector3d &other,
    double A,
    double B,
    double R,
    double tangentialGain)
  {
    ignition::math::Vector3d d = self - other;
    d.Z() = 0.0;

    double dist = d.Length();

    if (dist < 1e-6)
      return ignition::math::Vector3d::Zero;

    ignition::math::Vector3d n = d / dist;

    double mag = A * std::exp((R - dist) / std::max(1e-3, B));

    ignition::math::Vector3d F = mag * n;

    ignition::math::Vector3d t(-n.Y(), n.X(), 0.0);

    double side =
      (self.X() * other.Y() - self.Y() * other.X()) >= 0.0 ? 1.0 : -1.0;

    F += tangentialGain * mag * side * t;

    return F;
  }

  bool isHumanName(const std::string &name) const
  {
    return name.find(human_prefix) != std::string::npos;
  }

  double wrapPi(double a) const
  {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a < -M_PI) a += 2.0 * M_PI;
    return a;
  }

  double uniform(double a, double b)
  {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(gen);
  }

  double normal(double mean, double stddev)
  {
    std::normal_distribution<double> dist(mean, stddev);
    return dist(gen);
  }

private:
  physics::WorldPtr world;
  event::ConnectionPtr updateConn;
  common::Time last_time;

  std::vector<physics::ModelPtr> humans;
  std::vector<physics::ModelPtr> obstacles;
  std::vector<HumanState> states;

  std::string human_prefix;
  std::string robot_name;

  double xmin, xmax, ymin, ymax;

  double v0;
  double v_max;
  double tau;

  double A_human, B_human, R_human;
  double A_obs, B_obs, R_obs;
  double A_robot, B_robot, R_robot;

  double A_wall, B_wall;
  double wall_margin;

  double tangential_gain;

  double wander_heading_noise;
  double heading_change_rate;

  double max_acc;

  double hard_human_radius;
  double hard_obs_radius;
  double hard_robot_radius;

  double wall_turn_cooldown_s;
  bool wall_turn_active_only_if_outward;

  std::default_random_engine gen;
  
  void RefreshHumans()
{
  for (auto m : world->Models())
  {
    if (!m) continue;
    if (!isHumanName(m->GetName())) continue;

    bool exists = false;

    for (auto h : humans)
    {
      if (h && h->GetName() == m->GetName())
      {
        exists = true;
        break;
      }
    }

    if (!exists)
    {
      std::cout << "[HumanCrowd] Found human: "
                << m->GetName() << std::endl;

      humans.push_back(m);

      HumanState s;
      s.desired_heading = uniform(-M_PI, M_PI);
      s.last_wall_turn_time = 0.0;
      s.last_vel = ignition::math::Vector3d::Zero;

      states.push_back(s);
    }
  }
}

bool isRealObstacleName(const std::string &name) const
{
  if (name.find("wall")   != std::string::npos) return true;
  if (name.find("shelf")  != std::string::npos) return true;
  if (name.find("desk")   != std::string::npos) return true;
  if (name.find("box")    != std::string::npos) return true;
  if (name.find("bucket") != std::string::npos) return true;
  if (name.find("trash")  != std::string::npos) return true;
  if (name.find("pallet") != std::string::npos) return true;
  if (name.find("surenav") != std::string::npos) return true;

  return false;
}

void RefreshObstacles()
{
  obstacles.clear();

  for (auto m : world->Models())
  {
    if (!m) continue;

    std::string name = m->GetName();

    if (isHumanName(name)) continue;
    if (!isRealObstacleName(name)) continue;

    obstacles.push_back(m);
  }
}

};

GZ_REGISTER_MODEL_PLUGIN(SimpleHumanCrowdPlugin)

}