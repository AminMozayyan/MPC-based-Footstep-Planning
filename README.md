# Surena-V Core ROS Workspace

This repository contains the **ROS1 (Noetic) workspace** for operating the humanoid robot **Surena-V**, the fifth generation of Iran’s national humanoid robotics project, developed at the **Center for Advanced Systems and Technologies (CAST), University of Tehran**.  

It provides stable packages for **control, electronics, simulation, planning, and teleoperation**, along with the necessary communication interfaces for integrating AI modules such as perception (object/face detection), sound source localization, and language/vision models.


## Quick Start

### Simulation

Open separate terminals, navigate to `~/surena_ws`, **source the workspace** in each one (`source devel/setup.bash`), and then run the following commands (one per terminal):

```bash
  roslaunch surenaV_gazebo surenaV_gazebo.launch    # Launch Gazebo  
  rosrun robot_manager robot_manager_node           # Start manager  
  rosservice call /keyboard_walk                    # Example: start walking  
  rosrun keyboard_teleop keyboard_teleop_node       # Start keyboard teleop node
```

### Real Robot
```bash
  rosrun surena_eth surena_eth                      # Start hardware interface  
  rosservice call /ActivateLegs "nodeID: 0"         # Initialize legs  
  rosservice call /ActivateHands "nodeID: 255"      # Initialize hands  
  rosrun robot_manager robot_manager_node           # Start manager  
```


## Workspace Overview

The workspace is organized into modular ROS packages:

- **gait_planner**  
  Core locomotion package. Generates trajectories (walk, step, COM, ankle, DCM), provides joint control, and exposes services for commanding the legs.

- **hand_planner**  
  Manages humanoid hand kinematics and planning (e.g., minimum jerk interpolation, grasping, head/hand movement). Provides services for moving single/both hands, online gripping, head tracking, and object-class–based actions.

- **keyboard_teleop**  
  Lightweight keyboard interface for triggering basic teleoperation commands such as walking.

- **robot_manager**  
  High-level scenario manager. Coordinates robot behaviors through YAML-defined scenarios and exposes the `/ExecuteScenario` service.
<!-- 
- **robot_teleop**  
  Qt-based GUI teleoperation panel for higher-level control and monitoring.

- **offlinePublisher**  
  Tools for replaying data offline, GUI for visualizing logged trajectories. -->

- **simulation/surenaV_description & simulation/surenaV_gazebo**  
  URDF description, Gazebo integration, and controllers for Surena-V. Allows full humanoid simulation with physics.

- **surena_eth**  
  Core hardware interface. Handles *EtherCAT/UDP communication* with the physical robot’s electronics (motors, sensors, F/T sensors). This package must run first on the real robot.


## Installation

### Prerequisites
- [ROS Noetic](http://wiki.ros.org/noetic/Installation) (Ubuntu 20.04 recommended)
- Dependencies:  
```bash
  sudo apt install ros-noetic-ros-control ros-noetic-ros-controllers \
                   libeigen3-dev libqt5core5a libqt5gui5 libqt5widgets5 \
                   build-essential cmake git 
```
  (plus OpenCV, Eigen, and other standard ROS dependencies)

### Build
```bash
  mkdir -p ~/surena_ws/src  
  cd ~/surena_ws/src  
  git clone https://github.com/CAST-Robotics/surenaV_core.git  
  cd ..  
  catkin_make  
  source devel/setup.bash  
```

## Usage

### 1. Simulation Mode (Gazebo)
1. Launch simulation:  
```bash
  roslaunch surenaV_gazebo surenaV_gazebo.launch  
```
   > In Gazebo, press the ▶ Play button to start physics.  

2. Start the robot manager:  
```bash
  rosrun robot_manager robot_manager_node  
```
3. Call services (walking, hand motions, head, etc.).  

   **Example**: make the robot walk  
```bash   
  rosservice call /keyboard_walk  
```

**Note**: To enable simulation mode in hand_planner, set the `simulation` flag at the top of HandManager.cpp.


### 2. Real Robot Mode
1. Start the hardware communication node:  
```bash
  rosrun surena_eth surena_eth  
```

2. Initialize hardware:  
```bash
  rosservice call /ResetAllNodes "nodeID: 255" 
  rosservice call /ActivateLegs "nodeID: 0"  
  rosservice call /ActivateHands "nodeID: 255"   
```

3. Run the robot manager:  
```bash
  rosrun robot_manager robot_manager_node  
```

4. Use services exactly as in simulation (see below).  



## Core Services

| Service | Type | Description |
|---------|------|-------------|
| `/walk_service` | `gait_planner/Trajectory` | Generate and execute walking gait with custom parameters (step length, width, COM height, etc.). |
| `/keyboard_walk` | `std_srvs/Empty` | Start walking sequence from keyboard teleop. |
| `/get_data` | `gait_planner/getdata` | Stream robot state data for a specified duration. |
| `/joint_command` | `gait_planner/command` | Direct joint/motor command (IDs + angles). |
| `/move_hand_single_srv` | `hand_planner/move_hand_single` | Execute hand motion scenario on left/right hand. |
| `/move_hand_both_srv` | `hand_planner/move_hand_both` | Execute coordinated bimanual motion. |
| `/grip_online_srv` | `hand_planner/gripOnline` | Start/finish online grasping controller. |
| `/head_track_srv` | `hand_planner/head_track` | Perform head tracking for a given duration. |
| `/set_target_class_srv` | `hand_planner/SetTargetClass` | Set object class for manipulation. |
| `/home_service` | `std_srvs/Empty` | Return robot to home posture. |


## Example Commands

- **Walking with specific parameters**  
```bash
  rosservice call /walk_service "alpha: 0.0  
  t_double_support: 0.2  
  t_step: 0.8  
  step_length: 0.2  
  step_width: 0.1  
  COM_height: 0.85  
  step_count: 4  
  ankle_height: 0.05  
  dt: 0.005  
  theta: 0.0  
  step_height: 0.04  
  com_offset: 0.0  
  is_config: false"  
```
- **Perform an specific scenario with right hand**  
```bash
  rosservice call /move_hand_single_srv "mode: 'righthand'  
  ee_ini_pos: 'init'  
  scenario: ['shakeHands']  
  scen_count: 1  
  t_total: 8"  
```
- **Move a servo joint (degrees | absolute)**  
```bash
  rosservice call /joint_command "motor_id: 21 angle: 170"  
```
- **Move a non-servo joint (radians | relative)**  
```bash
  rosservice call /joint_command "motor_id: 0 angle: 0.05"  
```
- **Echo force–torque sensor data**  
```bash
  rostopic echo /surena/ft_r_state  
```


## Capabilities

Surena-V integrates multiple **AI and perception modules**, including:  
- Object detection & tracking  
- Face recognition & tracking  
- Microphone array for sound localization  
- Large Language Models (LLM) & Vision-Language Models (VLM) for interactive tasks  



## Contributing

We welcome contributions, bug reports, and feature requests!

- Issues: Submit in GitHub Issues tab with logs and steps.  
- Pull Requests: Fork, improve, and submit a PR.  


## License
This repository is released under [MIT](https://github.com/CAST-Robotics/surenaV_core/blob/main/LICENSE) license.  
