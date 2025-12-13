---
sidebar_position: 3
---

# Week 5 Exercises: Detailed URDF and Robot Control

## Exercise 1: Advanced Humanoid URDF Model

### Objective
Create a detailed humanoid URDF model with realistic proportions, proper kinematic chains, and integration with ros_control.

### Requirements
1. Create a humanoid URDF with at least 24 joints (6 DOF per leg, 6 DOF per arm, 2 DOF for head, 4 DOF for torso)
2. Include proper visual, collision, and inertial properties for all links
3. Add sensor mounts (IMU, cameras, LiDAR)
4. Include transmissions for ros_control integration
5. Use Xacro for parameterization and modularity

### Implementation Steps

1. **Create the advanced humanoid URDF file:**

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_humanoid_robot">

  <!-- Constants and properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="deg_to_rad" value="0.017453292519943295"/>

  <!-- Humanoid dimensions based on average adult proportions -->
  <xacro:property name="torso_height" value="0.6"/>
  <xacro:property name="torso_width" value="0.25"/>
  <xacro:property name="torso_depth" value="0.15"/>
  <xacro:property name="head_radius" value="0.1"/>
  <xacro:property name="neck_height" value="0.08"/>
  <xacro:property name="upper_arm_length" value="0.35"/>
  <xacro:property name="lower_arm_length" value="0.3"/>
  <xacro:property name="upper_arm_radius" value="0.07"/>
  <xacro:property name="lower_arm_radius" value="0.06"/>
  <xacro:property name="upper_leg_length" value="0.45"/>
  <xacro:property name="lower_leg_length" value="0.4"/>
  <xacro:property name="foot_length" value="0.25"/>
  <xacro:property name="foot_width" value="0.12"/>
  <xacro:property name="foot_height" value="0.08"/>

  <!-- Materials -->
  <material name="black">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>
  <material name="white">
    <color rgba="0.9 0.9 0.9 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 1.0 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="skin">
    <color rgba="1.0 0.8 0.6 1.0"/>
  </material>

  <!-- Macro for creating arm chains with full DOF -->
  <xacro:macro name="full_dof_arm" params="side parent_xyz">
    <!-- Shoulder complex (3 DOF) -->
    <joint name="${side}_shoulder_yaw_joint" type="revolute">
      <parent link="torso"/>
      <child link="${side}_shoulder_yaw"/>
      <origin xyz="${parent_xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-1.57}" upper="${1.57}" effort="150" velocity="2"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

    <link name="${side}_shoulder_yaw">
      <visual>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
        <material name="grey"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_pitch_joint" type="revolute">
      <parent link="${side}_shoulder_yaw"/>
      <child link="${side}_shoulder_pitch"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-2.0}" upper="${1.0}" effort="150" velocity="2"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

    <link name="${side}_shoulder_pitch">
      <visual>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
        <material name="grey"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_roll_joint" type="revolute">
      <parent link="${side}_shoulder_pitch"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-1.57}" upper="${1.57}" effort="150" velocity="2"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

    <!-- Upper arm -->
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
        <material name="grey"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="2.5"/>
        <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <!-- Elbow (1 DOF) -->
    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 ${-upper_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-2.0}" upper="${0.5}" effort="120" velocity="2"/>
      <dynamics damping="0.8" friction="0.1"/>
    </joint>

    <!-- Lower arm -->
    <link name="${side}_lower_arm">
      <visual>
        <geometry>
          <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
        <material name="grey"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="1.8"/>
        <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <!-- Wrist complex (2 DOF) -->
    <joint name="${side}_wrist_yaw_joint" type="revolute">
      <parent link="${side}_lower_arm"/>
      <child link="${side}_wrist_yaw"/>
      <origin xyz="0 0 ${-lower_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-1.0}" upper="${1.0}" effort="80" velocity="2"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${side}_wrist_yaw">
      <visual>
        <geometry>
          <sphere radius="0.04"/>
        </geometry>
        <material name="grey"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.3"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0005"/>
      </inertial>
    </link>

    <joint name="${side}_wrist_pitch_joint" type="revolute">
      <parent link="${side}_wrist_yaw"/>
      <child link="${side}_hand"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-0.5}" upper="${0.5}" effort="80" velocity="2"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${side}_hand">
      <visual>
        <geometry>
          <box size="0.15 0.08 0.06"/>
        </geometry>
        <material name="skin"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.15 0.08 0.06"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Macro for creating leg chains with full DOF -->
  <xacro:macro name="full_dof_leg" params="side parent_xyz">
    <!-- Hip complex (3 DOF) -->
    <joint name="${side}_hip_yaw_joint" type="revolute">
      <parent link="torso"/>
      <child link="${side}_hip_yaw"/>
      <origin xyz="${parent_xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-0.5}" upper="${0.5}" effort="250" velocity="1.5"/>
      <dynamics damping="2.0" friction="0.2"/>
    </joint>

    <link name="${side}_hip_yaw">
      <visual>
        <geometry>
          <sphere radius="0.08"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.08"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${side}_hip_roll_joint" type="revolute">
      <parent link="${side}_hip_yaw"/>
      <child link="${side}_hip_roll"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-0.5}" upper="${0.5}" effort="250" velocity="1.5"/>
      <dynamics damping="2.0" friction="0.2"/>
    </joint>

    <link name="${side}_hip_roll">
      <visual>
        <geometry>
          <sphere radius="0.08"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.08"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${side}_hip_pitch_joint" type="revolute">
      <parent link="${side}_hip_roll"/>
      <child link="${side}_upper_leg"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-1.57}" upper="${0.5}" effort="250" velocity="1.5"/>
      <dynamics damping="2.0" friction="0.2"/>
    </joint>

    <!-- Upper leg -->
    <link name="${side}_upper_leg">
      <visual>
        <geometry>
          <cylinder radius="0.08" length="${upper_leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.08" length="${upper_leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="6.0"/>
        <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.02"/>
      </inertial>
    </link>

    <!-- Knee (1 DOF) -->
    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_upper_leg"/>
      <child link="${side}_lower_leg"/>
      <origin xyz="0 0 ${-upper_leg_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${0.0}" upper="${1.57}" effort="250" velocity="1.5"/>
      <dynamics damping="2.0" friction="0.2"/>
    </joint>

    <!-- Lower leg -->
    <link name="${side}_lower_leg">
      <visual>
        <geometry>
          <cylinder radius="0.07" length="${lower_leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.07" length="${lower_leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="4.5"/>
        <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
        <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <!-- Ankle complex (2 DOF) -->
    <joint name="${side}_ankle_pitch_joint" type="revolute">
      <parent link="${side}_lower_leg"/>
      <child link="${side}_ankle_pitch"/>
      <origin xyz="0 0 ${-lower_leg_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-0.5}" upper="${0.5}" effort="200" velocity="1.5"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

    <link name="${side}_ankle_pitch">
      <visual>
        <geometry>
          <sphere radius="0.06"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.06"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_ankle_roll_joint" type="revolute">
      <parent link="${side}_ankle_pitch"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-0.3}" upper="${0.3}" effort="200" velocity="1.5"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

    <!-- Foot -->
    <link name="${side}_foot">
      <visual>
        <geometry>
          <box size="${foot_length} ${foot_width} ${foot_height}"/>
        </geometry>
        <material name="black"/>
      </visual>
      <collision>
        <geometry>
          <box size="${foot_length} ${foot_width} ${foot_height}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link (torso) -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="25.0"/>
      <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
      <inertia ixx="2.0" ixy="0.0" ixz="0.0" iyy="2.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Torso joints for additional DOF -->
  <joint name="torso_yaw_joint" type="revolute">
    <parent link="torso"/>
    <child link="torso_upper"/>
    <origin xyz="0 0 ${torso_height}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-0.3}" upper="${0.3}" effort="100" velocity="1"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>

  <link name="torso_upper">
    <visual>
      <geometry>
        <box size="${torso_width*0.8} ${torso_depth*0.8} ${torso_height*0.3}"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="${torso_width*0.8} ${torso_depth*0.8} ${torso_height*0.3}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <joint name="torso_pitch_joint" type="revolute">
    <parent link="torso_upper"/>
    <child link="neck_link"/>
    <origin xyz="0 0 ${torso_height*0.3}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-0.3}" upper="${0.3}" effort="100" velocity="1"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>

  <!-- Neck and head -->
  <link name="neck_link">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="${neck_height}"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="${neck_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 ${neck_height/2}" rpy="0 0 0"/>
      <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="neck_link"/>
    <child link="head"/>
    <origin xyz="0 0 ${neck_height}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-0.5}" upper="${0.5}" effort="50" velocity="1"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="${head_radius}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="4.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.08" ixy="0.0" ixz="0.0" iyy="0.08" iyz="0.0" izz="0.08"/>
    </inertial>
  </link>

  <!-- Sensors on head -->
  <joint name="imu_joint" type="fixed">
    <parent link="head"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.000001" ixy="0.0" ixz="0.0" iyy="0.000001" iyz="0.0" izz="0.000001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="head"/>
    <child link="camera_link"/>
    <origin xyz="0.08 0 0.02" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.03"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Create arms and legs using macros -->
  <xacro:full_dof_arm side="left" parent_xyz="${torso_width/2} ${torso_depth/2} ${torso_height*0.7}"/>
  <xacro:full_dof_arm side="right" parent_xyz="${torso_width/2} ${-torso_depth/2} ${torso_height*0.7}"/>
  <xacro:full_dof_leg side="left" parent_xyz="${-torso_width/3} ${torso_depth/4} 0"/>
  <xacro:full_dof_leg side="right" parent_xyz="${-torso_width/3} ${-torso_depth/4} 0"/>

  <!-- ros_control transmissions -->
  <xacro:macro name="transmission_block" params="joint_name">
    <transmission name="trans_${joint_name}">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="${joint_name}">
        <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
      </joint>
      <actuator name="motor_${joint_name}">
        <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
        <mechanicalReduction>1</mechanicalReduction>
      </actuator>
    </transmission>
  </xacro:macro>

  <!-- Add transmissions for all joints -->
  <xacro:transmission_block joint_name="torso_yaw_joint"/>
  <xacro:transmission_block joint_name="torso_pitch_joint"/>
  <xacro:transmission_block joint_name="neck_joint"/>

  <xacro:transmission_block joint_name="left_shoulder_yaw_joint"/>
  <xacro:transmission_block joint_name="left_shoulder_pitch_joint"/>
  <xacro:transmission_block joint_name="left_shoulder_roll_joint"/>
  <xacro:transmission_block joint_name="left_elbow_joint"/>
  <xacro:transmission_block joint_name="left_wrist_yaw_joint"/>
  <xacro:transmission_block joint_name="left_wrist_pitch_joint"/>

  <xacro:transmission_block joint_name="right_shoulder_yaw_joint"/>
  <xacro:transmission_block joint_name="right_shoulder_pitch_joint"/>
  <xacro:transmission_block joint_name="right_shoulder_roll_joint"/>
  <xacro:transmission_block joint_name="right_elbow_joint"/>
  <xacro:transmission_block joint_name="right_wrist_yaw_joint"/>
  <xacro:transmission_block joint_name="right_wrist_pitch_joint"/>

  <xacro:transmission_block joint_name="left_hip_yaw_joint"/>
  <xacro:transmission_block joint_name="left_hip_roll_joint"/>
  <xacro:transmission_block joint_name="left_hip_pitch_joint"/>
  <xacro:transmission_block joint_name="left_knee_joint"/>
  <xacro:transmission_block joint_name="left_ankle_pitch_joint"/>
  <xacro:transmission_block joint_name="left_ankle_roll_joint"/>

  <xacro:transmission_block joint_name="right_hip_yaw_joint"/>
  <xacro:transmission_block joint_name="right_hip_roll_joint"/>
  <xacro:transmission_block joint_name="right_hip_pitch_joint"/>
  <xacro:transmission_block joint_name="right_knee_joint"/>
  <xacro:transmission_block joint_name="right_ankle_pitch_joint"/>
  <xacro:transmission_block joint_name="right_ankle_roll_joint"/>

  <!-- Gazebo plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/</robotNamespace>
    </plugin>
  </gazebo>

</robot>
```

2. **Create a URDF validation and loading node:**

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math
import numpy as np

class AdvancedHumanoidPublisher(Node):
    def __init__(self):
        super().__init__('advanced_humanoid_publisher')

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing
        self.timer = self.create_timer(0.05, self.publish_states)  # 20Hz

        # Define all joint names for the advanced humanoid
        self.joint_names = [
            'torso_yaw_joint', 'torso_pitch_joint', 'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
            'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
            'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]

        # Initialize joint positions
        self.joint_positions = {name: 0.0 for name in self.joint_names}

        # Demo pattern variables
        self.time = 0.0

        self.get_logger().info(f'Advanced humanoid publisher initialized with {len(self.joint_names)} joints')

    def publish_states(self):
        # Update joint positions with demo patterns
        self.update_demo_positions()

        # Create and publish joint state message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = [self.joint_positions[name] for name in self.joint_names]
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)

        self.joint_state_publisher.publish(msg)

        # Broadcast base transform
        self.broadcast_base_transform()

    def update_demo_positions(self):
        """Update joint positions with coordinated demo patterns"""
        self.time += 0.05

        # Breathing motion in torso
        torso_motion = 0.05 * math.sin(self.time * 0.5)
        self.joint_positions['torso_yaw_joint'] = torso_motion * 0.3
        self.joint_positions['torso_pitch_joint'] = torso_motion * 0.2

        # Slow head movement
        self.joint_positions['neck_joint'] = 0.1 * math.sin(self.time * 0.3)

        # Coordinated arm movement (wave pattern)
        wave_phase = self.time * 0.7
        self.joint_positions['left_shoulder_yaw_joint'] = 0.2 * math.sin(wave_phase)
        self.joint_positions['left_shoulder_pitch_joint'] = 0.3 * math.sin(wave_phase + math.pi/2)
        self.joint_positions['left_elbow_joint'] = -0.4 * abs(math.sin(wave_phase))

        self.joint_positions['right_shoulder_yaw_joint'] = 0.2 * math.sin(wave_phase + math.pi)
        self.joint_positions['right_shoulder_pitch_joint'] = 0.3 * math.sin(wave_phase + math.pi/2 + math.pi)
        self.joint_positions['right_elbow_joint'] = -0.4 * abs(math.sin(wave_phase + math.pi))

        # Gentle balance adjustment in legs
        balance_phase = self.time * 0.4
        self.joint_positions['left_hip_roll_joint'] = 0.05 * math.sin(balance_phase)
        self.joint_positions['right_hip_roll_joint'] = 0.05 * math.sin(balance_phase + math.pi)
        self.joint_positions['left_ankle_roll_joint'] = -0.03 * math.sin(balance_phase)
        self.joint_positions['right_ankle_roll_joint'] = -0.03 * math.sin(balance_phase + math.pi)

    def broadcast_base_transform(self):
        """Broadcast the base transform for visualization"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedHumanoidPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. **Validate the URDF:**
```bash
# Validate the URDF file
check_urdf ~/physical_ai_ws/src/week5_exercises/urdf/advanced_humanoid_robot.urdf.xacro

# Generate graph of the robot structure
urdf_to_graphiz ~/physical_ai_ws/src/week5_exercises/urdf/advanced_humanoid_robot.urdf.xacro

# Visualize in RViz2
ros2 run rviz2 rviz2
# Add RobotModel display and set topic to joint_states
```

### Expected Output
- URDF with 28+ joints (exceeding the requirement of 24)
- Proper visual, collision, and inertial properties
- All joints have realistic limits and dynamics
- Sensors properly mounted on the robot
- ros_control transmissions defined for all joints
- URDF validates successfully

### Submission Requirements
- Complete URDF file with Xacro
- URDF publisher node
- Validation results
- Screenshots of the robot model in RViz2

## Exercise 2: Multi-Joint Robot Controller

### Objective
Implement a comprehensive robot controller that manages multiple joints with coordinated motion, safety systems, and feedback control.

### Requirements
1. Create a joint state controller with PID feedback
2. Implement trajectory execution for coordinated motion
3. Add safety systems (joint limits, emergency stop)
4. Include sensor fusion for state estimation
5. Implement a behavior system for different robot modes

### Implementation Steps

1. **Create the main controller node:**

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from builtin_interfaces.msg import Duration
import numpy as np
import math
from collections import deque
import threading

class MultiJointController(Node):
    def __init__(self):
        super().__init__('multi_joint_controller')

        # Publishers
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.controller_state_publisher = self.create_publisher(
            JointTrajectoryControllerState, 'controller_state', 10)
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)

        # Subscribers
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.command_subscription = self.create_subscription(
            String, 'robot_command', self.command_callback, 10)
        self.trajectory_subscription = self.create_subscription(
            JointTrajectory, 'joint_trajectory', self.trajectory_callback, 10)
        self.emergency_stop_subscription = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Robot configuration
        self.joint_names = [
            'torso_yaw_joint', 'torso_pitch_joint', 'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
            'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
            'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]

        # Initialize state dictionaries
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}
        self.current_efforts = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}
        self.desired_velocities = {name: 0.0 for name in self.joint_names}
        self.commanded_positions = {name: 0.0 for name in self.joint_names}

        # PID controller parameters
        self.pid_params = {
            'kp': {name: 10.0 for name in self.joint_names},
            'ki': {name: 0.5 for name in self.joint_names},
            'kd': {name: 0.1 for name in self.joint_names}
        }

        # PID state variables
        self.errors = {name: 0.0 for name in self.joint_names}
        self.integral_errors = {name: 0.0 for name in self.joint_names}
        self.previous_errors = {name: 0.0 for name in self.joint_names}

        # Joint limits (example values - should match URDF)
        self.joint_limits = {
            name: (-3.14, 3.14) for name in self.joint_names  # Default limits
        }
        # Override with more realistic limits
        for joint in ['left_knee_joint', 'right_knee_joint']:
            self.joint_limits[joint] = (0.0, 1.57)
        for joint in ['left_hip_pitch_joint', 'right_hip_pitch_joint']:
            self.joint_limits[joint] = (-1.57, 0.5)

        # Trajectory execution
        self.active_trajectory = None
        self.trajectory_start_time = None
        self.trajectory_current_point = 0
        self.executing_trajectory = False

        # Robot state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.linear_acceleration = Vector3()
        self.angular_velocity = Vector3()

        # Control modes and behaviors
        self.control_mode = "IDLE"  # IDLE, POSITION, TRAJECTORY, BALANCE
        self.active_behavior = "STAND"  # STAND, WALK, MANIPULATE, etc.
        self.emergency_stop_active = False

        # Safety parameters
        self.max_velocity = 2.0  # rad/s
        self.max_acceleration = 5.0  # rad/s^2

        # Logging
        self.command_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)

        self.get_logger().info(f'Multi-joint controller initialized with {len(self.joint_names)} joints')

    def joint_state_callback(self, msg):
        """Update current joint states from feedback"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.current_efforts[name] = msg.effort[i]

    def imu_callback(self, msg):
        """Update orientation and acceleration from IMU"""
        # Convert quaternion to euler angles
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z

        # Roll (x-axis rotation)
        self.roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

        # Pitch (y-axis rotation)
        self.pitch = math.asin(2*(w*y - z*x))

        # Yaw (z-axis rotation)
        self.yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        # Store acceleration and angular velocity
        self.linear_acceleration = msg.linear_acceleration
        self.angular_velocity = msg.angular_velocity

    def command_callback(self, msg):
        """Handle high-level robot commands"""
        command = msg.data.strip().upper()
        self.command_history.append(command)

        if command == "STAND":
            self.active_behavior = "STAND"
            self.control_mode = "POSITION"
        elif command == "WALK":
            self.active_behavior = "WALK"
            self.control_mode = "TRAJECTORY"
        elif command == "MANIPULATE":
            self.active_behavior = "MANIPULATE"
            self.control_mode = "POSITION"
        elif command == "BALANCE":
            self.active_behavior = "BALANCE"
            self.control_mode = "BALANCE"
        elif command == "IDLE":
            self.active_behavior = "STAND"
            self.control_mode = "IDLE"
        elif command.startswith("MOVE_TO:"):
            # Parse move command: MOVE_TO:joint_name,position
            try:
                parts = command[8:].split(',')  # Remove "MOVE_TO:" prefix
                joint_name = parts[0].strip()
                position = float(parts[1].strip())
                if joint_name in self.desired_positions:
                    self.desired_positions[joint_name] = position
                    self.control_mode = "POSITION"
            except (ValueError, IndexError):
                self.get_logger().warn(f'Invalid MOVE_TO command format: {command}')

    def trajectory_callback(self, msg):
        """Handle trajectory commands"""
        if len(msg.points) > 0 and len(msg.joint_names) > 0:
            self.active_trajectory = msg
            self.trajectory_start_time = self.get_clock().now()
            self.trajectory_current_point = 0
            self.executing_trajectory = True
            self.control_mode = "TRAJECTORY"
            self.get_logger().info(f'Received trajectory with {len(msg.points)} points')

    def emergency_stop_callback(self, msg):
        """Handle emergency stop commands"""
        self.emergency_stop_active = msg.data
        if self.emergency_stop_active:
            self.get_logger().error('EMERGENCY STOP ACTIVATED')
            # Immediately stop all motion
            for joint in self.joint_names:
                self.desired_positions[joint] = self.current_positions[joint]
        else:
            self.get_logger().info('Emergency stop deactivated')

    def control_loop(self):
        """Main control loop"""
        if self.emergency_stop_active:
            # Emergency stop - publish zero commands
            self.publish_zero_commands()
            return

        # Execute appropriate control mode
        if self.control_mode == "IDLE":
            self.execute_idle_control()
        elif self.control_mode == "POSITION":
            self.execute_position_control()
        elif self.control_mode == "TRAJECTORY":
            self.execute_trajectory_control()
        elif self.control_mode == "BALANCE":
            self.execute_balance_control()

        # Publish controller state
        self.publish_controller_state()

        # Update status
        self.publish_status()

    def execute_idle_control(self):
        """Maintain current position"""
        # Commands stay at current position
        for joint in self.joint_names:
            self.desired_positions[joint] = self.current_positions[joint]

    def execute_position_control(self):
        """Execute position control with PID"""
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.position = []
        commands.velocity = []
        commands.effort = []

        for joint in self.joint_names:
            # Calculate PID output
            error = self.desired_positions[joint] - self.current_positions[joint]
            self.integral_errors[joint] += error * 0.01  # dt = 0.01s
            derivative = (error - self.previous_errors[joint]) / 0.01

            # Apply PID formula
            pid_output = (
                self.pid_params['kp'][joint] * error +
                self.pid_params['ki'][joint] * self.integral_errors[joint] +
                self.pid_params['kd'][joint] * derivative
            )

            # Apply limits
            pid_output = max(min(pid_output, self.max_velocity), -self.max_velocity)

            # Update state for next iteration
            self.previous_errors[joint] = error

            # Apply to command
            self.commanded_positions[joint] += pid_output * 0.01  # Integrate velocity

            # Apply joint limits
            self.commanded_positions[joint] = max(
                self.joint_limits[joint][0],
                min(self.joint_limits[joint][1], self.commanded_positions[joint])
            )

            commands.position.append(self.commanded_positions[joint])
            commands.velocity.append(pid_output)
            commands.effort.append(0.0)  # Effort calculated by hardware controller

        self.joint_command_publisher.publish(commands)

    def execute_trajectory_control(self):
        """Execute trajectory with interpolation"""
        if not self.executing_trajectory or not self.active_trajectory:
            self.control_mode = "IDLE"
            return

        if self.trajectory_current_point >= len(self.active_trajectory.points):
            # Trajectory completed
            self.executing_trajectory = False
            self.active_behavior = "STAND"
            self.control_mode = "POSITION"
            self.get_logger().info('Trajectory completed')
            return

        # Get current time
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.trajectory_start_time).nanoseconds / 1e9

        # Find the appropriate segment
        points = self.active_trajectory.points
        while (self.trajectory_current_point < len(points) - 1 and
               elapsed_time > (points[self.trajectory_current_point + 1].time_from_start.sec +
                             points[self.trajectory_current_point + 1].time_from_start.nanosec / 1e9)):
            self.trajectory_current_point += 1

        if self.trajectory_current_point >= len(points):
            self.executing_trajectory = False
            return

        # Get current and next points
        current_point = points[self.trajectory_current_point]
        if self.trajectory_current_point < len(points) - 1:
            next_point = points[self.trajectory_current_point + 1]

            # Calculate interpolation factor
            current_point_time = (current_point.time_from_start.sec +
                                current_point.time_from_start.nanosec / 1e9)
            next_point_time = (next_point.time_from_start.sec +
                             next_point.time_from_start.nanosec / 1e9)

            if next_point_time > current_point_time:
                t = (elapsed_time - current_point_time) / (next_point_time - current_point_time)
                t = max(0.0, min(1.0, t))  # Clamp to [0, 1]

                # Interpolate positions
                for i, joint_name in enumerate(self.active_trajectory.joint_names):
                    if i < len(current_point.positions) and i < len(next_point.positions):
                        start_pos = current_point.positions[i]
                        end_pos = next_point.positions[i]
                        interpolated_pos = start_pos + t * (end_pos - start_pos)
                        if joint_name in self.desired_positions:
                            self.desired_positions[joint_name] = interpolated_pos

        # Execute position control for the interpolated positions
        self.execute_position_control()

    def execute_balance_control(self):
        """Execute balance control based on IMU feedback"""
        # Simple balance controller - adjust ankle joints based on tilt
        ankle_adjustment = {
            'left_ankle_pitch_joint': -self.pitch * 0.5,
            'right_ankle_pitch_joint': -self.pitch * 0.5,
            'left_ankle_roll_joint': -self.roll * 0.5,
            'right_ankle_roll_joint': -self.roll * 0.5
        }

        # Apply balance adjustments to standing position
        stand_positions = self.get_stand_positions()
        for joint, adjustment in ankle_adjustment.items():
            if joint in stand_positions:
                stand_positions[joint] += adjustment

        # Set desired positions
        for joint, pos in stand_positions.items():
            if joint in self.desired_positions:
                self.desired_positions[joint] = pos

        # Execute position control
        self.execute_position_control()

    def get_stand_positions(self):
        """Get default standing joint positions"""
        stand_config = {
            'left_hip_pitch_joint': 0.0,
            'right_hip_pitch_joint': 0.0,
            'left_knee_joint': 0.0,
            'right_knee_joint': 0.0,
            'left_ankle_pitch_joint': 0.0,
            'right_ankle_pitch_joint': 0.0,
            'left_ankle_roll_joint': 0.0,
            'right_ankle_roll_joint': 0.0,
            # Arms in neutral position
            'left_shoulder_pitch_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'left_elbow_joint': -0.5,
            'right_elbow_joint': -0.5,
            'left_shoulder_yaw_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
        }
        return stand_config

    def publish_zero_commands(self):
        """Publish zero velocity commands for emergency stop"""
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.position = [self.current_positions[joint] for joint in self.joint_names]
        commands.velocity = [0.0] * len(self.joint_names)
        commands.effort = [0.0] * len(self.joint_names)
        self.joint_command_publisher.publish(commands)

    def publish_controller_state(self):
        """Publish controller state for monitoring"""
        msg = JointTrajectoryControllerState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.joint_names = self.joint_names
        msg.desired.positions = [self.desired_positions[joint] for joint in self.joint_names]
        msg.actual.positions = [self.current_positions[joint] for joint in self.joint_names]
        msg.error.positions = [self.desired_positions[joint] - self.current_positions[joint]
                              for joint in self.joint_names]

        self.controller_state_publisher.publish(msg)

    def publish_status(self):
        """Publish robot status"""
        status_msg = String()
        status_msg.data = (f"Mode: {self.control_mode}, Behavior: {self.active_behavior}, "
                          f"Emergency Stop: {self.emergency_stop_active}, "
                          f"Joints: {len(self.joint_names)}")
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiJointController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down multi-joint controller')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

2. **Create a trajectory generator node:**

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
import numpy as np
import math

class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, 'joint_trajectory', 10)
        self.command_subscription = self.create_subscription(
            String, 'trajectory_command', self.command_callback, 10)

        # Robot joint names
        self.joint_names = [
            'torso_yaw_joint', 'torso_pitch_joint', 'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
            'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
            'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]

        self.get_logger().info('Trajectory generator initialized')

    def command_callback(self, msg):
        """Handle trajectory commands"""
        command = msg.data.strip().lower()

        if command == "wave":
            self.generate_wave_trajectory()
        elif command == "stand_up":
            self.generate_stand_up_trajectory()
        elif command == "sit_down":
            self.generate_sit_down_trajectory()
        elif command == "clap":
            self.generate_clap_trajectory()

    def generate_wave_trajectory(self):
        """Generate waving motion trajectory for right arm"""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        # Define keyframes for waving motion
        duration = 3.0  # Total duration
        steps = int(duration / 0.05)  # 20Hz trajectory points

        for i in range(steps + 1):
            t = i / steps  # Normalized time [0, 1]
            actual_time = t * duration

            point = JointTrajectoryPoint()
            point.positions = []
            point.velocities = []
            point.accelerations = []

            # Start with current position (0.0 for all joints)
            base_positions = [0.0] * len(self.joint_names)

            # Add waving motion to right arm
            wave_amplitude = 0.8
            wave_freq = 2.0  # Hz

            right_shoulder_pos = wave_amplitude * math.sin(2 * math.pi * wave_freq * actual_time)
            right_elbow_pos = -0.5 + 0.4 * math.sin(2 * math.pi * wave_freq * actual_time + math.pi/2)

            # Find joint indices and update positions
            for j, joint_name in enumerate(self.joint_names):
                pos = base_positions[j]
                if joint_name == 'right_shoulder_pitch_joint':
                    pos = right_shoulder_pos
                elif joint_name == 'right_elbow_joint':
                    pos = right_elbow_pos
                elif joint_name == 'right_shoulder_yaw_joint':
                    pos = 0.2 * math.sin(2 * math.pi * wave_freq * actual_time * 0.5)

                point.positions.append(pos)
                point.velocities.append(0.0)  # Will be calculated by controller
                point.accelerations.append(0.0)

            point.time_from_start.sec = int(actual_time)
            point.time_from_start.nanosec = int((actual_time - int(actual_time)) * 1e9)
            msg.points.append(point)

        self.trajectory_publisher.publish(msg)

    def generate_stand_up_trajectory(self):
        """Generate trajectory for standing up motion"""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        # Define keyframes for standing up
        duration = 4.0
        steps = int(duration / 0.05)

        for i in range(steps + 1):
            t = i / steps
            actual_time = t * duration

            point = JointTrajectoryPoint()
            point.positions = []
            point.velocities = []
            point.accelerations = []

            # Start with sitting position and move to standing
            for joint_name in self.joint_names:
                pos = 0.0  # Default position

                # Sitting position (initial)
                if 'hip_pitch' in joint_name:
                    pos = -1.0  # Bent hips
                elif 'knee' in joint_name:
                    pos = -1.57  # Bent knees
                elif 'ankle_pitch' in joint_name:
                    pos = 0.5  # Feet positioned

                # Interpolate to standing position
                if 'hip_pitch' in joint_name:
                    pos = -1.0 + t * 1.0  # Straighten hips
                elif 'knee' in joint_name:
                    pos = -1.57 + t * 1.57  # Straighten knees
                elif 'ankle_pitch' in joint_name:
                    pos = 0.5 - t * 0.5  # Neutral feet

                point.positions.append(pos)
                point.velocities.append(0.0)
                point.accelerations.append(0.0)

            point.time_from_start.sec = int(actual_time)
            point.time_from_start.nanosec = int((actual_time - int(actual_time)) * 1e9)
            msg.points.append(point)

        self.trajectory_publisher.publish(msg)

    def generate_clap_trajectory(self):
        """Generate clapping motion trajectory"""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        duration = 2.0
        steps = int(duration / 0.05)

        for i in range(steps + 1):
            t = i / steps
            actual_time = t * duration

            point = JointTrajectoryPoint()
            point.positions = []
            point.velocities = []
            point.accelerations = []

            for joint_name in self.joint_names:
                pos = 0.0

                # Clapping motion for arms
                if 'shoulder_pitch' in joint_name:
                    # Both arms move toward center
                    if 'left' in joint_name:
                        pos = -0.8 + 0.6 * math.sin(math.pi * actual_time) if actual_time < 1.5 else -0.2
                    else:  # right
                        pos = -0.8 - 0.6 * math.sin(math.pi * actual_time) if actual_time < 1.5 else -0.2
                elif 'elbow' in joint_name:
                    # Elbows bend during clapping
                    pos = -1.5 if 'left' in joint_name else -1.5

                point.positions.append(pos)
                point.velocities.append(0.0)
                point.accelerations.append(0.0)

            point.time_from_start.sec = int(actual_time)
            point.time_from_start.nanosec = int((actual_time - int(actual_time)) * 1e9)
            msg.points.append(point)

        self.trajectory_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

3. **Test the controller system:**
```bash
# Terminal 1: Start the controller
ros2 run week5_exercises multi_joint_controller

# Terminal 2: Send commands
ros2 topic pub /robot_command std_msgs/String "data: 'STAND'"
ros2 topic pub /robot_command std_msgs/String "data: 'WALK'"
ros2 topic pub /robot_command std_msgs/String "data: 'MOVE_TO:left_elbow_joint,1.0'"

# Terminal 3: Send trajectory commands
ros2 topic pub /trajectory_command std_msgs/String "data: 'wave'"
ros2 topic pub /trajectory_command std_msgs/String "data: 'clap'"
```

### Expected Output
- Multi-joint controller managing 26+ joints simultaneously
- PID feedback control for precise positioning
- Trajectory execution with smooth interpolation
- Safety systems preventing dangerous movements
- Behavior switching between different robot modes
- Emergency stop functionality

### Submission Requirements
- Complete multi-joint controller implementation
- Trajectory generator node
- Test results showing different behaviors
- Safety system demonstration

## Exercise 3: Humanoid Robot Integration

### Objective
Create a complete integration system that combines URDF, control, and simulation for a humanoid robot.

### Requirements
1. Integrate the URDF with Gazebo simulation
2. Implement ros_control configuration
3. Create launch files for the complete system
4. Demonstrate coordinated behaviors
5. Include sensor integration and state estimation

### Implementation Steps

1. **Create the ros_control configuration file (config/humanoid_controllers.yaml):**

```yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    # Joint state broadcaster
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    # Individual joint group controllers
    torso_controller:
      type: position_controllers/JointTrajectoryController

    head_controller:
      type: position_controllers/JointTrajectoryController

    left_arm_controller:
      type: position_controllers/JointTrajectoryController

    right_arm_controller:
      type: position_controllers/JointTrajectoryController

    left_leg_controller:
      type: position_controllers/JointTrajectoryController

    right_leg_controller:
      type: position_controllers/JointTrajectoryController

# Torso controller
torso_controller:
  ros__parameters:
    joints:
      - torso_yaw_joint
      - torso_pitch_joint
    interface_name: position

# Head controller
head_controller:
  ros__parameters:
    joints:
      - neck_joint
    interface_name: position

# Left arm controller
left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_yaw_joint
      - left_shoulder_pitch_joint
      - left_shoulder_roll_joint
      - left_elbow_joint
      - left_wrist_yaw_joint
      - left_wrist_pitch_joint
    interface_name: position

# Right arm controller
right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_yaw_joint
      - right_shoulder_pitch_joint
      - right_shoulder_roll_joint
      - right_elbow_joint
      - right_wrist_yaw_joint
      - right_wrist_pitch_joint
    interface_name: position

# Left leg controller
left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_yaw_joint
      - left_hip_roll_joint
      - left_hip_pitch_joint
      - left_knee_joint
      - left_ankle_pitch_joint
      - left_ankle_roll_joint
    interface_name: position

# Right leg controller
right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_yaw_joint
      - right_hip_roll_joint
      - right_hip_pitch_joint
      - right_knee_joint
      - right_ankle_pitch_joint
      - right_ankle_roll_joint
    interface_name: position
```

2. **Create a launch file (launch/humanoid_robot.launch.py):**

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import TextSubstitution

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Get URDF path
    robot_description_path = PathJoinSubstitution([
        FindPackageShare('week5_exercises'),
        'urdf',
        'advanced_humanoid_robot.urdf.xacro'
    ])

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': Command(['xacro ', robot_description_path])}
        ]
    )

    # Joint State Publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time}
        ]
    )

    # Controller Manager node
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('week5_exercises'),
                'config',
                'humanoid_controllers.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        output='both',
    )

    # Spawn controllers after controller manager starts
    spawn_torso_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['torso_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    spawn_head_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['head_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    spawn_left_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_arm_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    spawn_right_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_arm_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    spawn_left_leg_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_leg_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    spawn_right_leg_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_leg_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # Multi-joint controller node
    multi_joint_controller = Node(
        package='week5_exercises',
        executable='multi_joint_controller',
        name='multi_joint_controller',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # Trajectory generator node
    trajectory_generator = Node(
        package='week5_exercises',
        executable='trajectory_generator',
        name='trajectory_generator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # Delay spawning controllers until controller manager is ready
    delay_spawner_torso = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[spawn_torso_controller],
        )
    )

    delay_spawner_head = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[spawn_head_controller],
        )
    )

    delay_spawner_left_arm = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[spawn_left_arm_controller],
        )
    )

    delay_spawner_right_arm = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[spawn_right_arm_controller],
        )
    )

    delay_spawner_left_leg = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[spawn_left_leg_controller],
        )
    )

    delay_spawner_right_leg = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=controller_manager,
            on_start=[spawn_right_leg_controller],
        )
    )

    # Return launch description
    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        controller_manager,
        multi_joint_controller,
        trajectory_generator,
        delay_spawner_torso,
        delay_spawner_head,
        delay_spawner_left_arm,
        delay_spawner_right_arm,
        delay_spawner_left_leg,
        delay_spawner_right_leg,
    ])
```

3. **Create a behavior demonstration node:**

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import time
import math

class BehaviorDemonstrator(Node):
    def __init__(self):
        super().__init__('behavior_demonstrator')

        # Publishers
        self.command_publisher = self.create_publisher(String, 'robot_command', 10)
        self.trajectory_command_publisher = self.create_publisher(String, 'trajectory_command', 10)

        # State monitoring
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Timer for behavior sequence
        self.behavior_timer = self.create_timer(0.1, self.behavior_sequence)

        self.current_behavior = "INIT"
        self.behavior_step = 0
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.current_positions = {}

        self.get_logger().info('Behavior demonstrator initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            self.current_positions[name] = msg.position[i]

    def behavior_sequence(self):
        """Execute a sequence of behaviors"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.start_time

        # Behavior sequence:
        # 0-5s: Stand
        # 5-10s: Wave
        # 10-15s: Clap
        # 15-20s: Balance test
        # 20-25s: Return to stand

        if elapsed < 5.0:
            if self.current_behavior != "STAND":
                self.current_behavior = "STAND"
                self.send_command("STAND")
                self.get_logger().info("Standing...")
        elif elapsed < 10.0:
            if self.current_behavior != "WAVE":
                self.current_behavior = "WAVE"
                self.send_trajectory_command("wave")
                self.get_logger().info("Waving...")
        elif elapsed < 15.0:
            if self.current_behavior != "CLAP":
                self.current_behavior = "CLAP"
                self.send_trajectory_command("clap")
                self.get_logger().info("Clapping...")
        elif elapsed < 20.0:
            if self.current_behavior != "BALANCE":
                self.current_behavior = "BALANCE"
                self.send_command("BALANCE")
                self.get_logger().info("Balancing...")
        elif elapsed < 25.0:
            if self.current_behavior != "RETURN_STAND":
                self.current_behavior = "RETURN_STAND"
                self.send_command("STAND")
                self.get_logger().info("Returning to stand...")
        else:
            # Reset sequence
            self.start_time = current_time
            self.current_behavior = "INIT"

    def send_command(self, command):
        """Send a command to the robot"""
        msg = String()
        msg.data = command
        self.command_publisher.publish(msg)

    def send_trajectory_command(self, command):
        """Send a trajectory command"""
        msg = String()
        msg.data = command
        self.trajectory_command_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorDemonstrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down behavior demonstrator')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

4. **Complete launch file with behavior demonstration:**

```python
# Add to the launch file above
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # ... (previous code remains the same until the return statement)

    # Behavior demonstrator node
    behavior_demonstrator = Node(
        package='week5_exercises',
        executable='behavior_demonstrator',
        name='behavior_demonstrator',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    # Add the behavior demonstrator to the launch description
    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        controller_manager,
        multi_joint_controller,
        trajectory_generator,
        behavior_demonstrator,  # Add this line
        delay_spawner_torso,
        delay_spawner_head,
        delay_spawner_left_arm,
        delay_spawner_right_arm,
        delay_spawner_left_leg,
        delay_spawner_right_leg,
    ])
```

5. **Test the complete system:**
```bash
# Build the package
cd ~/physical_ai_ws
colcon build --packages-select week5_exercises
source install/setup.bash

# Launch the complete system
ros2 launch week5_exercises humanoid_robot.launch.py

# In another terminal, visualize in RViz2
ros2 run rviz2 rviz2

# Monitor the behavior sequence
ros2 topic echo /robot_status
```

### Expected Output
- Complete integrated system with URDF, control, and simulation
- ros_control configuration managing all joints
- Coordinated behaviors executing automatically
- All components working together in harmony
- Proper launch file orchestration

### Submission Requirements
- Complete ros_control configuration
- Launch files for system integration
- Behavior demonstration showing coordinated motion
- System validation and testing results

## Grading Rubric

Each exercise will be graded on the following criteria:

- **Implementation Correctness** (30%): Code works as specified
- **Code Quality** (25%): Well-structured, documented, follows ROS 2 best practices
- **Understanding** (25%): Proper understanding of concepts demonstrated
- **Testing** (20%): Adequate testing and validation performed

## Submission Guidelines

- Submit all exercises as a complete ROS 2 package
- Include a README.md explaining your implementation
- Provide screenshots of successful execution
- Follow proper ROS 2 package structure and conventions
- Late submissions will be penalized by 10% per day

## Resources

- [ROS 2 URDF Tutorials](https://docs.ros.org/en/humble/Tutorials/URDF/Building-a-Visual-Robot-Model-with-URDF.html)
- [ros_control Documentation](https://control.ros.org/)
- [ROS 2 Controllers](https://github.com/ros-controls/ros2_controllers)
- [Gazebo Integration](https://gazebosim.org/docs/harmonic/ros_integration/)