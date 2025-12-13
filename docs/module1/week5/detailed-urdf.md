---
sidebar_position: 1
---

# Week 5: Detailed URDF for Humanoid Robots and Robot Control

## Learning Objectives

By the end of this week, you will be able to:
- Create detailed URDF models for complex humanoid robots
- Implement robot state publishing for visualization and control
- Design and implement robot control systems using ROS 2
- Integrate URDF with simulation environments
- Create comprehensive humanoid robot models with proper kinematics

## Advanced URDF Concepts for Humanoid Robots

### Kinematic Chains and Degrees of Freedom

Humanoid robots require complex kinematic structures with multiple degrees of freedom (DOF) to achieve human-like movement. Each joint contributes to the overall kinematic chain:

- **Torso**: Provides the base for all other limbs
- **Arms**: 6+ DOF each for manipulation tasks
- **Legs**: 6+ DOF each for locomotion and balance
- **Head**: 2-3 DOF for vision and interaction

### Detailed Humanoid URDF Example

Let's build a comprehensive humanoid URDF with realistic proportions and kinematics:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="detailed_humanoid">

  <!-- Constants and properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="deg_to_rad" value="0.017453292519943295"/>

  <!-- Humanoid dimensions (approximate adult human proportions) -->
  <xacro:property name="torso_height" value="0.6"/>
  <xacro:property name="torso_width" value="0.3"/>
  <xacro:property name="torso_depth" value="0.2"/>
  <xacro:property name="head_radius" value="0.1"/>
  <xacro:property name="neck_height" value="0.1"/>
  <xacro:property name="upper_arm_length" value="0.35"/>
  <xacro:property name="lower_arm_length" value="0.3"/>
  <xacro:property name="upper_arm_radius" value="0.07"/>
  <xacro:property name="lower_arm_radius" value="0.06"/>
  <xacro:property name="hand_length" value="0.18"/>
  <xacro:property name="hand_radius" value="0.05"/>
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

  <!-- Macro for creating arm chains -->
  <xacro:macro name="arm_chain" params="side parent_xyz parent_rpy">
    <!-- Shoulder joint -->
    <joint name="${side}_shoulder_yaw_joint" type="revolute">
      <parent link="torso"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${parent_xyz}" rpy="${parent_rpy}"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-90 * deg_to_rad}" upper="${90 * deg_to_rad}" effort="100" velocity="2"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

    <link name="${side}_shoulder">
      <visual>
        <geometry>
          <sphere radius="0.08"/>
        </geometry>
        <material name="grey"/>
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

    <joint name="${side}_shoulder_pitch_joint" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-90 * deg_to_rad}" upper="${90 * deg_to_rad}" effort="100" velocity="2"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

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

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 ${-upper_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-120 * deg_to_rad}" upper="${0 * deg_to_rad}" effort="80" velocity="2"/>
      <dynamics damping="0.8" friction="0.1"/>
    </joint>

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

    <joint name="${side}_wrist_joint" type="revolute">
      <parent link="${side}_lower_arm"/>
      <child link="${side}_hand"/>
      <origin xyz="0 0 ${-lower_arm_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-45 * deg_to_rad}" upper="${45 * deg_to_rad}" effort="50" velocity="2"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${side}_hand">
      <visual>
        <geometry>
          <box size="${hand_length} ${hand_radius*2} ${hand_radius*2}"/>
        </geometry>
        <material name="skin"/>
      </visual>
      <collision>
        <geometry>
          <box size="${hand_length} ${hand_radius*2} ${hand_radius*2}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Macro for creating leg chains -->
  <xacro:macro name="leg_chain" params="side parent_xyz">
    <!-- Hip joint -->
    <joint name="${side}_hip_yaw_joint" type="revolute">
      <parent link="torso"/>
      <child link="${side}_hip"/>
      <origin xyz="${parent_xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-30 * deg_to_rad}" upper="${30 * deg_to_rad}" effort="200" velocity="1"/>
      <dynamics damping="2.0" friction="0.2"/>
    </joint>

    <link name="${side}_hip">
      <visual>
        <geometry>
          <sphere radius="0.08"/>
        </geometry>
        <material name="grey"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.08"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <origin xyz="0 0 0"/>
        <inertia ixx="0.004" ixy="0.0" ixz="0.0" iyy="0.004" iyz="0.0" izz="0.004"/>
      </inertial>
    </link>

    <joint name="${side}_hip_roll_joint" type="revolute">
      <parent link="${side}_hip"/>
      <child link="${side}_upper_leg"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-30 * deg_to_rad}" upper="${30 * deg_to_rad}" effort="200" velocity="1"/>
      <dynamics damping="2.0" friction="0.2"/>
    </joint>

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

    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_upper_leg"/>
      <child link="${side}_lower_leg"/>
      <origin xyz="0 0 ${-upper_leg_length}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${0 * deg_to_rad}" upper="${120 * deg_to_rad}" effort="200" velocity="1"/>
      <dynamics damping="2.0" friction="0.2"/>
    </joint>

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

    <joint name="${side}_ankle_joint" type="revolute">
      <parent link="${side}_lower_leg"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 ${-lower_leg_length}" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-30 * deg_to_rad}" upper="${30 * deg_to_rad}" effort="150" velocity="1"/>
      <dynamics damping="1.0" friction="0.1"/>
    </joint>

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

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 ${torso_height}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="${-30 * deg_to_rad}" upper="${30 * deg_to_rad}" effort="20" velocity="1"/>
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

  <!-- Arms using macro -->
  <xacro:arm_chain side="left"
                   parent_xyz="${torso_width/2} ${torso_depth/2} ${torso_height*0.6}"
                   parent_rpy="0 0 0"/>

  <xacro:arm_chain side="right"
                   parent_xyz="${torso_width/2} ${-torso_depth/2} ${torso_height*0.6}"
                   parent_rpy="0 0 0"/>

  <!-- Legs using macro -->
  <xacro:leg_chain side="left" parent_xyz="${-torso_width/3} ${torso_depth/4} 0"/>
  <xacro:leg_chain side="right" parent_xyz="${-torso_width/3} ${-torso_depth/4} 0"/>

  <!-- ROS Control interface -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Transmission elements for ros_control -->
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
  <xacro:transmission_block joint_name="neck_joint"/>
  <xacro:transmission_block joint_name="left_shoulder_yaw_joint"/>
  <xacro:transmission_block joint_name="left_shoulder_pitch_joint"/>
  <xacro:transmission_block joint_name="left_elbow_joint"/>
  <xacro:transmission_block joint_name="left_wrist_joint"/>
  <xacro:transmission_block joint_name="right_shoulder_yaw_joint"/>
  <xacro:transmission_block joint_name="right_shoulder_pitch_joint"/>
  <xacro:transmission_block joint_name="right_elbow_joint"/>
  <xacro:transmission_block joint_name="right_wrist_joint"/>
  <xacro:transmission_block joint_name="left_hip_yaw_joint"/>
  <xacro:transmission_block joint_name="left_hip_roll_joint"/>
  <xacro:transmission_block joint_name="left_knee_joint"/>
  <xacro:transmission_block joint_name="left_ankle_joint"/>
  <xacro:transmission_block joint_name="right_hip_yaw_joint"/>
  <xacro:transmission_block joint_name="right_hip_roll_joint"/>
  <xacro:transmission_block joint_name="right_knee_joint"/>
  <xacro:transmission_block joint_name="right_ankle_joint"/>

</robot>
```

## Robot State Publishing

### Robot State Publisher Node

The robot state publisher is responsible for publishing the current state of all joints in the robot, which is essential for visualization and control:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math
import numpy as np

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(JointState, 'joint_states', 10)

        # Transform broadcaster for tf
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing state
        self.timer = self.create_timer(0.05, self.publish_state)  # 20Hz

        # Initialize joint positions
        self.joint_names = [
            'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Initialize all joint positions to zero
        self.joint_positions = {name: 0.0 for name in self.joint_names}

        # Demo pattern variables
        self.time = 0.0

        self.get_logger().info('Robot state publisher initialized')

    def publish_state(self):
        # Update joint positions with demo pattern
        self.update_demo_positions()

        # Create joint state message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        msg.velocity = [0.0] * len(msg.position)
        msg.effort = [0.0] * len(msg.position)

        # Publish joint states
        self.joint_state_publisher.publish(msg)

        # Broadcast transforms
        self.broadcast_transforms()

    def update_demo_positions(self):
        """Update joint positions with demo patterns"""
        self.time += 0.05  # Increment time based on timer rate

        # Head movement
        self.joint_positions['neck_joint'] = 0.2 * math.sin(self.time * 0.5)

        # Left arm movement
        self.joint_positions['left_shoulder_yaw_joint'] = 0.3 * math.sin(self.time * 0.7)
        self.joint_positions['left_shoulder_pitch_joint'] = 0.4 * math.sin(self.time * 0.6)
        self.joint_positions['left_elbow_joint'] = -0.5 * abs(math.sin(self.time * 0.5))
        self.joint_positions['left_wrist_joint'] = 0.2 * math.sin(self.time * 0.8)

        # Right arm movement
        self.joint_positions['right_shoulder_yaw_joint'] = 0.3 * math.sin(self.time * 0.7 + math.pi)
        self.joint_positions['right_shoulder_pitch_joint'] = 0.4 * math.sin(self.time * 0.6 + math.pi)
        self.joint_positions['right_elbow_joint'] = -0.5 * abs(math.sin(self.time * 0.5 + math.pi))
        self.joint_positions['right_wrist_joint'] = 0.2 * math.sin(self.time * 0.8 + math.pi)

        # Left leg movement (walking pattern)
        self.joint_positions['left_hip_yaw_joint'] = 0.1 * math.sin(self.time * 0.4)
        self.joint_positions['left_hip_roll_joint'] = 0.1 * math.sin(self.time * 0.3)
        self.joint_positions['left_knee_joint'] = 0.6 * abs(math.sin(self.time * 0.4))
        self.joint_positions['left_ankle_joint'] = -0.1 * math.sin(self.time * 0.4)

        # Right leg movement (walking pattern)
        self.joint_positions['right_hip_yaw_joint'] = 0.1 * math.sin(self.time * 0.4 + math.pi)
        self.joint_positions['right_hip_roll_joint'] = 0.1 * math.sin(self.time * 0.3 + math.pi)
        self.joint_positions['right_knee_joint'] = 0.6 * abs(math.sin(self.time * 0.4 + math.pi))
        self.joint_positions['right_ankle_joint'] = -0.1 * math.sin(self.time * 0.4 + math.pi)

    def broadcast_transforms(self):
        """Broadcast transforms for visualization"""
        # In a real implementation, you would calculate transforms based on forward kinematics
        # For now, we'll broadcast a simple base transform
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
    node = RobotStatePublisher()

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

## Robot Control Systems

### Joint State Controller

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np

class JointStateController(Node):
    def __init__(self):
        super().__init__('joint_state_controller')

        # Subscription to desired joint states
        self.joint_command_subscription = self.create_subscription(
            JointState, 'joint_commands', self.joint_command_callback, 10)

        # Publisher for actual joint states
        self.joint_state_publisher = self.create_publisher(
            JointState, 'actual_joint_states', 10)

        # Publisher for controller state
        self.controller_state_publisher = self.create_publisher(
            JointTrajectoryControllerState, 'controller_state', 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Initialize joint states
        self.joint_names = [
            'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}
        self.velocities = {name: 0.0 for name in self.joint_names}

        # Control parameters
        self.control_gain = 10.0  # Proportional gain
        self.max_velocity = 2.0   # rad/s

        self.get_logger().info('Joint state controller initialized')

    def joint_command_callback(self, msg):
        """Receive desired joint positions"""
        for i, name in enumerate(msg.name):
            if name in self.desired_positions:
                self.desired_positions[name] = msg.position[i]

    def control_loop(self):
        """Main control loop implementing PD control"""
        # Calculate control commands for each joint
        for joint_name in self.joint_names:
            current_pos = self.current_positions[joint_name]
            desired_pos = self.desired_positions[joint_name]

            # Simple proportional control
            error = desired_pos - current_pos
            control_output = self.control_gain * error

            # Limit control output to max velocity
            control_output = max(min(control_output, self.max_velocity), -self.max_velocity)

            # Update position (simple integration)
            self.current_positions[joint_name] += control_output * 0.01  # dt = 0.01s

            # Update velocity
            self.velocities[joint_name] = control_output

        # Publish actual joint states
        self.publish_joint_states()

        # Publish controller state
        self.publish_controller_state()

    def publish_joint_states(self):
        """Publish current joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = [self.current_positions[name] for name in self.joint_names]
        msg.velocity = [self.velocities[name] for name in self.joint_names]
        msg.effort = [0.0] * len(self.joint_names)  # Effort not calculated in this example

        self.joint_state_publisher.publish(msg)

    def publish_controller_state(self):
        """Publish controller state"""
        msg = JointTrajectoryControllerState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.joint_names = self.joint_names
        msg.desired.positions = [self.desired_positions[name] for name in self.joint_names]
        msg.actual.positions = [self.current_positions[name] for name in self.joint_names]
        msg.error.positions = [self.desired_positions[name] - self.current_positions[name]
                              for name in self.joint_names]

        self.controller_state_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStateController()

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

### Trajectory Controller

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np
import math
from rclpy.duration import Duration

class TrajectoryController(Node):
    def __init__(self):
        super().__init__('trajectory_controller')

        # Subscription to trajectory commands
        self.trajectory_subscription = self.create_subscription(
            JointTrajectory, 'joint_trajectory', self.trajectory_callback, 10)

        # Publisher for joint commands
        self.joint_command_publisher = self.create_publisher(
            JointState, 'joint_commands', 10)

        # Timer for trajectory execution
        self.trajectory_timer = self.create_timer(0.01, self.trajectory_execution)

        # Current state
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.current_point_idx = 0
        self.trajectory_active = False

        # Robot joint names
        self.joint_names = [
            'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        self.current_positions = {name: 0.0 for name in self.joint_names}

        self.get_logger().info('Trajectory controller initialized')

    def trajectory_callback(self, msg):
        """Receive trajectory command"""
        if len(msg.points) > 0:
            self.current_trajectory = msg
            self.trajectory_start_time = self.get_clock().now()
            self.current_point_idx = 0
            self.trajectory_active = True
            self.get_logger().info(f'Received trajectory with {len(msg.points)} points')

    def trajectory_execution(self):
        """Execute trajectory interpolation"""
        if not self.trajectory_active or not self.current_trajectory:
            return

        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.trajectory_start_time).nanoseconds / 1e9

        # Find the appropriate segment in the trajectory
        points = self.current_trajectory.points
        if self.current_point_idx >= len(points) - 1:
            # Trajectory completed
            self.trajectory_active = False
            self.get_logger().info('Trajectory completed')
            return

        # Get start and end points for interpolation
        start_point = points[self.current_point_idx]
        end_point = points[self.current_point_idx + 1]

        # Calculate segment duration
        segment_duration = (end_point.time_from_start.sec + end_point.time_from_start.nanosec / 1e9) - \
                          (start_point.time_from_start.sec + start_point.time_from_start.nanosec / 1e9)

        # Calculate progress within segment
        start_time_from_start = start_point.time_from_start.sec + start_point.time_from_start.nanosec / 1e9
        if elapsed_time >= start_time_from_start + segment_duration:
            # Move to next segment
            self.current_point_idx += 1
            return

        # Interpolate between start and end points
        progress = (elapsed_time - start_time_from_start) / segment_duration
        progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]

        # Linear interpolation
        interpolated_positions = []
        for i in range(len(start_point.positions)):
            start_pos = start_point.positions[i]
            end_pos = end_point.positions[i]
            interpolated_pos = start_pos + progress * (end_pos - start_pos)
            interpolated_positions.append(interpolated_pos)

        # Publish interpolated positions
        self.publish_joint_command(interpolated_positions)

    def publish_joint_command(self, positions):
        """Publish joint command"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.current_trajectory.joint_names
        msg.position = positions

        self.joint_command_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryController()

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

## Integration with Simulation

### Gazebo Integration

To properly integrate with Gazebo simulation, we need to define proper transmissions and controllers:

```xml
<!-- Add to the URDF -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</gazebo>

<!-- Controller configuration file (controllers.yaml) -->
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    joint_state_controller:
      type: joint_state_controller/JointStateController

    left_arm_controller:
      type: position_controllers/JointTrajectoryController

    right_arm_controller:
      type: position_controllers/JointTrajectoryController

    left_leg_controller:
      type: position_controllers/JointTrajectoryController

    right_leg_controller:
      type: position_controllers/JointTrajectoryController

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_yaw_joint
      - left_shoulder_pitch_joint
      - left_elbow_joint
      - left_wrist_joint
    interface_name: position

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_yaw_joint
      - right_shoulder_pitch_joint
      - right_elbow_joint
      - right_wrist_joint
    interface_name: position

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_yaw_joint
      - left_hip_roll_joint
      - left_knee_joint
      - left_ankle_joint
    interface_name: position

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_yaw_joint
      - right_hip_roll_joint
      - right_knee_joint
      - right_ankle_joint
    interface_name: position
```

## Robot Control Architecture

### Hierarchical Control System

For humanoid robots, a hierarchical control architecture is typically used:

1. **High-level planner**: Generates desired trajectories
2. **Mid-level controller**: Executes trajectories with feedback
3. **Low-level controller**: Direct motor control

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String
import numpy as np
import math

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Publishers and subscribers
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Command subscriptions
        self.walk_command_subscription = self.create_subscription(
            Twist, 'cmd_walk', self.walk_command_callback, 10)
        self.arm_command_subscription = self.create_subscription(
            Pose, 'cmd_arm', self.arm_command_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Robot state
        self.current_joint_positions = {}
        self.desired_joint_positions = {}
        self.joint_names = [
            'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        # Initialize all joints
        for name in self.joint_names:
            self.current_joint_positions[name] = 0.0
            self.desired_joint_positions[name] = 0.0

        # Walking state
        self.walk_active = False
        self.walk_params = {'vx': 0.0, 'vy': 0.0, 'w': 0.0}
        self.walk_phase = 0.0

        # Arm control state
        self.arm_active = False
        self.arm_target = None

        self.get_logger().info('Humanoid controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if name in self.current_joint_positions:
                self.current_joint_positions[name] = msg.position[i]

    def walk_command_callback(self, msg):
        """Handle walking commands"""
        self.walk_params['vx'] = msg.linear.x
        self.walk_params['vy'] = msg.linear.y
        self.walk_params['w'] = msg.angular.z
        self.walk_active = (abs(msg.linear.x) > 0.01 or abs(msg.linear.y) > 0.01 or abs(msg.angular.z) > 0.01)

    def arm_command_callback(self, msg):
        """Handle arm commands"""
        self.arm_target = msg
        self.arm_active = True

    def control_loop(self):
        """Main control loop"""
        # Update walking pattern if active
        if self.walk_active:
            self.update_walking_pattern()

        # Update arm pattern if active
        if self.arm_active:
            self.update_arm_pattern()

        # Apply gravity compensation and balance
        self.apply_balance_control()

        # Publish joint commands
        self.publish_joint_commands()

        # Update timing
        self.walk_phase += 0.01  # Increment phase based on control rate

    def update_walking_pattern(self):
        """Generate walking joint patterns"""
        # Simplified walking pattern - in reality this would be much more complex
        phase = self.walk_phase
        step_height = 0.05 * abs(self.walk_params['vx'])  # Height proportional to speed

        # Left leg
        self.desired_joint_positions['left_hip_yaw_joint'] = 0.1 * math.sin(phase * 2)
        self.desired_joint_positions['left_knee_joint'] = 0.6 * abs(math.sin(phase * 2)) * (1 + step_height)
        self.desired_joint_positions['left_ankle_joint'] = -0.1 * math.sin(phase * 2)

        # Right leg
        self.desired_joint_positions['right_hip_yaw_joint'] = 0.1 * math.sin(phase * 2 + math.pi)
        self.desired_joint_positions['right_knee_joint'] = 0.6 * abs(math.sin(phase * 2 + math.pi)) * (1 + step_height)
        self.desired_joint_positions['right_ankle_joint'] = -0.1 * math.sin(phase * 2 + math.pi)

        # Adjust arms for balance
        self.desired_joint_positions['left_shoulder_pitch_joint'] = -0.2 * math.sin(phase * 2)
        self.desired_joint_positions['right_shoulder_pitch_joint'] = -0.2 * math.sin(phase * 2 + math.pi)

    def update_arm_pattern(self):
        """Generate arm movement patterns"""
        if self.arm_target:
            # Simple arm positioning - in reality this would involve inverse kinematics
            self.desired_joint_positions['left_shoulder_yaw_joint'] += 0.01  # Move toward target
            self.desired_joint_positions['left_elbow_joint'] -= 0.01

    def apply_balance_control(self):
        """Apply balance and stability control"""
        # Simple balance control based on current joint positions
        # In reality, this would use IMU data and more sophisticated control algorithms
        for joint_name in self.joint_names:
            if joint_name not in ['left_knee_joint', 'right_knee_joint']:
                # Apply slight damping to maintain stability
                current = self.current_joint_positions[joint_name]
                desired = self.desired_joint_positions[joint_name]
                self.desired_joint_positions[joint_name] = 0.95 * desired + 0.05 * current

    def publish_joint_commands(self):
        """Publish joint commands to the robot"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [self.desired_joint_positions[name] for name in self.joint_names]

        self.joint_command_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidController()

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

## Best Practices for Humanoid Robot Control

### 1. Safety Considerations
- Implement joint position limits
- Use velocity and acceleration limits
- Include emergency stop functionality
- Monitor for hardware failures

### 2. Control Architecture
- Use appropriate control frequencies
- Implement proper feedback loops
- Separate high-level planning from low-level control
- Include state estimation for balance

### 3. Simulation to Reality Transfer
- Model actuator dynamics accurately
- Include sensor noise and delays
- Validate control algorithms in simulation first
- Use system identification for model refinement

### 4. Performance Optimization
- Use efficient kinematic libraries
- Implement multi-threading where appropriate
- Optimize control algorithms for real-time performance
- Monitor computational load

## Summary

Week 5 covered detailed URDF modeling for humanoid robots, robot state publishing, and comprehensive control systems. We explored:

- Advanced URDF concepts with realistic humanoid proportions
- Robot state publisher implementation for visualization
- Joint state control with feedback
- Trajectory execution for coordinated movement
- Hierarchical control architecture for humanoid robots
- Integration with simulation environments

These concepts form the foundation for building complex humanoid robot systems that can be controlled and simulated effectively in ROS 2. The combination of detailed modeling, proper state publishing, and robust control systems enables the development of capable humanoid robots.