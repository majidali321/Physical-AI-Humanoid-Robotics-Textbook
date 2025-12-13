---
sidebar_position: 3
---

# Week 5: Module 1 Project - Basic Robot Controller

## Overview

In Week 5, you'll complete Module 1 by implementing a comprehensive robot control system that integrates all the concepts learned in Weeks 3-5. This project combines ROS 2 architecture, Python integration with rclpy, and detailed URDF robot models into a functional robot controller.

## Project Objectives

By the end of this project, you will have:

- Created a complete humanoid robot URDF model
- Implemented a Python-based robot controller using rclpy
- Integrated joint state publishing and robot state publishing
- Created a functional teleoperation interface
- Validated the system through simulation

## Project Requirements

### Core Components

1. **URDF Robot Model**: Complete humanoid robot with articulated joints
2. **Joint State Controller**: Publishes current joint positions
3. **Robot State Publisher**: Publishes TF transforms for the robot
4. **Teleoperation Node**: Allows user control of the robot
5. **Safety System**: Prevents dangerous joint configurations

### Technical Specifications

- Use rclpy for all Python nodes
- Implement proper ROS 2 communication patterns
- Include parameter configuration for different robot types
- Add logging and error handling
- Provide a launch file to start all components

## Project Structure

### 1. URDF Model Development

Create a detailed URDF for a simple humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.075"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.075"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.175" rpy="0 0 0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.02"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.02"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Add more joints and links for complete humanoid model -->
</robot>
```

### 2. Joint State Publisher Node

Create a Python node that publishes joint states:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)

        # Declare parameters for joint configuration
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('robot_name', 'simple_humanoid')

        # Get parameters
        publish_rate = self.get_parameter('publish_rate').value

        # Create timer for publishing
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_joint_states)

        # Initialize joint positions
        self.joint_names = [
            'left_shoulder_joint', 'right_shoulder_joint',
            'left_elbow_joint', 'right_elbow_joint'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

        self.time = 0.0

        self.get_logger().info('Joint State Publisher initialized')

    def publish_joint_states(self):
        # Create joint state message
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Update joint positions (example: oscillating motion)
        self.time += 0.01
        for i, joint_name in enumerate(self.joint_names):
            # Create oscillating motion for demonstration
            self.joint_positions[i] = 0.5 * math.sin(self.time + i)

        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        # Publish message
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

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

### 3. Robot State Publisher Node

Create a node that publishes TF transforms based on joint states:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Robot State Publisher initialized')

    def joint_state_callback(self, msg):
        # Process joint states and broadcast transforms
        for i, joint_name in enumerate(msg.name):
            if i < len(msg.position):
                # Calculate transform based on joint position
                transform = TransformStamped()

                transform.header.stamp = msg.header.stamp
                transform.header.frame_id = 'base_link'
                transform.child_frame_id = joint_name.replace('_joint', '_link')

                # Simple transform calculation (in a real robot, this would be more complex)
                transform.transform.translation.x = 0.0
                transform.transform.translation.y = 0.0
                transform.transform.translation.z = 0.0

                # Set rotation based on joint position
                transform.transform.rotation.x = 0.0
                transform.transform.rotation.y = 0.0
                transform.transform.rotation.z = math.sin(msg.position[i] / 2.0)
                transform.transform.rotation.w = math.cos(msg.position[i] / 2.0)

                # Broadcast transform
                self.tf_broadcaster.sendTransform(transform)

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

### 4. Teleoperation Node

Create a node that allows user control of the robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import sys, select, termios, tty

class TeleopHumanoid(Node):
    def __init__(self):
        super().__init__('teleop_humanoid')

        self.publisher_ = self.create_publisher(JointState, 'joint_command', 10)

        # Initialize joint positions
        self.joint_names = [
            'left_shoulder_joint', 'right_shoulder_joint',
            'left_elbow_joint', 'right_elbow_joint'
        ]
        self.joint_positions = [0.0] * len(self.joint_names)

        # Movement increment
        self.increment = 0.1

        self.get_logger().info('Teleoperation node ready')
        self.get_logger().info('Use keys: q/w/e - left arm, a/s/d - right arm, z/x - reset')

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_teleop(self):
        self.settings = termios.tcgetattr(sys.stdin)

        try:
            while rclpy.ok():
                key = self.get_key()

                if key == 'q':  # Left shoulder up
                    self.joint_positions[0] = min(1.5, self.joint_positions[0] + self.increment)
                    self.publish_joint_command()
                elif key == 'w':  # Left shoulder center
                    self.joint_positions[0] = 0.0
                    self.publish_joint_command()
                elif key == 'e':  # Left shoulder down
                    self.joint_positions[0] = max(-1.5, self.joint_positions[0] - self.increment)
                    self.publish_joint_command()
                elif key == 'a':  # Right shoulder up
                    self.joint_positions[1] = min(1.5, self.joint_positions[1] + self.increment)
                    self.publish_joint_command()
                elif key == 's':  # Right shoulder center
                    self.joint_positions[1] = 0.0
                    self.publish_joint_command()
                elif key == 'd':  # Right shoulder down
                    self.joint_positions[1] = max(-1.5, self.joint_positions[1] - self.increment)
                    self.publish_joint_command()
                elif key == 'z':  # Reset all positions
                    self.joint_positions = [0.0] * len(self.joint_positions)
                    self.publish_joint_command()
                elif key == 'x':  # Exit
                    break
                elif key == '\x03':  # Ctrl+C
                    break

        except Exception as e:
            self.get_logger().error(f'Error in teleoperation: {e}')
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def publish_joint_command(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_positions

        self.publisher_.publish(msg)

        # Log current positions
        for i, name in enumerate(self.joint_names):
            self.get_logger().info(f'{name}: {self.joint_positions[i]:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = TeleopHumanoid()

    try:
        node.run_teleop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Launch File

Create a launch file to start all components:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Joint State Publisher
        Node(
            package='your_package_name',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'publish_rate': 50.0},
                {'robot_name': 'simple_humanoid'}
            ]
        ),

        # Robot State Publisher
        Node(
            package='your_package_name',
            executable='robot_state_publisher',
            name='robot_state_publisher'
        ),

        # RViz2 for visualization (optional)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(get_package_share_directory('your_package_name'), 'config', 'robot.rviz')]
        )
    ])
```

## Validation Steps

### 1. URDF Validation
```bash
# Check URDF syntax
check_urdf /path/to/your/robot.urdf

# Visualize URDF
urdf_to_graphiz /path/to/your/robot.urdf
```

### 2. TF Tree Validation
```bash
# View TF tree
ros2 run tf2_tools view_frames
```

### 3. Node Communication
```bash
# Check topics
ros2 topic list
ros2 topic echo /joint_states

# Check TF transforms
ros2 run tf2_ros tf2_echo base_link left_upper_arm
```

## Project Deliverables

### Required Files
- `robot.urdf` - Complete URDF model
- `joint_state_publisher.py` - Joint state publisher node
- `robot_state_publisher.py` - Robot state publisher node
- `teleop_humanoid.py` - Teleoperation node
- `launch/robot.launch.py` - Launch file
- `config/robot.rviz` - RViz configuration (optional)

### Evaluation Criteria
- **URDF Quality**: Properly structured, realistic, and functional
- **Node Implementation**: Correct use of rclpy, proper error handling
- **System Integration**: All components work together seamlessly
- **Code Quality**: Well-documented, follows Python best practices
- **Functionality**: Robot responds to commands as expected

## Advanced Extensions (Optional)

1. **Inverse Kinematics**: Implement basic IK for arm positioning
2. **Walking Pattern Generator**: Create rhythmic walking motions
3. **Balance Controller**: Implement simple balance maintenance
4. **ROS 2 Actions**: Use actions for complex multi-step behaviors

## Troubleshooting

### Common Issues
- **TF Issues**: Ensure frame names match between URDF and TF publishing
- **Joint Limits**: Verify joint limits in URDF match controller constraints
- **Timing**: Joint state publishing rate should match robot controller expectations

### Debugging Commands
```bash
# Monitor joint states
ros2 topic echo /joint_states

# Check TF tree
ros2 run tf2_ros tf2_echo base_link left_upper_arm

# View all transforms
ros2 run tf2_tools view_frames
```

## Next Steps

After completing this project, you'll have a solid foundation in ROS 2 architecture and robot modeling. In Module 2, you'll learn to simulate this robot in Gazebo and Unity environments.