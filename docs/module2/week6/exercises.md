---
sidebar_position: 4
---

# Week 6: Gazebo Simulation Exercises

## Overview

This section provides hands-on exercises to reinforce your understanding of Gazebo simulation. Complete these exercises to gain practical experience with physics simulation, sensor configuration, and ROS 2 integration in Gazebo.

## Exercise 1: Basic Robot Simulation

### Objective
Create a simple wheeled robot and simulate it in Gazebo with basic sensors.

### Instructions
1. Create a URDF model of a simple differential drive robot
2. Add a camera and IMU to the robot
3. Create a Gazebo world with obstacles
4. Implement basic movement control using ROS 2

### Robot URDF Template
```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Add right wheel, joints, and sensors -->

  <!-- Gazebo plugins for ROS 2 control -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>
</robot>
```

### Python Controller Template
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu
import time

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers for sensor data
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Timer for movement commands
        self.timer = self.create_timer(0.1, self.move_robot)

        self.get_logger().info('Robot controller initialized')

    def move_robot(self):
        msg = Twist()
        # TODO: Implement movement logic
        # Example: Move forward with slight turn
        msg.linear.x = 0.5
        msg.angular.z = 0.1
        self.cmd_vel_pub.publish(msg)

    def camera_callback(self, msg):
        # TODO: Process camera data
        self.get_logger().info(f'Received camera image: {msg.width}x{msg.height}')

    def imu_callback(self, msg):
        # TODO: Process IMU data
        self.get_logger().info(f'IMU orientation: {msg.orientation}')

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Expected Output
- Robot moves in the Gazebo environment
- Sensor data is published and received
- Robot can be controlled via ROS 2 topics

## Exercise 2: Physics Parameter Tuning

### Objective
Tune physics parameters to achieve realistic robot behavior.

### Instructions
1. Create a simulation with a humanoid robot model
2. Experiment with different physics parameters
3. Compare the behavior under different configurations
4. Document the optimal parameters for stable walking

### Physics Configuration Template
```xml
<physics name="tuned_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>  <!-- Try different values: 10, 20, 50 -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>  <!-- Try different values: 0.1, 0.2, 0.5 -->
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Tasks
1. Test different solver iteration counts (10, 20, 50)
2. Test different ERP values (0.1, 0.2, 0.5)
3. Measure simulation stability and performance
4. Record optimal parameters for your specific robot model

## Exercise 3: Sensor Fusion Implementation

### Objective
Implement a sensor fusion algorithm that combines data from multiple sensors.

### Instructions
1. Create a robot with multiple sensors (camera, LiDAR, IMU)
2. Implement a particle filter or Kalman filter for localization
3. Compare performance with and without sensor fusion
4. Analyze the improvement in accuracy

### Sensor Fusion Node Template
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

class SensorFusion(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribers for different sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publisher for fused pose estimate
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'fused_pose', 10)

        # Initialize filter state
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0
        self.covariance = np.eye(3) * 0.1  # Initial uncertainty

        self.get_logger().info('Sensor fusion node initialized')

    def lidar_callback(self, msg):
        # Process LiDAR data for position estimation
        # TODO: Implement LiDAR-based position update
        pass

    def imu_callback(self, msg):
        # Process IMU data for orientation estimation
        # TODO: Implement IMU-based orientation update
        pass

    def publish_fused_pose(self):
        # Publish the fused pose estimate
        # TODO: Create and publish PoseWithCovarianceStamped message
        pass

def main(args=None):
    rclpy.init(args=args)
    fusion = SensorFusion()

    try:
        rclpy.spin(fusion)
    except KeyboardInterrupt:
        pass
    finally:
        fusion.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 4: Complex Environment Simulation

### Objective
Create a complex environment with multiple objects and simulate robot interaction.

### Instructions
1. Design a world with various obstacles and surfaces
2. Implement a navigation system for the robot
3. Test robot performance in different scenarios
4. Analyze the effect of environment complexity on performance

### World Design Requirements
- Include static obstacles (boxes, cylinders)
- Add dynamic objects (movable objects)
- Create different surface types (with different friction)
- Add lighting effects

### Navigation Node Template
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np

class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.lidar_sub = self.create_subscription(
            LaserScan, 'scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)

        # Navigation parameters
        self.target = np.array([5.0, 5.0])  # Target position
        self.safe_distance = 0.5  # Minimum distance to obstacles

        # Robot state
        self.position = np.array([0.0, 0.0])
        self.orientation = 0.0

        # Timer for navigation updates
        self.timer = self.create_timer(0.1, self.navigate)

        self.get_logger().info('Navigation controller initialized')

    def lidar_callback(self, msg):
        # Process LiDAR data for obstacle detection
        # TODO: Implement obstacle detection and avoidance
        pass

    def odom_callback(self, msg):
        # Update robot position from odometry
        # TODO: Extract position and orientation from odometry message
        pass

    def navigate(self):
        # Implement navigation algorithm
        # TODO: Calculate desired velocity based on target and obstacles
        msg = Twist()

        # Example: Simple proportional controller
        target_direction = self.target - self.position
        distance_to_target = np.linalg.norm(target_direction)

        if distance_to_target > 0.1:  # If not at target
            msg.linear.x = min(0.5, distance_to_target * 0.5)  # Move toward target
            # Add obstacle avoidance logic here

        self.cmd_vel_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    nav = NavigationController()

    try:
        rclpy.spin(nav)
    except KeyboardInterrupt:
        pass
    finally:
        nav.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Exercise 5: Humanoid Robot Simulation

### Objective
Create a humanoid robot model with proper physics and simulate basic walking.

### Instructions
1. Create a detailed humanoid URDF with all necessary joints
2. Configure physics parameters for stable bipedal locomotion
3. Implement a simple walking controller
4. Test the robot in various scenarios

### Humanoid URDF Template
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Add arms, legs, and joints -->

  <!-- Joints connecting body parts -->
  <joint name="torso_head" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Add more joints for arms and legs -->
</robot>
```

## Exercise 6: Advanced Sensor Simulation

### Objective
Implement realistic sensor simulation with proper noise models and validation.

### Instructions
1. Create a robot with multiple sensor types
2. Configure realistic noise parameters based on real sensor specifications
3. Validate sensor simulation against expected behavior
4. Implement sensor calibration procedures

### Tasks
1. Configure camera with realistic noise and distortion
2. Set up LiDAR with appropriate range and resolution
3. Implement IMU with bias and drift characteristics
4. Validate sensor data using known ground truth

## Solutions and Hints

### Exercise 1 Solution
- Use `ros2 run gazebo_ros spawn_entity.py` to spawn your robot
- Check that all joints are properly connected in your URDF
- Verify plugin configurations match your robot model

### Exercise 2 Solution
- Start with conservative parameters and gradually optimize
- Monitor simulation real-time factor for performance
- Use `gz stats` to monitor simulation performance

### Exercise 3 Solution
- Implement proper covariance combination for sensor fusion
- Consider time synchronization between sensors
- Use appropriate filter for your specific application

## Evaluation Criteria

- **Implementation Quality**: Code is well-structured and follows best practices
- **Functionality**: All exercises complete and working as expected
- **Analysis**: Thorough analysis of results and parameter effects
- **Documentation**: Clear comments and explanations
- **Problem-Solving**: Creative solutions to challenges encountered

## Next Steps

After completing these exercises, you should have a solid understanding of Gazebo simulation and be ready to explore Unity integration in Week 7. Consider extending these exercises with more complex scenarios or additional sensor types.