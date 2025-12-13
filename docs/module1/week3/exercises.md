---
sidebar_position: 3
---

# Week 3 Exercises: ROS 2 Architecture

## Exercise 1: Basic Publisher-Subscriber System

### Objective
Create a complete publisher-subscriber system that demonstrates ROS 2 communication patterns.

### Requirements
1. Create a ROS 2 package called `week3_exercises`
2. Implement a publisher node that sends robot status messages
3. Implement a subscriber node that receives and processes these messages
4. Use custom message types with appropriate fields
5. Include proper error handling and logging

### Implementation Steps

1. **Create the package:**
   ```bash
   cd ~/physical_ai_ws/src
   ros2 pkg create --build-type ament_python week3_exercises
   ```

2. **Create a custom message for robot status:**
   Create directory and file:
   ```
   week3_exercises/msg/RobotStatus.msg
   ```
   With content:
   ```
   string robot_name
   float64 battery_level
   bool is_moving
   geometry_msgs/Pose current_pose
   time last_update
   ```

3. **Implement the publisher node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from week3_exercises.msg import RobotStatus
   from geometry_msgs.msg import Pose
   import random
   import math

   class RobotStatusPublisher(Node):
       def __init__(self):
           super().__init__('robot_status_publisher')

           # Create publisher
           self.publisher = self.create_publisher(RobotStatus, 'robot_status', 10)

           # Create timer for periodic publishing
           self.timer = self.create_timer(1.0, self.publish_status)

           # Robot state variables
           self.robot_name = "humanoid_robot_01"
           self.position_x = 0.0
           self.position_y = 0.0
           self.angle = 0.0

           self.get_logger().info('Robot status publisher started')

       def publish_status(self):
           msg = RobotStatus()
           msg.robot_name = self.robot_name

           # Simulate battery level (decreasing over time)
           battery_depletion = random.uniform(0.001, 0.005)
           current_battery = 100.0 - (self.get_time().nanoseconds / 1e9 * battery_depletion)
           msg.battery_level = max(0.0, min(100.0, current_battery))

           # Simulate movement state
           msg.is_moving = random.choice([True, False])

           # Update position (simulate movement)
           self.angle += random.uniform(-0.1, 0.1)
           self.position_x += 0.1 * math.cos(self.angle)
           self.position_y += 0.1 * math.sin(self.angle)

           # Set pose
           msg.current_pose.position.x = self.position_x
           msg.current_pose.position.y = self.position_y
           msg.current_pose.position.z = 0.0  # Assuming 2D movement

           # Simple orientation (facing direction)
           msg.current_pose.orientation.z = math.sin(self.angle / 2)
           msg.current_pose.orientation.w = math.cos(self.angle / 2)

           # Set timestamp
           msg.last_update = self.get_clock().now().to_msg()

           self.publisher.publish(msg)
           self.get_logger().info(
               f'Published status: {msg.robot_name}, '
               f'Battery: {msg.battery_level:.1f}%, '
               f'Moving: {msg.is_moving}, '
               f'Position: ({msg.current_pose.position.x:.2f}, {msg.current_pose.position.y:.2f})'
           )

   def main(args=None):
       rclpy.init(args=args)
       node = RobotStatusPublisher()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Shutting down robot status publisher')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Implement the subscriber node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from week3_exercises.msg import RobotStatus

   class RobotStatusSubscriber(Node):
       def __init__(self):
           super().__init__('robot_status_subscriber')

           # Create subscription
           self.subscription = self.create_subscription(
               RobotStatus,
               'robot_status',
               self.status_callback,
               10
           )

           # Statistics tracking
           self.message_count = 0
           self.total_battery = 0.0
           self.moving_count = 0

           self.get_logger().info('Robot status subscriber started')

       def status_callback(self, msg):
           self.message_count += 1
           self.total_battery += msg.battery_level
           if msg.is_moving:
               self.moving_count += 1

           # Log received status
           self.get_logger().info(
               f'Received status #{self.message_count}: {msg.robot_name}, '
               f'Battery: {msg.battery_level:.1f}%, '
               f'Moving: {msg.is_moving}'
           )

           # Log statistics every 10 messages
           if self.message_count % 10 == 0:
               avg_battery = self.total_battery / self.message_count
               moving_percentage = (self.moving_count / self.message_count) * 100
               self.get_logger().info(
                   f'Statistics: Average battery {avg_battery:.1f}%, '
                   f'Moving {moving_percentage:.1f}% of time'
               )

   def main(args=None):
       rclpy.init(args=args)
       node = RobotStatusSubscriber()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Shutting down robot status subscriber')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

5. **Update package.xml with dependencies:**
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>week3_exercises</name>
     <version>0.0.0</version>
     <description>Week 3 exercises for ROS 2 architecture</description>
     <maintainer email="user@example.com">Your Name</maintainer>
     <license>Apache-2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>geometry_msgs</depend>

     <exec_depend>rosidl_default_runtime</exec_depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <member_of_group>rosidl_interface_packages</member_of_group>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

6. **Create setup files:**
   Create `setup.py`:
   ```python
   from setuptools import setup
   import os
   from glob import glob

   package_name = 'week3_exercises'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
           # Include any other files you want to install
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='user@example.com',
       description='Week 3 exercises for ROS 2 architecture',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'robot_status_publisher = week3_exercises.robot_status_publisher:main',
               'robot_status_subscriber = week3_exercises.robot_status_subscriber:main',
           ],
       },
   )
   ```

7. **Create setup.cfg:**
   ```
   [develop]
   script-dir=$base/lib/week3_exercises
   [install]
   install-scripts=$base/lib/week3_exercises
   ```

8. **Test the system:**
   ```bash
   cd ~/physical_ai_ws
   colcon build --packages-select week3_exercises
   source install/setup.bash

   # Terminal 1: Run publisher
   ros2 run week3_exercises robot_status_publisher

   # Terminal 2: Run subscriber
   ros2 run week3_exercises robot_status_subscriber
   ```

### Expected Output
- Publisher sending robot status messages every second
- Subscriber receiving and logging messages
- Statistics displayed periodically
- Proper error handling and graceful shutdown

### Submission Requirements
- Complete source code for publisher and subscriber
- Package configuration files
- Screenshots of successful execution
- Explanation of message design choices

## Exercise 2: Service-Based Robot Control

### Objective
Implement a service-based robot control system that allows remote command execution.

### Requirements
1. Create a service server that accepts robot commands
2. Implement multiple service types for different commands
3. Create a client that can send various commands
4. Include command validation and error handling

### Implementation Steps

1. **Create service definition files:**
   Create `srv/MoveRobot.srv`:
   ```
   float64 linear_velocity
   float64 angular_velocity
   ---
   bool success
   string message
   ```

   Create `srv/GetPosition.srv`:
   ```
   ---
   geometry_msgs/Pose current_pose
   bool success
   string message
   ```

   Create `srv/SetJointPositions.srv`:
   ```
   string[] joint_names
   float64[] positions
   ---
   bool success
   string message
   ```

2. **Implement the robot control service server:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from week3_exercises.srv import MoveRobot, GetPosition, SetJointPositions
   from geometry_msgs.msg import Pose
   import math

   class RobotControlServer(Node):
       def __init__(self):
           super().__init__('robot_control_server')

           # Create services
           self.move_service = self.create_service(
               MoveRobot, 'move_robot', self.move_robot_callback)
           self.position_service = self.create_service(
               GetPosition, 'get_position', self.get_position_callback)
           self.joint_service = self.create_service(
               SetJointPositions, 'set_joint_positions', self.set_joint_positions_callback)

           # Robot state
           self.current_x = 0.0
           self.current_y = 0.0
           self.current_theta = 0.0
           self.joint_positions = {'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0}

           self.get_logger().info('Robot control server started')

       def move_robot_callback(self, request, response):
           try:
               # Validate inputs
               if abs(request.linear_velocity) > 2.0 or abs(request.angular_velocity) > 1.0:
                   response.success = False
                   response.message = 'Velocity values out of safe range'
                   return response

               # Simulate movement (integrate velocity over 1 second)
               dt = 1.0  # 1 second movement
               self.current_x += request.linear_velocity * math.cos(self.current_theta) * dt
               self.current_y += request.linear_velocity * math.sin(self.current_theta) * dt
               self.current_theta += request.angular_velocity * dt

               # Normalize angle
               self.current_theta = math.atan2(
                   math.sin(self.current_theta), math.cos(self.current_theta))

               response.success = True
               response.message = f'Robot moved to ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_theta:.2f})'
               self.get_logger().info(response.message)

           except Exception as e:
               response.success = False
               response.message = f'Error moving robot: {str(e)}'
               self.get_logger().error(response.message)

           return response

       def get_position_callback(self, request, response):
           try:
               response.current_pose.position.x = self.current_x
               response.current_pose.position.y = self.current_y
               response.current_pose.position.z = 0.0
               response.current_pose.orientation.z = math.sin(self.current_theta / 2)
               response.current_pose.orientation.w = math.cos(self.current_theta / 2)

               response.success = True
               response.message = 'Position retrieved successfully'

           except Exception as e:
               response.success = False
               response.message = f'Error getting position: {str(e)}'
               self.get_logger().error(response.message)

           return response

       def set_joint_positions_callback(self, request, response):
           try:
               # Validate input
               if len(request.joint_names) != len(request.positions):
                   response.success = False
                   response.message = 'Joint names and positions arrays must have same length'
                   return response

               # Update joint positions
               for name, pos in zip(request.joint_names, request.positions):
                   if name in self.joint_positions:
                       self.joint_positions[name] = pos
                   else:
                       self.get_logger().warn(f'Unknown joint: {name}')

               response.success = True
               response.message = f'Updated {len(request.joint_names)} joint positions'
               self.get_logger().info(response.message)

           except Exception as e:
               response.success = False
               response.message = f'Error setting joint positions: {str(e)}'
               self.get_logger().error(response.message)

           return response

   def main(args=None):
       rclpy.init(args=args)
       node = RobotControlServer()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Shutting down robot control server')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Implement the client node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from week3_exercises.srv import MoveRobot, GetPosition, SetJointPositions
   import sys

   class RobotControlClient(Node):
       def __init__(self):
           super().__init__('robot_control_client')

           # Create clients
           self.move_client = self.create_client(MoveRobot, 'move_robot')
           self.position_client = self.create_client(GetPosition, 'get_position')
           self.joint_client = self.create_client(SetJointPositions, 'set_joint_positions')

           # Wait for services
           while not self.move_client.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Move service not available, waiting again...')
           while not self.position_client.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Position service not available, waiting again...')
           while not self.joint_client.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Joint service not available, waiting again...')

           # Create requests
           self.move_request = MoveRobot.Request()
           self.position_request = GetPosition.Request()
           self.joint_request = SetJointPositions.Request()

           self.get_logger().info('Robot control client started')

       def move_robot(self, linear_vel, angular_vel):
           self.move_request.linear_velocity = linear_vel
           self.move_request.angular_velocity = angular_vel

           future = self.move_client.call_async(self.move_request)
           rclpy.spin_until_future_complete(self, future)

           if future.result() is not None:
               result = future.result()
               self.get_logger().info(f'Move result: {result.message}')
               return result.success
           else:
               self.get_logger().error('Move service call failed')
               return False

       def get_position(self):
           future = self.position_client.call_async(self.position_request)
           rclpy.spin_until_future_complete(self, future)

           if future.result() is not None:
               result = future.result()
               if result.success:
                   pose = result.current_pose
                   self.get_logger().info(
                       f'Current position: ({pose.position.x:.2f}, {pose.position.y:.2f}) '
                       f'Orientation: {pose.orientation.z:.2f}')
                   return pose
               else:
                   self.get_logger().error(f'Get position failed: {result.message}')
                   return None
           else:
               self.get_logger().error('Get position service call failed')
               return None

       def set_joint_positions(self, joint_names, positions):
           self.joint_request.joint_names = joint_names
           self.joint_request.positions = positions

           future = self.joint_client.call_async(self.joint_request)
           rclpy.spin_until_future_complete(self, future)

           if future.result() is not None:
               result = future.result()
               self.get_logger().info(f'Joint result: {result.message}')
               return result.success
           else:
               self.get_logger().error('Set joint positions service call failed')
               return False

       def run_demo_sequence(self):
           self.get_logger().info('Starting robot control demo sequence...')

           # 1. Get initial position
           self.get_position()

           # 2. Move robot
           self.move_robot(0.5, 0.0)  # Move forward
           self.get_position()

           # 3. Turn robot
           self.move_robot(0.0, 0.5)  # Turn right
           self.get_position()

           # 4. Set joint positions
           joint_names = ['joint1', 'joint2', 'joint3']
           positions = [0.1, 0.2, 0.3]
           self.set_joint_positions(joint_names, positions)

           # 5. Move to final position
           self.move_robot(0.3, -0.2)  # Combined movement
           final_pose = self.get_position()

           self.get_logger().info('Demo sequence completed')

   def main(args=None):
       rclpy.init(args=args)
       client = RobotControlClient()

       try:
           if len(sys.argv) > 1 and sys.argv[1] == 'demo':
               client.run_demo_sequence()
           else:
               client.get_logger().info('Run with "demo" argument to execute demo sequence')
       except KeyboardInterrupt:
           client.get_logger().info('Shutting down robot control client')
       finally:
           client.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Test the service system:**
   ```bash
   # Terminal 1: Run service server
   ros2 run week3_exercises robot_control_server

   # Terminal 2: Run client with demo
   ros2 run week3_exercises robot_control_client demo
   ```

### Expected Output
- Service server handling multiple request types
- Client successfully sending various commands
- Proper command validation and error handling
- Robot state updates reflected in responses

### Submission Requirements
- Complete service server and client implementation
- All service definition files
- Test output showing successful service calls
- Error handling demonstration

## Exercise 3: Advanced Communication Patterns

### Objective
Implement advanced ROS 2 communication patterns including multiple subscribers, publishers, and services working together.

### Requirements
1. Create a node that subscribes to multiple topics
2. Implement data fusion from multiple sources
3. Use services to control the fusion process
4. Publish fused results to other nodes
5. Include performance monitoring

### Implementation Steps

1. **Create the advanced communication node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan, Imu
   from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
   from week3_exercises.srv import SetBool
   from std_msgs.msg import Float32
   import numpy as np
   import math
   import time

   class AdvancedCommunicationNode(Node):
       def __init__(self):
           super().__init__('advanced_communication')

           # Subscriptions for multiple sensor types
           self.scan_subscription = self.create_subscription(
               LaserScan, 'scan', self.scan_callback, 10)
           self.imu_subscription = self.create_subscription(
               Imu, 'imu/data', self.imu_callback, 10)

           # Publishers for fused data
           self.obstacle_publisher = self.create_publisher(Float32, 'obstacle_distance', 10)
           self.fused_pose_publisher = self.create_publisher(
               PoseWithCovarianceStamped, 'fused_pose', 10)
           self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

           # Service for controlling fusion process
           self.fusion_control_service = self.create_service(
               SetBool, 'control_fusion', self.control_fusion_callback)

           # Data storage
           self.latest_scan = None
           self.latest_imu = None
           self.fusion_enabled = True
           self.scan_callback_count = 0
           self.imu_callback_count = 0

           # Performance monitoring
           self.start_time = time.time()
           self.message_count = 0

           self.get_logger().info('Advanced communication node started')

       def scan_callback(self, msg):
           self.latest_scan = msg
           self.scan_callback_count += 1
           self.message_count += 1

           if self.fusion_enabled:
               self.process_scan_data()

       def imu_callback(self, msg):
           self.latest_imu = msg
           self.imu_callback_count += 1
           self.message_count += 1

           if self.fusion_enabled:
               self.process_imu_data()

       def process_scan_data(self):
           if self.latest_scan is None:
               return

           # Find closest obstacle in front of robot (forward 45 degrees)
           min_distance = float('inf')
           valid_ranges = []

           # Calculate the range of indices for forward-looking scan
           total_angles = len(self.latest_scan.ranges)
           center_idx = total_angles // 2
           range_width = int(45 * (math.pi / 180) / self.latest_scan.angle_increment) // 2

           start_idx = max(0, center_idx - range_width)
           end_idx = min(total_angles, center_idx + range_width)

           for i in range(start_idx, end_idx):
               if (self.latest_scan.range_min < self.latest_scan.ranges[i] <
                   self.latest_scan.range_max):
                   valid_ranges.append(self.latest_scan.ranges[i])
                   if self.latest_scan.ranges[i] < min_distance:
                       min_distance = self.latest_scan.ranges[i]

           # Publish obstacle distance
           if min_distance != float('inf'):
               obstacle_msg = Float32()
               obstacle_msg.data = min_distance
               self.obstacle_publisher.publish(obstacle_msg)

               # If obstacle is too close, send stop command
               if min_distance < 0.5:  # 50cm threshold
                   self.send_stop_command()

       def process_imu_data(self):
           if self.latest_imu is None:
               return

           # Create a fused pose message from IMU data
           pose_msg = PoseWithCovarianceStamped()
           pose_msg.header.stamp = self.get_clock().now().to_msg()
           pose_msg.header.frame_id = 'odom'

           # Use IMU orientation
           pose_msg.pose.pose.orientation = self.latest_imu.orientation

           # Position would normally come from odometry integration
           # For simulation, we'll keep it at origin
           pose_msg.pose.pose.position.x = 0.0
           pose_msg.pose.pose.position.y = 0.0
           pose_msg.pose.pose.position.z = 0.0

           # Set covariance (simplified)
           pose_msg.pose.covariance = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.1, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.1]

           self.fused_pose_publisher.publish(pose_msg)

       def send_stop_command(self):
           """Send stop command when obstacle is detected"""
           cmd = Twist()
           cmd.linear.x = 0.0
           cmd.angular.z = 0.0
           self.velocity_publisher.publish(cmd)
           self.get_logger().warn('Obstacle detected! Sending stop command.')

       def control_fusion_callback(self, request, response):
           self.fusion_enabled = request.data
           status = "enabled" if self.fusion_enabled else "disabled"
           response.success = True
           response.message = f'Data fusion {status}'
           self.get_logger().info(response.message)
           return response

       def get_performance_stats(self):
           """Calculate and return performance statistics"""
           elapsed_time = time.time() - self.start_time
           if elapsed_time > 0:
               avg_rate = self.message_count / elapsed_time
           else:
               avg_rate = 0

           stats = {
               'elapsed_time': elapsed_time,
               'total_messages': self.message_count,
               'average_rate': avg_rate,
               'scan_callbacks': self.scan_callback_count,
               'imu_callbacks': self.imu_callback_count,
               'fusion_enabled': self.fusion_enabled
           }
           return stats

   def main(args=None):
       rclpy.init(args=args)
       node = AdvancedCommunicationNode()

       # Add a timer to periodically print performance stats
       stats_timer = node.create_timer(5.0, lambda: print_stats(node))

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           stats = node.get_performance_stats()
           node.get_logger().info(f'Final stats: {stats}')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   def print_stats(node):
       stats = node.get_performance_stats()
       node.get_logger().info(
           f'Performance - Rate: {stats["average_rate"]:.2f} msg/s, '
           f'Total: {stats["total_messages"]}, '
           f'Fusion: {"ON" if stats["fusion_enabled"] else "OFF"}'
       )

   if __name__ == '__main__':
       main()
   ```

2. **Create a monitoring node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import Float32
   from geometry_msgs.msg import Twist

   class CommunicationMonitor(Node):
       def __init__(self):
           super().__init__('communication_monitor')

           # Subscriptions to monitor communication
           self.obstacle_subscription = self.create_subscription(
               Float32, 'obstacle_distance', self.obstacle_callback, 10)

           # Service client to control fusion
           self.fusion_control_client = self.create_client(
               SetBool, 'control_fusion')

           # Publisher to send commands when needed
           self.command_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

           # Wait for service
           while not self.fusion_control_client.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Fusion control service not available...')

           self.obstacle_threshold = 0.75  # meters
           self.get_logger().info('Communication monitor started')

       def obstacle_callback(self, msg):
           distance = msg.data
           self.get_logger().info(f'Obstacle distance: {distance:.2f}m')

           # If obstacle is too close, send control command
           if distance < self.obstacle_threshold:
               self.get_logger().warn(f'Obstacle at {distance:.2f}m - too close!')
               self.slow_down()

       def slow_down(self):
           # Send a service request to disable fusion temporarily
           request = SetBool.Request()
           request.data = False  # Disable fusion

           future = self.fusion_control_client.call_async(request)
           future.add_done_callback(self.fusion_control_response)

       def fusion_control_response(self, future):
           try:
               response = future.result()
               self.get_logger().info(f'Fusion control response: {response.message}')
           except Exception as e:
               self.get_logger().error(f'Service call failed: {e}')

   def main(args=None):
       rclpy.init(args=args)
       node = CommunicationMonitor()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Shutting down communication monitor')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Test the advanced system:**
   ```bash
   # Terminal 1: Run the advanced communication node
   ros2 run week3_exercises advanced_communication

   # Terminal 2: Run the monitor
   ros2 run week3_exercises communication_monitor

   # Terminal 3: Use ros2cli tools to interact
   ros2 service call /control_fusion std_srvs/srv/SetBool '{data: true}'
   ros2 topic echo /obstacle_distance
   ```

### Expected Output
- Multiple nodes communicating through various patterns
- Data fusion from multiple sensors
- Service-based control of the fusion process
- Performance monitoring and statistics
- Proper error handling and graceful operation

### Submission Requirements
- Complete advanced communication node implementation
- Monitoring node code
- Test results showing multi-node communication
- Performance analysis and statistics
- Explanation of communication patterns used

## Grading Rubric

Each exercise will be graded on the following criteria:

- **Implementation Correctness** (30%): Code works as specified
- **Code Quality** (25%): Well-structured, documented, follows ROS 2 best practices
- **Understanding** (25%): Proper understanding of ROS 2 concepts demonstrated
- **Testing** (20%): Adequate testing and validation performed

## Submission Guidelines

- Submit all exercises as a complete ROS 2 package
- Include a README.md explaining your implementation
- Provide screenshots of successful execution
- Follow proper ROS 2 package structure and conventions
- Late submissions will be penalized by 10% per day

## Resources

- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS 2 Services and Actions](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Service-And-Client.html)
- [ROS 2 Quality of Service](https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html)