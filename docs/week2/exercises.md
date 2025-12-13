---
sidebar_position: 4
---

# Week 2 Exercises: Sensor Systems

## Exercise 1: Sensor Data Publisher Implementation

### Objective
Implement a ROS 2 node that publishes simulated sensor data using standard message types.

### Requirements
1. Create a ROS 2 package called `sensor_exercises`
2. Implement a node that publishes:
   - LaserScan messages (simulated LiDAR data)
   - Image messages (simulated camera data)
   - IMU messages (simulated inertial data)
3. Use proper header information including timestamps and frame IDs
4. Implement parameter configuration for sensor characteristics

### Implementation Steps
1. Create the package:
   ```bash
   cd ~/physical_ai_ws/src
   ros2 pkg create --build-type ament_python sensor_exercises
   ```

2. Implement the sensor simulator node:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan, Image, Imu
   from cv_bridge import CvBridge
   import numpy as np
   import math

   class SensorSimulator(Node):
       def __init__(self):
           super().__init__('sensor_simulator')

           # Declare parameters
           self.declare_parameter('publish_rate', 10.0)
           self.declare_parameter('laser_range_min', 0.1)
           self.declare_parameter('laser_range_max', 10.0)

           # Create publishers
           self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)
           self.image_pub = self.create_publisher(Image, 'image_raw', 10)
           self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)

           # Create timer
           publish_rate = self.get_parameter('publish_rate').value
           self.timer = self.create_timer(1.0/publish_rate, self.publish_sensor_data)

           # Initialize CV bridge
           self.bridge = CvBridge()

           self.get_logger().info('Sensor simulator node started')

       def publish_sensor_data(self):
           # Publish simulated laser scan
           scan_msg = self.create_laser_scan()
           self.scan_pub.publish(scan_msg)

           # Publish simulated image
           image_msg = self.create_image()
           self.image_pub.publish(image_msg)

           # Publish simulated IMU
           imu_msg = self.create_imu()
           self.imu_pub.publish(imu_msg)

       def create_laser_scan(self):
           msg = LaserScan()
           msg.header.stamp = self.get_clock().now().to_msg()
           msg.header.frame_id = 'laser_frame'

           # Set scan parameters
           msg.angle_min = -math.pi / 2
           msg.angle_max = math.pi / 2
           msg.angle_increment = math.pi / 180  # 1 degree
           msg.time_increment = 0.0
           msg.scan_time = 0.0
           msg.range_min = self.get_parameter('laser_range_min').value
           msg.range_max = self.get_parameter('laser_range_max').value

           # Generate simulated ranges (with some obstacles)
           num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1
           ranges = []
           for i in range(num_readings):
               angle = msg.angle_min + i * msg.angle_increment
               # Simulate a circular obstacle at 2m, 0rad
               distance = 2.0 + 0.5 * math.sin(5 * angle)  # Add some variation
               ranges.append(distance)

           msg.ranges = ranges
           msg.intensities = [1.0] * len(ranges)

           return msg

       def create_image(self):
           # Create a simulated image (simple pattern)
           img_array = np.zeros((480, 640, 3), dtype=np.uint8)
           # Add some simulated features
           cv2.rectangle(img_array, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue square
           cv2.circle(img_array, (320, 240), 50, (0, 255, 0), -1)  # Green circle

           return self.bridge.cv2_to_imgmsg(img_array, encoding="bgr8")

       def create_imu(self):
           from geometry_msgs.msg import Vector3, Quaternion
           msg = Imu()
           msg.header.stamp = self.get_clock().now().to_msg()
           msg.header.frame_id = 'imu_frame'

           # Simulate IMU data (with some noise)
           msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
           msg.angular_velocity = Vector3(x=0.1, y=0.05, z=0.02)
           msg.linear_acceleration = Vector3(x=0.0, y=0.0, z=9.81)  # Gravity

           return msg

   def main(args=None):
       rclpy.init(args=args)
       node = SensorSimulator()
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

3. Update the package.xml with proper dependencies:
   ```xml
   <depend>rclpy</depend>
   <depend>sensor_msgs</depend>
   <depend>cv_bridge</depend>
   <depend>std_msgs</depend>
   <depend>geometry_msgs</depend>
   ```

4. Make the Python file executable and test:
   ```bash
   chmod +x sensor_exercises/sensor_exercises/sensor_simulator.py
   ros2 run sensor_exercises sensor_simulator
   ```

### Expected Output
- Sensor data published at the specified rate
- Messages visible with `ros2 topic echo`
- Proper frame IDs and timestamps

### Submission Requirements
- Complete source code
- Screenshots of topic echo output
- Configuration parameters documentation

## Exercise 2: Sensor Data Processing Node

### Objective
Create a ROS 2 node that subscribes to sensor data and performs basic processing.

### Requirements
1. Create a node that subscribes to the LaserScan topic from Exercise 1
2. Implement obstacle detection algorithm
3. Filter out noise from the sensor data
4. Publish processed results

### Implementation Steps
1. Create a new node called `sensor_processor`:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan
   from std_msgs.msg import Float32
   import numpy as np

   class SensorProcessor(Node):
       def __init__(self):
           super().__init__('sensor_processor')

           # Subscribe to laser scan
           self.scan_subscription = self.create_subscription(
               LaserScan, 'scan', self.scan_callback, 10)

           # Publish processed data
           self.obstacle_distance_pub = self.create_publisher(
               Float32, 'obstacle_distance', 10)

           self.get_logger().info('Sensor processor node started')

       def scan_callback(self, msg):
           # Filter out invalid ranges
           valid_ranges = [r for r in msg.ranges if 0 < r < float('inf')]

           if valid_ranges:
               # Find minimum distance (closest obstacle)
               min_distance = min(valid_ranges)

               # Publish obstacle distance
               obstacle_msg = Float32()
               obstacle_msg.data = min_distance
               self.obstacle_distance_pub.publish(obstacle_msg)

               self.get_logger().info(f'Closest obstacle: {min_distance:.2f}m')

   def main(args=None):
       rclpy.init(args=args)
       node = SensorProcessor()
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

2. Test the processing node with the simulator:
   ```bash
   # Terminal 1: Run simulator
   ros2 run sensor_exercises sensor_simulator

   # Terminal 2: Run processor
   ros2 run sensor_exercises sensor_processor
   ```

### Expected Output
- Processed obstacle distance published
- Log messages showing closest obstacle distance
- No errors in processing

### Submission Requirements
- Complete source code
- Test output showing processing results
- Explanation of processing algorithm

## Exercise 3: Coordinate Frame Transformation

### Objective
Implement a node that demonstrates coordinate frame transformations with sensor data.

### Requirements
1. Create static transforms for sensor frames
2. Transform sensor data between coordinate frames
3. Visualize the transformation relationship

### Implementation Steps
1. Create a transform publisher node:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   import tf2_ros
   from geometry_msgs.msg import TransformStamped
   import math

   class SensorTfPublisher(Node):
       def __init__(self):
           super().__init__('sensor_tf_publisher')

           # Create transform broadcaster
           self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

           # Publish static transforms
           self.publish_static_transforms()

       def publish_static_transforms(self):
           # Robot base to laser frame transform
           t = TransformStamped()
           t.header.stamp = self.get_clock().now().to_msg()
           t.header.frame_id = 'base_link'
           t.child_frame_id = 'laser_frame'

           t.transform.translation.x = 0.2  # 20cm forward
           t.transform.translation.y = 0.0
           t.transform.translation.z = 0.5  # 50cm up

           # No rotation (laser aligned with base)
           t.transform.rotation.x = 0.0
           t.transform.rotation.y = 0.0
           t.transform.rotation.z = 0.0
           t.transform.rotation.w = 1.0

           self.tf_broadcaster.sendTransform(t)

   def main(args=None):
       rclpy.init(args=args)
       node = SensorTfPublisher()
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

2. Test transforms with RViz2:
   ```bash
   # Run all nodes
   ros2 run sensor_exercises sensor_simulator
   ros2 run sensor_exercises sensor_tf_publisher
   rviz2
   ```

3. In RViz2, add TF display to visualize the coordinate frames.

### Expected Output
- TF tree showing base_link and laser_frame relationship
- Proper transformation visualization in RViz2
- Coordinate frames properly aligned

### Submission Requirements
- Transform publisher source code
- RViz2 screenshots showing TF tree
- Explanation of coordinate frame relationships

## Exercise 4: Multi-Sensor Fusion

### Objective
Implement a simple sensor fusion algorithm combining data from multiple sensors.

### Requirements
1. Subscribe to multiple sensor topics
2. Implement basic fusion algorithm
3. Publish fused results
4. Compare fused results with individual sensor data

### Implementation Steps
1. Create a fusion node that combines IMU and simulated joint state data:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Imu, JointState
   from geometry_msgs.msg import Vector3
   import numpy as np

   class SensorFusionNode(Node):
       def __init__(self):
           super().__init__('sensor_fusion_node')

           # Subscribe to multiple sensors
           self.imu_subscription = self.create_subscription(
               Imu, 'imu/data', self.imu_callback, 10)
           self.joint_subscription = self.create_subscription(
               JointState, 'joint_states', self.joint_callback, 10)

           # Publish fused orientation
           self.fused_orientation_pub = self.create_publisher(
               Vector3, 'fused_orientation', 10)

           # Store latest readings
           self.latest_imu = None
           self.latest_joints = None

       def imu_callback(self, msg):
           self.latest_imu = msg
           self.perform_fusion()

       def joint_callback(self, msg):
           self.latest_joints = msg
           self.perform_fusion()

       def perform_fusion(self):
           if self.latest_imu is None:
               return

           # Simple fusion: extract roll, pitch, yaw from IMU
           # In practice, use more sophisticated fusion algorithms
           imu = self.latest_imu

           # Convert quaternion to Euler angles (simplified)
           # In practice, use tf2 or quaternion libraries
           orientation = Vector3()
           orientation.x = self.quaternion_to_roll(imu.orientation)
           orientation.y = self.quaternion_to_pitch(imu.orientation)
           orientation.z = self.quaternion_to_yaw(imu.orientation)

           self.fused_orientation_pub.publish(orientation)

       def quaternion_to_roll(self, q):
           # Simplified conversion - use proper library in practice
           sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
           cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
           return math.atan2(sinr_cosp, cosr_cosp)

       def quaternion_to_pitch(self, q):
           # Simplified conversion
           sinp = 2 * (q.w * q.y - q.z * q.x)
           if abs(sinp) >= 1:
               return math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
           return math.asin(sinp)

       def quaternion_to_yaw(self, q):
           # Simplified conversion
           siny_cosp = 2 * (q.w * q.z + q.x * q.y)
           cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
           return math.atan2(siny_cosp, cosy_cosp)

   def main(args=None):
       rclpy.init(args=args)
       node = SensorFusionNode()
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

2. Test the fusion node with sensor simulator:
   ```bash
   # Terminal 1
   ros2 run sensor_exercises sensor_simulator

   # Terminal 2
   ros2 run sensor_exercises sensor_fusion_node

   # Terminal 3 - monitor results
   ros2 topic echo /fused_orientation
   ```

### Expected Output
- Fused orientation data published
- Reasonable roll, pitch, yaw values
- No errors in fusion process

### Submission Requirements
- Fusion algorithm source code
- Test results showing fused data
- Comparison between individual and fused sensor data

## Exercise 5: Sensor Diagnostics and Health Monitoring

### Objective
Implement a node that monitors sensor health and provides diagnostic information.

### Requirements
1. Monitor sensor data quality
2. Detect sensor failures or anomalies
3. Publish diagnostic messages
4. Implement basic sensor calibration

### Implementation Steps
1. Create a sensor diagnostic node:
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan
   from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
   import numpy as np

   class SensorDiagnosticsNode(Node):
       def __init__(self):
           super().__init__('sensor_diagnostics')

           # Subscribe to sensor data
           self.scan_subscription = self.create_subscription(
               LaserScan, 'scan', self.scan_callback, 10)

           # Publish diagnostics
           self.diag_publisher = self.create_publisher(
               DiagnosticArray, '/diagnostics', 10)

           # Diagnostic timer
           self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

           # Sensor state tracking
           self.last_scan_time = None
           self.scan_frequency = 0
           self.data_quality_score = 0.0
           self.error_count = 0

       def scan_callback(self, msg):
           # Update timing information
           current_time = self.get_clock().now().nanoseconds
           if self.last_scan_time is not None:
               time_diff = (current_time - self.last_scan_time) / 1e9
               if time_diff > 0:
                   self.scan_frequency = 1.0 / time_diff

           self.last_scan_time = current_time

           # Evaluate data quality
           self.evaluate_data_quality(msg)

       def evaluate_data_quality(self, scan_msg):
           # Check for valid ranges
           valid_ranges = [r for r in scan_msg.ranges if 0 < r < float('inf')]
           total_ranges = len(scan_msg.ranges)

           if total_ranges > 0:
               quality_ratio = len(valid_ranges) / total_ranges
               self.data_quality_score = quality_ratio

               # Check for unexpected patterns
               if len(valid_ranges) == 0:
                   self.error_count += 1
                   self.get_logger().warn('All ranges invalid!')

       def publish_diagnostics(self):
           diag_array = DiagnosticArray()
           diag_array.header.stamp = self.get_clock().now().to_msg()

           status = DiagnosticStatus()
           status.name = 'Laser Scanner Health'

           # Determine status level
           if self.error_count > 5:
               status.level = DiagnosticStatus.ERROR
               status.message = 'Sensor error detected'
               self.error_count = 0  # Reset after reporting
           elif self.data_quality_score < 0.5:
               status.level = DiagnosticStatus.WARN
               status.message = 'Low data quality'
           else:
               status.level = DiagnosticStatus.OK
               status.message = 'Sensor operational'

           # Add key-value pairs
           status.values.extend([
               {'key': 'Frequency (Hz)', 'value': f'{self.scan_frequency:.2f}'},
               {'key': 'Data Quality', 'value': f'{self.data_quality_score:.2f}'},
               {'key': 'Valid Ranges', 'value': f'{int(self.data_quality_score * 100)}%'},
               {'key': 'Status', 'value': status.message}
           ])

           diag_array.status.append(status)
           self.diag_publisher.publish(diag_array)

   def main(args=None):
       rclpy.init(args=args)
       node = SensorDiagnosticsNode()
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

2. Test diagnostics with the sensor simulator:
   ```bash
   # Terminal 1
   ros2 run sensor_exercises sensor_simulator

   # Terminal 2
   ros2 run sensor_exercises sensor_diagnostics

   # Terminal 3 - monitor diagnostics
   ros2 topic echo /diagnostics
   ```

### Expected Output
- Diagnostic messages published regularly
- Health status based on sensor data quality
- Error detection and reporting

### Submission Requirements
- Diagnostic node source code
- Test results showing diagnostic output
- Explanation of diagnostic criteria used

## Grading Rubric

Each exercise will be graded on the following criteria:

- **Implementation Correctness** (30%): Code works as specified
- **Code Quality** (25%): Well-structured, documented, follows ROS 2 best practices
- **Understanding** (25%): Proper understanding of sensor concepts demonstrated
- **Testing** (20%): Adequate testing and validation performed

## Submission Guidelines

- Submit all exercises as a single package or separate files
- Include a README.md explaining your implementation
- Provide screenshots of successful execution
- Follow proper ROS 2 package structure and conventions
- Late submissions will be penalized by 10% per day

## Resources

- [ROS 2 Sensor Messages Documentation](https://docs.ros.org/en/humble/p(sensor_msgs.html))
- [TF2 Tutorials](https://docs.ros.org/en/humble/Tutorials/Advanced/Tf2/index.html)
- [Robot Operating System Sensor Integration](https://example.com/ros-sensors)