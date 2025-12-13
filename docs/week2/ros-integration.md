---
sidebar_position: 3
---

# ROS 2 Sensor Integration

## Sensor Drivers in ROS 2

### Overview
Sensor drivers are ROS 2 nodes that interface with physical sensors, converting raw sensor data into standardized ROS 2 message types. These drivers form the foundation of any robot's perception system.

### Standard Sensor Driver Structure
A typical sensor driver follows this pattern:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu, JointState
import time

class SensorDriver(Node):
    def __init__(self):
        super().__init__('sensor_driver')

        # Create publishers for sensor data
        self.scan_publisher = self.create_publisher(LaserScan, 'scan', 10)
        self.image_publisher = self.create_publisher(Image, 'image_raw', 10)
        self.imu_publisher = self.create_publisher(Imu, 'imu/data', 10)

        # Timer for data acquisition
        self.timer = self.create_timer(0.1, self.acquire_sensor_data)

        # Initialize sensor hardware
        self.initialize_hardware()

    def initialize_hardware(self):
        """Initialize physical sensor hardware"""
        # Hardware-specific initialization code
        pass

    def acquire_sensor_data(self):
        """Acquire data from physical sensors and publish ROS 2 messages"""
        # Read raw data from sensors
        raw_scan_data = self.read_lidar()
        raw_image_data = self.read_camera()
        raw_imu_data = self.read_imu()

        # Convert to ROS 2 messages
        scan_msg = self.create_scan_message(raw_scan_data)
        image_msg = self.create_image_message(raw_image_data)
        imu_msg = self.create_imu_message(raw_imu_data)

        # Publish messages
        self.scan_publisher.publish(scan_msg)
        self.image_publisher.publish(image_msg)
        self.imu_publisher.publish(imu_msg)
```

### Hardware Abstraction Layer
It's important to separate hardware-specific code from ROS 2-specific code:

```python
class HardwareInterface:
    """Abstract interface for sensor hardware"""
    def read_sensor(self):
        raise NotImplementedError

class RealLidarInterface(HardwareInterface):
    def read_sensor(self):
        # Code to read from actual LiDAR hardware
        pass

class SimulatedLidarInterface(HardwareInterface):
    def read_sensor(self):
        # Code to simulate LiDAR data
        pass
```

## Common Sensor Message Types

### LaserScan Messages
For LiDAR and range finder sensors:

```python
from sensor_msgs.msg import LaserScan
import math

def create_laser_scan_msg():
    msg = LaserScan()

    # Timing information
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = 'laser_frame'

    # Measurement parameters
    msg.angle_min = -math.pi / 2  # -90 degrees
    msg.angle_max = math.pi / 2   # 90 degrees
    msg.angle_increment = math.pi / 180  # 1 degree

    msg.time_increment = 0.0
    msg.scan_time = 0.0

    # Range limits
    msg.range_min = 0.1
    msg.range_max = 10.0

    # Measurement data
    num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment) + 1
    msg.ranges = [float('inf')] * num_readings  # Initialize with max range
    msg.intensities = [0.0] * num_readings

    return msg
```

### Image Messages
For camera sensors:

```python
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

def create_image_msg(cv_image):
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    image_msg.header.stamp = self.get_clock().now().to_msg()
    image_msg.header.frame_id = 'camera_frame'
    return image_msg
```

### IMU Messages
For inertial measurement units:

```python
from sensor_msgs.msg import Imu
import geometry_msgs.msg

def create_imu_msg(acceleration, angular_velocity, orientation):
    msg = Imu()

    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = 'imu_frame'

    # Orientation (quaternion)
    msg.orientation = orientation

    # Angular velocity
    msg.angular_velocity = angular_velocity

    # Linear acceleration
    msg.linear_acceleration = acceleration

    # Covariance matrices (set to appropriate values)
    msg.orientation_covariance = [0.0] * 9
    msg.angular_velocity_covariance = [0.0] * 9
    msg.linear_acceleration_covariance = [0.0] * 9

    return msg
```

## Sensor Processing Pipeline

### Data Preprocessing Node
Often, raw sensor data needs preprocessing before it's useful:

```python
class SensorPreprocessor(Node):
    def __init__(self):
        super().__init__('sensor_preprocessor')

        # Subscriptions to raw sensor data
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publishers for processed data
        self.processed_scan_publisher = self.create_publisher(
            LaserScan, 'scan_filtered', 10)
        self.processed_imu_publisher = self.create_publisher(
            Imu, 'imu/data_filtered', 10)

    def scan_callback(self, msg):
        # Apply filtering to remove noise
        filtered_msg = self.filter_laser_scan(msg)
        self.processed_scan_publisher.publish(filtered_msg)

    def filter_laser_scan(self, scan_msg):
        # Example: replace inf values with max range
        filtered_ranges = []
        for r in scan_msg.ranges:
            if r == float('inf'):
                filtered_ranges.append(scan_msg.range_max)
            elif r < scan_msg.range_min:
                filtered_ranges.append(scan_msg.range_min)
            else:
                filtered_ranges.append(r)

        scan_msg.ranges = filtered_ranges
        return scan_msg
```

### Sensor Fusion Node
Combining data from multiple sensors:

```python
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np

class ImuFusionNode(Node):
    def __init__(self):
        super().__init__('imu_fusion')

        # Subscribe to multiple IMU sensors
        self.imu1_subscription = self.create_subscription(
            Imu, 'imu1/data', self.imu1_callback, 10)
        self.imu2_subscription = self.create_subscription(
            Imu, 'imu2/data', self.imu2_callback, 10)

        # Publisher for fused data
        self.fused_imu_publisher = self.create_publisher(
            Imu, 'imu/fused', 10)

        # Store latest readings
        self.latest_imu1 = None
        self.latest_imu2 = None

    def imu1_callback(self, msg):
        self.latest_imu1 = msg
        self.fuse_imu_data()

    def imu2_callback(self, msg):
        self.latest_imu2 = msg
        self.fuse_imu_data()

    def fuse_imu_data(self):
        if self.latest_imu1 is None or self.latest_imu2 is None:
            return

        # Simple averaging fusion (in practice, use more sophisticated methods)
        fused_msg = Imu()
        fused_msg.header.stamp = self.get_clock().now().to_msg()
        fused_msg.header.frame_id = 'base_link'

        # Average orientation (simplified - use proper quaternion averaging in practice)
        fused_msg.orientation.x = (self.latest_imu1.orientation.x + self.latest_imu2.orientation.x) / 2.0
        fused_msg.orientation.y = (self.latest_imu1.orientation.y + self.latest_imu2.orientation.y) / 2.0
        fused_msg.orientation.z = (self.latest_imu1.orientation.z + self.latest_imu2.orientation.z) / 2.0
        fused_msg.orientation.w = (self.latest_imu1.orientation.w + self.latest_imu2.orientation.w) / 2.0

        # Average angular velocity
        fused_msg.angular_velocity.x = (self.latest_imu1.angular_velocity.x + self.latest_imu2.angular_velocity.x) / 2.0
        fused_msg.angular_velocity.y = (self.latest_imu1.angular_velocity.y + self.latest_imu2.angular_velocity.y) / 2.0
        fused_msg.angular_velocity.z = (self.latest_imu1.angular_velocity.z + self.latest_imu2.angular_velocity.z) / 2.0

        # Average linear acceleration
        fused_msg.linear_acceleration.x = (self.latest_imu1.linear_acceleration.x + self.latest_imu2.linear_acceleration.x) / 2.0
        fused_msg.linear_acceleration.y = (self.latest_imu1.linear_acceleration.y + self.latest_imu2.linear_acceleration.y) / 2.0
        fused_msg.linear_acceleration.z = (self.latest_imu1.linear_acceleration.z + self.latest_imu2.linear_acceleration.z) / 2.0

        self.fused_imu_publisher.publish(fused_msg)
```

## Coordinate Frames and tf2

### Importance of Coordinate Frames
In robotics, sensors are mounted at specific locations and orientations on the robot. The `tf2` library manages these relationships:

```python
import tf2_ros
from geometry_msgs.msg import TransformStamped

class SensorTfPublisher(Node):
    def __init__(self):
        super().__init__('sensor_tf_publisher')

        # Create transform broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publish static transforms
        self.publish_sensor_transforms()

    def publish_sensor_transforms(self):
        """Publish static transforms for sensor frames"""
        # Example: camera mounted 0.1m forward and 0.2m up from base_link
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_frame'

        t.transform.translation.x = 0.1
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.2

        # No rotation (camera aligned with base frame)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
```

### Using Transforms with Sensor Data
```python
import tf2_ros
from tf2_ros import TransformException

class SensorProcessorWithTf(Node):
    def __init__(self):
        super().__init__('sensor_processor_with_tf')

        # Create transform buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to sensor data
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

    def scan_callback(self, scan_msg):
        try:
            # Get transform from sensor frame to robot base frame
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # Target frame
                scan_msg.header.frame_id,  # Source frame
                rclpy.time.Time(),  # Latest available
                rclpy.duration.Duration(seconds=1.0))  # Timeout

            # Process scan data using the transform
            self.process_scan_with_transform(scan_msg, transform)

        except TransformException as ex:
            self.get_logger().error(f'Could not transform laser scan: {ex}')
```

## Sensor Configuration and Parameters

### Using ROS 2 Parameters for Sensor Configuration
```python
class ConfigurableSensorDriver(Node):
    def __init__(self):
        super().__init__('configurable_sensor_driver')

        # Declare parameters with defaults
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('range_min', 0.1)
        self.declare_parameter('range_max', 10.0)
        self.declare_parameter('sensor_frame_id', 'laser_frame')

        # Get parameter values
        self.publish_rate = self.get_parameter('publish_rate').value
        self.range_min = self.get_parameter('range_min').value
        self.range_max = self.get_parameter('range_max').value
        self.sensor_frame_id = self.get_parameter('sensor_frame_id').value

        # Create publisher
        self.publisher = self.create_publisher(LaserScan, 'scan', 10)

        # Create timer based on parameter
        self.timer = self.create_timer(1.0/self.publish_rate, self.publish_scan)

    def publish_scan(self):
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = self.sensor_frame_id
        scan_msg.range_min = self.range_min
        scan_msg.range_max = self.range_max
        # ... fill in other fields
        self.publisher.publish(scan_msg)
```

## Launch Files for Sensor Systems

### Example Launch File
```python
# launch/sensor_system.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # LiDAR driver
        Node(
            package='your_sensor_package',
            executable='lidar_driver',
            name='lidar_driver',
            parameters=[
                {'range_min': 0.1},
                {'range_max': 30.0},
                {'scan_topic': 'scan'},
            ],
            remappings=[
                ('scan', 'front_scan')
            ]
        ),

        # Camera driver
        Node(
            package='your_sensor_package',
            executable='camera_driver',
            name='camera_driver',
            parameters=[
                {'camera_name': 'front_camera'},
                {'image_width': 640},
                {'image_height': 480},
            ]
        ),

        # IMU driver
        Node(
            package='your_sensor_package',
            executable='imu_driver',
            name='imu_driver',
            parameters=[
                {'frame_id': 'imu_link'},
                {'calibration_file': 'imu_calibration.yaml'}
            ]
        ),

        # Sensor processing node
        Node(
            package='your_sensor_package',
            executable='sensor_processor',
            name='sensor_processor',
            parameters=[
                {'use_imu': True},
                {'use_lidar': True}
            ]
        )
    ])
```

## Best Practices for Sensor Integration

### Error Handling
```python
def safe_sensor_read(self):
    try:
        raw_data = self.read_hardware()
        if raw_data is None:
            self.get_logger().warn('Sensor returned None')
            return None

        # Validate data ranges
        if not self.validate_sensor_data(raw_data):
            self.get_logger().warn('Sensor data validation failed')
            return None

        return raw_data

    except Exception as e:
        self.get_logger().error(f'Sensor read failed: {e}')
        return None
```

### Data Validation
```python
def validate_sensor_data(self, data):
    # Check for NaN or infinite values
    if any(math.isnan(x) or math.isinf(x) for x in data):
        return False

    # Check for expected data ranges
    if min(data) < self.expected_min or max(data) > self.expected_max:
        return False

    return True
```

### Monitoring and Diagnostics
```python
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

class SensorWithDiagnostics(Node):
    def __init__(self):
        super().__init__('sensor_with_diagnostics')

        self.diag_publisher = self.create_publisher(
            DiagnosticArray, '/diagnostics', 10)

        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

    def publish_diagnostics(self):
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = 'Lidar Sensor'
        status.level = DiagnosticStatus.OK
        status.message = 'Sensor operational'

        # Add key-value pairs for sensor status
        status.values.extend([
            {'key': 'Temperature', 'value': '25.0 C'},
            {'key': 'Readings/sec', 'value': '10'},
            {'key': 'Status', 'value': 'OK'}
        ])

        diag_array.status.append(status)
        self.diag_publisher.publish(diag_array)
```

## Summary

ROS 2 provides a comprehensive framework for sensor integration in robotic systems. By following standardized message types, implementing proper coordinate frame management, and using parameterized configurations, you can create robust and maintainable sensor systems for humanoid robots. The modular approach allows for easy testing, calibration, and replacement of sensor components while maintaining system integrity.