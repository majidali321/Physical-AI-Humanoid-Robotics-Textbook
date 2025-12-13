---
sidebar_position: 3
---

# Week 6: Sensor Simulation in Gazebo

## Overview

Sensor simulation is crucial for developing perception systems in humanoid robotics. This section covers how to simulate various types of sensors in Gazebo, including cameras, LiDAR, IMUs, and force/torque sensors. Understanding sensor simulation allows you to develop and test perception algorithms before deploying on real hardware.

## Learning Objectives

By the end of this section, you will be able to:

- Configure and simulate different types of sensors in Gazebo
- Understand the characteristics and limitations of simulated sensors
- Integrate sensor data with ROS 2 for perception algorithms
- Validate sensor simulation against real-world sensor behavior
- Optimize sensor parameters for humanoid robotics applications

## Introduction to Sensor Simulation

### Why Simulate Sensors?

Sensor simulation in Gazebo provides several advantages:

- **Cost-Effective**: No need to purchase expensive sensors for testing
- **Risk-Free**: Test perception algorithms without damaging hardware
- **Controllable**: Create specific scenarios and conditions
- **Repeatable**: Exact same conditions for testing and validation
- **Safe**: Test dangerous scenarios without risk

### Types of Sensors in Gazebo

Gazebo supports simulation of many common robotic sensors:

- **Visual Sensors**: Cameras, depth cameras, RGB-D sensors
- **Range Sensors**: LiDAR, sonar, infrared
- **Inertial Sensors**: IMU, accelerometers, gyroscopes
- **Force/Torque Sensors**: Joint force sensors, tactile sensors
- **GPS and Other Position Sensors**: For outdoor navigation

## Camera Simulation

### Basic Camera Configuration

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head">
    <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_frame</frame_name>
    <topic_name>/camera/image_raw</topic_name>
  </plugin>
</sensor>
```

### Camera Parameters

**Field of View (FOV):**
- Horizontal FOV affects the width of the captured image
- Typical values: 60°-90° for standard cameras, 180°+ for wide-angle

**Image Resolution:**
- Higher resolution = more detail but slower processing
- Balance between quality and performance needs

**Clipping Planes:**
- Near plane: Minimum distance for objects to be visible
- Far plane: Maximum distance for objects to be visible

### Depth Camera Simulation

```xml
<sensor name="depth_camera" type="depth">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="depth_head">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.1</stddev>
    </noise>
  </camera>
  <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <baseline>0.2</baseline>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <point_cloud_cutoff>0.3</point_cloud_cutoff>
    <point_cloud_cutoff_max>3.0</point_cloud_cutoff_max>
    <frame_name>depth_camera_frame</frame_name>
    <topic_name>/camera/depth/image_raw</topic_name>
    <depth_image_topic_name>/camera/depth/image_raw</depth_image_topic_name>
    <point_cloud_topic_name>/camera/depth/points</point_cloud_topic_name>
  </plugin>
</sensor>
```

## LiDAR Simulation

### 2D LiDAR Configuration

```xml
<sensor name="laser" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
    <topic_name>/scan</topic_name>
    <frame_name>laser_frame</frame_name>
  </plugin>
</sensor>
```

### 3D LiDAR Configuration (Velodyne-style)

```xml
<sensor name="velodyne" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>800</samples>
        <resolution>1</resolution>
        <min_angle>-3.141593</min_angle>  <!-- -180 degrees -->
        <max_angle>3.141593</max_angle>   <!-- 180 degrees -->
      </horizontal>
      <vertical>
        <samples>32</samples>
        <resolution>1</resolution>
        <min_angle>-0.436332</min_angle>  <!-- -25 degrees -->
        <max_angle>0.209440</max_angle>   <!-- 12 degrees -->
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>100.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_laser.so">
    <topic_name>/velodyne_points</topic_name>
    <frame_name>velodyne_frame</frame_name>
    <min_range>0.9</min_range>
    <max_range>130.0</max_range>
    <gaussian_noise>0.008</gaussian_noise>
  </plugin>
</sensor>
```

### LiDAR Parameters

**Resolution Parameters:**
- **Samples**: Number of rays in each scan direction
- **Resolution**: Multiplier for sample density
- **Min/Max Angle**: Angular range of the sensor

**Range Parameters:**
- **Min Range**: Closest detectable distance
- **Max Range**: Farthest detectable distance
- **Resolution**: Distance resolution (accuracy)

## IMU Simulation

### Basic IMU Configuration

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.0003</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.0003</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.0003</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
    <topic_name>/imu/data</topic_name>
    <body_name>imu_link</body_name>
    <frame_name>imu_frame</frame_name>
    <update_rate>100</update_rate>
  </plugin>
</sensor>
```

### IMU Parameters

**Angular Velocity Noise:**
- **StdDev**: Standard deviation of noise
- **Bias Mean/StdDev**: Systematic errors in the sensor

**Linear Acceleration Noise:**
- Similar noise parameters for acceleration measurements

## Force/Torque Sensor Simulation

### Joint Force/Torque Sensors

```xml
<sensor name="ft_sensor" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>sensor</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin name="ft_sensor_controller" filename="libgazebo_ros_ft_sensor.so">
    <topic_name>/joint_wrench</topic_name>
    <joint_name>sensor_joint</joint_name>
  </plugin>
</sensor>
```

## Sensor Noise and Realism

### Adding Realistic Noise

Real sensors have various types of noise and imperfections:

```xml
<sensor name="noisy_camera" type="camera">
  <camera>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>  <!-- Noise level -->
    </noise>
  </camera>
</sensor>

<sensor name="noisy_lidar" type="ray">
  <ray>
    <range>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.01</stddev>  <!-- 1cm standard deviation -->
    </noise>
  </ray>
</sensor>
```

### Common Noise Models

**Gaussian Noise:**
- Most common model for sensor noise
- Represents random measurement errors

**Bias:**
- Systematic errors that remain constant
- Important for sensor calibration

**Drift:**
- Slowly changing systematic errors
- Simulates temperature effects, aging

## ROS 2 Integration

### Sensor Data Topics

Sensors publish data to ROS 2 topics following standard message types:

- **Camera**: `sensor_msgs/Image` and `sensor_msgs/CameraInfo`
- **Depth Camera**: `sensor_msgs/Image`, `sensor_msgs/PointCloud2`
- **LiDAR**: `sensor_msgs/LaserScan` or `sensor_msgs/PointCloud2`
- **IMU**: `sensor_msgs/Imu`
- **Force/Torque**: `geometry_msgs/WrenchStamped`

### Example ROS 2 Node Processing Sensor Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from cv_bridge import CvBridge
import numpy as np

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Initialize bridge for camera images
        self.bridge = CvBridge()

        # Subscribe to sensor topics
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)

        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)

        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        self.get_logger().info('Sensor processor initialized')

    def camera_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process image (example: simple edge detection)
            # Your perception algorithm here

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def lidar_callback(self, msg):
        # Process LiDAR data
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges
        valid_ranges = ranges[(ranges >= msg.range_min) &
                             (ranges <= msg.range_max)]

        # Your LiDAR processing algorithm here

    def imu_callback(self, msg):
        # Process IMU data
        orientation = [msg.orientation.x, msg.orientation.y,
                      msg.orientation.z, msg.orientation.w]
        angular_velocity = [msg.angular_velocity.x, msg.angular_velocity.y,
                           msg.angular_velocity.z]
        linear_acceleration = [msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z]

        # Your IMU processing algorithm here

def main(args=None):
    rclpy.init(args=args)
    processor = SensorProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Humanoid-Specific Sensor Considerations

### Head-Mounted Sensors

For humanoid robots, head-mounted sensors are common:

```xml
<!-- Head camera for object recognition -->
<sensor name="head_camera" type="camera">
  <!-- Configuration similar to basic camera -->
  <pose>0 0 0.1 0 0 0</pose>  <!-- Position relative to head link -->
</sensor>

<!-- Stereo cameras for depth perception -->
<sensor name="left_eye" type="camera">
  <pose>-0.05 0 0 0 0 0</pose>  <!-- Left of center -->
</sensor>
<sensor name="right_eye" type="camera">
  <pose>0.05 0 0 0 0 0</pose>   <!-- Right of center -->
</sensor>
```

### Body-Mounted Sensors

```xml
<!-- IMU in torso for balance -->
<sensor name="torso_imu" type="imu">
  <pose>0 0 0 0 0 0</pose>
</sensor>

<!-- Force sensors in feet for balance -->
<sensor name="left_foot_ft" type="force_torque">
  <pose>0.05 0 -0.05 0 0 0</pose>  <!-- Front center of foot -->
</sensor>
```

## Sensor Validation

### Validation Techniques

1. **Compare with Real Sensors**: When possible, compare simulated vs. real sensor data
2. **Statistical Analysis**: Analyze noise characteristics and distributions
3. **Perception Algorithm Testing**: Verify algorithms work on both simulated and real data
4. **Cross-Sensor Validation**: Use multiple sensors to validate each other

### Validation Metrics

- **Accuracy**: How close simulated measurements are to expected values
- **Precision**: Consistency of repeated measurements
- **Latency**: Time delay in sensor data
- **Update Rate**: How frequently data is published

## Troubleshooting Sensor Issues

### Common Problems

**Problem: Sensor not publishing data**
- Check sensor configuration in URDF/SDF
- Verify plugin is properly loaded
- Check topic names and permissions

**Problem: Sensor data is noisy or inaccurate**
- Review noise parameters in configuration
- Check simulation time step and physics parameters
- Verify sensor placement and orientation

**Problem: Performance issues with sensors**
- Reduce update rates or resolution
- Simplify sensor models if possible
- Check GPU/CPU utilization

### Debugging Commands

```bash
# List all sensor topics
ros2 topic list | grep -E "(camera|scan|imu)"

# Monitor sensor data
ros2 topic echo /camera/image_raw
ros2 topic echo /scan
ros2 topic echo /imu/data

# Check sensor frame transforms
ros2 run tf2_tools view_frames
```

## Best Practices

### Sensor Placement

1. **Strategic Positioning**: Place sensors where they'll be most useful
2. **Redundancy**: Consider multiple sensors for critical functions
3. **Clear Field of View**: Ensure sensors have unobstructed views
4. **Protection**: Consider sensor protection in collision scenarios

### Performance Optimization

1. **Appropriate Update Rates**: Balance quality with performance
2. **Selective Simulation**: Only simulate sensors you're actively using
3. **Parameter Tuning**: Optimize noise and accuracy parameters
4. **Resource Management**: Monitor and manage computational resources

## Summary

Sensor simulation in Gazebo provides powerful capabilities for developing perception systems in humanoid robotics. Understanding how to configure, integrate, and validate simulated sensors is essential for effective robotics development. Properly simulated sensors enable comprehensive testing of perception algorithms before deployment on real hardware.