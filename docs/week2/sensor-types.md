---
sidebar_position: 2
---

# Sensor Types and Applications

## Proprioceptive Sensors

### Joint Encoders
Joint encoders measure the position of robot joints, providing critical feedback for control systems.

#### Types of Encoders
- **Incremental Encoders**: Measure relative position changes
- **Absolute Encoders**: Provide absolute position information
- **Optical Encoders**: Use light interruption for position detection
- **Magnetic Encoders**: Use magnetic fields for position sensing

#### Implementation in ROS 2
```python
from sensor_msgs.msg import JointState
import rclpy
from rclpy.node import Node

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)

    def publish_joint_states(self):
        msg = JointState()
        msg.name = ['joint1', 'joint2', 'joint3']
        msg.position = [0.1, 0.2, 0.3]  # Radians
        msg.velocity = [0.0, 0.0, 0.0]  # rad/s
        msg.effort = [0.0, 0.0, 0.0]   # N*m
        self.publisher.publish(msg)
```

#### Applications
- Joint position feedback for control
- Kinematic state estimation
- Collision detection
- Calibration procedures

### Force/Torque Sensors
Force/torque sensors measure forces and torques applied to the robot, crucial for manipulation and interaction.

#### Types
- **6-axis F/T Sensors**: Measure forces in 3 directions and torques around 3 axes
- **Strain Gauge Sensors**: Measure deformation under load
- **Piezoelectric Sensors**: Generate electrical charge under mechanical stress

#### Implementation Considerations
- Mounting location affects measurement accuracy
- Temperature compensation may be needed
- Filtering to reduce noise is often necessary

### Inertial Measurement Units (IMUs)
IMUs measure acceleration, angular velocity, and sometimes orientation, essential for balance and navigation.

#### Components
- **Accelerometer**: Measures linear acceleration
- **Gyroscope**: Measures angular velocity
- **Magnetometer**: Measures magnetic field (for orientation)

#### ROS 2 Message Type
```python
from sensor_msgs.msg import Imu

# IMU message includes:
# - Orientation (quaternion)
# - Angular velocity
# - Linear acceleration
# - Covariance matrices for uncertainty
```

#### Applications in Humanoid Robotics
- Balance control
- Gait analysis
- Fall detection
- Motion tracking

## Exteroceptive Sensors

### Vision Sensors
Vision sensors provide rich environmental information, making them crucial for humanoid robots.

#### Camera Types
- **RGB Cameras**: Provide color image data
- **Depth Cameras**: Provide depth information per pixel
- **Stereo Cameras**: Use two cameras to compute depth
- **Thermal Cameras**: Detect heat signatures

#### ROS 2 Vision Integration
```python
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Process image with OpenCV
        processed_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
```

#### Common Vision Tasks
- Object detection and recognition
- Person detection and tracking
- Scene understanding
- Visual SLAM (Simultaneous Localization and Mapping)

### Range Sensors

#### LiDAR (Light Detection and Ranging)
LiDAR sensors provide accurate distance measurements using laser light.

##### Advantages
- High accuracy and precision
- Works in various lighting conditions
- Fast update rates

##### Disadvantages
- Expensive compared to other sensors
- Can be affected by weather conditions
- Limited resolution compared to cameras

##### ROS 2 Message Type
```python
from sensor_msgs.msg import LaserScan

# LaserScan message includes:
# - Angle min/max and increment
# - Time increment between measurements
# - Range min/max
# - Intensity values
# - Range measurements
```

#### Ultrasonic Sensors
Simple and cost-effective for short-range distance measurement.

##### Applications
- Obstacle detection
- Cliff detection
- Simple navigation

### Tactile Sensors
Tactile sensors provide information about physical contact and pressure.

#### Types
- **Contact Sensors**: Binary touch detection
- **Pressure Sensors**: Measure force magnitude
- **Tactile Arrays**: Distributed pressure sensing

#### Applications in Humanoid Robots
- Grasp stability monitoring
- Surface texture recognition
- Safe human interaction
- Tool use feedback

### Audio Sensors
Microphones enable humanoid robots to perceive and respond to sound.

#### Applications
- Voice command recognition
- Sound source localization
- Environmental monitoring
- Human-robot interaction

## Sensor Fusion

### Why Sensor Fusion?
Individual sensors have limitations. Sensor fusion combines multiple sensors to:
- Improve accuracy and precision
- Increase reliability through redundancy
- Extend operational capabilities
- Handle sensor failures gracefully

### Common Fusion Techniques

#### Kalman Filtering
Optimal estimation for linear systems with Gaussian noise.

```python
import numpy as np
from scipy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  # State transition model
        self.H = H  # Observation model
        self.Q = Q  # Process noise
        self.R = R  # Observation noise
        self.P = P  # Error covariance
        self.x = x  # State estimate

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)  # Innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(len(self.x))
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
```

#### Particle Filtering
Non-parametric approach for non-linear, non-Gaussian systems.

#### Complementary Filtering
Simple approach for combining sensors with different frequency characteristics.

## Sensor Calibration

### Importance of Calibration
Sensors often have systematic errors that must be corrected for accurate operation.

### Common Calibration Procedures
- **Camera Calibration**: Determine intrinsic and extrinsic parameters
- **IMU Calibration**: Correct for bias, scale factor, and misalignment
- **LiDAR Calibration**: Align with robot coordinate frame
- **Multi-sensor Calibration**: Determine relationships between sensors

### ROS 2 Calibration Tools
- `camera_calibration` package
- `imu_calib` tools
- `robot_calibration` package

## Challenges in Humanoid Sensor Systems

### Form Factor Constraints
- Limited space for sensors
- Need to maintain human-like appearance
- Cable management challenges

### Power and Processing
- Battery life limitations
- Real-time processing requirements
- Heat dissipation

### Environmental Robustness
- Dust, moisture, and temperature variations
- Electromagnetic interference
- Physical impact protection

## Best Practices

### Sensor Selection
- Match sensor capabilities to task requirements
- Consider environmental conditions
- Balance performance with cost and complexity

### Data Processing
- Implement appropriate filtering to reduce noise
- Validate sensor data before use
- Monitor sensor health and calibration

### Integration
- Use standardized ROS 2 message types
- Implement proper error handling
- Design modular sensor processing nodes

## Summary

Sensor systems are fundamental to robot perception and interaction. Understanding different sensor types, their characteristics, and how to integrate them effectively is crucial for developing capable humanoid robots. The next section will explore practical implementation of sensor systems in ROS 2.