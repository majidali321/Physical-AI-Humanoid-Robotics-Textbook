---
sidebar_position: 2
---

# Week 8: NVIDIA Isaac ROS Integration

## Overview

NVIDIA Isaac ROS is a collection of hardware-accelerated perception and manipulation packages that bridge the gap between AI and robotics. This section covers the integration of Isaac ROS with Isaac Sim and ROS 2, focusing on perception pipelines, sensor processing, and AI-based robotics applications for humanoid systems.

## Learning Objectives

By the end of this section, you will be able to:

- Install and configure Isaac ROS packages
- Integrate Isaac ROS with Isaac Sim and ROS 2
- Implement perception pipelines using Isaac ROS
- Utilize hardware acceleration for AI-based robotics
- Create end-to-end AI robotics workflows

## Introduction to Isaac ROS

### Overview and Architecture

NVIDIA Isaac ROS is a collection of ROS 2 packages that leverage NVIDIA hardware acceleration to enable real-time AI-based robotics applications. The architecture includes:

- **Hardware Acceleration**: Utilizes Jetson, RTX, and Aerial platforms
- **CUDA Acceleration**: GPU-accelerated processing for perception and planning
- **ROS 2 Integration**: Native ROS 2 packages with standard interfaces
- **Modular Design**: Independent packages that can be combined as needed

### Key Isaac ROS Packages

1. **Isaac ROS Visual SLAM**: Real-time visual-inertial SLAM
2. **Isaac ROS Stereo Dense Reconstruction**: 3D scene reconstruction
3. **Isaac ROS Apriltag**: Marker-based pose estimation
4. **Isaac ROS NITROS**: Network Interface for Time-based, Resilient, Ordered, and Synchronous communication
5. **Isaac ROS DNN Inference**: GPU-accelerated neural network inference
6. **Isaac ROS Image Pipeline**: Optimized image processing pipeline
7. **Isaac ROS Manipulator**: AI-based manipulation planning

### Hardware Requirements

**Minimum Hardware**:
- NVIDIA GPU with Tensor Cores (RTX 2060 or better)
- CUDA Compute Capability 7.0+
- 8GB+ VRAM recommended
- Jetson AGX Xavier or better for edge deployment

**Recommended Hardware**:
- RTX 3080/4080 or A6000 for development
- 16GB+ VRAM for complex AI models
- Jetson Orin AGX for edge deployment

## Installing Isaac ROS

### Prerequisites

Before installing Isaac ROS, ensure your system has the required prerequisites:

```bash
# Check NVIDIA GPU and driver
nvidia-smi

# Verify CUDA installation
nvcc --version

# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential
sudo apt install -y libssl-dev libffi-dev libeigen3-dev
```

### Installation Methods

#### Method 1: Docker Installation (Recommended)

```bash
# Pull Isaac ROS Docker image
docker pull nvcr.io/nvidia/isaac-ros:latest

# Run Isaac ROS container
docker run --gpus all -it --rm \
  --network=host \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
  --volume="${PWD}:/workspaces" \
  --name isaac_ros_dev \
  nvcr.io/nvidia/isaac-ros:latest
```

#### Method 2: Source Installation

```bash
# Create ROS workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Clone Isaac ROS repositories
cd src
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git

# Install dependencies
cd ~/isaac_ros_ws
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build --symlink-install
source install/setup.bash
```

### Verification Installation

```bash
# Check Isaac ROS packages
ros2 pkg list | grep isaac_ros

# Run a simple test
ros2 run isaac_ros_test test_image_pipeline
```

## Isaac ROS Perception Pipeline

### Image Pipeline

The Isaac ROS Image Pipeline provides optimized image processing capabilities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacImageProcessor(Node):
    def __init__(self):
        super().__init__('isaac_image_processor')

        # Create subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Create publisher for processed image
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_processed',
            10
        )

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        self.get_logger().info('Isaac Image Processor initialized')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Apply Isaac ROS optimized processing
            processed_image = self.process_image(cv_image)

            # Convert back to ROS Image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, "bgr8")
            processed_msg.header = msg.header

            # Publish processed image
            self.publisher.publish(processed_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_image(self, image):
        # Apply various image processing techniques
        # This is where Isaac ROS optimizations would be applied

        # Example: Edge detection using GPU acceleration
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Combine with original image
        result = image.copy()
        result[edges > 0] = [0, 255, 0]  # Mark edges in green

        return result

def main(args=None):
    rclpy.init(args=args)
    processor = IsaacImageProcessor()

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

### Stereo Processing

Isaac ROS provides optimized stereo processing capabilities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from cv_bridge import CvBridge
import numpy as np

class IsaacStereoProcessor(Node):
    def __init__(self):
        super().__init__('isaac_stereo_processor')

        # Subscriptions for stereo pair
        self.left_subscription = self.create_subscription(
            Image,
            '/stereo/left/image_raw',
            self.left_image_callback,
            10
        )

        self.right_subscription = self.create_subscription(
            Image,
            '/stereo/right/image_raw',
            self.right_image_callback,
            10
        )

        # Publisher for disparity map
        self.disparity_publisher = self.create_publisher(
            DisparityImage,
            '/stereo/disparity',
            10
        )

        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None

        self.get_logger().info('Isaac Stereo Processor initialized')

    def left_image_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_stereo()

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_stereo()

    def process_stereo(self):
        if self.left_image is not None and self.right_image is not None:
            # Apply stereo processing using Isaac ROS optimized methods
            # This would typically use CUDA-accelerated stereo matching
            disparity = self.compute_disparity(self.left_image, self.right_image)

            # Create disparity message
            disp_msg = DisparityImage()
            disp_msg.image = self.bridge.cv2_to_imgmsg(disparity, "32FC1")
            disp_msg.header = self.left_image.header
            disp_msg.f = 1.0  # Focal length (to be calibrated)
            disp_msg.T = 0.1  # Baseline (to be calibrated)

            self.disparity_publisher.publish(disp_msg)

    def compute_disparity(self, left_img, right_img):
        # Use optimized stereo matching algorithm
        # Isaac ROS provides CUDA-accelerated stereo matching
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Use SGBM for better results
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        return disparity

def main(args=None):
    rclpy.init(args=args)
    processor = IsaacStereoProcessor()

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

## Isaac ROS Visual SLAM

### Setting up Visual SLAM

Isaac ROS Visual SLAM provides real-time localization and mapping:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import message_filters

class IsaacVisualSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam')

        # Subscriptions with synchronization
        self.image_sub = message_filters.Subscriber(
            self, Image, '/camera/image_raw'
        )
        self.imu_sub = message_filters.Subscriber(
            self, Imu, '/imu/data'
        )

        # Synchronize image and IMU data
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub], 10, 0.1
        )
        self.sync.registerCallback(self.slam_callback)

        # Publishers
        self.pose_publisher = self.create_publisher(
            PoseStamped, '/visual_slam/pose', 10
        )
        self.odom_publisher = self.create_publisher(
            Odometry, '/visual_slam/odometry', 10
        )

        self.get_logger().info('Isaac Visual SLAM node initialized')

    def slam_callback(self, image_msg, imu_msg):
        # Process synchronized image and IMU data for SLAM
        # This would interface with Isaac ROS Visual SLAM pipeline
        self.get_logger().info('Processing SLAM data')

        # Publish estimated pose
        pose_msg = PoseStamped()
        pose_msg.header = image_msg.header
        # Fill pose with estimated position and orientation
        self.pose_publisher.publish(pose_msg)

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header = image_msg.header
        # Fill odometry with estimated position, velocity, etc.
        self.odom_publisher.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    slam_node = IsaacVisualSLAMNode()

    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch Configuration for Visual SLAM

Create a launch file to configure the Visual SLAM system:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Declare launch arguments
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace for the nodes'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),

        # Isaac ROS Visual SLAM node
        Node(
            package='isaac_ros_visual_slam',
            executable='visual_slam_node',
            name='visual_slam',
            namespace=namespace,
            parameters=[
                {'use_sim_time': use_sim_time},
                {'enable_slam': True},
                {'enable_rectification': True},
                {'rectified_images': True},
                {'map_frame': 'map'},
                {'odom_frame': 'odom'},
                {'base_frame': 'base_link'},
                {'publish_odom_tf': True}
            ],
            remappings=[
                ('/visual_slam/image', '/camera/image_rect'),
                ('/visual_slam/camera_info', '/camera/camera_info'),
                ('/visual_slam/imu', '/imu/data')
            ]
        ),

        # Image rectification (if needed)
        Node(
            package='isaac_ros_image_proc',
            executable='rectify_node',
            name='rectify',
            namespace=namespace,
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('image', '/camera/image_raw'),
                ('camera_info', '/camera/camera_info'),
                ('image_rect', '/camera/image_rect')
            ]
        )
    ])
```

## Isaac ROS DNN Inference

### Neural Network Inference Setup

Isaac ROS provides optimized DNN inference capabilities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np

class IsaacDNNInferenceNode(Node):
    def __init__(self):
        super().__init__('isaac_dnn_inference')

        # Subscription to camera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for detections
        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )

        self.bridge = CvBridge()

        # Initialize DNN model (placeholder - would use Isaac ROS DNN)
        self.initialize_model()

        self.get_logger().info('Isaac DNN Inference node initialized')

    def initialize_model(self):
        # Initialize neural network model
        # In real implementation, this would load a TensorRT optimized model
        self.get_logger().info('Initializing DNN model...')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform inference using Isaac ROS optimized pipeline
            detections = self.perform_inference(cv_image)

            # Publish detections
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_publisher.publish(detection_msg)

        except Exception as e:
            self.get_logger().error(f'Error in DNN inference: {e}')

    def perform_inference(self, image):
        # Perform DNN inference using Isaac ROS optimized methods
        # This would typically use TensorRT for acceleration

        # Placeholder implementation
        # In real implementation, this would use Isaac ROS DNN inference
        detections = []

        # Example: Simple object detection (replace with actual DNN)
        height, width = image.shape[:2]
        # Simulate detection of a person
        detection = {
            'class': 'person',
            'confidence': 0.95,
            'bbox': [width//4, height//4, width//2, height//2]
        }
        detections.append(detection)

        return detections

    def create_detection_message(self, detections, header):
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            detection_msg = Detection2D()
            detection_msg.header = header

            # Set bounding box
            bbox = detection['bbox']
            detection_msg.bbox.center.x = bbox[0] + bbox[2] / 2
            detection_msg.bbox.center.y = bbox[1] + bbox[3] / 2
            detection_msg.bbox.size_x = bbox[2]
            detection_msg.bbox.size_y = bbox[3]

            # Set results
            result = detection_msg.results.add()
            result.id = detection['class']
            result.score = detection['confidence']

            detection_array.detections.append(detection_msg)

        return detection_array

def main(args=None):
    rclpy.init(args=args)
    dnn_node = IsaacDNNInferenceNode()

    try:
        rclpy.spin(dnn_node)
    except KeyboardInterrupt:
        pass
    finally:
        dnn_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Manipulation

### Manipulation Planning

Isaac ROS provides AI-based manipulation capabilities:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from moveit_msgs.msg import MoveItErrorCodes
from sensor_msgs.msg import JointState
from visualization_msgs.msg import MarkerArray

class IsaacManipulationNode(Node):
    def __init__(self):
        super().__init__('isaac_manipulation')

        # Subscriptions
        self.target_sub = self.create_subscription(
            PointStamped,
            '/manipulation/target',
            self.target_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publishers
        self.command_pub = self.create_publisher(
            JointState,
            '/joint_commands',
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/manipulation/markers',
            10
        )

        # Internal state
        self.current_joints = JointState()
        self.target_pose = None

        self.get_logger().info('Isaac Manipulation node initialized')

    def target_callback(self, msg):
        self.target_pose = msg.point
        self.plan_manipulation()

    def joint_state_callback(self, msg):
        self.current_joints = msg

    def plan_manipulation(self):
        if self.target_pose is not None:
            # Plan manipulation trajectory using Isaac ROS manipulation
            # This would interface with Isaac ROS manipulation planning
            trajectory = self.compute_manipulation_trajectory()

            if trajectory is not None:
                # Execute trajectory
                self.execute_trajectory(trajectory)

    def compute_manipulation_trajectory(self):
        # Use Isaac ROS manipulation planning
        # This would typically involve:
        # 1. Inverse kinematics
        # 2. Trajectory optimization
        # 3. Collision checking
        # 4. AI-based grasp planning

        # Placeholder implementation
        trajectory = []
        self.get_logger().info('Computing manipulation trajectory...')

        # Return trajectory (in real implementation, this would be a planned path)
        return trajectory

    def execute_trajectory(self, trajectory):
        # Execute the planned trajectory
        for waypoint in trajectory:
            # Publish joint commands
            joint_cmd = JointState()
            joint_cmd.name = waypoint['joint_names']
            joint_cmd.position = waypoint['joint_positions']

            self.command_pub.publish(joint_cmd)
            # Wait for execution (simplified)
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.1))

def main(args=None):
    rclpy.init(args=args)
    manip_node = IsaacManipulationNode()

    try:
        rclpy.spin(manip_node)
    except KeyboardInterrupt:
        pass
    finally:
        manip_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS NITROS (Network Interface)

### Optimized Data Transport

Isaac ROS NITROS provides optimized transport for robotics data:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from isaac_ros_nitros_bridge_interfaces.msg import NitrosBridgeImage
import time

class NitrosOptimizedNode(Node):
    def __init__(self):
        super().__init__('nitros_optimized_node')

        # Use NITROS for optimized transport
        self.nitros_publisher = self.create_publisher(
            NitrosBridgeImage,
            '/nitros_optimized_topic',
            10
        )

        self.image_subscription = self.create_subscription(
            Image,
            '/input_image',
            self.image_callback,
            10
        )

        # Timer for performance testing
        self.timer = self.create_timer(0.033, self.performance_callback)  # ~30Hz
        self.message_count = 0
        self.start_time = time.time()

        self.get_logger().info('NITROS Optimized node initialized')

    def image_callback(self, msg):
        # Convert to NITROS format for optimized transport
        nitros_msg = NitrosBridgeImage()
        nitros_msg.header = msg.header
        nitros_msg.width = msg.width
        nitros_msg.height = msg.height
        nitros_msg.encoding = msg.encoding
        nitros_msg.data = msg.data
        nitros_msg.step = msg.step

        # Publish via NITROS
        self.nitros_publisher.publish(nitros_msg)

        self.message_count += 1

    def performance_callback(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.message_count / elapsed_time
            self.get_logger().info(f'NITROS Performance: {fps:.2f} FPS')

def main(args=None):
    rclpy.init(args=args)
    nitros_node = NitrosOptimizedNode()

    try:
        rclpy.spin(nitros_node)
    except KeyboardInterrupt:
        pass
    finally:
        nitros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with Isaac Sim

### Isaac Sim to Isaac ROS Bridge

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np

class IsaacSimBridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_bridge')

        # Publishers for Isaac Sim data
        self.camera_pub = self.create_publisher(Image, '/sim/camera/image_raw', 10)
        self.imu_pub = self.create_publisher(Imu, '/sim/imu/data', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/sim/lidar/points', 10)

        # Subscriptions for commands to Isaac Sim
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        # Timer to simulate Isaac Sim data publishing
        self.timer = self.create_timer(0.033, self.publish_sim_data)  # 30 Hz

        self.get_logger().info('Isaac Sim Bridge initialized')

    def publish_sim_data(self):
        # Simulate publishing data from Isaac Sim
        # In real implementation, this would interface with Isaac Sim's Python API

        # Publish camera data
        camera_msg = Image()
        camera_msg.header.stamp = self.get_clock().now().to_msg()
        camera_msg.header.frame_id = 'sim_camera'
        camera_msg.height = 480
        camera_msg.width = 640
        camera_msg.encoding = 'rgb8'
        camera_msg.is_bigendian = 0
        camera_msg.step = 640 * 3  # Width * channels
        camera_msg.data = [0] * (640 * 480 * 3)  # Simulated image data

        self.camera_pub.publish(camera_msg)

        # Publish IMU data
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'sim_imu'
        # Fill with simulated IMU data
        self.imu_pub.publish(imu_msg)

    def cmd_callback(self, msg):
        # Process commands for Isaac Sim
        # In real implementation, this would send commands to Isaac Sim
        self.get_logger().info(f'Received command: linear={msg.linear.x}, angular={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    bridge = IsaacSimBridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### GPU Utilization Monitoring

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import subprocess
import json

class GPUMonitor(Node):
    def __init__(self):
        super().__init__('gpu_monitor')

        self.gpu_util_pub = self.create_publisher(Float32, '/gpu/utilization', 10)
        self.gpu_memory_pub = self.create_publisher(Float32, '/gpu/memory_usage', 10)

        self.timer = self.create_timer(1.0, self.monitor_gpu)

        self.get_logger().info('GPU Monitor initialized')

    def monitor_gpu(self):
        try:
            # Get GPU status using nvidia-smi
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                gpu_util = float(gpu_data[0])
                gpu_memory_used = float(gpu_data[1])
                gpu_memory_total = float(gpu_data[2])
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100

                # Publish GPU utilization
                util_msg = Float32()
                util_msg.data = gpu_util
                self.gpu_util_pub.publish(util_msg)

                # Publish GPU memory usage
                mem_msg = Float32()
                mem_msg.data = gpu_memory_percent
                self.gpu_memory_pub.publish(mem_msg)

                self.get_logger().info(f'GPU Util: {gpu_util}%, Memory: {gpu_memory_percent}%')

        except Exception as e:
            self.get_logger().error(f'Error monitoring GPU: {e}')

def main(args=None):
    rclpy.init(args=args)
    monitor = GPUMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices and Troubleshooting

### Common Issues and Solutions

**Issue: High GPU Memory Usage**
- Use TensorRT optimization for neural networks
- Implement memory pooling for large data structures
- Monitor and limit concurrent inference requests

**Issue: Low Inference Performance**
- Verify TensorRT installation and optimization
- Check CUDA compute capability requirements
- Profile and optimize bottlenecks in the pipeline

**Issue: Integration Problems**
- Ensure all Isaac ROS packages are compatible versions
- Check network and time synchronization
- Verify message format compatibility

### Performance Tips

1. **Use TensorRT**: Convert models to TensorRT format for acceleration
2. **Batch Processing**: Process multiple inputs together when possible
3. **Memory Management**: Reuse buffers and avoid unnecessary allocations
4. **Pipeline Optimization**: Use NITROS for optimized data transport
5. **Multi-Processing**: Use separate processes for different pipeline stages

## Summary

Isaac ROS provides a comprehensive set of hardware-accelerated packages that significantly enhance robotics applications with AI capabilities. By leveraging NVIDIA's hardware acceleration, Isaac ROS enables real-time processing of complex perception and manipulation tasks that would be impossible with CPU-only systems. The integration with Isaac Sim and ROS 2 creates a complete development pipeline from simulation to deployment for humanoid robotics applications.

In the next sections, we'll explore VSLAM implementation and advanced navigation capabilities using the Isaac ecosystem.