---
sidebar_position: 3
---

# Week 8: Visual SLAM Implementation

## Overview

Visual Simultaneous Localization and Mapping (VSLAM) is a critical technology for humanoid robots to understand and navigate their environment. This section covers the implementation of VSLAM systems using NVIDIA Isaac tools, including visual-inertial SLAM, 3D reconstruction, and mapping techniques specifically tailored for humanoid robotics applications.

## Learning Objectives

By the end of this section, you will be able to:

- Understand the principles of Visual SLAM and its variants
- Implement visual-inertial SLAM for humanoid robots
- Configure and optimize VSLAM systems for real-time performance
- Integrate VSLAM with Isaac ROS and Isaac Sim
- Evaluate VSLAM performance and accuracy

## Introduction to Visual SLAM

### What is Visual SLAM?

Visual SLAM is a technique that allows robots to simultaneously localize themselves in an unknown environment while building a map of that environment using visual sensors (cameras). For humanoid robots, VSLAM is particularly important because it:

- Provides spatial awareness without requiring external infrastructure
- Enables autonomous navigation in dynamic environments
- Supports complex tasks like object recognition and manipulation
- Works in GPS-denied environments

### Types of Visual SLAM

1. **Monocular SLAM**: Uses a single camera, computationally efficient but scale ambiguous
2. **Stereo SLAM**: Uses stereo cameras, provides metric scale and depth
3. **Visual-Inertial SLAM**: Combines visual and IMU data for improved robustness
4. **RGB-D SLAM**: Uses depth cameras, provides direct depth measurements

### VSLAM Pipeline

The typical VSLAM pipeline consists of:

1. **Feature Detection**: Identify distinctive points in images
2. **Feature Tracking**: Match features across frames
3. **Pose Estimation**: Estimate camera motion between frames
4. **Mapping**: Build 3D map of environment
5. **Loop Closure**: Detect revisited locations to correct drift
6. **Optimization**: Refine map and trajectory estimates

## Visual-Inertial SLAM Fundamentals

### Sensor Fusion Principles

Visual-inertial SLAM combines visual and inertial measurements:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class VisualInertialFusion:
    def __init__(self):
        # State vector: [position, orientation, velocity, bias_gyro, bias_accel]
        self.state = np.zeros(16)  # 3 pos + 4 rot + 3 vel + 3 gyro_bias + 3 accel_bias
        self.covariance = np.eye(16) * 0.1  # Initial uncertainty

        # IMU parameters
        self.gyro_noise = 1e-3
        self.accel_noise = 1e-2
        self.gyro_bias_noise = 1e-5
        self.accel_bias_noise = 1e-4

    def predict_state(self, dt, gyro_measurement, accel_measurement):
        """
        Predict state using IMU measurements
        """
        # Extract state components
        pos = self.state[0:3]
        quat = self.state[3:7]  # [w, x, y, z]
        vel = self.state[7:10]
        gyro_bias = self.state[10:13]
        accel_bias = self.state[13:16]

        # Correct measurements with bias
        corrected_gyro = gyro_measurement - gyro_bias
        corrected_accel = accel_measurement - accel_bias

        # Convert quaternion to rotation matrix
        rot_matrix = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

        # Update orientation (angular integration)
        angular_velocity = corrected_gyro
        dq_dt = self.quaternion_derivative(quat, angular_velocity)
        new_quat = quat + dq_dt * dt
        new_quat = new_quat / np.linalg.norm(new_quat)  # Normalize

        # Update velocity (acceleration integration)
        gravity = np.array([0, 0, -9.81])
        world_accel = rot_matrix @ (corrected_accel) + gravity
        new_vel = vel + world_accel * dt

        # Update position (velocity integration)
        new_pos = pos + vel * dt + 0.5 * world_accel * dt**2

        # Update state
        self.state[0:3] = new_pos
        self.state[3:7] = new_quat
        self.state[7:10] = new_vel

        # Update covariance (simplified)
        self.propagate_covariance(dt, corrected_gyro, corrected_accel)

    def quaternion_derivative(self, q, omega):
        """
        Compute quaternion derivative from angular velocity
        """
        wx, wy, wz = omega
        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ])
        return 0.5 * Omega @ q

    def propagate_covariance(self, dt, gyro, accel):
        """
        Propagate uncertainty covariance matrix
        """
        # Jacobian of state transition
        F = self.compute_jacobian(dt, gyro, accel)

        # Process noise covariance
        Q = self.compute_process_noise(dt)

        # Propagate covariance
        self.covariance = F @ self.covariance @ F.T + Q

    def compute_jacobian(self, dt, gyro, accel):
        """
        Compute state transition Jacobian
        """
        F = np.eye(16)
        # Simplified - in practice this would be more complex
        return F

    def compute_process_noise(self, dt):
        """
        Compute process noise covariance
        """
        Q = np.zeros((16, 16))
        # Position noise
        Q[0:3, 0:3] = np.eye(3) * self.accel_noise**2 * dt**4 / 4
        # Velocity noise
        Q[7:10, 7:10] = np.eye(3) * self.accel_noise**2 * dt**2
        # Angular noise
        Q[10:13, 10:13] = np.eye(3) * self.gyro_noise**2 * dt
        return Q
```

### Key Challenges in VSLAM

1. **Feature Degradation**: Poor lighting, textureless surfaces, motion blur
2. **Scale Ambiguity**: Monocular systems cannot determine absolute scale
3. **Drift Accumulation**: Small errors accumulate over time
4. **Computational Complexity**: Real-time processing requirements
5. **Initialization**: Proper system initialization is critical

## NVIDIA Isaac Visual SLAM Implementation

### Isaac ROS Visual SLAM Setup

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import message_filters
from cv_bridge import CvBridge
import numpy as np

class IsaacVisualSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam_node')

        # Create CV bridge
        self.bridge = CvBridge()

        # Set up synchronized subscriptions for stereo + IMU
        self.left_image_sub = message_filters.Subscriber(
            self, Image, '/camera/left/image_rect')
        self.right_image_sub = message_filters.Subscriber(
            self, Image, '/camera/right/image_rect')
        self.imu_sub = message_filters.Subscriber(
            self, Imu, '/imu/data')
        self.left_info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/left/camera_info')

        # Synchronize inputs with appropriate time window
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.left_image_sub, self.right_image_sub, self.imu_sub, self.left_info_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.sync.registerCallback(self.slam_callback)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_slam/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/visual_slam/odometry', 10)
        self.map_pub = self.create_publisher(MarkerArray, '/visual_slam/map', 10)

        # SLAM state
        self.initialized = False
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.map_points = []  # 3D map points
        self.keyframes = []   # Keyframe poses

        self.get_logger().info('Isaac Visual SLAM node initialized')

    def slam_callback(self, left_img_msg, right_img_msg, imu_msg, cam_info_msg):
        """
        Process synchronized stereo and IMU data for SLAM
        """
        try:
            # Convert ROS images to OpenCV
            left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")

            # Extract IMU data
            gyro = np.array([imu_msg.angular_velocity.x,
                           imu_msg.angular_velocity.y,
                           imu_msg.angular_velocity.z])
            accel = np.array([imu_msg.linear_acceleration.x,
                            imu_msg.linear_acceleration.y,
                            imu_msg.linear_acceleration.z])

            # Perform Visual SLAM
            success, pose_update, new_features = self.process_slam(
                left_img, right_img, gyro, accel, cam_info_msg)

            if success:
                # Update pose
                self.update_pose(pose_update)

                # Publish results
                self.publish_pose(left_img_msg.header)
                self.publish_odometry(left_img_msg.header)

                if len(new_features) > 0:
                    self.publish_map()

        except Exception as e:
            self.get_logger().error(f'Error in SLAM callback: {e}')

    def process_slam(self, left_img, right_img, gyro, accel, cam_info):
        """
        Core SLAM processing using Isaac-optimized methods
        """
        # Feature detection and matching
        features = self.detect_features(left_img, right_img)

        if not self.initialized:
            # Initialize SLAM system
            success = self.initialize_slam(features, cam_info)
            if success:
                self.initialized = True
                return True, np.eye(4), features
            else:
                return False, np.eye(4), []
        else:
            # Track features and estimate motion
            pose_update, tracked_features = self.track_features(
                features, gyro, accel)
            return True, pose_update, tracked_features

    def detect_features(self, left_img, right_img):
        """
        Detect and match features between stereo images
        """
        # Use Isaac-optimized feature detection (e.g., ORB, SIFT)
        # In practice, this would use Isaac ROS optimized feature detectors
        import cv2

        # Convert to grayscale
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Detect features using ORB (for efficiency)
        orb = cv2.ORB_create(nfeatures=1000)
        kp_left, desc_left = orb.detectAndCompute(gray_left, None)
        kp_right, desc_right = orb.detectAndCompute(gray_right, None)

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc_left, desc_right)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        points_left = []
        points_right = []
        for match in matches[:100]:  # Take top 100 matches
            points_left.append(kp_left[match.queryIdx].pt)
            points_right.append(kp_right[match.trainIdx].pt)

        return {
            'points_left': np.array(points_left),
            'points_right': np.array(points_right),
            'matches': matches
        }

    def initialize_slam(self, features, cam_info):
        """
        Initialize SLAM system with initial pose and scale
        """
        if len(features['points_left']) < 10:
            return False

        # Estimate fundamental matrix and extract pose
        points_l = features['points_left']
        points_r = features['points_right']

        if len(points_l) >= 8:
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                points_l, points_r,
                focal=cam_info.k[0],  # fx
                pp=(cam_info.k[2], cam_info.k[5]),  # cx, cy
                method=cv2.RANSAC,
                threshold=1.0
            )

            if E is not None:
                # Recover pose from essential matrix
                _, R, t, mask_pose = cv2.recoverPose(E, points_l, points_r)

                # Create transformation matrix
                T = np.eye(4)
                T[0:3, 0:3] = R
                T[0:3, 3] = t.flatten()

                self.current_pose = T
                return True

        return False

    def track_features(self, features, gyro, accel):
        """
        Track features and estimate motion
        """
        # Use IMU prediction as initial estimate
        imu_prediction = self.predict_from_imu(gyro, accel)

        # Refine with visual tracking
        if len(features['points_left']) >= 5:
            # Use PnP to estimate pose from 3D-2D correspondences
            # This is simplified - real implementation would use bundle adjustment
            pass

        # Return identity for now (simplified)
        return np.eye(4), features

    def predict_from_imu(self, gyro, accel):
        """
        Predict motion from IMU data
        """
        # Integrate gyro to get rotation
        dt = 0.033  # Assume 30Hz IMU rate
        rotation_vec = gyro * dt
        angle = np.linalg.norm(rotation_vec)

        if angle > 1e-6:
            axis = rotation_vec / angle
            # Convert to rotation matrix using Rodrigues formula
            k = axis
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*K@K
        else:
            R = np.eye(3)

        # Integrate acceleration to get position change
        gravity = np.array([0, 0, 9.81])
        world_accel = R @ (accel - gravity)
        position_change = 0.5 * world_accel * dt**2

        # Create transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = position_change

        return T

    def update_pose(self, pose_update):
        """
        Update current pose with new transformation
        """
        self.current_pose = self.current_pose @ pose_update

    def publish_pose(self, header):
        """
        Publish current pose estimate
        """
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "map"

        # Extract position and orientation from transformation matrix
        position = self.current_pose[0:3, 3]
        rotation_matrix = self.current_pose[0:3, 0:3]

        # Convert rotation matrix to quaternion
        import tf_transformations
        quat = tf_transformations.quaternion_from_matrix(self.current_pose)

        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

    def publish_odometry(self, header):
        """
        Publish odometry information
        """
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = "base_link"

        # Set pose (same as above)
        position = self.current_pose[0:3, 3]
        quat = tf_transformations.quaternion_from_matrix(self.current_pose)

        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Velocity would be computed from pose differences
        # Simplified for this example
        self.odom_pub.publish(odom_msg)

    def publish_map(self):
        """
        Publish 3D map visualization
        """
        marker_array = MarkerArray()
        # Create markers for map points
        # This would be implemented based on actual map points
        self.map_pub.publish(marker_array)

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

## Feature Detection and Tracking

### Optimized Feature Detection

```python
import cv2
import numpy as np
import torch
from cuda import cudart

class OptimizedFeatureDetector:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.feature_detector = cv2.ORB_create(nfeatures=2000)

        # For GPU acceleration, we would use Isaac ROS optimized detectors
        if use_gpu:
            self.setup_gpu_detector()

    def setup_gpu_detector(self):
        """
        Setup GPU-accelerated feature detection using Isaac ROS
        """
        # This would interface with Isaac ROS CUDA-optimized feature detectors
        self.get_logger().info("Using GPU-accelerated feature detection")

    def detect_and_describe(self, image):
        """
        Detect and describe features in an image
        """
        if self.use_gpu:
            # Use Isaac ROS optimized feature detection
            return self.gpu_detect_and_describe(image)
        else:
            # Fallback to CPU
            return self.cpu_detect_and_describe(image)

    def cpu_detect_and_describe(self, image):
        """
        CPU-based feature detection and description
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def gpu_detect_and_describe(self, image):
        """
        GPU-accelerated feature detection and description
        """
        # Placeholder for Isaac ROS GPU feature detection
        # In practice, this would use CUDA kernels for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return keypoints, descriptors

class FeatureTracker:
    def __init__(self):
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.tracked_features = []

    def track_features(self, curr_keypoints, curr_descriptors, prev_keypoints, prev_descriptors):
        """
        Track features between consecutive frames
        """
        if prev_descriptors is None or curr_descriptors is None:
            return [], []

        # Use FLANN matcher for efficient matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(prev_descriptors, curr_descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        # Extract matched points
        prev_points = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_points = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        return prev_points, curr_points
```

## Loop Closure Detection

### Place Recognition for Loop Closure

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class LoopClosureDetector:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocabulary = None
        self.bow_model = MiniBatchKMeans(n_clusters=vocab_size, random_state=42)
        self.is_trained = False
        self.image_descriptors = []  # Store descriptors for training
        self.image_poses = []       # Store poses for verification

    def add_image_features(self, descriptors, pose):
        """
        Add image features for vocabulary training or loop closure detection
        """
        if descriptors is not None:
            self.image_descriptors.append(descriptors)
            self.image_poses.append(pose)

            if not self.is_trained and len(self.image_descriptors) >= 100:
                self.train_vocabulary()

    def train_vocabulary(self):
        """
        Train bag-of-words vocabulary using image descriptors
        """
        # Collect all descriptors
        all_descriptors = []
        for desc_list in self.image_descriptors:
            if desc_list is not None and len(desc_list) > 0:
                all_descriptors.append(desc_list)

        if len(all_descriptors) > 0:
            # Concatenate all descriptors
            all_desc_concat = np.vstack(all_descriptors)

            # Train vocabulary
            self.bow_model.fit(all_desc_concat)
            self.is_trained = True

            # Create vocabulary
            self.vocabulary = self.bow_model.cluster_centers_

            print(f"Vocabulary trained with {self.vocab_size} words")

    def compute_bow_descriptor(self, descriptors):
        """
        Compute bag-of-words descriptor for an image
        """
        if not self.is_trained or descriptors is None:
            return None

        # Assign descriptors to vocabulary words
        assignments = self.bow_model.predict(descriptors)

        # Create histogram
        bow_hist = np.bincount(assignments, minlength=self.vocab_size)
        bow_hist = bow_hist.astype(np.float32)

        # Normalize
        if np.sum(bow_hist) > 0:
            bow_hist = bow_hist / np.sum(bow_hist)

        return bow_hist

    def detect_loop_closure(self, current_descriptors, current_pose, min_similarity=0.7):
        """
        Detect if current view matches a previous view (loop closure)
        """
        if not self.is_trained:
            return None, 0.0

        # Compute BoW descriptor for current image
        current_bow = self.compute_bow_descriptor(current_descriptors)
        if current_bow is None:
            return None, 0.0

        # Compare with previous images
        best_match_idx = -1
        best_similarity = 0.0

        for i, desc_list in enumerate(self.image_descriptors):
            if desc_list is not None:
                prev_bow = self.compute_bow_descriptor(desc_list)
                if prev_bow is not None:
                    # Compute similarity
                    similarity = cosine_similarity([current_bow], [prev_bow])[0][0]

                    if similarity > best_similarity and similarity > min_similarity:
                        best_similarity = similarity
                        best_match_idx = i

        if best_match_idx >= 0:
            return self.image_poses[best_match_idx], best_similarity
        else:
            return None, 0.0
```

## Map Management and Optimization

### Graph-based SLAM Optimization

```python
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import networkx as nx

class MapOptimizer:
    def __init__(self):
        self.graph = nx.Graph()  # Pose graph
        self.poses = {}          # Pose estimates
        self.constraints = []    # Relative pose constraints

    def add_pose_node(self, node_id, pose=None):
        """
        Add a pose node to the graph
        """
        if pose is None:
            pose = np.eye(4)  # Identity pose if not provided

        self.graph.add_node(node_id, pose=pose)
        self.poses[node_id] = pose

    def add_constraint(self, node1_id, node2_id, relative_pose, information_matrix=None):
        """
        Add a relative pose constraint between two nodes
        """
        if information_matrix is None:
            # Default information matrix (identity)
            information_matrix = np.eye(6)

        self.graph.add_edge(node1_id, node2_id,
                           relative_pose=relative_pose,
                           information=information_matrix)

        self.constraints.append({
            'node1': node1_id,
            'node2': node2_id,
            'relative_pose': relative_pose,
            'information': information_matrix
        })

    def optimize_poses(self, max_iterations=10):
        """
        Optimize poses using graph SLAM
        """
        # This is a simplified implementation
        # Real implementation would use g2o, Ceres, or similar

        for iteration in range(max_iterations):
            # Linearize the system around current estimates
            A, b = self.linearize_system()

            # Solve the linear system
            if A.shape[0] > 0:
                delta = spsolve(csc_matrix(A), b)

                # Update poses
                self.update_poses(delta)

    def linearize_system(self):
        """
        Linearize the pose graph around current estimates
        """
        # This would create the linear system Ax = b
        # where x is the pose update vector
        # For now, return empty matrices
        return np.array([]), np.array([])

    def update_poses(self, delta):
        """
        Update poses with computed deltas
        """
        # Apply pose updates
        # This is a simplified implementation
        pass

class MapManager:
    def __init__(self):
        self.map_optimizer = MapOptimizer()
        self.map_points = []  # 3D points in the map
        self.keyframes = []   # Keyframe poses
        self.max_map_size = 10000  # Maximum number of map points

    def add_keyframe(self, frame_id, pose, features_3d):
        """
        Add a keyframe to the map
        """
        self.map_optimizer.add_pose_node(frame_id, pose)

        # Add map points from this keyframe
        for point_3d in features_3d:
            if len(self.map_points) < self.max_map_size:
                self.map_points.append({
                    'point': point_3d,
                    'observed_in': [frame_id],
                    'descriptor': None  # Feature descriptor
                })

        self.keyframes.append({
            'id': frame_id,
            'pose': pose,
            'features': features_3d
        })

    def merge_maps(self, other_map):
        """
        Merge with another map (for multi-session mapping)
        """
        # This would handle map merging and loop closure
        # between different mapping sessions
        pass

    def optimize_map(self):
        """
        Optimize the entire map using bundle adjustment
        """
        self.map_optimizer.optimize_poses()
```

## Performance Optimization

### Real-time SLAM Optimization

```python
import threading
import queue
import time
from collections import deque

class RealTimeSLAMOptimizer:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_running = False
        self.fps_counter = deque(maxlen=30)  # Last 30 frame times

        # Threading for parallel processing
        self.feature_queue = queue.Queue()
        self.tracking_queue = queue.Queue()

    def start_processing(self):
        """
        Start real-time SLAM processing
        """
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.start()

    def stop_processing(self):
        """
        Stop real-time SLAM processing
        """
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()

    def process_loop(self):
        """
        Main processing loop for real-time SLAM
        """
        while self.is_running:
            try:
                # Get input data
                data = self.input_queue.get(timeout=0.1)

                start_time = time.time()

                # Process SLAM pipeline
                self.process_frame(data)

                # Record processing time
                processing_time = time.time() - start_time
                self.fps_counter.append(1.0 / processing_time if processing_time > 0 else 0)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def process_frame(self, data):
        """
        Process a single frame through SLAM pipeline
        """
        # Feature detection (can be parallelized)
        features = self.detect_features(data['image'])

        # Tracking
        pose_update = self.track_features(features)

        # Mapping
        self.update_map(features, pose_update)

        # Optimization (periodically, not every frame)
        if self.should_optimize():
            self.optimize_map()

    def detect_features(self, image):
        """
        Detect features in image (optimized for real-time)
        """
        # Use optimized detector
        # Skip detection if processing is behind
        if self.input_queue.qsize() > 5:
            # Drop frame to maintain real-time performance
            return None

        # Perform feature detection
        # In practice, use Isaac ROS optimized detectors
        return []

    def should_optimize(self):
        """
        Determine if map optimization should be performed
        """
        # Optimize every N frames or when certain conditions are met
        return len(self.fps_counter) > 0 and np.mean(self.fps_counter) > 10  # Only if we have processing headroom
```

## Evaluation and Validation

### SLAM Accuracy Assessment

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class SLAMEvaluator:
    def __init__(self):
        self.ground_truth_poses = []
        self.estimated_poses = []
        self.alignment_transform = None

    def add_ground_truth_pose(self, pose):
        """
        Add ground truth pose for evaluation
        """
        self.ground_truth_poses.append(pose)

    def add_estimated_pose(self, pose):
        """
        Add estimated pose from SLAM system
        """
        self.estimated_poses.append(pose)

    def compute_trajectory_error(self):
        """
        Compute trajectory errors (ATE and RPE)
        """
        if len(self.ground_truth_poses) != len(self.estimated_poses):
            print("Warning: Ground truth and estimated trajectories have different lengths")
            return None, None

        # Align estimated trajectory to ground truth
        self.alignment_transform = self.align_trajectories()

        # Apply alignment
        aligned_estimated = []
        for est_pose in self.estimated_poses:
            aligned_pose = self.alignment_transform @ est_pose
            aligned_estimated.append(aligned_pose)

        # Compute Absolute Trajectory Error (ATE)
        ate = self.compute_ate(self.ground_truth_poses, aligned_estimated)

        # Compute Relative Pose Error (RPE)
        rpe = self.compute_rpe(self.ground_truth_poses, aligned_estimated)

        return ate, rpe

    def align_trajectories(self):
        """
        Align estimated trajectory to ground truth using Umeyama algorithm
        """
        if len(self.ground_truth_poses) < 3:
            return np.eye(4)

        # Extract positions
        gt_positions = np.array([pose[0:3, 3] for pose in self.ground_truth_poses])
        est_positions = np.array([pose[0:3, 3] for pose in self.estimated_poses])

        # Compute alignment using Umeyama algorithm
        # Simplified implementation
        gt_mean = np.mean(gt_positions, axis=0)
        est_mean = np.mean(est_positions, axis=0)

        # Center the point clouds
        gt_centered = gt_positions - gt_mean
        est_centered = est_positions - est_mean

        # Compute covariance matrix
        H = est_centered.T @ gt_centered
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation
        R = Vt.T @ U.T
        # Ensure proper rotation matrix (not reflection)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = gt_mean - R @ est_mean

        # Create transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t

        return T

    def compute_ate(self, gt_poses, est_poses):
        """
        Compute Absolute Trajectory Error
        """
        errors = []
        for gt, est in zip(gt_poses, est_poses):
            pos_error = np.linalg.norm(gt[0:3, 3] - est[0:3, 3])
            errors.append(pos_error)

        return {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'rmse': np.sqrt(np.mean(np.square(errors))),
            'median': np.median(errors)
        }

    def compute_rpe(self, gt_poses, est_poses):
        """
        Compute Relative Pose Error
        """
        errors = []
        for i in range(len(gt_poses) - 1):
            # Compute relative transformation in ground truth
            gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i+1]
            est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i+1]

            # Compute error
            error = np.linalg.inv(est_rel) @ gt_rel
            pos_error = np.linalg.norm(error[0:3, 3])

            errors.append(pos_error)

        return {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'rmse': np.sqrt(np.mean(np.square(errors))),
            'median': np.median(errors)


# Example usage of evaluation
def evaluate_slam_performance():
    evaluator = SLAMEvaluator()

    # Add some example poses (in practice, these would come from ground truth and SLAM output)
    for i in range(100):
        # Ground truth: simple circular trajectory
        angle = i * 0.1
        gt_pose = np.eye(4)
        gt_pose[0, 3] = np.cos(angle) * 5  # Circle of radius 5
        gt_pose[1, 3] = np.sin(angle) * 5
        gt_pose[2, 3] = 0

        # Estimated: with some error
        est_pose = gt_pose.copy()
        est_pose[0, 3] += np.random.normal(0, 0.1)  # Add position noise
        est_pose[1, 3] += np.random.normal(0, 0.1)

        evaluator.add_ground_truth_pose(gt_pose)
        evaluator.add_estimated_pose(est_pose)

    ate, rpe = evaluator.compute_trajectory_error()

    print(f"ATE: {ate}")
    print(f"RPE: {rpe}")
```

## Best Practices and Troubleshooting

### Common VSLAM Issues and Solutions

**Issue: Drift Accumulation**
- **Cause**: Integration of small errors over time
- **Solution**: Implement loop closure detection and pose graph optimization

**Issue: Feature Degradation**
- **Cause**: Poor lighting, textureless surfaces
- **Solution**: Use multiple feature types, add IMU fusion, improve lighting

**Issue: Scale Drift (Monocular)**
- **Cause**: Cannot determine absolute scale from monocular images
- **Solution**: Use stereo/RGB-D cameras or add scale constraints

**Issue: Computational Bottleneck**
- **Cause**: Heavy processing requirements
- **Solution**: Optimize algorithms, use GPU acceleration, selective processing

### Performance Tips

1. **Use Appropriate Feature Detectors**: ORB for speed, SIFT for accuracy
2. **Implement Keyframing**: Process every Nth frame to reduce computational load
3. **Multi-threading**: Separate tracking, mapping, and optimization threads
4. **GPU Acceleration**: Use Isaac ROS optimized CUDA kernels
5. **Adaptive Processing**: Adjust processing based on scene complexity

## Summary

Visual SLAM is a fundamental capability for autonomous humanoid robots, enabling them to understand and navigate their environment without external infrastructure. The implementation of VSLAM systems requires careful consideration of sensor fusion, computational efficiency, and accuracy optimization. NVIDIA Isaac provides optimized tools and packages that significantly enhance the performance and robustness of VSLAM systems, making them practical for real-world humanoid robotics applications.

The combination of visual and inertial sensors, along with optimized algorithms and hardware acceleration, enables humanoid robots to build accurate maps of their environment and maintain precise localization over extended periods of operation.