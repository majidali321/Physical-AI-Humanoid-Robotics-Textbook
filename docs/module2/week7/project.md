---
sidebar_position: 4
---

# Week 7: Module 2 Project - Complete Digital Twin Implementation

## Overview

In Week 7, you'll complete Module 2 by implementing a comprehensive digital twin system that integrates both Gazebo and Unity simulation environments with ROS 2. This project demonstrates the complete pipeline from physical robot representation to digital simulation and visualization.

## Project Objectives

By the end of this project, you will have:

- Created a complete humanoid robot model compatible with both Gazebo and Unity
- Implemented ROS 2 interfaces for both simulation environments
- Developed a digital twin system with synchronized visualization
- Validated the digital twin through simulation experiments
- Demonstrated the advantages of multi-environment simulation

## Project Requirements

### Core Components

1. **Unified Robot Model**: URDF that works in both Gazebo and Unity
2. **Gazebo Simulation**: Physics-based simulation with sensors
3. **Unity Visualization**: Realistic 3D rendering and environment
4. **ROS 2 Bridge**: Communication between both environments
5. **Synchronization System**: Keep both environments in sync

### Technical Specifications

- Robot model must include realistic kinematics and dynamics
- Both simulation environments must respond to ROS 2 commands
- Sensor data must be consistent between environments
- Visualization must be smooth and responsive
- System must handle failure gracefully

## Project Structure

### 1. Unified Robot Model

Create a URDF that works well in both environments:

```xml
<?xml version="1.0"?>
<robot name="digital_twin_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1.0"/>
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
      <material name="white">
        <color rgba="1.0 1.0 1.0 1.0"/>
      </material>
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

  <joint name="head_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1.0 0.0 0.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder"/>
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Right Arm -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_shoulder"/>
    <origin xyz="-0.15 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Gazebo-specific configurations -->
  <gazebo reference="base_link">
    <material>Gazebo/Gray</material>
  </gazebo>

  <gazebo reference="head">
    <material>Gazebo/White</material>
  </gazebo>

  <gazebo reference="left_shoulder">
    <material>Gazebo/Red</material>
  </gazebo>

  <gazebo reference="right_shoulder">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Gazebo plugins for control -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <joint_name>left_shoulder_joint, right_shoulder_joint</joint_name>
    </plugin>
  </gazebo>

</robot>
```

### 2. Gazebo World Configuration

Create a world file that includes your robot and environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="digital_twin_world">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Sun lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics engine -->
    <physics name="ode" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>20</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Room environment -->
    <model name="room">
      <pose>0 0 2.5 0 0 0</pose>
      <static>true</static>
      <link name="room_link">
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture -->
    <model name="table">
      <pose>2 0 0.4 0 0 0</pose>
      <include>
        <uri>model://table</uri>
      </include>
    </model>

    <!-- Your robot will be spawned here -->
  </world>
</sdf>
```

### 3. Unity Digital Twin Controller

Create the main controller script for Unity:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using System.Collections.Generic;

public class DigitalTwinController : MonoBehaviour
{
    [Header("ROS Configuration")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Configuration")]
    public GameObject robotPrefab;
    public Transform spawnPoint;
    public float syncFrequency = 50.0f; // Hz

    [Header("Visualization")]
    public Material[] robotMaterials;
    public bool showTrajectory = true;
    public int trajectoryPoints = 100;

    private ROSConnection ros;
    private GameObject robotInstance;
    private List<Vector3> trajectoryPointsList = new List<Vector3>();
    private LineRenderer trajectoryLine;
    private float syncInterval;
    private float lastSyncTime;

    // Robot joint controllers
    private Dictionary<string, Transform> jointMap = new Dictionary<string, Transform>();

    void Start()
    {
        ros = ROSConnection.instance;
        ros.Initialize(rosIPAddress, rosPort);

        syncInterval = 1.0f / syncFrequency;
        lastSyncTime = Time.time;

        SpawnRobot();
        SetupTrajectoryVisualization();
        SubscribeToTopics();
    }

    void SpawnRobot()
    {
        if (robotPrefab != null && spawnPoint != null)
        {
            robotInstance = Instantiate(robotPrefab, spawnPoint.position, spawnPoint.rotation);
            robotInstance.name = "DigitalTwinRobot";

            // Map joint names to transforms
            MapRobotJoints();
        }
    }

    void MapRobotJoints()
    {
        // Find all joints by name convention
        Transform[] allTransforms = robotInstance.GetComponentsInChildren<Transform>();

        foreach (Transform t in allTransforms)
        {
            if (t.name.Contains("joint") || t.name.Contains("shoulder") ||
                t.name.Contains("elbow") || t.name.Contains("hip"))
            {
                jointMap[t.name.ToLower()] = t;
            }
        }
    }

    void SetupTrajectoryVisualization()
    {
        if (showTrajectory)
        {
            GameObject trajectoryObj = new GameObject("Trajectory");
            trajectoryObj.transform.SetParent(robotInstance.transform);
            trajectoryLine = trajectoryObj.AddComponent<LineRenderer>();
            trajectoryLine.material = new Material(Shader.Find("Sprites/Default"));
            trajectoryLine.startWidth = 0.02f;
            trajectoryLine.endWidth = 0.02f;
            trajectoryLine.startColor = Color.green;
            trajectoryLine.endColor = Color.red;
            trajectoryLine.positionCount = 0;
        }
    }

    void SubscribeToTopics()
    {
        // Subscribe to joint states from Gazebo simulation
        ros.Subscribe<JointStateMsg>("/joint_states", JointStateCallback);

        // Subscribe to robot pose from Gazebo
        ros.Subscribe<OdometryMsg>("/odom", OdometryCallback);

        // Subscribe to sensor data
        ros.Subscribe<ImuMsg>("/imu/data", ImuCallback);
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        if (jointState.name.Count != jointState.position.Count)
            return;

        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i].ToLower();
            float position = (float)jointState.position[i];

            if (jointMap.ContainsKey(jointName))
            {
                Transform jointTransform = jointMap[jointName];

                // Apply rotation based on joint position
                // This is a simplified example - you may need more complex IK
                jointTransform.localRotation = Quaternion.Euler(0, position * Mathf.Rad2Deg, 0);
            }
        }
    }

    void OdometryCallback(OdometryMsg odom)
    {
        if (robotInstance != null)
        {
            // Convert ROS pose to Unity position
            Vector3 position = new Vector3(
                (float)odom.pose.pose.position.x,
                (float)odom.pose.pose.position.z, // Swap Y and Z for Unity
                (float)odom.pose.pose.position.y
            );

            // Apply position with smoothing
            robotInstance.transform.position = Vector3.Lerp(
                robotInstance.transform.position,
                position,
                Time.deltaTime * 5f // Smoothing factor
            );

            // Update trajectory
            if (showTrajectory)
            {
                UpdateTrajectory(position);
            }
        }
    }

    void ImuCallback(ImuMsg imu)
    {
        // Process IMU data for visualization or additional effects
        // For example, show orientation changes
        if (robotInstance != null)
        {
            Quaternion orientation = new Quaternion(
                (float)imu.orientation.x,
                (float)imu.orientation.z, // Swap for Unity coordinate system
                (float)imu.orientation.y,
                (float)imu.orientation.w
            );

            robotInstance.transform.rotation = orientation;
        }
    }

    void UpdateTrajectory(Vector3 position)
    {
        trajectoryPointsList.Add(position);

        if (trajectoryPointsList.Count > trajectoryPoints)
        {
            trajectoryPointsList.RemoveAt(0);
        }

        if (trajectoryLine != null)
        {
            trajectoryLine.positionCount = trajectoryPointsList.Count;
            trajectoryLine.SetPositions(trajectoryPointsList.ToArray());
        }
    }

    void Update()
    {
        // Send Unity robot state back to ROS if needed
        if (Time.time - lastSyncTime >= syncInterval)
        {
            PublishUnityRobotState();
            lastSyncTime = Time.time;
        }
    }

    void PublishUnityRobotState()
    {
        // Publish Unity robot state back to ROS
        // This creates a bidirectional digital twin
        if (robotInstance != null)
        {
            var poseMsg = new geometry_msgs.PoseMsg
            {
                position = new geometry_msgs.Vector3Msg
                {
                    x = robotInstance.transform.position.x,
                    y = robotInstance.transform.position.z, // Unity to ROS coordinate conversion
                    z = robotInstance.transform.position.y
                },
                orientation = new geometry_msgs.QuaternionMsg
                {
                    x = robotInstance.transform.rotation.x,
                    y = robotInstance.transform.rotation.z,
                    z = robotInstance.transform.rotation.y,
                    w = robotInstance.transform.rotation.w
                }
            };

            ros.Publish("/unity_robot_pose", poseMsg);
        }
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros = null;
        }
    }
}
```

### 4. Synchronization System

Create a synchronization manager to keep both environments in sync:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

public class SynchronizationManager : MonoBehaviour
{
    [Header("Synchronization Settings")]
    public float syncFrequency = 30.0f; // Hz
    public float maxSyncError = 0.1f; // Maximum allowed position error
    public bool autoCorrect = true; // Auto-correct synchronization errors

    [Header("ROS Topics")]
    public string gazeboPoseTopic = "/gazebo_robot_pose";
    public string unityPoseTopic = "/unity_robot_pose";
    public string syncCommandTopic = "/sync_command";

    private ROSConnection ros;
    private float syncInterval;
    private float lastSyncTime;
    private Vector3 lastGazeboPosition;
    private Vector3 lastUnityPosition;
    private Quaternion lastGazeboOrientation;
    private Quaternion lastUnityOrientation;

    void Start()
    {
        ros = ROSConnection.instance;
        syncInterval = 1.0f / syncFrequency;
        lastSyncTime = Time.time;

        SubscribeToSynchronizationTopics();
    }

    void SubscribeToSynchronizationTopics()
    {
        ros.Subscribe<PoseMsg>(gazeboPoseTopic, GazeboPoseCallback);
        ros.Subscribe<PoseMsg>(unityPoseTopic, UnityPoseCallback);
    }

    void GazeboPoseCallback(PoseMsg pose)
    {
        lastGazeboPosition = new Vector3(
            (float)pose.position.x,
            (float)pose.position.z,
            (float)pose.position.y
        );

        lastGazeboOrientation = new Quaternion(
            (float)pose.orientation.x,
            (float)pose.orientation.z,
            (float)pose.orientation.y,
            (float)pose.orientation.w
        );
    }

    void UnityPoseCallback(PoseMsg pose)
    {
        lastUnityPosition = new Vector3(
            (float)pose.position.x,
            (float)pose.position.z,
            (float)pose.position.y
        );

        lastUnityOrientation = new Quaternion(
            (float)pose.orientation.x,
            (float)pose.orientation.z,
            (float)pose.orientation.y,
            (float)pose.orientation.w
        );
    }

    void Update()
    {
        if (Time.time - lastSyncTime >= syncInterval)
        {
            CheckSynchronization();
            lastSyncTime = Time.time;
        }
    }

    void CheckSynchronization()
    {
        if (lastGazeboPosition != Vector3.zero && lastUnityPosition != Vector3.zero)
        {
            float positionError = Vector3.Distance(lastGazeboPosition, lastUnityPosition);

            if (positionError > maxSyncError)
            {
                Debug.LogWarning($"Synchronization error detected: {positionError:F3}m");

                if (autoCorrect)
                {
                    CorrectSynchronization(positionError);
                }
            }

            // Check orientation error
            float angleError = Quaternion.Angle(lastGazeboOrientation, lastUnityOrientation);
            if (angleError > 10f) // 10 degree threshold
            {
                Debug.LogWarning($"Orientation synchronization error: {angleError:F2} degrees");
            }
        }
    }

    void CorrectSynchronization(float error)
    {
        // Send correction command to Unity to match Gazebo position
        var correctionMsg = new PoseMsg
        {
            position = new geometry_msgs.Vector3Msg
            {
                x = lastGazeboPosition.x,
                y = lastGazeboPosition.z, // Unity to ROS coordinate conversion
                z = lastGazeboPosition.y
            },
            orientation = new geometry_msgs.QuaternionMsg
            {
                x = lastGazeboOrientation.x,
                y = lastGazeboOrientation.z,
                z = lastGazeboOrientation.y,
                w = lastGazeboOrientation.w
            }
        };

        ros.Publish(syncCommandTopic, correctionMsg);
    }

    public void ForceSynchronization()
    {
        // Force Unity to match Gazebo state
        if (lastGazeboPosition != Vector3.zero)
        {
            var syncMsg = new PoseMsg
            {
                position = new geometry_msgs.Vector3Msg
                {
                    x = lastGazeboPosition.x,
                    y = lastGazeboPosition.z,
                    z = lastGazeboPosition.y
                },
                orientation = new geometry_msgs.QuaternionMsg
                {
                    x = lastGazeboOrientation.x,
                    y = lastGazeboOrientation.z,
                    z = lastGazeboOrientation.y,
                    w = lastGazeboOrientation.w
                }
            };

            ros.Publish(syncCommandTopic, syncMsg);
        }
    }
}
```

### 5. Launch System

Create a launch file to start the complete digital twin system:

```xml
<launch>
  <!-- Start Gazebo with the digital twin world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find your_package)/worlds/digital_twin_world.sdf"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- Spawn the robot in Gazebo -->
  <node name="spawn_robot" pkg="gazebo_ros" type="spawn_model"
        args="-file $(find your_package)/urdf/digital_twin_humanoid.urdf
              -urdf -model digital_twin_humanoid
              -x 0 -y 0 -z 0.5"/>

  <!-- Start robot state publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher"
        type="robot_state_publisher">
    <param name="publish_frequency" value="50.0"/>
  </node>

  <!-- Start joint state publisher -->
  <node name="joint_state_publisher" pkg="joint_state_publisher"
        type="joint_state_publisher">
    <param name="rate" value="50"/>
  </node>

  <!-- Unity ROS TCP Endpoint (this would be configured in Unity) -->
  <!-- Note: Unity needs to be started separately with ROS TCP Endpoint configured -->

</launch>
```

## Validation and Testing

### 1. Basic Functionality Tests

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import time

class DigitalTwinValidator(Node):
    def __init__(self):
        super().__init__('digital_twin_validator')

        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers for validation
        self.gazebo_joint_sub = self.create_subscription(
            JointState, '/gazebo_joint_states', self.gazebo_joint_callback, 10)
        self.unity_joint_sub = self.create_subscription(
            JointState, '/unity_joint_states', self.unity_joint_callback, 10)

        # Test variables
        self.gazebo_joints = {}
        self.unity_joints = {}

        self.get_logger().info('Digital Twin Validator started')

    def gazebo_joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.gazebo_joints[name] = msg.position[i]

    def unity_joint_callback(self, msg):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.unity_joints[name] = msg.position[i]

    def run_validation_tests(self):
        """Run a series of validation tests"""
        self.get_logger().info('Starting validation tests...')

        # Test 1: Basic movement
        self.test_basic_movement()

        # Test 2: Joint synchronization
        self.test_joint_synchronization()

        # Test 3: Sensor data consistency
        self.test_sensor_consistency()

    def test_basic_movement(self):
        """Test that robot moves consistently in both environments"""
        self.get_logger().info('Testing basic movement...')

        # Send movement command
        twist_msg = Twist()
        twist_msg.linear.x = 0.5  # Move forward
        twist_msg.angular.z = 0.2  # Turn slightly

        # Send command for 3 seconds
        start_time = time.time()
        while time.time() - start_time < 3.0:
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(0.1)

        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)

        self.get_logger().info('Movement test completed')

    def test_joint_synchronization(self):
        """Test that joints are synchronized between environments"""
        self.get_logger().info('Testing joint synchronization...')

        # Wait for joint data
        time.sleep(2.0)

        # Compare joint positions
        sync_count = 0
        total_count = 0

        for joint_name in self.gazebo_joints:
            if joint_name in self.unity_joints:
                gazebo_pos = self.gazebo_joints[joint_name]
                unity_pos = self.unity_joints[joint_name]

                # Check if positions are similar (within tolerance)
                if abs(gazebo_pos - unity_pos) < 0.1:
                    sync_count += 1
                total_count += 1

        sync_percentage = (sync_count / total_count) * 100 if total_count > 0 else 0
        self.get_logger().info(f'Joint synchronization: {sync_percentage:.1f}%')

    def test_sensor_consistency(self):
        """Test that sensor data is consistent between environments"""
        self.get_logger().info('Testing sensor consistency...')
        # Implementation would depend on your specific sensor setup
        pass

def main(args=None):
    rclpy.init(args=args)
    validator = DigitalTwinValidator()

    # Run validation tests
    validator.run_validation_tests()

    rclpy.spin(validator)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization

### 1. Unity Optimization Script

```csharp
using UnityEngine;
using System.Collections.Generic;

public class DigitalTwinOptimizer : MonoBehaviour
{
    [Header("Performance Settings")]
    public int maxFps = 60;
    public float physicsUpdateRate = 0.02f; // 50 Hz
    public bool useLOD = true;
    public int maxRenderedRobots = 10;

    [Header("Resource Management")]
    public bool enableOcclusionCulling = true;
    public bool useTextureCompression = true;

    private List<GameObject> activeRobots = new List<GameObject>();
    private float lastPhysicsUpdate;

    void Start()
    {
        ConfigurePerformanceSettings();
    }

    void ConfigurePerformanceSettings()
    {
        // Set target frame rate
        Application.targetFrameRate = maxFps;

        // Quality settings for performance
        QualitySettings.vSyncCount = 0; // Disable V-Sync for consistent frame rate

        // Physics settings
        Time.fixedDeltaTime = physicsUpdateRate;
    }

    void Update()
    {
        // Throttle physics updates if needed
        if (Time.time - lastPhysicsUpdate >= physicsUpdateRate)
        {
            UpdatePhysics();
            lastPhysicsUpdate = Time.time;
        }
    }

    void UpdatePhysics()
    {
        // Custom physics update logic if needed
        // This could include custom joint constraints or special physics handling
    }

    public void AddRobot(GameObject robot)
    {
        if (activeRobots.Count < maxRenderedRobots)
        {
            activeRobots.Add(robot);
        }
        else
        {
            Debug.LogWarning("Maximum robot count reached. Consider optimization.");
        }
    }

    public void RemoveRobot(GameObject robot)
    {
        activeRobots.Remove(robot);
    }

    void OnDestroy()
    {
        // Clean up resources
        activeRobots.Clear();
    }
}
```

## Project Deliverables

### Required Files
- `urdf/digital_twin_humanoid.urdf` - Unified robot model
- `worlds/digital_twin_world.sdf` - Gazebo world file
- `launch/digital_twin.launch.py` - ROS 2 launch file
- `Unity/Assets/Scripts/DigitalTwinController.cs` - Unity controller
- `Unity/Assets/Scripts/SynchronizationManager.cs` - Sync manager
- `scripts/validator.py` - Validation script
- `config/digital_twin.rviz` - RViz configuration

### Evaluation Criteria
- **Model Compatibility**: Robot works in both Gazebo and Unity
- **Synchronization**: Environments stay properly synchronized
- **Performance**: System runs smoothly in real-time
- **Robustness**: Handles errors and edge cases gracefully
- **Documentation**: Clear setup and usage instructions

## Advanced Extensions (Optional)

1. **Multi-Robot Digital Twin**: Extend to multiple robots
2. **Real-Time Analytics**: Add performance monitoring
3. **Cloud Integration**: Deploy to cloud-based Unity instances
4. **VR/AR Support**: Add virtual/augmented reality visualization

## Troubleshooting

### Common Issues
- **Synchronization Drift**: Check time synchronization between environments
- **Performance Problems**: Optimize Unity scene complexity and ROS message rates
- **Connection Issues**: Verify ROS TCP Endpoint configuration

### Debugging Commands
```bash
# Monitor synchronization
ros2 topic echo /gazebo_robot_pose
ros2 topic echo /unity_robot_pose

# Check joint states
ros2 topic echo /joint_states

# Monitor system performance
ros2 run tf2_tools view_frames
```

## Next Steps

After completing this project, you'll have a comprehensive understanding of digital twin systems for robotics. In Module 3, you'll learn to integrate NVIDIA Isaac tools for advanced AI and perception capabilities in your humanoid robots.