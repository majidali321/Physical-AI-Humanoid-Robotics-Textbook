---
sidebar_position: 5
---

# Week 7: Unity Digital Twin Exercises

## Overview

This section provides hands-on exercises to reinforce your understanding of Unity-ROS integration and digital twin implementation. Complete these exercises to gain practical experience with advanced Unity simulation, sensor integration, and real-time synchronization between Unity and Gazebo environments.

## Exercise 1: Advanced Robot Visualization

### Objective
Create a detailed humanoid robot model in Unity that responds to ROS joint state commands and provides realistic visual feedback.

### Instructions
1. Import a URDF model into Unity using the URDF Importer
2. Create custom shaders for robot materials
3. Implement joint-based animation
4. Add visual effects for sensor feedback

### Robot Model Template
```csharp
using UnityEngine;
using System.Collections.Generic;

public class AdvancedRobotVisualizer : MonoBehaviour
{
    [Header("Robot Configuration")]
    public List<JointVisualizer> jointVisualizers = new List<JointVisualizer>();
    public Material[] robotMaterials;
    public GameObject[] sensorVisualizers;

    [Header("Animation")]
    public float animationSmoothing = 0.1f;
    public bool showJointLimits = true;

    [System.Serializable]
    public class JointVisualizer
    {
        public string jointName;
        public Transform jointTransform;
        public float minAngle = -90f;
        public float maxAngle = 90f;
        public float currentAngle;
        public float targetAngle;
        public AnimationCurve motionCurve = AnimationCurve.Linear(0, 0, 1, 1);
    }

    void Start()
    {
        InitializeRobot();
    }

    void InitializeRobot()
    {
        // Find all joints in the robot hierarchy
        FindRobotJoints();

        // Apply initial materials
        ApplyRobotMaterials();

        // Initialize sensor visualizers
        InitializeSensorVisualizers();
    }

    void FindRobotJoints()
    {
        // Find all joints by name pattern
        Transform[] allTransforms = GetComponentsInChildren<Transform>();

        foreach (Transform t in allTransforms)
        {
            if (t.name.Contains("joint") || t.name.Contains("link"))
            {
                JointVisualizer newJoint = new JointVisualizer
                {
                    jointName = t.name,
                    jointTransform = t,
                    currentAngle = t.localEulerAngles.y,
                    targetAngle = t.localEulerAngles.y
                };

                jointVisualizers.Add(newJoint);
            }
        }
    }

    void ApplyRobotMaterials()
    {
        if (robotMaterials.Length == 0) return;

        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        int materialIndex = 0;

        foreach (Renderer renderer in renderers)
        {
            renderer.material = robotMaterials[materialIndex % robotMaterials.Length];
            materialIndex++;
        }
    }

    void InitializeSensorVisualizers()
    {
        // Find and initialize sensor objects
        foreach (Transform child in transform)
        {
            if (child.name.Contains("sensor") || child.name.Contains("camera"))
            {
                sensorVisualizers = sensorVisualizers ?? new GameObject[0];
                GameObject[] newSensors = new GameObject[sensorVisualizers.Length + 1];
                System.Array.Copy(sensorVisualizers, newSensors, sensorVisualizers.Length);
                newSensors[newSensors.Length - 1] = child.gameObject;
                sensorVisualizers = newSensors;
            }
        }
    }

    public void UpdateJointPositions(Dictionary<string, float> jointPositions)
    {
        foreach (JointVisualizer joint in jointVisualizers)
        {
            if (jointPositions.ContainsKey(joint.jointName))
            {
                joint.targetAngle = jointPositions[joint.jointName] * Mathf.Rad2Deg;
            }
        }
    }

    void Update()
    {
        AnimateJoints();
        UpdateSensorVisualizers();
    }

    void AnimateJoints()
    {
        foreach (JointVisualizer joint in jointVisualizers)
        {
            if (joint.jointTransform != null)
            {
                // Apply smoothing to joint movement
                joint.currentAngle = Mathf.Lerp(
                    joint.currentAngle,
                    joint.targetAngle,
                    animationSmoothing
                );

                // Clamp to joint limits
                joint.currentAngle = Mathf.Clamp(
                    joint.currentAngle,
                    joint.minAngle,
                    joint.maxAngle
                );

                // Apply rotation
                Vector3 eulerAngles = joint.jointTransform.localEulerAngles;
                eulerAngles.y = joint.currentAngle;
                joint.jointTransform.localEulerAngles = eulerAngles;
            }
        }
    }

    void UpdateSensorVisualizers()
    {
        // Update sensor visualization (e.g., show active sensors)
        foreach (GameObject sensor in sensorVisualizers)
        {
            // Example: Make active sensors glow
            if (sensor.activeSelf)
            {
                Renderer renderer = sensor.GetComponent<Renderer>();
                if (renderer != null)
                {
                    // Add glowing effect or other visual feedback
                    renderer.material.SetFloat("_EmissionScaleUI", 1.0f);
                }
            }
        }
    }
}
```

### Unity Shader for Robot Materials
Create a custom shader for realistic robot materials:

```hlsl
// RobotMaterial.shader
Shader "Custom/RobotMaterial"
{
    Properties
    {
        _Color ("Color", Color) = (0.8, 0.8, 0.8, 1.0)
        _Metallic ("Metallic", Range(0.0, 1.0)) = 0.5
        _Smoothness ("Smoothness", Range(0.0, 1.0)) = 0.5
        _EmissionColor ("Emission Color", Color) = (0,0,0,1)
        _EmissionScaleUI ("Emission Scale", Range(0.0, 2.0)) = 0.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        struct Input
        {
            float2 uv_MainTex;
        };

        fixed4 _Color;
        half _Metallic;
        half _Smoothness;
        fixed4 _EmissionColor;
        float _EmissionScaleUI;

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            fixed4 c = _Color;
            o.Albedo = c.rgb;
            o.Metallic = _Metallic;
            o.Smoothness = _Smoothness;
            o.Alpha = c.a;
            o.Emission = _EmissionColor.rgb * _EmissionScaleUI;
        }
        ENDCG
    }
    FallBack "Diffuse"
}
```

### Expected Output
- Robot responds to joint state commands with smooth animation
- Materials have realistic metallic and smoothness properties
- Sensors provide visual feedback when active

## Exercise 2: Multi-Sensor Integration

### Objective
Integrate multiple sensor types in Unity and publish their data to ROS topics.

### Instructions
1. Create a robot with multiple sensor types (camera, LiDAR, IMU)
2. Implement sensor simulation for each type
3. Publish sensor data to appropriate ROS topics
4. Validate sensor data consistency

### Multi-Sensor Controller Template
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class MultiSensorController : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public UnityCameraSensor cameraSensor;
    public UnityLidarSensor lidarSensor;
    public UnityIMUSensor imuSensor;

    [Header("Sensor Fusion")]
    public bool enableFusion = true;
    public string fusedTopic = "/sensors/fused_data";

    private ROSConnection ros;
    private float lastFusionUpdate;
    private float fusionInterval = 0.1f; // 10Hz fusion

    void Start()
    {
        ros = ROSConnection.instance;

        // Initialize individual sensors
        InitializeSensors();

        lastFusionUpdate = Time.time;
    }

    void InitializeSensors()
    {
        // Create and configure individual sensors
        if (cameraSensor == null)
        {
            GameObject cameraGO = new GameObject("CameraSensor");
            cameraGO.transform.SetParent(transform);
            cameraGO.transform.localPosition = new Vector3(0, 0.5f, 0.1f); // Head position
            cameraSensor = cameraGO.AddComponent<UnityCameraSensor>();
        }

        if (lidarSensor == null)
        {
            GameObject lidarGO = new GameObject("LidarSensor");
            lidarGO.transform.SetParent(transform);
            lidarGO.transform.localPosition = new Vector3(0, 0.6f, 0); // Torso position
            lidarSensor = lidarGO.AddComponent<UnityLidarSensor>();
        }

        if (imuSensor == null)
        {
            GameObject imuGO = new GameObject("IMUSensor");
            imuGO.transform.SetParent(transform);
            imuGO.transform.localPosition = new Vector3(0, 0.4f, 0); // Center of mass
            imuSensor = imuGO.AddComponent<UnityIMUSensor>();
        }
    }

    void Update()
    {
        if (enableFusion && Time.time - lastFusionUpdate >= fusionInterval)
        {
            PublishFusedSensorData();
            lastFusionUpdate = Time.time;
        }
    }

    void PublishFusedSensorData()
    {
        // Create fused sensor message
        var fusedMsg = new UnitySensorMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = "fused_sensor_frame"
            }
        };

        // Combine data from all sensors
        fusedMsg.position = new float[] {
            transform.position.x,
            transform.position.y,
            transform.position.z
        };

        fusedMsg.rotation = new float[] {
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        };

        // Add sensor-specific data
        fusedMsg.sensor_data = new float[20]; // Example: 20 fused values
        fusedMsg.sensor_data[0] = (float)Time.time; // Timestamp
        // Add more fused data here...

        ros.Publish(fusedTopic, fusedMsg);
    }

    // Method to handle sensor calibration
    public void CalibrateSensors()
    {
        // Implement sensor calibration logic
        Debug.Log("Calibrating all sensors...");

        // Example: Reset sensor offsets
        if (imuSensor != null)
        {
            // Reset IMU bias
        }
    }
}
```

## Exercise 3: Real-time Environment Synchronization

### Objective
Create a system that synchronizes Unity and Gazebo environments in real-time.

### Instructions
1. Implement a synchronization manager
2. Create bidirectional communication between environments
3. Handle synchronization errors and corrections
4. Monitor and visualize synchronization quality

### Synchronization Manager Template
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using System.Collections.Generic;

public class RealTimeSyncManager : MonoBehaviour
{
    [Header("Synchronization Settings")]
    public float syncFrequency = 50.0f; // Hz
    public float maxPositionError = 0.1f;
    public float maxOrientationError = 5.0f; // degrees
    public bool autoCorrect = true;

    [Header("Topics")]
    public string gazeboPoseTopic = "/gazebo/robot_pose";
    public string unityPoseTopic = "/unity/robot_pose";
    public string syncStatusTopic = "/sync/status";

    private ROSConnection ros;
    private float syncInterval;
    private float lastSyncTime;
    private Queue<PoseSyncData> syncHistory = new Queue<PoseSyncData>();
    private const int HISTORY_SIZE = 10;

    [System.Serializable]
    public class PoseSyncData
    {
        public Vector3 gazeboPosition;
        public Quaternion gazeboOrientation;
        public Vector3 unityPosition;
        public Quaternion unityOrientation;
        public float timestamp;
        public float positionError;
        public float orientationError;
    }

    void Start()
    {
        ros = ROSConnection.instance;
        syncInterval = 1.0f / syncFrequency;
        lastSyncTime = Time.time;

        // Subscribe to pose updates
        ros.Subscribe<PoseMsg>(gazeboPoseTopic, GazeboPoseCallback);
        ros.Subscribe<PoseMsg>(unityPoseTopic, UnityPoseCallback);
    }

    void GazeboPoseCallback(PoseMsg pose)
    {
        // Convert Gazebo pose to Unity coordinate system
        Vector3 gazeboPos = new Vector3(
            (float)pose.position.x,
            (float)pose.position.z, // Swap Y and Z for Unity
            (float)pose.position.y
        );

        Quaternion gazeboRot = new Quaternion(
            (float)pose.orientation.x,
            (float)pose.orientation.z,
            (float)pose.orientation.y,
            (float)pose.orientation.w
        );

        UpdateSynchronization(gazeboPos, gazeboRot, true);
    }

    void UnityPoseCallback(PoseMsg pose)
    {
        // Convert Unity pose to ROS coordinate system
        Vector3 unityPos = new Vector3(
            (float)pose.position.x,
            (float)pose.position.z, // Swap Y and Z for Unity
            (float)pose.position.y
        );

        Quaternion unityRot = new Quaternion(
            (float)pose.orientation.x,
            (float)pose.orientation.z,
            (float)pose.orientation.y,
            (float)pose.orientation.w
        );

        UpdateSynchronization(unityPos, unityRot, false);
    }

    void UpdateSynchronization(Vector3 position, Quaternion orientation, bool isGazebo)
    {
        if (isGazebo)
        {
            // Store Gazebo data, keep Unity data from last sync
            if (syncHistory.Count > 0)
            {
                PoseSyncData lastSync = syncHistory.Peek();
                AddSyncData(position, orientation, lastSync.unityPosition, lastSync.unityOrientation);
            }
            else
            {
                AddSyncData(position, orientation, transform.position, transform.rotation);
            }
        }
        else
        {
            // Store Unity data, keep Gazebo data from last sync
            if (syncHistory.Count > 0)
            {
                PoseSyncData lastSync = syncHistory.Peek();
                AddSyncData(lastSync.gazeboPosition, lastSync.gazeboOrientation, position, orientation);
            }
            else
            {
                AddSyncData(transform.position, transform.rotation, position, orientation);
            }
        }
    }

    void AddSyncData(Vector3 gazeboPos, Quaternion gazeboRot, Vector3 unityPos, Quaternion unityRot)
    {
        PoseSyncData syncData = new PoseSyncData
        {
            gazeboPosition = gazeboPos,
            gazeboOrientation = gazeboRot,
            unityPosition = unityPos,
            unityOrientation = unityRot,
            timestamp = Time.time,
            positionError = Vector3.Distance(gazeboPos, unityPos),
            orientationError = Quaternion.Angle(gazeboRot, unityRot)
        };

        syncHistory.Enqueue(syncData);

        if (syncHistory.Count > HISTORY_SIZE)
        {
            syncHistory.Dequeue();
        }

        // Check for synchronization errors
        if (syncData.positionError > maxPositionError || syncData.orientationError > maxOrientationError)
        {
            Debug.LogWarning($"Synchronization error - Pos: {syncData.positionError:F3}, Rot: {syncData.orientationError:F2}");

            if (autoCorrect)
            {
                ApplySynchronizationCorrection(syncData);
            }
        }

        // Publish synchronization status
        PublishSyncStatus(syncData);
    }

    void ApplySynchronizationCorrection(PoseSyncData syncData)
    {
        // Correct Unity position to match Gazebo
        transform.position = syncData.gazeboPosition;
        transform.rotation = syncData.gazeboOrientation;
    }

    void PublishSyncStatus(PoseSyncData syncData)
    {
        var statusMsg = new geometry_msgs.TwistMsg // Using Twist as a generic status message
        {
            linear = new geometry_msgs.Vector3Msg
            {
                x = syncData.positionError,
                y = syncData.orientationError,
                z = syncData.timestamp
            }
        };

        ros.Publish(syncStatusTopic, statusMsg);
    }

    void Update()
    {
        if (Time.time - lastSyncTime >= syncInterval)
        {
            // Publish current Unity pose
            var poseMsg = new PoseMsg
            {
                position = new geometry_msgs.Vector3Msg
                {
                    x = transform.position.x,
                    y = transform.position.z, // Unity to ROS conversion
                    z = transform.position.y
                },
                orientation = new geometry_msgs.QuaternionMsg
                {
                    x = transform.rotation.x,
                    y = transform.rotation.z,
                    z = transform.rotation.y,
                    w = transform.rotation.w
                }
            };

            ros.Publish(unityPoseTopic, poseMsg);
            lastSyncTime = Time.time;
        }
    }

    // Method to visualize synchronization quality
    public void VisualizeSyncQuality()
    {
        if (syncHistory.Count > 0)
        {
            PoseSyncData latest = syncHistory.Peek();
            Debug.Log($"Sync Quality - Error: {latest.positionError:F3}m, {latest.orientationError:F2}°");
        }
    }
}
```

## Exercise 4: Interactive Environment with Physics

### Objective
Create an interactive Unity environment where robots can manipulate objects with realistic physics.

### Instructions
1. Design an environment with movable objects
2. Implement physics-based interaction
3. Create manipulation challenges
4. Validate physics realism against Gazebo

### Interactive Environment Template
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class InteractiveEnvironment : MonoBehaviour
{
    [Header("Environment Configuration")]
    public GameObject[] manipulableObjects;
    public Transform[] spawnPoints;
    public PhysicsMaterial[] surfaceMaterials;

    [Header("Manipulation")]
    public string manipulationTopic = "/manipulation/command";
    public float manipulationForce = 10.0f;

    [Header("Challenges")]
    public ManipulationChallenge[] challenges;

    private ROSConnection ros;
    private int currentChallenge = 0;

    [System.Serializable]
    public class ManipulationChallenge
    {
        public string name;
        public GameObject[] targetObjects;
        public Transform[] targetPositions;
        public string description;
    }

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to manipulation commands
        ros.Subscribe<PoseStampedMsg>(manipulationTopic, ManipulationCallback);

        // Initialize environment
        InitializeEnvironment();
    }

    void InitializeEnvironment()
    {
        // Spawn manipulable objects
        for (int i = 0; i < spawnPoints.Length && i < manipulableObjects.Length; i++)
        {
            GameObject obj = Instantiate(manipulableObjects[i], spawnPoints[i].position, Quaternion.identity);
            obj.name = $"ManipulableObject_{i}";

            // Add physics properties
            Rigidbody rb = obj.AddComponent<Rigidbody>();
            rb.mass = Random.Range(0.5f, 5.0f);
            rb.drag = 0.5f;
            rb.angularDrag = 0.5f;

            // Add collider if not present
            if (obj.GetComponent<Collider>() == null)
            {
                obj.AddComponent<BoxCollider>();
            }
        }
    }

    void ManipulationCallback(PoseStampedMsg command)
    {
        // Parse manipulation command
        Vector3 targetPosition = new Vector3(
            (float)command.pose.position.x,
            (float)command.pose.position.z, // Unity coordinate conversion
            (float)command.pose.position.y
        );

        // Find nearest manipulable object
        GameObject nearestObject = FindNearestObject(targetPosition);

        if (nearestObject != null)
        {
            // Apply manipulation force
            Rigidbody rb = nearestObject.GetComponent<Rigidbody>();
            if (rb != null)
            {
                Vector3 direction = (targetPosition - nearestObject.transform.position).normalized;
                rb.AddForce(direction * manipulationForce, ForceMode.Impulse);
            }
        }
    }

    GameObject FindNearestObject(Vector3 position)
    {
        GameObject nearest = null;
        float minDistance = float.MaxValue;

        foreach (Transform child in transform)
        {
            if (child.CompareTag("ManipulableObject"))
            {
                float distance = Vector3.Distance(child.position, position);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearest = child.gameObject;
                }
            }
        }

        return nearest;
    }

    // Method to create manipulation challenges
    public void CreateChallenge(int challengeIndex)
    {
        if (challengeIndex < challenges.Length)
        {
            currentChallenge = challengeIndex;
            ManipulationChallenge challenge = challenges[challengeIndex];

            Debug.Log($"Starting challenge: {challenge.name}");
            Debug.Log(challenge.description);

            // Reset challenge objects to initial positions
            for (int i = 0; i < challenge.targetObjects.Length; i++)
            {
                if (i < challenge.targetPositions.Length)
                {
                    challenge.targetObjects[i].transform.position = challenge.targetPositions[i].position;
                }
            }
        }
    }

    // Method to validate physics realism
    public void ValidatePhysics()
    {
        // Compare Unity physics behavior with expected Gazebo behavior
        Debug.Log("Validating physics realism...");

        // This would typically involve comparing simulation results
        // with real-world data or Gazebo simulation data
    }
}
```

## Exercise 5: Performance Optimization and Scalability

### Objective
Optimize the Unity simulation for performance and scalability with multiple robots.

### Instructions
1. Implement Level of Detail (LOD) systems
2. Create object pooling for frequently instantiated objects
3. Optimize rendering and physics for multiple robots
4. Test scalability with increasing robot counts

### Performance Optimization Template
```csharp
using UnityEngine;
using System.Collections.Generic;

public class PerformanceOptimizer : MonoBehaviour
{
    [Header("LOD Configuration")]
    public LODGroup[] robotLODGroups;
    public float[] lodDistances = { 10f, 30f, 50f };
    public int maxVisibleRobots = 50;

    [Header("Object Pooling")]
    public GameObject[] pooledObjects;
    public int poolSize = 100;

    [Header("Performance Monitoring")]
    public bool enableMonitoring = true;
    public float monitoringInterval = 1.0f;
    public Text performanceText; // Unity UI Text element

    private Dictionary<GameObject, Queue<GameObject>> objectPools = new Dictionary<GameObject, Queue<GameObject>>();
    private List<GameObject> activeRobots = new List<GameObject>();
    private float lastMonitoringTime;
    private int frameCount = 0;
    private float fpsTimer = 0f;

    void Start()
    {
        InitializeObjectPools();
        lastMonitoringTime = Time.time;
    }

    void InitializeObjectPools()
    {
        foreach (GameObject prefab in pooledObjects)
        {
            Queue<GameObject> pool = new Queue<GameObject>();

            for (int i = 0; i < poolSize; i++)
            {
                GameObject obj = Instantiate(prefab);
                obj.SetActive(false);
                pool.Enqueue(obj);
            }

            objectPools[prefab] = pool;
        }
    }

    public GameObject GetPooledObject(GameObject prefab)
    {
        if (objectPools.ContainsKey(prefab))
        {
            Queue<GameObject> pool = objectPools[prefab];

            if (pool.Count > 0)
            {
                GameObject obj = pool.Dequeue();
                obj.SetActive(true);
                return obj;
            }
            else
            {
                // Pool exhausted, create new object
                return Instantiate(prefab);
            }
        }

        return Instantiate(prefab);
    }

    public void ReturnToPool(GameObject obj, GameObject prefab)
    {
        if (objectPools.ContainsKey(prefab))
        {
            obj.SetActive(false);
            objectPools[prefab].Enqueue(obj);
        }
        else
        {
            Destroy(obj);
        }
    }

    void Update()
    {
        UpdateLOD();
        UpdatePerformanceMetrics();

        if (enableMonitoring && Time.time - lastMonitoringTime >= monitoringInterval)
        {
            LogPerformanceMetrics();
            lastMonitoringTime = Time.time;
        }
    }

    void UpdateLOD()
    {
        foreach (GameObject robot in activeRobots)
        {
            if (robot != null && robot.GetComponent<LODGroup>() != null)
            {
                LODGroup lodGroup = robot.GetComponent<LODGroup>();
                float distance = Vector3.Distance(robot.transform.position, Camera.main.transform.position);

                // Simple LOD selection based on distance
                if (distance > lodDistances[2])
                {
                    lodGroup.ForceLOD(2); // Lowest detail
                }
                else if (distance > lodDistances[1])
                {
                    lodGroup.ForceLOD(1);
                }
                else
                {
                    lodGroup.ForceLOD(0); // Highest detail
                }
            }
        }
    }

    void UpdatePerformanceMetrics()
    {
        frameCount++;
        fpsTimer += Time.unscaledDeltaTime;

        if (fpsTimer >= 1.0f)
        {
            float currentFPS = frameCount / fpsTimer;

            // Update UI if available
            if (performanceText != null)
            {
                performanceText.text = $"FPS: {currentFPS:F1}\nRobots: {activeRobots.Count}";
            }

            frameCount = 0;
            fpsTimer = 0f;
        }
    }

    void LogPerformanceMetrics()
    {
        float memoryUsage = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() / (1024f * 1024f); // MB
        float currentFPS = 1.0f / Time.unscaledDeltaTime;

        Debug.Log($"Performance - FPS: {currentFPS:F1}, Memory: {memoryUsage:F1}MB, Robots: {activeRobots.Count}");
    }

    public void AddRobot(GameObject robot)
    {
        if (activeRobots.Count < maxVisibleRobots)
        {
            activeRobots.Add(robot);
        }
        else
        {
            Debug.LogWarning("Maximum robot count reached. Consider optimization or culling.");
        }
    }

    public void RemoveRobot(GameObject robot)
    {
        activeRobots.Remove(robot);
    }

    // Method to implement robot culling
    public void CullDistantRobots()
    {
        List<GameObject> robotsToRemove = new List<GameObject>();

        foreach (GameObject robot in activeRobots)
        {
            if (robot != null)
            {
                float distance = Vector3.Distance(robot.transform.position, Camera.main.transform.position);

                if (distance > 100f) // Cull robots beyond 100m
                {
                    robotsToRemove.Add(robot);
                }
            }
        }

        foreach (GameObject robot in robotsToRemove)
        {
            RemoveRobot(robot);
            ReturnToPool(robot, robot); // Return to pool instead of destroying
        }
    }
}
```

## Exercise 6: Advanced Visualization and Analytics

### Objective
Implement advanced visualization tools and analytics for the digital twin system.

### Instructions
1. Create real-time analytics dashboard
2. Implement trajectory visualization
3. Add performance monitoring
4. Create data export functionality

### Analytics Dashboard Template
```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class AnalyticsDashboard : MonoBehaviour
{
    [Header("UI Elements")]
    public Text fpsText;
    public Text robotCountText;
    public Text syncErrorText;
    public Text memoryText;
    public LineRenderer trajectoryRenderer;
    public Text eventLog;

    [Header("Analytics Configuration")]
    public int maxTrajectoryPoints = 1000;
    public float analyticsUpdateRate = 0.1f;
    public string dataExportPath = "AnalyticsData/";

    private float lastAnalyticsUpdate;
    private List<Vector3> trajectoryPoints = new List<Vector3>();
    private Queue<string> eventLogEntries = new Queue<string>();
    private const int MAX_LOG_ENTRIES = 50;

    void Start()
    {
        lastAnalyticsUpdate = Time.time;
        InitializeDashboard();
    }

    void InitializeDashboard()
    {
        // Initialize trajectory renderer
        if (trajectoryRenderer != null)
        {
            trajectoryRenderer.positionCount = 0;
        }

        // Initialize event log
        if (eventLog != null)
        {
            eventLog.text = "";
        }
    }

    void Update()
    {
        if (Time.time - lastAnalyticsUpdate >= analyticsUpdateRate)
        {
            UpdateAnalytics();
            lastAnalyticsUpdate = Time.time;
        }
    }

    void UpdateAnalytics()
    {
        // Update FPS
        if (fpsText != null)
        {
            float fps = 1.0f / Time.unscaledDeltaTime;
            fpsText.text = $"FPS: {fps:F1}";
        }

        // Update memory usage
        if (memoryText != null)
        {
            float memoryMB = UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong() / (1024f * 1024f);
            memoryText.text = $"Memory: {memoryMB:F1} MB";
        }

        // Add current position to trajectory
        AddTrajectoryPoint(transform.position);
    }

    void AddTrajectoryPoint(Vector3 position)
    {
        trajectoryPoints.Add(position);

        if (trajectoryPoints.Count > maxTrajectoryPoints)
        {
            trajectoryPoints.RemoveAt(0);
        }

        if (trajectoryRenderer != null)
        {
            trajectoryRenderer.positionCount = trajectoryPoints.Count;
            trajectoryRenderer.SetPositions(trajectoryPoints.ToArray());
        }
    }

    public void LogEvent(string eventMessage)
    {
        string timestampedMessage = $"[{System.DateTime.Now:HH:mm:ss}] {eventMessage}";
        eventLogEntries.Enqueue(timestampedMessage);

        if (eventLogEntries.Count > MAX_LOG_ENTRIES)
        {
            eventLogEntries.Dequeue();
        }

        if (eventLog != null)
        {
            eventLog.text = string.Join("\n", eventLogEntries.ToArray());
        }
    }

    public void ExportAnalyticsData()
    {
        // Export trajectory data
        string trajectoryData = "Time,X,Y,Z\n";
        for (int i = 0; i < trajectoryPoints.Count; i++)
        {
            Vector3 point = trajectoryPoints[i];
            trajectoryData += $"{i * analyticsUpdateRate:F3},{point.x:F3},{point.y:F3},{point.z:F3}\n";
        }

        // Write to file
        string fileName = $"{dataExportPath}trajectory_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv";
        System.IO.Directory.CreateDirectory(dataExportPath);
        System.IO.File.WriteAllText(fileName, trajectoryData);

        LogEvent($"Analytics data exported to: {fileName}");
    }
}
```

## Solutions and Hints

### Exercise 1 Solution
- Use Unity's built-in animation system for smooth joint movements
- Apply materials with metallic and smoothness maps for realistic appearance
- Use coroutines for asynchronous sensor updates

### Exercise 2 Solution
- Implement separate update loops for different sensor types
- Use Unity's physics raycasting for LiDAR simulation
- Apply realistic noise models to sensor data

### Exercise 3 Solution
- Implement a feedback control system for synchronization
- Use interpolation for smooth position updates
- Monitor and log synchronization errors

## Evaluation Criteria

- **Implementation Quality**: Code follows Unity and ROS best practices
- **Functionality**: All exercises complete and working as expected
- **Performance**: Systems run efficiently with good frame rates
- **Integration**: Unity-ROS communication works reliably
- **Documentation**: Clear comments and explanations
- **Innovation**: Creative solutions and extensions

## Next Steps

After completing these exercises, you should have a comprehensive understanding of Unity-ROS integration and digital twin systems. Consider exploring advanced topics like machine learning integration, cloud deployment, or VR/AR interfaces for your digital twin systems.