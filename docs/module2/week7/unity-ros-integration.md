---
sidebar_position: 3
---

# Week 7: Unity-ROS Integration

## Overview

Unity-ROS integration enables seamless communication between Unity's powerful 3D visualization capabilities and ROS 2's robotics framework. This section covers advanced techniques for integrating Unity with ROS 2, including message passing, sensor simulation, robot control, and bidirectional communication patterns.

## Learning Objectives

By the end of this section, you will be able to:

- Implement bidirectional communication between Unity and ROS 2
- Simulate various types of sensors in Unity and publish data to ROS
- Control Unity objects using ROS messages
- Create custom message types for Unity-ROS communication
- Implement advanced robotics workflows with Unity visualization

## Advanced ROS TCP Endpoint Configuration

### Connection Management

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class AdvancedROSConnection : MonoBehaviour
{
    [Header("Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;
    public float connectionTimeout = 10.0f;
    public bool autoReconnect = true;

    private ROSConnection ros;
    private bool isConnected = false;
    private float lastConnectionAttempt;

    void Start()
    {
        InitializeROSConnection();
    }

    void InitializeROSConnection()
    {
        ros = ROSConnection.instance;
        ros.OnConnected += OnConnected;
        ros.OnDisconnected += OnDisconnected;

        // Configure connection settings
        ros.Initialize(rosIPAddress, rosPort);
    }

    void OnConnected()
    {
        isConnected = true;
        Debug.Log("Connected to ROS successfully!");

        // Subscribe to topics after connection
        SubscribeToTopics();
    }

    void OnDisconnected()
    {
        isConnected = false;
        Debug.Log("Disconnected from ROS");

        if (autoReconnect && Time.time - lastConnectionAttempt > 5.0f)
        {
            lastConnectionAttempt = Time.time;
            ros.Initialize(rosIPAddress, rosPort);
        }
    }

    void SubscribeToTopics()
    {
        // Subscribe to robot control commands
        ros.Subscribe<geometry_msgs.TwistMsg>("cmd_vel", HandleCmdVel);
        ros.Subscribe<std_msgs.StringMsg>("robot_command", HandleRobotCommand);
    }

    void Update()
    {
        if (!isConnected && autoReconnect && Time.time - lastConnectionAttempt > 5.0f)
        {
            lastConnectionAttempt = Time.time;
            ros.Initialize(rosIPAddress, rosPort);
        }
    }

    void OnApplicationQuit()
    {
        if (ros != null)
        {
            ros.OnConnected -= OnConnected;
            ros.OnDisconnected -= OnDisconnected;
        }
    }
}
```

### Custom Message Types

Creating custom message types for Unity-specific data:

```csharp
// Custom message for Unity-specific sensor data
namespace RosMessageTypes.Unity
{
    [System.Serializable]
    public class UnitySensorMsg : Message
    {
        public const string k_RosMessageName = "unity_msgs/UnitySensor";
        public override string RosMessageName => k_RosMessageName;

        public RosMessageTypes.Std.Header header;
        public float[] position; // [x, y, z]
        public float[] rotation; // [x, y, z, w] (quaternion)
        public float[] velocity; // [x, y, z]
        public float[] angular_velocity; // [x, y, z]
        public float[] sensor_data; // Custom sensor readings

        public UnitySensorMsg()
        {
            header = new RosMessageTypes.Std.Header();
            position = new float[3];
            rotation = new float[4];
            velocity = new float[3];
            angular_velocity = new float[3];
            sensor_data = new float[10]; // Example: 10 custom sensor values
        }

        public UnitySensorMsg(RosMessageTypes.Std.Header header, float[] position, float[] rotation,
                             float[] velocity, float[] angular_velocity, float[] sensor_data)
        {
            this.header = header;
            this.position = position;
            this.rotation = rotation;
            this.velocity = velocity;
            this.angular_velocity = angular_velocity;
            this.sensor_data = sensor_data;
        }

        public override void FillFrom(Unity.Robotics.ROSTCPConnector.MessageDeserializer deserializer)
        {
            deserializer.Read(out header);
            deserializer.Read(out position);
            deserializer.Read(out rotation);
            deserializer.Read(out velocity);
            deserializer.Read(out angular_velocity);
            deserializer.Read(out sensor_data);
        }

        public override string ToString()
        {
            return $"UnitySensorMsg: Pos=({position[0]:F2}, {position[1]:F2}, {position[2]:F2}) " +
                   $"Rot=({rotation[0]:F2}, {rotation[1]:F2}, {rotation[2]:F2}, {rotation[3]:F2})";
        }
    }
}
```

## Advanced Sensor Simulation

### Camera Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera unityCamera;
    public string topicName = "/camera/image_raw";
    public int publishFrequency = 30; // Hz
    public int imageWidth = 640;
    public int imageHeight = 480;

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.instance;

        if (unityCamera == null)
            unityCamera = GetComponent<Camera>();

        InitializeCameraSensor();
    }

    void InitializeCameraSensor()
    {
        // Create render texture for camera capture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        unityCamera.targetTexture = renderTexture;

        // Create 2D texture for conversion
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

        publishInterval = 1.0f / publishFrequency;
        lastPublishTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishCameraData();
            lastPublishTime = Time.time;
        }
    }

    void PublishCameraData()
    {
        // Capture camera image
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();

        // Convert to ROS message
        var imageMsg = CreateImageMessage(texture2D);

        // Publish to ROS
        ros.Publish(topicName, imageMsg);
    }

    ImageMsg CreateImageMessage(Texture2D texture)
    {
        byte[] imageData = texture.EncodeToPNG();

        var imageMsg = new ImageMsg
        {
            header = new HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = transform.name + "_frame"
            },
            height = (uint)texture.height,
            width = (uint)texture.width,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(texture.width * 3), // 3 bytes per pixel (RGB)
            data = imageData
        };

        return imageMsg;
    }

    void OnDestroy()
    {
        if (renderTexture != null)
            Destroy(renderTexture);
        if (texture2D != null)
            Destroy(texture2D);
    }
}
```

### LiDAR Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityLidarSensor : MonoBehaviour
{
    [Header("LiDAR Configuration")]
    public string topicName = "/scan";
    public float minAngle = -Mathf.PI / 2; // -90 degrees
    public float maxAngle = Mathf.PI / 2;  // 90 degrees
    public int rayCount = 720;
    public float maxRange = 30.0f;
    public float minRange = 0.1f;
    public float publishFrequency = 10.0f; // Hz

    private ROSConnection ros;
    private float[] ranges;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.instance;
        ranges = new float[rayCount];
        publishInterval = 1.0f / publishFrequency;
        lastPublishTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            SimulateLidarScan();
            PublishLidarData();
            lastPublishTime = Time.time;
        }
    }

    void SimulateLidarScan()
    {
        float angleIncrement = (maxAngle - minAngle) / rayCount;

        for (int i = 0; i < rayCount; i++)
        {
            float angle = minAngle + i * angleIncrement;

            // Calculate ray direction in world space
            Vector3 rayDirection = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            rayDirection = transform.TransformDirection(rayDirection);

            // Perform raycast
            RaycastHit hit;
            if (Physics.Raycast(transform.position, rayDirection, out hit, maxRange))
            {
                float distance = hit.distance;
                ranges[i] = distance >= minRange ? distance : 0; // Invalid if too close
            }
            else
            {
                ranges[i] = maxRange + 1; // Invalid range
            }
        }
    }

    void PublishLidarData()
    {
        var scanMsg = new LaserScanMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = transform.name + "_frame"
            },
            angle_min = minAngle,
            angle_max = maxAngle,
            angle_increment = (maxAngle - minAngle) / rayCount,
            time_increment = 0, // Not used for simulation
            scan_time = 1.0f / publishFrequency,
            range_min = minRange,
            range_max = maxRange,
            ranges = ranges,
            intensities = new float[rayCount] // Initialize with zeros
        };

        ros.Publish(topicName, scanMsg);
    }
}
```

### IMU Sensor Simulation

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityIMUSensor : MonoBehaviour
{
    [Header("IMU Configuration")]
    public string topicName = "/imu/data";
    public float publishFrequency = 100.0f; // Hz
    public Vector3 noiseLevel = new Vector3(0.01f, 0.01f, 0.01f); // Noise in measurements

    private ROSConnection ros;
    private Rigidbody attachedRigidbody;
    private float publishInterval;
    private float lastPublishTime;

    void Start()
    {
        ros = ROSConnection.instance;
        attachedRigidbody = GetComponent<Rigidbody>() ?? GetComponentInParent<Rigidbody>();
        publishInterval = 1.0f / publishFrequency;
        lastPublishTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastPublishTime >= publishInterval)
        {
            PublishIMUData();
            lastPublishTime = Time.time;
        }
    }

    void PublishIMUData()
    {
        var imuMsg = new ImuMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1000000000)
                },
                frame_id = transform.name + "_frame"
            }
        };

        // Set orientation (from Unity rotation to ROS quaternion)
        Quaternion unityRotation = transform.rotation;
        imuMsg.orientation = new geometry_msgs.QuaternionMsg
        {
            x = unityRotation.x,
            y = unityRotation.y,
            z = unityRotation.z,
            w = unityRotation.w
        };

        // Add noise to orientation
        imuMsg.orientation.x += Random.Range(-noiseLevel.x, noiseLevel.x);
        imuMsg.orientation.y += Random.Range(-noiseLevel.y, noiseLevel.y);
        imuMsg.orientation.z += Random.Range(-noiseLevel.z, noiseLevel.z);

        // Set angular velocity (from Rigidbody if available)
        if (attachedRigidbody != null)
        {
            Vector3 angularVel = attachedRigidbody.angularVelocity;
            imuMsg.angular_velocity = new geometry_msgs.Vector3Msg
            {
                x = angularVel.x,
                y = angularVel.y,
                z = angularVel.z
            };

            // Add noise to angular velocity
            imuMsg.angular_velocity.x += Random.Range(-noiseLevel.x, noiseLevel.x);
            imuMsg.angular_velocity.y += Random.Range(-noiseLevel.y, noiseLevel.y);
            imuMsg.angular_velocity.z += Random.Range(-noiseLevel.z, noiseLevel.z);
        }

        // Set linear acceleration
        Vector3 gravity = Physics.gravity;
        Vector3 linearAcc = -gravity; // Compensate for Unity's gravity

        // If we have a rigidbody, add its acceleration
        if (attachedRigidbody != null)
        {
            linearAcc += attachedRigidbody.velocity / Time.deltaTime; // Approximate acceleration
        }

        imuMsg.linear_acceleration = new geometry_msgs.Vector3Msg
        {
            x = linearAcc.x,
            y = linearAcc.y,
            z = linearAcc.z
        };

        // Add noise to linear acceleration
        imuMsg.linear_acceleration.x += Random.Range(-noiseLevel.x, noiseLevel.x);
        imuMsg.linear_acceleration.y += Random.Range(-noiseLevel.y, noiseLevel.y);
        imuMsg.linear_acceleration.z += Random.Range(-noiseLevel.z, noiseLevel.z);

        ros.Publish(topicName, imuMsg);
    }
}
```

## Robot Control from ROS

### Unity Robot Controller

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    [Header("Control Configuration")]
    public float linearSpeed = 1.0f;
    public float angularSpeed = 1.0f;
    public float maxVelocity = 2.0f;

    private ROSConnection ros;
    private Rigidbody rb;
    private Vector3 targetLinearVelocity;
    private Vector3 targetAngularVelocity;

    void Start()
    {
        ros = ROSConnection.instance;
        rb = GetComponent<Rigidbody>() ?? gameObject.AddComponent<Rigidbody>();

        // Configure rigidbody for robot physics
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
        rb.mass = 50.0f; // Realistic humanoid mass

        // Subscribe to control commands
        ros.Subscribe<TwistMsg>("cmd_vel", CmdVelCallback);
    }

    void CmdVelCallback(TwistMsg cmd)
    {
        // Convert ROS Twist to Unity vectors
        Vector3 linear = new Vector3((float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.linear.z);
        Vector3 angular = new Vector3((float)cmd.angular.x, (float)cmd.angular.y, (float)cmd.angular.z);

        // Apply scaling for Unity coordinate system
        targetLinearVelocity = linear * linearSpeed;
        targetAngularVelocity = angular * angularSpeed;
    }

    void FixedUpdate()
    {
        ApplyControlCommands();
    }

    void ApplyControlCommands()
    {
        // Apply linear velocity
        if (rb != null)
        {
            // For ground-based robots, only apply X and Z (forward/backward, left/right)
            Vector3 currentVelocity = rb.velocity;
            rb.velocity = new Vector3(targetLinearVelocity.x, currentVelocity.y, targetLinearVelocity.z);

            // Apply angular velocity (rotation around Y axis for differential drive)
            float targetAngularZ = -targetAngularVelocity.y; // Unity uses different axis convention
            transform.Rotate(Vector3.up, targetAngularZ * Time.fixedDeltaTime, Space.World);
        }
        else
        {
            // If no rigidbody, use transform translation (less realistic)
            transform.Translate(targetLinearVelocity * Time.fixedDeltaTime, Space.World);
            transform.Rotate(Vector3.up, -targetAngularVelocity.y * Time.fixedDeltaTime, Space.World);
        }
    }
}
```

### Advanced Joint Control for Humanoid Robots

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;

public class UnityHumanoidController : MonoBehaviour
{
    [Header("Joint Configuration")]
    public List<JointController> jointControllers = new List<JointController>();

    private ROSConnection ros;

    [System.Serializable]
    public class JointController
    {
        public string jointName;
        public Transform jointTransform;
        public float minAngle = -90f;
        public float maxAngle = 90f;
        public float speed = 90f; // degrees per second
        [HideInInspector] public float targetAngle;
    }

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to joint states
        ros.Subscribe<JointStateMsg>("joint_states", JointStateCallback);
    }

    void JointStateCallback(JointStateMsg jointState)
    {
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float position = (float)jointState.position[i];

            // Find corresponding joint controller
            JointController controller = jointControllers.Find(jc => jc.jointName == jointName);
            if (controller != null)
            {
                controller.targetAngle = Mathf.Rad2Deg * position; // Convert from radians
            }
        }
    }

    void Update()
    {
        MoveJoints();
    }

    void MoveJoints()
    {
        foreach (JointController controller in jointControllers)
        {
            if (controller.jointTransform != null)
            {
                // Clamp target angle
                float clampedTarget = Mathf.Clamp(controller.targetAngle,
                                                controller.minAngle,
                                                controller.maxAngle);

                // Smoothly move to target angle
                float currentAngle = controller.jointTransform.localEulerAngles.y;
                float newAngle = Mathf.MoveTowards(currentAngle, clampedTarget,
                                                 controller.speed * Time.deltaTime);

                // Apply rotation
                Vector3 eulerAngles = controller.jointTransform.localEulerAngles;
                eulerAngles.y = newAngle;
                controller.jointTransform.localEulerAngles = eulerAngles;
            }
        }
    }
}
```

## Advanced Communication Patterns

### Service-like Communication

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class UnityServiceServer : MonoBehaviour
{
    [Header("Service Configuration")]
    public string serviceName = "/unity_service";
    public string requestTopic = "/unity_service_request";
    public string responseTopic = "/unity_service_response";

    private ROSConnection ros;
    private bool isServiceActive = false;

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to service requests
        ros.Subscribe<std_msgs.StringMsg>(requestTopic, HandleServiceRequest);
    }

    void HandleServiceRequest(std_msgs.StringMsg request)
    {
        Debug.Log($"Received service request: {request.data}");

        // Process the request and generate response
        std_msgs.StringMsg response = ProcessRequest(request.data);

        // Publish response
        ros.Publish(responseTopic, response);

        Debug.Log($"Sent service response: {response.data}");
    }

    std_msgs.StringMsg ProcessRequest(string requestData)
    {
        // Implement your service logic here
        string result = $"Processed: {requestData} at Unity time {Time.time}";

        return new std_msgs.StringMsg
        {
            data = result
        };
    }

    // Example: Trigger service from Unity
    public void TriggerService(string request)
    {
        var requestMsg = new std_msgs.StringMsg { data = request };
        ros.Publish(requestTopic, requestMsg);
    }
}
```

### Action-like Communication

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Actionlib;

public class UnityActionServer : MonoBehaviour
{
    [Header("Action Configuration")]
    public string actionName = "/unity_action";
    public string goalTopic = "/unity_action/goal";
    public string feedbackTopic = "/unity_action/feedback";
    public string resultTopic = "/unity_action/result";
    public string statusTopic = "/unity_action/status";

    private ROSConnection ros;
    private bool isGoalActive = false;
    private string currentGoal = "";
    private float goalStartTime = 0f;

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to action goals
        ros.Subscribe<GoalIDMsg>(goalTopic, HandleGoal);
    }

    void HandleGoal(GoalIDMsg goal)
    {
        Debug.Log($"Received action goal: {goal.id}");

        // Cancel current goal if any
        if (isGoalActive)
        {
            CancelCurrentGoal();
        }

        // Start new goal
        currentGoal = goal.id;
        isGoalActive = true;
        goalStartTime = Time.time;

        // Start goal execution
        StartCoroutine(ExecuteGoal());
    }

    System.Collections.IEnumerator ExecuteGoal()
    {
        float goalDuration = 5.0f; // Example: 5 second goal

        while (isGoalActive && (Time.time - goalStartTime) < goalDuration)
        {
            // Send feedback
            var feedback = new TestActionFeedbackMsg
            {
                status = new GoalStatusMsg
                {
                    goal_id = new GoalIDMsg { id = currentGoal },
                    status = GoalStatusMsg.ACTIVE
                },
                feedback = new TestFeedbackMsg
                {
                    percent_complete = (Time.time - goalStartTime) / goalDuration
                }
            };

            ros.Publish(feedbackTopic, feedback);

            yield return new WaitForSeconds(0.1f); // Feedback every 100ms
        }

        if (isGoalActive)
        {
            // Complete goal
            CompleteGoal();
        }
    }

    void CompleteGoal()
    {
        // Send result
        var result = new TestActionResultMsg
        {
            status = new GoalStatusMsg
            {
                goal_id = new GoalIDMsg { id = currentGoal },
                status = GoalStatusMsg.SUCCEEDED
            },
            result = new TestResultMsg { success = true }
        };

        ros.Publish(resultTopic, result);

        isGoalActive = false;
        currentGoal = "";
    }

    void CancelCurrentGoal()
    {
        var result = new TestActionResultMsg
        {
            status = new GoalStatusMsg
            {
                goal_id = new GoalIDMsg { id = currentGoal },
                status = GoalStatusMsg.PREEMPTED
            },
            result = new TestResultMsg { success = false }
        };

        ros.Publish(resultTopic, result);

        isGoalActive = false;
        currentGoal = "";
    }
}
```

## Performance Optimization

### Message Batching

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using System.Collections.Generic;

public class MessageBatcher : MonoBehaviour
{
    [Header("Batching Configuration")]
    public float batchInterval = 0.1f; // 10Hz batching

    private ROSConnection ros;
    private Dictionary<string, List<object>> batchedMessages = new Dictionary<string, List<object>>();
    private float lastBatchTime;

    void Start()
    {
        ros = ROSConnection.instance;
        lastBatchTime = Time.time;
    }

    void Update()
    {
        if (Time.time - lastBatchTime >= batchInterval)
        {
            PublishBatchedMessages();
            lastBatchTime = Time.time;
        }
    }

    public void QueueMessage(string topic, object message)
    {
        if (!batchedMessages.ContainsKey(topic))
        {
            batchedMessages[topic] = new List<object>();
        }

        batchedMessages[topic].Add(message);
    }

    void PublishBatchedMessages()
    {
        foreach (var kvp in batchedMessages)
        {
            foreach (var message in kvp.Value)
            {
                ros.Publish(kvp.Key, message);
            }
        }

        // Clear batched messages
        foreach (var kvp in batchedMessages)
        {
            kvp.Value.Clear();
        }
    }
}
```

## Best Practices and Troubleshooting

### Common Issues and Solutions

**Issue: High Latency in Communication**
- Check network connection quality
- Reduce message frequency for non-critical data
- Use message compression for large data

**Issue: Message Loss**
- Increase ROS TCP Endpoint buffer sizes
- Check for network congestion
- Implement message acknowledgment if critical

**Issue: Performance Degradation**
- Limit the number of simultaneous publishers/subscribers
- Use object pooling for frequently sent messages
- Optimize Unity scene complexity

### Security Considerations

When deploying Unity-ROS systems:

1. **Network Security**: Use secure connections and authentication
2. **Message Validation**: Validate incoming ROS messages
3. **Resource Limits**: Implement rate limiting and resource management

## Summary

Unity-ROS integration provides powerful capabilities for robotics simulation and visualization. By implementing proper communication patterns, sensor simulation, and robot control systems, you can create sophisticated simulation environments that bridge the gap between virtual and real robotics development. The key to successful integration lies in understanding both Unity's real-time capabilities and ROS 2's communication patterns, then designing systems that leverage the strengths of both platforms.