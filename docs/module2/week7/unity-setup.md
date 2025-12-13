---
sidebar_position: 1
---

# Week 7: Unity Setup and Configuration

## Overview

This section introduces Unity as a simulation environment for humanoid robotics. Unity provides advanced 3D rendering capabilities, realistic physics simulation, and powerful environment building tools that complement traditional robotics simulators. This week focuses on setting up Unity for robotics applications and establishing the connection with ROS 2.

## Learning Objectives

By the end of this section, you will be able to:

- Install and configure Unity for robotics applications
- Set up the Unity-ROS bridge for communication
- Create basic 3D environments for robot simulation
- Configure Unity physics for realistic robot interaction
- Understand the advantages and limitations of Unity vs. traditional simulators

## Unity Installation and Setup

### System Requirements

Before installing Unity, ensure your system meets the following requirements:

**Minimum Requirements:**
- **OS**: Windows 10 (64-bit), macOS 10.14+, or Ubuntu 20.04/22.04 LTS
- **CPU**: SSE2 instruction set support
- **GPU**: DX10 (shader model 4.0) capabilities
- **RAM**: 4GB+ (8GB+ recommended)
- **Disk Space**: 20GB+ for installation

**Recommended Requirements:**
- **OS**: Windows 10/11 (64-bit) or Ubuntu 22.04 LTS
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **GPU**: Dedicated graphics card with 4GB+ VRAM
- **RAM**: 16GB+ for complex scenes
- **Disk Space**: 50GB+ for projects

### Installing Unity Hub and Unity Editor

1. **Download Unity Hub**:
   - Visit [unity.com](https://unity.com/download)
   - Download Unity Hub (required for managing Unity installations)

2. **Install Unity Hub**:
   - Run the installer and follow the setup wizard
   - Create or sign in to your Unity ID

3. **Install Unity Editor**:
   - Open Unity Hub
   - Go to the "Installs" tab
   - Click "Add" to install a Unity version
   - Select **Unity 2022.3 LTS** (Long Term Support) - recommended for robotics
   - Select the following modules:
     - Windows Build Support (IL2CPP) or Linux Build Support
     - Visual Studio Editor (for Windows) or Visual Studio Code Editor
     - Android Build Support (optional, for mobile deployment)

4. **Create a New Project**:
   - In Unity Hub, click "New Project"
   - Select the "3D (Built-in Render Pipeline)" template
   - Name your project (e.g., "RoboticsSimulation")
   - Choose a location and click "Create Project"

## Unity-ROS Bridge Setup

### Overview of Unity-ROS Integration

The Unity-ROS bridge enables communication between Unity and ROS 2, allowing you to:
- Send and receive ROS messages from Unity
- Control Unity objects with ROS nodes
- Simulate sensors and publish data to ROS topics
- Integrate Unity visualization with ROS tools

### Installing ROS TCP Endpoint

The ROS TCP Endpoint is a Unity package that provides the bridge functionality:

1. **Download the Package**:
   - Visit the [Unity Robotics GitHub repository](https://github.com/Unity-Technologies/Unity-Robotics-Hub)
   - Download the latest release of the ROS TCP Endpoint package

2. **Import into Unity**:
   - In Unity, go to `Window > Package Manager`
   - Click the "+" button and select "Add package from disk..."
   - Navigate to the downloaded `.unitypackage` file
   - Import the package

3. **Alternative: Git Integration**:
   - In Package Manager, click the "+" button
   - Select "Add package from git URL"
   - Enter: `https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git`

### Setting up the ROS TCP Endpoint

1. **Add the TCP Endpoint to Your Scene**:
   - In the Unity hierarchy, right-click and select `Create Empty`
   - Name it "ROSTCPConnection"
   - Add the "ROS Connection" component by going to `Add Component > ROS TCP Endpoint > ROS Connection`

2. **Configure Connection Settings**:
   - Select the "ROSTCPConnection" object
   - In the Inspector, configure:
     - **ROS IP Address**: "127.0.0.1" (localhost) or the IP of your ROS machine
     - **ROS Port**: 10000 (default)
     - **ROS WebSocket Port**: 9090 (for web interface, if needed)

3. **Test the Connection**:
   - Build and run your Unity scene
   - In a terminal, test the connection:
   ```bash
   telnet 127.0.0.1 10000
   ```

### Installing ROS 2 Unity Packages

For enhanced functionality, install additional Unity packages:

1. **URDF Importer**:
   - Enables importing URDF files directly into Unity
   - Import from the Unity Robotics Hub

2. **Robotics Examples**:
   - Sample scenes and scripts for robotics applications
   - Helpful for learning Unity-ROS integration patterns

## Unity Interface and Navigation

### Main Interface Components

1. **Scene View**: 3D view of your scene where you can position objects
2. **Game View**: Preview of how the scene will look during gameplay
3. **Hierarchy**: List of all objects in the current scene
4. **Inspector**: Properties and components of the selected object
5. **Project**: Assets, scripts, and resources for your project
6. **Console**: Error messages, warnings, and logs

### Navigation Controls

**Scene View Navigation**:
- **Orbit**: Alt + Left Mouse Button drag
- **Pan**: Alt + Right Mouse Button drag or Middle Mouse Button
- **Zoom**: Mouse Wheel or Alt + Right Mouse Button drag vertically
- **Focus**: Select an object and press F

**Gizmo Tools**:
- **Move Tool** (W): Move objects in 3D space
- **Rotate Tool** (E): Rotate objects
- **Scale Tool** (R): Scale objects
- **Transform Tool** (Q): Select tool

## Creating Basic Robotics Environments

### Setting up the Scene

1. **Remove Default Objects**:
   - Delete the default "Main Camera" and "Directional Light"
   - These will be replaced with robotics-specific objects

2. **Add Ground Plane**:
   - Right-click in Hierarchy → 3D Object → Plane
   - Position at (0, 0, 0)
   - Scale as needed (e.g., scale 10, 1, 10 for a large ground)

3. **Configure Physics Materials**:
   - Create a new Physic Material in Project window
   - Right-click → Create → Physic Material
   - Configure friction and bounce properties for realistic interaction

### Adding Basic Objects

```csharp
// Example C# script for creating objects programmatically
using UnityEngine;

public class RobotEnvironment : MonoBehaviour
{
    public GameObject robotPrefab;
    public Transform spawnPoint;

    void Start()
    {
        // Spawn robot at specified location
        if (robotPrefab != null && spawnPoint != null)
        {
            Instantiate(robotPrefab, spawnPoint.position, spawnPoint.rotation);
        }

        // Create some obstacles
        CreateObstacle(new Vector3(2, 0.5, 2), new Vector3(1, 1, 1));
        CreateObstacle(new Vector3(-2, 0.5, -2), new Vector3(1, 1, 1));
    }

    GameObject CreateObstacle(Vector3 position, Vector3 size)
    {
        GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Cube);
        obstacle.transform.position = position;
        obstacle.transform.localScale = size;

        // Add realistic physics properties
        Rigidbody rb = obstacle.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.mass = 1.0f;
        }

        return obstacle;
    }
}
```

### Lighting Setup

For realistic robotics simulation:

1. **Add Directional Light**:
   - Represents the sun
   - Adjust rotation for desired lighting direction
   - Set shadows to "Soft Shadows" for realism

2. **Configure Lighting Settings**:
   - Go to `Window > Rendering > Lighting Settings`
   - Set environment lighting
   - Configure real-time global illumination if needed

## Physics Configuration for Robotics

### Unity Physics System

Unity uses NVIDIA PhysX for physics simulation. For robotics applications:

1. **Physics Settings**:
   - Go to `Edit > Project Settings > Physics`
   - Adjust default values for robotics simulation:
     - **Gravity**: (0, -9.81, 0) - matches real world
     - **Default Material**: Set appropriate friction values
     - **Solver Iteration Count**: Increase for stability (e.g., 10-20)

2. **Rigidbody Configuration**:
   - Add Rigidbody component to all objects that should be physically simulated
   - Configure mass based on real robot specifications
   - Set appropriate constraints for joints

### Collision Detection

For accurate robot-environment interaction:

1. **Collider Components**:
   - Add appropriate colliders to all objects
   - Use primitive colliders when possible for performance
   - Use mesh colliders for complex shapes (use sparingly)

2. **Layer-Based Collision**:
   - Use Unity's layer system to control which objects interact
   - Configure collision matrix in Physics settings

## Unity-ROS Communication Patterns

### Publishing Data from Unity

```csharp
using UnityEngine;
using RosMessageTypes.Sensor;
using Unity.Robotics.ROSTCPConnector;

public class UnitySensorPublisher : MonoBehaviour
{
    ROSConnection ros;
    string topicName = "unity_sensor_data";

    void Start()
    {
        ros = ROSConnection.instance;
    }

    void Update()
    {
        // Publish sensor data periodically
        if (Time.time % 0.1f < Time.deltaTime) // Publish at 10Hz
        {
            var sensorMsg = new UnitySensorMsg
            {
                header = new std_msgs.Header()
                {
                    stamp = new builtin_interfaces.Time()
                    {
                        sec = (int)Time.time,
                        nanosec = (uint)((Time.time % 1) * 1000000000)
                    }
                },
                position = new float[] { transform.position.x, transform.position.y, transform.position.z },
                rotation = new float[] { transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w }
            };

            ros.Publish(topicName, sensorMsg);
        }
    }
}
```

### Subscribing to ROS Topics

```csharp
using UnityEngine;
using RosMessageTypes.Geometry;
using Unity.Robotics.ROSTCPConnector;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to ROS topic
        ros.Subscribe<TwistMsg>("cmd_vel", CmdVelCallback);
    }

    void CmdVelCallback(TwistMsg cmd)
    {
        // Apply movement commands to robot
        Vector3 linear = new Vector3((float)cmd.linear.x, (float)cmd.linear.y, (float)cmd.linear.z);
        Vector3 angular = new Vector3((float)cmd.angular.x, (float)cmd.angular.y, (float)cmd.angular.z);

        // Apply the movement (implementation depends on your robot setup)
        transform.Translate(linear * Time.deltaTime);
        transform.Rotate(angular * Time.deltaTime);
    }
}
```

## Performance Optimization

### Unity Performance Settings

For real-time robotics simulation:

1. **Quality Settings**:
   - Go to `Edit > Project Settings > Quality`
   - Balance visual quality with performance requirements
   - Consider using "Fastest" or "Simple" for real-time applications

2. **Script Optimization**:
   - Minimize use of `Update()` method calls
   - Use object pooling for frequently created/destroyed objects
   - Optimize physics calculations

3. **Rendering Optimization**:
   - Use appropriate LOD (Level of Detail) settings
   - Reduce draw calls by batching objects
   - Optimize textures and materials

## Troubleshooting Common Issues

### Connection Problems

**Issue: Cannot connect to ROS**
- Check that ROS IP address and port are correct
- Verify firewall settings allow connections
- Ensure ROS TCP Endpoint is running in Unity

**Issue: High latency in communication**
- Check network connection quality
- Reduce message frequency if possible
- Optimize message size

### Physics Issues

**Issue: Robot falls through ground**
- Verify colliders are properly configured
- Check that ground plane has appropriate physics properties
- Increase solver iteration count if needed

**Issue: Unstable physics simulation**
- Reduce time step in Unity Time settings
- Increase solver iterations
- Verify mass and inertia values

### Performance Issues

**Issue: Low frame rate**
- Reduce scene complexity
- Optimize lighting and shadows
- Use occlusion culling for large environments

## Best Practices for Robotics Simulation

### Scene Organization

1. **Use Prefabs**: Create prefabs for common robot models and environments
2. **Layer Management**: Use Unity layers for different object types (robots, obstacles, sensors)
3. **Naming Conventions**: Use clear, descriptive names for all objects

### Code Organization

1. **Modular Scripts**: Separate concerns into different scripts
2. **Configuration Files**: Use ScriptableObjects for parameter configuration
3. **Event Systems**: Use Unity's event system for communication between components

## Summary

Unity provides a powerful platform for robotics simulation with advanced graphics capabilities and flexible environment creation tools. Proper setup of the Unity-ROS bridge enables seamless integration between Unity's visualization capabilities and ROS 2's robotics framework. Understanding Unity's physics system and optimization techniques is crucial for creating realistic and performant robotics simulations.

In the next sections, we'll explore environment building in Unity and advanced integration with ROS 2.