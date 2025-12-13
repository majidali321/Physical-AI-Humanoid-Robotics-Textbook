---
sidebar_position: 1
---

# Week 6: Gazebo Basics

## Overview

This section introduces Gazebo, a powerful physics-based simulation environment for robotics. Gazebo provides realistic simulation of robots in complex environments with accurate physics, sensors, and rendering. Understanding Gazebo is crucial for testing and validating humanoid robotics applications before deployment on real hardware.

## Learning Objectives

By the end of this section, you will be able to:

- Launch and navigate the Gazebo simulation environment
- Create and modify basic Gazebo worlds
- Spawn and control robots in simulation
- Configure physics properties and parameters
- Integrate Gazebo with ROS 2

## Introduction to Gazebo

Gazebo is a 3D dynamic simulator with accurate physics and rendering capabilities. It's widely used in robotics research and development for:

- **Robot Testing**: Validate algorithms without physical hardware
- **Sensor Simulation**: Test perception systems with realistic sensor data
- **Environment Simulation**: Create complex scenarios for robot navigation
- **Multi-robot Systems**: Simulate interactions between multiple robots

### Key Features

- **Physics Engine**: Supports ODE, Bullet, Simbody, and DART physics engines
- **Sensor Simulation**: Cameras, LiDAR, IMU, GPS, and force/torque sensors
- **Rendering**: High-quality 3D graphics with OpenGL
- **Plugins**: Extensible architecture with plugin support
- **ROS Integration**: Native ROS/ROS 2 support

## Installing and Launching Gazebo

### Installation

Gazebo Garden is the recommended version for this course:

```bash
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/gazebo-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo Garden
sudo apt update
sudo apt install gazebo
```

### Basic Launch

```bash
# Launch Gazebo with default empty world
gazebo

# Launch with a specific world
gazebo worlds/willow garage.world
```

## Gazebo Interface and Navigation

### Main Interface Components

1. **Menu Bar**: File, Edit, View, Plugins, and Window options
2. **Toolbar**: Common actions like play/pause simulation
3. **Scene**: 3D view of the simulation environment
4. **Layers**: Scene hierarchy and object management
5. **Properties**: Selected object properties and parameters
6. **Console**: Command output and error messages

### Camera Navigation

- **Orbit**: Right-click and drag to rotate around a point
- **Pan**: Shift + right-click and drag to move the camera
- **Zoom**: Scroll wheel or right-click + drag vertically
- **Focus**: Double-click on an object to center the view

## Creating Gazebo Worlds

### World File Structure

Gazebo worlds are defined in SDF (Simulation Description Format) files:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define physics engine -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Custom models go here -->
  </world>
</sdf>
```

### Basic World Example

Create a simple world file (`my_world.sdf`):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Physics -->
    <physics name="ode" type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- A simple box obstacle -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1667</iyy>
            <iyz>0</iyz>
            <izz>0.1667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Launching Custom Worlds

```bash
# Launch your custom world
gazebo my_world.sdf
```

## Working with Models in Gazebo

### Built-in Models

Gazebo comes with many built-in models:

- **Primitives**: Boxes, spheres, cylinders
- **Robots**: PR2, TurtleBot, Pioneer
- **Environment**: Ground plane, sun, objects
- **Furniture**: Tables, chairs, doors

### Adding Models

1. **Through GUI**: Use the "Insert" tab to browse and add models
2. **Through SDF**: Include models in world files using `<include>` tags
3. **Programmatically**: Use Gazebo services to spawn models

### Model Spawning via Command Line

```bash
# Spawn a model at specific pose
gz model -f /path/to/model.sdf -m my_model_name -x 1.0 -y 2.0 -z 0.0
```

## Physics Configuration

### Physics Engines

Gazebo supports multiple physics engines:

- **ODE**: Open Dynamics Engine (default, good balance of speed and accuracy)
- **Bullet**: Fast and robust for most applications
- **Simbody**: High accuracy for complex articulated systems
- **DART**: Good for humanoid robots with complex contact

### Physics Parameters

Key physics parameters to configure:

- **Max Step Size**: Time step for physics simulation (smaller = more accurate but slower)
- **Real Time Factor**: Simulation speed relative to real time (1.0 = real-time)
- **Real Time Update Rate**: Updates per second (1000 = 1ms step size)

### Example Physics Configuration

```xml
<physics name="my_physics" type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
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
```

## Gazebo Plugins

### Overview

Gazebo plugins extend functionality and enable integration with external systems like ROS 2.

### Common Plugin Types

- **World Plugins**: Affect the entire simulation world
- **Model Plugins**: Attach to specific models
- **Sensor Plugins**: Process sensor data
- **GUI Plugins**: Extend the user interface

### ROS 2 Integration Plugin

The ROS 2 bridge enables communication between Gazebo and ROS 2:

```xml
<plugin filename="gz-transport-ros-bridge-system" name="ros_gz_bridge">
  <ros>
    <namespace>/gazebo</namespace>
  </ros>
  <services>
    <service>/spawn_entity</service>
    <service>/delete_entity</service>
  </services>
  <topics>
    <topic>/model_states</topic>
    <topic>/clock</topic>
  </topics>
</plugin>
```

## Controlling Gazebo Programmatically

### Command Line Tools

Gazebo provides command-line tools for automation:

```bash
# List all models in simulation
gz model -l

# Get model state
gz model -m my_robot -i

# Set model pose
gz model -m my_robot -x 1.0 -y 2.0 -z 0.0

# Apply force to a model
gz model -m my_box -f "0 0 10" -o "0 0 0"
```

### Using Gazebo Services

From ROS 2 nodes, you can interact with Gazebo using services:

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')

        # Create clients for Gazebo services
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.pause_client = self.create_client(Empty, '/pause_physics')
        self.unpause_client = self.create_client(Empty, '/unpause_physics')

    def spawn_model(self, model_xml, model_name, pose):
        # Implementation for spawning model
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()
    rclpy.spin(controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Gazebo Simulation

### Performance Optimization

1. **Reduce Physics Complexity**: Simplify collision geometries when possible
2. **Optimize Update Rates**: Balance accuracy with performance
3. **Limit Model Count**: Too many models can slow simulation
4. **Use Appropriate Physics Engine**: Choose based on simulation needs

### Simulation Accuracy

1. **Validate Against Real Data**: Compare simulation results with real-world data
2. **Tune Physics Parameters**: Adjust for realistic behavior
3. **Model Real Sensors**: Use accurate sensor models with realistic noise
4. **Include Environmental Effects**: Lighting, weather, and terrain variations

### Debugging Tips

1. **Use Visualization Tools**: Enable contact visualization and wireframe mode
2. **Monitor Performance**: Check simulation step time and real-time factor
3. **Log Model States**: Monitor joint positions, velocities, and forces
4. **Validate URDF/SDF**: Ensure models are properly defined

## Summary

Gazebo provides a powerful platform for robotics simulation with realistic physics and rendering. Understanding how to create and configure simulation environments is essential for developing and testing humanoid robotics applications. In the next sections, we'll explore more advanced features and integrate Gazebo with ROS 2 for complete robot simulation.