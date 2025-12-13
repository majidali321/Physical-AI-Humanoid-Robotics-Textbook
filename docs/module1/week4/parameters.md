---
sidebar_position: 1
---

# Week 4: ROS 2 Advanced Topics - Parameters, Lifecycle Nodes, and URDF Basics

## Learning Objectives

By the end of this week, you will be able to:
- Use ROS 2 parameters for node configuration
- Implement and manage lifecycle nodes for complex systems
- Create basic URDF (Unified Robot Description Format) models for humanoid robots
- Understand the relationship between URDF and robot kinematics
- Apply parameters to control robot behavior dynamically

## ROS 2 Parameters

### Introduction to Parameters
Parameters in ROS 2 provide a way to configure nodes at runtime without recompiling code. They enable dynamic reconfiguration of robot systems and support different deployment scenarios.

### Parameter Declaration and Usage

#### Basic Parameter Declaration
```python
import rclpy
from rclpy.node import Node

class ParameterExampleNode(Node):
    def __init__(self):
        super().__init__('parameter_example')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('debug_mode', False)

        # Access parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.debug_mode = self.get_parameter('debug_mode').value

        self.get_logger().info(
            f'Initialized with parameters:\n'
            f'  Robot name: {self.robot_name}\n'
            f'  Max velocity: {self.max_velocity}\n'
            f'  Safety distance: {self.safety_distance}\n'
            f'  Debug mode: {self.debug_mode}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = ParameterExampleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Parameter Descriptors
You can add additional constraints and descriptions to parameters:

```python
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import IntegerRange, FloatingPointRange

class ParameterDescriptorNode(Node):
    def __init__(self):
        super().__init__('parameter_descriptor_example')

        # String parameter with description
        string_desc = ParameterDescriptor()
        string_desc.description = 'Name of the robot'
        string_desc.read_only = False
        self.declare_parameter('robot_name', 'default_robot', string_desc)

        # Integer parameter with range
        int_desc = ParameterDescriptor()
        int_desc.description = 'Number of joints'
        int_desc.integer_range = [IntegerRange(from_value=1, to_value=10, step=1)]
        self.declare_parameter('num_joints', 6, int_desc)

        # Double parameter with range
        double_desc = ParameterDescriptor()
        double_desc.description = 'Maximum speed in m/s'
        double_desc.floating_range = [FloatingPointRange(from_value=0.0, to_value=5.0, step=0.1)]
        self.declare_parameter('max_speed', 1.0, double_desc)

        # Boolean parameter
        bool_desc = ParameterDescriptor()
        bool_desc.description = 'Enable advanced features'
        self.declare_parameter('enable_advanced', False, bool_desc)
```

### Parameter Callbacks
Nodes can be notified when parameters change:

```python
from rcl_interfaces.msg import SetParametersResult

class ParameterCallbackNode(Node):
    def __init__(self):
        super().__init__('parameter_callback_example')

        # Declare parameters
        self.declare_parameter('speed_limit', 1.0)
        self.declare_parameter('safety_enabled', True)

        # Set parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Store current values
        self.speed_limit = self.get_parameter('speed_limit').value
        self.safety_enabled = self.get_parameter('safety_enabled').value

    def parameter_callback(self, params):
        """Callback for parameter changes"""
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'speed_limit':
                if param.value > 5.0:
                    result.successful = False
                    result.reason = 'Speed limit cannot exceed 5.0 m/s'
                    return result
                else:
                    self.speed_limit = param.value
                    self.get_logger().info(f'Speed limit updated to {self.speed_limit}')

            elif param.name == 'safety_enabled':
                self.safety_enabled = param.value
                status = 'enabled' if self.safety_enabled else 'disabled'
                self.get_logger().info(f'Safety system {status}')

        return result
```

### Working with Parameters at Runtime

#### Command Line Parameter Setting
```bash
# Set parameter when running node
ros2 run my_package my_node --ros-args -p robot_name:=my_robot -p max_velocity:=2.0

# Use parameter file
ros2 run my_package my_node --ros-args --params-file config.yaml
```

#### Parameter File Example (config.yaml)
```yaml
parameter_example:
  ros__parameters:
    robot_name: 'configured_robot'
    max_velocity: 2.5
    safety_distance: 0.8
    debug_mode: true
```

### Parameter Services
ROS 2 provides built-in services for parameter management:
- `ros2 param list`: List all parameters for a node
- `ros2 param get <node_name> <param_name>`: Get parameter value
- `ros2 param set <node_name> <param_name> <value>`: Set parameter value

## Lifecycle Nodes

### Understanding Lifecycle Nodes
Lifecycle nodes provide a structured way to manage the state of complex systems. They follow a well-defined state machine with clear transitions between states.

### Lifecycle Node States
- **Unconfigured**: Node created but not configured
- **Inactive**: Node configured but not active
- **Active**: Node running and processing callbacks
- **Finalized**: Node destroyed and cleaned up
- **Error**: Node in error state

### Implementing a Lifecycle Node
```python
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.qos import QoSProfile

class LifecycleExampleNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_example')
        self.get_logger().info('Lifecycle node created, current state: unconfigured')

    def on_configure(self, state):
        """Called when transitioning to configuring state"""
        self.get_logger().info(f'Configuring node, state: {state.label}')

        # Initialize publishers, subscribers, etc.
        self.publisher = self.create_publisher(
            String, 'lifecycle_topic', QoSProfile(depth=10))

        # Return success to continue transition
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Called when transitioning to activating state"""
        self.get_logger().info(f'Activating node, state: {state.label}')

        # Activate publishers and subscribers
        self.publisher.on_activate()

        # Create timer for periodic publishing
        self.timer = self.create_timer(1.0, self.timer_callback)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Called when transitioning to deactivating state"""
        self.get_logger().info(f'Deactivating node, state: {state.label}')

        # Deactivate publishers and subscribers
        self.publisher.on_deactivate()

        # Destroy timer
        self.timer.destroy()

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Called when transitioning to cleaning up state"""
        self.get_logger().info(f'Cleaning up node, state: {state.label}')

        # Clean up resources
        self.publisher.destroy()

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        """Called when transitioning to shutting down state"""
        self.get_logger().info(f'Shutting down node, state: {state.label}')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        """Called when transitioning to error state"""
        self.get_logger().info(f'Node in error state: {state.label}')
        return TransitionCallbackReturn.SUCCESS

    def timer_callback(self):
        """Timer callback for active state"""
        msg = String()
        msg.data = f'Hello from lifecycle node at {self.get_clock().now().nanoseconds}'
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    # Create lifecycle node
    node = LifecycleExampleNode()

    # Spin the node
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Managing Lifecycle Nodes
```bash
# List lifecycle nodes
ros2 lifecycle list my_lifecycle_node

# Get current state
ros2 lifecycle get my_lifecycle_node

# Trigger state transitions
ros2 lifecycle configure my_lifecycle_node
ros2 lifecycle activate my_lifecycle_node
ros2 lifecycle deactivate my_lifecycle_node
ros2 lifecycle cleanup my_lifecycle_node
ros2 lifecycle shutdown my_lifecycle_node
```

### Complex Lifecycle Node Example
```python
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RobotControllerLifecycle(LifecycleNode):
    def __init__(self):
        super().__init__('robot_controller_lifecycle')
        self.get_logger().info('Robot controller lifecycle node initialized')

        # Initialize state variables
        self.scan_subscription = None
        self.cmd_vel_publisher = None
        self.safety_distance = 0.5
        self.max_velocity = 1.0

    def on_configure(self, state):
        self.get_logger().info('Configuring robot controller')

        # Get parameters
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('max_velocity', 1.0)
        self.safety_distance = self.get_parameter('safety_distance').value
        self.max_velocity = self.get_parameter('max_velocity').value

        # Create publisher and subscription
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)

        self.get_logger().info('Robot controller configured successfully')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating robot controller')

        # Activate communications
        self.cmd_vel_publisher.on_activate()
        # Note: subscriptions are always active

        # Start safety timer
        self.safety_timer = self.create_timer(0.1, self.safety_check)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating robot controller')

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)

        # Deactivate publisher
        self.cmd_vel_publisher.on_deactivate()

        # Destroy timer
        self.safety_timer.destroy()

        return TransitionCallbackReturn.SUCCESS

    def scan_callback(self, msg):
        # Store latest scan for safety check
        self.latest_scan = msg

    def safety_check(self):
        if hasattr(self, 'latest_scan') and self.latest_scan:
            # Check for obstacles in front
            center_idx = len(self.latest_scan.ranges) // 2
            front_distance = self.latest_scan.ranges[center_idx]

            cmd = Twist()
            if front_distance > self.safety_distance:
                cmd.linear.x = self.max_velocity
            else:
                cmd.linear.x = 0.0  # Stop to avoid collision

            self.cmd_vel_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerLifecycle()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## URDF (Unified Robot Description Format) Basics

### Introduction to URDF
URDF (Unified Robot Description Format) is an XML format used to describe robots in ROS. It defines the physical and kinematic properties of a robot, including:
- Links (rigid parts)
- Joints (connections between links)
- Visual and collision properties
- Inertial properties

### Basic URDF Structure
```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Links define rigid parts of the robot -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0.0 0.0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
  </link>
</robot>
```

### URDF Components

#### Links
Links represent rigid bodies in the robot:
- **visual**: How the link appears in visualization
- **collision**: How the link interacts in collision detection
- **inertial**: Physical properties for simulation

#### Joints
Joints define connections between links:
- **revolute**: Rotational joint with limits
- **continuous**: Rotational joint without limits
- **prismatic**: Linear sliding joint with limits
- **fixed**: No movement between links
- **floating**: 6 DOF movement
- **planar**: Movement on a plane

### URDF for Humanoid Robots

#### Basic Humanoid Structure
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="fixed">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.35" rpy="0 0 0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
  </link>

  <!-- Left Arm -->
  <joint name="torso_to_left_shoulder" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0.0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
  </link>

  <!-- Additional joints and links would continue... -->
</robot>
```

### URDF with Xacro
Xacro is a macro language that extends URDF with features like variables, math, and includes:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_length" value="0.5" />
  <xacro:property name="base_height" value="0.2" />

  <!-- Macro for creating a wheel -->
  <xacro:macro name="wheel" params="prefix parent x_reflect y_reflect">
    <joint name="${prefix}_wheel_joint" type="continuous">
      <origin xyz="${x_reflect*0.1} ${y_reflect*0.13} 0" rpy="0 0 0" />
      <parent link="${parent}" />
      <child link="${prefix}_wheel_link" />
      <axis xyz="0 1 0" />
    </joint>

    <link name="${prefix}_wheel_link">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0" />
        <geometry>
          <cylinder radius="0.05" length="0.02" />
        </geometry>
      </visual>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_length} ${base_height}"/>
      </geometry>
    </visual>
  </link>

  <!-- Use the wheel macro -->
  <xacro:wheel prefix="front_left" parent="base_link" x_reflect="1" y_reflect="1" />
  <xacro:wheel prefix="front_right" parent="base_link" x_reflect="1" y_reflect="-1" />
  <xacro:wheel prefix="rear_left" parent="base_link" x_reflect="-1" y_reflect="1" />
  <xacro:wheel prefix="rear_right" parent="base_link" x_reflect="-1" y_reflect="-1" />
</robot>
```

### Loading and Using URDF

#### Publishing URDF to ROS
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math

class URDFPublisher(Node):
    def __init__(self):
        super().__init__('urdf_publisher')

        # Publish robot description
        self.description_publisher = self.create_publisher(
            String, 'robot_description', 1)

        # Load URDF from file or string
        self.load_urdf()

        # Timer to publish robot description
        self.timer = self.create_timer(5.0, self.publish_description)

    def load_urdf(self):
        # Load URDF from file or define as string
        self.robot_description = """<?xml version="1.0"?>
        <robot name="simple_robot">
          <link name="base_link">
            <visual>
              <geometry>
                <box size="0.5 0.5 0.2"/>
              </geometry>
            </visual>
          </link>
        </robot>"""

    def publish_description(self):
        msg = String()
        msg.data = self.robot_description
        self.description_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = URDFPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### URDF Tools and Visualization

#### Command Line Tools
```bash
# Check URDF syntax
check_urdf my_robot.urdf

# Show URDF structure
urdf_to_graphiz my_robot.urdf

# Visualize URDF
ros2 run rviz2 rviz2
# Then add RobotModel display and set robot description topic
```

## Integration: Parameters with URDF

### Parameterized URDF Loading
```python
class ParameterizedURDFNode(Node):
    def __init__(self):
        super().__init__('parameterized_urdf')

        # Declare parameters for robot dimensions
        self.declare_parameter('robot_width', 0.3)
        self.declare_parameter('robot_length', 0.5)
        self.declare_parameter('robot_height', 0.2)

        # Get parameter values
        self.width = self.get_parameter('robot_width').value
        self.length = self.get_parameter('robot_length').value
        self.height = self.get_parameter('robot_height').value

        # Create parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Publish parameterized robot description
        self.description_publisher = self.create_publisher(
            String, 'robot_description', 1)

        # Timer to update robot description when parameters change
        self.publish_timer = self.create_timer(1.0, self.publish_robot_description)

    def parameter_callback(self, params):
        """Update robot dimensions when parameters change"""
        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'robot_width':
                self.width = param.value
            elif param.name == 'robot_length':
                self.length = param.value
            elif param.name == 'robot_height':
                self.height = param.value

        self.get_logger().info(
            f'Robot dimensions updated: {self.width}x{self.length}x{self.height}')

        # Update URDF with new dimensions
        self.update_urdf()

        return result

    def update_urdf(self):
        """Generate URDF with current dimensions"""
        self.robot_description = f"""<?xml version="1.0"?>
        <robot name="parameterized_robot">
          <link name="base_link">
            <visual>
              <geometry>
                <box size="{self.width} {self.length} {self.height}"/>
              </geometry>
            </visual>
            <collision>
              <geometry>
                <box size="{self.width} {self.length} {self.height}"/>
              </geometry>
            </collision>
          </link>
        </robot>"""

    def publish_robot_description(self):
        msg = String()
        msg.data = self.robot_description
        self.description_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParameterizedURDFNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

### Parameter Best Practices
- Use meaningful parameter names with proper prefixes
- Provide sensible default values
- Validate parameter ranges in callbacks
- Document parameters clearly
- Group related parameters logically

### URDF Best Practices
- Use consistent naming conventions
- Include proper inertial properties for simulation
- Separate visual and collision geometries when needed
- Use xacro for complex robots to avoid repetition
- Test URDF with visualization tools

### Lifecycle Node Best Practices
- Implement proper cleanup in each transition
- Use lifecycle nodes for complex systems that need initialization
- Handle errors gracefully in callbacks
- Document the state machine clearly
- Test all state transitions

## Summary

Week 4 covered advanced ROS 2 topics including parameters for dynamic configuration, lifecycle nodes for complex system management, and URDF basics for robot description. These concepts are fundamental for building sophisticated robotic systems that can adapt to different configurations and environments. The integration of parameters with URDF demonstrates how these concepts work together in practical applications.

In the next section, we'll explore more advanced URDF concepts and how to use it with ROS 2 for humanoid robot applications.