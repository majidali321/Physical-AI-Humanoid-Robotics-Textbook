---
sidebar_position: 3
---

# URDF Basics for Humanoid Robots

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML-based format used in ROS to describe robot models. It defines the physical and kinematic properties of a robot, including links, joints, visual appearance, collision properties, and inertial characteristics.

For humanoid robots, URDF is essential for:
- Simulation in Gazebo
- Visualization in RViz
- Kinematic analysis
- Motion planning
- Control system development

## URDF Structure and Components

### Basic URDF Document

```xml
<?xml version="1.0"?>
<robot name="my_humanoid_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>
</robot>
```

### Components Breakdown

#### 1. Links
Links represent rigid bodies in the robot. Each link contains:
- **Visual**: How the link appears in visualization
- **Collision**: How the link interacts in collision detection
- **Inertial**: Physical properties for simulation

#### 2. Joints
Joints connect links and define their relative motion:
- **parent**: The link that the joint connects from
- **child**: The link that the joint connects to
- **type**: The type of joint (fixed, revolute, continuous, prismatic, etc.)
- **origin**: Position and orientation of the joint relative to the parent

## Humanoid Robot URDF Structure

### Basic Humanoid Topology

A humanoid robot typically has this kinematic structure:
```
base_link (torso)
├── head
├── left_arm
│   ├── left_upper_arm
│   ├── left_lower_arm
│   └── left_hand
├── right_arm
│   ├── right_upper_arm
│   ├── right_lower_arm
│   └── right_hand
├── left_leg
│   ├── left_upper_leg
│   ├── left_lower_leg
│   └── left_foot
└── right_leg
    ├── right_upper_leg
    ├── right_lower_leg
    └── right_foot
```

### Complete Humanoid URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/Torso Link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0.15 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="arm_color">
        <color rgba="0.5 0.5 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="arm_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Right Arm (similar structure) -->
  <joint name="right_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_arm"/>
    <origin xyz="0.2 -0.15 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="arm_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="arm_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_leg"/>
    <origin xyz="-0.1 0.1 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="1"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
      <material name="leg_color">
        <color rgba="0.3 0.3 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1.57" effort="200" velocity="1"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.35"/>
      </geometry>
      <material name="leg_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.35"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="150" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="foot_color">
        <color rgba="0.2 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Right Leg (similar structure) -->
  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_upper_leg"/>
    <origin xyz="-0.1 -0.1 -0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="1"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
      <material name="leg_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1.57" effort="200" velocity="1"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.35"/>
      </geometry>
      <material name="leg_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.35"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.35" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="150" velocity="1"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="foot_color"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>
</robot>
```

## URDF Joint Types for Humanoid Robots

### 1. Fixed Joints
Used for permanent connections (e.g., sensor mounts):
```xml
<joint name="sensor_mount" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>
```

### 2. Revolute Joints
Rotational joints with limits (e.g., elbows, knees):
```xml
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="lower_arm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-2.0" upper="1.5" effort="50" velocity="2"/>
</joint>
```

### 3. Continuous Joints
Rotational joints without limits (e.g., shoulders):
```xml
<joint name="shoulder_joint" type="continuous">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0.2 0.15 0.6" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
</joint>
```

### 4. Prismatic Joints
Linear sliding joints (less common in humanoid robots):
```xml
<joint name="linear_joint" type="prismatic">
  <parent link="base"/>
  <child link="slider"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="0.5"/>
</joint>
```

## Visual and Collision Properties

### Visual Elements
```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.1"/>
    <!-- Other geometry types: <sphere radius="0.1"/>, <cylinder radius="0.05" length="0.2"/> -->
  </geometry>
  <material name="red">
    <color rgba="1 0 0 1"/>
    <!-- Or reference a material defined elsewhere -->
  </material>
</visual>
```

### Collision Elements
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.1"/>
  </geometry>
</collision>
```

### Inertial Properties
```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

## Using Xacro for Complex Humanoid URDFs

Xacro allows for parameterization, macros, and includes to simplify complex URDFs:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="torso_width" value="0.3"/>
  <xacro:property name="torso_depth" value="0.3"/>
  <xacro:property name="torso_height" value="1.0"/>

  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>

  <!-- Macro for creating symmetric limbs -->
  <xacro:macro name="arm" params="side parent xyz_rpy">
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${xyz_rpy}"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.3"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 -0.3"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_lower_arm">
      <visual>
        <geometry>
          <cylinder radius="0.04" length="0.25"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.04" length="0.25"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="${torso_width} ${torso_depth} ${torso_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Use the arm macro for both arms -->
  <xacro:arm side="left" parent="base_link" xyz_rpy="0.2 0.15 0.6"/>
  <xacro:arm side="right" parent="base_link" xyz_rpy="0.2 -0.15 0.6"/>

</robot>
```

## URDF Validation and Tools

### Checking URDF Syntax
```bash
# Check if URDF is valid
check_urdf my_humanoid.urdf

# Generate graph of robot structure
urdf_to_graphiz my_humanoid.urdf
```

### Loading URDF in ROS 2
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class URDFLoader(Node):
    def __init__(self):
        super().__init__('urdf_loader')

        # Publisher for robot description
        self.urdf_publisher = self.create_publisher(
            String, 'robot_description', 1)

        # Load URDF from file
        self.load_urdf_from_file()

        # Publish URDF
        self.publish_urdf()

    def load_urdf_from_file(self):
        """Load URDF from file"""
        try:
            with open('/path/to/my_humanoid.urdf', 'r') as file:
                self.robot_description = file.read()
        except FileNotFoundError:
            self.get_logger().error('URDF file not found')
            self.robot_description = ""

    def publish_urdf(self):
        """Publish the robot description"""
        msg = String()
        msg.data = self.robot_description
        self.urdf_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = URDFLoader()

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

## Integration with ROS 2 Parameters

### Parameterized URDF Loading
```python
class ParameterizedURDFNode(Node):
    def __init__(self):
        super().__init__('parameterized_urdf')

        # Declare parameters for robot dimensions
        self.declare_parameter('torso_width', 0.3)
        self.declare_parameter('torso_height', 1.0)
        self.declare_parameter('arm_length', 0.5)

        # Get parameter values
        self.torso_width = self.get_parameter('torso_width').value
        self.torso_height = self.get_parameter('torso_height').value
        self.arm_length = self.get_parameter('arm_length').value

        # Publisher for robot description
        self.urdf_publisher = self.create_publisher(
            String, 'robot_description', 1)

        # Create parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Generate and publish initial URDF
        self.generate_and_publish_urdf()

    def parameter_callback(self, params):
        """Update URDF when parameters change"""
        from rcl_interfaces.msg import SetParametersResult

        result = SetParametersResult()
        result.successful = True

        for param in params:
            if param.name == 'torso_width':
                self.torso_width = param.value
            elif param.name == 'torso_height':
                self.torso_height = param.value
            elif param.name == 'arm_length':
                self.arm_length = param.value

        # Regenerate URDF with new parameters
        self.generate_and_publish_urdf()

        return result

    def generate_and_publish_urdf(self):
        """Generate URDF with current parameters and publish"""
        urdf_string = f"""<?xml version="1.0"?>
        <robot name="parameterized_humanoid">
          <link name="base_link">
            <visual>
              <geometry>
                <box size="{self.torso_width} 0.3 {self.torso_height}"/>
              </geometry>
              <material name="white">
                <color rgba="1 1 1 1"/>
              </material>
            </visual>
            <collision>
              <geometry>
                <box size="{self.torso_width} 0.3 {self.torso_height}"/>
              </geometry>
            </collision>
            <inertial>
              <mass value="10.0"/>
              <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="0.2"/>
            </inertial>
          </link>
        </robot>"""

        msg = String()
        msg.data = urdf_string
        self.urdf_publisher.publish(msg)

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

## Best Practices for Humanoid URDF

### 1. Naming Conventions
- Use consistent naming (e.g., `left_upper_arm`, `right_lower_leg`)
- Use descriptive names that indicate function and position
- Follow ROS conventions for frame names

### 2. Mass and Inertia Properties
- Use realistic values for simulation
- Consider the actual robot's weight distribution
- Use the parallel axis theorem when needed

### 3. Joint Limits
- Set appropriate limits based on actual hardware capabilities
- Consider safety margins
- Account for desired operational range

### 4. Visual vs Collision
- Use simplified geometries for collision to improve performance
- Use detailed geometries for visual representation
- Ensure collision geometry encompasses visual geometry

### 5. Organization
- Group related links and joints together
- Use comments to separate body parts
- Consider using Xacro for complex models

## Common Issues and Troubleshooting

### 1. Invalid URDF
- Check for proper XML syntax
- Ensure all joints have valid parent/child links
- Verify all referenced materials exist

### 2. Kinematic Issues
- Check joint types and limits
- Verify joint axes are correctly oriented
- Ensure no kinematic loops exist

### 3. Simulation Problems
- Verify inertial properties are reasonable
- Check that collision geometries are properly defined
- Ensure mass properties are realistic

## Summary

URDF is fundamental for humanoid robotics in ROS 2, providing the robot model needed for simulation, visualization, and control. Proper URDF definition requires attention to kinematic structure, physical properties, and visualization elements. The use of Xacro and parameters allows for flexible, configurable robot models that can adapt to different hardware configurations while maintaining proper kinematic relationships.

In the next section, we'll explore how to integrate URDF with ROS 2 for practical humanoid robot applications.