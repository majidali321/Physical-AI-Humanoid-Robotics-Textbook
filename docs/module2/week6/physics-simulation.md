---
sidebar_position: 2
---

# Week 6: Physics Simulation in Gazebo

## Overview

Physics simulation is the core of Gazebo's capability to provide realistic robot testing environments. This section covers the fundamentals of physics simulation in Gazebo, including how to configure physics properties, understand the different physics engines, and optimize simulations for humanoid robotics applications.

## Learning Objectives

By the end of this section, you will be able to:

- Configure physics parameters for optimal simulation performance
- Understand the differences between physics engines and their applications
- Implement realistic contact models for humanoid robots
- Optimize simulation parameters for accuracy and speed
- Debug physics-related issues in simulations

## Understanding Physics Simulation

### Core Concepts

Physics simulation in Gazebo involves solving equations of motion to predict how objects move and interact over time. The simulation calculates:

- **Position and Orientation**: How objects move through space
- **Velocity and Acceleration**: How fast objects move and how their speed changes
- **Forces and Torques**: How objects affect each other through contact and other interactions
- **Collisions**: When objects come into contact and how they respond

### Simulation Loop

The physics simulation runs in a continuous loop:

1. **Force Calculation**: Compute forces acting on all objects
2. **Integration**: Update positions and velocities based on forces
3. **Collision Detection**: Identify object intersections
4. **Collision Response**: Calculate resulting forces from collisions
5. **Constraint Resolution**: Apply joint constraints and other limitations

## Physics Engines in Gazebo

### Open Dynamics Engine (ODE)

**Pros:**
- Good balance of speed and accuracy
- Mature and well-tested
- Good for most humanoid robotics applications

**Cons:**
- Can be unstable with complex contact scenarios
- Limited support for soft body dynamics

**Best for:** General robotics simulation, humanoid walking, manipulation

### Bullet Physics

**Pros:**
- Fast and robust
- Good for complex contact scenarios
- Supports more collision shapes

**Cons:**
- Less accurate for some scenarios
- Can be less stable with complex articulated systems

**Best for:** Multi-robot scenarios, complex environments

### Simbody

**Pros:**
- High accuracy for articulated systems
- Excellent for complex joint constraints
- Good for biomechanical simulations

**Cons:**
- Slower than other engines
- More complex to configure

**Best for:** High-precision humanoid simulation, complex kinematic chains

### DART (Dynamic Animation and Robotics Toolkit)

**Pros:**
- Excellent for humanoid robots
- Advanced contact handling
- Good for bipedal locomotion

**Cons:**
- Less mature than ODE
- Can be resource-intensive

**Best for:** Humanoid robotics, bipedal locomotion

## Physics Configuration Parameters

### Time Step Configuration

```xml
<physics name="ode" type="ode">
  <max_step_size>0.001</max_step_size>  <!-- Physics time step (seconds) -->
  <real_time_factor>1.0</real_time_factor>  <!-- Simulation speed multiplier -->
  <real_time_update_rate>1000.0</real_time_update_rate>  <!-- Updates per second -->
</physics>
```

**Key Parameters:**
- **max_step_size**: Smaller values = more accurate but slower
- **real_time_factor**: 1.0 = real-time, >1.0 = faster than real-time
- **real_time_update_rate**: Usually 1/max_step_size

### Solver Configuration

```xml
<ode>
  <solver>
    <type>quick</type>  <!-- Solver type: quick, world -->
    <iters>10</iters>  <!-- Number of solver iterations -->
    <sor>1.3</sor>  <!-- Successive over-relaxation parameter -->
  </solver>
</ode>
```

**Solver Parameters:**
- **iters**: More iterations = more accurate but slower
- **sor**: Over-relaxation factor, typically 1.0-1.9

### Constraint Configuration

```xml
<constraints>
  <cfm>0.0</cfm>  <!-- Constraint force mixing -->
  <erp>0.2</erp>  <!-- Error reduction parameter -->
  <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
  <contact_surface_layer>0.001</contact_surface_layer>
</constraints>
```

**Constraint Parameters:**
- **cfm**: Helps with numerical stability
- **erp**: How quickly constraint errors are corrected
- **contact_max_correcting_vel**: Maximum velocity for contact correction
- **contact_surface_layer**: Penetration depth before contact forces apply

## Configuring for Humanoid Robots

### Mass and Inertia Properties

For realistic humanoid simulation, accurate mass and inertia properties are crucial:

```xml
<inertial>
  <mass>5.0</mass>  <!-- Mass in kg -->
  <inertia>
    <ixx>0.1</ixx>  <!-- Moments of inertia -->
    <ixy>0.0</ixy>
    <ixz>0.0</ixz>
    <iyy>0.1</iyy>
    <iyz>0.0</iyz>
    <izz>0.1</izz>
  </inertia>
</inertial>
```

### Center of Mass

The center of mass affects balance and stability:

```xml
<inertial>
  <mass>5.0</mass>
  <pose>0.0 0.0 0.1 0 0 0</pose>  <!-- Offset center of mass -->
  <inertia>...</inertia>
</inertial>
```

### Contact Properties

For humanoid feet and hands, configure contact properties carefully:

```xml
<collision name="foot_collision">
  <surface>
    <friction>
      <ode>
        <mu>1.0</mu>  <!-- Static friction coefficient -->
        <mu2>1.0</mu2>  <!-- Secondary friction direction -->
        <slip1>0.0</slip1>  <!-- Slip in primary direction -->
        <slip2>0.0</slip2>  <!-- Slip in secondary direction -->
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>  <!-- Bounciness -->
      <threshold>100000.0</threshold>  <!-- Velocity threshold for bouncing -->
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0.0</soft_cfm>  <!-- Soft constraint force mixing -->
        <soft_erp>0.2</soft_erp>  <!-- Soft error reduction parameter -->
        <kp>1000000.0</kp>  <!-- Contact stiffness -->
        <kd>100.0</kd>  <!-- Contact damping -->
        <max_vel>100.0</max_vel>  <!-- Maximum contact correction velocity -->
        <min_depth>0.001</min_depth>  <!-- Minimum contact depth -->
      </ode>
    </contact>
  </surface>
</collision>
```

## Advanced Physics Concepts

### Joint Dynamics

Configure joint properties for realistic movement:

```xml
<joint name="hip_joint" type="revolute">
  <parent>torso</parent>
  <child>thigh</child>
  <axis>
    <xyz>0 0 1</xyz>
    <limit>
      <lower>-1.57</lower>  <!-- Joint limits in radians -->
      <upper>1.57</upper>
      <effort>100.0</effort>  <!-- Maximum torque -->
      <velocity>3.0</velocity>  <!-- Maximum velocity -->
    </limit>
    <dynamics>
      <damping>0.1</damping>  <!-- Joint damping -->
      <friction>0.0</friction>  <!-- Joint friction -->
    </dynamics>
  </axis>
</joint>
```

### Actuator Models

For more realistic joint control, model actuators explicitly:

```xml
<plugin name="joint_position_controller" filename="libgazebo_ros_joint_position.so">
  <command_topic>joint_position/command</command_topic>
  <feedback_topic>joint_position/feedback</feedback_topic>
  <joint_name>hip_joint</joint_name>
  <pid>
    <p>100.0</p>  <!-- Proportional gain -->
    <i>0.1</i>   <!-- Integral gain -->
    <d>10.0</d>  <!-- Derivative gain -->
  </pid>
</plugin>
```

## Optimization Strategies

### Performance vs. Accuracy Trade-offs

| Parameter | Performance Impact | Accuracy Impact |
|-----------|-------------------|-----------------|
| Time Step | Smaller = slower | Smaller = better |
| Solver Iterations | More = slower | More = better |
| Contact Parameters | High values = slower | High values = more stable |

### Adaptive Configuration

For complex humanoid simulations, consider adaptive parameters:

```xml
<!-- Use different parameters for different simulation phases -->
<physics name="normal" type="ode">
  <!-- Normal operation parameters -->
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
</physics>

<!-- Or use multiple physics engines for different parts -->
```

## Debugging Physics Issues

### Common Problems and Solutions

**Problem: Robot falls through the ground**
- Check collision geometries in URDF/SDF
- Verify ground plane exists and is properly configured
- Adjust contact parameters (increase ERP, decrease CFM)

**Problem: Robot is unstable or "explodes"**
- Increase solver iterations
- Decrease time step
- Adjust joint limits and damping
- Check mass and inertia values

**Problem: Robot doesn't move as expected**
- Verify joint types and limits
- Check actuator configurations
- Validate controller commands

### Debugging Tools

**Gazebo GUI:**
- Enable contact visualization to see contact points
- Use wireframe mode to see collision geometries
- Monitor real-time factor and step times

**Command Line:**
```bash
# Monitor simulation statistics
gz stats

# Check model states
gz model -m robot_name -i
```

**ROS 2 Tools:**
```bash
# Monitor joint states
ros2 topic echo /joint_states

# Check TF transforms
ros2 run tf2_tools view_frames
```

## Physics Validation

### Validation Techniques

1. **Compare with Analytical Solutions**: For simple cases, compare simulation with known solutions
2. **Energy Conservation**: Monitor total energy in closed systems
3. **Stability Testing**: Verify that static objects remain stable
4. **Parameter Sensitivity**: Test how results change with parameter variations

### Validation Metrics

- **Position Accuracy**: How closely simulated positions match expected values
- **Velocity Accuracy**: How well velocities are maintained
- **Force Accuracy**: How well contact forces are computed
- **Timing Accuracy**: How well real-time performance is maintained

## Best Practices for Humanoid Physics

### Model Preparation

1. **Validate URDF**: Use `check_urdf` to verify model structure
2. **Realistic Masses**: Use actual robot masses and inertias when available
3. **Proper Joint Limits**: Set realistic joint limits based on hardware constraints
4. **Collision Geometries**: Use simplified but accurate collision models

### Simulation Configuration

1. **Start Simple**: Begin with basic physics parameters and refine
2. **Match Hardware**: Configure simulation to match real robot characteristics
3. **Iterative Tuning**: Adjust parameters based on simulation behavior
4. **Validation Loop**: Continuously validate against real-world data

## Summary

Physics simulation is fundamental to creating realistic humanoid robot simulations in Gazebo. Proper configuration of physics parameters, understanding the trade-offs between performance and accuracy, and validating simulation results are essential skills for effective robotics development. In the next sections, we'll explore sensor simulation and ROS 2 integration with Gazebo.