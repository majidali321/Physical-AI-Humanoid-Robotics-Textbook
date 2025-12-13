---
sidebar_position: 2
---

# Robot Control Systems for Humanoid Robots

## Introduction to Robot Control

Robot control is the process of commanding a robot's actuators to achieve desired motions or behaviors. For humanoid robots, control systems must manage complex multi-degree-of-freedom systems while maintaining balance, executing tasks, and ensuring safety.

### Control System Hierarchy

Robot control systems typically follow a hierarchical structure:

```
High-Level Planner
    ↓ (Trajectories/Tasks)
Mid-Level Controller
    ↓ (Joint Commands)
Low-Level Actuator Control
```

Each level operates at different frequencies and with different objectives:

- **High-level**: Path planning, task planning, decision making (1-10 Hz)
- **Mid-level**: Trajectory following, balance control, feedback control (100-1000 Hz)
- **Low-level**: Direct motor control, current control (1-10 kHz)

## Types of Control Systems

### 1. Position Control
Position control commands specific joint angles. It's the most common type for humanoid robots:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np

class PositionController(Node):
    def __init__(self):
        super().__init__('position_controller')

        # Publishers and subscribers
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Robot configuration
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']  # Example joints
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}

        # PID controller parameters
        self.kp = 10.0  # Proportional gain
        self.ki = 0.5   # Integral gain
        self.kd = 0.1   # Derivative gain

        # PID state variables
        self.errors = {name: 0.0 for name in self.joint_names}
        self.integral_errors = {name: 0.0 for name in self.joint_names}
        self.previous_errors = {name: 0.0 for name in self.joint_names}

        self.get_logger().info('Position controller initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]

    def set_desired_positions(self, positions_dict):
        """Set desired positions for joints"""
        for joint_name, desired_pos in positions_dict.items():
            if joint_name in self.desired_positions:
                self.desired_positions[joint_name] = desired_pos

    def control_loop(self):
        """Main PID control loop"""
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.position = []

        for joint_name in self.joint_names:
            # Calculate error
            error = self.desired_positions[joint_name] - self.current_positions[joint_name]

            # Update integral and derivative terms
            self.integral_errors[joint_name] += error * 0.01  # dt = 0.01s
            derivative = (error - self.previous_errors[joint_name]) / 0.01

            # Calculate PID output
            output = (self.kp * error +
                     self.ki * self.integral_errors[joint_name] +
                     self.kd * derivative)

            # Apply limits
            output = max(min(output, 2.0), -2.0)  # Limit to ±2 rad/s

            # Update state for next iteration
            self.previous_errors[joint_name] = error
            commands.position.append(output)

        self.joint_command_publisher.publish(commands)

def main(args=None):
    rclpy.init(args=args)
    node = PositionController()

    # Example: Move to a specific configuration
    node.set_desired_positions({
        'joint1': 0.5,
        'joint2': -0.3,
        'joint3': 0.8,
        'joint4': -0.2
    })

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

### 2. Velocity Control
Velocity control commands specific joint velocities, useful for smooth motion:

```python
class VelocityController(Node):
    def __init__(self):
        super().__init__('velocity_controller')

        self.joint_command_publisher = self.create_publisher(JointState, 'joint_velocity_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.desired_velocities = {name: 0.0 for name in self.joint_names}

        self.get_logger().info('Velocity controller initialized')

    def control_loop(self):
        """Publish velocity commands"""
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.velocity = [self.desired_velocities[name] for name in self.joint_names]

        self.joint_command_publisher.publish(commands)
```

### 3. Torque Control
Torque control directly commands joint torques, providing precise force control:

```python
class TorqueController(Node):
    def __init__(self):
        super().__init__('torque_controller')

        self.joint_command_publisher = self.create_publisher(JointState, 'joint_effort_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.control_timer = self.create_timer(0.001, self.control_loop)  # 1kHz for torque control

        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}
        self.desired_torques = {name: 0.0 for name in self.joint_names}

        self.get_logger().info('Torque controller initialized')

    def control_loop(self):
        """Publish torque commands"""
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.effort = [self.desired_torques[name] for name in self.joint_names]

        self.joint_command_publisher.publish(commands)
```

## Advanced Control Techniques

### Impedance Control
Impedance control makes the robot behave like a spring-damper system, useful for compliant manipulation:

```python
class ImpedanceController(Node):
    def __init__(self):
        super().__init__('impedance_controller')

        self.joint_command_publisher = self.create_publisher(JointState, 'joint_impedance_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.control_timer = self.create_timer(0.001, self.control_loop)

        # Impedance parameters (stiffness and damping)
        self.stiffness = {name: 100.0 for name in ['joint1', 'joint2', 'joint3']}
        self.damping = {name: 10.0 for name in ['joint1', 'joint2', 'joint3']}

        # Desired equilibrium positions
        self.equilibrium_positions = {name: 0.0 for name in ['joint1', 'joint2', 'joint3']}

        self.current_positions = {name: 0.0 for name in ['joint1', 'joint2', 'joint3']}
        self.current_velocities = {name: 0.0 for name in ['joint1', 'joint2', 'joint3']}

        self.get_logger().info('Impedance controller initialized')

    def set_equilibrium(self, positions_dict):
        """Set equilibrium positions for impedance control"""
        for joint_name, pos in positions_dict.items():
            if joint_name in self.equilibrium_positions:
                self.equilibrium_positions[joint_name] = pos

    def control_loop(self):
        """Calculate impedance control torques"""
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = list(self.equilibrium_positions.keys())
        commands.effort = []

        for joint_name in self.equilibrium_positions.keys():
            # Calculate position and velocity errors
            pos_error = self.equilibrium_positions[joint_name] - self.current_positions[joint_name]
            vel_error = -self.current_velocities[joint_name]  # Relative to equilibrium velocity (0)

            # Calculate impedance force: F = K*(x_desired - x_current) + D*(v_desired - v_current)
            torque = (self.stiffness[joint_name] * pos_error +
                     self.damping[joint_name] * vel_error)

            commands.effort.append(torque)

        self.joint_command_publisher.publish(commands)
```

### Operational Space Control
Operational space control allows controlling end-effector position and orientation directly:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class OperationalSpaceController(Node):
    def __init__(self):
        super().__init__('operational_space_controller')

        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Robot configuration (example 3-DOF arm)
        self.joint_names = ['shoulder_joint', 'elbow_joint', 'wrist_joint']
        self.current_positions = {name: 0.0 for name in self.joint_names}

        # Desired end-effector pose
        self.desired_position = np.array([0.5, 0.0, 0.5])
        self.desired_orientation = R.from_euler('xyz', [0, 0, 0]).as_matrix()

        self.get_logger().info('Operational space controller initialized')

    def forward_kinematics(self, joint_angles):
        """Calculate end-effector position from joint angles (simplified)"""
        # This is a simplified example - real implementation would use DH parameters or URDF
        l1, l2, l3 = 0.3, 0.3, 0.2  # Link lengths
        q1, q2, q3 = joint_angles

        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2) + l3 * np.cos(q1 + q2 + q3)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2) + l3 * np.sin(q1 + q2 + q3)
        z = 0.5  # Fixed height for this example

        return np.array([x, y, z])

    def jacobian(self, joint_angles):
        """Calculate geometric Jacobian (simplified)"""
        # This is a simplified Jacobian - real implementation would be more complex
        l1, l2, l3 = 0.3, 0.3, 0.2
        q1, q2, q3 = joint_angles

        # Jacobian matrix (3x3 for position only)
        J = np.zeros((3, 3))

        # dx/dq
        J[0, 0] = -l1*np.sin(q1) - l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3)
        J[0, 1] = -l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3)
        J[0, 2] = -l3*np.sin(q1+q2+q3)

        # dy/dq
        J[1, 0] = l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
        J[1, 1] = l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3)
        J[1, 2] = l3*np.cos(q1+q2+q3)

        # dz/dq (all zeros in this simplified model)
        J[2, :] = 0

        return J

    def control_loop(self):
        """Operational space control"""
        # Get current joint angles
        q = np.array([self.current_positions[name] for name in self.joint_names])

        # Calculate current end-effector position
        current_pos = self.forward_kinematics(q)

        # Calculate position error
        pos_error = self.desired_position - current_pos

        # Calculate Jacobian
        J = self.jacobian(q)

        # Calculate joint velocities using Jacobian transpose
        # dq = J^T * dx (for position control)
        dq = J.T @ pos_error * 0.1  # 0.1 is a gain factor

        # Create joint command
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.velocity = dq.tolist()

        self.joint_command_publisher.publish(commands)
```

## Balance Control for Humanoid Robots

### Center of Mass (CoM) Control
Maintaining balance in humanoid robots requires careful CoM management:

```python
class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        self.joint_command_publisher = self.create_publisher(JointState, 'balance_joint_commands', 10)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.control_timer = self.create_timer(0.01, self.balance_control_loop)

        # Balance control parameters
        self.com_gain = 5.0
        self.angle_gain = 10.0
        self.velocity_gain = 1.0

        # Robot state
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        self.current_positions = {}
        self.desired_positions = {}

        self.get_logger().info('Balance controller initialized')

    def imu_callback(self, msg):
        """Update orientation from IMU"""
        # Convert quaternion to roll, pitch, yaw
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z

        # Roll (x-axis rotation)
        self.roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))

        # Pitch (y-axis rotation)
        self.pitch = np.arcsin(2*(w*y - z*x))

        # Yaw (z-axis rotation)
        self.yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        # Angular velocity
        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    def balance_control_loop(self):
        """Main balance control loop"""
        # Calculate balance corrections based on IMU data
        roll_correction = -self.angle_gain * self.roll - self.velocity_gain * self.angular_velocity[0]
        pitch_correction = -self.angle_gain * self.pitch - self.velocity_gain * self.angular_velocity[1]

        # Apply corrections to hip joints to maintain balance
        balance_adjustments = {
            'left_hip_roll_joint': roll_correction,
            'right_hip_roll_joint': -roll_correction,
            'left_hip_pitch_joint': pitch_correction,
            'right_hip_pitch_joint': pitch_correction
        }

        # Create balance command
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = list(balance_adjustments.keys())
        commands.position = list(balance_adjustments.values())

        self.joint_command_publisher.publish(commands)
```

### Zero Moment Point (ZMP) Control
ZMP control is crucial for dynamic balance in walking humanoid robots:

```python
class ZMPController(Node):
    def __init__(self):
        super().__init__('zmp_controller')

        self.joint_command_publisher = self.create_publisher(JointState, 'zmp_joint_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.control_timer = self.create_timer(0.01, self.zmp_control_loop)

        # ZMP control parameters
        self.zmp_reference = np.array([0.0, 0.0])  # Desired ZMP position
        self.zmp_current = np.array([0.0, 0.0])    # Current ZMP estimate
        self.com_height = 0.8  # Center of mass height

        # Walking parameters
        self.step_length = 0.3
        self.step_width = 0.2
        self.step_height = 0.05
        self.step_phase = 0.0

        self.get_logger().info('ZMP controller initialized')

    def estimate_zmp(self):
        """Estimate Zero Moment Point from robot state"""
        # Simplified ZMP estimation
        # In practice, this would use force/torque sensors in the feet
        com_pos = self.get_com_position()

        # ZMP = CoM projected to ground with dynamic compensation
        zmp_x = com_pos[0] - self.com_height / 9.81 * self.com_acceleration[0]
        zmp_y = com_pos[1] - self.com_height / 9.81 * self.com_acceleration[1]

        return np.array([zmp_x, zmp_y])

    def get_com_position(self):
        """Get estimated center of mass position (simplified)"""
        # This would use forward kinematics and mass distribution
        # For now, return a simplified estimate
        return np.array([0.0, 0.0])

    def zmp_control_loop(self):
        """ZMP-based balance control"""
        # Estimate current ZMP
        self.zmp_current = self.estimate_zmp()

        # Calculate ZMP error
        zmp_error = self.zmp_reference - self.zmp_current

        # Generate corrective joint commands
        # This is a simplified example - real implementation would be more complex
        corrective_torques = {
            'left_hip_roll_joint': zmp_error[1] * 50.0,  # Y-direction ZMP error -> roll correction
            'right_hip_roll_joint': -zmp_error[1] * 50.0,
            'left_hip_pitch_joint': zmp_error[0] * 30.0,  # X-direction ZMP error -> pitch correction
            'right_hip_pitch_joint': zmp_error[0] * 30.0
        }

        # Create command message
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = list(corrective_torques.keys())
        commands.effort = list(corrective_torques.values())

        self.joint_command_publisher.publish(commands)
```

## Trajectory Generation and Execution

### Joint Space Trajectory Generation
Generating smooth trajectories in joint space:

```python
class TrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('trajectory_generator')

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, 'joint_trajectory', 10)

        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.current_positions = {name: 0.0 for name in self.joint_names}

        self.get_logger().info('Trajectory generator initialized')

    def generate_trap_trajectory(self, start_pos, end_pos, duration, dt=0.01):
        """Generate trapezoidal velocity trajectory"""
        # Calculate trajectory parameters
        total_time = duration
        accel_time = total_time * 0.2  # 20% acceleration, 60% constant, 20% deceleration
        decel_time = accel_time
        const_time = total_time - 2 * accel_time

        # Calculate max velocity and acceleration
        total_move = end_pos - start_pos
        max_vel = 1.5 * total_move / (total_time - accel_time/3 - decel_time/3)
        max_accel = max_vel / accel_time

        # Generate trajectory points
        trajectory = []
        t = 0.0

        while t <= total_time:
            if t < accel_time:
                # Acceleration phase
                pos = start_pos + 0.5 * max_accel * t**2
                vel = max_accel * t
            elif t < accel_time + const_time:
                # Constant velocity phase
                pos = start_pos + 0.5 * max_accel * accel_time**2 + max_vel * (t - accel_time)
                vel = max_vel
            else:
                # Deceleration phase
                phase_t = t - (accel_time + const_time)
                pos = end_pos - 0.5 * max_accel * (decel_time - phase_t)**2
                vel = max_accel * (decel_time - phase_t)

            trajectory.append((t, pos, vel))
            t += dt

        return trajectory

    def create_joint_trajectory_msg(self, trajectory_points, joint_names):
        """Create JointTrajectory message from trajectory points"""
        msg = JointTrajectory()
        msg.joint_names = joint_names

        for t, positions, velocities in trajectory_points:
            point = JointTrajectoryPoint()
            point.positions = positions if isinstance(positions, list) else [positions]
            point.velocities = velocities if isinstance(velocities, list) else [velocities]
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            msg.points.append(point)

        return msg

    def move_to_pose(self, target_positions, duration=2.0):
        """Move all joints to target positions"""
        # Generate trajectory for each joint
        all_trajectories = []

        for i, joint_name in enumerate(self.joint_names):
            start_pos = self.current_positions[joint_name]
            end_pos = target_positions[i] if i < len(target_positions) else start_pos

            joint_traj = self.generate_trap_trajectory(start_pos, end_pos, duration)
            all_trajectories.append(joint_traj)

        # Combine into single trajectory message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        # Synchronize all joint trajectories
        num_points = len(all_trajectories[0])
        for i in range(num_points):
            point = JointTrajectoryPoint()
            point.positions = [all_trajectories[j][i][1] for j in range(len(self.joint_names))]
            point.velocities = [all_trajectories[j][i][2] for j in range(len(self.joint_names))]

            # Use time from first joint trajectory
            t = all_trajectories[0][i][0]
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)

            msg.points.append(point)

        self.trajectory_publisher.publish(msg)
```

### Cartesian Space Trajectory Generation
Generating trajectories in Cartesian space and converting to joint space:

```python
class CartesianTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('cartesian_trajectory_generator')

        self.trajectory_publisher = self.create_publisher(
            JointTrajectory, 'cartesian_joint_trajectory', 10)

        # Robot parameters (example 3-DOF arm)
        self.link_lengths = [0.3, 0.3, 0.2]
        self.joint_names = ['shoulder_joint', 'elbow_joint', 'wrist_joint']

        self.get_logger().info('Cartesian trajectory generator initialized')

    def inverse_kinematics(self, target_pos):
        """Calculate joint angles for target Cartesian position (2D planar example)"""
        x, y, _ = target_pos  # Ignore z for planar arm

        # Calculate distance from base to target
        r = np.sqrt(x**2 + y**2)

        if r > sum(self.link_lengths):
            # Target is out of reach, return closest position
            scale = sum(self.link_lengths) / r
            x *= scale
            y *= scale
            r = sum(self.link_lengths)

        # Two-link planar manipulator inverse kinematics
        l1, l2 = self.link_lengths[0], self.link_lengths[1]

        # Calculate elbow angle using law of cosines
        cos_theta2 = (l1**2 + l2**2 - r**2) / (2 * l1 * l2)
        cos_theta2 = np.clip(cos_theta2, -1, 1)  # Clamp to valid range
        theta2 = np.arccos(cos_theta2)

        # Calculate shoulder angle
        k1 = l1 + l2 * np.cos(theta2)
        k2 = l2 * np.sin(theta2)
        theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

        # Third joint for orientation (simplified)
        theta3 = 0.0  # Maintain fixed orientation

        return np.array([theta1, theta2, theta3])

    def generate_cartesian_trajectory(self, start_pos, end_pos, duration, dt=0.01):
        """Generate Cartesian trajectory and convert to joint space"""
        steps = int(duration / dt)
        trajectory = []

        for i in range(steps + 1):
            t = i / steps  # Interpolation parameter [0, 1]

            # Linear interpolation in Cartesian space
            current_pos = start_pos + t * (end_pos - start_pos)

            # Convert to joint space
            joint_angles = self.inverse_kinematics(current_pos)

            trajectory.append((t * duration, joint_angles))

        return trajectory

    def execute_cartesian_move(self, start_pos, end_pos, duration=2.0):
        """Execute movement from start to end position in Cartesian space"""
        trajectory = self.generate_cartesian_trajectory(start_pos, end_pos, duration)

        # Create trajectory message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        for t, joint_angles in trajectory:
            point = JointTrajectoryPoint()
            point.positions = joint_angles.tolist()
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t - int(t)) * 1e9)
            msg.points.append(point)

        self.trajectory_publisher.publish(msg)
```

## Safety and Emergency Systems

### Joint Limit Protection
```python
class JointLimitProtector(Node):
    def __init__(self):
        super().__init__('joint_limit_protector')

        self.command_subscription = self.create_subscription(
            JointState, 'joint_commands_raw', self.command_callback, 10)
        self.command_publisher = self.create_publisher(
            JointState, 'joint_commands', 10)
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Define joint limits (example values)
        self.joint_limits = {
            'shoulder_joint': (-1.57, 1.57),
            'elbow_joint': (-2.0, 0.5),
            'wrist_joint': (-1.0, 1.0)
        }

        self.current_positions = {}

        self.get_logger().info('Joint limit protector initialized')

    def command_callback(self, msg):
        """Process and limit joint commands"""
        limited_msg = JointState()
        limited_msg.header = msg.header
        limited_msg.name = msg.name
        limited_msg.position = []

        for i, name in enumerate(msg.name):
            if name in self.joint_limits:
                # Apply position limits
                pos = max(self.joint_limits[name][0],
                         min(self.joint_limits[name][1], msg.position[i]))
                limited_msg.position.append(pos)
            else:
                limited_msg.position.append(msg.position[i])

        # Also check for velocity and effort limits
        if msg.velocity:
            limited_msg.velocity = []
            for i, name in enumerate(msg.name):
                if name in self.current_positions:
                    # Calculate approximate velocity
                    vel = (limited_msg.position[i] - self.current_positions[name]) / 0.01  # dt = 0.01s
                    vel = max(-5.0, min(5.0, vel))  # Limit to ±5 rad/s
                    limited_msg.velocity.append(vel)
                else:
                    limited_msg.velocity.append(msg.velocity[i])

        self.command_publisher.publish(limited_msg)

    def joint_state_callback(self, msg):
        """Update current positions"""
        for i, name in enumerate(msg.name):
            self.current_positions[name] = msg.position[i]
```

### Emergency Stop System
```python
class EmergencyStopSystem(Node):
    def __init__(self):
        super().__init__('emergency_stop_system')

        self.emergency_stop_subscription = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10)
        self.fault_publisher = self.create_publisher(String, 'fault_status', 10)

        self.emergency_stop_active = False
        self.fault_status = "OK"

        self.get_logger().info('Emergency stop system initialized')

    def emergency_stop_callback(self, msg):
        """Handle emergency stop commands"""
        if msg.data:
            self.activate_emergency_stop()
        else:
            self.deactivate_emergency_stop()

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop_active = True
        self.fault_status = "EMERGENCY_STOP_ACTIVATED"

        # Publish fault status
        fault_msg = String()
        fault_msg.data = self.fault_status
        self.fault_publisher.publish(fault_msg)

        self.get_logger().error('EMERGENCY STOP ACTIVATED')

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop_active = False
        self.fault_status = "OK"

        # Publish fault status
        fault_msg = String()
        fault_msg.data = self.fault_status
        self.fault_publisher.publish(fault_msg)

        self.get_logger().info('Emergency stop deactivated')
```

## Control System Integration

### Complete Control Node
```python
class CompleteHumanoidController(Node):
    def __init__(self):
        super().__init__('complete_humanoid_controller')

        # Publishers
        self.joint_command_publisher = self.create_publisher(JointState, 'joint_commands', 10)
        self.trajectory_publisher = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)

        # Subscribers
        self.joint_state_subscription = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.command_subscription = self.create_subscription(
            String, 'robot_command', self.command_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(0.01, self.control_loop)

        # Robot state
        self.joint_names = [
            'neck_joint',
            'left_shoulder_yaw_joint', 'left_shoulder_pitch_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_yaw_joint', 'right_shoulder_pitch_joint', 'right_elbow_joint', 'right_wrist_joint',
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        self.current_positions = {name: 0.0 for name in self.joint_names}
        self.current_velocities = {name: 0.0 for name in self.joint_names}
        self.desired_positions = {name: 0.0 for name in self.joint_names}

        # Control modes
        self.control_mode = "IDLE"  # IDLE, POSITION, WALK, MANIPULATE
        self.active_behavior = "STAND"  # STAND, WALK, REACH, etc.

        # IMU data
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.get_logger().info('Complete humanoid controller initialized')

    def joint_state_callback(self, msg):
        """Update robot state from joint feedback"""
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_velocities[name] = msg.velocity[i]

    def imu_callback(self, msg):
        """Update orientation from IMU"""
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
        self.roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        self.pitch = np.arcsin(2*(w*y - z*x))
        self.yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def command_callback(self, msg):
        """Handle high-level commands"""
        command = msg.data.lower()

        if command == "stand":
            self.active_behavior = "STAND"
            self.control_mode = "POSITION"
        elif command == "walk_forward":
            self.active_behavior = "WALK"
            self.control_mode = "TRAJECTORY"
        elif command == "wave":
            self.active_behavior = "WAVE"
            self.control_mode = "TRAJECTORY"
        elif command == "idle":
            self.active_behavior = "STAND"
            self.control_mode = "IDLE"

    def control_loop(self):
        """Main control loop with behavior switching"""
        if self.control_mode == "IDLE":
            self.execute_idle_behavior()
        elif self.control_mode == "POSITION":
            if self.active_behavior == "STAND":
                self.execute_stand_behavior()
            elif self.active_behavior == "WAVE":
                self.execute_wave_behavior()
        elif self.control_mode == "TRAJECTORY":
            if self.active_behavior == "WALK":
                self.execute_walk_behavior()

    def execute_idle_behavior(self):
        """Maintain current position"""
        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.position = [self.current_positions[name] for name in self.joint_names]
        self.joint_command_publisher.publish(commands)

    def execute_stand_behavior(self):
        """Maintain standing position with balance control"""
        # Default standing position
        stand_config = {
            'left_hip_pitch_joint': 0.0,
            'right_hip_pitch_joint': 0.0,
            'left_knee_joint': 0.0,
            'right_knee_joint': 0.0,
            'left_ankle_joint': 0.0,
            'right_ankle_joint': 0.0,
            # Arms in neutral position
            'left_shoulder_pitch_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'left_elbow_joint': -0.5,
            'right_elbow_joint': -0.5,
        }

        # Apply balance corrections based on IMU
        balance_correction = {
            'left_ankle_joint': -self.roll * 0.5,
            'right_ankle_joint': -self.roll * 0.5,
            'left_hip_roll_joint': self.pitch * 0.2,
            'right_hip_roll_joint': self.pitch * 0.2,
        }

        # Combine stand configuration with balance corrections
        final_positions = self.current_positions.copy()
        for joint, pos in stand_config.items():
            final_positions[joint] = pos
        for joint, correction in balance_correction.items():
            if joint in final_positions:
                final_positions[joint] += correction

        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.position = [final_positions[name] for name in self.joint_names]
        self.joint_command_publisher.publish(commands)

    def execute_wave_behavior(self):
        """Wave behavior for right arm"""
        time_in_behavior = (self.get_clock().now().nanoseconds / 1e9) % 4.0  # 4 second cycle

        # Wave motion for right arm
        wave_amplitude = 0.5
        wave_freq = 2.0  # Hz

        right_shoulder_pos = wave_amplitude * np.sin(2 * np.pi * wave_freq * time_in_behavior)
        right_elbow_pos = -0.5 + 0.3 * np.sin(2 * np.pi * wave_freq * time_in_behavior + np.pi/2)

        # Update desired positions
        self.desired_positions['right_shoulder_pitch_joint'] = right_shoulder_pos
        self.desired_positions['right_elbow_joint'] = right_elbow_pos

        commands = JointState()
        commands.header.stamp = self.get_clock().now().to_msg()
        commands.name = self.joint_names
        commands.position = [self.desired_positions[name] for name in self.joint_names]
        self.joint_command_publisher.publish(commands)

def main(args=None):
    rclpy.init(args=args)
    node = CompleteHumanoidController()

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

## Best Practices for Robot Control

### 1. Control Frequency Considerations
- Use appropriate frequencies for each control level
- Ensure real-time performance for safety-critical controls
- Consider computational load when setting frequencies

### 2. Safety First
- Implement multiple safety layers
- Use joint limits and velocity limits
- Include emergency stop functionality
- Monitor for hardware failures

### 3. Smooth Transitions
- Use trajectory generation for smooth motion
- Implement proper interpolation between waypoints
- Avoid discontinuities in control signals

### 4. Feedback and Monitoring
- Use appropriate sensors for feedback
- Monitor control performance
- Implement logging for debugging
- Include health monitoring systems

## Summary

Robot control for humanoid systems requires sophisticated multi-layered approaches that handle position control, balance, safety, and coordination of multiple degrees of freedom. The examples in this section demonstrate various control techniques from basic PID control to advanced operational space control, along with safety systems and integration approaches that are essential for humanoid robot operation.

The key to successful humanoid robot control lies in combining these techniques appropriately, ensuring safety at all levels, and maintaining stable, predictable behavior across all operating conditions.