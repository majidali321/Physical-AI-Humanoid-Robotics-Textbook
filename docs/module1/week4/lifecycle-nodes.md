---
sidebar_position: 2
---

# Lifecycle Nodes in Depth

## Understanding the Need for Lifecycle Nodes

Traditional ROS nodes have a simple lifecycle: they start, run until terminated, and then clean up. However, complex robotic systems often require more sophisticated state management:

- **Initialization**: Resources need to be acquired in a specific order
- **Configuration**: Parameters and settings need to be loaded
- **Activation**: Components need to be enabled when ready
- **Deactivation**: Safe shutdown procedures for components
- **Cleanup**: Proper resource release

Lifecycle nodes provide a structured approach to manage these requirements.

## The Lifecycle Node State Machine

### State Transitions
The lifecycle node follows a specific state machine with well-defined transitions:

```
[Unconfigured] ←→ [Inactive] ←→ [Active]
     ↑                  ↑           ↑
   [Finalized] ←---- [Error] ←----------
```

### Available Transitions
- `configure()` - Unconfigured → Inactive
- `cleanup()` - Inactive → Unconfigured
- `activate()` - Inactive → Active
- `deactivate()` - Active → Inactive
- `shutdown()` - Any state → Finalized
- `error()` - Any state → Error (internal, not user-triggered)

## Detailed Lifecycle Node Implementation

### Complete Lifecycle Node Example

```python
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import String, Bool
import threading
import time

class ComprehensiveLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('comprehensive_lifecycle_node')

        # State tracking
        self.current_state_label = 'unconfigured'

        # Components that will be managed through lifecycle
        self.scan_subscription = None
        self.imu_subscription = None
        self.cmd_vel_publisher = None
        self.status_publisher = None
        self.control_timer = None

        # Configuration parameters
        self.safety_distance = 0.5
        self.max_velocity = 1.0
        self.robot_name = 'default_robot'

        # Runtime state
        self.latest_scan = None
        self.latest_imu = None
        self.is_operational = False

    def on_configure(self, state):
        """Called when transitioning to configuring state"""
        self.get_logger().info(f'Configuring node from state: {state.label}')

        # Get parameters (these should be declared in constructor in real implementation)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('robot_name', 'default_robot')

        self.safety_distance = self.get_parameter('safety_distance').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.robot_name = self.get_parameter('robot_name').value

        # Create publishers and subscriptions (but don't activate them yet)
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.status_publisher = self.create_publisher(String, 'robot_status', 10)
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, qos_profile)

        # Simulate complex initialization that might fail
        if not self.initialize_hardware():
            self.get_logger().error('Hardware initialization failed')
            return TransitionCallbackReturn.FAILURE

        # Simulate configuration validation
        if not self.validate_configuration():
            self.get_logger().error('Configuration validation failed')
            return TransitionCallbackReturn.FAILURE

        self.current_state_label = 'inactive'
        self.get_logger().info('Node configured successfully')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Called when transitioning to cleaning up state"""
        self.get_logger().info(f'Cleaning up node from state: {state.label}')

        # Reset configuration
        self.safety_distance = 0.5
        self.max_velocity = 1.0
        self.robot_name = 'default_robot'

        # Reset runtime state
        self.latest_scan = None
        self.latest_imu = None
        self.is_operational = False

        # Destroy publishers/subscriptions
        if self.cmd_vel_publisher:
            self.destroy_publisher(self.cmd_vel_publisher)
            self.cmd_vel_publisher = None
        if self.status_publisher:
            self.destroy_publisher(self.status_publisher)
            self.status_publisher = None
        if self.scan_subscription:
            self.destroy_subscription(self.scan_subscription)
            self.scan_subscription = None
        if self.imu_subscription:
            self.destroy_subscription(self.imu_subscription)
            self.imu_subscription = None

        self.current_state_label = 'unconfigured'
        self.get_logger().info('Node cleaned up successfully')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Called when transitioning to activating state"""
        self.get_logger().info(f'Activating node from state: {state.label}')

        # Activate all communications
        if self.cmd_vel_publisher:
            self.cmd_vel_publisher.on_activate()
        if self.status_publisher:
            self.status_publisher.on_activate()
        # Subscriptions are always active, but we can enable processing

        # Start control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Publish activation status
        status_msg = String()
        status_msg.data = f'{self.robot_name} activated'
        self.status_publisher.publish(status_msg)

        self.is_operational = True
        self.current_state_label = 'active'
        self.get_logger().info('Node activated successfully')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Called when transitioning to deactivating state"""
        self.get_logger().info(f'Deactivating node from state: {state.label}')

        # Stop robot movement
        stop_cmd = Twist()
        if self.cmd_vel_publisher:
            self.cmd_vel_publisher.publish(stop_cmd)

        # Deactivate publishers
        if self.cmd_vel_publisher:
            self.cmd_vel_publisher.on_deactivate()
        if self.status_publisher:
            self.status_publisher.on_deactivate()

        # Destroy timer
        if self.control_timer:
            self.control_timer.destroy()
            self.control_timer = None

        # Update operational status
        self.is_operational = False
        self.current_state_label = 'inactive'

        self.get_logger().info('Node deactivated successfully')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        """Called when transitioning to shutting down state"""
        self.get_logger().info(f'Shutting down node from state: {state.label}')

        # Perform final cleanup
        self.cleanup_resources()

        self.current_state_label = 'finalized'
        self.get_logger().info('Node shutdown complete')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        """Called when transitioning to error state"""
        self.get_logger().info(f'Node entered error state from: {state.label}')

        # Try to recover or at least clean up
        self.handle_error_state()

        self.current_state_label = 'error'
        return TransitionCallbackReturn.SUCCESS

    def initialize_hardware(self):
        """Simulate hardware initialization"""
        # In real implementation, this would initialize actual hardware
        self.get_logger().info('Initializing hardware...')
        time.sleep(0.5)  # Simulate initialization time
        return True  # Return False to simulate failure

    def validate_configuration(self):
        """Validate current configuration"""
        if self.max_velocity <= 0 or self.max_velocity > 10.0:
            self.get_logger().error(f'Invalid max_velocity: {self.max_velocity}')
            return False
        if self.safety_distance <= 0:
            self.get_logger().error(f'Invalid safety_distance: {self.safety_distance}')
            return False
        return True

    def cleanup_resources(self):
        """Clean up all resources"""
        self.get_logger().info('Cleaning up all resources...')
        # Release any allocated resources
        pass

    def handle_error_state(self):
        """Handle error state - try to recover or at least stop safely"""
        self.get_logger().warn('Handling error state - stopping all operations')
        if self.cmd_vel_publisher:
            stop_cmd = Twist()
            self.cmd_vel_publisher.publish(stop_cmd)

    def scan_callback(self, msg):
        """Store latest scan data"""
        if self.is_operational:
            self.latest_scan = msg

    def imu_callback(self, msg):
        """Store latest IMU data"""
        if self.is_operational:
            self.latest_imu = msg

    def control_loop(self):
        """Main control loop when node is active"""
        if not self.is_operational:
            return

        # Safety check using scan data
        if self.latest_scan:
            self.perform_safety_check()

        # Publish periodic status
        if self.get_clock().now().nanoseconds % 1000000000 < 100000000:  # Every ~1 second
            self.publish_status()

    def perform_safety_check(self):
        """Perform safety checks based on sensor data"""
        # Check for obstacles in front of robot
        if len(self.latest_scan.ranges) > 0:
            center_idx = len(self.latest_scan.ranges) // 2
            front_distance = self.latest_scan.ranges[center_idx]

            if 0 < front_distance < self.safety_distance:
                # Stop robot if obstacle is too close
                stop_cmd = Twist()
                self.cmd_vel_publisher.publish(stop_cmd)
                self.get_logger().warn(f'Safety stop: obstacle at {front_distance:.2f}m')

    def publish_status(self):
        """Publish robot status"""
        status_msg = String()
        status_msg.data = f'{self.robot_name}: operational, state={self.current_state_label}'
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ComprehensiveLifecycleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Lifecycle Patterns

### Conditional Transitions

```python
class ConditionalLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('conditional_lifecycle')
        self.hardware_ready = False
        self.calibration_complete = False

    def on_configure(self, state):
        """Only proceed if hardware is ready"""
        if not self.check_hardware_ready():
            self.get_logger().error('Hardware not ready, cannot configure')
            return TransitionCallbackReturn.FAILURE

        # Perform configuration
        self.hardware_ready = True
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Only activate if calibrated"""
        if not self.calibration_complete:
            self.get_logger().error('Not calibrated, cannot activate')
            return TransitionCallbackReturn.FAILURE

        return TransitionCallbackReturn.SUCCESS

    def check_hardware_ready(self):
        """Check if hardware is ready for operation"""
        # In real implementation, check actual hardware status
        return True  # Simulated
```

### State-Dependent Behavior

```python
class StateDependentNode(LifecycleNode):
    def __init__(self):
        super().__init__('state_dependent')
        self.state_behavior_map = {
            'unconfigured': self.behavior_unconfigured,
            'inactive': self.behavior_inactive,
            'active': self.behavior_active
        }

    def on_configure(self, state):
        self.current_behavior = self.state_behavior_map['inactive']
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.current_behavior = self.state_behavior_map['active']
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.current_behavior = self.state_behavior_map['inactive']
        return TransitionCallbackReturn.SUCCESS

    def timer_callback(self):
        """Execute behavior based on current state"""
        if self.current_behavior:
            self.current_behavior()

    def behavior_unconfigured(self):
        self.get_logger().debug('In unconfigured state - minimal activity')

    def behavior_inactive(self):
        self.get_logger().debug('In inactive state - monitoring only')

    def behavior_active(self):
        self.get_logger().debug('In active state - full operation')
        # Perform full robot operations
```

## Integration with ROS 2 Ecosystem

### Using Lifecycle Manager

```python
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition

class LifecycleManagerClient(Node):
    def __init__(self):
        super().__init__('lifecycle_manager_client')

        # Create clients for lifecycle management
        self.change_state_client = self.create_client(
            ChangeState, '/my_lifecycle_node/change_state')
        self.get_state_client = self.create_client(
            GetState, '/my_lifecycle_node/get_state')

    def configure_node(self, node_name):
        """Configure a lifecycle node"""
        request = ChangeState.Request()
        request.transition.id = Transition.TRANSITION_CONFIGURE
        request.transition.label = 'configure'

        future = self.change_state_client.call_async(request)
        return future

    def activate_node(self, node_name):
        """Activate a lifecycle node"""
        request = ChangeState.Request()
        request.transition.id = Transition.TRANSITION_ACTIVATE
        request.transition.label = 'activate'

        future = self.change_state_client.call_async(request)
        return future
```

### Lifecycle Launch Files

```python
# launch/lifecycle_example.launch.py
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.actions import EmitEvent
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.events import ChangeState

def generate_launch_description():
    # Create lifecycle node
    lifecycle_node = LifecycleNode(
        package='my_package',
        executable='comprehensive_lifecycle_node',
        name='my_lifecycle_node',
        namespace='',
        parameters=[
            {'safety_distance': 0.5},
            {'max_velocity': 1.0}
        ]
    )

    # Automatically configure and activate the node
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher='my_lifecycle_node',
            transition_id=Transition.TRANSITION_CONFIGURE
        )
    )

    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher='my_lifecycle_node',
            transition_id=Transition.TRANSITION_ACTIVATE
        )
    )

    return LaunchDescription([
        lifecycle_node,
        # Configure after 2 seconds
        configure_event,
        # Activate after 4 seconds
        activate_event,
    ])
```

## URDF Integration with Lifecycle Nodes

### Parameterized URDF Loading in Lifecycle Node

```python
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from std_msgs.msg import String

class URDFLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('urdf_lifecycle_node')

        # URDF-related components
        self.urdf_publisher = None
        self.robot_description = ""

        # Configuration parameters
        self.robot_dimensions = {
            'width': 0.3,
            'length': 0.5,
            'height': 0.2
        }

    def on_configure(self, state):
        """Configure URDF publishing"""
        self.get_logger().info('Configuring URDF publisher')

        # Declare and get URDF-related parameters
        self.declare_parameter('robot_width', 0.3)
        self.declare_parameter('robot_length', 0.5)
        self.declare_parameter('robot_height', 0.2)

        self.robot_dimensions['width'] = self.get_parameter('robot_width').value
        self.robot_dimensions['length'] = self.get_parameter('robot_length').value
        self.robot_dimensions['height'] = self.get_parameter('robot_height').value

        # Create publisher for robot description
        self.urdf_publisher = self.create_publisher(
            String, 'robot_description', 1)

        # Generate initial URDF
        self.generate_urdf()

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Activate URDF publishing"""
        self.get_logger().info('Activating URDF publisher')

        # Start publishing URDF
        self.urdf_timer = self.create_timer(5.0, self.publish_urdf)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Deactivate URDF publishing"""
        self.get_logger().info('Deactivating URDF publisher')

        # Stop publishing
        if hasattr(self, 'urdf_timer'):
            self.urdf_timer.destroy()

        return TransitionCallbackReturn.SUCCESS

    def generate_urdf(self):
        """Generate URDF based on current parameters"""
        width = self.robot_dimensions['width']
        length = self.robot_dimensions['length']
        height = self.robot_dimensions['height']

        self.robot_description = f"""<?xml version="1.0"?>
        <robot name="parameterized_robot">
          <link name="base_link">
            <visual>
              <geometry>
                <box size="{width} {length} {height}"/>
              </geometry>
            </visual>
            <collision>
              <geometry>
                <box size="{width} {length} {height}"/>
              </geometry>
            </collision>
            <inertial>
              <mass value="1.0"/>
              <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
            </inertial>
          </link>
        </robot>"""

    def publish_urdf(self):
        """Publish the robot description"""
        msg = String()
        msg.data = self.robot_description
        self.urdf_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = URDFLifecycleNode()

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

## Best Practices for Lifecycle Nodes

### 1. Error Handling
```python
def on_configure(self, state):
    try:
        # Configuration code that might fail
        result = self.setup_resources()
        if not result:
            self.get_logger().error('Resource setup failed')
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS
    except Exception as e:
        self.get_logger().error(f'Configuration error: {e}')
        return TransitionCallbackReturn.ERROR
```

### 2. Resource Management
```python
def on_cleanup(self, state):
    # Always properly destroy resources
    if hasattr(self, 'publisher') and self.publisher:
        self.destroy_publisher(self.publisher)
    if hasattr(self, 'subscription') and self.subscription:
        self.destroy_subscription(self.subscription)
    if hasattr(self, 'timer') and self.timer:
        self.timer.destroy()
    return TransitionCallbackReturn.SUCCESS
```

### 3. State Validation
```python
def safe_transition(self):
    """Only perform transition if conditions are met"""
    if not self.required_resources_available():
        return False
    if not self.current_state_allows_transition():
        return False
    return True
```

### 4. Logging and Monitoring
```python
def log_state_change(self, from_state, to_state):
    """Log all state changes for debugging"""
    self.get_logger().info(f'State change: {from_state} → {to_state}')

    # Publish state change to monitoring system
    state_msg = String()
    state_msg.data = f'{from_state}_to_{to_state}'
    self.state_monitor_publisher.publish(state_msg)
```

## Common Lifecycle Node Patterns

### 1. Sensor Node Pattern
```python
class SensorLifecycleNode(LifecycleNode):
    def on_configure(self, state):
        # Initialize sensor hardware
        # Set up data buffers
        pass

    def on_activate(self, state):
        # Start sensor data acquisition
        # Begin publishing sensor data
        pass

    def on_deactivate(self, state):
        # Stop sensor data acquisition
        # Preserve sensor calibration
        pass
```

### 2. Controller Node Pattern
```python
class ControllerLifecycleNode(LifecycleNode):
    def on_configure(self, state):
        # Load control parameters
        # Initialize control algorithms
        pass

    def on_activate(self, state):
        # Start control loop
        # Begin accepting commands
        pass

    def on_deactivate(self, state):
        # Stop control loop
        # Hold current position safely
        pass
```

### 3. Processing Node Pattern
```python
class ProcessingLifecycleNode(LifecycleNode):
    def on_configure(self, state):
        # Load processing algorithms
        # Initialize data structures
        pass

    def on_activate(self, state):
        # Start processing pipeline
        # Begin processing data
        pass

    def on_deactivate(self, state):
        # Stop processing
        # Flush any buffered data
        pass
```

## Summary

Lifecycle nodes provide a powerful framework for managing complex robotic systems with proper initialization, configuration, and shutdown procedures. They are essential for production robotic systems where reliability and proper resource management are critical.

The integration with URDF and parameters demonstrates how lifecycle nodes can be used to create dynamic, configurable robotic systems that can adapt to different hardware configurations while maintaining proper state management.

In the next section, we'll explore URDF in more depth, focusing on humanoid robot applications.