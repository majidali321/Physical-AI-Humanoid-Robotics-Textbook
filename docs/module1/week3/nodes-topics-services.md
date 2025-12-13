---
sidebar_position: 2
---

# Nodes, Topics, and Services in Detail

## Deep Dive into Nodes

### Node Lifecycle
A ROS 2 node goes through several states during its lifetime:

1. **Unconfigured**: Node created but not yet configured
2. **Inactive**: Node configured but not active
3. **Active**: Node running and processing callbacks
4. **Finalized**: Node destroyed and cleaned up

### Node Creation and Management

#### Basic Node Structure
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('my_node_name')

        # Node initialization code goes here
        self.get_logger().info('Node initialized')

def main(args=None):
    rclpy.init(args=args)

    # Create node instance
    node = MyNode()

    try:
        # Keep node running
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Node Parameters
Nodes can accept parameters to customize their behavior:

```python
import rclpy
from rclpy.node import Node

class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with default values
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('topic_name', 'my_topic')
        self.declare_parameter('robot_name', 'default_robot')

        # Get parameter values
        self.publish_rate = self.get_parameter('publish_rate').value
        self.topic_name = self.get_parameter('topic_name').value
        self.robot_name = self.get_parameter('robot_name').value

        self.get_logger().info(
            f'Initialized with parameters: '
            f'rate={self.publish_rate}, '
            f'topic={self.topic_name}, '
            f'robot={self.robot_name}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = ParameterizedNode()

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

#### Running with Parameters
```bash
# Using command line
ros2 run my_package parameterized_node --ros-args -p publish_rate:=2.0 -p robot_name:=my_robot

# Using parameter file
ros2 run my_package parameterized_node --ros-args --params-file params.yaml
```

### Node Names and Namespacing
```python
class NamespacedNode(Node):
    def __init__(self):
        # Create node with namespace
        super().__init__('my_node', namespace='robot1')
        # This creates a node named '/robot1/my_node'

        # Publishers and subscribers will be namespaced automatically
        self.publisher = self.create_publisher(String, 'data', 10)
        # This creates topic '/robot1/data'
```

## Advanced Topic Communication

### Publisher Configuration

#### Quality of Service (QoS) Settings
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class QoSPublisher(Node):
    def __init__(self):
        super().__init__('qos_publisher')

        # Different QoS profiles for different use cases

        # For sensor data (best-effort, volatile)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # For critical commands (reliable, transient-local)
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # For configuration data (reliable, persistent)
        config_qos = QoSProfile(
            history=HistoryPolicy.KEEP_ALL,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.sensor_publisher = self.create_publisher(String, 'sensor_data', sensor_qos)
        self.command_publisher = self.create_publisher(Twist, 'cmd_vel', command_qos)
        self.config_publisher = self.create_publisher(String, 'config', config_qos)
```

#### Publisher with Callbacks
```python
class PublisherWithCallbacks(Node):
    def __init__(self):
        super().__init__('publisher_callbacks')

        # Create publisher with callbacks
        self.publisher = self.create_publisher(String, 'data', 10)

        # Add publisher callbacks for monitoring
        self.publisher.add_wait_set_ready_callback(
            lambda: self.get_logger().info('Publisher ready')
        )

    def publish_with_check(self, msg):
        if self.publisher.get_subscription_count() > 0:
            self.publisher.publish(msg)
            self.get_logger().info(f'Published: {msg.data}')
        else:
            self.get_logger().warn('No subscribers for this topic')
```

### Subscriber Configuration

#### Subscription Callbacks
```python
class AdvancedSubscriber(Node):
    def __init__(self):
        super().__init__('advanced_subscriber')

        # Create subscription with custom callback
        self.subscription = self.create_subscription(
            String,
            'data',
            self.data_callback,
            10,
            # Custom QoS if needed
            # qos_profile=custom_qos
        )

        # Store received messages
        self.message_history = []

    def data_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

        # Store message for later processing
        self.message_history.append({
            'timestamp': self.get_clock().now(),
            'data': msg.data
        })

        # Limit history size
        if len(self.message_history) > 100:
            self.message_history.pop(0)
```

#### Subscription with Header Information
```python
from std_msgs.msg import Header

class TimestampedSubscriber(Node):
    def __init__(self):
        super().__init__('timestamped_subscriber')
        self.subscription = self.create_subscription(
            String,  # This would typically be a custom message with Header
            'data',
            self.data_callback,
            10
        )

    def data_callback(self, msg):
        # In practice, you'd use a message type that includes Header
        # For now, we'll simulate the concept
        current_time = self.get_clock().now()
        self.get_logger().info(f'Received at {current_time.to_msg()}: {msg.data}')
```

### Message Types and Custom Messages

#### Using Built-in Message Types
```python
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import LaserScan, Image, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Int32, Float64

class MessageExample(Node):
    def __init__(self):
        super().__init__('message_example')

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # Publisher for sensor data
        self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)

    def send_velocity_command(self, linear_x, angular_z):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)

    def send_joint_positions(self, names, positions):
        joint_msg = JointState()
        joint_msg.name = names
        joint_msg.position = positions
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'
        self.joint_pub.publish(joint_msg)
```

#### Creating Custom Messages
Custom messages are defined in `.msg` files in the `msg/` directory of a package:

```
# In msg/RobotStatus.msg
string robot_name
int32 battery_level
bool is_charging
float64[] joint_angles
geometry_msgs/Pose current_pose
```

Then use in Python:
```python
from my_robot_msgs.msg import RobotStatus

class StatusPublisher(Node):
    def __init__(self):
        super().__init__('status_publisher')
        self.status_pub = self.create_publisher(RobotStatus, 'robot_status', 10)

    def publish_status(self):
        msg = RobotStatus()
        msg.robot_name = 'my_robot'
        msg.battery_level = 85
        msg.is_charging = False
        msg.joint_angles = [0.1, 0.2, 0.3]
        # Pose would be populated similarly
        self.status_pub.publish(msg)
```

## Services in Depth

### Service Server Implementation

#### Basic Service Server
```python
from example_interfaces.srv import AddTwoInts

class BasicServiceServer(Node):
    def __init__(self):
        super().__init__('basic_service_server')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response
```

#### Service Server with Error Handling
```python
from example_interfaces.srv import AddTwoInts

class RobustServiceServer(Node):
    def __init__(self):
        super().__init__('robust_service_server')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints_safe',
            self.safe_add_callback
        )
        self.request_count = 0

    def safe_add_callback(self, request, response):
        try:
            # Validate inputs
            if not isinstance(request.a, int) or not isinstance(request.b, int):
                self.get_logger().error('Invalid input types')
                response.sum = 0
                return response

            # Perform calculation
            result = request.a + request.b

            # Check for overflow (simplified)
            if result > 2**31 - 1 or result < -2**31:
                self.get_logger().warn('Potential overflow detected')

            response.sum = result
            self.request_count += 1
            self.get_logger().info(f'Request #{self.request_count}: {request.a} + {request.b} = {response.sum}')

        except Exception as e:
            self.get_logger().error(f'Service error: {e}')
            response.sum = 0

        return response
```

### Service Client Implementation

#### Basic Service Client
```python
from example_interfaces.srv import AddTwoInts

class BasicServiceClient(Node):
    def __init__(self):
        super().__init__('basic_service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = AddTwoInts.Request()

    def send_request(self, a, b):
        self.request.a = a
        self.request.b = b

        # Send request asynchronously
        self.future = self.cli.call_async(self.request)
        return self.future
```

#### Async Service Client with Callback
```python
from example_interfaces.srv import AddTwoInts

class AsyncServiceClient(Node):
    def __init__(self):
        super().__init__('async_service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

        self.request = AddTwoInts.Request()

    def send_request_async(self, a, b):
        self.request.a = a
        self.request.b = b

        # Send request and add callback
        future = self.cli.call_async(self.request)
        future.add_done_callback(self.service_response_callback)

        return future

    def service_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Service response: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
```

### Service with Custom Message Types

#### Custom Service Definition
Create a service file `srv/CalculateTrajectory.srv`:
```
# Request
geometry_msgs/Pose start_pose
geometry_msgs/Pose goal_pose
float64 max_velocity
---
# Response
nav_msgs/Path trajectory
bool success
string message
```

#### Using Custom Service
```python
from my_robot_msgs.srv import CalculateTrajectory

class TrajectoryServiceServer(Node):
    def __init__(self):
        super().__init__('trajectory_service')
        self.srv = self.create_service(
            CalculateTrajectory,
            'calculate_trajectory',
            self.calculate_trajectory_callback
        )

    def calculate_trajectory_callback(self, request, response):
        try:
            # In a real implementation, this would calculate a path
            # For now, we'll simulate the process

            # Create a simple trajectory (straight line)
            path = nav_msgs.msg.Path()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = 'map'

            # Add some waypoints (simplified)
            for i in range(10):
                pose_stamped = geometry_msgs.msg.PoseStamped()
                pose_stamped.pose.position.x = request.start_pose.position.x + \
                    (request.goal_pose.position.x - request.start_pose.position.x) * i / 9
                pose_stamped.pose.position.y = request.start_pose.position.y + \
                    (request.goal_pose.position.y - request.start_pose.position.y) * i / 9
                path.poses.append(pose_stamped)

            response.trajectory = path
            response.success = True
            response.message = 'Trajectory calculated successfully'

        except Exception as e:
            response.success = False
            response.message = f'Error calculating trajectory: {str(e)}'

        return response
```

## Communication Patterns and Best Practices

### Publisher-Subscriber Pattern Variations

#### Fan-out Pattern
```python
class FanOutPublisher(Node):
    def __init__(self):
        super().__init__('fan_out_publisher')

        # Publish to multiple topics
        self.status_pub = self.create_publisher(String, 'status', 10)
        self.log_pub = self.create_publisher(String, 'log', 10)
        self.debug_pub = self.create_publisher(String, 'debug', 10)

    def publish_to_all(self, message):
        msg = String()
        msg.data = message

        self.status_pub.publish(msg)
        self.log_pub.publish(msg)
        self.debug_pub.publish(msg)
```

#### Data Aggregation Pattern
```python
class DataAggregator(Node):
    def __init__(self):
        super().__init__('data_aggregator')

        # Subscribe to multiple data sources
        self.sub1 = self.create_subscription(String, 'sensor1_data', self.sensor1_callback, 10)
        self.sub2 = self.create_subscription(String, 'sensor2_data', self.sensor2_callback, 10)
        self.sub3 = self.create_subscription(String, 'sensor3_data', self.sensor3_callback, 10)

        # Publish aggregated data
        self.agg_pub = self.create_publisher(String, 'aggregated_data', 10)

        # Store data from each sensor
        self.sensor_data = {'sensor1': None, 'sensor2': None, 'sensor3': None}

    def sensor1_callback(self, msg):
        self.sensor_data['sensor1'] = msg.data
        self.check_and_publish_aggregated()

    def sensor2_callback(self, msg):
        self.sensor_data['sensor2'] = msg.data
        self.check_and_publish_aggregated()

    def sensor3_callback(self, msg):
        self.sensor_data['sensor3'] = msg.data
        self.check_and_publish_aggregated()

    def check_and_publish_aggregated(self):
        # Only publish when all sensors have provided data
        if all(self.sensor_data.values()):
            aggregated = f"Sensor1: {self.sensor_data['sensor1']}, " \
                        f"Sensor2: {self.sensor_data['sensor2']}, " \
                        f"Sensor3: {self.sensor_data['sensor3']}"

            agg_msg = String()
            agg_msg.data = aggregated
            self.agg_pub.publish(agg_msg)

            # Reset for next cycle
            self.sensor_data = {'sensor1': None, 'sensor2': None, 'sensor3': None}
```

### Service Design Patterns

#### Command Pattern
```python
from std_srvs.srv import Trigger

class RobotCommander(Node):
    def __init__(self):
        super().__init__('robot_commander')

        # Multiple command services
        self.move_service = self.create_service(Trigger, 'move_forward', self.move_forward_callback)
        self.stop_service = self.create_service(Trigger, 'stop', self.stop_callback)
        self.home_service = self.create_service(Trigger, 'go_home', self.go_home_callback)

    def move_forward_callback(self, request, response):
        # Implement move forward logic
        self.get_logger().info('Moving forward')
        response.success = True
        response.message = 'Moving forward'
        return response

    def stop_callback(self, request, response):
        # Implement stop logic
        self.get_logger().info('Stopping')
        response.success = True
        response.message = 'Stopped'
        return response

    def go_home_callback(self, request, response):
        # Implement go home logic
        self.get_logger().info('Going home')
        response.success = True
        response.message = 'Going home'
        return response
```

#### Configuration Service Pattern
```python
from example_interfaces.srv import SetBool

class ConfigurableNode(Node):
    def __init__(self):
        super().__init__('configurable_node')

        # Configuration services
        self.enable_service = self.create_service(SetBool, 'enable', self.enable_callback)
        self.disable_service = self.create_service(SetBool, 'disable', self.disable_callback)

        # Runtime parameters
        self.enabled = True

    def enable_callback(self, request, response):
        self.enabled = request.data
        response.success = True
        response.message = f'Node {"enabled" if self.enabled else "disabled"}'
        return response

    def disable_callback(self, request, response):
        self.enabled = not request.data
        response.success = True
        response.message = f'Node {"enabled" if self.enabled else "disabled"}'
        return response
```

## Performance Considerations

### Efficient Message Handling
```python
class EfficientNode(Node):
    def __init__(self):
        super().__init__('efficient_node')

        # Use appropriate queue sizes
        self.subscription = self.create_subscription(
            String,
            'data',
            self.data_callback,
            1  # Small queue for real-time processing
        )

        # Pre-allocate message objects to reduce garbage collection
        self.msg_buffer = String()

    def data_callback(self, msg):
        # Process message efficiently
        # Avoid creating unnecessary objects in callback
        processed_data = self.process_data(msg.data)

        # If publishing result, do it efficiently
        self.msg_buffer.data = processed_data
        # self.publisher.publish(self.msg_buffer)  # Uncomment if needed
```

### Memory Management
```python
class MemoryAwareNode(Node):
    def __init__(self):
        super().__init__('memory_aware_node')

        self.subscription = self.create_subscription(String, 'data', self.data_callback, 10)

        # Limit message history to prevent memory buildup
        self.message_buffer = []
        self.max_buffer_size = 100

    def data_callback(self, msg):
        # Add to buffer
        self.message_buffer.append(msg.data)

        # Maintain buffer size
        if len(self.message_buffer) > self.max_buffer_size:
            self.message_buffer.pop(0)  # Remove oldest
```

## Summary

This detailed exploration of nodes, topics, and services covers the essential communication mechanisms in ROS 2. Understanding these concepts is crucial for building robust robotic systems. The examples demonstrate both basic and advanced usage patterns that you'll encounter in real-world robotics applications.

In the next section, we'll explore Python integration with ROS 2 using rclpy and practical implementation patterns.