---
sidebar_position: 1
---

# Week 3: ROS 2 Architecture

## Learning Objectives

By the end of this week, you will be able to:
- Explain the fundamental architecture of ROS 2
- Create and run basic ROS 2 nodes
- Implement topic-based communication between nodes
- Use services for request-response communication
- Understand the role of ROS 2 in robotic systems

## ROS 2 Architecture Overview

ROS 2 follows a distributed computing architecture where multiple processes (nodes) communicate with each other through a publish-subscribe messaging system. The architecture is designed to be:

- **Distributed**: Nodes can run on the same or different machines
- **Decentralized**: No single point of failure
- **Language-agnostic**: Supports multiple programming languages
- **Platform-independent**: Works across different operating systems

### Key Components

#### Nodes
Nodes are the fundamental building blocks of ROS 2 applications. Each node is a process that performs computation and communicates with other nodes.

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

    # Keep the node running
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

#### Communication Primitives
ROS 2 provides three main communication patterns:

1. **Topics** (Publish-Subscribe): Asynchronous, one-to-many communication
2. **Services** (Request-Response): Synchronous, one-to-one communication
3. **Actions** (Goal-Feedback-Result): Asynchronous, long-running tasks with feedback

## Topics and Messages

### Understanding Topics
Topics enable asynchronous communication through a publish-subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics they're interested in.

### Message Types
Messages are standardized data structures that flow through topics. ROS 2 provides many built-in message types and allows custom message creation.

#### Common Built-in Message Types:
- `std_msgs`: Basic data types (String, Int32, Float64, etc.)
- `geometry_msgs`: Geometric primitives (Point, Pose, Twist, etc.)
- `sensor_msgs`: Sensor data (LaserScan, Image, JointState, etc.)
- `nav_msgs`: Navigation-related messages (Odometry, Path, etc.)

### Publisher Implementation
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()

    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Implementation
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()

    try:
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Running Publisher and Subscriber
```bash
# Terminal 1: Run publisher
ros2 run your_package minimal_publisher

# Terminal 2: Run subscriber
ros2 run your_package minimal_subscriber
```

## Services

### Understanding Services
Services provide synchronous request-response communication. A service client sends a request and waits for a response from the service server.

### Service Definition
Services are defined using `.srv` files that specify the request and response message types:

```
# Request (before ---)
string name
int32 age
---
# Response (after ---)
bool success
string message
```

### Service Server Implementation
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\n')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation
```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))

    minimal_client.get_logger().info(
        f'Result of add_two_ints: {response.sum}')

    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

### Understanding Actions
Actions are used for long-running tasks that require:
- Goal requests
- Continuous feedback
- Final results
- Cancellation capability

### Action Definition
Actions are defined using `.action` files:

```
# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] partial_sequence
```

## ROS 2 Command-Line Tools

### Common Commands
- `ros2 node list`: List all active nodes
- `ros2 topic list`: List all active topics
- `ros2 service list`: List all active services
- `ros2 action list`: List all active actions
- `ros2 topic echo <topic_name>`: Display messages on a topic
- `ros2 topic info <topic_name>`: Get information about a topic
- `ros2 run <package> <executable>`: Run a node

### Node Management
- `ros2 run <package> <node_name>`: Run a specific node
- `ros2 node info <node_name>`: Get information about a node
- `ros2 lifecycle`: Manage lifecycle nodes

## ROS 2 Middleware (RMW)

### Understanding RMW
ROS 2 uses a middleware abstraction layer called RMW (ROS Middleware) that allows different communication implementations:

- **Fast DDS**: Default middleware, optimized for robotics
- **Cyclone DDS**: Lightweight alternative
- **RTI Connext DDS**: Commercial implementation

### Configuration
Middleware can be selected at runtime or compile time, allowing flexibility for different deployment scenarios.

## Quality of Service (QoS)

### QoS Profiles
ROS 2 provides Quality of Service settings to control communication behavior:

- **Reliability**: Reliable vs. best-effort delivery
- **Durability**: Volatile vs. transient-local durability
- **History**: Keep-all vs. keep-last history
- **Deadline**: Time constraints for message delivery

### Example QoS Configuration
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile for sensor data (best-effort, volatile)
sensor_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)

# Create a QoS profile for critical commands (reliable, transient-local)
command_qos = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)
```

## ROS 2 Launch System

### Launch Files
Launch files allow starting multiple nodes with specific configurations:

```python
# example_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='minimal_listener',
        ),
        Node(
            package='demo_nodes_py',
            executable='talker',
            name='minimal_talker',
        ),
    ])
```

### Launch Commands
- `ros2 launch <package> <launch_file>.py`: Run a launch file
- `ros2 launch <package> <launch_file>.py param:=value`: Pass parameters to launch

## Practical Example: Robot Command System

Let's create a practical example that demonstrates ROS 2 architecture for a simple robot command system:

```python
# robot_commander.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class RobotCommander(Node):
    def __init__(self):
        super().__init__('robot_commander')

        # Create publisher for robot velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create publisher for status messages
        self.status_publisher = self.create_publisher(String, 'status', 10)

        # Create timer to send commands periodically
        self.timer = self.create_timer(1.0, self.send_command)
        self.command_count = 0

    def send_command(self):
        # Create and publish velocity command
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.5  # Move forward at 0.5 m/s
        cmd_msg.angular.z = 0.2  # Turn right at 0.2 rad/s

        self.cmd_vel_publisher.publish(cmd_msg)

        # Publish status message
        status_msg = String()
        status_msg.data = f'Command {self.command_count} sent'
        self.status_publisher.publish(status_msg)

        self.get_logger().info(f'Sent command: linear.x={cmd_msg.linear.x}, angular.z={cmd_msg.angular.z}')
        self.command_count += 1

def main(args=None):
    rclpy.init(args=args)
    commander = RobotCommander()

    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        pass
    finally:
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

### Node Design
- Keep nodes focused on single responsibilities
- Use meaningful names for nodes and topics
- Implement proper error handling and logging
- Use parameters for configuration

### Communication Design
- Choose appropriate communication patterns for your use case
- Use QoS settings appropriate for your application
- Consider bandwidth and latency requirements
- Plan for system scalability

### Code Organization
- Follow ROS 2 package conventions
- Use proper message types for your data
- Document your interfaces clearly
- Test components individually before integration

## Summary

This week introduced the fundamental architecture of ROS 2, including nodes, topics, services, and basic communication patterns. You learned how to create publishers and subscribers, work with services, and understand the middleware abstraction. These concepts form the foundation for building complex robotic systems in subsequent weeks.

In the next section, we'll explore more advanced topics including rclpy integration and practical implementation patterns.