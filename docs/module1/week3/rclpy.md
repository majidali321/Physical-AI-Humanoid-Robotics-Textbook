---
sidebar_position: 3
---

# Week 3: Python Integration with rclpy

## Overview

This section covers the Python client library for ROS 2 (rclpy), which allows you to write ROS 2 nodes in Python. Python is often preferred for rapid prototyping and experimentation in robotics due to its simplicity and extensive ecosystem of scientific libraries.

## Learning Objectives

By the end of this section, you will be able to:

- Create ROS 2 nodes using rclpy
- Implement publishers and subscribers in Python
- Use services and actions with Python
- Apply Python-specific ROS 2 patterns and best practices

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing a Pythonic interface to the ROS 2 graph. It wraps the underlying ROS client library (rcl) and provides Python developers with access to ROS 2 functionality.

### Key Features of rclpy

- **Node Creation**: Simple Python classes to create ROS 2 nodes
- **Topic Communication**: Publishers and subscribers with type checking
- **Service Calls**: Synchronous and asynchronous service interactions
- **Parameter Management**: Dynamic parameter configuration
- **Logging**: Integrated logging with ROS 2 logging system

## Basic Node Structure

Here's a minimal ROS 2 node using rclpy:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from rclpy!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

    # Keep node alive
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

## Publishers and Subscribers

### Creating a Publisher

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
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

### Creating a Subscriber

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

## Services in Python

### Service Server

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
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}')
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

### Service Client

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
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Parameters

Parameters in rclpy allow you to configure nodes at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('robot_name', 'turtlebot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        my_param = self.get_parameter('my_parameter').value
        robot_name = self.get_parameter('robot_name').value
        max_vel = self.get_parameter('max_velocity').value

        self.get_logger().info(f'Parameter values: {my_param}, {robot_name}, {max_vel}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

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

## Best Practices for rclpy

### 1. Use Context Managers for Cleanup

```python
import rclpy
from rclpy.node import Node

def main():
    rclpy.init()
    try:
        node = Node('my_node')
        # Your node logic here
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### 2. Handle Callbacks Properly

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.safe_callback,
            10)

    def safe_callback(self, msg):
        try:
            # Process message
            self.get_logger().info(f'Received: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Error in callback: {e}')
```

### 3. Use QoS Profiles Appropriately

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# For real-time systems
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

publisher = self.create_publisher(String, 'topic', qos_profile)
```

## Advanced rclpy Concepts

### Custom Message Types

When using custom messages, ensure they're properly imported:

```python
from my_package_msgs.msg import MyCustomMessage

class CustomMessageNode(Node):
    def __init__(self):
        super().__init__('custom_message_node')
        self.publisher = self.create_publisher(MyCustomMessage, 'custom_topic', 10)
```

### Timer-based Execution

```python
class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.counter = 0

    def timer_callback(self):
        self.get_logger().info(f'Timer callback {self.counter}')
        self.counter += 1
```

## Common Pitfalls and Solutions

### 1. Threading Issues

rclpy is not thread-safe by default. Use separate contexts for threading:

```python
import rclpy
from rclpy.context import Context

# Create separate contexts for different threads if needed
context1 = Context()
context2 = Context()
```

### 2. Memory Management

Always destroy nodes properly to prevent memory leaks:

```python
def cleanup_node(node):
    if node is not None:
        node.destroy_node()
```

### 3. Parameter Validation

Validate parameters before using them:

```python
def validate_parameters(self):
    if not self.has_parameter('required_param'):
        self.get_logger().error('Required parameter not found')
        return False
    return True
```

## Summary

rclpy provides a Pythonic interface to ROS 2 functionality, making it accessible for rapid prototyping and development. Understanding how to properly use rclpy is crucial for developing humanoid robotics applications in Python.

In the next section, we'll explore practical exercises to reinforce these concepts.