---
sidebar_position: 4
---

# Week 4 Exercises: Parameters, Lifecycle Nodes, and URDF Basics

## Exercise 1: Parameter Management System

### Objective
Create a comprehensive parameter management system for a robot controller that demonstrates dynamic reconfiguration capabilities.

### Requirements
1. Create a ROS 2 package called `week4_exercises`
2. Implement a parameter server node with validation callbacks
3. Create a parameter client that can modify parameters at runtime
4. Include parameter groups for different robot subsystems
5. Implement parameter persistence using YAML files

### Implementation Steps

1. **Create the package:**
   ```bash
   cd ~/physical_ai_ws/src
   ros2 pkg create --build-type ament_python week4_exercises
   ```

2. **Create the parameter server node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
   from rcl_interfaces.msg import IntegerRange, FloatingPointRange
   from std_msgs.msg import String
   import yaml
   import os

   class ParameterServerNode(Node):
       def __init__(self):
           super().__init__('parameter_server')

           # Declare parameters with descriptors
           self.declare_parameter_with_descriptor()

           # Set up parameter callback
           self.add_on_set_parameters_callback(self.parameter_callback)

           # Publishers for parameter updates
           self.param_change_publisher = self.create_publisher(String, 'parameter_changes', 10)

           # Parameter persistence
           self.config_file = os.path.expanduser('~/.ros/robot_config.yaml')

           self.get_logger().info('Parameter server initialized')

       def declare_parameter_with_descriptor(self):
           # Navigation parameters
           nav_desc = ParameterDescriptor()
           nav_desc.description = 'Maximum linear velocity (m/s)'
           nav_desc.floating_range = [FloatingPointRange(from_value=0.0, to_value=5.0, step=0.1)]
           self.declare_parameter('nav.max_velocity', 1.0, nav_desc)

           nav_desc2 = ParameterDescriptor()
           nav_desc2.description = 'Safety distance threshold (m)'
           nav_desc2.floating_range = [FloatingPointRange(from_value=0.1, to_value=2.0, step=0.05)]
           self.declare_parameter('nav.safety_distance', 0.5, nav_desc2)

           # Arm control parameters
           arm_desc = ParameterDescriptor()
           arm_desc.description = 'Arm movement speed (rad/s)'
           arm_desc.floating_range = [FloatingPointRange(from_value=0.1, to_value=2.0, step=0.1)]
           self.declare_parameter('arm.speed', 0.5, arm_desc)

           # Head control parameters
           head_desc = ParameterDescriptor()
           head_desc.description = 'Head pan range (rad)'
           head_desc.floating_range = [FloatingPointRange(from_value=0.1, to_value=1.57, step=0.01)]
           self.declare_parameter('head.pan_range', 0.785, head_desc)

           # Robot identification
           id_desc = ParameterDescriptor()
           id_desc.description = 'Robot name'
           self.declare_parameter('robot.name', 'humanoid_robot_01', id_desc)

       def parameter_callback(self, params):
           """Validate and process parameter changes"""
           result = SetParametersResult()
           result.successful = True

           for param in params:
               param_name = param.name
               param_value = param.value

               # Validate navigation parameters
               if param_name.startswith('nav.'):
                   if param_name == 'nav.max_velocity':
                       if not (0.0 <= param_value <= 5.0):
                           result.successful = False
                           result.reason = f'Max velocity {param_value} out of range [0.0, 5.0]'
                           return result
                   elif param_name == 'nav.safety_distance':
                       if not (0.1 <= param_value <= 2.0):
                           result.successful = False
                           result.reason = f'Safety distance {param_value} out of range [0.1, 2.0]'
                           return result

               # Validate arm parameters
               elif param_name.startswith('arm.'):
                   if param_name == 'arm.speed':
                       if not (0.1 <= param_value <= 2.0):
                           result.successful = False
                           result.reason = f'Arm speed {param_value} out of range [0.1, 2.0]'
                           return result

               # Validate head parameters
               elif param_name.startswith('head.'):
                   if param_name == 'head.pan_range':
                       if not (0.1 <= param_value <= 1.57):
                           result.successful = False
                           result.reason = f'Head pan range {param_value} out of range [0.1, 1.57]'
                           return result

               # Log parameter change
               self.get_logger().info(f'Parameter updated: {param_name} = {param_value}')

               # Publish parameter change notification
               change_msg = String()
               change_msg.data = f'PARAMETER_CHANGE: {param_name} = {param_value}'
               self.param_change_publisher.publish(change_msg)

           # Save parameters to file after successful update
           self.save_parameters_to_file()

           return result

       def save_parameters_to_file(self):
           """Save current parameters to a YAML file"""
           try:
               params = {}
               for param_name in self._parameters.keys():
                   params[param_name] = self.get_parameter(param_name).value

               # Create directory if it doesn't exist
               os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

               with open(self.config_file, 'w') as f:
                   yaml.dump(params, f, default_flow_style=False)

               self.get_logger().info(f'Parameters saved to {self.config_file}')
           except Exception as e:
               self.get_logger().error(f'Failed to save parameters: {e}')

       def load_parameters_from_file(self):
           """Load parameters from a YAML file"""
           try:
               if os.path.exists(self.config_file):
                   with open(self.config_file, 'r') as f:
                       params = yaml.safe_load(f)

                   for param_name, param_value in params.items():
                       if self.has_parameter(param_name):
                           self.set_parameters([rclpy.Parameter(param_name, value=param_value)])
                           self.get_logger().info(f'Loaded parameter: {param_name} = {param_value}')
                       else:
                           self.get_logger().warn(f'Unknown parameter in config: {param_name}')
               else:
                   self.get_logger().info('No config file found, using defaults')
           except Exception as e:
               self.get_logger().error(f'Failed to load parameters: {e}')

   def main(args=None):
       rclpy.init(args=args)
       node = ParameterServerNode()

       # Load parameters from file
       node.load_parameters_from_file()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Shutting down parameter server')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create the parameter client node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from rclpy.parameter import Parameter
   from std_msgs.msg import String
   import time
   import random

   class ParameterClientNode(Node):
       def __init__(self):
           super().__init__('parameter_client')

           # Create parameter client
           self.declare_parameter('target_node', 'parameter_server')
           self.target_node = self.get_parameter('target_node').value

           # Create timer to periodically update parameters
           self.timer = self.create_timer(5.0, self.update_parameters)

           # Subscription to parameter changes
           self.change_subscription = self.create_subscription(
               String, 'parameter_changes', self.change_callback, 10)

           self.get_logger().info(f'Parameter client initialized, targeting: {self.target_node}')

       def update_parameters(self):
           """Periodically update robot parameters"""
           # Update navigation parameters
           max_vel = random.uniform(0.5, 2.0)
           safety_dist = random.uniform(0.3, 1.0)

           # Update parameters
           self.set_parameters([
               Parameter('nav.max_velocity', value=max_vel),
               Parameter('nav.safety_distance', value=safety_dist)
           ])

           self.get_logger().info(f'Updated navigation params: vel={max_vel:.2f}, dist={safety_dist:.2f}')

       def change_callback(self, msg):
           """Log parameter changes"""
           self.get_logger().info(f'Observed: {msg.data}')

   def main(args=None):
       rclpy.init(args=args)
       node = ParameterClientNode()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Shutting down parameter client')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Update package.xml:**
   ```xml
   <?xml version="1.0"?>
   <?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
   <package format="3">
     <name>week4_exercises</name>
     <version>0.0.0</version>
     <description>Week 4 exercises for ROS 2 advanced topics</description>
     <maintainer email="user@example.com">Your Name</maintainer>
     <license>Apache-2.0</license>

     <depend>rclpy</depend>
     <depend>std_msgs</depend>
     <depend>rcl_interfaces</depend>

     <test_depend>ament_copyright</test_depend>
     <test_depend>ament_flake8</test_depend>
     <test_depend>ament_pep257</test_depend>
     <test_depend>python3-pytest</test_depend>

     <export>
       <build_type>ament_python</build_type>
     </export>
   </package>
   ```

5. **Create setup files and run the system:**
   ```bash
   cd ~/physical_ai_ws
   colcon build --packages-select week4_exercises
   source install/setup.bash

   # Terminal 1: Run parameter server
   ros2 run week4_exercises parameter_server

   # Terminal 2: Run parameter client
   ros2 run week4_exercises parameter_client
   ```

### Expected Output
- Parameter server validates all parameter changes
- Parameter client periodically updates navigation parameters
- Changes are logged and published
- Parameters are saved to and loaded from file

### Submission Requirements
- Complete parameter server and client implementation
- Demonstration of parameter validation
- File persistence functionality
- Screenshots of successful parameter updates

## Exercise 2: Lifecycle Node System

### Objective
Implement a complete lifecycle node system for a humanoid robot with multiple coordinated components.

### Requirements
1. Create a lifecycle node for robot control
2. Implement multiple lifecycle states with proper transitions
3. Include sensor and actuator management
4. Add error handling and recovery mechanisms
5. Create a lifecycle manager client

### Implementation Steps

1. **Create the lifecycle robot controller:**
   ```python
   #!/usr/bin/env python3
   from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
   from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
   from sensor_msgs.msg import JointState, LaserScan
   from geometry_msgs.msg import Twist
   from std_msgs.msg import String, Bool
   import threading
   import time

   class RobotLifecycleController(LifecycleNode):
       def __init__(self):
           super().__init__('robot_lifecycle_controller')

           # Component initialization
           self.joint_publisher = None
           self.cmd_vel_publisher = None
           self.scan_subscriber = None
           self.status_publisher = None
           self.control_timer = None

           # Robot state
           self.current_state_label = 'unconfigured'
           self.joint_positions = {}
           self.emergency_stop = False
           self.hardware_status = {'sensors': 'ok', 'actuators': 'ok', 'comms': 'ok'}

       def on_configure(self, state):
           self.get_logger().info(f'Configuring robot controller from state: {state.label}')

           # Create communications (inactive)
           qos_profile = QoSProfile(depth=10)

           self.joint_publisher = self.create_publisher(JointState, 'joint_states', qos_profile)
           self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', qos_profile)
           self.scan_subscriber = self.create_subscription(
               LaserScan, 'scan', self.scan_callback, qos_profile)
           self.status_publisher = self.create_publisher(String, 'robot_status', 10)

           # Initialize joint positions
           self.initialize_joints()

           # Validate hardware
           if not self.validate_hardware():
               self.get_logger().error('Hardware validation failed')
               return TransitionCallbackReturn.FAILURE

           self.current_state_label = 'inactive'
           self.get_logger().info('Robot controller configured successfully')
           return TransitionCallbackReturn.SUCCESS

       def on_activate(self, state):
           self.get_logger().info(f'Activating robot controller from state: {state.label}')

           # Activate communications
           self.joint_publisher.on_activate()
           self.cmd_vel_publisher.on_activate()

           # Start control timer
           self.control_timer = self.create_timer(0.1, self.control_loop)

           # Publish activation status
           status_msg = String()
           status_msg.data = f'Robot controller activated, state: active'
           self.status_publisher.publish(status_msg)

           self.current_state_label = 'active'
           self.get_logger().info('Robot controller activated successfully')
           return TransitionCallbackReturn.SUCCESS

       def on_deactivate(self, state):
           self.get_logger().info(f'Deactivating robot controller from state: {state.label}')

           # Stop robot movement
           stop_cmd = Twist()
           self.cmd_vel_publisher.publish(stop_cmd)

           # Deactivate publishers
           self.joint_publisher.on_deactivate()
           self.cmd_vel_publisher.on_deactivate()

           # Stop control timer
           if self.control_timer:
               self.control_timer.destroy()
               self.control_timer = None

           self.current_state_label = 'inactive'
           self.get_logger().info('Robot controller deactivated successfully')
           return TransitionCallbackReturn.SUCCESS

       def on_cleanup(self, state):
           self.get_logger().info(f'Cleaning up robot controller from state: {state.label}')

           # Destroy all communications
           if self.joint_publisher:
               self.destroy_publisher(self.joint_publisher)
               self.joint_publisher = None
           if self.cmd_vel_publisher:
               self.destroy_publisher(self.cmd_vel_publisher)
               self.cmd_vel_publisher = None
           if self.scan_subscriber:
               self.destroy_subscription(self.scan_subscriber)
               self.scan_subscriber = None
           if self.status_publisher:
               self.destroy_publisher(self.status_publisher)
               self.status_publisher = None

           self.current_state_label = 'unconfigured'
           self.get_logger().info('Robot controller cleaned up successfully')
           return TransitionCallbackReturn.SUCCESS

       def on_shutdown(self, state):
           self.get_logger().info(f'Shutting down robot controller from state: {state.label}')

           # Perform emergency stop
           self.emergency_stop = True
           stop_cmd = Twist()
           if self.cmd_vel_publisher:
               self.cmd_vel_publisher.publish(stop_cmd)

           self.current_state_label = 'finalized'
           self.get_logger().info('Robot controller shutdown complete')
           return TransitionCallbackReturn.SUCCESS

       def on_error(self, state):
           self.get_logger().info(f'Robot controller entered error state from: {state.label}')

           # Emergency stop
           self.emergency_stop = True
           stop_cmd = Twist()
           if self.cmd_vel_publisher:
               self.cmd_vel_publisher.publish(stop_cmd)

           self.current_state_label = 'error'
           return TransitionCallbackReturn.SUCCESS

       def initialize_joints(self):
           """Initialize joint positions"""
           joint_names = [
               'left_hip', 'left_knee', 'left_ankle',
               'right_hip', 'right_knee', 'right_ankle',
               'left_shoulder', 'left_elbow', 'right_shoulder', 'right_elbow'
           ]

           for name in joint_names:
               self.joint_positions[name] = 0.0

       def validate_hardware(self):
           """Validate hardware connections"""
           self.get_logger().info('Validating hardware connections...')
           time.sleep(0.5)  # Simulate hardware check
           return True  # In real implementation, check actual hardware status

       def scan_callback(self, msg):
           """Process laser scan data"""
           if self.current_state_label == 'active' and not self.emergency_stop:
               # Check for obstacles and trigger safety if needed
               if self.detect_obstacles(msg):
                   self.trigger_safety_stop()

       def detect_obstacles(self, scan_msg):
           """Detect obstacles in front of robot"""
           if len(scan_msg.ranges) > 0:
               center_idx = len(scan_msg.ranges) // 2
               front_distance = scan_msg.ranges[center_idx]
               return 0 < front_distance < 0.5  # 50cm threshold
           return False

       def trigger_safety_stop(self):
           """Trigger safety stop procedure"""
           self.get_logger().warn('Obstacle detected! Triggering safety stop.')
           stop_cmd = Twist()
           self.cmd_vel_publisher.publish(stop_cmd)

       def control_loop(self):
           """Main control loop when active"""
           if self.emergency_stop or self.current_state_label != 'active':
               return

           # Publish joint states
           self.publish_joint_states()

           # Publish status periodically
           if self.get_clock().now().nanoseconds % 2000000000 < 100000000:  # Every 2 seconds
               self.publish_status()

       def publish_joint_states(self):
           """Publish current joint states"""
           msg = JointState()
           msg.header.stamp = self.get_clock().now().to_msg()
           msg.name = list(self.joint_positions.keys())
           msg.position = list(self.joint_positions.values())
           msg.velocity = [0.0] * len(msg.position)
           msg.effort = [0.0] * len(msg.position)

           self.joint_publisher.publish(msg)

       def publish_status(self):
           """Publish robot status"""
           status_msg = String()
           status_msg.data = f'Robot operational, joints: {len(self.joint_positions)}, state: {self.current_state_label}'
           self.status_publisher.publish(status_msg)

   def main(args=None):
       rclpy.init(args=args)
       node = RobotLifecycleController()

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

2. **Create a lifecycle manager client:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from lifecycle_msgs.srv import ChangeState, GetState
   from lifecycle_msgs.msg import Transition
   import time

   class LifecycleManagerClient(Node):
       def __init__(self):
           super().__init__('lifecycle_manager_client')

           # Create clients for lifecycle management
           self.change_state_client = self.create_client(
               ChangeState, 'robot_lifecycle_controller/change_state')
           self.get_state_client = self.create_client(
               GetState, 'robot_lifecycle_controller/get_state')

           # Wait for services
           while not self.change_state_client.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Change state service not available, waiting...')
           while not self.get_state_client.wait_for_service(timeout_sec=1.0):
               self.get_logger().info('Get state service not available, waiting...')

           # Create timer to run lifecycle sequence
           self.timer = self.create_timer(10.0, self.lifecycle_sequence)
           self.sequence_step = 0

           self.get_logger().info('Lifecycle manager client initialized')

       def lifecycle_sequence(self):
           """Run a sequence of lifecycle transitions"""
           if self.sequence_step == 0:
               self.configure_node()
           elif self.sequence_step == 1:
               self.activate_node()
           elif self.sequence_step == 2:
               self.deactivate_node()
           elif self.sequence_step == 3:
               self.cleanup_node()
           elif self.sequence_step == 4:
               self.get_logger().info('Lifecycle sequence completed')
               self.timer.cancel()  # Stop the timer
               return

           self.sequence_step += 1

       def configure_node(self):
           """Configure the lifecycle node"""
           request = ChangeState.Request()
           request.transition.id = Transition.TRANSITION_CONFIGURE
           request.transition.label = 'configure'

           future = self.change_state_client.call_async(request)
           future.add_done_callback(self.configuration_response)

       def activate_node(self):
           """Activate the lifecycle node"""
           request = ChangeState.Request()
           request.transition.id = Transition.TRANSITION_ACTIVATE
           request.transition.label = 'activate'

           future = self.change_state_client.call_async(request)
           future.add_done_callback(self.activation_response)

       def deactivate_node(self):
           """Deactivate the lifecycle node"""
           request = ChangeState.Request()
           request.transition.id = Transition.TRANSITION_DEACTIVATE
           request.transition.label = 'deactivate'

           future = self.change_state_client.call_async(request)
           future.add_done_callback(self.deactivation_response)

       def cleanup_node(self):
           """Clean up the lifecycle node"""
           request = ChangeState.Request()
           request.transition.id = Transition.TRANSITION_CLEANUP
           request.transition.label = 'cleanup'

           future = self.change_state_client.call_async(request)
           future.add_done_callback(self.cleanup_response)

       def get_current_state(self):
           """Get current state of the lifecycle node"""
           request = GetState.Request()

           future = self.get_state_client.call_async(request)
           future.add_done_callback(self.state_response)

       def configuration_response(self, future):
           try:
               response = future.result()
               if response.success:
                   self.get_logger().info('Node configured successfully')
                   self.get_current_state()
               else:
                   self.get_logger().error(f'Configuration failed: {response.error_message}')
           except Exception as e:
               self.get_logger().error(f'Configuration service call failed: {e}')

       def activation_response(self, future):
           try:
               response = future.result()
               if response.success:
                   self.get_logger().info('Node activated successfully')
                   self.get_current_state()
               else:
                   self.get_logger().error(f'Activation failed: {response.error_message}')
           except Exception as e:
               self.get_logger().error(f'Activation service call failed: {e}')

       def deactivation_response(self, future):
           try:
               response = future.result()
               if response.success:
                   self.get_logger().info('Node deactivated successfully')
                   self.get_current_state()
               else:
                   self.get_logger().error(f'Deactivation failed: {response.error_message}')
           except Exception as e:
               self.get_logger().error(f'Deactivation service call failed: {e}')

       def cleanup_response(self, future):
           try:
               response = future.result()
               if response.success:
                   self.get_logger().info('Node cleaned up successfully')
                   self.get_current_state()
               else:
                   self.get_logger().error(f'Cleanup failed: {response.error_message}')
           except Exception as e:
               self.get_logger().error(f'Cleanup service call failed: {e}')

       def state_response(self, future):
           try:
               response = future.result()
               self.get_logger().info(f'Current state: {response.current_state.label}')
           except Exception as e:
               self.get_logger().error(f'Get state service call failed: {e}')

   def main(args=None):
       rclpy.init(args=args)
       node = LifecycleManagerClient()

       try:
           rclpy.spin(node)
       except KeyboardInterrupt:
           node.get_logger().info('Shutting down lifecycle manager client')
       finally:
           node.destroy_node()
           rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Test the lifecycle system:**
   ```bash
   # Terminal 1: Run lifecycle controller
   ros2 run week4_exercises robot_lifecycle_controller

   # Terminal 2: Run lifecycle manager client
   ros2 run week4_exercises lifecycle_manager_client

   # Terminal 3: Use command line tools
   ros2 lifecycle list robot_lifecycle_controller
   ros2 lifecycle get robot_lifecycle_controller
   ros2 lifecycle configure robot_lifecycle_controller
   ros2 lifecycle activate robot_lifecycle_controller
   ```

### Expected Output
- Lifecycle node transitions through all states properly
- Manager client controls state transitions
- Proper error handling and resource management
- State information published and accessible

### Submission Requirements
- Complete lifecycle controller implementation
- Lifecycle manager client code
- Demonstration of all state transitions
- Error handling and recovery procedures

## Exercise 3: URDF for Humanoid Robot

### Objective
Create a complete URDF model for a humanoid robot with realistic proportions and proper kinematic structure.

### Requirements
1. Create a detailed humanoid URDF with at least 20 joints
2. Include proper visual, collision, and inertial properties
3. Use Xacro for parameterization and modularity
4. Include sensor mounts and realistic dimensions
5. Validate the URDF and test with RViz2

### Implementation Steps

1. **Create the main URDF file with Xacro:**
   ```xml
   <?xml version="1.0"?>
   <robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_humanoid">

     <!-- Properties and constants -->
     <xacro:property name="M_PI" value="3.1415926535897931" />
     <xacro:property name="torso_height" value="0.8" />
     <xacro:property name="torso_width" value="0.3" />
     <xacro:property name="torso_depth" value="0.25" />
     <xacro:property name="head_radius" value="0.12" />
     <xacro:property name="upper_arm_length" value="0.35" />
     <xacro:property name="lower_arm_length" value="0.3" />
     <xacro:property name="upper_arm_radius" value="0.06" />
     <xacro:property name="lower_arm_radius" value="0.05" />
     <xacro:property name="upper_leg_length" value="0.45" />
     <xacro:property name="lower_leg_length" value="0.4" />
     <xacro:property name="foot_length" value="0.25" />
     <xacro:property name="foot_width" value="0.12" />
     <xacro:property name="foot_height" value="0.08" />

     <!-- Materials -->
     <material name="black">
       <color rgba="0.1 0.1 0.1 1.0"/>
     </material>
     <material name="white">
       <color rgba="0.9 0.9 0.9 1.0"/>
     </material>
     <material name="blue">
       <color rgba="0.0 0.0 1.0 1.0"/>
     </material>
     <material name="red">
       <color rgba="1.0 0.0 0.0 1.0"/>
     </material>
     <material name="green">
       <color rgba="0.0 1.0 0.0 1.0"/>
     </material>
     <material name="grey">
       <color rgba="0.5 0.5 0.5 1.0"/>
     </material>

     <!-- Macro for creating limbs -->
     <xacro:macro name="limb" params="side type parent xyz_rpy axis joint_limits mass">
       <xacro:if value="${type == 'arm'}">
         <!-- Upper limb -->
         <joint name="${side}_${type}_shoulder_joint" type="revolute">
           <parent link="${parent}"/>
           <child link="${side}_upper_${type}"/>
           <origin xyz="${xyz_rpy}"/>
           <axis xyz="${axis}"/>
           <xacro:insert_block name="joint_limits"/>
           <dynamics damping="0.5" friction="0.1"/>
         </joint>

         <link name="${side}_upper_${type}">
           <visual>
             <geometry>
               <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
             </geometry>
             <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
             <material name="grey"/>
           </visual>
           <collision>
             <geometry>
               <cylinder radius="${upper_arm_radius}" length="${upper_arm_length}"/>
             </geometry>
             <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
           </collision>
           <inertial>
             <mass value="${mass}"/>
             <origin xyz="0 0 ${-upper_arm_length/2}" rpy="0 0 0"/>
             <inertia ixx="${mass * (3*upper_arm_radius*upper_arm_radius + upper_arm_length*upper_arm_length) / 12}"
                      ixy="0" ixz="0"
                      iyy="${mass * (3*upper_arm_radius*upper_arm_radius + upper_arm_length*upper_arm_length) / 12}"
                      iyz="0"
                      izz="${mass * upper_arm_radius * upper_arm_radius / 2}"/>
           </inertial>
         </link>

         <!-- Lower limb -->
         <joint name="${side}_${type}_elbow_joint" type="revolute">
           <parent link="${side}_upper_${type}"/>
           <child link="${side}_lower_${type}"/>
           <origin xyz="0 0 ${-upper_arm_length}" rpy="0 0 0"/>
           <axis xyz="0 0 1"/>
           <limit lower="-2.0" upper="1.5" effort="50" velocity="2"/>
           <dynamics damping="0.5" friction="0.1"/>
         </joint>

         <link name="${side}_lower_${type}">
           <visual>
             <geometry>
               <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
             </geometry>
             <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
             <material name="grey"/>
           </visual>
           <collision>
             <geometry>
               <cylinder radius="${lower_arm_radius}" length="${lower_arm_length}"/>
             </geometry>
             <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
           </collision>
           <inertial>
             <mass value="${mass*0.8}"/>
             <origin xyz="0 0 ${-lower_arm_length/2}" rpy="0 0 0"/>
             <inertia ixx="${mass*0.8 * (3*lower_arm_radius*lower_arm_radius + lower_arm_length*lower_arm_length) / 12}"
                      ixy="0" ixz="0"
                      iyy="${mass*0.8 * (3*lower_arm_radius*lower_arm_radius + lower_arm_length*lower_arm_length) / 12}"
                      iyz="0"
                      izz="${mass*0.8 * lower_arm_radius * lower_arm_radius / 2}"/>
           </inertial>
         </link>
       </xacro:if>
     </xacro:macro>

     <!-- Base link (torso) -->
     <link name="base_link">
       <visual>
         <geometry>
           <box size="${torso_width} ${torso_depth} ${torso_height}"/>
         </geometry>
         <material name="white"/>
       </visual>
       <collision>
         <geometry>
           <box size="${torso_width} ${torso_depth} ${torso_height}"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="20.0"/>
         <origin xyz="0 0 ${torso_height/2}" rpy="0 0 0"/>
         <inertia ixx="1.5" ixy="0.0" ixz="0.0" iyy="1.5" iyz="0.0" izz="0.8"/>
       </inertial>
     </link>

     <!-- Head -->
     <joint name="neck_joint" type="revolute">
       <parent link="base_link"/>
       <child link="head"/>
       <origin xyz="0 0 ${torso_height}" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.7" upper="0.7" effort="20" velocity="1"/>
       <dynamics damping="0.5" friction="0.1"/>
     </joint>

     <link name="head">
       <visual>
         <geometry>
           <sphere radius="${head_radius}"/>
         </geometry>
         <material name="white"/>
       </visual>
       <collision>
         <geometry>
           <sphere radius="${head_radius}"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="3.0"/>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
       </inertial>
     </link>

     <!-- Camera mount on head -->
     <joint name="camera_mount_joint" type="fixed">
       <parent link="head"/>
       <child link="camera_link"/>
       <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
     </joint>

     <link name="camera_link">
       <visual>
         <geometry>
           <box size="0.05 0.05 0.03"/>
         </geometry>
         <material name="black"/>
       </visual>
       <collision>
         <geometry>
           <box size="0.05 0.05 0.03"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.1"/>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
       </inertial>
     </link>

     <!-- Arms using macro -->
     <xacro:limb side="left" type="arm"
                parent="base_link"
                xyz_rpy="0.15 0.15 0.4"
                axis="0 1 0"
                joint_limits="<limit lower='-1.5' upper='1.5' effort='100' velocity='2'/>"
                mass="2.0"/>

     <xacro:limb side="right" type="arm"
                parent="base_link"
                xyz_rpy="0.15 -0.15 0.4"
                axis="0 1 0"
                joint_limits="<limit lower='-1.5' upper='1.5' effort='100' velocity='2'/>"
                mass="2.0"/>

     <!-- Legs -->
     <joint name="left_hip_joint" type="revolute">
       <parent link="base_link"/>
       <child link="left_upper_leg"/>
       <origin xyz="-0.1 0.1 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.0" upper="1.0" effort="200" velocity="1"/>
       <dynamics damping="1.0" friction="0.2"/>
     </joint>

     <link name="left_upper_leg">
       <visual>
         <geometry>
           <cylinder radius="0.08" length="${upper_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
         <material name="blue"/>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.08" length="${upper_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
       </collision>
       <inertial>
         <mass value="5.0"/>
         <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
         <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <joint name="left_knee_joint" type="revolute">
       <parent link="left_upper_leg"/>
       <child link="left_lower_leg"/>
       <origin xyz="0 0 ${-upper_leg_length}" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="0" upper="1.5" effort="200" velocity="1"/>
       <dynamics damping="1.0" friction="0.2"/>
     </joint>

     <link name="left_lower_leg">
       <visual>
         <geometry>
           <cylinder radius="0.07" length="${lower_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
         <material name="blue"/>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.07" length="${lower_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
       </collision>
       <inertial>
         <mass value="4.0"/>
         <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
         <inertia ixx="0.15" ixy="0.0" ixz="0.0" iyy="0.15" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <joint name="left_ankle_joint" type="revolute">
       <parent link="left_lower_leg"/>
       <child link="left_foot"/>
       <origin xyz="0 0 ${-lower_leg_length}" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.5" upper="0.5" effort="150" velocity="1"/>
       <dynamics damping="0.5" friction="0.1"/>
     </joint>

     <link name="left_foot">
       <visual>
         <geometry>
           <box size="${foot_length} ${foot_width} ${foot_height}"/>
         </geometry>
         <material name="black"/>
       </visual>
       <collision>
         <geometry>
           <box size="${foot_length} ${foot_width} ${foot_height}"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.5"/>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
       </inertial>
     </link>

     <!-- Right leg (similar to left) -->
     <joint name="right_hip_joint" type="revolute">
       <parent link="base_link"/>
       <child link="right_upper_leg"/>
       <origin xyz="-0.1 -0.1 0" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-1.0" upper="1.0" effort="200" velocity="1"/>
       <dynamics damping="1.0" friction="0.2"/>
     </joint>

     <link name="right_upper_leg">
       <visual>
         <geometry>
           <cylinder radius="0.08" length="${upper_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
         <material name="blue"/>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.08" length="${upper_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
       </collision>
       <inertial>
         <mass value="5.0"/>
         <origin xyz="0 0 ${-upper_leg_length/2}" rpy="0 0 0"/>
         <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <joint name="right_knee_joint" type="revolute">
       <parent link="right_upper_leg"/>
       <child link="right_lower_leg"/>
       <origin xyz="0 0 ${-upper_leg_length}" rpy="0 0 0"/>
       <axis xyz="0 0 1"/>
       <limit lower="0" upper="1.5" effort="200" velocity="1"/>
       <dynamics damping="1.0" friction="0.2"/>
     </joint>

     <link name="right_lower_leg">
       <visual>
         <geometry>
           <cylinder radius="0.07" length="${lower_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
         <material name="blue"/>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.07" length="${lower_leg_length}"/>
         </geometry>
         <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
       </collision>
       <inertial>
         <mass value="4.0"/>
         <origin xyz="0 0 ${-lower_leg_length/2}" rpy="0 0 0"/>
         <inertia ixx="0.15" ixy="0.0" ixz="0.0" iyy="0.15" iyz="0.0" izz="0.01"/>
       </inertial>
     </link>

     <joint name="right_ankle_joint" type="revolute">
       <parent link="right_lower_leg"/>
       <child link="right_foot"/>
       <origin xyz="0 0 ${-lower_leg_length}" rpy="0 0 0"/>
       <axis xyz="0 1 0"/>
       <limit lower="-0.5" upper="0.5" effort="150" velocity="1"/>
       <dynamics damping="0.5" friction="0.1"/>
     </joint>

     <link name="right_foot">
       <visual>
         <geometry>
           <box size="${foot_length} ${foot_width} ${foot_height}"/>
         </geometry>
         <material name="black"/>
       </visual>
       <collision>
         <geometry>
           <box size="${foot_length} ${foot_width} ${foot_height}"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1.5"/>
         <origin xyz="0 0 0" rpy="0 0 0"/>
         <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
       </inertial>
     </link>

     <!-- ROS Control interface -->
     <gazebo>
       <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
         <robotNamespace>/</robotNamespace>
       </plugin>
     </gazebo>

   </robot>
   ```

2. **Create a URDF loader node:**
   ```python
   #!/usr/bin/env python3
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   import xacro
   import os

   class URDFLoaderNode(Node):
       def __init__(self):
           super().__init__('urdf_loader')

           # Publisher for robot description
           self.urdf_publisher = self.create_publisher(
               String, 'robot_description', 1)

           # Load and publish URDF
           self.load_and_publish_urdf()

           self.get_logger().info('URDF loader initialized')

       def load_and_publish_urdf(self):
           """Load URDF from Xacro and publish"""
           try:
               # Get the path to the URDF file
               urdf_path = os.path.join(
                   os.path.dirname(__file__),
                   '..', '..', 'urdf', 'advanced_humanoid.urdf.xacro'
               )

               # Process Xacro to URDF
               urdf_content = xacro.process_file(urdf_path).toprettyxml(indent='  ')

               # Publish URDF
               msg = String()
               msg.data = urdf_content
               self.urdf_publisher.publish(msg)

               self.get_logger().info('URDF published successfully')

           except Exception as e:
               self.get_logger().error(f'Failed to load URDF: {e}')
               # Create a simple fallback URDF
               fallback_urdf = """<?xml version="1.0"?>
               <robot name="fallback_robot">
                 <link name="base_link">
                   <visual>
                     <geometry>
                       <box size="0.5 0.5 0.5"/>
                     </geometry>
                   </visual>
                 </link>
               </robot>"""

               msg = String()
               msg.data = fallback_urdf
               self.urdf_publisher.publish(msg)

   def main(args=None):
       rclpy.init(args=args)
       node = URDFLoaderNode()

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

3. **Validate and test the URDF:**
   ```bash
   # Validate URDF syntax
   check_urdf ~/physical_ai_ws/src/week4_exercises/urdf/advanced_humanoid.urdf.xacro

   # Generate graph
   urdf_to_graphiz ~/physical_ai_ws/src/week4_exercises/urdf/advanced_humanoid.urdf.xacro

   # Visualize in RViz2
   ros2 run rviz2 rviz2
   # Then add RobotModel display and set topic to robot_description
   ```

### Expected Output
- Complete humanoid URDF with 20+ joints
- Proper visual, collision, and inertial properties
- Parameterized using Xacro
- Valid URDF that can be visualized in RViz2

### Submission Requirements
- Complete URDF file with Xacro
- URDF loader node implementation
- Validation results
- Screenshots of URDF visualization in RViz2

## Grading Rubric

Each exercise will be graded on the following criteria:

- **Implementation Correctness** (30%): Code works as specified
- **Code Quality** (25%): Well-structured, documented, follows ROS 2 best practices
- **Understanding** (25%): Proper understanding of concepts demonstrated
- **Testing** (20%): Adequate testing and validation performed

## Submission Guidelines

- Submit all exercises as a complete ROS 2 package
- Include a README.md explaining your implementation
- Provide screenshots of successful execution
- Follow proper ROS 2 package structure and conventions
- Late submissions will be penalized by 10% per day

## Resources

- [ROS 2 Parameters Guide](https://docs.ros.org/en/humble/How-To-Guides/Using-Parameters-In-A-Class-Python.html)
- [ROS 2 Lifecycle Nodes](https://docs.ros.org/en/humble/Tutorials/Managed-Nodes.html)
- [URDF Tutorials](https://docs.ros.org/en/humble/Tutorials/URDF/Building-a-Visual-Robot-Model-with-URDF.html)
- [Xacro Documentation](https://wiki.ros.org/xacro)