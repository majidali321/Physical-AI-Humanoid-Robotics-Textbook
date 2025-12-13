---
sidebar_position: 4
---

# Week 10 Exercises: AI Decision Making and Bipedal Movement

## Exercise 1: Cognitive Path Planning Implementation

### Objective
Implement a cognitive path planning algorithm that considers multiple factors for humanoid navigation.

### Tasks
1. Create a path planning system that evaluates paths based on:
   - Path length (efficiency)
   - Safety (distance from obstacles)
   - Energy consumption (smoothness of path)
2. Implement a weighted scoring system to select optimal paths
3. Test the system with different environment configurations

### Code Template
```python
import numpy as np

class CognitivePathPlanner:
    def __init__(self):
        self.weights = {
            'efficiency': 0.4,
            'safety': 0.3,
            'energy': 0.3
        }

    def evaluate_path(self, path, environment_map):
        """
        Evaluate a path based on multiple cognitive factors
        """
        # TODO: Implement path evaluation logic
        # Calculate efficiency score (shorter paths score higher)
        # Calculate safety score (paths further from obstacles score higher)
        # Calculate energy score (smoother paths score higher)
        # Return weighted sum of all scores
        pass

    def plan_path(self, start, goal, environment_map):
        """
        Plan path considering cognitive factors
        """
        # TODO: Generate multiple path options
        # Evaluate each path using evaluate_path method
        # Return the path with highest score
        pass

# Test the implementation
def test_cognitive_planner():
    planner = CognitivePathPlanner()

    # Define test environment (simplified)
    start = (0, 0)
    goal = (10, 10)
    environment_map = {
        'obstacles': [(5, 5), (6, 6), (7, 7)],  # Obstacle positions
        'traversable': True
    }

    # TODO: Implement test and verify the planner works correctly
    pass
```

### Expected Output
- Path planning algorithm that considers multiple factors
- Working evaluation function with appropriate scoring
- Test results showing path selection based on different priorities

### Solution Guidelines
- Use geometric calculations for path length and obstacle distances
- Implement smoothness evaluation using angle changes between path segments
- Test with different weight configurations to see how priorities affect path selection

---

## Exercise 2: Balance Control System

### Objective
Implement a balance control system for bipedal locomotion that maintains stability during movement.

### Tasks
1. Create a Center of Mass (CoM) tracking and control system
2. Implement stability checking based on Zero Moment Point (ZMP) or support polygon
3. Design a feedback control system to maintain balance

### Code Template
```python
import numpy as np

class BalanceController:
    def __init__(self):
        self.com_position = np.array([0.0, 0.0, 0.8])  # Initial CoM at 0.8m height
        self.com_velocity = np.array([0.0, 0.0, 0.0])
        self.com_acceleration = np.array([0.0, 0.0, 0.0])
        self.support_polygon = []  # Define support polygon based on foot positions
        self.gravity = 9.81
        self.control_gain = 10.0  # Adjust as needed

    def update_balance(self, dt, external_forces=np.array([0.0, 0.0, 0.0])):
        """
        Update balance state based on dynamics and external forces
        """
        # TODO: Update CoM dynamics using Newton's laws
        # F = ma, so a = F/m (simplified assuming unit mass for control purposes)
        # Update velocity and position using Euler integration
        pass

    def check_stability(self):
        """
        Check if the robot is stable based on CoM position
        """
        # TODO: Implement stability check
        # For ZMP: check if ZMP is within support polygon
        # For simplified: check if CoM projection is within support polygon
        pass

    def generate_balance_control(self):
        """
        Generate control commands to maintain balance
        """
        # TODO: Implement feedback control to move CoM toward stable position
        # Return control commands for joint angles or foot placement
        pass

# Test the balance controller
def test_balance_controller():
    controller = BalanceController()

    # Simulate some external disturbance
    disturbance = np.array([0.5, 0.0, 0.0])  # Force pushing robot to the right

    # TODO: Run simulation loop with balance control
    # Apply disturbance and verify balance recovery
    pass
```

### Expected Output
- Working balance control system that maintains CoM within stable region
- Stability checking function that correctly identifies balance state
- Control commands that actively maintain balance during disturbances

### Solution Guidelines
- Use simple 2D projection of CoM for basic implementation
- Implement PID or similar control for CoM tracking
- Consider the relationship between foot placement and balance

---

## Exercise 3: AI Decision Making for Navigation

### Objective
Create an AI system that makes navigation decisions based on environmental conditions and robot state.

### Tasks
1. Implement a decision tree or rule-based system for navigation choices
2. Integrate sensor data processing for situation assessment
3. Create action selection based on current context

### Code Template
```python
import numpy as np

class NavigationDecisionMaker:
    def __init__(self):
        self.robot_state = "idle"
        self.environment_state = {}
        self.decision_history = []

    def assess_situation(self, sensor_data):
        """
        Assess current situation from sensor data
        """
        # TODO: Process sensor data to determine:
        # - Obstacle locations and types
        # - Goal distance and direction
        # - Robot energy/battery level
        # - Environmental conditions
        situation = {
            'obstacles': self.process_obstacles(sensor_data.get('laser_scan', [])),
            'goal_direction': self.calculate_goal_direction(sensor_data.get('goal', (0,0))),
            'robot_state': self.robot_state,
            'environment': sensor_data.get('environment', {})
        }
        return situation

    def process_obstacles(self, laser_scan):
        """Process laser scan data to identify obstacles"""
        # TODO: Analyze laser scan to find obstacles
        # Return structured obstacle information
        obstacles = []
        return obstacles

    def calculate_goal_direction(self, goal_position):
        """Calculate direction to goal"""
        # TODO: Calculate direction vector to goal
        return np.array([0.0, 0.0])

    def make_decision(self, situation):
        """
        Make navigation decision based on situation
        """
        # TODO: Implement decision logic
        # Consider: obstacle avoidance, goal reaching, energy conservation
        # Return action: move_forward, turn_left, turn_right, wait, etc.
        if situation['obstacles']:
            # Handle obstacle situation
            return self.handle_obstacles(situation)
        else:
            # Move toward goal
            return "move_toward_goal"

    def handle_obstacles(self, situation):
        """Handle obstacle avoidance decision"""
        # TODO: Implement obstacle avoidance logic
        # Consider: obstacle size, location, alternative paths
        return "avoid_obstacle"

    def execute_decision(self, decision):
        """
        Execute the decision by generating control commands
        """
        # TODO: Convert decision to specific control commands
        # Return velocity commands or other robot controls
        if decision == "move_toward_goal":
            return {'linear': 0.3, 'angular': 0.0}
        elif decision == "avoid_obstacle":
            return {'linear': 0.0, 'angular': 0.5}
        else:
            return {'linear': 0.0, 'angular': 0.0}

# Test the decision maker
def test_decision_maker():
    decision_maker = NavigationDecisionMaker()

    # Simulate sensor data
    sensor_data = {
        'laser_scan': [1.0, 1.0, 0.5, 0.3, 0.5, 1.0, 1.0],  # Simplified scan
        'goal': (5.0, 5.0),
        'environment': {'lighting': 'good', 'terrain': 'flat'}
    }

    # TODO: Test the complete decision-making pipeline
    situation = decision_maker.assess_situation(sensor_data)
    decision = decision_maker.make_decision(situation)
    commands = decision_maker.execute_decision(decision)

    print(f"Situation: {situation}")
    print(f"Decision: {decision}")
    print(f"Commands: {commands}")
```

### Expected Output
- Working situation assessment system
- Decision-making logic that responds appropriately to different situations
- Action execution that produces valid control commands

### Solution Guidelines
- Use state machines or rule-based systems for decision making
- Consider multiple factors in decision process
- Implement fallback behaviors for uncertain situations

---

## Exercise 4: Perception-Action Integration

### Objective
Create a system that integrates perception data with action execution for humanoid navigation.

### Tasks
1. Implement sensor data fusion from multiple sources
2. Create an action selection mechanism based on integrated perception
3. Implement execution monitoring and feedback

### Code Template
```python
import numpy as np

class PerceptionActionSystem:
    def __init__(self):
        self.perception_buffer = {}
        self.action_queue = []
        self.execution_monitor = {}

    def integrate_sensors(self, sensor_inputs):
        """
        Integrate data from multiple sensors
        """
        # TODO: Fuse data from:
        # - LIDAR for obstacle detection
        # - Camera for visual information
        # - IMU for balance and orientation
        # - Joint encoders for position
        integrated_perception = {
            'obstacles': self.fuse_obstacle_data(sensor_inputs),
            'environment': self.fuse_environment_data(sensor_inputs),
            'robot_state': self.fuse_robot_state(sensor_inputs)
        }
        return integrated_perception

    def fuse_obstacle_data(self, sensor_inputs):
        """Fuse obstacle data from multiple sensors"""
        # TODO: Combine LIDAR, camera, and other obstacle data
        obstacles = []
        return obstacles

    def fuse_environment_data(self, sensor_inputs):
        """Fuse environmental data"""
        # TODO: Combine visual, tactile, and other environmental data
        environment = {}
        return environment

    def fuse_robot_state(self, sensor_inputs):
        """Fuse robot state data"""
        # TODO: Combine pose, velocity, and internal state data
        robot_state = {}
        return robot_state

    def select_action(self, integrated_perception):
        """
        Select appropriate action based on integrated perception
        """
        # TODO: Choose action based on:
        # - Current goals
        # - Environmental conditions
        # - Robot capabilities and state
        # - Safety constraints
        return "selected_action"

    def execute_with_monitoring(self, action):
        """
        Execute action with monitoring for success/failure
        """
        # TODO: Execute action and monitor:
        # - Execution progress
        # - Success criteria
        # - Failure detection
        # - Need for plan adjustment
        result = {
            'success': True,
            'progress': 1.0,
            'adjustments_needed': False
        }
        return result

# Test the perception-action system
def test_perception_action():
    system = PerceptionActionSystem()

    # Simulate sensor inputs
    sensor_inputs = {
        'laser': [1.0, 0.8, 0.3, 0.2, 0.3, 0.8, 1.0],
        'camera': {'image': np.random.rand(480, 640, 3)},  # Simulated image
        'imu': {'orientation': [0.0, 0.0, 0.0], 'acceleration': [0.0, 0.0, 9.81]},
        'joints': {'positions': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    }

    # TODO: Test the complete perception-action pipeline
    integrated = system.integrate_sensors(sensor_inputs)
    action = system.select_action(integrated)
    result = system.execute_with_monitoring(action)

    print(f"Integrated perception: {integrated}")
    print(f"Selected action: {action}")
    print(f"Execution result: {result}")
```

### Expected Output
- Working sensor fusion system
- Action selection based on integrated perception
- Execution monitoring with feedback capabilities

### Solution Guidelines
- Implement data association to match sensor readings
- Use confidence measures for different sensor modalities
- Implement feedback loops for continuous improvement

---

## Exercise 5: Project Integration Challenge

### Objective
Combine all the components from previous exercises into a complete navigation system.

### Tasks
1. Integrate cognitive path planning with balance control
2. Combine AI decision making with perception-action integration
3. Create a complete navigation pipeline
4. Test the integrated system in simulation

### Implementation Requirements
- Use the classes and functions developed in previous exercises
- Create a main control loop that coordinates all components
- Implement state management for the complete system
- Add error handling and recovery mechanisms

### Testing Scenario
Create a test scenario where the humanoid robot must navigate through a complex environment with:
- Static obstacles
- Dynamic obstacles (moving objects)
- Terrain variations
- Balance challenges (narrow passages, slopes)

### Evaluation Criteria
- **Integration Quality**: How well components work together
- **System Performance**: Navigation efficiency and safety
- **Robustness**: Handling of unexpected situations
- **Innovation**: Creative solutions to integration challenges

### Solution Approach
1. Create a main navigation node that orchestrates all components
2. Implement a state machine to manage navigation phases
3. Add monitoring and logging for system debugging
4. Test with increasingly complex scenarios

This exercise combines all the concepts learned in Module 3, requiring you to create a complete, integrated system for humanoid robot navigation with AI decision making capabilities.