---
sidebar_position: 2
---

# Week 12 Exercises: LLM Integration and Cognitive Planning

## Exercise 1: Basic LLM Integration for Robot Planning

### Objective
Implement basic LLM integration to generate simple robot action plans from natural language commands.

### Tasks
1. Set up OpenAI API integration for robotic planning
2. Create a simple prompt engineering system for robot commands
3. Implement JSON response parsing for action sequences
4. Test with basic navigation and manipulation commands

### Code Template
```python
import openai
import json
from typing import List, Dict, Any
import asyncio

class BasicLLMPlanner:
    def __init__(self, api_key: str):
        """
        Initialize the LLM planner with API key
        """
        # TODO: Set the OpenAI API key
        openai.api_key = api_key

    async def generate_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Generate a plan for the given goal using LLM
        """
        # TODO: Create a prompt that asks the LLM to generate robot actions
        # The prompt should:
        # 1. Describe the robot's capabilities
        # 2. Specify the goal
        # 3. Request a JSON response with actions
        # 4. Include action types like "navigate", "grasp", "speak", etc.

        prompt = f"""
        You are a planning system for a humanoid robot. Generate a sequence of actions to achieve this goal:
        Goal: {goal}

        Robot capabilities: navigate, grasp, speak, detect_objects, place_object
        Available locations: kitchen, living_room, bedroom, entrance

        Return a JSON array of actions. Each action should have:
        - "action": the action type
        - "parameters": required parameters for the action
        - "description": human-readable description

        Example response:
        [
          {{
            "action": "navigate",
            "parameters": {{"location": "kitchen"}},
            "description": "Move to the kitchen"
          }},
          {{
            "action": "detect_objects",
            "parameters": {{"object_type": "cup"}},
            "description": "Look for a cup"
          }}
        ]
        """

        # TODO: Call the OpenAI API asynchronously
        # Use gpt-3.5-turbo or gpt-4 model
        # Request JSON response format
        # Handle potential API errors
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",  # Use appropriate model
                messages=[
                    {"role": "system", "content": "You are a robot planning system. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=500,
                response_format={"type": "json_object"}  # Request JSON response
            )

            # TODO: Parse the response and return the action list
            response_text = response.choices[0].message.content
            plan = json.loads(response_text)
            return plan.get("actions", [])

        except Exception as e:
            print(f"Error generating plan: {e}")
            return []

    def validate_plan(self, plan: List[Dict[str, Any]]) -> bool:
        """
        Validate that the plan contains valid actions
        """
        # TODO: Implement plan validation
        # Check that each action has required fields
        # Verify action types are supported
        # Return True if valid, False otherwise
        for action in plan:
            if not isinstance(action, dict):
                return False
            if "action" not in action or "parameters" not in action:
                return False
            # Add more validation as needed
        return True

# Test the implementation
async def test_basic_planner():
    # TODO: Initialize planner with API key
    # You'll need to provide a valid OpenAI API key
    # For testing purposes, you can use a placeholder
    planner = BasicLLMPlanner("your-api-key-here")  # Replace with real key

    test_goals = [
        "Go to the kitchen and bring me a cup",
        "Find my keys in the living room",
        "Navigate to the bedroom"
    ]

    # TODO: Test each goal and print results
    for goal in test_goals:
        print(f"\nGoal: {goal}")
        plan = await planner.generate_plan(goal)
        print(f"Generated plan: {plan}")

        # Validate the plan
        is_valid = planner.validate_plan(plan)
        print(f"Plan valid: {is_valid}")

# Uncomment to test when you have an API key
# asyncio.run(test_basic_planner())
```

### Expected Output
- Working OpenAI API integration
- Valid JSON responses with robot action sequences
- Proper error handling for API calls
- Plan validation functionality

### Solution Guidelines
- Use appropriate temperature settings (0.3-0.5) for consistent planning
- Always request JSON response format for reliable parsing
- Implement proper error handling for API failures
- Validate responses before using them for robot control

---

## Exercise 2: Context-Aware Planning System

### Objective
Create a context-aware planning system that considers robot state and environment when generating plans.

### Tasks
1. Implement robot state tracking
2. Create environment state representation
3. Include context in LLM prompts for better planning
4. Implement plan adaptation based on changing conditions

### Code Template
```python
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio

@dataclass
class RobotState:
    """
    Represents the current state of the robot
    """
    # TODO: Define robot state attributes
    # Consider position, battery, carrying object, etc.
    position: Dict[str, float] = None
    battery_level: float = 1.0
    carrying_object: str = None
    available_actions: List[str] = None

@dataclass
class EnvironmentState:
    """
    Represents the current environment state
    """
    # TODO: Define environment state attributes
    # Consider objects, locations, obstacles, etc.
    objects: List[Dict[str, Any]] = None
    locations: List[Dict[str, Any]] = None
    obstacles: List[Dict[str, Any]] = None

class ContextAwarePlanner:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.robot_state = RobotState(
            position={"x": 0.0, "y": 0.0},
            battery_level=1.0,
            carrying_object=None,
            available_actions=["navigate", "grasp", "speak", "detect"]
        )
        self.environment_state = EnvironmentState(
            objects=[],
            locations=[],
            obstacles=[]
        )

    def update_robot_state(self, new_state: RobotState):
        """
        Update the robot's current state
        """
        # TODO: Update robot state with new information
        self.robot_state = new_state

    def update_environment_state(self, new_env: EnvironmentState):
        """
        Update the environment state
        """
        # TODO: Update environment state with new information
        self.environment_state = new_env

    async def generate_context_aware_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Generate a plan considering current robot and environment state
        """
        # TODO: Create a comprehensive prompt that includes context
        # Include robot state, environment state, and the goal
        # Ask the LLM to consider constraints and capabilities
        context_prompt = f"""
        Robot State:
        - Position: {self.robot_state.position}
        - Battery: {self.robot_state.battery_level:.1%}
        - Carrying: {self.robot_state.carrying_object or 'nothing'}
        - Available actions: {self.robot_state.available_actions}

        Environment State:
        - Objects: {self.environment_state.objects}
        - Locations: {self.environment_state.locations}
        - Obstacles: {self.environment_state.obstacles}

        Goal: {goal}

        Generate a plan that considers the robot's current state and environment.
        Take into account battery level, what the robot is carrying, and environmental constraints.
        Return JSON with action sequence.
        """

        # TODO: Call OpenAI API with context prompt
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a context-aware robot planning system."},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.3,
                max_tokens=600,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content
            plan = json.loads(response_text)
            return plan.get("actions", [])

        except Exception as e:
            print(f"Error generating context-aware plan: {e}")
            return []

    def adapt_plan_to_context(self, original_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adapt the original plan based on current context
        """
        # TODO: Implement plan adaptation logic
        # Consider:
        # - Current battery level (avoid long navigation if battery is low)
        # - What robot is currently carrying (plan accordingly)
        # - Environmental obstacles (adjust navigation)
        # Return adapted plan
        adapted_plan = []

        for action in original_plan:
            # Example adaptation: if battery is low, avoid complex actions
            if self.robot_state.battery_level < 0.2 and action["action"] == "navigate":
                # Modify navigation to go to charging station first
                if "charging_station" in [loc["name"] for loc in self.environment_state.locations]:
                    adapted_plan.append({
                        "action": "navigate",
                        "parameters": {"location": "charging_station"},
                        "description": "Go to charging station first (low battery)"
                    })

            adapted_plan.append(action)

        return adapted_plan

# Test the context-aware planner
async def test_context_aware_planner():
    # TODO: Create test scenarios with different contexts
    planner = ContextAwarePlanner("your-api-key-here")  # Replace with real key

    # Test with low battery context
    low_battery_state = RobotState(
        position={"x": 5.0, "y": 5.0},
        battery_level=0.15,  # Low battery
        carrying_object="cup",
        available_actions=["navigate", "grasp", "speak"]
    )

    planner.update_robot_state(low_battery_state)

    environment_with_charger = EnvironmentState(
        objects=[{"name": "cup", "location": "kitchen"}],
        locations=[
            {"name": "kitchen", "x": 10.0, "y": 0.0},
            {"name": "charging_station", "x": 0.0, "y": 0.0}
        ],
        obstacles=[]
    )

    planner.update_environment_state(environment_with_charger)

    goal = "Go to kitchen and pick up the cup"
    print(f"Goal: {goal}")
    print(f"Robot battery: {planner.robot_state.battery_level:.1%}")

    # Generate plan considering context
    plan = await planner.generate_context_aware_plan(goal)
    print(f"Original plan: {plan}")

    # Adapt plan based on context
    adapted_plan = planner.adapt_plan_to_context(plan)
    print(f"Adapted plan: {adapted_plan}")

# Uncomment to test when ready
# asyncio.run(test_context_aware_planner())
```

### Expected Output
- Context-aware plan generation considering robot state
- Environment state integration in planning
- Plan adaptation based on changing conditions
- Proper handling of constraints (battery, carrying, etc.)

### Solution Guidelines
- Include comprehensive context information in prompts
- Implement intelligent plan adaptation logic
- Consider multiple constraints simultaneously
- Maintain plan coherence during adaptation

---

## Exercise 3: Multi-Step Task Decomposition

### Objective
Implement LLM-based task decomposition that breaks complex goals into manageable subtasks.

### Tasks
1. Create a task decomposition system using LLM reasoning
2. Implement subtask dependency management
3. Generate executable action sequences from subtasks
4. Handle task failures and recovery

### Code Template
```python
import asyncio
from typing import List, Dict, Any, Optional

class TaskDecomposer:
    def __init__(self, api_key: str):
        openai.api_key = api_key

    async def decompose_task(self, high_level_goal: str) -> Dict[str, Any]:
        """
        Decompose a high-level goal into subtasks using LLM
        """
        # TODO: Create prompt for task decomposition
        # Include information about robot capabilities
        # Request hierarchical task structure with dependencies
        decomposition_prompt = f"""
        Decompose the following high-level goal into specific, executable subtasks:
        Goal: {high_level_goal}

        Robot capabilities: navigate, grasp, speak, detect_objects, place_object, open_door, close_door

        Return a JSON object with:
        - "main_goal": the original goal
        - "subtasks": array of subtasks, each with:
          - "id": unique identifier
          - "description": what needs to be done
          - "type": task type (navigation, manipulation, perception, etc.)
          - "dependencies": list of subtask IDs that must be completed first
          - "estimated_duration": in seconds
          - "success_criteria": how to determine if task is complete

        Example format:
        {{
          "main_goal": "{high_level_goal}",
          "subtasks": [
            {{
              "id": 1,
              "description": "Navigate to kitchen",
              "type": "navigation",
              "dependencies": [],
              "estimated_duration": 30,
              "success_criteria": "Robot is in kitchen area"
            }},
            {{
              "id": 2,
              "description": "Detect cup",
              "type": "perception",
              "dependencies": [1],
              "estimated_duration": 10,
              "success_criteria": "Cup is located and positioned"
            }}
          ]
        }}
        """

        # TODO: Call OpenAI API for task decomposition
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",  # Using GPT-4 for better reasoning
                messages=[
                    {"role": "system", "content": "You are a task decomposition expert for robotic systems."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0.2,  # Lower temperature for more structured output
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content
            decomposition = json.loads(response_text)
            return decomposition

        except Exception as e:
            print(f"Error decomposing task: {e}")
            return {"main_goal": high_level_goal, "subtasks": []}

    def resolve_dependencies(self, subtasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Resolve task dependencies and return execution order
        Returns list of task batches that can be executed in parallel
        """
        # TODO: Implement dependency resolution
        # Create execution batches where tasks in each batch have no dependencies
        # on each other but depend on previous batches
        # Return list of task batches
        task_dict = {task["id"]: task for task in subtasks}
        remaining_tasks = set(task["id"] for task in subtasks)
        batches = []

        while remaining_tasks:
            batch = []
            for task_id in list(remaining_tasks):
                task = task_dict[task_id]
                dependencies = set(task.get("dependencies", []))

                # Check if all dependencies are satisfied
                if dependencies.issubset(set(batch_task["id"] for batch_tasks in batches for batch_task in batch_tasks)):
                    batch.append(task)

            if not batch:
                # No progress made - possible circular dependency
                print("Warning: Could not resolve dependencies, possible circular dependency")
                break

            # Remove processed tasks from remaining
            for task in batch:
                remaining_tasks.remove(task["id"])

            batches.append(batch)

        return batches

    async def convert_subtasks_to_actions(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert subtasks to executable robot actions
        """
        # TODO: For each subtask, generate specific robot actions
        # This could involve another LLM call or rule-based conversion
        all_actions = []

        for subtask in subtasks:
            # Generate actions based on subtask type
            if subtask["type"] == "navigation":
                actions = [
                    {
                        "action": "navigate",
                        "parameters": {"target": subtask.get("location", "unknown")},
                        "description": f"Navigate to location for: {subtask['description']}"
                    }
                ]
            elif subtask["type"] == "manipulation":
                actions = [
                    {
                        "action": "grasp",
                        "parameters": {"object": subtask.get("object", "unknown")},
                        "description": f"Grasp object for: {subtask['description']}"
                    }
                ]
            elif subtask["type"] == "perception":
                actions = [
                    {
                        "action": "detect_objects",
                        "parameters": {"object_type": subtask.get("object_type", "unknown")},
                        "description": f"Detect objects for: {subtask['description']}"
                    }
                ]
            else:
                # Default action for other types
                actions = [
                    {
                        "action": "wait",
                        "parameters": {"duration": 1},
                        "description": f"Wait action for: {subtask['description']}"
                    }
                ]

            all_actions.extend(actions)

        return all_actions

class TaskExecutionManager:
    def __init__(self):
        self.completed_tasks = set()
        self.failed_tasks = set()

    async def execute_task_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Execute a batch of tasks in parallel
        Returns dictionary with successful and failed task IDs
        """
        # TODO: Execute tasks in the batch
        # In a real system, this would call robot action services
        # For simulation, we'll randomly succeed/fail tasks
        results = {"successful": [], "failed": []}

        # Simulate parallel execution
        execution_tasks = []
        for subtask in batch:
            execution_tasks.append(self._execute_single_subtask(subtask))

        results_list = await asyncio.gather(*execution_tasks, return_exceptions=True)

        for i, result in enumerate(results_list):
            task_id = batch[i]["id"]
            if result is True:
                results["successful"].append(task_id)
            else:
                results["failed"].append(task_id)

        return results

    async def _execute_single_subtask(self, subtask: Dict[str, Any]) -> bool:
        """
        Simulate execution of a single subtask
        """
        # TODO: In real implementation, this would interface with robot
        # For simulation, add some randomness to success/failure
        import random
        await asyncio.sleep(0.1)  # Simulate task duration
        return random.random() > 0.2  # 80% success rate for simulation

# Test the task decomposition system
async def test_task_decomposition():
    decomposer = TaskDecomposer("your-api-key-here")  # Replace with real key
    executor = TaskExecutionManager()

    complex_goal = "Go to the kitchen, find a red cup, pick it up, and bring it to the living room table"

    print(f"Decomposing complex goal: {complex_goal}")

    # Decompose the task
    decomposition = await decomposer.decompose_task(complex_goal)
    print(f"Decomposed into {len(decomposition['subtasks'])} subtasks")

    # Resolve dependencies
    batches = decomposer.resolve_dependencies(decomposition["subtasks"])
    print(f"Resolved into {len(batches)} execution batches")

    # Execute batches sequentially
    for i, batch in enumerate(batches):
        print(f"\nExecuting batch {i+1}/{len(batches)}:")
        for subtask in batch:
            print(f"  - {subtask['id']}: {subtask['description']} (type: {subtask['type']})")

        # Execute the batch
        results = await executor.execute_task_batch(batch)
        print(f"  Successful: {results['successful']}, Failed: {results['failed']}")

    print("\nTask decomposition and execution completed")

# Uncomment to test when ready
# asyncio.run(test_task_decomposition())
```

### Expected Output
- Working task decomposition with dependency resolution
- Proper handling of subtask dependencies
- Conversion of subtasks to executable actions
- Task execution management with failure handling

### Solution Guidelines
- Use GPT-4 or similar for better reasoning in decomposition
- Implement proper dependency resolution algorithms
- Consider task parallelization opportunities
- Handle failures gracefully with recovery options

---

## Exercise 4: LLM-ROS Integration Node

### Objective
Create a ROS 2 node that integrates LLM planning with robot execution.

### Tasks
1. Implement ROS 2 service for goal planning requests
2. Create action execution with feedback
3. Implement state management between planning and execution
4. Add monitoring and logging capabilities

### Code Template
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Define custom action message (simplified)
class PlanAndExecuteAction:
    """
    Simplified action definition for planning and execution
    In real implementation, define proper action interface
    """
    def __init__(self):
        self.goal = ""
        self.plan = []
        self.result = {"success": False, "message": ""}

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Publishers
        self.status_pub = self.create_publisher(String, '/llm_planning_status', 10)
        self.feedback_pub = self.create_publisher(String, '/llm_execution_feedback', 10)

        # Initialize LLM components
        api_key = self.declare_parameter('openai_api_key', '').value
        if not api_key:
            self.get_logger().error('OpenAI API key not provided')
            return

        self.decomposer = TaskDecomposer(api_key)
        self.executor_manager = TaskExecutionManager()

        # Robot state tracking
        self.robot_position = {"x": 0.0, "y": 0.0}
        self.battery_level = 1.0
        self.carrying_object = None

        # Async execution setup
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.loop = asyncio.new_event_loop()
        self.executor.submit(self._run_async_loop, self.loop)

        # Subscribe to robot state updates
        self.robot_state_sub = self.create_subscription(
            String, '/robot_state', self.robot_state_callback, 10)

        self.get_logger().info('LLM Planning Node initialized')

    def _run_async_loop(self, loop):
        """Run asyncio event loop in separate thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def robot_state_callback(self, msg):
        """Update robot state from ROS messages"""
        # TODO: Parse robot state message and update internal state
        # This would typically be a custom message type
        try:
            state_data = json.loads(msg.data)
            self.robot_position = state_data.get("position", self.robot_position)
            self.battery_level = state_data.get("battery", self.battery_level)
            self.carrying_object = state_data.get("carrying", self.carrying_object)
        except json.JSONDecodeError:
            self.get_logger().warn('Invalid robot state message format')

    async def plan_goal_async(self, goal_text: str) -> List[Dict[str, Any]]:
        """Async method to plan a goal using LLM"""
        # TODO: Update environment context with current robot state
        # Then call the decomposer to create a plan
        self.get_logger().info(f'Planning for goal: {goal_text}')

        # This would involve updating context with current state
        # and calling the LLM to generate a plan
        decomposition = await self.decomposer.decompose_task(goal_text)
        return decomposition.get("subtasks", [])

    def plan_and_execute_service(self, request, response):
        """
        Service callback for planning and executing a goal
        """
        goal = request.goal  # Assuming request has a 'goal' field
        self.get_logger().info(f'Received planning request: {goal}')

        # Run planning in async loop
        future = asyncio.run_coroutine_threadsafe(
            self.plan_goal_async(goal), self.loop)

        try:
            # Wait for planning to complete (with timeout)
            subtasks = future.result(timeout=10.0)  # 10 second timeout

            if subtasks:
                # Convert subtasks to executable actions
                actions = asyncio.run_coroutine_threadsafe(
                    self.convert_subtasks_to_actions(subtasks), self.loop).result(timeout=5.0)

                # Execute the plan
                execution_success = self.execute_plan(actions)

                response.success = execution_success
                response.message = "Plan executed successfully" if execution_success else "Execution failed"
            else:
                response.success = False
                response.message = "Failed to generate plan"

        except Exception as e:
            response.success = False
            response.message = f"Planning or execution failed: {str(e)}"

        return response

    def convert_subtasks_to_actions(self, subtasks):
        """Convert subtasks to executable actions (async wrapper)"""
        return self.decomposer.convert_subtasks_to_actions(subtasks)

    def execute_plan(self, actions: List[Dict[str, Any]]) -> bool:
        """
        Execute the plan with ROS integration
        """
        # TODO: Implement plan execution with ROS service calls
        # This would call actual robot services for navigation, manipulation, etc.
        success_count = 0

        for i, action in enumerate(actions):
            self.get_logger().info(f'Executing action {i+1}/{len(actions)}: {action["description"]}')

            # Publish feedback
            feedback_msg = String()
            feedback_msg.data = f"Executing: {action['description']}"
            self.feedback_pub.publish(feedback_msg)

            # Simulate action execution
            action_success = self.execute_single_action(action)

            if action_success:
                success_count += 1
            else:
                self.get_logger().error(f'Action failed: {action["description"]}')
                return False  # Stop execution on failure for this example

        return success_count == len(actions)

    def execute_single_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute a single action (placeholder implementation)
        """
        # TODO: In real implementation, call appropriate ROS services
        # For example:
        # - For navigation: call navigation2 services
        # - For grasping: call manipulation services
        # - For speaking: call text-to-speech services

        action_type = action['action']
        parameters = action['parameters']

        self.get_logger().info(f'Executing {action_type} with parameters: {parameters}')

        # Simulate action execution
        import time
        time.sleep(0.1)  # Simulate action time

        # Return success (in real system, check actual execution result)
        return True

    def destroy_node(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    node = LLMPlanningNode()

    # Use multi-threaded executor to handle async operations
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM Planning Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

# Test client for the planning node
class PlanningClient(Node):
    def __init__(self):
        super().__init__('planning_client')

        # TODO: Create client for the planning service
        # This would call the service created in LLMPlanningNode
        pass

# Uncomment to run when ready
# main()
```

### Expected Output
- Working ROS 2 node with LLM integration
- Service for planning and execution requests
- Proper state management between planning and execution
- Monitoring and feedback capabilities

### Solution Guidelines
- Use proper ROS 2 service definitions for planning requests
- Implement async handling for LLM calls without blocking ROS
- Include proper error handling and timeouts
- Add comprehensive logging for debugging

---

## Exercise 5: Integration Challenge - Complete Cognitive Planning System

### Objective
Integrate all components into a complete cognitive planning system for a humanoid robot.

### Tasks
1. Combine LLM planning, context awareness, and task decomposition
2. Implement complete perception-planning-action loop
3. Add multi-modal inputs (text, vision, sensor data)
4. Create comprehensive monitoring and evaluation system

### Implementation Requirements
- Use components from previous exercises
- Implement state management across the entire system
- Add performance monitoring and optimization
- Include safety and validation checks

### System Architecture
Your complete system should include:
- Multi-modal input processing (voice, text, vision)
- Context-aware LLM planning
- Task decomposition and execution
- ROS 2 integration for robot control
- Monitoring and evaluation components

### Testing Scenarios
Create test scenarios that cover:
1. Complex multi-step tasks with dependencies
2. Context-aware planning with changing conditions
3. Multi-modal input processing
4. Error handling and recovery
5. Performance optimization with caching

### Evaluation Criteria
- **Integration Quality**: How well all components work together
- **Robustness**: Handling of various inputs and error conditions
- **Performance**: Response time and efficiency
- **Safety**: Proper validation and safety checks
- **Scalability**: Ability to handle complex tasks

### Solution Approach
1. Create a main cognitive system class that orchestrates all components
2. Implement comprehensive state management
3. Add monitoring and logging for all system components
4. Test with realistic humanoid robot scenarios
5. Optimize for your specific use case and constraints

This exercise combines all the concepts learned in Week 12 to create a complete LLM-based cognitive planning system for humanoid robots, preparing you for the capstone project in Week 13.