---
sidebar_position: 2
---

# Week 11 Exercises: Voice Command Systems

## Exercise 1: Whisper Integration and Audio Processing

### Objective
Implement a basic voice command system using OpenAI Whisper with proper audio processing.

### Tasks
1. Set up OpenAI Whisper model for voice recognition
2. Implement real-time audio capture and processing
3. Create a simple command recognition system
4. Test the system with various voice commands

### Code Template
```python
import whisper
import pyaudio
import numpy as np
import threading
import queue
import time

class BasicVoiceRecognizer:
    def __init__(self):
        print("Loading Whisper model...")
        # TODO: Load the Whisper model (use "base" model for balance of speed/accuracy)
        self.model = None  # Replace with actual model loading

        # Audio parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works best at 16kHz
        self.record_seconds = 3

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Create queues for audio processing
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        print("Whisper model loaded successfully")

    def record_audio(self):
        """Record audio from microphone"""
        # TODO: Implement audio recording using PyAudio
        # 1. Open audio stream with the specified parameters
        # 2. Record audio in chunks for self.record_seconds
        # 3. Add recorded audio to self.audio_queue
        pass

    def process_audio(self):
        """Process audio data with Whisper"""
        # TODO: Process audio from queue and transcribe with Whisper
        # 1. Get audio data from self.audio_queue
        # 2. Convert to appropriate format for Whisper (float32, normalized)
        # 3. Transcribe using self.model.transcribe()
        # 4. Add results to self.result_queue
        pass

    def start_recognition(self):
        """Start the voice recognition system"""
        # TODO: Start recording and processing threads
        # 1. Start audio recording thread
        # 2. Start audio processing thread
        # 3. Return immediately to allow other operations
        pass

    def get_recognition_result(self):
        """Get the latest recognition result"""
        # TODO: Return the latest recognized text from result queue
        # Return None if no new results are available
        pass

# Test the implementation
def test_voice_recognizer():
    recognizer = BasicVoiceRecognizer()

    # TODO: Start the recognition system
    # TODO: Simulate some audio input (or run with real microphone)
    # TODO: Retrieve and print results
    pass

# Uncomment to test when ready
# test_voice_recognizer()
```

### Expected Output
- Working Whisper model integration
- Real-time audio capture and processing
- Text output of recognized speech
- Proper threading for non-blocking operation

### Solution Guidelines
- Use appropriate audio format (16kHz, mono, 16-bit) for Whisper
- Implement proper normalization of audio data
- Handle threading carefully to avoid blocking
- Add error handling for audio device access

---

## Exercise 2: Natural Language Command Parser

### Objective
Create a natural language command parser that can interpret human commands for robot control.

### Tasks
1. Implement pattern matching for common robot commands
2. Parse command arguments and parameters
3. Validate commands for safety and feasibility
4. Generate structured command objects

### Code Template
```python
import re
from typing import Dict, List, Optional

class CommandParser:
    def __init__(self):
        # Define command patterns with regex
        self.patterns = {
            # Movement commands
            'move_forward': [r'move forward', r'go forward', r'forward'],
            'move_backward': [r'move backward', r'go backward', r'back', r'backward'],
            'turn_left': [r'turn left', r'left turn', r'rotate left'],
            'turn_right': [r'turn right', r'right turn', r'rotate right'],
            'stop': [r'stop', r'halt', r'freeze'],

            # Action commands
            'wave': [r'wave', r'wave hello', r'hello'],
            'sit': [r'sit', r'sit down'],
            'stand': [r'stand', r'stand up'],

            # Navigation commands
            'go_to': [r'go to (.+)', r'move to (.+)', r'go to the (.+)'],
            'find': [r'find (.+)', r'look for (.+)', r'where is (.+)']
        }

        # Distance/parameter patterns
        self.distance_pattern = r'(\d+(?:\.\d+)?)\s*(m|meter|cm|ft|feet)?'
        self.angle_pattern = r'(\d+(?:\.\d+)?)\s*(deg|degree|rad|radian)?'

    def parse_command(self, text: str) -> Optional[Dict]:
        """
        Parse natural language command and return structured representation
        """
        # TODO: Implement command parsing
        # 1. Normalize the input text (lowercase, remove extra spaces)
        # 2. Try each pattern to match the command
        # 3. Extract command type and arguments
        # 4. Return dictionary with 'type', 'arguments', and 'parameters'
        pass

    def extract_parameters(self, text: str) -> Dict:
        """
        Extract numerical parameters like distances and angles
        """
        # TODO: Extract distance and angle parameters from text
        # Use self.distance_pattern and self.angle_pattern
        # Return dictionary with extracted parameters
        pass

    def validate_command(self, parsed_command: Dict) -> bool:
        """
        Validate if the command is safe and feasible
        """
        # TODO: Implement command validation
        # 1. Check if command type is recognized
        # 2. Validate parameters (e.g., distance limits)
        # 3. Check for safety constraints
        # 4. Return True if command is valid, False otherwise
        pass

class RobotCommandValidator:
    def __init__(self):
        self.max_distance = 10.0  # meters
        self.max_angle = 180.0   # degrees
        self.safe_commands = ['move_forward', 'move_backward', 'turn_left', 'turn_right', 'stop']

    def validate(self, command: Dict) -> tuple[bool, str]:
        """
        Validate command with detailed feedback
        Returns (is_valid, reason)
        """
        # TODO: Implement detailed validation
        # Check command type, parameters, safety limits
        # Return validation result and reason string
        pass

# Test the implementation
def test_command_parser():
    parser = CommandParser()
    validator = RobotCommandValidator()

    test_commands = [
        "Please move forward",
        "Turn left 90 degrees",
        "Go to the kitchen",
        "Wave hello",
        "Move forward 2 meters",
        "Find the red ball"
    ]

    # TODO: Test each command
    # Parse, validate, and print results
    for cmd in test_commands:
        print(f"Command: {cmd}")
        # parsed = parser.parse_command(cmd)
        # if parsed:
        #     is_valid, reason = validator.validate(parsed)
        #     print(f"  Parsed: {parsed}")
        #     print(f"  Valid: {is_valid} - {reason}")
        # else:
        #     print("  Could not parse command")
        print("  [Implement parsing logic above]")
        print()

# Uncomment when ready
# test_command_parser()
```

### Expected Output
- Working natural language command parser
- Recognition of various command patterns
- Extraction of parameters from commands
- Validation of commands for safety/feasibility

### Solution Guidelines
- Use regex groups to extract command arguments
- Implement hierarchical pattern matching
- Consider command context and ambiguity
- Add comprehensive validation checks

---

## Exercise 3: Voice Command Execution System

### Objective
Create a ROS 2 node that receives voice commands and executes them on a simulated robot.

### Tasks
1. Create a ROS 2 subscriber for voice commands
2. Implement command execution based on parsed commands
3. Add safety checks and validation
4. Provide feedback to users

### Code Template
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time
import time

class VoiceCommandExecutor(Node):
    def __init__(self):
        super().__init__('voice_command_executor')

        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.feedback_pub = self.create_publisher(String, '/voice_feedback', 10)

        # Subscriber for parsed commands
        self.command_sub = self.create_subscription(
            String, '/parsed_command', self.command_callback, 10)

        # Command parser and validator
        self.command_parser = CommandParser()
        self.command_validator = RobotCommandValidator()

        # Robot state
        self.is_moving = False
        self.current_speed = 0.0
        self.safety_enabled = True

        # Command history for context
        self.command_history = []

        self.get_logger().info('Voice command executor initialized')

    def command_callback(self, msg):
        """
        Callback for incoming voice commands
        """
        # TODO: Process the incoming command message
        # 1. Parse the command text
        # 2. Validate the command
        # 3. Execute if valid, provide feedback if invalid
        # 4. Update command history
        pass

    def execute_command(self, parsed_command):
        """
        Execute the parsed command on the robot
        """
        cmd_type = parsed_command.get('type', 'unknown')

        # TODO: Implement command execution
        # Based on cmd_type, call appropriate execution method
        # Add safety checks before execution
        # Provide feedback after execution
        pass

    def execute_movement_command(self, cmd_type, distance=None, angle=None):
        """
        Execute movement commands (forward, backward, turn)
        """
        # TODO: Implement movement execution
        # 1. Create Twist message for movement
        # 2. Set appropriate linear/angular velocities
        # 3. Publish to cmd_vel topic
        # 4. Consider duration based on distance/angle
        pass

    def execute_action_command(self, cmd_type, params=None):
        """
        Execute action commands (wave, sit, stand, etc.)
        """
        # TODO: Implement action execution
        # 1. Create JointState message for actions
        # 2. Set appropriate joint positions
        # 3. Publish to joint_commands topic
        pass

    def safety_check(self, command):
        """
        Perform safety checks before executing command
        """
        # TODO: Implement safety validation
        # 1. Check if robot is in safe state
        # 2. Validate command parameters
        # 3. Check for collision risks
        # 4. Return True if safe, False otherwise
        pass

    def provide_feedback(self, message):
        """
        Provide feedback to user about command execution
        """
        feedback_msg = String()
        feedback_msg.data = message
        self.feedback_pub.publish(feedback_msg)
        self.get_logger().info(f"Feedback: {message}")

    def emergency_stop(self):
        """
        Emergency stop for safety
        """
        # TODO: Implement emergency stop
        # Stop all robot movement immediately
        # Publish zero velocities to cmd_vel
        # Stop all joint movements
        pass

# Test publisher node to simulate command input
class TestCommandPublisher(Node):
    def __init__(self):
        super().__init__('test_command_publisher')
        self.pub = self.create_publisher(String, '/parsed_command', 10)

        # Timer to send test commands
        self.timer = self.create_timer(2.0, self.send_test_command)
        self.command_index = 0
        self.test_commands = [
            "move forward",
            "turn left",
            "stop",
            "wave",
            "sit",
            "stand"
        ]

    def send_test_command(self):
        if self.command_index < len(self.test_commands):
            cmd_msg = String()
            cmd_msg.data = self.test_commands[self.command_index]
            self.pub.publish(cmd_msg)
            self.get_logger().info(f"Sent test command: {cmd_msg.data}")
            self.command_index += 1

def main(args=None):
    rclpy.init(args=args)

    # Create both nodes
    executor = VoiceCommandExecutor()
    publisher = TestCommandPublisher()  # For testing

    try:
        # Run both nodes
        exec_future = rclpy.executors.MultiThreadedExecutor()
        exec_future.add_node(executor)
        exec_future.add_node(publisher)
        exec_future.spin()
    except KeyboardInterrupt:
        executor.get_logger().info('Shutting down voice command executor')
    finally:
        executor.destroy_node()
        publisher.destroy_node()
        rclpy.shutdown()

# Uncomment to run when ready
# main()
```

### Expected Output
- ROS 2 node that processes voice commands
- Movement and action execution on simulated robot
- Safety checks and validation
- User feedback system

### Solution Guidelines
- Use proper ROS 2 node structure and lifecycle
- Implement safety checks before command execution
- Add appropriate message types for different commands
- Consider timing and duration for movement commands

---

## Exercise 4: Robust Voice Processing with Error Handling

### Objective
Create a robust voice processing system that handles errors, ambiguity, and provides user feedback.

### Tasks
1. Implement confidence estimation for speech recognition
2. Add error handling for recognition failures
3. Create ambiguity resolution mechanisms
4. Implement user feedback and correction systems

### Code Template
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger, SetBool
import numpy as np
from collections import deque
import threading
import time

class RobustVoiceProcessor(Node):
    def __init__(self):
        super().__init__('robust_voice_processor')

        # Publishers and subscribers
        self.command_pub = self.create_publisher(String, '/voice_command', 10)
        self.feedback_pub = self.create_publisher(String, '/voice_feedback', 10)

        # Services for user interaction
        self.confirmation_srv = self.create_service(
            SetBool, 'command_confirmation', self.confirmation_callback)
        self.correction_srv = self.create_service(
            Trigger, 'request_correction', self.correction_callback)

        # Voice processing components
        self.whisper_model = None  # Will be loaded later
        self.command_parser = CommandParser()

        # Confidence and quality metrics
        self.confidence_threshold = 0.7
        self.audio_quality_threshold = 0.1

        # Command history and context
        self.command_history = deque(maxlen=5)
        self.pending_confirmation = None

        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        self.audio_buffer = []

        # Initialize Whisper model
        self.initialize_whisper()

        self.get_logger().info('Robust voice processor initialized')

    def initialize_whisper(self):
        """
        Initialize Whisper model (placeholder for actual loading)
        """
        # TODO: Load Whisper model
        # Handle potential loading errors gracefully
        pass

    def process_voice_input(self, audio_data):
        """
        Process voice input with confidence estimation
        """
        # TODO: Process audio data through Whisper
        # 1. Calculate audio quality metrics
        # 2. Transcribe speech to text
        # 3. Estimate confidence in recognition
        # 4. Handle low-confidence results appropriately
        pass

    def estimate_confidence(self, text, audio_features):
        """
        Estimate confidence in the recognized text
        """
        # TODO: Implement confidence estimation
        # Consider factors like:
        # - Audio quality (noise level, clarity)
        # - Text structure (grammar, common phrases)
        # - Acoustic features
        # Return confidence value between 0 and 1
        pass

    def handle_low_confidence(self, recognized_text, confidence):
        """
        Handle low-confidence recognition results
        """
        # TODO: Implement low-confidence handling
        # 1. Request user confirmation
        # 2. Store for potential correction
        # 3. Provide appropriate feedback
        pass

    def confirmation_callback(self, request, response):
        """
        Handle user confirmation of recognized command
        """
        # TODO: Process user confirmation
        # 1. Check if there's a pending command for confirmation
        # 2. If confirmed, execute the command
        # 3. If not confirmed, request re-recognition
        # 4. Return appropriate response
        pass

    def correction_callback(self, request, response):
        """
        Handle request for command correction
        """
        # TODO: Handle command correction request
        # 1. Provide current recognized command for user correction
        # 2. Set up system to receive corrected command
        # 3. Return response with current status
        pass

    def validate_command_context(self, command):
        """
        Validate command in context of recent commands and robot state
        """
        # TODO: Implement contextual validation
        # Consider:
        # - Previous commands (sequence makes sense?)
        # - Robot's current state
        # - Environmental context
        # Return validation result
        pass

    def add_to_history(self, command, success=True):
        """
        Add command to history with success status
        """
        entry = {
            'command': command,
            'timestamp': time.time(),
            'success': success,
            'attempts': 1
        }
        self.command_history.append(entry)

    def get_feedback_for_recognition(self, text, confidence):
        """
        Generate appropriate feedback based on recognition confidence
        """
        # TODO: Generate feedback message
        # If confidence is high: "Executing command: {text}"
        # If confidence is medium: "Did you say: {text}?"
        # If confidence is low: "I didn't understand that clearly"
        pass

class AudioQualityAnalyzer:
    def __init__(self):
        self.noise_threshold = 0.01
        self.snr_threshold = 10  # Signal-to-noise ratio

    def analyze_audio(self, audio_data):
        """
        Analyze audio quality metrics
        """
        # TODO: Calculate audio quality metrics
        # - Signal power
        # - Noise level
        # - Signal-to-noise ratio
        # - Spectral features
        # Return dictionary of quality metrics
        pass

    def is_quality_acceptable(self, metrics):
        """
        Check if audio quality is acceptable for recognition
        """
        # TODO: Evaluate if audio quality is sufficient
        # Return True if quality is acceptable, False otherwise
        pass

# Test the robust processor
def test_robust_processor():
    # TODO: Create test scenario for robust processing
    # 1. Simulate various audio quality conditions
    # 2. Test confidence estimation
    # 3. Test error handling and user feedback
    # 4. Test command correction workflow
    pass

# Uncomment when ready
# test_robust_processor()
```

### Expected Output
- Voice processing with confidence estimation
- Error handling for low-quality audio
- User feedback and correction mechanisms
- Context-aware command validation

### Solution Guidelines
- Implement multiple confidence indicators
- Use statistical methods for quality assessment
- Design clear user interaction flows
- Consider edge cases and error conditions

---

## Exercise 5: Integration Challenge - Complete Voice Control System

### Objective
Integrate all components into a complete voice control system for a humanoid robot.

### Tasks
1. Combine Whisper integration, command parsing, and execution
2. Implement complete user interaction flow
3. Add comprehensive error handling and safety
4. Test the complete system

### Implementation Requirements
- Use components developed in previous exercises
- Implement state management for the complete system
- Add monitoring and logging capabilities
- Create a simple user interface for testing

### System Architecture
Your complete system should include:
- Audio input and preprocessing
- Speech recognition with Whisper
- Natural language understanding
- Command validation and safety checks
- Robot command execution
- User feedback and interaction

### Testing Scenarios
Create test scenarios that cover:
1. Normal command execution (high-confidence recognition)
2. Low-confidence recognition with user confirmation
3. Command correction when recognition fails
4. Safety checks and emergency stops
5. Context-aware command validation

### Evaluation Criteria
- **Integration Quality**: How well components work together
- **Robustness**: Handling of various audio conditions and errors
- **User Experience**: Clarity of feedback and interaction flow
- **Safety**: Proper validation and safety mechanisms
- **Performance**: Responsiveness and accuracy

### Solution Approach
1. Create a main system class that orchestrates all components
2. Implement state management for different system modes
3. Add comprehensive logging for debugging
4. Test with various real-world scenarios
5. Optimize for your specific hardware constraints

This exercise combines all the concepts learned in Week 11 to create a complete voice command system for humanoid robots, preparing you for the more advanced LLM integration in Week 12.