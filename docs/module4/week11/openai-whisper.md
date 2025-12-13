---
sidebar_position: 2
---

# Week 11: OpenAI Whisper Integration

## Learning Objectives

By the end of this section, you will be able to:
- Implement OpenAI Whisper for real-time speech recognition in robotic systems
- Configure Whisper models for optimal performance in robotics applications
- Integrate Whisper with ROS 2 for voice command processing
- Optimize Whisper for edge deployment on robotic platforms
- Handle Whisper's computational requirements in real-time robotic systems

## OpenAI Whisper Overview

OpenAI Whisper is a state-of-the-art speech recognition model that has revolutionized automatic speech recognition (ASR) capabilities. For humanoid robots, Whisper provides the ability to understand natural human speech, enabling more intuitive human-robot interaction.

### Key Features of Whisper

1. **Multilingual Support**: Supports multiple languages with high accuracy
2. **Robustness**: Handles various accents, background noise, and audio quality
3. **Zero-Shot Learning**: Performs well without domain-specific training
4. **Multiple Model Sizes**: From tiny models for edge devices to large models for accuracy

### Whisper Model Variants

| Model | Size | Required VRAM | Relative Speed | Accuracy |
|-------|------|---------------|----------------|----------|
| tiny  | 75 MB | ~1 GB | 32x | Lower |
| base  | 145 MB | ~1 GB | 16x | Low |
| small | 485 MB | ~2 GB | 6x | Medium |
| medium | 1.5 GB | ~5 GB | 2x | High |
| large | 3.0 GB | ~10 GB | 1x | Highest |

## Whisper Installation and Setup

### Basic Installation

```bash
# Install Whisper and dependencies
pip install openai-whisper

# For GPU acceleration (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional dependencies for audio processing
pip install pyaudio soundfile
```

### Model Download and Caching

```python
import whisper

# Download and cache models
model_tiny = whisper.load_model("tiny")
model_base = whisper.load_model("base")
model_small = whisper.load_model("small")
model_medium = whisper.load_model("medium")
model_large = whisper.load_model("large")
```

## Real-Time Whisper Integration

### Audio Processing Pipeline

```python
# real_time_whisper.py
import whisper
import torch
import pyaudio
import numpy as np
import queue
import threading
import time
from typing import Optional, Dict, Any

class RealTimeWhisper:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize real-time Whisper processor
        """
        self.model_size = model_size
        self.device = device

        # Load Whisper model
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size).to(device)
        print(f"Model loaded on {device}")

        # Audio parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.record_seconds = 3  # Process audio in 3-second chunks

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Queues for audio processing
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Processing flags
        self.is_recording = False
        self.processing_thread = None

    def start_recording(self):
        """Start real-time audio recording and processing"""
        if self.is_recording:
            return

        self.is_recording = True

        # Start audio recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.start()

        print("Real-time Whisper started")

    def stop_recording(self):
        """Stop audio recording and processing"""
        self.is_recording = False

        if hasattr(self, 'recording_thread'):
            self.recording_thread.join(timeout=2)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)

        # Terminate PyAudio
        self.audio.terminate()

        print("Real-time Whisper stopped")

    def _record_audio(self):
        """Record audio from microphone in chunks"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        print("Recording... Speak now!")

        while self.is_recording:
            # Record a chunk
            frames = []
            for _ in range(0, int(self.sample_rate / self.chunk_size * self.record_seconds)):
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"Audio read error: {e}")
                    break

            # Add audio chunk to processing queue
            audio_chunk = b''.join(frames)
            self.audio_queue.put(audio_chunk)

        stream.stop_stream()
        stream.close()

    def _process_audio(self):
        """Process audio chunks with Whisper"""
        while self.is_recording:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1)

                # Convert to numpy array
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

                # Normalize to [-1, 1] and convert to float32
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Process with Whisper
                result = self._transcribe_audio(audio_float)

                if result and result.strip():
                    print(f"Recognized: {result}")

                    # Add result to output queue
                    self.result_queue.put({
                        'text': result,
                        'timestamp': time.time(),
                        'confidence': self._estimate_confidence(result, audio_float)
                    })

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper"""
        try:
            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio_data).to(self.device)

            # Transcribe
            result = self.model.transcribe(
                audio_tensor,
                language='en',  # Set to appropriate language
                fp16=(self.device == 'cuda')  # Use fp16 for GPU inference
            )

            return result['text'].strip()

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def _estimate_confidence(self, text: str, audio_data: np.ndarray) -> float:
        """Estimate confidence in recognition result"""
        # Simple confidence estimation based on audio quality
        # In practice, you might use more sophisticated methods
        audio_energy = np.mean(np.abs(audio_data))

        # Higher energy generally indicates clearer speech
        energy_score = min(1.0, audio_energy * 100)

        # Shorter, more common phrases might be more reliable
        word_count = len(text.split())
        length_score = 1.0 if 2 <= word_count <= 10 else 0.7

        return energy_score * length_score

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get the latest recognition result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

# Example usage
def main():
    # Initialize Whisper processor
    whisper_processor = RealTimeWhisper(model_size="base", device="cpu")

    try:
        # Start recording
        whisper_processor.start_recording()

        # Process results for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            result = whisper_processor.get_latest_result()
            if result:
                print(f"Result: {result['text']} (Conf: {result['confidence']:.2f})")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        whisper_processor.stop_recording()

if __name__ == "__main__":
    main()
```

## Whisper Optimization for Robotics

### Model Optimization Techniques

```python
# whisper_optimization.py
import whisper
import torch
from torch.quantization import quantize_dynamic
import numpy as np

class OptimizedWhisper:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._load_optimized_model()

    def _load_optimized_model(self):
        """Load and optimize Whisper model for robotics use"""
        print(f"Loading optimized Whisper {self.model_size} model...")

        # Load model
        self.model = whisper.load_model(self.model_size).to(self.device)

        # Apply optimizations based on device
        if self.device == "cuda":
            # Use half precision for GPU
            self.model = self.model.half()
        else:
            # Quantize for CPU
            self.model = quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        print(f"Model optimized for {self.device}")

    def transcribe_with_timing(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio with performance metrics"""
        start_time = time.time()

        # Convert to tensor
        if isinstance(audio_data, np.ndarray):
            audio_tensor = torch.from_numpy(audio_data).to(self.device)
            if self.device == "cuda":
                audio_tensor = audio_tensor.half()

        # Transcribe
        result = self.model.transcribe(audio_tensor)

        end_time = time.time()
        processing_time = end_time - start_time

        return {
            'text': result['text'],
            'processing_time': processing_time,
            'model_size': self.model_size,
            'device': self.device
        }

    def batch_transcribe(self, audio_chunks: list) -> list:
        """Process multiple audio chunks efficiently"""
        results = []

        for chunk in audio_chunks:
            result = self.transcribe_with_timing(chunk)
            results.append(result)

        return results
```

## ROS 2 Integration with Whisper

### Creating a Whisper ROS 2 Node

```python
# whisper_ros_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import whisper
import torch
import numpy as np
import threading
import queue
import time

class WhisperROSNode(Node):
    def __init__(self):
        super().__init__('whisper_ros_node')

        # Parameters
        model_size = self.declare_parameter('model_size', 'base').value
        self.device = self.declare_parameter('device', 'cpu').value

        # Publishers
        self.transcript_pub = self.create_publisher(String, '/whisper/transcript', 10)
        self.confidence_pub = self.create_publisher(String, '/whisper/confidence', 10)
        self.status_pub = self.create_publisher(String, '/whisper/status', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, '/audio_input', self.audio_callback, 10)
        self.enable_sub = self.create_subscription(
            Bool, '/whisper/enable', self.enable_callback, 10)

        # Initialize Whisper model
        self.get_logger().info(f'Loading Whisper {model_size} model...')
        self.model = whisper.load_model(model_size).to(self.device)
        self.get_logger().info('Whisper model loaded successfully')

        # Processing queues
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()

        # Control flags
        self.enabled = True
        self.processing_active = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_queue)
        self.processing_thread.start()

        # Timer for status updates
        self.status_timer = self.create_timer(5.0, self.publish_status)

        self.get_logger().info('Whisper ROS Node initialized')

    def audio_callback(self, msg):
        """Handle incoming audio data"""
        if not self.enabled:
            return

        # Convert AudioData to numpy array
        # Note: AudioData format may vary, adjust based on your audio source
        try:
            audio_array = np.frombuffer(msg.data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Add to processing queue
            self.audio_queue.put(audio_float)

        except Exception as e:
            self.get_logger().error(f'Error processing audio: {e}')

    def enable_callback(self, msg):
        """Handle enable/disable commands"""
        self.enabled = msg.data
        status_msg = String()
        status_msg.data = f"Whisper enabled: {self.enabled}"
        self.status_pub.publish(status_msg)

    def process_audio_queue(self):
        """Process audio data from queue using Whisper"""
        while self.processing_active:
            try:
                # Get audio from queue with timeout
                audio_data = self.audio_queue.get(timeout=1.0)

                # Transcribe using Whisper
                result = self.model.transcribe(audio_data)

                # Publish results
                transcript_msg = String()
                transcript_msg.data = result['text']
                self.transcript_pub.publish(transcript_msg)

                confidence_msg = String()
                confidence_msg.data = f"Confidence: {result.get('avg_logprob', 0.0):.2f}"
                self.confidence_pub.publish(confidence_msg)

                self.get_logger().info(f'Transcribed: {result["text"]}')

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Whisper processing error: {e}')

    def publish_status(self):
        """Publish status information"""
        status_msg = String()
        status_msg.data = f"Whisper active, enabled: {self.enabled}, queue_size: {self.audio_queue.qsize()}"
        self.status_pub.publish(status_msg)

    def destroy_node(self):
        """Clean up resources"""
        self.processing_active = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = WhisperROSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Whisper ROS Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Voice Command Processing Pipeline

### Advanced Voice Command Processing

```python
# voice_command_processor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import whisper
import torch
import numpy as np
import re
from typing import Dict, List, Optional

class VoiceCommandProcessor(Node):
    def __init__(self):
        super().__init__('voice_command_processor')

        # Publishers
        self.command_pub = self.create_publisher(String, '/parsed_command', 10)
        self.motion_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize Whisper
        self.model = whisper.load_model("base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.model = self.model.to(self.device)

        # Command patterns for natural language understanding
        self.command_patterns = {
            # Movement commands
            'move_forward': [
                r'move forward|go forward|move ahead|go ahead',
                r'forward|straight|go straight'
            ],
            'move_backward': [
                r'move backward|go backward|move back|go back',
                r'backward|back|reverse'
            ],
            'turn_left': [
                r'turn left|rotate left|turn anti-clockwise|go left',
                r'left|turn counter clockwise'
            ],
            'turn_right': [
                r'turn right|rotate right|turn clockwise|go right',
                r'right|turn clock wise'
            ],
            'stop': [
                r'stop|halt|freeze|stand still|pause',
                r'wait|hold|cease'
            ],
            'speed_control': [
                r'slow down|reduce speed|go slower',
                r'speed up|increase speed|go faster',
                r'go (slow|medium|fast)'
            ]
        }

        # Initialize audio processing (simplified - in practice you'd connect to audio source)
        self.setup_audio_input()

        self.get_logger().info('Voice Command Processor initialized')

    def setup_audio_input(self):
        """Setup audio input processing"""
        # In a real implementation, this would connect to audio input
        # For now, we'll provide a method to manually process audio
        pass

    def process_audio_command(self, audio_data: np.ndarray) -> Optional[str]:
        """Process audio data and extract command"""
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data).to(self.device)

            # Transcribe
            result = self.model.transcribe(audio_tensor)
            text = result['text'].strip().lower()

            self.get_logger().info(f'Recognized: {text}')

            # Parse command
            parsed_command = self.parse_command(text)

            if parsed_command:
                # Publish parsed command
                cmd_msg = String()
                cmd_msg.data = parsed_command
                self.command_pub.publish(cmd_msg)

                # Execute command
                self.execute_command(parsed_command)

                return parsed_command

        except Exception as e:
            self.get_logger().error(f'Error processing audio command: {e}')

        return None

    def parse_command(self, text: str) -> Optional[str]:
        """Parse natural language command"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text).strip()

        # Try to match command patterns
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return command_type

        # Handle more complex commands with parameters
        if 'go to' in text or 'move to' in text:
            # Extract location
            location_match = re.search(r'(kitchen|living room|bedroom|office|entrance)', text)
            if location_match:
                return f'go_to_{location_match.group(1).replace(" ", "_")}'

        # Handle speed commands
        speed_match = re.search(r'go (slow|medium|fast)', text)
        if speed_match:
            speed = speed_match.group(1)
            return f'speed_{speed}'

        return None

    def execute_command(self, command: str):
        """Execute parsed command"""
        if command == 'move_forward':
            self.move_forward()
        elif command == 'move_backward':
            self.move_backward()
        elif command == 'turn_left':
            self.turn_left()
        elif command == 'turn_right':
            self.turn_right()
        elif command == 'stop':
            self.stop_robot()
        elif command.startswith('go_to_'):
            location = command[6:]  # Remove 'go_to_' prefix
            self.navigate_to_location(location)
        elif command.startswith('speed_'):
            speed_level = command[6:]  # Remove 'speed_' prefix
            self.set_speed(speed_level)
        else:
            self.get_logger().warn(f'Unknown command: {command}')

    def move_forward(self):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = 0.3  # m/s
        cmd.angular.z = 0.0
        self.motion_pub.publish(cmd)

    def move_backward(self):
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -0.3  # m/s
        cmd.angular.z = 0.0
        self.motion_pub.publish(cmd)

    def turn_left(self):
        """Turn robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # rad/s
        self.motion_pub.publish(cmd)

    def turn_right(self):
        """Turn robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5  # rad/s
        self.motion_pub.publish(cmd)

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.motion_pub.publish(cmd)

    def navigate_to_location(self, location: str):
        """Navigate to specified location (placeholder)"""
        self.get_logger().info(f'Navigating to {location}')
        # In real implementation, this would use navigation stack

    def set_speed(self, speed_level: str):
        """Set robot speed based on level"""
        speeds = {'slow': 0.1, 'medium': 0.3, 'fast': 0.5}
        speed = speeds.get(speed_level, 0.3)  # default to medium

        # Send current motion command with new speed
        # This would require tracking current direction
        pass

def main(args=None):
    rclpy.init(args=args)
    processor = VoiceCommandProcessor()

    try:
        # In a real implementation, this would connect to audio input
        # For demonstration, we'll just keep the node running
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Shutting down Voice Command Processor')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization for Robotics

### Efficient Whisper Usage in Robotics

```python
# whisper_robotics_optimization.py
import whisper
import torch
import numpy as np
import time
from collections import deque
from typing import Dict, List, Optional

class RoboticsWhisperOptimizer:
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Performance metrics
        self.processing_times = deque(maxlen=100)
        self.energy_levels = deque(maxlen=100)

        # Initialize optimized model
        self._initialize_model()

        # VAD (Voice Activity Detection) simulation
        self.vad_threshold = 0.01
        self.silence_count = 0
        self.max_silence = 50  # frames of silence before reducing processing

    def _initialize_model(self):
        """Initialize Whisper with robotics-specific optimizations"""
        print(f"Initializing optimized Whisper for robotics...")

        # Load model
        self.model = whisper.load_model(self.model_size)

        if self.device == "cuda":
            self.model = self.model.to(self.device).half()  # Half precision for GPU
        else:
            # For CPU, we might want to use a smaller model or quantized version
            if self.model_size in ["large", "medium"]:
                print("Warning: Large models on CPU may be slow for real-time robotics")

        print(f"Model initialized on {self.device}")

    def is_voice_activity(self, audio_chunk: np.ndarray, threshold: float = None) -> bool:
        """Detect voice activity in audio chunk"""
        if threshold is None:
            threshold = self.vad_threshold

        energy = np.mean(np.abs(audio_chunk))
        return energy > threshold

    def optimize_processing_frequency(self) -> float:
        """Determine optimal processing frequency based on activity"""
        if len(self.energy_levels) < 10:
            return 1.0  # Process every second initially

        avg_energy = np.mean(list(self.energy_levels))

        # Higher energy (more voice activity) = higher processing frequency
        if avg_energy > 0.05:
            return 2.0  # Process twice per second
        elif avg_energy > 0.02:
            return 1.0  # Process once per second
        else:
            return 0.5  # Process every 2 seconds

    def transcribe_efficiently(self, audio_data: np.ndarray) -> Optional[Dict]:
        """Transcribe audio with efficiency optimizations"""
        start_time = time.time()

        # Check for voice activity first
        if not self.is_voice_activity(audio_data):
            self.silence_count += 1
            if self.silence_count > self.max_silence:
                # Reduce processing during long silence periods
                return None
        else:
            self.silence_count = 0  # Reset silence counter

        # Add energy level to metrics
        energy = np.mean(np.abs(audio_data))
        self.energy_levels.append(energy)

        try:
            # Convert to tensor
            if self.device == "cuda":
                audio_tensor = torch.from_numpy(audio_data).to(self.device).half()
            else:
                audio_tensor = torch.from_numpy(audio_data).to(self.device)

            # Transcribe
            result = self.model.transcribe(audio_tensor)

            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            return {
                'text': result['text'],
                'processing_time': processing_time,
                'energy': energy,
                'timestamp': time.time()
            }

        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.processing_times:
            return {
                'avg_processing_time': 0.0,
                'processing_frequency': 0.0,
                'avg_energy': 0.0
            }

        return {
            'avg_processing_time': np.mean(list(self.processing_times)),
            'processing_frequency': self.optimize_processing_frequency(),
            'avg_energy': np.mean(list(self.energy_levels)) if self.energy_levels else 0.0,
            'model_size': self.model_size,
            'device': self.device
        }

    def adaptive_processing(self, audio_stream_generator):
        """Process audio stream with adaptive optimization"""
        for audio_chunk in audio_stream_generator:
            # Determine processing frequency
            freq = self.optimize_processing_frequency()

            # Process based on frequency
            if np.random.random() < freq:  # Adaptive sampling
                result = self.transcribe_efficiently(audio_chunk)
                if result and result['text'].strip():
                    yield result

# Example usage in robotics context
def robotics_voice_pipeline():
    """Example of using optimized Whisper in a robotics pipeline"""
    optimizer = RoboticsWhisperOptimizer(model_size="base")

    # Simulate audio stream (in real robotics, this would come from microphone)
    def audio_stream():
        # This would be replaced with actual audio input
        for i in range(100):  # Simulate 100 audio chunks
            # Generate simulated audio (silence with occasional "speech")
            if i % 20 == 0:  # Every 20th chunk has "speech"
                audio = np.random.normal(0.1, 0.05, 16000)  # Higher energy = speech
            else:
                audio = np.random.normal(0.0, 0.01, 16000)  # Lower energy = silence
            yield audio

    print("Starting optimized voice processing...")

    for result in optimizer.adaptive_processing(audio_stream()):
        if result:
            print(f"Recognized: {result['text'][:50]}... (Time: {result['processing_time']:.2f}s)")

    # Print performance metrics
    metrics = optimizer.get_performance_metrics()
    print(f"Performance metrics: {metrics}")

if __name__ == "__main__":
    robotics_voice_pipeline()
```

## Integration with Humanoid Robot Systems

### Complete Voice Command System for Humanoid Robots

```python
# complete_voice_system.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import whisper
import torch
import numpy as np
import threading
import queue
import time
import re
from typing import Dict, Optional

class CompleteVoiceSystem(Node):
    def __init__(self):
        super().__init__('complete_voice_system')

        # Publishers
        self.command_pub = self.create_publisher(String, '/robot_command', 10)
        self.motion_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.feedback_pub = self.create_publisher(String, '/voice_feedback', 10)

        # Initialize Whisper
        self.get_logger().info('Loading Whisper model...')
        self.model = whisper.load_model("base")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.model = self.model.to(self.device)

        # Command patterns for humanoid-specific commands
        self.command_patterns = {
            # Navigation commands
            'navigate_to': r'go to (the )?(?P<location>\w+)|move to (the )?(?P<location2>\w+)',
            'move_forward': r'move forward|go forward|move ahead',
            'move_backward': r'move backward|go backward|move back',
            'turn_left': r'turn left|rotate left',
            'turn_right': r'turn right|rotate right',
            'stop': r'stop|halt|freeze|wait|pause',

            # Manipulation commands
            'grasp': r'pick up (?P<object>\w+)|grasp (?P<object2>\w+)|get (?P<object3>\w+)',
            'place': r'put down|place|release|drop',
            'wave': r'wave|waving|wave hello|hello',
            'point': r'point to|point at',

            # Posture commands
            'sit': r'sit down|sit|take a seat',
            'stand': r'stand up|stand|stand straight',
            'crouch': r'crouch|bend down|squat',

            # Interaction commands
            'speak': r'say (?P<text>.+)|speak (?P<text2>.+)|tell (?P<text3>.+)',
        }

        # System state
        self.enabled = True
        self.robot_state = "idle"  # idle, moving, manipulating, etc.
        self.current_location = "unknown"

        # Audio processing setup (simplified - connect to actual audio source)
        self.audio_queue = queue.Queue()
        self.processing_active = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

        self.get_logger().info('Complete Voice System initialized')

    def process_text_command(self, text: str):
        """Process text command from Whisper transcription"""
        if not self.enabled:
            return

        self.get_logger().info(f'Processing command: {text}')

        # Parse command using regex patterns
        command = self.parse_command(text)

        if command:
            # Publish command for logging/debugging
            cmd_msg = String()
            cmd_msg.data = f"{command['type']}: {command.get('params', {})}"
            self.command_pub.publish(cmd_msg)

            # Execute command
            success = self.execute_command(command)

            # Provide feedback
            if success:
                feedback = f"Executing: {text}"
            else:
                feedback = f"Failed to execute: {text}"

            feedback_msg = String()
            feedback_msg.data = feedback
            self.feedback_pub.publish(feedback_msg)

    def parse_command(self, text: str) -> Optional[Dict]:
        """Parse natural language command using regex patterns"""
        text_lower = text.lower().strip()

        for cmd_type, pattern in self.command_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                params = match.groupdict()

                # Clean up parameter names (remove numbered variants)
                cleaned_params = {}
                for key, value in params.items():
                    if value is not None:
                        # Remove numbers from parameter names (e.g., 'location2' -> 'location')
                        clean_key = re.sub(r'\d+$', '', key)
                        cleaned_params[clean_key] = value

                return {
                    'type': cmd_type,
                    'params': cleaned_params,
                    'original': text
                }

        return None

    def execute_command(self, command: Dict) -> bool:
        """Execute parsed command on humanoid robot"""
        cmd_type = command['type']
        params = command['params']

        try:
            if cmd_type == 'move_forward':
                self.move_forward()
            elif cmd_type == 'move_backward':
                self.move_backward()
            elif cmd_type == 'turn_left':
                self.turn_left()
            elif cmd_type == 'turn_right':
                self.turn_right'
            elif cmd_type == 'stop':
                self.stop_robot()
            elif cmd_type == 'navigate_to':
                location = params.get('location') or params.get('location2')
                if location:
                    self.navigate_to_location(location)
                else:
                    self.get_logger().warn('Navigate command missing location')
                    return False
            elif cmd_type == 'grasp':
                obj = params.get('object') or params.get('object2') or params.get('object3')
                if obj:
                    self.grasp_object(obj)
                else:
                    self.get_logger().warn('Grasp command missing object')
                    return False
            elif cmd_type == 'wave':
                self.wave()
            elif cmd_type == 'sit':
                self.sit_down()
            elif cmd_type == 'stand':
                self.stand_up()
            elif cmd_type == 'speak':
                text = params.get('text') or params.get('text2') or params.get('text3')
                if text:
                    self.speak_text(text)
                else:
                    self.get_logger().warn('Speak command missing text')
                    return False
            else:
                self.get_logger().warn(f'Unknown command type: {cmd_type}')
                return False

            return True

        except Exception as e:
            self.get_logger().error(f'Error executing command {cmd_type}: {e}')
            return False

    def move_forward(self):
        """Move robot forward"""
        cmd = Twist()
        cmd.linear.x = 0.3  # m/s
        cmd.angular.z = 0.0
        self.motion_pub.publish(cmd)
        self.robot_state = "moving_forward"

    def move_backward(self):
        """Move robot backward"""
        cmd = Twist()
        cmd.linear.x = -0.3  # m/s
        cmd.angular.z = 0.0
        self.motion_pub.publish(cmd)
        self.robot_state = "moving_backward"

    def turn_left(self):
        """Turn robot left"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # rad/s
        self.motion_pub.publish(cmd)
        self.robot_state = "turning_left"

    def turn_right(self):
        """Turn robot right"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -0.5  # rad/s
        self.motion_pub.publish(cmd)
        self.robot_state = "turning_right"

    def stop_robot(self):
        """Stop robot movement"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.motion_pub.publish(cmd)
        self.robot_state = "idle"

    def navigate_to_location(self, location: str):
        """Navigate to specified location"""
        self.get_logger().info(f'Navigating to {location}')
        # In real implementation, this would use navigation stack
        # For now, just move forward as a placeholder
        self.move_forward()

    def grasp_object(self, obj_name: str):
        """Grasp specified object"""
        self.get_logger().info(f'Attempting to grasp {obj_name}')
        # In real implementation, this would use manipulation stack
        # For now, just wave as a placeholder
        self.wave()

    def wave(self):
        """Perform waving motion"""
        joint_state = JointState()
        joint_state.name = ['right_shoulder_pitch', 'right_elbow_yaw']
        joint_state.position = [0.5, 0.3]  # Example positions for waving
        self.joint_pub.publish(joint_state)

    def sit_down(self):
        """Move to sitting position"""
        joint_state = JointState()
        joint_state.name = ['hip_pitch', 'knee_pitch', 'ankle_pitch']
        joint_state.position = [-0.8, 1.2, -0.4]  # Example sitting positions
        self.joint_pub.publish(joint_state)

    def stand_up(self):
        """Move to standing position"""
        joint_state = JointState()
        joint_state.name = ['hip_pitch', 'knee_pitch', 'ankle_pitch']
        joint_state.position = [0.0, 0.0, 0.0]  # Example standing positions
        self.joint_pub.publish(joint_state)

    def speak_text(self, text: str):
        """Speak text (placeholder - in real system, use TTS)"""
        self.get_logger().info(f'Speaking: {text}')
        # In real implementation, this would use text-to-speech system

    def process_audio(self):
        """Process audio from queue (placeholder for real audio input)"""
        # This would connect to actual audio input in a real implementation
        # For now, this method exists to maintain the threading structure
        pass

    def destroy_node(self):
        """Clean up resources"""
        self.processing_active = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    voice_system = CompleteVoiceSystem()

    try:
        # In a real implementation, this would connect to audio input
        # For demonstration, we'll just keep the node running
        rclpy.spin(voice_system)
    except KeyboardInterrupt:
        voice_system.get_logger().info('Shutting down Complete Voice System')
    finally:
        voice_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This section covered the integration of OpenAI Whisper for voice command processing in humanoid robots, including:
- Whisper model selection and optimization for robotics applications
- Real-time audio processing and transcription
- ROS 2 integration for robotics systems
- Natural language understanding for robot commands
- Performance optimization techniques for edge deployment
- Complete voice command processing pipeline for humanoid robots

The integration of Whisper enables humanoid robots to understand and respond to natural human speech, creating more intuitive and accessible human-robot interaction.