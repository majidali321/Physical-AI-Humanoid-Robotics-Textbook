---
sidebar_position: 3
---

# Week 1 Exercises

## Exercise 1: Environment Setup Verification

### Objective
Verify that your development environment is properly configured for the course.

### Steps
1. Open a terminal and source ROS 2:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Check that ROS 2 is properly installed:
   ```bash
   ros2 --version
   ```

3. Verify Python packages are available:
   ```bash
   python3 -c "import rclpy; print('rclpy:', rclpy.__version__)"
   python3 -c "import numpy; print('numpy:', numpy.__version__)"
   ```

4. Test basic ROS 2 functionality:
   ```bash
   ros2 topic list
   ros2 node list
   ```

### Expected Output
- ROS 2 version number displayed
- Python packages imported successfully
- Empty or default topic/node lists (no errors)

### Submission Requirements
- Screenshot of successful environment verification
- Brief description of any issues encountered and how they were resolved

## Exercise 2: Physical AI Research Exploration

### Objective
Research and analyze recent developments in Physical AI and humanoid robotics.

### Steps
1. Research three recent papers (published within the last 2 years) in Physical AI or humanoid robotics
2. For each paper, identify:
   - Main contribution to the field
   - Physical AI concepts demonstrated
   - Technical approach used
   - Potential applications

3. Write a brief summary (1-2 paragraphs) for each paper

### Requirements
- Papers must be from reputable venues (conferences like ICRA, IROS, RSS, or journals like Science Robotics, IEEE RA-L)
- Include proper citations in academic format
- Total length: 3-4 pages including citations

### Submission Requirements
- Research summary document
- Reflection on how the papers relate to course content

## Exercise 3: ROS 2 Basics Tutorial

### Objective
Complete basic ROS 2 tutorials to familiarize yourself with the framework.

### Steps
1. Complete the ROS 2 beginner tutorials:
   - [Understanding ROS 2 nodes](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Nodes/Understanding-ROS2-Nodes.html)
   - [Understanding ROS 2 topics](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.html)
   - [Writing a simple publisher and subscriber](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html)

2. Modify the publisher/subscriber example to:
   - Publish a custom message type
   - Include a timestamp in the message
   - Add a counter to track message number

3. Create a launch file that starts both nodes simultaneously

### Expected Output
- Working publisher/subscriber nodes
- Custom message type implementation
- Launch file that starts both nodes
- Demonstration of message passing

### Submission Requirements
- Source code files
- Launch file
- Screenshot of running system with messages
- Brief explanation of modifications made

## Exercise 4: Physical AI Concept Analysis

### Objective
Analyze a real-world Physical AI system and identify its components.

### Steps
1. Choose a physical AI system (e.g., Boston Dynamics robot, Tesla Autopilot, Amazon warehouse robots, etc.)

2. Analyze the system by identifying:
   - Perception components (sensors, data processing)
   - Cognition components (planning, learning, reasoning)
   - Action components (locomotion, manipulation, communication)
   - How the system demonstrates embodied intelligence
   - Real-time constraints and how they're addressed

3. Create a system diagram showing the main components and their interactions

4. Discuss potential improvements or extensions to the system

### Requirements
- Analysis should be 2-3 pages
- Include at least 2 diagrams or figures
- Cite at least 3 sources for your analysis

### Submission Requirements
- Analysis document with diagrams
- Source citations
- Reflection on how this system relates to humanoid robotics

## Exercise 5: Development Workspace Setup

### Objective
Set up a proper development workspace for the course projects.

### Steps
1. Create a ROS 2 workspace:
   ```bash
   mkdir -p ~/physical_ai_ws/src
   cd ~/physical_ai_ws
   colcon build
   source install/setup.bash
   ```

2. Create a package for course exercises:
   ```bash
   cd ~/physical_ai_ws/src
   ros2 pkg create --build-type ament_python week1_exercises
   ```

3. Implement a simple node in your package that:
   - Publishes system status information
   - Subscribes to a sensor topic (simulated)
   - Logs important events
   - Follows ROS 2 best practices

4. Create a README.md file explaining your package structure

### Expected Output
- Properly structured ROS 2 workspace
- Functional package with nodes
- Working build and execution
- Proper documentation

### Submission Requirements
- Workspace structure screenshots
- Source code
- README.md file
- Demonstration of working nodes

## Grading Rubric

Each exercise will be graded on the following criteria:

- **Technical Correctness** (40%): Implementation works as expected
- **Code Quality** (25%): Well-structured, documented, and following best practices
- **Analysis Depth** (20%): Thorough understanding and thoughtful analysis
- **Documentation** (15%): Clear explanations and proper formatting

## Submission Guidelines

- Submit all exercises via the course management system
- Include all source code, documentation, and required files
- Follow the naming convention: `week1_exerciseN_yourname.ext`
- Late submissions will be penalized by 10% per day

## Resources

- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [Physical AI Research Papers](https://example.com/physical-ai-papers)
- [Course GitHub Repository](https://github.com/your-organization/physical-ai-textbook)