---
sidebar_position: 2
---

# Troubleshooting Guide

## Overview

This guide provides solutions to common issues you may encounter during the Physical AI & Humanoid Robotics course. Use this as a reference when you encounter problems with software installations, simulations, or code execution.

## General Troubleshooting Principles

### 1. Check Your Environment
Always verify that your environment is properly configured:
```bash
# Check ROS 2 distribution
echo $ROS_DISTRO

# Check Python environment
which python3
pip list | grep rclpy

# Check workspace sourcing
printenv | grep ROS
```

### 2. Verify Installation
Before troubleshooting specific issues, verify all installations:
```bash
# ROS 2
ros2 --version
ros2 run demo_nodes_cpp talker

# Python packages
python3 -c "import rclpy; import numpy; import cv2"

# Gazebo
gazebo --version
```

### 3. Read Error Messages Carefully
Error messages often contain specific information about what went wrong. Look for:
- Missing packages or dependencies
- Path issues
- Permission problems
- Version conflicts

## ROS 2 Issues

### Nodes Not Communicating
**Symptoms**: Publishers and subscribers not exchanging messages

**Solutions**:
1. Ensure nodes are on the same ROS domain ID:
   ```bash
   export ROS_DOMAIN_ID=0  # Use same value for all terminals
   ```

2. Check network configuration if using multiple machines:
   ```bash
   export ROS_LOCALHOST_ONLY=0  # For multi-machine setups
   ```

3. Verify topic names match exactly (including namespaces)

### Missing ROS Packages
**Symptoms**: "Package not found" errors

**Solutions**:
1. Source your workspace:
   ```bash
   source ~/your_workspace/install/setup.bash
   ```

2. Check if package is properly built:
   ```bash
   ros2 pkg list | grep package_name
   ```

3. Rebuild if necessary:
   ```bash
   cd ~/your_workspace
   colcon build --packages-select package_name
   ```

### Python Import Errors
**Symptoms**: "No module named rclpy" or similar

**Solutions**:
1. Ensure ROS 2 is sourced:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. Check Python path:
   ```bash
   python3 -c "import sys; print(sys.path)"
   ```

3. Install Python packages in the correct environment:
   ```bash
   pip3 install rclpy  # In your virtual environment
   ```

## Gazebo Issues

### Gazebo Won't Start
**Symptoms**: Gazebo crashes immediately or shows no GUI

**Solutions**:
1. Check graphics drivers:
   ```bash
   nvidia-smi  # For NVIDIA GPUs
   glxinfo | grep "OpenGL renderer"  # Check for hardware acceleration
   ```

2. Try software rendering:
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   gazebo
   ```

3. Check for missing models:
   ```bash
   export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models
   ```

### Simulation Performance Issues
**Symptoms**: Low frame rate, stuttering, or lag

**Solutions**:
1. Reduce visual quality in Gazebo settings
2. Close unnecessary applications
3. Check available RAM and CPU usage
4. Consider using faster physics engine (DART vs ODE)

## Unity Issues

### Unity ROS Bridge Problems
**Symptoms**: Unity and ROS 2 cannot communicate

**Solutions**:
1. Verify ROS TCP endpoint settings in Unity
2. Check firewall settings
3. Ensure both Unity and ROS are using the same ROS domain ID
4. Verify ROS TCP Endpoint package is properly installed

### Scene Loading Issues
**Symptoms**: Unity scenes fail to load or crash

**Solutions**:
1. Check system requirements (especially GPU)
2. Update graphics drivers
3. Verify Unity version compatibility
4. Check for asset loading errors in Unity console

## NVIDIA Isaac Issues

### Isaac Sim Not Launching
**Symptoms**: Isaac Sim fails to start or crashes

**Solutions**:
1. Verify NVIDIA driver compatibility:
   ```bash
   nvidia-smi
   # Driver should be 525+ for Isaac Sim 2023.1+
   ```

2. Check CUDA installation:
   ```bash
   nvcc --version
   nvidia-ml-py3 --version
   ```

3. Verify GPU compute capability:
   ```bash
   nvidia-ml-py3 -c "import pynvml; pynvml.nvmlInit(); handle = pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetName(handle))"
   ```

### Isaac ROS Package Issues
**Symptoms**: Isaac ROS nodes fail to start or crash

**Solutions**:
1. Check Isaac ROS package installation:
   ```bash
   ros2 pkg list | grep isaac
   ```

2. Verify CUDA compatibility with Isaac ROS:
   ```bash
   # Check if Isaac ROS nodes are built
   ros2 run --packages-path ~/isaac_ros_ws/src
   ```

3. Check for missing dependencies:
   ```bash
   rosdep check --from-paths ~/isaac_ros_ws/src --ignore-src
   ```

## Voice Processing Issues (Whisper)

### OpenAI Whisper Installation
**Symptoms**: Whisper not available or import errors

**Solutions**:
1. Install with proper dependencies:
   ```bash
   pip install openai-whisper
   # May need to install additional dependencies:
   sudo apt update
   sudo apt install ffmpeg
   ```

2. Check Python version compatibility (requires Python 3.8+)

### Whisper Performance Issues
**Symptoms**: Slow processing or memory errors

**Solutions**:
1. Reduce audio quality for testing:
   ```python
   # Use smaller models for testing
   model = whisper.load_model("tiny")  # Instead of "large"
   ```

2. Check available RAM and CPU usage

## Development Environment Issues

### Virtual Environment Problems
**Symptoms**: Package conflicts or missing dependencies

**Solutions**:
1. Create a clean environment:
   ```bash
   python3 -m venv ~/clean_robotics_env
   source ~/clean_robotics_env/bin/activate
   pip install --upgrade pip
   ```

2. Install packages in correct order (ROS packages first)

### Git Issues
**Symptoms**: Cannot clone repositories or pull changes

**Solutions**:
1. Check Git configuration:
   ```bash
   git config --global user.name
   git config --global user.email
   ```

2. Verify network connectivity:
   ```bash
   git ls-remote https://github.com/your-repo
   ```

3. Check for large file issues:
   ```bash
   git lfs install  # If using Git LFS
   ```

## Performance Optimization

### System Performance
1. Close unnecessary applications
2. Monitor resource usage:
   ```bash
   htop  # CPU and memory usage
   nvidia-smi  # GPU usage (if applicable)
   ```

### Simulation Optimization
1. Reduce simulation frequency
2. Use simplified models for testing
3. Close visualizers when not needed

## Getting Help

### When to Seek Help
- Issues persist after trying troubleshooting steps
- Error messages are unclear
- Need clarification on course content

### Resources
- Course Discord server
- ROS Answers: answers.ros.org
- NVIDIA Isaac forums
- Unity documentation

### Information to Include When Asking for Help
1. Complete error message
2. Operating system and versions
3. Steps to reproduce the issue
4. What you've already tried