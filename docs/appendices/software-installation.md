---
sidebar_position: 1
---

# Software Installation Guide

## Overview

This guide provides detailed instructions for installing all the software tools required for the Physical AI & Humanoid Robotics course. Follow these steps in order to ensure proper setup.

## Prerequisites

Before beginning the installation process, ensure you have:

- Administrative access to your computer
- Stable internet connection
- At least 50GB of free disk space
- 16GB+ RAM recommended

## Operating System Requirements

### Ubuntu 22.04 LTS (Recommended)
This is the preferred operating system for the course as it provides the best compatibility with all tools:

1. Download Ubuntu 22.04 LTS from [ubuntu.com](https://ubuntu.com/download/desktop)
2. Create a bootable USB drive using Rufus (Windows) or Etcher (Mac/Linux)
3. Install Ubuntu alongside or instead of your current operating system
4. Update the system: `sudo apt update && sudo apt upgrade`

### Alternative: Windows Subsystem for Linux (WSL2)
For Windows users, WSL2 provides a Linux environment with good ROS 2 compatibility:

1. Install WSL2 following [Microsoft's guide](https://docs.microsoft.com/en-us/windows/wsl/install)
2. Install Ubuntu 22.04 distribution from Microsoft Store
3. Configure WSL2 with GUI support if needed

### macOS
While macOS is supported, some tools may have limited functionality:

1. Install Xcode Command Line Tools: `xcode-select --install`
2. Install Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
3. Consider using Docker for some tools to ensure compatibility

## ROS 2 Installation (Humble Hawksbill)

### Ubuntu Installation

```bash
# Set locale
sudo locale-gen en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe

# Add the repository to your sources list
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

### Environment Setup
Add to your `~/.bashrc`:
```bash
source /opt/ros/humble/setup.bash
```

## Gazebo Installation

Gazebo Garden is the recommended version for this course:

```bash
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/gazebo-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo Garden
sudo apt update
sudo apt install gazebo
```

## Unity Installation

### System Requirements
- Windows 10/11 or Ubuntu 20.04/22.04
- Dedicated GPU with DirectX 11 support
- 8GB+ RAM

### Installation Steps
1. Download Unity Hub from [unity.com](https://unity.com/download)
2. Install Unity Hub
3. Use Unity Hub to install Unity 2022.3 LTS version
4. Install the "Linux Build Support" module if on Windows

## NVIDIA Isaac Installation

### Prerequisites
- NVIDIA GPU with CUDA support (GTX 1060/RTX 2060 or better)
- NVIDIA driver 525 or newer
- CUDA 11.8+ installed

### Isaac Sim Installation
1. Sign up for NVIDIA Developer account
2. Download Isaac Sim from [NVIDIA Developer](https://developer.nvidia.com/isaac-sim)
3. Follow the installation guide for your platform
4. Verify installation with: `isaac-sim --version`

### Isaac ROS Packages
```bash
# Create workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Install dependencies
sudo apt update
sudo apt install python3-colcon-common-extensions python3-rosdep

# Clone Isaac ROS packages
cd src
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
# Additional packages as needed for the course

# Build workspace
cd ~/isaac_ros_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

## Python Environment

### Virtual Environment Setup
```bash
# Install Python tools
sudo apt install python3-pip python3-venv

# Create virtual environment
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Install course requirements
pip install --upgrade pip
pip install numpy scipy matplotlib pyyaml
pip install openai-whisper  # For voice processing
```

## Development Tools

### Visual Studio Code
```bash
# Ubuntu
sudo snap install code --classic

# Install ROS extension
code --install-extension ms-iot.vscode-ros
```

### Git Configuration
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "code --wait"
```

## Testing the Installation

### Basic ROS 2 Test
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Test installation
ros2 run demo_nodes_cpp talker
```

### Python Integration Test
```bash
# Source ROS 2 and Python environment
source /opt/ros/humble/setup.bash
source ~/robotics_env/bin/activate

# Test Python ROS 2 interface
python3 -c "import rclpy; print('ROS 2 Python interface working!')"
```

## Troubleshooting

### Common Issues

1. **Permission denied errors**: Ensure you've sourced the ROS 2 setup.bash file
2. **CUDA not found**: Verify NVIDIA drivers and CUDA installation
3. **Package not found**: Update apt repositories and rosdep database
4. **Python import errors**: Activate your virtual environment

### Verification Commands

```bash
# Check ROS 2 installation
echo $ROS_DISTRO
ros2 --version

# Check Python packages
python3 -c "import rclpy; import numpy; print('All packages available')"
```

## Next Steps

Once you've completed the installation:

1. Verify all components work with the test commands above
2. Follow the course setup guide in Week 1 documentation
3. Join the course Discord for support and community