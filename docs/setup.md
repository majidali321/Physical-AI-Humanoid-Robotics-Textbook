---
sidebar_position: 3
---

# Setup Guide

## Prerequisites

Before beginning this course, ensure your system meets the following requirements:

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space minimum
- **GPU**: Dedicated GPU with CUDA support recommended for NVIDIA Isaac components
- **Processor**: Multi-core processor (Intel i7 or equivalent AMD recommended)

### Software Prerequisites
- Git version control system
- Python 3.8 or higher
- Node.js and npm (for Docusaurus documentation)
- Docker (for containerized simulation environments)

## Software Installation

### 1. ROS 2 Installation

Install ROS 2 Humble Hawksbill following the official installation guide:

```bash
# Add ROS 2 repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2
sudo apt install python3-colcon-common-extensions
sudo rosdep init
rosdep update
```

Source ROS 2 in your shell:
```bash
source /opt/ros/humble/setup.bash
```

### 2. Gazebo Installation

Install Gazebo Garden (or compatible version):
```bash
curl -sSL http://get.gazebosim.org | sh
```

### 3. Python Dependencies

Create a virtual environment and install Python dependencies:
```bash
python3 -m venv physical_ai_env
source physical_ai_env/bin/activate  # On Windows: physical_ai_env\Scripts\activate
pip install --upgrade pip
pip install rclpy numpy scipy matplotlib pybullet openai-whisper
```

### 4. Unity Installation

Download and install Unity Hub, then install Unity 2022.3 LTS or later. Install the ROS-TCP-Endpoint package for ROS integration.

### 5. NVIDIA Isaac Installation

Follow the NVIDIA Isaac Sim installation guide for your platform. Ensure you have CUDA-compatible hardware and drivers installed.

### 6. Docusaurus Documentation Setup

Install Node.js and npm, then set up the documentation:
```bash
npm install
npm run start
```

## Verification Steps

After installation, verify each component works:

### ROS 2 Verification
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
```

### Python Verification
```bash
python3 -c "import rclpy; print('rclpy imported successfully')"
python3 -c "import numpy; print('numpy imported successfully')"
```

### Gazebo Verification
```bash
gz sim
```

## Course Repository Setup

Clone the course repository and set up your workspace:
```bash
git clone https://github.com/your-organization/physical-ai-textbook.git
cd physical-ai-textbook
source /opt/ros/humble/setup.bash
```

## Troubleshooting Common Issues

### ROS 2 Environment Not Found
- Ensure you've sourced the ROS 2 setup file: `source /opt/ros/humble/setup.bash`
- Add this to your `.bashrc` file to make it permanent

### Python Package Import Errors
- Ensure you're using the correct virtual environment
- Check that all required packages are installed

### Gazebo Not Launching
- Verify your graphics drivers are up to date
- Check that you have sufficient system resources

### Unity-ROS Connection Issues
- Ensure the ROS-TCP-Endpoint is running
- Verify IP addresses and ports are correctly configured

## Next Steps

Once you've completed the setup, proceed to Week 1: Physical AI Foundations to begin the course. Each week will build upon the previous week's concepts and implementations.