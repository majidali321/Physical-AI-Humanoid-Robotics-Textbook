---
sidebar_position: 3
---

# Week 1: Course Setup

## Overview

This section provides step-by-step instructions for setting up your development environment for the Physical AI & Humanoid Robotics course. Follow these instructions carefully to ensure all tools are properly installed and configured.

## Prerequisites

Before beginning the setup process, ensure you have:

- Administrative access to your computer
- At least 50GB of free disk space
- 16GB+ RAM (32GB recommended)
- Stable internet connection
- Basic familiarity with command line tools

## Operating System Setup

### Ubuntu 22.04 LTS (Recommended)

For the best experience with all course tools, we recommend using Ubuntu 22.04 LTS:

1. **Download Ubuntu**:
   - Visit [ubuntu.com](https://ubuntu.com/download/desktop)
   - Download the 22.04 LTS version

2. **Create Bootable USB**:
   - Use Rufus (Windows) or Etcher (Mac/Linux) to create a bootable USB drive
   - Follow the installation guide for your hardware

3. **System Configuration**:
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade -y

   # Install basic development tools
   sudo apt install build-essential cmake git python3-pip python3-dev
   ```

### Alternative: Windows Subsystem for Linux (WSL2)

If you prefer to keep Windows as your primary OS:

1. **Install WSL2**:
   - Follow Microsoft's [WSL installation guide](https://docs.microsoft.com/en-us/windows/wsl/install)
   - Install Ubuntu 22.04 distribution

2. **Configure WSL2 for ROS**:
   ```bash
   # In WSL2 terminal
   sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'
   ```

## ROS 2 Installation

### Install ROS 2 Humble Hawksbill

```bash
# Set up locale
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8

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

### Set up Environment
Add the following to your `~/.bashrc`:
```bash
source /opt/ros/humble/setup.bash
```

Then reload your environment:
```bash
source ~/.bashrc
```

## Python Environment Setup

### Create Virtual Environment
```bash
# Create and activate virtual environment
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install essential packages
pip install numpy scipy matplotlib pyyaml
```

### Install Additional Course Dependencies
```bash
# In your activated virtual environment
pip install rclpy  # Python ROS 2 client library
pip install openai-whisper  # For voice processing (Week 11+)
```

## Development Tools

### Install Visual Studio Code
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

## Verification Steps

### Test ROS 2 Installation
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Test basic functionality
ros2 --version

# Test Python interface
python3 -c "import rclpy; print('ROS 2 Python interface working!')"
```

### Test Basic Communication
```bash
# Terminal 1: Start a simple publisher
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker

# Terminal 2: Start a subscriber (in a new terminal)
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp listener
```

You should see messages being passed between the talker and listener nodes.

## Troubleshooting Common Issues

### Permission Issues
If you encounter permission errors:
```bash
# Check if ROS is properly sourced
echo $ROS_DISTRO  # Should show "humble"
```

### Python Import Errors
If Python packages are not found:
```bash
# Ensure virtual environment is activated
source ~/robotics_env/bin/activate
which python  # Should show path to your virtual environment
```

### Network Issues
For multi-machine ROS communication:
```bash
# Check ROS domain ID
echo $ROS_DOMAIN_ID  # Set to same value on all machines if needed
```

## Next Steps

Once you've completed the setup:

1. **Complete Week 1 Exercises**: Practice with basic ROS 2 commands
2. **Prepare for Week 2**: Review sensor concepts and data processing
3. **Join Course Community**: Connect with fellow students for support

## Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Ubuntu Installation Guide](https://tutorials.ubuntu.com/tutorial/tutorial-install-ubuntu-desktop)
- [VS Code ROS Extension](https://marketplace.visualstudio.com/items?itemName=ms-iot.vscode-ros)

Your development environment is now ready for the exciting journey ahead in Physical AI and Humanoid Robotics!