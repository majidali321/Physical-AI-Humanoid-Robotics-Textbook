---
sidebar_position: 1
---

# Hardware Requirements

## Overview

This document outlines the hardware requirements for the Physical AI & Humanoid Robotics course. Requirements are categorized into essential, recommended, and optional hardware for different aspects of the course.

## Development Environment

### Essential Hardware
- **Computer**: Modern laptop or desktop with 16GB+ RAM
- **Processor**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **Storage**: 256GB+ SSD storage (500GB+ recommended)
- **Operating System**: Linux (Ubuntu 22.04 LTS preferred), Windows 10/11, or macOS

### Recommended Hardware
- **Graphics**: Dedicated GPU with 4GB+ VRAM (NVIDIA GPU recommended for Isaac Sim)
- **Memory**: 32GB RAM for complex simulations
- **Storage**: 1TB+ SSD for simulation environments and datasets

## Simulation Requirements

### Gazebo Simulation
- **CPU**: 4+ cores recommended for real-time physics
- **GPU**: Dedicated graphics card for visual rendering
- **RAM**: 8GB+ available for complex environments
- **OS**: Linux preferred for best Gazebo performance

### Unity Digital Twin
- **CPU**: Multi-core processor (4+ cores)
- **GPU**: Dedicated graphics card with DirectX 11 support
- **RAM**: 16GB+ for complex 3D environments
- **OS**: Windows, macOS, or Linux (with compatibility considerations)

### NVIDIA Isaac Sim
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060/RTX 2060 or better)
- **VRAM**: 6GB+ VRAM for complex scenes
- **OS**: Ubuntu 20.04/22.04 LTS (Windows support limited)
- **CUDA**: CUDA 11.8+ with compatible driver

## Optional Physical Hardware

### For Advanced Projects
- **Humanoid Robot Platform**: Any ROS 2 compatible humanoid robot
- **Development Kit**: NVIDIA Jetson platform for edge AI (Jetson Orin, Xavier)
- **Sensors**: LiDAR, RGB-D cameras, IMUs for real-world validation
- **Actuators**: Servo motors, motor controllers for custom builds

### Recommended Platforms
- **Simulation-Only**: Any computer meeting development requirements
- **Light Robotics**: TurtleBot3, ROSbot, or similar platforms
- **Advanced Robotics**: Unitree Go1, ANYmal, or custom humanoid platforms

## Network Requirements

- **Internet**: Stable connection for package downloads and API access
- **Bandwidth**: 10 Mbps+ for large asset downloads
- **Local Network**: For ROS 2 multi-machine setups (optional)

## Special Considerations

### For Isaac Sim Users
- NVIDIA GPU with RTX support provides best experience
- Ensure adequate cooling for intensive simulations
- Consider cloud GPU instances for machines without suitable hardware

### For Multi-Machine Setups
- Network latency should be &lt;10ms for real-time control
- Use wired connection when possible for robotics applications
- Ensure consistent ROS 2 distribution across all machines

## Cost Estimates

- **Minimum Setup**: $800-1200 (laptop meeting requirements)
- **Recommended Setup**: $1500-2500 (dedicated GPU, 32GB RAM)
- **Professional Setup**: $3000+ (high-end GPU, robotics hardware)

## Troubleshooting

### Performance Issues
- Close unnecessary applications during simulations
- Ensure adequate cooling and ventilation
- Consider reducing simulation complexity for lower-end hardware

### Compatibility Issues
- Use Ubuntu 22.04 LTS for best compatibility with all tools
- Check NVIDIA driver compatibility for Isaac Sim
- Verify ROS 2 distribution compatibility with all components

## Updates and Maintenance

Hardware requirements may evolve as simulation tools advance. Check this document regularly for updates, especially before starting new modules that introduce new simulation environments.