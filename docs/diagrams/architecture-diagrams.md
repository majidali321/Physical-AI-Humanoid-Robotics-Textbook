# Architecture Diagrams for Physical AI & Humanoid Robotics

This document contains the key architectural diagrams referenced throughout the textbook, showing the system architecture, component relationships, and data flows for the humanoid robotics system.

## System Architecture Overview

```mermaid
graph TB
    subgraph "User Interaction Layer"
        A[Voice Commands] --> B[Speech Recognition]
        C[Visual Input] --> D[Computer Vision]
    end

    subgraph "Cognitive Processing Layer"
        B --> E[LLM Planning]
        D --> E
        F[Sensor Fusion] --> E
        E --> G[Task Decomposition]
        G --> H[Action Sequencing]
    end

    subgraph "Execution Layer"
        H --> I[Navigation Control]
        H --> J[Manipulation Control]
        H --> K[Posture Control]
    end

    subgraph "Robot Platform"
        I --> L[Humanoid Robot]
        J --> L
        K --> L
        L --> M[Environment]
        M --> F
        M --> C
    end

    subgraph "Safety System"
        N[Safety Monitor] --> I
        N --> J
        N --> K
        L --> N
        M --> N
    end

    style A fill:#cde4ff
    style L fill:#9ffcc0
    style M fill:#f8d7da
    style E fill:#f9c5d1
    style N fill:#fff3cd
```

## Module 1: ROS 2 Architecture

```mermaid
graph LR
    subgraph "ROS 2 Framework"
        A[Node] --> B[Topics]
        A --> C[Services]
        A --> D[Actions]
        B --> E[Message Passing]
        C --> E
        D --> E
    end

    subgraph "Robot Hardware"
        F[Sensor Drivers] --> A
        G[Actuator Controllers] --> A
        H[Communication Interfaces] --> A
    end

    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#e8f5e8
    style G fill:#fff3e0
```

## Module 2: Digital Twin Architecture

```mermaid
graph TB
    subgraph "Physical Robot"
        A[Real Robot] --> B[Real Sensors]
        A --> C[Real Actuators]
        B --> D[Sensor Data]
        C --> D
    end

    subgraph "Simulation Environment"
        E[Gazebo Simulation] --> F[Simulated Robot]
        E --> G[Simulated Sensors]
        E --> H[Simulated Environment]
        F --> I[Simulated Data]
        G --> I
    end

    subgraph "Unity Digital Twin"
        J[Unity Environment] --> K[3D Visualization]
        J --> L[Physics Simulation]
        K --> M[User Interface]
        L --> M
    end

    D <--> N[Data Bridge]
    I <--> N
    M <--> N

    style A fill:#c8e6c9
    style E fill:#e3f2fd
    style J fill:#f3e5f5
    style N fill:#fff3e0
```

## Module 3: AI-Robot Brain Architecture

```mermaid
graph LR
    subgraph "Perception System"
        A[Visual SLAM] --> B[Environment Map]
        C[Object Detection] --> B
        D[Lidar Processing] --> B
    end

    subgraph "Planning System"
        E[Path Planning] --> F[Navigation System]
        G[Task Planning] --> F
        H[Behavior Trees] --> G
    end

    subgraph "Control System"
        I[Locomotion Control] --> J[Bipedal Controller]
        K[Manipulation Control] --> J
        L[Balance Control] --> J
    end

    B --> G
    F --> J

    style A fill:#e8f5e8
    style E fill:#e3f2fd
    style I fill:#fff3e0
    style J fill:#ffebee
```

## Module 4: Vision-Language-Action (VLA) Architecture

```mermaid
graph TB
    subgraph "Input Processing"
        A[Voice Input] --> B[Speech Recognition]
        C[Visual Input] --> D[Computer Vision]
        E[Tactile Input] --> F[Sensor Processing]
    end

    subgraph "Cognitive Processing"
        B --> G[LLM Understanding]
        D --> G
        F --> G
        G --> H[Context Integration]
        H --> I[Plan Generation]
    end

    subgraph "Action Execution"
        I --> J[Navigation Actions]
        I --> K[Manipulation Actions]
        I --> L[Communication Actions]
        J --> M[Robot Execution]
        K --> M
        L --> M
    end

    M --> N[Environment Feedback]
    N --> H

    style A fill:#cde4ff
    style G fill:#f9c5d1
    style M fill:#9ffcc0
    style N fill:#f8d7da
```

## Complete Autonomous System Integration

```mermaid
graph TD
    subgraph "User Interface"
        A[Voice Commands] --> B[Command Processor]
        C[Mobile App] --> B
        D[Gesture Recognition] --> B
    end

    subgraph "Core Intelligence"
        B --> E[LLM Planner]
        F[Memory System] --> E
        G[Learning System] --> E
        E --> H[Task Sequencer]
    end

    subgraph "Perception System"
        I[Camera Array] --> J[Scene Understanding]
        K[Lidar Array] --> J
        L[IMU/Tactile] --> J
        J --> F
    end

    subgraph "Action System"
        H --> M[Navigation Module]
        H --> N[Manipulation Module]
        H --> O[Communication Module]
        M <--> P[Robot Control]
        N <--> P
        O <--> P
    end

    subgraph "Safety & Monitoring"
        Q[Safety Monitor] --> P
        R[Performance Monitor] --> Q
        S[Emergency Handler] --> Q
        P --> R
    end

    P --> T[Physical Robot]
    T --> I
    T --> K
    T --> L

    style A fill:#cde4ff
    style E fill:#f9c5d1
    style P fill:#9ffcc0
    style T fill:#f8d7da
    style Q fill:#fff3cd
```

## ROS 2 Node Architecture

```mermaid
graph LR
    subgraph "Perception Nodes"
        A[Camera Node] --> B[Image Processing Node]
        C[Lidar Node] --> D[Obstacle Detection Node]
        E[IMU Node] --> F[State Estimation Node]
    end

    subgraph "Planning Nodes"
        G[LLM Interface Node] --> H[Task Planning Node]
        I[Navigation Planner Node] --> H
        J[Manipulation Planner Node] --> H
    end

    subgraph "Control Nodes"
        K[Navigation Controller Node] --> L[Move Base Node]
        M[Manipulation Controller Node] --> N[Joint Controller Node]
        O[Posture Controller Node] --> N
    end

    subgraph "Integration Nodes"
        P[Command Router Node] --> A
        P --> G
        P --> K
        Q[State Monitor Node] --> P
        R[Emergency Handler Node] --> P
    end

    B --> G
    D --> G
    F --> G
    H --> L
    H --> N

    style A fill:#e1f5fe
    style G fill:#f3e5f5
    style K fill:#e8f5e8
    style P fill:#fff3e0
    style Q fill:#ffebee
```

## Hardware Architecture

```mermaid
graph TD
    subgraph "Computing Unit"
        A[Main Computer] --> B[GPU for AI]
        A --> C[Real-time Processor]
        A --> D[Communication Module]
    end

    subgraph "Sensor Array"
        E[RGB Camera] --> A
        F[Depth Camera] --> A
        G[2D Lidar] --> A
        H[3D Lidar] --> A
        I[IMU] --> A
        J[Force/Torque Sensors] --> A
        K[Joint Encoders] --> A
    end

    subgraph "Actuation System"
        A --> L[Joint Motors]
        A --> M[Motor Drivers]
        A --> N[Power Management]
        L --> O[Robot Joints]
    end

    subgraph "Communication"
        D --> P[Wi-Fi/Ethernet]
        D --> Q[ROS Bridge]
        P <--> R[External Systems]
    end

    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style L fill:#fff3e0
    style P fill:#f3e5f5
```

## Software Stack Architecture

```mermaid
graph BT
    subgraph "Application Layer"
        A[Humanoid Applications]
        B[Task Planning]
        C[Behavior Control]
    end

    subgraph "AI Layer"
        D[LLM Integration]
        E[Computer Vision]
        F[NLP Processing]
        G[Learning Systems]
    end

    subgraph "Robotics Middleware"
        H[ROS 2 Framework]
        I[Navigation Stack]
        J[Manipulation Stack]
        K[Perception Stack]
    end

    subgraph "Hardware Abstraction"
        L[Hardware Drivers]
        M[Real-time Control]
        N[Sensor Interfaces]
    end

    subgraph "Operating System"
        O[Real-time Linux]
        P[Container System]
    end

    A --> D
    A --> H
    B --> D
    B --> I
    C --> J
    D --> H
    E --> K
    F --> H
    G --> D
    H --> L
    I --> M
    J --> M
    K --> N
    L --> O
    M --> O
    N --> O
    P --> O

    style A fill:#c8e6c9
    style D fill:#f9c5d1
    style H fill:#e1f5fe
    style L fill:#fff3e0
    style O fill:#e8f5e8
```