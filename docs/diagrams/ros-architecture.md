# ROS 2 Architecture for Humanoid Robotics

This document details the ROS 2 architecture specifically designed for humanoid robotics applications, covering the node structure, message passing, and system integration.

## ROS 2 Node Structure

```mermaid
graph TD
    subgraph "ROS 2 Core"
        A[roscore] --> B[Parameter Server]
        A --> C[Master/Discovery]
        A --> D[Logging System]
    end

    subgraph "Perception Nodes"
        E[Camera Driver] --> F[Image Proc]
        G[Lidar Driver] --> H[Laser Proc]
        I[IMU Driver] --> J[State Est]
        K[Joint States] --> L[Robot State]
    end

    subgraph "AI/Planning Nodes"
        M[LLM Interface] --> N[Task Planner]
        O[Object Detector] --> P[Scene Analyzer]
        Q[Speech Recognizer] --> R[Command Parser]
    end

    subgraph "Control Nodes"
        S[Navigation] --> T[Move Base]
        U[Manipulation] --> V[MoveIt!]
        W[Posture Control] --> X[Balance Ctrl]
    end

    subgraph "Integration Nodes"
        Y[Command Router] --> Z[State Monitor]
        AA[Emergency Handler] --> Y
        BB[TF Manager] --> Y
    end

    F --> N
    H --> N
    J --> N
    L --> N
    P --> N
    R --> N
    N --> T
    N --> V
    N --> X
    T --> BB
    V --> BB
    X --> BB

    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style M fill:#f3e5f5
    style S fill:#fff3e0
    style Y fill:#ffebee
```

## Message Flow Architecture

```mermaid
graph LR
    subgraph "Sensor Data Flow"
        A[sensor_msgs/Image] --> B[image_transport]
        C[sensor_msgs/LaserScan] --> D[laser_filters]
        E[sensor_msgs/Imu] --> F[robot_localization]
        G[sensor_msgs/JointState] --> H[robot_state_publisher]
    end

    subgraph "Planning Data Flow"
        I[std_msgs/String] --> J[nlp_pipeline]
        K[geometry_msgs/Pose] --> L[move_base]
        M[moveit_msgs/RobotTrajectory] --> N[trajectory_execution]
    end

    subgraph "Control Data Flow"
        O[geometry_msgs/Twist] --> P[base_controller]
        Q[trajectory_msgs/JointTrajectory] --> R[joint_trajectory_controller]
        S[std_msgs/Float64] --> T[effort_controller]
    end

    B --> I
    D --> K
    F --> K
    H --> K
    J --> K
    L --> O
    L --> Q
    L --> S
    P --> A
    R --> A
    T --> A

    style A fill:#e8f5e8
    style I fill:#e1f5fe
    style O fill:#fff3e0
```

## Service and Action Architecture

```mermaid
graph TD
    subgraph "Services"
        A[Get Map Service] -.-> B[Map Server]
        C[Set Pose Service] -.-> D[AMCL]
        E[Plan Path Service] -.-> F[Global Planner]
        G[Execute Traj Service] -.-> H[Trajectory Controller]
    end

    subgraph "Actions"
        I[MoveBase Action] ==> J[Move Base Server]
        K[PickPlace Action] ==> L[Grasp Server]
        M[FollowJoint Action] ==> N[Joint Controller]
        O[LookAt Action] ==> P[Head Controller]
    end

    subgraph "Topics"
        Q[cmd_vel] --> R[Base Controller]
        S[scan] --> T[Obstacle Detector]
        U[joint_states] --> V[State Estimator]
        W[tf/tf_static] --> X[TF Tree]
    end

    J --> R
    L --> N
    N --> V
    P --> V
    T --> J
    X --> J
    X --> L
    X --> P

    style A fill:#e1f5fe
    style I fill:#c8e6c9
    style Q fill:#fff3e0
```

## Hardware Interface Architecture

```mermaid
graph LR
    subgraph "Hardware Abstraction Layer"
        A[Hardware Interface] --> B[Joint State Interface]
        A --> C[Joint Command Interface]
        A --> D[Sensor Interface]
        A --> E[Actuator Interface]
    end

    subgraph "Real Robot Interface"
        F[Real Robot Driver] --> G[Motor Controllers]
        H[Real Sensor Driver] --> I[Physical Sensors]
        G --> J[Physical Joints]
        I --> K[Physical Environment]
    end

    subgraph "Simulation Interface"
        L[Gazebo Plugin] --> M[Simulated Robot]
        N[Gazebo Sensors] --> O[Simulated Sensors]
        M --> P[Simulated Joints]
        O --> Q[Simulated Environment]
    end

    B --> F
    C --> F
    D --> H
    E --> F
    B --> L
    C --> L
    D --> N
    E --> L

    style A fill:#e1f5fe
    style F fill:#c8e6c9
    style L fill:#f3e5f5
```

## Communication Patterns

```mermaid
graph TD
    subgraph "Publisher-Subscriber Pattern"
        A[Sensor Node] -->|sensor_msgs| B[Perception Node]
        C[State Node] -->|nav_msgs| D[Planning Node]
        E[Cmd Node] -->|geometry_msgs| F[Control Node]
    end

    subgraph "Service Pattern"
        G[Request Node] -.->|Request| H[Service Node]
        H -.->|Response| G
    end

    subgraph "Action Pattern"
        I[Client Node] ==>|Goal| J[Action Server]
        J ==>|Feedback| I
        J ==>|Result| I
    end

    subgraph "Transform Pattern"
        K[TF Publisher] ~~~ L[TF Tree]
        L ~~~ M[TF Subscriber]
    end

    style A fill:#e8f5e8
    style G fill:#e1f5fe
    style I fill:#fff3e0
    style K fill:#f3e5f5
```

## Node Lifecycle Management

```mermaid
stateDiagram-v2
    [*] --> Unconfigured
    Unconfigured --> Inactive : configure()
    Inactive --> Active : activate()
    Active --> Inactive : deactivate()
    Inactive --> Unconfigured : cleanup()
    Unconfigured --> Finalized : shutdown()
    Active --> Error : error()
    Error --> Unconfigured : reset()
    Error --> [*]
```

## Package Organization

```
humanoid_robot/
├── perception/
│   ├── camera_driver/
│   ├── lidar_processing/
│   └── object_detection/
├── planning/
│   ├── task_planning/
│   ├── motion_planning/
│   └── llm_interface/
├── control/
│   ├── base_controller/
│   ├── joint_controllers/
│   └── balance_control/
├── integration/
│   ├── command_router/
│   ├── state_monitor/
│   └── emergency_handler/
└── utils/
    ├── tf_utils/
    ├── param_utils/
    └── diag_utils/
```

## Launch File Architecture

```mermaid
graph TD
    A[main.launch.py] --> B[perception.launch.py]
    A --> C[planning.launch.py]
    A --> D[control.launch.py]
    A --> E[integration.launch.py]
    B --> F[camera.launch.py]
    B --> G[lidar.launch.py]
    C --> H[llm.launch.py]
    C --> I[navigation.launch.py]
    D --> J[controllers.launch.py]
    E --> K[monitoring.launch.py]
    E --> L[emergency.launch.py]

    style A fill:#e1f5fe
    style B fill:#c8e6c9
    style C fill:#f3e5f5
    style D fill:#fff3e0
    style E fill:#ffebee
```

## Parameter Management

```mermaid
graph LR
    subgraph "Parameter Sources"
        A[Default Params] --> B[Parameter Server]
        C[Launch File Params] --> B
        D[Config File Params] --> B
        E[Runtime Params] --> B
    end

    subgraph "Parameter Usage"
        B --> F[Perception Nodes]
        B --> G[Planning Nodes]
        B --> H[Control Nodes]
        B --> I[Integration Nodes]
    end

    F --> J[Sensor Parameters]
    G --> K[Planning Parameters]
    H --> L[Control Parameters]
    I --> M[System Parameters]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style F fill:#c8e6c9
```