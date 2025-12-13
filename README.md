# Physical AI & Humanoid Robotics Textbook

A comprehensive 13-week course covering the complete stack of Physical AI and Humanoid Robotics, from foundational concepts to advanced AI integration.

## Course Overview

This textbook provides a structured learning path through 4 main modules:

- **Module 1: The Robotic Nervous System (ROS 2)** - Weeks 3-5
  - ROS 2 architecture, nodes, topics, services
  - Python integration with rclpy
  - URDF for humanoid robots

- **Module 2: The Digital Twin (Gazebo & Unity)** - Weeks 6-7
  - Physics simulation in Gazebo
  - Environment building in Unity
  - Sensor simulation (LiDAR, Depth Cameras, IMUs)

- **Module 3: The AI-Robot Brain (NVIDIA Isaac™)** - Weeks 8-10
  - NVIDIA Isaac Sim and Isaac ROS
  - VSLAM and navigation
  - Nav2 for bipedal movement

- **Module 4: Vision-Language-Action (VLA)** - Weeks 11-13
  - Voice commands with OpenAI Whisper
  - LLM-based cognitive planning
  - Capstone project: Autonomous Humanoid

**Introductory Weeks (1-2):** Physical AI foundations and sensor systems

## Features

- **Docusaurus-based**: Modern, responsive documentation
- **Mobile-friendly**: Accessible on all devices
- **GitHub Pages**: Easy deployment and hosting
- **Practical Focus**: Hands-on exercises and projects
- **Industry Tools**: Real-world technology stack
- **Progressive Learning**: Carefully structured difficulty progression
- **Mermaid Diagrams**: Visual representations of concepts
- **Dark Mode**: Support for light/dark theme preferences

## Technology Stack

- **Documentation**: Docusaurus v3
- **Language**: JavaScript/React
- **Content**: Markdown/MDX
- **Code Highlighting**: Prism
- **Diagrams**: Mermaid
- **Robotics Framework**: ROS 2 (Humble Hawksbill)
- **Simulation**: Gazebo, Unity
- **AI/ML**: NVIDIA Isaac Sim, Isaac ROS, OpenAI Whisper
- **Programming**: Python 3.8+

## Local Development

### Prerequisites

- Node.js (version 18 or higher)
- npm or yarn package manager

### Installation

```bash
npm install
```

### Local Development

```bash
npm start
```

This command starts a local development server and opens the documentation in your browser. Most changes are reflected live without requiring a restart.

### Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static hosting service.

### Deployment

The site is configured for deployment to GitHub Pages. Use the deploy script:

```bash
npm run deploy
```

## Project Structure

```
.
├── docs/                    # Course content organized by modules and weeks
│   ├── intro.md            # Introduction to the textbook
│   ├── course-overview.md  # Complete course overview
│   ├── setup.md            # Environment setup guide
│   ├── week1/              # Week 1 content: Physical AI foundations
│   ├── week2/              # Week 2 content: Sensor systems
│   ├── module1/            # Module 1: The Robotic Nervous System
│   ├── module2/            # Module 2: The Digital Twin
│   ├── module3/            # Module 3: The AI-Robot Brain
│   ├── module4/            # Module 4: Vision-Language-Action
│   ├── hardware/           # Hardware requirements
│   ├── assessments/        # Assessment guidelines
│   └── appendices/         # Additional reference materials
├── src/                    # Custom React components and CSS
├── static/                 # Static assets like images and PDFs
├── docusaurus.config.js    # Site configuration
├── sidebars.js             # Navigation structure
├── package.json            # Dependencies and scripts
└── README.md               # This file
```

## Contributing

This textbook is designed for educators and students in Physical AI and Humanoid Robotics. Contributions are welcome for improving content, examples, and exercises.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Deployment

The textbook is configured for GitHub Pages deployment. The following settings are pre-configured:

- **Organization Name**: your-organization (to be customized)
- **Project Name**: physical-ai-textbook
- **Deployment Branch**: gh-pages

To deploy to your own GitHub Pages:
1. Update the `organizationName` and `projectName` in `docusaurus.config.js`
2. Update the GitHub URL in `docusaurus.config.js`
3. Run `npm run deploy`

## License

[Specify license type here]