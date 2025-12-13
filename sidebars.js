// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: [
        'intro',
        'course-overview',
        'setup',
      ],
    },
    {
      type: 'category',
      label: 'Week 1: Physical AI Foundations',
      items: [
        'week1/intro',
        'week1/physical-ai-concepts',
        'week1/course-setup',
        'week1/exercises',
      ],
    },
    {
      type: 'category',
      label: 'Week 2: Sensor Systems',
      items: [
        'week2/intro',
        'week2/sensor-types',
        'week2/ros-integration',
        'week2/exercises',
      ],
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module1/intro',
        {
          type: 'category',
          label: 'Week 3: ROS 2 Architecture',
          items: [
            'module1/week3/architecture',
            'module1/week3/nodes-topics-services',
            'module1/week3/rclpy',
            'module1/week3/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 4: ROS 2 Advanced Topics',
          items: [
            'module1/week4/parameters',
            'module1/week4/lifecycle-nodes',
            'module1/week4/urdf-basics',
            'module1/week4/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 5: URDF and Robot Control',
          items: [
            'module1/week5/detailed-urdf',
            'module1/week5/robot-control',
            'module1/week5/project',
            'module1/week5/exercises',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module2/intro',
        {
          type: 'category',
          label: 'Week 6: Gazebo Simulation',
          items: [
            'module2/week6/gazebo-basics',
            'module2/week6/physics-simulation',
            'module2/week6/sensor-simulation',
            'module2/week6/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 7: Unity Digital Twin',
          items: [
            'module2/week7/unity-setup',
            'module2/week7/environment-building',
            'module2/week7/unity-ros-integration',
            'module2/week7/project',
            'module2/week7/exercises',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module3/intro',
        {
          type: 'category',
          label: 'Week 8: NVIDIA Isaac Introduction',
          items: [
            'module3/week8/isaac-sim',
            'module3/week8/isaac-ros',
            'module3/week8/vslam',
          ],
        },
        {
          type: 'category',
          label: 'Week 9: Navigation Systems',
          items: [
            'module3/week9/nav2',
          ],
        },
        {
          type: 'category',
          label: 'Week 10: AI Decision Making',
          items: [
            'module3/week10/ai-integration',
            'module3/week10/project',
            'module3/week10/exercises',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module4/intro',
        {
          type: 'category',
          label: 'Week 11: Voice Command Systems',
          items: [
            'module4/week11/voice-commands',
            'module4/week11/openai-whisper',
            'module4/week11/voice-integration',
            'module4/week11/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 12: LLM Integration',
          items: [
            'module4/week12/llm-planning',
            'module4/week12/llm-integration',
            'module4/week12/cognitive-architectures',
            'module4/week12/llm-robot-integration',
            'module4/week12/exercises',
          ],
        },
        {
          type: 'category',
          label: 'Week 13: Capstone Project',
          items: [
            'module4/week13/capstone-overview',
            'module4/week13/autonomous-humanoid',
            'module4/week13/project-requirements',
            'module4/week13/final-presentation',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Hardware Requirements',
      items: [
        'hardware/requirements',
      ],
    },
    {
      type: 'category',
      label: 'Assessment Guidelines',
      items: [
        'assessments/guidelines',
      ],
    },
    {
      type: 'category',
      label: 'System Diagrams',
      items: [
        'diagrams/architecture-diagrams',
        'diagrams/ros-architecture',
        'diagrams/ai-architecture',
        'diagrams/navigation-control',
        'diagrams/vla-architecture',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/software-installation',
        'appendices/troubleshooting',
        'appendices/glossary',
      ],
    },
  ],
};

module.exports = sidebars;