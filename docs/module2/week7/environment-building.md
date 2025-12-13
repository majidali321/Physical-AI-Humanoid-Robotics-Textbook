---
sidebar_position: 2
---

# Week 7: Environment Building in Unity

## Overview

Creating realistic and functional environments is crucial for effective robotics simulation. This section covers advanced techniques for building 3D environments in Unity specifically designed for humanoid robotics applications. You'll learn to create indoor and outdoor scenes, configure physics properties, and optimize environments for real-time simulation.

## Learning Objectives

By the end of this section, you will be able to:

- Design and build complex indoor and outdoor environments
- Configure realistic physics properties for different surfaces
- Create modular and reusable environment components
- Optimize environments for real-time robotics simulation
- Implement dynamic elements and interactive objects

## Indoor Environment Design

### Planning Indoor Spaces

When designing indoor environments for humanoid robots, consider:

- **Navigation Space**: Ensure sufficient space for robot movement and turning
- **Obstacle Placement**: Realistic furniture and objects that robots might encounter
- **Surface Properties**: Different materials affecting robot mobility
- **Lighting Conditions**: Various lighting scenarios for perception testing

### Creating Rooms and Corridors

```csharp
using UnityEngine;

public class RoomBuilder : MonoBehaviour
{
    public float roomWidth = 10f;
    public float roomDepth = 8f;
    public float roomHeight = 3f;
    public float wallThickness = 0.2f;

    void Start()
    {
        BuildRoom();
    }

    void BuildRoom()
    {
        // Create floor
        CreatePlane("Floor", new Vector3(0, 0, 0), new Vector3(roomWidth, 1, roomDepth));

        // Create ceiling
        CreatePlane("Ceiling", new Vector3(0, roomHeight, 0), new Vector3(roomWidth, 1, roomDepth));

        // Create walls
        CreateWall("Wall_Left", new Vector3(-roomWidth/2, roomHeight/2, 0), new Vector3(wallThickness, roomHeight, roomDepth));
        CreateWall("Wall_Right", new Vector3(roomWidth/2, roomHeight/2, 0), new Vector3(wallThickness, roomHeight, roomDepth));
        CreateWall("Wall_Front", new Vector3(0, roomHeight/2, -roomDepth/2), new Vector3(roomWidth, roomHeight, wallThickness));
        CreateWall("Wall_Back", new Vector3(0, roomHeight/2, roomDepth/2), new Vector3(roomWidth, roomHeight, wallThickness));
    }

    GameObject CreatePlane(string name, Vector3 position, Vector3 scale)
    {
        GameObject plane = GameObject.CreatePrimitive(PrimitiveType.Cube);
        plane.name = name;
        plane.transform.position = position;
        plane.transform.localScale = scale;

        // Remove mesh collider and add box collider for better performance
        DestroyImmediate(plane.GetComponent<MeshCollider>());
        plane.AddComponent<BoxCollider>();

        return plane;
    }

    GameObject CreateWall(string name, Vector3 position, Vector3 scale)
    {
        GameObject wall = CreatePlane(name, position, scale);

        // Add realistic material properties
        PhysicMaterial physicMat = new PhysicMaterial();
        physicMat.staticFriction = 0.5f;
        physicMat.dynamicFriction = 0.4f;
        physicMat.bounciness = 0.1f;

        BoxCollider boxCollider = wall.GetComponent<BoxCollider>();
        if (boxCollider != null)
        {
            boxCollider.material = physicMat;
        }

        return wall;
    }
}
```

### Adding Furniture and Obstacles

```csharp
using UnityEngine;

public class FurniturePlacer : MonoBehaviour
{
    public GameObject[] furniturePrefabs; // Tables, chairs, etc.
    public Transform[] spawnPoints;
    public float spacing = 1.0f;

    void Start()
    {
        PlaceFurniture();
    }

    void PlaceFurniture()
    {
        for (int i = 0; i < spawnPoints.Length; i++)
        {
            if (furniturePrefabs.Length > 0)
            {
                GameObject prefab = furniturePrefabs[i % furniturePrefabs.Length];
                Vector3 spawnPos = spawnPoints[i].position;

                // Randomize rotation for more natural placement
                Quaternion rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);

                GameObject furniture = Instantiate(prefab, spawnPos, rotation);

                // Add physics properties suitable for robot interaction
                Rigidbody rb = furniture.GetComponent<Rigidbody>();
                if (rb != null)
                {
                    rb.mass = Random.Range(5f, 20f); // Realistic furniture weights
                    rb.interpolation = RigidbodyInterpolation.Interpolate; // Smoother physics
                }
            }
        }
    }
}
```

### Indoor Lighting Setup

For realistic indoor lighting:

```csharp
using UnityEngine;

public class IndoorLightingSetup : MonoBehaviour
{
    public Light[] ceilingLights;
    public float baseIntensity = 1.0f;
    public Color lightColor = Color.white;

    void Start()
    {
        SetupLighting();
    }

    void SetupLighting()
    {
        foreach (Light light in ceilingLights)
        {
            light.color = lightColor;
            light.intensity = baseIntensity;
            light.range = 10f; // Adjust based on room size
            light.spotAngle = 60f; // For spotlights
            light.shadows = LightShadows.Soft; // Realistic shadows
        }
    }
}
```

## Outdoor Environment Design

### Terrain Creation

Unity's terrain system is ideal for outdoor robotics environments:

1. **Create Terrain**:
   - Right-click in Hierarchy → 3D Object → Terrain
   - Adjust terrain size in the Terrain component

2. **Sculpting Tools**:
   - **Raise/Lower**: Create hills and valleys
   - **Smooth**: Smooth terrain transitions
   - **Paint**: Add different surface materials

3. **Adding Details**:
   - **Trees**: Place trees with appropriate density
   - **Grass**: Add grass for natural appearance
   - **Details**: Flowers, rocks, and other small objects

### Outdoor Lighting

```csharp
using UnityEngine;

public class OutdoorLighting : MonoBehaviour
{
    public Light sunLight;
    public AnimationCurve dayNightCycle;

    void Start()
    {
        SetupOutdoorLighting();
    }

    void SetupOutdoorLighting()
    {
        if (sunLight != null)
        {
            sunLight.type = LightType.Directional;
            sunLight.color = Color.white;
            sunLight.intensity = 1.0f;
            sunLight.shadows = LightShadows.Soft;
            sunLight.shadowStrength = 0.8f;

            // Set sun angle for realistic outdoor lighting
            sunLight.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }
    }

    // For day/night cycle simulation
    void Update()
    {
        float timeOfDay = (Mathf.Sin(Time.time * 0.1f) + 1) / 2; // 0-1 cycle
        float intensity = dayNightCycle.Evaluate(timeOfDay);

        if (sunLight != null)
        {
            sunLight.intensity = intensity;
        }
    }
}
```

### Weather and Environmental Effects

```csharp
using UnityEngine;

public class WeatherController : MonoBehaviour
{
    public GameObject rainSystem;
    public GameObject fogSystem;
    public float rainIntensity = 0f;

    void Update()
    {
        // Example: Random weather changes
        if (Random.value < 0.001f) // Low probability event
        {
            ToggleRain();
        }
    }

    void ToggleRain()
    {
        if (rainSystem != null)
        {
            rainSystem.SetActive(!rainSystem.activeSelf);
        }
    }

    public void SetRainIntensity(float intensity)
    {
        rainIntensity = Mathf.Clamp01(intensity);
        if (rainSystem != null)
        {
            var ps = rainSystem.GetComponent<ParticleSystem>();
            var emission = ps.emission;
            emission.rateOverTime = rainIntensity * 1000f; // Scale to desired range
        }
    }
}
```

## Physics Material Configuration

### Creating Realistic Surface Properties

```csharp
using UnityEngine;

[CreateAssetMenu(fileName = "PhysicsMaterial", menuName = "Robotics/Physics Material")]
public class PhysicsMaterialConfig : ScriptableObject
{
    [Header("Friction Properties")]
    public float staticFriction = 0.5f;
    public float dynamicFriction = 0.4f;

    [Header("Bounce Properties")]
    public float bounciness = 0.1f;

    [Header("Robot Interaction")]
    public bool isSlippery = false; // Affects robot foot placement
    public float tractionCoefficient = 1.0f; // Affects robot movement

    public PhysicMaterial CreatePhysicMaterial()
    {
        PhysicMaterial material = new PhysicMaterial();
        material.staticFriction = staticFriction;
        material.dynamicFriction = dynamicFriction;
        material.bounciness = bounciness;
        return material;
    }
}
```

### Common Surface Types for Robotics

```csharp
using UnityEngine;

public class SurfaceManager : MonoBehaviour
{
    [System.Serializable]
    public class SurfaceType
    {
        public string name;
        public PhysicMaterial material;
        public float robotSpeedModifier = 1.0f; // How surface affects robot speed
        public bool isSafeForWalking = true; // Whether robot can safely walk on surface
    }

    public SurfaceType[] surfaceTypes;

    void Start()
    {
        CreateCommonSurfaces();
    }

    void CreateCommonSurfaces()
    {
        // Create materials for different surfaces
        PhysicMaterial wood = CreateMaterial("Wood", 0.6f, 0.5f, 0.1f);
        PhysicMaterial metal = CreateMaterial("Metal", 0.4f, 0.3f, 0.05f);
        PhysicMaterial carpet = CreateMaterial("Carpet", 0.8f, 0.7f, 0.05f);
        PhysicMaterial ice = CreateMaterial("Ice", 0.1f, 0.05f, 0.1f);

        surfaceTypes = new SurfaceType[]
        {
            new SurfaceType { name = "Wood", material = wood, robotSpeedModifier = 1.0f, isSafeForWalking = true },
            new SurfaceType { name = "Metal", material = metal, robotSpeedModifier = 0.9f, isSafeForWalking = true },
            new SurfaceType { name = "Carpet", material = carpet, robotSpeedModifier = 0.8f, isSafeForWalking = true },
            new SurfaceType { name = "Ice", material = ice, robotSpeedModifier = 0.5f, isSafeForWalking = false }
        };
    }

    PhysicMaterial CreateMaterial(string name, float staticFric, float dynamicFric, float bounce)
    {
        PhysicMaterial material = new PhysicMaterial(name);
        material.staticFriction = staticFric;
        material.dynamicFriction = dynamicFric;
        material.bounciness = bounce;
        return material;
    }

    public PhysicMaterial GetMaterialForSurface(string surfaceName)
    {
        foreach (SurfaceType surface in surfaceTypes)
        {
            if (surface.name == surfaceName)
            {
                return surface.material;
            }
        }
        return surfaceTypes[0].material; // Default to first material
    }
}
```

## Modular Environment Building

### Creating Reusable Components

```csharp
using UnityEngine;

public class ModularRoomComponent : MonoBehaviour
{
    [Header("Component Configuration")]
    public string componentType; // "wall", "door", "window", etc.
    public Vector3 dimensions = Vector3.one;
    public bool isInteractive = false;

    [Header("Connection Points")]
    public Transform[] connectionPoints;

    void Start()
    {
        ConfigureComponent();
    }

    void ConfigureComponent()
    {
        // Apply dimensions
        transform.localScale = dimensions;

        // Add appropriate physics and colliders
        if (GetComponent<Collider>() == null)
        {
            gameObject.AddComponent<BoxCollider>();
        }

        // Configure based on type
        switch (componentType)
        {
            case "wall":
                ConfigureWall();
                break;
            case "door":
                ConfigureDoor();
                break;
            case "window":
                ConfigureWindow();
                break;
        }
    }

    void ConfigureWall()
    {
        // Add realistic wall properties
        Rigidbody rb = gameObject.AddComponent<Rigidbody>();
        rb.mass = dimensions.x * dimensions.y * dimensions.z * 100f; // Density-based mass
        rb.isKinematic = true; // Walls don't move
    }

    void ConfigureDoor()
    {
        // Make door hinged
        HingeJoint hinge = gameObject.AddComponent<HingeJoint>();
        hinge.axis = Vector3.up;
        hinge.useLimits = true;
        JointLimits limits = hinge.limits;
        limits.min = 0f;
        limits.max = 90f;
        hinge.limits = limits;
    }

    void ConfigureWindow()
    {
        // Make window transparent and non-collidable (for robot path planning)
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material.color = new Color(1, 1, 1, 0.3f); // Semi-transparent
        }
    }
}
```

### Procedural Environment Generation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class ProceduralEnvironmentGenerator : MonoBehaviour
{
    public int gridSize = 10;
    public float cellSize = 2.0f;
    public GameObject[] buildingPrefabs;
    public GameObject[] obstaclePrefabs;

    void Start()
    {
        GenerateEnvironment();
    }

    void GenerateEnvironment()
    {
        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                Vector3 position = new Vector3(x * cellSize, 0, z * cellSize);

                // Randomly place buildings
                if (Random.value > 0.7f && buildingPrefabs.Length > 0)
                {
                    int buildingIndex = Random.Range(0, buildingPrefabs.Length);
                    GameObject building = Instantiate(buildingPrefabs[buildingIndex], position, Quaternion.identity);

                    // Randomize building rotation
                    building.transform.rotation = Quaternion.Euler(0, Random.Range(0, 4) * 90, 0);
                }
                // Randomly place obstacles
                else if (Random.value > 0.9f && obstaclePrefabs.Length > 0)
                {
                    int obstacleIndex = Random.Range(0, obstaclePrefabs.Length);
                    Instantiate(obstaclePrefabs[obstacleIndex], position, Quaternion.identity);
                }
            }
        }
    }
}
```

## Optimization Techniques

### Level of Detail (LOD) for Robotics

```csharp
using UnityEngine;

public class RoboticsLODManager : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public GameObject[] objects; // Objects to show at this level
        public float distance; // Distance threshold
    }

    public LODLevel[] lodLevels;
    public Transform viewer; // Robot's camera or position
    private int currentLOD = 0;

    void Start()
    {
        if (viewer == null)
        {
            viewer = Camera.main.transform; // Default to main camera
        }
    }

    void Update()
    {
        UpdateLOD();
    }

    void UpdateLOD()
    {
        if (viewer == null) return;

        float distance = Vector3.Distance(transform.position, viewer.position);

        // Find appropriate LOD level
        int newLOD = 0;
        for (int i = 0; i < lodLevels.Length; i++)
        {
            if (distance <= lodLevels[i].distance)
            {
                newLOD = i;
                break;
            }
        }

        // Update visibility based on LOD
        for (int i = 0; i < lodLevels.Length; i++)
        {
            bool visible = (i == newLOD);
            foreach (GameObject obj in lodLevels[i].objects)
            {
                if (obj != null)
                {
                    obj.SetActive(visible);
                }
            }
        }

        currentLOD = newLOD;
    }
}
```

### Occlusion Culling Setup

For large environments, implement occlusion culling:

1. **Mark Static Objects**:
   - Select all static environment objects
   - Check "Static" in the Inspector
   - Select "Occluder Static" and "Occludee Static"

2. **Bake Occlusion Culling**:
   - Go to `Window > Rendering > Occlusion Culling`
   - Click "Bake" to compute visibility

## Dynamic Elements and Interactive Objects

### Moving Obstacles

```csharp
using UnityEngine;

public class MovingObstacle : MonoBehaviour
{
    public Transform[] waypoints;
    public float speed = 2.0f;
    public float waitTime = 1.0f;

    private int currentWaypoint = 0;
    private bool movingForward = true;

    void Update()
    {
        if (waypoints.Length == 0) return;

        MoveToWaypoint();
    }

    void MoveToWaypoint()
    {
        if (currentWaypoint >= waypoints.Length) return;

        Transform target = waypoints[currentWaypoint];
        transform.position = Vector3.MoveTowards(transform.position, target.position, speed * Time.deltaTime);

        // Check if reached waypoint
        if (Vector3.Distance(transform.position, target.position) < 0.1f)
        {
            Invoke("ChangeWaypoint", waitTime);
        }
    }

    void ChangeWaypoint()
    {
        if (movingForward)
        {
            currentWaypoint++;
            if (currentWaypoint >= waypoints.Length)
            {
                currentWaypoint = waypoints.Length - 2;
                movingForward = false;
            }
        }
        else
        {
            currentWaypoint--;
            if (currentWaypoint < 0)
            {
                currentWaypoint = 1;
                movingForward = true;
            }
        }
    }
}
```

### Interactive Elements

```csharp
using UnityEngine;

public class InteractiveObject : MonoBehaviour
{
    public string objectType; // "button", "lever", "door", etc.
    public bool isActivated = false;
    public GameObject linkedObject; // Object this affects

    void OnTriggerEnter(Collider other)
    {
        // Check if triggered by robot
        if (other.CompareTag("Robot"))
        {
            Activate();
        }
    }

    public void Activate()
    {
        isActivated = !isActivated;

        switch (objectType)
        {
            case "button":
                OnButtonPressed();
                break;
            case "lever":
                OnLeverToggled();
                break;
            case "door":
                OnDoorToggled();
                break;
        }
    }

    void OnButtonPressed()
    {
        // Change appearance
        GetComponent<Renderer>().material.color = isActivated ? Color.green : Color.red;

        // Activate linked object if exists
        if (linkedObject != null)
        {
            linkedObject.SetActive(isActivated);
        }
    }

    void OnLeverToggled()
    {
        // Rotate lever
        transform.Rotate(Vector3.forward, isActivated ? 90 : -90);
    }

    void OnDoorToggled()
    {
        // Animate door opening/closing
        if (linkedObject != null)
        {
            // Add door animation logic here
        }
    }
}
```

## Best Practices for Robotics Environments

### Performance Considerations

1. **Polygon Count**: Keep polygon counts reasonable for real-time simulation
2. **Texture Optimization**: Use appropriate texture sizes and compression
3. **Light Baking**: Bake static lighting when possible
4. **Object Pooling**: Reuse objects instead of constantly creating/destroying

### Realism vs. Performance

Balance realism with performance requirements:
- Use simplified collision meshes
- Implement LOD systems
- Optimize rendering for distant objects
- Use efficient physics materials

### Testing and Validation

1. **Navigation Testing**: Ensure robots can navigate through all areas
2. **Physics Validation**: Test that objects behave realistically
3. **Performance Testing**: Monitor frame rates with target robots
4. **Edge Case Testing**: Test with various robot sizes and capabilities

## Summary

Creating effective environments for humanoid robotics simulation requires careful consideration of both visual realism and physical accuracy. By understanding Unity's environment building tools and implementing proper physics configurations, you can create simulation environments that accurately reflect real-world conditions for robot testing and development. Modular design principles and optimization techniques ensure that your environments remain performant and maintainable as they grow in complexity.