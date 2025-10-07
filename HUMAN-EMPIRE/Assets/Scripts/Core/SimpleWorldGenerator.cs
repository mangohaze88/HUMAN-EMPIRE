using UnityEngine;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Simple, clean procedural world generator
    /// Creates organized, visually appealing lands with proper spacing
    /// </summary>
    public class SimpleWorldGenerator : MonoBehaviour
    {
        [Header("World Settings")]
        [SerializeField] private int numberOfLands = 8;
        [SerializeField] private float circleRadius = 20f;
        [SerializeField] private bool generateOnStart = true;

        [Header("Visual Settings")]
        [SerializeField] private float landSize = 5f;
        [SerializeField] private float landHeight = 1f;

        private void Start()
        {
            if (generateOnStart)
            {
                GenerateCleanWorld();
            }
        }

        /// <summary>
        /// Generate a clean, organized world
        /// </summary>
        [ContextMenu("Generate Clean World")]
        public void GenerateCleanWorld()
        {
            Debug.Log("🌍 Generating clean procedural world...");

            // Setup camera
            SetupCamera();

            // Setup lighting
            SetupLighting();

            // Generate lands in organized circle
            for (int i = 0; i < numberOfLands; i++)
            {
                CreateLand(i);
            }

            Debug.Log($"✅ Generated {numberOfLands} lands!");
        }

        /// <summary>
        /// Create a single land with environment
        /// </summary>
        private void CreateLand(int index)
        {
            // Calculate position in circle
            float angle = (index * 360f / numberOfLands) * Mathf.Deg2Rad;
            Vector3 position = new Vector3(
                Mathf.Cos(angle) * circleRadius,
                0,
                Mathf.Sin(angle) * circleRadius
            );

            // Create land type based on index
            LandType landType = (LandType)(index % 7);

            // Create land container
            GameObject landObject = new GameObject($"Land_{landType}_{index}");
            landObject.transform.position = position;

            // Create base
            CreateLandBase(landObject, landType);

            // Add environment details
            AddEnvironmentToLand(landObject, landType);

            // Add label
            CreateLandLabel(landObject, landType, position);
        }

        /// <summary>
        /// Create visual base for land
        /// </summary>
        private void CreateLandBase(GameObject landObject, LandType landType)
        {
            GameObject baseObject = null;
            Material material = new Material(Shader.Find("Standard"));

            switch (landType)
            {
                case LandType.Grass:
                    baseObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    baseObject.transform.localScale = new Vector3(landSize, landHeight * 0.5f, landSize);
                    material.color = new Color(0.3f, 0.7f, 0.2f); // Green
                    break;

                case LandType.Water:
                    baseObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    baseObject.transform.localScale = new Vector3(landSize, landHeight * 0.3f, landSize);
                    baseObject.transform.localPosition = Vector3.down * 0.5f;
                    material.color = new Color(0.2f, 0.5f, 0.9f); // Blue
                    material.SetFloat("_Metallic", 0.8f);
                    material.SetFloat("_Smoothness", 0.95f);
                    break;

                case LandType.Mountain:
                    baseObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    baseObject.transform.localScale = new Vector3(landSize * 0.8f, landHeight * 3f, landSize * 0.8f);
                    material.color = new Color(0.5f, 0.5f, 0.5f); // Gray
                    break;

                case LandType.Desert:
                    baseObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    baseObject.transform.localScale = new Vector3(landSize, landHeight * 0.4f, landSize);
                    material.color = new Color(0.9f, 0.8f, 0.4f); // Sandy
                    break;

                case LandType.Snow:
                    baseObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    baseObject.transform.localScale = new Vector3(landSize * 0.9f, landHeight * 1.5f, landSize * 0.9f);
                    material.color = new Color(0.95f, 0.95f, 1f); // White
                    break;

                case LandType.Volcano:
                    baseObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    baseObject.transform.localScale = new Vector3(landSize * 0.8f, landHeight * 2.5f, landSize * 0.8f);
                    material.color = new Color(0.3f, 0.1f, 0.1f); // Dark red
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", Color.red * 0.5f);
                    break;

                case LandType.Magic:
                    baseObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    baseObject.transform.localScale = Vector3.one * landSize * 0.9f;
                    material.color = new Color(0.8f, 0.4f, 0.9f); // Purple
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", new Color(0.8f, 0.4f, 0.9f) * 0.3f);
                    break;
            }

            if (baseObject != null)
            {
                baseObject.name = "Base";
                baseObject.transform.SetParent(landObject.transform);
                baseObject.GetComponent<Renderer>().material = material;
            }
        }

        /// <summary>
        /// Add environment objects to land
        /// </summary>
        private void AddEnvironmentToLand(GameObject landObject, LandType landType)
        {
            switch (landType)
            {
                case LandType.Grass:
                    // Add trees
                    for (int i = 0; i < 3; i++)
                    {
                        CreateTree(landObject, GetRandomPositionOnLand(2f));
                    }
                    break;

                case LandType.Mountain:
                    // Add rocks
                    for (int i = 0; i < 2; i++)
                    {
                        CreateRock(landObject, GetRandomPositionOnLand(1.5f));
                    }
                    break;

                case LandType.Desert:
                    // Add cactus
                    CreateCactus(landObject, GetRandomPositionOnLand(1.5f));
                    break;

                case LandType.Snow:
                    // Add snowy trees
                    for (int i = 0; i < 2; i++)
                    {
                        CreateSnowyTree(landObject, GetRandomPositionOnLand(2f));
                    }
                    break;

                case LandType.Magic:
                    // Add crystals
                    for (int i = 0; i < 3; i++)
                    {
                        CreateCrystal(landObject, GetRandomPositionOnLand(2f));
                    }
                    break;
            }
        }

        /// <summary>
        /// Get random position on land surface
        /// </summary>
        private Vector3 GetRandomPositionOnLand(float radius)
        {
            Vector2 randomCircle = Random.insideUnitCircle * radius;
            return new Vector3(randomCircle.x, 1.5f, randomCircle.y);
        }

        /// <summary>
        /// Create a simple tree
        /// </summary>
        private void CreateTree(GameObject parent, Vector3 localPosition)
        {
            GameObject tree = new GameObject("Tree");
            tree.transform.SetParent(parent.transform);
            tree.transform.localPosition = localPosition;

            // Trunk
            GameObject trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            trunk.transform.SetParent(tree.transform);
            trunk.transform.localScale = new Vector3(0.2f, 0.8f, 0.2f);
            trunk.transform.localPosition = Vector3.zero;
            trunk.GetComponent<Renderer>().material.color = new Color(0.4f, 0.2f, 0.1f);

            // Leaves
            GameObject leaves = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            leaves.transform.SetParent(tree.transform);
            leaves.transform.localScale = Vector3.one * 1.2f;
            leaves.transform.localPosition = Vector3.up * 1.2f;
            leaves.GetComponent<Renderer>().material.color = new Color(0.2f, 0.6f, 0.2f);
        }

        /// <summary>
        /// Create a snowy tree
        /// </summary>
        private void CreateSnowyTree(GameObject parent, Vector3 localPosition)
        {
            GameObject tree = new GameObject("SnowyTree");
            tree.transform.SetParent(parent.transform);
            tree.transform.localPosition = localPosition;

            // Trunk
            GameObject trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            trunk.transform.SetParent(tree.transform);
            trunk.transform.localScale = new Vector3(0.2f, 0.8f, 0.2f);
            trunk.GetComponent<Renderer>().material.color = new Color(0.3f, 0.2f, 0.1f);

            // Snow leaves (using cylinder as pine tree shape)
            GameObject leaves = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            leaves.transform.SetParent(tree.transform);
            leaves.transform.localScale = new Vector3(1f, 1.2f, 1f);
            leaves.transform.localPosition = Vector3.up * 1.2f;
            leaves.GetComponent<Renderer>().material.color = new Color(0.9f, 0.95f, 1f);
        }

        /// <summary>
        /// Create a rock
        /// </summary>
        private void CreateRock(GameObject parent, Vector3 localPosition)
        {
            GameObject rock = GameObject.CreatePrimitive(PrimitiveType.Cube);
            rock.name = "Rock";
            rock.transform.SetParent(parent.transform);
            rock.transform.localPosition = localPosition;
            rock.transform.localScale = new Vector3(0.5f, 0.3f, 0.5f);
            rock.transform.localRotation = Quaternion.Euler(Random.Range(-15, 15), Random.Range(0, 360), Random.Range(-15, 15));
            rock.GetComponent<Renderer>().material.color = new Color(0.5f, 0.5f, 0.5f);
        }

        /// <summary>
        /// Create a cactus
        /// </summary>
        private void CreateCactus(GameObject parent, Vector3 localPosition)
        {
            GameObject cactus = new GameObject("Cactus");
            cactus.transform.SetParent(parent.transform);
            cactus.transform.localPosition = localPosition;

            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            body.transform.SetParent(cactus.transform);
            body.transform.localScale = new Vector3(0.3f, 0.8f, 0.3f);
            body.GetComponent<Renderer>().material.color = new Color(0.2f, 0.5f, 0.2f);
        }

        /// <summary>
        /// Create a magical crystal
        /// </summary>
        private void CreateCrystal(GameObject parent, Vector3 localPosition)
        {
            GameObject crystal = GameObject.CreatePrimitive(PrimitiveType.Cube);
            crystal.name = "Crystal";
            crystal.transform.SetParent(parent.transform);
            crystal.transform.localPosition = localPosition;
            crystal.transform.localScale = new Vector3(0.3f, 1f, 0.3f);
            crystal.transform.localRotation = Quaternion.Euler(Random.Range(-15, 15), Random.Range(0, 360), Random.Range(-15, 15));

            Material mat = crystal.GetComponent<Renderer>().material;
            mat.color = new Color(0.8f, 0.4f, 0.9f, 0.8f);
            mat.EnableKeyword("_EMISSION");
            mat.SetColor("_EmissionColor", new Color(0.8f, 0.4f, 0.9f));
        }

        /// <summary>
        /// Create text label for land
        /// </summary>
        private void CreateLandLabel(GameObject landObject, LandType landType, Vector3 position)
        {
            GameObject label = new GameObject("Label");
            label.transform.SetParent(landObject.transform);
            label.transform.localPosition = Vector3.up * 3f;

            // Add text mesh (simple 3D text)
            TextMesh textMesh = label.AddComponent<TextMesh>();
            textMesh.text = landType.ToString();
            textMesh.fontSize = 20;
            textMesh.color = Color.white;
            textMesh.alignment = TextAlignment.Center;
            textMesh.anchor = TextAnchor.MiddleCenter;
            textMesh.characterSize = 0.1f;
        }

        /// <summary>
        /// Setup camera for good view
        /// </summary>
        private void SetupCamera()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                GameObject cameraObject = new GameObject("Main Camera");
                mainCamera = cameraObject.AddComponent<Camera>();
                cameraObject.tag = "MainCamera";
                cameraObject.AddComponent<AudioListener>();
            }

            mainCamera.transform.position = new Vector3(0, 30, -25);
            mainCamera.transform.rotation = Quaternion.Euler(45, 0, 0);
            mainCamera.clearFlags = CameraClearFlags.SolidColor;
            mainCamera.backgroundColor = new Color(0.1f, 0.1f, 0.15f);
        }

        /// <summary>
        /// Setup nice lighting
        /// </summary>
        private void SetupLighting()
        {
            // Find or create directional light
            Light[] lights = FindObjectsByType<Light>(FindObjectsSortMode.None);
            Light sunLight = null;

            foreach (Light light in lights)
            {
                if (light.type == LightType.Directional)
                {
                    sunLight = light;
                    break;
                }
            }

            if (sunLight == null)
            {
                GameObject lightObject = new GameObject("Directional Light");
                sunLight = lightObject.AddComponent<Light>();
                sunLight.type = LightType.Directional;
            }

            sunLight.transform.rotation = Quaternion.Euler(50f, 30f, 0);
            sunLight.color = Color.white;
            sunLight.intensity = 1.5f;
            sunLight.shadows = LightShadows.Soft;

            // Setup ambient light
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.3f, 0.3f, 0.4f);
        }

        /// <summary>
        /// Clear all generated lands
        /// </summary>
        [ContextMenu("Clear World")]
        public void ClearWorld()
        {
            GameObject[] allObjects = GameObject.FindObjectsOfType<GameObject>();
            foreach (GameObject obj in allObjects)
            {
                if (obj.name.StartsWith("Land_"))
                {
                    DestroyImmediate(obj);
                }
            }
            Debug.Log("🧹 Cleared all lands");
        }

        private enum LandType
        {
            Grass,
            Water,
            Mountain,
            Desert,
            Snow,
            Volcano,
            Magic
        }
    }
}
