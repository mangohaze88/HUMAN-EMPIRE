using UnityEngine;
using UnityEngine.UI;
using TMPro;
using WorldNavigator.Navigation;
using WorldNavigator.UI;
using WorldNavigator.Lands;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Main world manager that sets up the entire game world
    /// Instantiates all lands and manages the game state
    /// </summary>
    public class WorldManager : MonoBehaviour
    {
        [Header("Core References")]
        [SerializeField] private LandDatabase landDatabase;
        [SerializeField] private LandPrefabGenerator prefabGenerator;
        [SerializeField] private GameManager gameManager;
        [SerializeField] private NavigationController navigationController;
        
        [Header("UI References")]
        [SerializeField] private Canvas mainCanvas;
        [SerializeField] private LandInfoPanel landInfoPanel;
        [SerializeField] private GameObject discoveryUI;
        [SerializeField] private TextMeshProUGUI discoveryCountText;
        [SerializeField] private TextMeshProUGUI instructionsText;
        
        [Header("World Settings")]
        [SerializeField] private bool generateWorldOnStart = true;
        [SerializeField] private Transform landContainer;
        
        // Runtime data
        private LandType[] allInstantiatedLands;
        private bool worldGenerated = false;
        
        private void Start()
        {
            InitializeWorld();
        }
        
        /// <summary>
        /// Initialize the entire world
        /// </summary>
        private void InitializeWorld()
        {
            // Create land container if not assigned
            if (landContainer == null)
            {
                GameObject container = new GameObject("Land Container");
                landContainer = container.transform;
            }
            
            // Initialize database if needed
            if (landDatabase == null)
            {
                CreateDefaultDatabase();
            }
            
            // Generate world
            if (generateWorldOnStart)
            {
                GenerateWorld();
            }
            
            // Setup UI
            SetupUI();
            
            Debug.Log("World Navigator initialized successfully!");
        }
        
        /// <summary>
        /// Generate the entire world with all land types
        /// </summary>
        [ContextMenu("Generate World")]
        public void GenerateWorld()
        {
            if (worldGenerated)
            {
                Debug.LogWarning("World already generated!");
                return;
            }
            
            if (landDatabase == null || landDatabase.allLands.Length == 0)
            {
                Debug.LogError("No land data available!");
                return;
            }
            
            Debug.Log($"Generating world with {landDatabase.allLands.Length} land types...");
            
            // Clear existing lands
            ClearExistingLands();
            
            // Instantiate all lands
            allInstantiatedLands = new LandType[landDatabase.allLands.Length];
            
            for (int i = 0; i < landDatabase.allLands.Length; i++)
            {
                LandData landData = landDatabase.allLands[i];
                GameObject landObject = CreateLandFromData(landData);
                
                if (landObject != null)
                {
                    allInstantiatedLands[i] = landObject.GetComponent<LandType>();
                }
            }
            
            worldGenerated = true;
            
            // Setup starting position
            SetupStartingPosition();
            
            Debug.Log("World generation complete!");
        }
        
        /// <summary>
        /// Create a land object from land data
        /// </summary>
        private GameObject CreateLandFromData(LandData landData)
        {
            // Create main land object
            GameObject landObject = new GameObject($"Land_{landData.landName}");
            landObject.transform.SetParent(landContainer);
            
            // Add LandType component and set data
            LandType landType = landObject.AddComponent<LandType>();
            
            // Use reflection or direct assignment to set private landData field
            SetLandData(landType, landData);
            
            // Generate visual components
            GenerateLandVisuals(landObject, landData);
            
            // Position in world
            landObject.transform.position = GetLandPosition(landData);
            
            return landObject;
        }
        
        /// <summary>
        /// Set land data using reflection (since it's private)
        /// </summary>
        private void SetLandData(LandType landType, LandData landData)
        {
            // Use reflection to set private field
            var field = typeof(LandType).GetField("landData", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field != null)
            {
                field.SetValue(landType, landData);
            }
        }
        
        /// <summary>
        /// Generate visual components for a land
        /// </summary>
        private void GenerateLandVisuals(GameObject landObject, LandData landData)
        {
            // Create basic mesh
            CreateBasicMesh(landObject, landData);
            
            // Add particle effects
            CreateParticleEffects(landObject, landData);
            
            // Add lighting
            CreateLandLighting(landObject, landData);
            
            // Add audio
            CreateLandAudio(landObject, landData);
        }
        
        /// <summary>
        /// Create basic mesh for land
        /// </summary>
        private void CreateBasicMesh(GameObject landObject, LandData landData)
        {
            // Create a simple plane or cube based on land type
            GameObject primitive;
            
            switch (landData.category)
            {
                case LandCategory.Mountain:
                case LandCategory.Volcanic:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    primitive.transform.localScale = new Vector3(4f, 2f, 4f);
                    break;
                case LandCategory.Water:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Plane);
                    primitive.transform.localScale = new Vector3(2f, 1f, 2f);
                    primitive.transform.position = Vector3.down * 0.5f;
                    break;
                default:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    primitive.transform.localScale = new Vector3(4f, 0.5f, 4f);
                    break;
            }
            
            // Parent to land object
            primitive.transform.SetParent(landObject.transform);
            primitive.transform.localPosition = Vector3.zero;
            
            // Create and apply material
            Material landMaterial = new Material(Shader.Find("Standard"));
            landMaterial.color = landData.primaryColor;
            landMaterial.name = $"{landData.landName}_Material";
            
            // Set material properties
            switch (landData.category)
            {
                case LandCategory.Water:
                    landMaterial.SetFloat("_Metallic", 0.8f);
                    landMaterial.SetFloat("_Smoothness", 0.9f);
                    break;
                case LandCategory.Volcanic:
                    landMaterial.SetFloat("_Metallic", 0.3f);
                    landMaterial.SetFloat("_Smoothness", 0.1f);
                    landMaterial.EnableKeyword("_EMISSION");
                    landMaterial.SetColor("_EmissionColor", landData.primaryColor * 0.5f);
                    break;
                case LandCategory.Special:
                    landMaterial.SetFloat("_Metallic", 0.5f);
                    landMaterial.SetFloat("_Smoothness", 0.7f);
                    landMaterial.EnableKeyword("_EMISSION");
                    landMaterial.SetColor("_EmissionColor", landData.primaryColor * 0.3f);
                    break;
            }
            
            primitive.GetComponent<Renderer>().material = landMaterial;
            landData.landMaterial = landMaterial;
            
            // Add collider for interaction
            if (primitive.GetComponent<Collider>() == null)
            {
                primitive.AddComponent<BoxCollider>();
            }
        }
        
        /// <summary>
        /// Create particle effects
        /// </summary>
        private void CreateParticleEffects(GameObject landObject, LandData landData)
        {
            GameObject particleObject = new GameObject("Particles");
            particleObject.transform.SetParent(landObject.transform);
            particleObject.transform.localPosition = Vector3.up * 2f;
            
            ParticleSystem particles = particleObject.AddComponent<ParticleSystem>();
            var main = particles.main;
            var emission = particles.emission;
            
            main.startColor = landData.primaryColor;
            main.startSize = 0.1f;
            main.startLifetime = 3f;
            emission.rateOverTime = 5f;
            
            // Special effects for certain categories
            if (landData.category == LandCategory.Volcanic)
            {
                emission.rateOverTime = 20f;
                main.startColor = Color.red;
            }
            else if (landData.category == LandCategory.Special)
            {
                emission.rateOverTime = 15f;
                main.startColor = new Color(landData.primaryColor.r, landData.primaryColor.g, landData.primaryColor.b, 0.7f);
            }
        }
        
        /// <summary>
        /// Create land lighting
        /// </summary>
        private void CreateLandLighting(GameObject landObject, LandData landData)
        {
            GameObject lightObject = new GameObject("Light");
            lightObject.transform.SetParent(landObject.transform);
            lightObject.transform.localPosition = Vector3.up * 5f;
            
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Point;
            light.color = landData.primaryColor;
            light.intensity = landData.category == LandCategory.Volcanic ? 2f : 1f;
            light.range = 10f;
        }
        
        /// <summary>
        /// Create land audio
        /// </summary>
        private void CreateLandAudio(GameObject landObject, LandData landData)
        {
            AudioSource audioSource = landObject.AddComponent<AudioSource>();
            audioSource.loop = true;
            audioSource.volume = 0.3f;
            audioSource.spatialBlend = 0.8f;
            audioSource.maxDistance = 15f;
            audioSource.playOnAwake = false;
        }
        
        /// <summary>
        /// Get position for land based on category and index
        /// </summary>
        private Vector3 GetLandPosition(LandData landData)
        {
            // Arrange lands in a circular pattern by category
            int categoryIndex = (int)landData.category;
            
            // Get lands in this category
            LandData[] categoryLands = landDatabase.GetLandsByCategory(landData.category);
            int landIndex = System.Array.IndexOf(categoryLands, landData);
            
            // Calculate position
            float categoryAngle = categoryIndex * (360f / 7f) * Mathf.Deg2Rad;
            float landAngle = landIndex * (360f / Mathf.Max(1, categoryLands.Length)) * Mathf.Deg2Rad;
            
            float categoryRadius = 25f;
            float landRadius = 8f;
            
            Vector3 categoryCenter = new Vector3(
                Mathf.Cos(categoryAngle) * categoryRadius,
                0,
                Mathf.Sin(categoryAngle) * categoryRadius
            );
            
            Vector3 landOffset = new Vector3(
                Mathf.Cos(categoryAngle + landAngle) * landRadius,
                0,
                Mathf.Sin(categoryAngle + landAngle) * landRadius
            );
            
            return categoryCenter + landOffset;
        }
        
        /// <summary>
        /// Clear existing lands
        /// </summary>
        private void ClearExistingLands()
        {
            if (landContainer != null)
            {
                foreach (Transform child in landContainer)
                {
                    DestroyImmediate(child.gameObject);
                }
            }
        }
        
        /// <summary>
        /// Setup starting position
        /// </summary>
        private void SetupStartingPosition()
        {
            if (navigationController == null || allInstantiatedLands == null) return;
            
            // Find a good starting land (common lands only)
            LandType startingLand = null;
            foreach (LandType land in allInstantiatedLands)
            {
                if (land != null && land.Data.rarity <= 2)
                {
                    startingLand = land;
                    break;
                }
            }
            
            if (startingLand != null)
            {
                navigationController.SetStartingLand(startingLand);
            }
        }
        
        /// <summary>
        /// Setup UI elements
        /// </summary>
        private void SetupUI()
        {
            // Create main canvas if not exists
            if (mainCanvas == null)
            {
                CreateMainCanvas();
            }
            
            // Setup discovery counter
            if (discoveryCountText != null)
            {
                UpdateDiscoveryUI();
            }
            
            // Setup instructions
            if (instructionsText != null)
            {
                instructionsText.text = "Click on discovered lands to explore them!\\nUse mouse to navigate and discover new territories.";
            }
        }
        
        /// <summary>
        /// Create main canvas
        /// </summary>
        private void CreateMainCanvas()
        {
            GameObject canvasObject = new GameObject("Main Canvas");
            mainCanvas = canvasObject.AddComponent<Canvas>();
            mainCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
            
            canvasObject.AddComponent<CanvasScaler>();
            canvasObject.AddComponent<GraphicRaycaster>();
        }
        
        /// <summary>
        /// Update discovery UI
        /// </summary>
        private void UpdateDiscoveryUI()
        {
            if (allInstantiatedLands == null) return;
            
            int discoveredCount = 0;
            foreach (LandType land in allInstantiatedLands)
            {
                if (land != null && land.IsDiscovered)
                    discoveredCount++;
            }
            
            if (discoveryCountText != null)
            {
                discoveryCountText.text = $"Discovered: {discoveredCount}/{allInstantiatedLands.Length}";
            }
        }
        
        /// <summary>
        /// Create default database if none exists
        /// </summary>
        private void CreateDefaultDatabase()
        {
            landDatabase = ScriptableObject.CreateInstance<LandDatabase>();
            landDatabase.InitializeDefaultLands();
            Debug.Log("Created default land database with all land types!");
        }
        
        /// <summary>
        /// Get all instantiated lands
        /// </summary>
        public LandType[] GetAllLands()
        {
            return allInstantiatedLands;
        }
    }
}