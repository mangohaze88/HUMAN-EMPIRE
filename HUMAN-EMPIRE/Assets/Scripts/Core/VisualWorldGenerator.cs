using UnityEngine;
using UnityEngine.UI;
using TMPro;
using WorldNavigator.Core;
using WorldNavigator.Lands;

namespace WorldNavigator.Visual
{
    /// <summary>
    /// Enhanced visual world generator with animations and effects
    /// </summary>
    public class VisualWorldGenerator : MonoBehaviour
    {
        [Header("World Generation")]
        [SerializeField] private LandDatabase landDatabase;
        [SerializeField] private Transform landContainer;
        [SerializeField] private bool generateOnStart = true;
        
        [Header("Visual Settings")]
        [SerializeField] private float landSpacing = 15f;
        [SerializeField] private float animationDuration = 2f;
        [SerializeField] private AnimationCurve spawnCurve = AnimationCurve.EaseInOut(0, 0, 1, 1);
        
        [Header("Effects")]
        [SerializeField] private ParticleSystem discoveryEffect;
        [SerializeField] private AudioSource ambientAudio;
        
        // Generated objects
        private GameObject[] landObjects;
        private Material[] landMaterials;
        private bool isGenerating = false;
        
        private void Start()
        {
            InitializeWorld();
        }
        
        /// <summary>
        /// Initialize the visual world
        /// </summary>
        private void InitializeWorld()
        {
            SetupCamera();
            SetupLighting();
            SetupUI();
            
            if (generateOnStart)
            {
                StartCoroutine(GenerateWorldAnimated());
            }
        }
        
        /// <summary>
        /// Setup camera for better viewing
        /// </summary>
        private void SetupCamera()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                GameObject cameraObject = new GameObject("Main Camera");
                mainCamera = cameraObject.AddComponent<Camera>();
                cameraObject.tag = "MainCamera";
            }
            
            // Position camera for overview
            mainCamera.transform.position = new Vector3(0, 40, -30);
            mainCamera.transform.rotation = Quaternion.Euler(45, 0, 0);
            mainCamera.fieldOfView = 60f;
            mainCamera.backgroundColor = new Color(0.1f, 0.1f, 0.2f, 1f);
            
            // Add camera controller
            if (mainCamera.GetComponent<CameraController>() == null)
            {
                mainCamera.gameObject.AddComponent<CameraController>();
            }
        }
        
        /// <summary>
        /// Setup atmospheric lighting
        /// </summary>
        private void SetupLighting()
        {
            // Main directional light (sun)
            GameObject sunLight = new GameObject("Sun Light");
            Light sun = sunLight.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(1f, 0.95f, 0.8f, 1f);
            sun.intensity = 1.2f;
            sun.shadows = LightShadows.Soft;
            sunLight.transform.rotation = Quaternion.Euler(45f, 30f, 0);
            
            // Ambient lighting
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.3f, 0.4f, 0.6f, 1f);
            RenderSettings.ambientEquatorColor = new Color(0.2f, 0.3f, 0.4f, 1f);
            RenderSettings.ambientGroundColor = new Color(0.1f, 0.1f, 0.2f, 1f);
            
            // Fog for atmosphere
            RenderSettings.fog = true;
            RenderSettings.fogColor = new Color(0.5f, 0.6f, 0.8f, 1f);
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = 0.01f;
        }
        
        /// <summary>
        /// Setup basic UI
        /// </summary>
        private void SetupUI()
        {
            // Create canvas if not exists
            Canvas canvas = FindFirstObjectByType<Canvas>();
            if (canvas == null)
            {
                GameObject canvasObject = new GameObject("World UI Canvas");
                canvas = canvasObject.AddComponent<Canvas>();
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                canvasObject.AddComponent<CanvasScaler>();
                canvasObject.AddComponent<GraphicRaycaster>();
            }
            
            // Create title text
            CreateTitleText(canvas.transform);
            CreateInstructionsText(canvas.transform);
        }
        
        /// <summary>
        /// Create animated title
        /// </summary>
        private void CreateTitleText(Transform parent)
        {
            GameObject titleObject = new GameObject("Title");
            titleObject.transform.SetParent(parent);
            
            TextMeshProUGUI title = titleObject.AddComponent<TextMeshProUGUI>();
            title.text = "🌍 WORLD NAVIGATOR 🌍";
            title.fontSize = 36;
            title.fontStyle = FontStyles.Bold;
            title.color = new Color(1f, 1f, 1f, 0.9f);
            title.alignment = TextAlignmentOptions.Center;
            
            RectTransform rect = title.rectTransform;
            rect.anchorMin = new Vector2(0.5f, 1f);
            rect.anchorMax = new Vector2(0.5f, 1f);
            rect.anchoredPosition = new Vector2(0, -50);
            rect.sizeDelta = new Vector2(600, 80);
            
            // Add glow effect
            title.fontMaterial = CreateGlowMaterial();
        }
        
        /// <summary>
        /// Create instructions text
        /// </summary>
        private void CreateInstructionsText(Transform parent)
        {
            GameObject instructionsObject = new GameObject("Instructions");
            instructionsObject.transform.SetParent(parent);
            
            TextMeshProUGUI instructions = instructionsObject.AddComponent<TextMeshProUGUI>();
            instructions.text = "🖱️ Click lands to explore • 🔍 Mouse wheel to zoom • 🎯 Right-click to pan";
            instructions.fontSize = 18;
            instructions.color = new Color(1f, 1f, 1f, 0.7f);
            instructions.alignment = TextAlignmentOptions.Center;
            
            RectTransform rect = instructions.rectTransform;
            rect.anchorMin = new Vector2(0.5f, 0f);
            rect.anchorMax = new Vector2(0.5f, 0f);
            rect.anchoredPosition = new Vector2(0, 30);
            rect.sizeDelta = new Vector2(800, 40);
        }
        
        /// <summary>
        /// Generate world with animations
        /// </summary>
        private System.Collections.IEnumerator GenerateWorldAnimated()
        {
            if (isGenerating) yield break;
            isGenerating = true;
            
            Debug.Log("Starting animated world generation...");
            
            // Initialize database if needed
            if (landDatabase == null)
            {
                landDatabase = ScriptableObject.CreateInstance<LandDatabase>();
                landDatabase.InitializeDefaultLands();
            }
            
            // Create container
            if (landContainer == null)
            {
                GameObject container = new GameObject("World Lands");
                landContainer = container.transform;
            }
            
            // Generate lands with delay
            landObjects = new GameObject[landDatabase.allLands.Length];
            landMaterials = new Material[landDatabase.allLands.Length];
            
            for (int i = 0; i < landDatabase.allLands.Length; i++)
            {
                LandData landData = landDatabase.allLands[i];
                Vector3 position = GetLandPosition(i, landDatabase.allLands.Length);
                
                GameObject landObject = CreateVisualLand(landData, position, i);
                landObjects[i] = landObject;
                
                // Animate spawn
                StartCoroutine(AnimateLandSpawn(landObject, i * 0.1f));
                
                yield return new WaitForSeconds(0.05f); // Small delay between spawns
            }
            
            isGenerating = false;
            Debug.Log($"Generated {landDatabase.allLands.Length} lands with visual effects!");
        }
        
        /// <summary>
        /// Create visually enhanced land
        /// </summary>
        private GameObject CreateVisualLand(LandData landData, Vector3 position, int index)
        {
            GameObject landObject = new GameObject($"Land_{landData.landName}");
            landObject.transform.SetParent(landContainer);
            landObject.transform.position = position;
            
            // Add LandType component
            LandType landType = landObject.AddComponent<LandType>();
            SetLandData(landType, landData);
            
            // Create visual components
            CreateLandMesh(landObject, landData, index);
            CreateLandParticles(landObject, landData);
            CreateLandLight(landObject, landData);
            CreateLandAudio(landObject, landData);
            CreateLandInteraction(landObject, landData);
            
            return landObject;
        }
        
        /// <summary>
        /// Create enhanced land mesh with better visuals
        /// </summary>
        private void CreateLandMesh(GameObject landObject, LandData landData, int materialIndex)
        {
            GameObject meshObject = CreateBasicMeshForCategory(landData.category);
            meshObject.transform.SetParent(landObject.transform);
            meshObject.transform.localPosition = Vector3.zero;
            
            // Create enhanced material
            Material material = CreateEnhancedMaterial(landData);
            landMaterials[materialIndex] = material;
            
            Renderer renderer = meshObject.GetComponent<Renderer>();
            renderer.material = material;
            
            // Add floating animation
            landObject.AddComponent<FloatingAnimation>();
        }
        
        /// <summary>
        /// Create different mesh types for different categories
        /// </summary>
        private GameObject CreateBasicMeshForCategory(LandCategory category)
        {
            GameObject primitive;
            
            switch (category)
            {
                case LandCategory.Mountain:
                case LandCategory.Volcanic:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    primitive.transform.localScale = new Vector3(3f, 4f, 3f);
                    break;
                    
                case LandCategory.Water:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    primitive.transform.localScale = new Vector3(4f, 0.3f, 4f);
                    primitive.transform.localPosition = Vector3.down * 0.3f;
                    break;
                    
                case LandCategory.Arid:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    primitive.transform.localScale = new Vector3(3.5f, 0.8f, 3.5f);
                    break;
                    
                case LandCategory.Cold:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    primitive.transform.localScale = new Vector3(3f, 1.5f, 3f);
                    break;
                    
                case LandCategory.Special:
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    primitive.transform.localScale = Vector3.one * 3f;
                    break;
                    
                default: // Temperate
                    primitive = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                    primitive.transform.localScale = new Vector3(3.5f, 1.2f, 3.5f);
                    break;
            }
            
            return primitive;
        }
        
        /// <summary>
        /// Create enhanced materials with better visuals
        /// </summary>
        private Material CreateEnhancedMaterial(LandData landData)
        {
            Material material = new Material(Shader.Find("Standard"));
            material.name = $"{landData.landName}_Enhanced";
            
            // Base color
            material.color = landData.primaryColor;
            
            // Category-specific material properties
            switch (landData.category)
            {
                case LandCategory.Water:
                    material.SetFloat("_Metallic", 0.8f);
                    material.SetFloat("_Smoothness", 0.95f);
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", landData.primaryColor * 0.3f);
                    break;
                    
                case LandCategory.Volcanic:
                    material.SetFloat("_Metallic", 0.2f);
                    material.SetFloat("_Smoothness", 0.1f);
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", Color.red * 0.5f);
                    break;
                    
                case LandCategory.Special:
                    material.SetFloat("_Metallic", 0.6f);
                    material.SetFloat("_Smoothness", 0.8f);
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", landData.primaryColor * 0.4f);
                    break;
                    
                case LandCategory.Mountain:
                    material.SetFloat("_Metallic", 0.1f);
                    material.SetFloat("_Smoothness", 0.2f);
                    break;
                    
                default:
                    material.SetFloat("_Metallic", 0.0f);
                    material.SetFloat("_Smoothness", 0.5f);
                    break;
            }
            
            return material;
        }
        
        /// <summary>
        /// Create enhanced particle systems
        /// </summary>
        private void CreateLandParticles(GameObject landObject, LandData landData)
        {
            GameObject particleObject = new GameObject("Particles");
            particleObject.transform.SetParent(landObject.transform);
            particleObject.transform.localPosition = Vector3.up * 3f;
            
            ParticleSystem particles = particleObject.AddComponent<ParticleSystem>();
            var main = particles.main;
            var emission = particles.emission;
            var shape = particles.shape;
            
            // Basic settings
            main.startLifetime = 4f;
            main.startSpeed = 2f;
            main.startSize = 0.2f;
            main.startColor = landData.primaryColor;
            emission.rateOverTime = 8f;
            
            // Shape
            shape.shapeType = ParticleSystemShapeType.Circle;
            shape.radius = 2f;
            
            // Category-specific effects
            switch (landData.category)
            {
                case LandCategory.Volcanic:
                    emission.rateOverTime = 25f;
                    main.startColor = Color.red;
                    main.startSpeed = 5f;
                    break;
                    
                case LandCategory.Water:
                    main.startColor = Color.cyan;
                    emission.rateOverTime = 15f;
                    break;
                    
                case LandCategory.Special:
                    emission.rateOverTime = 20f;
                    var colorOverLifetime = particles.colorOverLifetime;
                    colorOverLifetime.enabled = true;
                    Gradient gradient = new Gradient();
                    gradient.SetKeys(
                        new GradientColorKey[] { 
                            new GradientColorKey(landData.primaryColor, 0.0f), 
                            new GradientColorKey(Color.white, 1.0f) 
                        },
                        new GradientAlphaKey[] { 
                            new GradientAlphaKey(1.0f, 0.0f), 
                            new GradientAlphaKey(0.0f, 1.0f) 
                        }
                    );
                    colorOverLifetime.color = gradient;
                    break;
            }
        }
        
        /// <summary>
        /// Create atmospheric lighting for each land
        /// </summary>
        private void CreateLandLight(GameObject landObject, LandData landData)
        {
            GameObject lightObject = new GameObject("Atmosphere Light");
            lightObject.transform.SetParent(landObject.transform);
            lightObject.transform.localPosition = Vector3.up * 6f;
            
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Point;
            light.color = landData.primaryColor;
            light.intensity = landData.category == LandCategory.Volcanic ? 3f : 1.5f;
            light.range = 12f;
            light.shadows = LightShadows.Soft;
            
            // Add light animation
            lightObject.AddComponent<LightPulseAnimation>();
        }
        
        /// <summary>
        /// Create ambient audio for each land
        /// </summary>
        private void CreateLandAudio(GameObject landObject, LandData landData)
        {
            AudioSource audio = landObject.AddComponent<AudioSource>();
            audio.spatialBlend = 0.9f;
            audio.volume = 0.3f;
            audio.maxDistance = 20f;
            audio.rolloffMode = AudioRolloffMode.Linear;
            audio.loop = true;
            audio.playOnAwake = false;
            
            // Note: In a real project, you'd assign different audio clips here
            // audio.clip = GetAudioClipForCategory(landData.category);
        }
        
        /// <summary>
        /// Create interaction components
        /// </summary>
        private void CreateLandInteraction(GameObject landObject, LandData landData)
        {
            // Add collider for interaction
            if (landObject.GetComponent<Collider>() == null)
            {
                landObject.AddComponent<SphereCollider>().radius = 4f;
            }
            
            // Add interaction component
            landObject.AddComponent<LandInteraction>();
        }
        
        /// <summary>
        /// Animate land spawn
        /// </summary>
        private System.Collections.IEnumerator AnimateLandSpawn(GameObject landObject, float delay)
        {
            yield return new WaitForSeconds(delay);
            
            Vector3 originalScale = landObject.transform.localScale;
            Vector3 originalPosition = landObject.transform.position;
            
            // Start from above and small
            landObject.transform.localScale = Vector3.zero;
            landObject.transform.position = originalPosition + Vector3.up * 10f;
            
            float elapsed = 0f;
            while (elapsed < animationDuration)
            {
                elapsed += Time.deltaTime;
                float progress = elapsed / animationDuration;
                float curveValue = spawnCurve.Evaluate(progress);
                
                landObject.transform.localScale = Vector3.Lerp(Vector3.zero, originalScale, curveValue);
                landObject.transform.position = Vector3.Lerp(originalPosition + Vector3.up * 10f, originalPosition, curveValue);
                
                yield return null;
            }
            
            landObject.transform.localScale = originalScale;
            landObject.transform.position = originalPosition;
            
            // Play discovery effect
            if (discoveryEffect != null)
            {
                discoveryEffect.transform.position = originalPosition;
                discoveryEffect.Play();
            }
        }
        
        /// <summary>
        /// Get position for land in circular arrangement
        /// </summary>
        private Vector3 GetLandPosition(int index, int totalLands)
        {
            float angle = (index * 360f / totalLands) * Mathf.Deg2Rad;
            float radius = 20f;
            
            // Add some randomness for natural look
            radius += Random.Range(-3f, 3f);
            
            return new Vector3(
                Mathf.Cos(angle) * radius,
                Random.Range(-1f, 1f), // Slight height variation
                Mathf.Sin(angle) * radius
            );
        }
        
        /// <summary>
        /// Set land data using reflection
        /// </summary>
        private void SetLandData(LandType landType, LandData landData)
        {
            var field = typeof(LandType).GetField("landData", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field != null)
            {
                field.SetValue(landType, landData);
            }
        }
        
        /// <summary>
        /// Create glow material for title
        /// </summary>
        private Material CreateGlowMaterial()
        {
            Material glowMat = new Material(Shader.Find("TextMeshPro/Distance Field"));
            glowMat.SetFloat("_GlowPower", 0.3f);
            glowMat.SetColor("_GlowColor", Color.cyan);
            return glowMat;
        }
        
        /// <summary>
        /// Public method to regenerate world
        /// </summary>
        [ContextMenu("Regenerate World")]
        public void RegenerateWorld()
        {
            if (landContainer != null)
            {
                DestroyImmediate(landContainer.gameObject);
            }
            
            StartCoroutine(GenerateWorldAnimated());
        }
    }
}