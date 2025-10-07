using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using TMPro;
using WorldNavigator.Navigation;
using WorldNavigator.UI;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Sets up the complete Unity scene programmatically
    /// Creates camera, UI, lighting, and all necessary components
    /// </summary>
    public class SceneSetup : MonoBehaviour
    {
        [Header("Scene Configuration")]
        [SerializeField] private bool setupOnStart = true;
        [SerializeField] private bool createLighting = true;
        [SerializeField] private bool createUI = true;
        [SerializeField] private bool createCamera = true;
        
        [Header("Camera Settings")]
        [SerializeField] private Vector3 cameraStartPosition = new Vector3(0, 10, -10);
        [SerializeField] private Vector3 cameraStartRotation = new Vector3(30, 0, 0);
        
        private void Start()
        {
            if (setupOnStart)
            {
                SetupCompleteScene();
            }
        }
        
        /// <summary>
        /// Setup the complete scene
        /// </summary>
        [ContextMenu("Setup Complete Scene")]
        public void SetupCompleteScene()
        {
            Debug.Log("Setting up complete World Navigator scene...");
            
            // Setup scene lighting
            if (createLighting)
                SetupLighting();
            
            // Setup camera and navigation
            if (createCamera)
                SetupCamera();
            
            // Setup UI system
            if (createUI)
                SetupUI();
            
            // Setup world manager
            SetupWorldManager();
            
            // Setup game manager
            SetupGameManager();
            
            Debug.Log("Scene setup complete!");
        }
        
        /// <summary>
        /// Setup scene lighting
        /// </summary>
        private void SetupLighting()
        {
            // Create directional light (sun)
            GameObject sunLight = new GameObject("Directional Light");
            Light sun = sunLight.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(1f, 0.95f, 0.8f);
            sun.intensity = 1.2f;
            sun.shadows = LightShadows.Soft;
            sunLight.transform.rotation = Quaternion.Euler(45f, 30f, 0f);
            
            // Create ambient light
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.54f, 0.67f, 0.84f);
            RenderSettings.ambientEquatorColor = new Color(0.4f, 0.4f, 0.4f);
            RenderSettings.ambientGroundColor = new Color(0.2f, 0.2f, 0.2f);
            
            // Create skybox
            Material skyboxMaterial = new Material(Shader.Find("Skybox/Procedural"));
            skyboxMaterial.SetFloat("_SunSize", 0.04f);
            skyboxMaterial.SetFloat("_SunSizeConvergence", 5f);
            skyboxMaterial.SetFloat("_AtmosphereThickness", 1f);
            skyboxMaterial.SetColor("_SkyTint", new Color(0.5f, 0.5f, 0.5f));
            skyboxMaterial.SetColor("_GroundColor", new Color(0.369f, 0.349f, 0.341f));
            RenderSettings.skybox = skyboxMaterial;
        }
        
        /// <summary>
        /// Setup camera and navigation controller
        /// </summary>
        private void SetupCamera()
        {
            // Find existing camera or create new one
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                GameObject cameraObject = new GameObject("Main Camera");
                mainCamera = cameraObject.AddComponent<Camera>();
                cameraObject.tag = "MainCamera";
                
                // Add audio listener
                cameraObject.AddComponent<AudioListener>();
            }
            
            // Setup camera properties
            mainCamera.transform.position = cameraStartPosition;
            mainCamera.transform.rotation = Quaternion.Euler(cameraStartRotation);
            mainCamera.clearFlags = CameraClearFlags.Skybox;
            mainCamera.fieldOfView = 60f;
            mainCamera.nearClipPlane = 0.1f;
            mainCamera.farClipPlane = 1000f;
            
            // Add navigation controller
            NavigationController navController = mainCamera.GetComponent<NavigationController>();
            if (navController == null)
            {
                navController = mainCamera.gameObject.AddComponent<NavigationController>();
            }
        }
        
        /// <summary>
        /// Setup UI system
        /// </summary>
        private void SetupUI()
        {
            // Create event system if not exists
            EventSystem eventSystem = FindObjectOfType<EventSystem>();
            if (eventSystem == null)
            {
                GameObject eventSystemObject = new GameObject("EventSystem");
                eventSystem = eventSystemObject.AddComponent<EventSystem>();
                eventSystemObject.AddComponent<StandaloneInputModule>();
            }
            
            // Create main canvas
            GameObject canvasObject = new GameObject("Main Canvas");
            Canvas canvas = canvasObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 0;
            
            CanvasScaler scaler = canvasObject.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
            scaler.matchWidthOrHeight = 0.5f;
            
            canvasObject.AddComponent<GraphicRaycaster>();
            
            // Create UI panels
            CreateDiscoveryPanel(canvas.transform);
            CreateLandInfoPanel(canvas.transform);
            CreateNavigationPanel(canvas.transform);
            CreateSettingsPanel(canvas.transform);
            
            // Add main UI controller
            MainUIController uiController = canvasObject.AddComponent<MainUIController>();
            
            // Setup UI references (would be set in inspector normally)
            SetupUIReferences(uiController, canvas.transform);
        }
        
        /// <summary>
        /// Create discovery panel
        /// </summary>
        private void CreateDiscoveryPanel(Transform canvasTransform)
        {
            GameObject panel = CreateUIPanel("Discovery Panel", canvasTransform, 
                new Vector2(300, 150), new Vector2(10, -10), TextAnchor.UpperLeft);
            
            // Discovery count text
            CreateUIText("Discovery Count", panel.transform, 
                "Discovered: 0/22 Lands", new Vector2(280, 30), new Vector2(0, -10));
            
            // Progress slider
            GameObject sliderObject = new GameObject("Discovery Progress");
            sliderObject.transform.SetParent(panel.transform);
            
            Slider slider = sliderObject.AddComponent<Slider>();
            slider.minValue = 0f;
            slider.maxValue = 1f;
            slider.value = 0f;
            
            RectTransform sliderRect = sliderObject.GetComponent<RectTransform>();
            sliderRect.anchoredPosition = new Vector2(0, -50);
            sliderRect.sizeDelta = new Vector2(280, 20);
            
            // Instructions text
            CreateUIText("Instructions", panel.transform,
                "Click discovered lands to explore them!", 
                new Vector2(280, 60), new Vector2(0, -80));
        }
        
        /// <summary>
        /// Create land info panel
        /// </summary>
        private void CreateLandInfoPanel(Transform canvasTransform)
        {
            GameObject panel = CreateUIPanel("Land Info Panel", canvasTransform,
                new Vector2(400, 500), new Vector2(-10, 10), TextAnchor.UpperRight);
            
            panel.SetActive(false); // Start hidden
            
            // Land name
            CreateUIText("Land Name", panel.transform,
                "Land Name", new Vector2(380, 40), new Vector2(0, -20), 24);
            
            // Land category
            CreateUIText("Land Category", panel.transform,
                "Category", new Vector2(380, 25), new Vector2(0, -60), 16);
            
            // Description
            GameObject descText = CreateUIText("Land Description", panel.transform,
                "Description will appear here...", 
                new Vector2(380, 100), new Vector2(0, -120), 14);
            
            // Make description scrollable
            TextMeshProUGUI descTextComp = descText.GetComponent<TextMeshProUGUI>();
            descTextComp.enableWordWrapping = true;
            
            // Characteristics header
            CreateUIText("Characteristics Header", panel.transform,
                "Characteristics:", new Vector2(380, 20), new Vector2(0, -230), 16);
            
            // Characteristics container
            CreateScrollableContainer("Characteristics Container", panel.transform,
                new Vector2(380, 80), new Vector2(0, -290));
            
            // Resources header
            CreateUIText("Resources Header", panel.transform,
                "Resources:", new Vector2(380, 20), new Vector2(0, -380), 16);
            
            // Resources container
            CreateScrollableContainer("Resources Container", panel.transform,
                new Vector2(380, 60), new Vector2(0, -420));
            
            // Navigate button
            CreateUIButton("Navigate Button", panel.transform,
                "Navigate Here", new Vector2(150, 40), new Vector2(-60, -470));
            
            // Close button
            CreateUIButton("Close Button", panel.transform,
                "Close", new Vector2(80, 40), new Vector2(60, -470));
        }
        
        /// <summary>
        /// Create navigation panel
        /// </summary>
        private void CreateNavigationPanel(Transform canvasTransform)
        {
            GameObject panel = CreateUIPanel("Navigation Panel", canvasTransform,
                new Vector2(250, 100), new Vector2(10, 10), TextAnchor.UpperLeft);
            
            // Current location
            CreateUIText("Current Location", panel.transform,
                "Current: Exploring", new Vector2(230, 25), new Vector2(0, -10));
            
            // Status text
            CreateUIText("Navigation Status", panel.transform,
                "Ready to explore", new Vector2(230, 20), new Vector2(0, -40));
            
            // Center camera button
            CreateUIButton("Center Camera Button", panel.transform,
                "Center View", new Vector2(100, 30), new Vector2(0, -70));
        }
        
        /// <summary>
        /// Create settings panel
        /// </summary>
        private void CreateSettingsPanel(Transform canvasTransform)
        {
            GameObject panel = CreateUIPanel("Settings Panel", canvasTransform,
                new Vector2(300, 200), new Vector2(0, 0), TextAnchor.MiddleCenter);
            
            panel.SetActive(false); // Start hidden
            
            // Title
            CreateUIText("Settings Title", panel.transform,
                "Settings", new Vector2(280, 30), new Vector2(0, 70), 20);
            
            // Volume slider
            CreateUIText("Volume Label", panel.transform,
                "Volume:", new Vector2(280, 20), new Vector2(0, 30));
            
            GameObject volumeSlider = CreateUISlider("Volume Slider", panel.transform,
                new Vector2(200, 20), new Vector2(0, 10));
            
            // Particle effects toggle
            CreateUIToggle("Particle Effects Toggle", panel.transform,
                "Particle Effects", new Vector2(200, 20), new Vector2(0, -20));
            
            // Reset button
            CreateUIButton("Reset Button", panel.transform,
                "Reset Progress", new Vector2(120, 30), new Vector2(-50, -60));
            
            // Close button
            CreateUIButton("Close Settings Button", panel.transform,
                "Close", new Vector2(80, 30), new Vector2(50, -60));
        }
        
        /// <summary>
        /// Setup world manager
        /// </summary>
        private void SetupWorldManager()
        {
            GameObject worldManagerObject = new GameObject("World Manager");
            WorldManager worldManager = worldManagerObject.AddComponent<WorldManager>();
            
            // Would normally assign references in inspector
            // For now, WorldManager will find components automatically
        }
        
        /// <summary>
        /// Setup game manager
        /// </summary>
        private void SetupGameManager()
        {
            GameObject gameManagerObject = new GameObject("Game Manager");
            GameManager gameManager = gameManagerObject.AddComponent<GameManager>();
            
            // Would normally assign references in inspector
            // For now, GameManager will find components automatically
        }
        
        /// <summary>
        /// Setup UI references for main UI controller
        /// </summary>
        private void SetupUIReferences(MainUIController uiController, Transform canvasTransform)
        {
            // This would normally be done in the inspector
            // For programmatic setup, we'd need to use reflection or public setters
            Debug.Log("UI Controller created - references should be assigned in inspector");
        }
        
        #region UI Helper Methods
        
        /// <summary>
        /// Create a UI panel
        /// </summary>
        private GameObject CreateUIPanel(string name, Transform parent, Vector2 size, Vector2 position, TextAnchor anchor)
        {
            GameObject panel = new GameObject(name);
            panel.transform.SetParent(parent);
            
            RectTransform rect = panel.AddComponent<RectTransform>();
            rect.sizeDelta = size;
            rect.anchoredPosition = position;
            
            // Set anchor based on parameter
            SetRectAnchor(rect, anchor);
            
            Image image = panel.AddComponent<Image>();
            image.color = new Color(0.1f, 0.1f, 0.1f, 0.8f);
            
            return panel;
        }
        
        /// <summary>
        /// Create UI text
        /// </summary>
        private GameObject CreateUIText(string name, Transform parent, string text, Vector2 size, Vector2 position, int fontSize = 16)
        {
            GameObject textObject = new GameObject(name);
            textObject.transform.SetParent(parent);
            
            RectTransform rect = textObject.AddComponent<RectTransform>();
            rect.sizeDelta = size;
            rect.anchoredPosition = position;
            
            TextMeshProUGUI textComponent = textObject.AddComponent<TextMeshProUGUI>();
            textComponent.text = text;
            textComponent.fontSize = fontSize;
            textComponent.color = Color.white;
            textComponent.alignment = TextAlignmentOptions.Center;
            
            return textObject;
        }
        
        /// <summary>
        /// Create UI button
        /// </summary>
        private GameObject CreateUIButton(string name, Transform parent, string text, Vector2 size, Vector2 position)
        {
            GameObject buttonObject = new GameObject(name);
            buttonObject.transform.SetParent(parent);
            
            RectTransform rect = buttonObject.AddComponent<RectTransform>();
            rect.sizeDelta = size;
            rect.anchoredPosition = position;
            
            Image image = buttonObject.AddComponent<Image>();
            image.color = new Color(0.3f, 0.3f, 0.3f, 1f);
            
            Button button = buttonObject.AddComponent<Button>();
            
            // Create text child
            GameObject textObject = new GameObject("Text");
            textObject.transform.SetParent(buttonObject.transform);
            
            RectTransform textRect = textObject.AddComponent<RectTransform>();
            textRect.anchorMin = Vector2.zero;
            textRect.anchorMax = Vector2.one;
            textRect.sizeDelta = Vector2.zero;
            textRect.anchoredPosition = Vector2.zero;
            
            TextMeshProUGUI textComponent = textObject.AddComponent<TextMeshProUGUI>();
            textComponent.text = text;
            textComponent.fontSize = 14;
            textComponent.color = Color.white;
            textComponent.alignment = TextAlignmentOptions.Center;
            
            button.targetGraphic = image;
            
            return buttonObject;
        }
        
        /// <summary>
        /// Create UI slider
        /// </summary>
        private GameObject CreateUISlider(string name, Transform parent, Vector2 size, Vector2 position)
        {
            GameObject sliderObject = new GameObject(name);
            sliderObject.transform.SetParent(parent);
            
            RectTransform rect = sliderObject.AddComponent<RectTransform>();
            rect.sizeDelta = size;
            rect.anchoredPosition = position;
            
            Slider slider = sliderObject.AddComponent<Slider>();
            slider.minValue = 0f;
            slider.maxValue = 1f;
            slider.value = 1f;
            
            return sliderObject;
        }
        
        /// <summary>
        /// Create UI toggle
        /// </summary>
        private GameObject CreateUIToggle(string name, Transform parent, string text, Vector2 size, Vector2 position)
        {
            GameObject toggleObject = new GameObject(name);
            toggleObject.transform.SetParent(parent);
            
            RectTransform rect = toggleObject.AddComponent<RectTransform>();
            rect.sizeDelta = size;
            rect.anchoredPosition = position;
            
            Toggle toggle = toggleObject.AddComponent<Toggle>();
            toggle.isOn = true;
            
            return toggleObject;
        }
        
        /// <summary>
        /// Create scrollable container
        /// </summary>
        private GameObject CreateScrollableContainer(string name, Transform parent, Vector2 size, Vector2 position)
        {
            GameObject container = new GameObject(name);
            container.transform.SetParent(parent);
            
            RectTransform rect = container.AddComponent<RectTransform>();
            rect.sizeDelta = size;
            rect.anchoredPosition = position;
            
            VerticalLayoutGroup layout = container.AddComponent<VerticalLayoutGroup>();
            layout.spacing = 2f;
            layout.padding = new RectOffset(5, 5, 5, 5);
            
            return container;
        }
        
        /// <summary>
        /// Set rect anchor
        /// </summary>
        private void SetRectAnchor(RectTransform rect, TextAnchor anchor)
        {
            switch (anchor)
            {
                case TextAnchor.UpperLeft:
                    rect.anchorMin = new Vector2(0, 1);
                    rect.anchorMax = new Vector2(0, 1);
                    break;
                case TextAnchor.UpperRight:
                    rect.anchorMin = new Vector2(1, 1);
                    rect.anchorMax = new Vector2(1, 1);
                    break;
                case TextAnchor.MiddleCenter:
                    rect.anchorMin = new Vector2(0.5f, 0.5f);
                    rect.anchorMax = new Vector2(0.5f, 0.5f);
                    break;
            }
        }
        
        #endregion
    }
}