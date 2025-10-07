using UnityEngine;
using UnityEngine.UI;
using TMPro;
using WorldNavigator.Core;
using WorldNavigator.Lands;
using WorldNavigator.Navigation;

namespace WorldNavigator.UI
{
    /// <summary>
    /// Main UI controller managing all UI elements
    /// Coordinates between different UI panels and game systems
    /// </summary>
    public class MainUIController : MonoBehaviour
    {
        [Header("UI Panels")]
        [SerializeField] private GameObject landInfoPanel;
        [SerializeField] private GameObject discoveryPanel;
        [SerializeField] private GameObject navigationPanel;
        [SerializeField] private GameObject settingsPanel;
        
        [Header("Land Info Panel")]
        [SerializeField] private TextMeshProUGUI landNameText;
        [SerializeField] private TextMeshProUGUI landCategoryText;
        [SerializeField] private TextMeshProUGUI landDescriptionText;
        [SerializeField] private Image landColorImage;
        [SerializeField] private Transform characteristicsContainer;
        [SerializeField] private Transform resourcesContainer;
        [SerializeField] private Button navigateButton;
        [SerializeField] private Button closeLandInfoButton;
        
        [Header("Discovery Panel")]
        [SerializeField] private TextMeshProUGUI discoveryCountText;
        [SerializeField] private TextMeshProUGUI instructionsText;
        [SerializeField] private Slider discoveryProgressSlider;
        
        [Header("Navigation Panel")]
        [SerializeField] private TextMeshProUGUI currentLocationText;
        [SerializeField] private Button centerCameraButton;
        [SerializeField] private TextMeshProUGUI navigationStatusText;
        
        [Header("Settings Panel")]
        [SerializeField] private Slider volumeSlider;
        [SerializeField] private Toggle particleEffectsToggle;
        [SerializeField] private Button resetProgressButton;
        [SerializeField] private Button closeSettingsButton;
        
        [Header("UI Prefabs")]
        [SerializeField] private GameObject characteristicItemPrefab;
        [SerializeField] private GameObject resourceItemPrefab;
        
        // Private references
        private GameManager gameManager;
        private NavigationController navigationController;
        private WorldManager worldManager;
        private LandType currentDisplayedLand;
        private CanvasGroup landInfoCanvasGroup;
        
        // UI state
        private bool isLandInfoVisible = false;
        
        private void Awake()
        {
            InitializeReferences();
            SetupUI();
        }
        
        private void Start()
        {
            SetupEventListeners();
            UpdateAllUI();
        }
        
        /// <summary>
        /// Initialize component references
        /// </summary>
        private void InitializeReferences()
        {
            gameManager = FindObjectOfType<GameManager>();
            navigationController = FindObjectOfType<NavigationController>();
            worldManager = FindObjectOfType<WorldManager>();
            
            if (landInfoPanel != null)
            {
                landInfoCanvasGroup = landInfoPanel.GetComponent<CanvasGroup>();
                if (landInfoCanvasGroup == null)
                    landInfoCanvasGroup = landInfoPanel.AddComponent<CanvasGroup>();
            }
        }
        
        /// <summary>
        /// Setup UI components
        /// </summary>
        private void SetupUI()
        {
            // Create prefabs if not assigned
            CreateUIPrefabs();
            
            // Initialize panel states
            SetLandInfoPanelVisible(false, true);
            SetSettingsPanelVisible(false);
            
            // Setup initial text
            if (instructionsText != null)
            {
                instructionsText.text = "🌍 World Navigator\\n\\n" +
                    "• Click on discovered lands to explore\\n" +
                    "• Use right mouse to pan camera\\n" +
                    "• Mouse wheel to zoom\\n" +
                    "• Discover new lands by visiting nearby areas";
            }
        }
        
        /// <summary>
        /// Setup event listeners
        /// </summary>
        private void SetupEventListeners()
        {
            // Land info panel buttons
            if (navigateButton != null)
                navigateButton.onClick.AddListener(OnNavigateButtonClicked);
                
            if (closeLandInfoButton != null)
                closeLandInfoButton.onClick.AddListener(() => SetLandInfoPanelVisible(false));
            
            // Navigation panel buttons
            if (centerCameraButton != null)
                centerCameraButton.onClick.AddListener(OnCenterCameraClicked);
            
            // Settings panel buttons
            if (closeSettingsButton != null)
                closeSettingsButton.onClick.AddListener(() => SetSettingsPanelVisible(false));
                
            if (resetProgressButton != null)
                resetProgressButton.onClick.AddListener(OnResetProgressClicked);
            
            // Settings sliders and toggles
            if (volumeSlider != null)
                volumeSlider.onValueChanged.AddListener(OnVolumeChanged);
                
            if (particleEffectsToggle != null)
                particleEffectsToggle.onValueChanged.AddListener(OnParticleEffectsToggled);
            
            // Game manager events
            if (gameManager != null)
            {
                gameManager.OnLandVisited += OnLandVisited;
                gameManager.OnStatsUpdated += OnStatsUpdated;
            }
            
            // Navigation controller events
            if (navigationController != null)
            {
                navigationController.OnNavigationStarted += OnNavigationStarted;
                navigationController.OnNavigationCompleted += OnNavigationCompleted;
                navigationController.OnLandReached += OnLandReached;
            }
        }
        
        /// <summary>
        /// Update method for UI animations and state
        /// </summary>
        private void Update()
        {
            UpdateLandInfoPanelAnimation();
            HandleUIInput();
        }
        
        /// <summary>
        /// Handle UI input
        /// </summary>
        private void HandleUIInput()
        {
            // Toggle settings with ESC
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                if (isLandInfoVisible)
                    SetLandInfoPanelVisible(false);
                else
                    ToggleSettingsPanel();
            }
            
            // Show/hide UI with Tab
            if (Input.GetKeyDown(KeyCode.Tab))
            {
                ToggleUIVisibility();
            }
        }
        
        /// <summary>
        /// Show land information
        /// </summary>
        public void ShowLandInfo(LandType land)
        {
            if (land == null) return;
            
            currentDisplayedLand = land;
            UpdateLandInfoDisplay();
            SetLandInfoPanelVisible(true);
        }
        
        /// <summary>
        /// Update land info display
        /// </summary>
        private void UpdateLandInfoDisplay()
        {
            if (currentDisplayedLand == null) return;
            
            LandData data = currentDisplayedLand.Data;
            
            // Update basic info
            if (landNameText != null)
                landNameText.text = data.landName;
                
            if (landCategoryText != null)
                landCategoryText.text = data.category.ToString();
                
            if (landDescriptionText != null)
                landDescriptionText.text = data.description;
                
            if (landColorImage != null)
                landColorImage.color = data.primaryColor;
            
            // Update characteristics
            UpdateCharacteristicsList(data.characteristics);
            UpdateResourcesList(data.resources);
            
            // Update navigate button
            if (navigateButton != null)
                navigateButton.interactable = data.isDiscovered;
        }
        
        /// <summary>
        /// Update characteristics list
        /// </summary>
        private void UpdateCharacteristicsList(string[] characteristics)
        {
            if (characteristicsContainer == null) return;
            
            // Clear existing
            foreach (Transform child in characteristicsContainer)
                Destroy(child.gameObject);
            
            // Add characteristics
            if (characteristics != null)
            {
                foreach (string characteristic in characteristics)
                {
                    CreateListItem(characteristic, characteristicsContainer, "🔹 ");
                }
            }
        }
        
        /// <summary>
        /// Update resources list
        /// </summary>
        private void UpdateResourcesList(string[] resources)
        {
            if (resourcesContainer == null) return;
            
            // Clear existing
            foreach (Transform child in resourcesContainer)
                Destroy(child.gameObject);
            
            // Add resources
            if (resources != null)
            {
                foreach (string resource in resources)
                {
                    CreateListItem(resource, resourcesContainer, "📦 ");
                }
            }
        }
        
        /// <summary>
        /// Create a list item for characteristics or resources
        /// </summary>
        private void CreateListItem(string text, Transform parent, string prefix = "")
        {
            GameObject item = new GameObject("ListItem");
            item.transform.SetParent(parent);
            
            TextMeshProUGUI textComponent = item.AddComponent<TextMeshProUGUI>();
            textComponent.text = prefix + text;
            textComponent.fontSize = 14f;
            textComponent.color = Color.white;
            textComponent.margin = new Vector4(5, 2, 5, 2);
            
            // Add layout element
            LayoutElement layoutElement = item.AddComponent<LayoutElement>();
            layoutElement.preferredHeight = 20f;
        }
        
        /// <summary>
        /// Update discovery UI
        /// </summary>
        private void UpdateDiscoveryUI()
        {
            if (worldManager == null) return;
            
            LandType[] allLands = worldManager.GetAllLands();
            if (allLands == null) return;
            
            int discoveredCount = 0;
            foreach (LandType land in allLands)
            {
                if (land != null && land.IsDiscovered)
                    discoveredCount++;
            }
            
            // Update discovery text
            if (discoveryCountText != null)
            {
                discoveryCountText.text = $"Discovered: {discoveredCount}/{allLands.Length} Lands";
            }
            
            // Update progress slider
            if (discoveryProgressSlider != null)
            {
                discoveryProgressSlider.value = (float)discoveredCount / allLands.Length;
            }
        }
        
        /// <summary>
        /// Update navigation UI
        /// </summary>
        private void UpdateNavigationUI()
        {
            if (navigationController == null) return;
            
            // Update current location
            if (currentLocationText != null)
            {
                LandType currentLand = navigationController.CurrentLand;
                if (currentLand != null)
                {
                    currentLocationText.text = $"Current: {currentLand.Data.landName}";
                }
                else
                {
                    currentLocationText.text = "Current: Exploring";
                }
            }
        }
        
        /// <summary>
        /// Update all UI elements
        /// </summary>
        private void UpdateAllUI()
        {
            UpdateDiscoveryUI();
            UpdateNavigationUI();
        }
        
        /// <summary>
        /// Set land info panel visibility
        /// </summary>
        private void SetLandInfoPanelVisible(bool visible, bool immediate = false)
        {
            isLandInfoVisible = visible;
            
            if (landInfoPanel != null)
            {
                landInfoPanel.SetActive(visible);
                
                if (immediate && landInfoCanvasGroup != null)
                {
                    landInfoCanvasGroup.alpha = visible ? 1f : 0f;
                }
            }
        }
        
        /// <summary>
        /// Animate land info panel visibility
        /// </summary>
        private void UpdateLandInfoPanelAnimation()
        {
            if (landInfoCanvasGroup == null) return;
            
            float targetAlpha = isLandInfoVisible ? 1f : 0f;
            landInfoCanvasGroup.alpha = Mathf.Lerp(landInfoCanvasGroup.alpha, targetAlpha, Time.deltaTime * 6f);
        }
        
        /// <summary>
        /// Set settings panel visibility
        /// </summary>
        private void SetSettingsPanelVisible(bool visible)
        {
            if (settingsPanel != null)
                settingsPanel.SetActive(visible);
        }
        
        /// <summary>
        /// Toggle settings panel
        /// </summary>
        private void ToggleSettingsPanel()
        {
            if (settingsPanel != null)
                settingsPanel.SetActive(!settingsPanel.activeSelf);
        }
        
        /// <summary>
        /// Toggle UI visibility
        /// </summary>
        private void ToggleUIVisibility()
        {
            CanvasGroup canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null)
                canvasGroup = gameObject.AddComponent<CanvasGroup>();
                
            canvasGroup.alpha = canvasGroup.alpha > 0.5f ? 0.2f : 1f;
        }
        
        /// <summary>
        /// Create UI prefabs if not assigned
        /// </summary>
        private void CreateUIPrefabs()
        {
            if (characteristicItemPrefab == null)
            {
                characteristicItemPrefab = CreateTextItemPrefab("CharacteristicItem");
            }
            
            if (resourceItemPrefab == null)
            {
                resourceItemPrefab = CreateTextItemPrefab("ResourceItem");
            }
        }
        
        /// <summary>
        /// Create a text item prefab
        /// </summary>
        private GameObject CreateTextItemPrefab(string name)
        {
            GameObject prefab = new GameObject(name);
            TextMeshProUGUI text = prefab.AddComponent<TextMeshProUGUI>();
            text.fontSize = 14f;
            text.color = Color.white;
            
            LayoutElement layout = prefab.AddComponent<LayoutElement>();
            layout.preferredHeight = 20f;
            
            return prefab;
        }
        
        #region Event Handlers
        
        private void OnNavigateButtonClicked()
        {
            if (currentDisplayedLand != null && navigationController != null)
            {
                navigationController.NavigateToLand(currentDisplayedLand);
                SetLandInfoPanelVisible(false);
            }
        }
        
        private void OnCenterCameraClicked()
        {
            if (navigationController != null && navigationController.CurrentLand != null)
            {
                navigationController.NavigateToLand(navigationController.CurrentLand);
            }
        }
        
        private void OnResetProgressClicked()
        {
            // TODO: Implement reset functionality
            Debug.Log("Reset progress clicked");
        }
        
        private void OnVolumeChanged(float value)
        {
            AudioListener.volume = value;
        }
        
        private void OnParticleEffectsToggled(bool enabled)
        {
            // TODO: Implement particle effects toggle
            Debug.Log($"Particle effects: {enabled}");
        }
        
        private void OnLandVisited(LandType land)
        {
            UpdateNavigationUI();
            UpdateDiscoveryUI();
        }
        
        private void OnStatsUpdated(GameManager.GameStats stats)
        {
            UpdateDiscoveryUI();
        }
        
        private void OnNavigationStarted()
        {
            if (navigationStatusText != null)
                navigationStatusText.text = "Traveling...";
        }
        
        private void OnNavigationCompleted()
        {
            if (navigationStatusText != null)
                navigationStatusText.text = "Arrived!";
                
            UpdateNavigationUI();
        }
        
        private void OnLandReached(LandType land)
        {
            UpdateNavigationUI();
        }
        
        #endregion
    }
}