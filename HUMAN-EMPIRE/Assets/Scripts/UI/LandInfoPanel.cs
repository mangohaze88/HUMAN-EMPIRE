using UnityEngine;
using UnityEngine.UI;
using TMPro;
using WorldNavigator.Core;
using WorldNavigator.Lands;

namespace WorldNavigator.UI
{
    /// <summary>
    /// UI panel that displays information about selected lands
    /// </summary>
    public class LandInfoPanel : MonoBehaviour
    {
        [Header("UI References")]
        [SerializeField] private GameObject panelRoot;
        [SerializeField] private TextMeshProUGUI landNameText;
        [SerializeField] private TextMeshProUGUI landCategoryText;
        [SerializeField] private TextMeshProUGUI landDescriptionText;
        [SerializeField] private Transform characteristicsContainer;
        [SerializeField] private Transform resourcesContainer;
        [SerializeField] private GameObject characteristicItemPrefab;
        [SerializeField] private Button navigateButton;
        [SerializeField] private Button closeButton;
        
        [Header("Visual Effects")]
        [SerializeField] private Image landColorIndicator;
        [SerializeField] private Image backgroundImage;
        [SerializeField] private float fadeSpeed = 3f;
        
        // Private variables
        private LandType currentDisplayedLand;
        private CanvasGroup canvasGroup;
        private bool isVisible = false;
        
        // Events
        public System.Action<LandType> OnNavigateRequested;
        public System.Action OnPanelClosed;
        
        public bool IsVisible => isVisible;
        public LandType CurrentLand => currentDisplayedLand;
        
        private void Awake()
        {
            canvasGroup = GetComponent<CanvasGroup>();
            if (canvasGroup == null)
                canvasGroup = gameObject.AddComponent<CanvasGroup>();
            
            SetupButtons();
            Hide(true); // Start hidden
        }
        
        private void Start()
        {
            // Create characteristic item prefab if none assigned
            if (characteristicItemPrefab == null)
                CreateCharacteristicItemPrefab();
        }
        
        private void Update()
        {
            UpdateVisibility();
        }
        
        /// <summary>
        /// Setup button event listeners
        /// </summary>
        private void SetupButtons()
        {
            if (navigateButton != null)
            {
                navigateButton.onClick.AddListener(() => {
                    OnNavigateRequested?.Invoke(currentDisplayedLand);
                });
            }
            
            if (closeButton != null)
            {
                closeButton.onClick.AddListener(() => {
                    Hide();
                    OnPanelClosed?.Invoke();
                });
            }
        }
        
        /// <summary>
        /// Display information for a specific land
        /// </summary>
        public void ShowLandInfo(LandType land)
        {
            if (land == null) return;
            
            currentDisplayedLand = land;
            UpdateLandDisplay();
            Show();
        }
        
        /// <summary>
        /// Update the display with current land information
        /// </summary>
        private void UpdateLandDisplay()
        {
            if (currentDisplayedLand == null) return;
            
            LandData data = currentDisplayedLand.Data;
            
            // Update text fields
            if (landNameText != null)
                landNameText.text = data.landName;
                
            if (landCategoryText != null)
                landCategoryText.text = data.category.ToString();
                
            if (landDescriptionText != null)
                landDescriptionText.text = data.description;
            
            // Update visual indicators
            if (landColorIndicator != null)
                landColorIndicator.color = data.primaryColor;
                
            if (backgroundImage != null)
            {
                Color bgColor = data.primaryColor;
                bgColor.a = 0.1f;
                backgroundImage.color = bgColor;
            }
            
            // Update characteristics
            UpdateCharacteristics(data.characteristics);
            UpdateResources(data.resources);
            
            // Update navigate button
            if (navigateButton != null)
                navigateButton.interactable = data.isDiscovered;
        }
        
        /// <summary>
        /// Update characteristics list
        /// </summary>
        private void UpdateCharacteristics(string[] characteristics)
        {
            if (characteristicsContainer == null) return;
            
            // Clear existing items
            foreach (Transform child in characteristicsContainer)
            {
                Destroy(child.gameObject);
            }
            
            // Add characteristic items
            if (characteristics != null)
            {
                foreach (string characteristic in characteristics)
                {
                    CreateCharacteristicItem(characteristic, characteristicsContainer);
                }
            }
        }
        
        /// <summary>
        /// Update resources list
        /// </summary>
        private void UpdateResources(string[] resources)
        {
            if (resourcesContainer == null) return;
            
            // Clear existing items
            foreach (Transform child in resourcesContainer)
            {
                Destroy(child.gameObject);
            }
            
            // Add resource items
            if (resources != null)
            {
                foreach (string resource in resources)
                {
                    CreateCharacteristicItem(resource, resourcesContainer);
                }
            }
        }
        
        /// <summary>
        /// Create a characteristic/resource item
        /// </summary>
        private void CreateCharacteristicItem(string text, Transform container)
        {
            GameObject item;
            
            if (characteristicItemPrefab != null)
            {
                item = Instantiate(characteristicItemPrefab, container);
            }
            else
            {
                // Create simple text item
                item = new GameObject("Characteristic");
                item.transform.SetParent(container);
                
                TextMeshProUGUI textComponent = item.AddComponent<TextMeshProUGUI>();
                textComponent.text = "• " + text;
                textComponent.fontSize = 14f;
                textComponent.color = Color.white;
            }
            
            // Set text if the item has a text component
            TextMeshProUGUI textComp = item.GetComponent<TextMeshProUGUI>();
            if (textComp != null)
                textComp.text = "• " + text;
        }
        
        /// <summary>
        /// Create default characteristic item prefab
        /// </summary>
        private void CreateCharacteristicItemPrefab()
        {
            GameObject prefab = new GameObject("CharacteristicItem");
            
            TextMeshProUGUI text = prefab.AddComponent<TextMeshProUGUI>();
            text.text = "• Characteristic";
            text.fontSize = 14f;
            text.color = Color.white;
            
            characteristicItemPrefab = prefab;
        }
        
        /// <summary>
        /// Show the panel
        /// </summary>
        public void Show()
        {
            isVisible = true;
            if (panelRoot != null)
                panelRoot.SetActive(true);
        }
        
        /// <summary>
        /// Hide the panel
        /// </summary>
        public void Hide(bool immediate = false)
        {
            isVisible = false;
            
            if (immediate)
            {
                canvasGroup.alpha = 0f;
                if (panelRoot != null)
                    panelRoot.SetActive(false);
            }
        }
        
        /// <summary>
        /// Update panel visibility with smooth transitions
        /// </summary>
        private void UpdateVisibility()
        {
            if (canvasGroup == null) return;
            
            float targetAlpha = isVisible ? 1f : 0f;
            canvasGroup.alpha = Mathf.Lerp(canvasGroup.alpha, targetAlpha, fadeSpeed * Time.deltaTime);
            
            // Disable panel when fully faded out
            if (!isVisible && canvasGroup.alpha < 0.01f)
            {
                if (panelRoot != null)
                    panelRoot.SetActive(false);
            }
        }
        
        /// <summary>
        /// Check if mouse is over this panel
        /// </summary>
        public bool IsMouseOver()
        {
            if (!isVisible || canvasGroup.alpha < 0.5f) return false;
            
            RectTransform rectTransform = GetComponent<RectTransform>();
            if (rectTransform == null) return false;
            
            Vector2 mousePosition = Input.mousePosition;
            return RectTransformUtility.RectangleContainsScreenPoint(rectTransform, mousePosition);
        }
        
        /// <summary>
        /// Toggle panel visibility
        /// </summary>
        public void Toggle()
        {
            if (isVisible)
                Hide();
            else
                Show();
        }
    }
}