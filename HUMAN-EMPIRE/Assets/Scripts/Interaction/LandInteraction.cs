using UnityEngine;
using WorldNavigator.Lands;

namespace WorldNavigator.Interaction
{
    /// <summary>
    /// Handles mouse interaction with land objects
    /// </summary>
    public class LandInteraction : MonoBehaviour
    {
        [Header("Visual Feedback")]
        [SerializeField] private Color hoverColor = Color.white;
        [SerializeField] private Color selectedColor = Color.yellow;
        [SerializeField] private float glowIntensity = 2f;
        
        private LandType landType;
        private Renderer landRenderer;
        private Material originalMaterial;
        private Material glowMaterial;
        private bool isHovered = false;
        private bool isSelected = false;
        
        private void Start()
        {
            landType = GetComponent<LandType>();
            landRenderer = GetComponentInChildren<Renderer>();
            
            if (landRenderer != null)
            {
                originalMaterial = landRenderer.material;
                CreateGlowMaterial();
            }
        }
        
        /// <summary>
        /// Create glow material for highlighting
        /// </summary>
        private void CreateGlowMaterial()
        {
            glowMaterial = new Material(originalMaterial);
            glowMaterial.EnableKeyword("_EMISSION");
        }
        
        /// <summary>
        /// Handle mouse enter
        /// </summary>
        private void OnMouseEnter()
        {
            if (landType != null && landType.IsDiscovered)
            {
                isHovered = true;
                UpdateVisuals();
                
                // Show tooltip or info
                ShowLandTooltip();
            }
        }
        
        /// <summary>
        /// Handle mouse exit
        /// </summary>
        private void OnMouseExit()
        {
            isHovered = false;
            UpdateVisuals();
            
            // Hide tooltip
            HideLandTooltip();
        }
        
        /// <summary>
        /// Handle mouse click
        /// </summary>
        private void OnMouseDown()
        {
            if (landType != null && landType.IsDiscovered)
            {
                isSelected = !isSelected;
                UpdateVisuals();
                
                // Notify land clicked
                landType.OnClick();
                
                // Focus camera on this land
                CameraController camera = FindFirstObjectByType<CameraController>();
                if (camera != null)
                {
                    camera.FocusOn(transform.position);
                }
                
                Debug.Log("Clicked on " + landType.Data.landName);
            }
        }
        
        /// <summary>
        /// Update visual feedback
        /// </summary>
        private void UpdateVisuals()
        {
            if (landRenderer == null || glowMaterial == null) return;
            
            if (isSelected)
            {
                glowMaterial.SetColor("_EmissionColor", selectedColor * glowIntensity);
                landRenderer.material = glowMaterial;
            }
            else if (isHovered)
            {
                glowMaterial.SetColor("_EmissionColor", hoverColor * glowIntensity);
                landRenderer.material = glowMaterial;
            }
            else
            {
                landRenderer.material = originalMaterial;
            }
        }
        
        /// <summary>
        /// Show land information tooltip
        /// </summary>
        private void ShowLandTooltip()
        {
            if (landType != null)
            {
                // Create floating tooltip
                CreateFloatingTooltip(landType.Data.landName, landType.Data.description);
            }
        }
        
        /// <summary>
        /// Hide land tooltip
        /// </summary>
        private void HideLandTooltip()
        {
            // Remove tooltip
            GameObject tooltip = GameObject.Find("LandTooltip");
            if (tooltip != null)
            {
                Destroy(tooltip);
            }
        }
        
        /// <summary>
        /// Create floating tooltip UI
        /// </summary>
        private void CreateFloatingTooltip(string title, string description)
        {
            // Remove existing tooltip
            HideLandTooltip();
            
            // Create tooltip object
            GameObject tooltipObject = new GameObject("LandTooltip");
            
            // Position above land
            tooltipObject.transform.position = transform.position + Vector3.up * 8f;
            
            // Create UI components
            Canvas canvas = tooltipObject.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = Camera.main;
            
            // Background panel
            GameObject panel = new GameObject("Panel");
            panel.transform.SetParent(tooltipObject.transform);
            
            UnityEngine.UI.Image background = panel.AddComponent<UnityEngine.UI.Image>();
            background.color = new Color(0, 0, 0, 0.8f);
            
            RectTransform panelRect = panel.GetComponent<RectTransform>();
            panelRect.sizeDelta = new Vector2(300, 100);
            panelRect.localScale = Vector3.one * 0.01f; // Scale for world space
            
            // Title text
            GameObject titleObject = new GameObject("Title");
            titleObject.transform.SetParent(panel.transform);
            
            TMPro.TextMeshProUGUI titleText = titleObject.AddComponent<TMPro.TextMeshProUGUI>();
            titleText.text = title;
            titleText.fontSize = 24;
            titleText.fontStyle = TMPro.FontStyles.Bold;
            titleText.color = Color.white;
            titleText.alignment = TMPro.TextAlignmentOptions.Center;
            
            RectTransform titleRect = titleText.rectTransform;
            titleRect.anchorMin = Vector2.zero;
            titleRect.anchorMax = Vector2.one;
            titleRect.offsetMin = new Vector2(10, 50);
            titleRect.offsetMax = new Vector2(-10, -10);
            
            // Description text
            GameObject descObject = new GameObject("Description");
            descObject.transform.SetParent(panel.transform);
            
            TMPro.TextMeshProUGUI descText = descObject.AddComponent<TMPro.TextMeshProUGUI>();
            descText.text = description;
            descText.fontSize = 16;
            descText.color = new Color(0.9f, 0.9f, 0.9f, 1f);
            descText.alignment = TMPro.TextAlignmentOptions.Center;
            descText.textWrappingMode = TMPro.TextWrappingModes.Normal;
            
            RectTransform descRect = descText.rectTransform;
            descRect.anchorMin = Vector2.zero;
            descRect.anchorMax = Vector2.one;
            descRect.offsetMin = new Vector2(10, 10);
            descRect.offsetMax = new Vector2(-10, -50);
            
            // Make tooltip face camera
            tooltipObject.AddComponent<BillboardEffect>();
            
            // Auto-destroy after time
            Destroy(tooltipObject, 5f);
        }
        
        /// <summary>
        /// Set selection state
        /// </summary>
        public void SetSelected(bool selected)
        {
            isSelected = selected;
            UpdateVisuals();
        }
    }
}