using UnityEngine;
using WorldNavigator.Core;

namespace WorldNavigator.Lands
{
    /// <summary>
    /// Base component for all land types in the world
    /// Handles interaction, visual effects, and audio
    /// </summary>
    public class LandType : MonoBehaviour
    {
        [Header("Land Configuration")]
        [SerializeField] private LandData landData;
        
        [Header("Visual Effects")]
        [SerializeField] private ParticleSystem ambientParticles;
        [SerializeField] private Light landLight;
        [SerializeField] private Renderer landRenderer;
        
        [Header("Interaction")]
        [SerializeField] private bool isInteractable = true;
        [SerializeField] private float hoverScale = 1.1f;
        
        // Private variables
        private Vector3 originalScale;
        private AudioSource audioSource;
        private bool isHovered = false;
        private bool isSelected = false;
        
        // Events
        public System.Action<LandType> OnLandClicked;
        public System.Action<LandType> OnLandHovered;
        public System.Action<LandType> OnLandUnhovered;
        
        public LandData Data => landData;
        public bool IsDiscovered => landData.isDiscovered;
        
        private void Awake()
        {
            originalScale = transform.localScale;
            audioSource = GetComponent<AudioSource>();
            if (audioSource == null)
                audioSource = gameObject.AddComponent<AudioSource>();
                
            InitializeLand();
        }
        
        private void Start()
        {
            SetupVisuals();
            SetupAudio();
        }
        
        /// <summary>
        /// Initialize land with data
        /// </summary>
        private void InitializeLand()
        {
            if (landData == null) return;
            
            name = $"Land_{landData.landName}";
            landData.worldPosition = transform.position;
            
            // Set initial discovery state (first few lands discovered by default)
            if (landData.rarity <= 2)
                landData.isDiscovered = true;
        }
        
        /// <summary>
        /// Setup visual components
        /// </summary>
        private void SetupVisuals()
        {
            if (landData == null) return;
            
            // Apply material if available
            if (landRenderer != null && landData.landMaterial != null)
            {
                landRenderer.material = landData.landMaterial;
            }
            
            // Setup lighting
            if (landLight != null)
            {
                landLight.color = landData.primaryColor;
            }
            
            // Show/hide based on discovery state
            UpdateVisualState();
        }
        
        /// <summary>
        /// Setup audio components
        /// </summary>
        private void SetupAudio()
        {
            if (landData == null || audioSource == null) return;
            
            if (landData.ambientSound != null)
            {
                audioSource.clip = landData.ambientSound;
                audioSource.loop = true;
                audioSource.volume = 0.3f;
                audioSource.spatialBlend = 0.7f; // 3D sound
                audioSource.maxDistance = 20f;
            }
        }
        
        /// <summary>
        /// Update visual state based on discovery and interaction
        /// </summary>
        private void UpdateVisualState()
        {
            if (!landData.isDiscovered)
            {
                // Undiscovered lands are dimmed
                if (landRenderer != null)
                {
                    var material = landRenderer.material;
                    var color = material.color;
                    color.a = 0.3f;
                    material.color = color;
                }
                return;
            }
            
            // Visual feedback for hover/selection
            float targetScale = isHovered ? hoverScale : 1f;
            if (isSelected) targetScale *= 1.05f;
            
            transform.localScale = Vector3.Lerp(transform.localScale, 
                originalScale * targetScale, Time.deltaTime * 8f);
        }
        
        private void Update()
        {
            UpdateVisualState();
        }
        
        /// <summary>
        /// Handle mouse hover enter
        /// </summary>
        private void OnMouseEnter()
        {
            if (!isInteractable || !landData.isDiscovered) return;
            
            isHovered = true;
            OnLandHovered?.Invoke(this);
            
            // Play ambient sound
            if (audioSource != null && !audioSource.isPlaying)
                audioSource.Play();
        }
        
        /// <summary>
        /// Handle mouse hover exit
        /// </summary>
        private void OnMouseExit()
        {
            if (!isInteractable) return;
            
            isHovered = false;
            OnLandUnhovered?.Invoke(this);
            
            // Stop ambient sound if not selected
            if (!isSelected && audioSource != null && audioSource.isPlaying)
                audioSource.Stop();
        }
        
        /// <summary>
        /// Handle mouse click
        /// </summary>
        private void OnMouseDown()
        {
            if (!isInteractable || !landData.isDiscovered) return;
            
            OnLandClicked?.Invoke(this);
            
            // Play click sound
            if (landData.clickSound != null && audioSource != null)
            {
                audioSource.PlayOneShot(landData.clickSound);
            }
        }
        
        /// <summary>
        /// Set land as selected
        /// </summary>
        public void SetSelected(bool selected)
        {
            isSelected = selected;
            
            if (selected)
            {
                // Keep ambient sound playing when selected
                if (audioSource != null && !audioSource.isPlaying)
                    audioSource.Play();
            }
            else
            {
                // Stop ambient sound when deselected (unless hovered)
                if (!isHovered && audioSource != null && audioSource.isPlaying)
                    audioSource.Stop();
            }
        }
        
        /// <summary>
        /// Discover this land (make it interactable)
        /// </summary>
        public void DiscoverLand()
        {
            if (landData.isDiscovered) return;
            
            landData.isDiscovered = true;
            UpdateVisualState();
            
            // Play discovery effect
            if (ambientParticles != null)
                ambientParticles.Play();
        }
        
        /// <summary>
        /// Get distance to another land
        /// </summary>
        public float GetDistanceTo(LandType otherLand)
        {
            return Vector3.Distance(transform.position, otherLand.transform.position);
        }
        
        /// <summary>
        /// Check if this land is adjacent to another
        /// </summary>
        public bool IsAdjacentTo(LandType otherLand, float maxDistance = 10f)
        {
            return GetDistanceTo(otherLand) <= maxDistance;
        }
        
        /// <summary>
        /// Public method to trigger click event
        /// </summary>
        public void OnClick()
        {
            OnLandClicked?.Invoke(this);
        }
    }
}