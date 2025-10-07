using UnityEngine;
using WorldNavigator.Lands;

namespace WorldNavigator.Navigation
{
    /// <summary>
    /// Handles player navigation between different lands
    /// Manages camera movement and player positioning
    /// </summary>
    public class NavigationController : MonoBehaviour
    {
        [Header("Camera Settings")]
        [SerializeField] private Camera playerCamera;
        [SerializeField] private float moveSpeed = 5f;
        [SerializeField] private float rotationSpeed = 2f;
        [SerializeField] private float zoomSpeed = 3f;
        [SerializeField] private float minZoom = 5f;
        [SerializeField] private float maxZoom = 20f;
        
        [Header("Navigation Settings")]
        [SerializeField] private LayerMask landLayerMask = 1;
        [SerializeField] private float landVisitDistance = 2f;
        
        // Private variables
        private LandType currentLand;
        private LandType targetLand;
        private Vector3 targetPosition;
        private bool isMoving = false;
        
        // Input handling
        private Vector3 lastMousePosition;
        private bool isDragging = false;
        
        // Events
        public System.Action<LandType> OnLandReached;
        public System.Action OnNavigationStarted;
        public System.Action OnNavigationCompleted;
        
        public LandType CurrentLand => currentLand;
        public bool IsMoving => isMoving;
        
        private void Awake()
        {
            if (playerCamera == null)
                playerCamera = Camera.main;
                
            if (playerCamera == null)
                playerCamera = FindObjectOfType<Camera>();
        }
        
        private void Update()
        {
            HandleInput();
            UpdateMovement();
            HandleCameraControls();
        }
        
        /// <summary>
        /// Handle mouse and keyboard input
        /// </summary>
        private void HandleInput()
        {
            // Handle land clicking
            if (Input.GetMouseButtonDown(0) && !isDragging)
            {
                CheckLandClick();
            }
            
            // Handle camera dragging
            if (Input.GetMouseButtonDown(1)) // Right mouse button
            {
                lastMousePosition = Input.mousePosition;
                isDragging = true;
            }
            
            if (Input.GetMouseButtonUp(1))
            {
                isDragging = false;
            }
            
            // Handle zoom with mouse wheel
            float scroll = Input.GetAxis(\"Mouse ScrollWheel\");
            if (Mathf.Abs(scroll) > 0.01f)
            {
                ZoomCamera(-scroll * zoomSpeed);
            }
        }
        
        /// <summary>
        /// Check if player clicked on a land
        /// </summary>
        private void CheckLandClick()
        {
            Ray ray = playerCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            
            if (Physics.Raycast(ray, out hit, Mathf.Infinity, landLayerMask))
            {
                LandType clickedLand = hit.collider.GetComponent<LandType>();
                if (clickedLand != null && clickedLand.IsDiscovered)
                {
                    NavigateToLand(clickedLand);
                }
            }
        }
        
        /// <summary>
        /// Navigate to a specific land
        /// </summary>
        public void NavigateToLand(LandType land)
        {
            if (land == null || isMoving) return;
            
            targetLand = land;
            targetPosition = land.transform.position + Vector3.up * 2f; // Slightly above the land
            isMoving = true;
            
            OnNavigationStarted?.Invoke();
        }
        
        /// <summary>
        /// Update camera movement towards target
        /// </summary>
        private void UpdateMovement()
        {
            if (!isMoving || targetLand == null) return;
            
            // Move camera towards target position
            float distance = Vector3.Distance(transform.position, targetPosition);
            
            if (distance > 0.1f)
            {
                Vector3 direction = (targetPosition - transform.position).normalized;
                transform.position += direction * moveSpeed * Time.deltaTime;
                
                // Look at the target land
                Vector3 lookDirection = (targetLand.transform.position - transform.position).normalized;
                if (lookDirection != Vector3.zero)
                {
                    Quaternion targetRotation = Quaternion.LookRotation(lookDirection);
                    transform.rotation = Quaternion.Slerp(transform.rotation, targetRotation, 
                        rotationSpeed * Time.deltaTime);
                }
            }
            else
            {
                // Reached destination
                transform.position = targetPosition;
                isMoving = false;
                
                // Update current land
                LandType previousLand = currentLand;
                currentLand = targetLand;
                targetLand = null;
                
                OnLandReached?.Invoke(currentLand);
                OnNavigationCompleted?.Invoke();
                
                // Discover adjacent lands
                DiscoverAdjacentLands();
            }
        }
        
        /// <summary>
        /// Handle camera controls (pan and zoom)
        /// </summary>
        private void HandleCameraControls()
        {
            if (isDragging)
            {
                Vector3 deltaMousePosition = Input.mousePosition - lastMousePosition;
                
                // Pan camera based on mouse movement
                Vector3 panDirection = new Vector3(-deltaMousePosition.x, 0, -deltaMousePosition.y);
                panDirection = transform.TransformDirection(panDirection);
                panDirection.y = 0; // Keep movement horizontal
                
                transform.position += panDirection * moveSpeed * 0.01f;
                
                lastMousePosition = Input.mousePosition;
            }
            
            // Keyboard movement
            float horizontal = Input.GetAxis(\"Horizontal\");
            float vertical = Input.GetAxis(\"Vertical\");
            
            if (Mathf.Abs(horizontal) > 0.01f || Mathf.Abs(vertical) > 0.01f)
            {
                Vector3 moveDirection = new Vector3(horizontal, 0, vertical);
                moveDirection = transform.TransformDirection(moveDirection);
                moveDirection.y = 0;
                
                transform.position += moveDirection * moveSpeed * Time.deltaTime;
            }
        }
        
        /// <summary>
        /// Zoom camera in or out
        /// </summary>
        private void ZoomCamera(float zoomAmount)
        {
            if (playerCamera.orthographic)
            {
                playerCamera.orthographicSize = Mathf.Clamp(
                    playerCamera.orthographicSize + zoomAmount,
                    minZoom, maxZoom);
            }
            else
            {
                Vector3 forward = transform.forward * zoomAmount;
                transform.position += forward;
            }
        }
        
        /// <summary>
        /// Discover lands adjacent to current land
        /// </summary>
        private void DiscoverAdjacentLands()
        {
            if (currentLand == null) return;
            
            // Find all lands within discovery range
            LandType[] allLands = FindObjectsOfType<LandType>();
            
            foreach (LandType land in allLands)
            {
                if (land == currentLand || land.IsDiscovered) continue;
                
                float distance = currentLand.GetDistanceTo(land);
                if (distance <= landVisitDistance * 3f) // Larger range for discovery
                {
                    land.DiscoverLand();
                }
            }
        }
        
        /// <summary>
        /// Set starting position at a specific land
        /// </summary>
        public void SetStartingLand(LandType startLand)
        {
            if (startLand == null) return;
            
            currentLand = startLand;
            transform.position = startLand.transform.position + Vector3.up * 2f;
            
            // Look at the land
            Vector3 lookDirection = (startLand.transform.position - transform.position).normalized;
            if (lookDirection != Vector3.zero)
            {
                transform.rotation = Quaternion.LookRotation(lookDirection);
            }
            
            // Discover starting land and nearby lands
            startLand.DiscoverLand();
            DiscoverAdjacentLands();
        }
        
        /// <summary>
        /// Get all discovered lands
        /// </summary>
        public LandType[] GetDiscoveredLands()
        {
            LandType[] allLands = FindObjectsOfType<LandType>();
            System.Collections.Generic.List<LandType> discoveredLands = 
                new System.Collections.Generic.List<LandType>();
            
            foreach (LandType land in allLands)
            {
                if (land.IsDiscovered)
                    discoveredLands.Add(land);
            }
            
            return discoveredLands.ToArray();
        }
    }
}