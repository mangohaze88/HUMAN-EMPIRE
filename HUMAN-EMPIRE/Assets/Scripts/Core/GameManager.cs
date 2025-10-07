using UnityEngine;
using WorldNavigator.Navigation;
using WorldNavigator.UI;
using WorldNavigator.Lands;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Main game manager for the World Navigator game
    /// Coordinates navigation, UI, and land management
    /// </summary>
    public class GameManager : MonoBehaviour
    {
        [Header("Core Components")]
        [SerializeField] private NavigationController navigationController;
        [SerializeField] private LandInfoPanel landInfoPanel;
        
        [Header("Game Settings")]
        [SerializeField] private bool startWithRandomLand = true;
        [SerializeField] private LandType[] startingLands;
        
        // Statistics
        [System.Serializable]
        public class GameStats
        {
            public int landsDiscovered = 0;
            public int landsVisited = 0;
            public float totalPlayTime = 0f;
            public string[] discoveredLandNames;
        }
        
        public GameStats currentStats = new GameStats();
        
        // Events
        public System.Action<LandType> OnLandDiscovered;
        public System.Action<LandType> OnLandVisited;
        public System.Action<GameStats> OnStatsUpdated;
        
        private void Awake()
        {
            // Find components if not assigned
            if (navigationController == null)
                navigationController = FindFirstObjectByType<NavigationController>();
                
            if (landInfoPanel == null)
                landInfoPanel = FindFirstObjectByType<LandInfoPanel>();
        }
        
        private void Start()
        {
            SetupEventListeners();
            InitializeGame();
        }
        
        private void Update()
        {
            // Update play time
            currentStats.totalPlayTime += Time.deltaTime;
            
            // Handle UI input
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                if (landInfoPanel != null && landInfoPanel.IsVisible)
                    landInfoPanel.Hide();
            }
        }
        
        /// <summary>
        /// Setup event listeners for game components
        /// </summary>
        private void SetupEventListeners()
        {
            // Navigation events
            if (navigationController != null)
            {
                navigationController.OnLandReached += OnPlayerReachedLand;
            }
            
            // Land interaction events
            LandType[] allLands = FindObjectsByType<LandType>(FindObjectsSortMode.None);
            foreach (LandType land in allLands)
            {
                land.OnLandClicked += OnLandClicked;
                land.OnLandHovered += OnLandHovered;
            }
            
            // UI events
            if (landInfoPanel != null)
            {
                landInfoPanel.OnNavigateRequested += OnNavigateRequested;
            }
        }
        
        /// <summary>
        /// Initialize the game
        /// </summary>
        private void InitializeGame()
        {
            // Set starting position
            SetStartingPosition();
            
            // Update initial stats
            UpdateDiscoveryStats();
        }
        
        /// <summary>
        /// Set the starting position for the player
        /// </summary>
        private void SetStartingPosition()
        {
            LandType startLand = null;
            
            if (startWithRandomLand)
            {
                // Find a suitable starting land
                LandType[] allLands = FindObjectsByType<LandType>(FindObjectsSortMode.None);
                LandType[] suitableStarts = System.Array.FindAll(allLands, 
                    land => land.Data.rarity <= 2); // Easy/common lands only
                
                if (suitableStarts.Length > 0)
                {
                    startLand = suitableStarts[Random.Range(0, suitableStarts.Length)];
                }
            }
            else if (startingLands != null && startingLands.Length > 0)
            {
                startLand = startingLands[Random.Range(0, startingLands.Length)];
            }
            
            // Fallback to first land found
            if (startLand == null)
            {
                LandType[] allLands = FindObjectsByType<LandType>(FindObjectsSortMode.None);
                if (allLands.Length > 0)
                    startLand = allLands[0];
            }
            
            if (startLand != null && navigationController != null)
            {
                navigationController.SetStartingLand(startLand);
            }
        }
        
        /// <summary>
        /// Handle player reaching a land
        /// </summary>
        private void OnPlayerReachedLand(LandType land)
        {
            if (land == null) return;
            
            // Update stats
            currentStats.landsVisited++;
            OnLandVisited?.Invoke(land);
            OnStatsUpdated?.Invoke(currentStats);
            
            Debug.Log("Reached " + land.Data.landName + "!");
        }
        
        /// <summary>
        /// Handle land being clicked
        /// </summary>
        private void OnLandClicked(LandType land)
        {
            if (land == null || !land.IsDiscovered) return;
            
            // Show land information
            if (landInfoPanel != null)
            {
                landInfoPanel.ShowLandInfo(land);
            }
            
            // Deselect other lands
            LandType[] allLands = FindObjectsByType<LandType>(FindObjectsSortMode.None);
            foreach (LandType otherLand in allLands)
            {
                otherLand.SetSelected(otherLand == land);
            }
            
            Debug.Log("Selected " + land.Data.landName);
        }
        
        /// <summary>
        /// Handle land being hovered
        /// </summary>
        private void OnLandHovered(LandType land)
        {
            if (land == null || !land.IsDiscovered) return;
            
            // Show brief tooltip or highlight
            Debug.Log("Hovering over " + land.Data.landName);
        }
        
        /// <summary>
        /// Handle navigation request from UI
        /// </summary>
        private void OnNavigateRequested(LandType land)
        {
            if (land == null || !land.IsDiscovered || navigationController == null) return;
            
            navigationController.NavigateToLand(land);
            
            // Hide info panel
            if (landInfoPanel != null)
                landInfoPanel.Hide();
        }
        
        /// <summary>
        /// Update discovery statistics
        /// </summary>
        private void UpdateDiscoveryStats()
        {
            LandType[] allLands = FindObjectsByType<LandType>(FindObjectsSortMode.None);
            int discoveredCount = 0;
            System.Collections.Generic.List<string> discoveredNames = 
                new System.Collections.Generic.List<string>();
            
            foreach (LandType land in allLands)
            {
                if (land.IsDiscovered)
                {
                    discoveredCount++;
                    discoveredNames.Add(land.Data.landName);
                }
            }
            
            currentStats.landsDiscovered = discoveredCount;
            currentStats.discoveredLandNames = discoveredNames.ToArray();
            
            OnStatsUpdated?.Invoke(currentStats);
        }
        
        /// <summary>
        /// Get all lands of a specific category
        /// </summary>
        public LandType[] GetLandsByCategory(LandCategory category)
        {
            LandType[] allLands = FindObjectsByType<LandType>(FindObjectsSortMode.None);
            return System.Array.FindAll(allLands, land => land.Data.category == category);
        }
        
        /// <summary>
        /// Get all discovered lands
        /// </summary>
        public LandType[] GetDiscoveredLands()
        {
            if (navigationController != null)
                return navigationController.GetDiscoveredLands();
                
            LandType[] allLands = FindObjectsByType<LandType>(FindObjectsSortMode.None);
            return System.Array.FindAll(allLands, land => land.IsDiscovered);
        }
        
        /// <summary>
        /// Save game progress (for future implementation)
        /// </summary>
        public void SaveProgress()
        {
            // TODO: Implement save system
            Debug.Log("Game progress saved!");
        }
        
        /// <summary>
        /// Load game progress (for future implementation)
        /// </summary>
        public void LoadProgress()
        {
            // TODO: Implement load system
            Debug.Log("Game progress loaded!");
        }
    }
}