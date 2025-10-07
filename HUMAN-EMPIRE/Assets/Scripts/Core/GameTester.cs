using UnityEngine;
using WorldNavigator.Lands;
using WorldNavigator.Navigation;
using WorldNavigator.UI;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Test script to verify all game systems work correctly
    /// Can be used to test navigation, land discovery, UI, etc.
    /// </summary>
    public class GameTester : MonoBehaviour
    {
        [Header("Testing Options")]
        [SerializeField] private bool enableTestMode = false;
        [SerializeField] private bool testNavigation = true;
        [SerializeField] private bool testLandDiscovery = true;
        [SerializeField] private bool testUI = true;
        [SerializeField] private bool showDebugInfo = true;
        
        [Header("Test Settings")]
        [SerializeField] private float testInterval = 2f;
        [SerializeField] private KeyCode testKey = KeyCode.T;
        [SerializeField] private KeyCode discoverAllKey = KeyCode.R;
        
        // Component references
        private GameManager gameManager;
        private NavigationController navigationController;
        private WorldManager worldManager;
        private MainUIController uiController;
        private LandDatabase landDatabase;
        
        // Test state
        private float lastTestTime;
        private int testIndex = 0;
        private LandType[] allLands;
        
        private void Start()
        {
            InitializeReferences();
            
            if (enableTestMode)
            {
                Debug.Log("🧪 Game Tester initialized - Press T to run tests, R to discover all lands");
                LogSystemStatus();
            }
        }
        
        private void Update()
        {
            if (!enableTestMode) return;
            
            HandleTestInput();
            
            if (showDebugInfo)
            {
                ShowDebugInfo();
            }
        }
        
        /// <summary>
        /// Initialize component references
        /// </summary>
        private void InitializeReferences()
        {
            gameManager = FindFirstObjectByType<GameManager>();
            navigationController = FindFirstObjectByType<NavigationController>();
            worldManager = FindFirstObjectByType<WorldManager>();
            uiController = FindFirstObjectByType<MainUIController>();
            landDatabase = FindFirstObjectByType<LandDatabase>();
            
            if (worldManager != null)
            {
                allLands = worldManager.GetAllLands();
            }
        }
        
        /// <summary>
        /// Handle test input
        /// </summary>
        private void HandleTestInput()
        {
            // Run test cycle
            if (Input.GetKeyDown(testKey))
            {
                RunTestCycle();
            }
            
            // Discover all lands
            if (Input.GetKeyDown(discoverAllKey))
            {
                DiscoverAllLands();
            }
            
            // Auto-test mode
            if (Time.time - lastTestTime > testInterval)
            {
                lastTestTime = Time.time;
                // AutoTest(); // Uncomment to enable auto-testing
            }
        }
        
        /// <summary>
        /// Run a complete test cycle
        /// </summary>
        [ContextMenu("Run Test Cycle")]
        public void RunTestCycle()
        {
            Debug.Log("🧪 Starting test cycle...");
            
            if (testNavigation)
                TestNavigation();
                
            if (testLandDiscovery)
                TestLandDiscovery();
                
            if (testUI)
                TestUI();
            
            Debug.Log("✅ Test cycle complete!");
        }
        
        /// <summary>
        /// Test navigation system
        /// </summary>
        private void TestNavigation()
        {
            Debug.Log("🧭 Testing navigation system...");
            
            if (navigationController == null)
            {
                Debug.LogError("❌ NavigationController not found!");
                return;
            }
            
            if (allLands == null || allLands.Length == 0)
            {
                Debug.LogError("❌ No lands available for navigation test!");
                return;
            }
            
            // Find a discovered land to navigate to
            LandType targetLand = null;
            foreach (LandType land in allLands)
            {
                if (land != null && land.IsDiscovered && land != navigationController.CurrentLand)
                {
                    targetLand = land;
                    break;
                }
            }
            
            if (targetLand != null)
            {
                Debug.Log($"🎯 Navigating to {targetLand.Data.landName}");
                navigationController.NavigateToLand(targetLand);
            }
            else
            {
                Debug.LogWarning("⚠️ No suitable target land found for navigation test");
            }
        }
        
        /// <summary>
        /// Test land discovery system
        /// </summary>
        private void TestLandDiscovery()
        {
            Debug.Log("🔍 Testing land discovery system...");
            
            if (allLands == null || allLands.Length == 0)
            {
                Debug.LogError("❌ No lands available for discovery test!");
                return;
            }
            
            // Find an undiscovered land
            LandType undiscoveredLand = null;
            foreach (LandType land in allLands)
            {
                if (land != null && !land.IsDiscovered)
                {
                    undiscoveredLand = land;
                    break;
                }
            }
            
            if (undiscoveredLand != null)
            {
                Debug.Log($"🌟 Discovering {undiscoveredLand.Data.landName}");
                undiscoveredLand.DiscoverLand();
            }
            else
            {
                Debug.Log("🎉 All lands already discovered!");
            }
        }
        
        /// <summary>
        /// Test UI system
        /// </summary>
        private void TestUI()
        {
            Debug.Log("🖥️ Testing UI system...");
            
            if (uiController == null)
            {
                Debug.LogError("❌ MainUIController not found!");
                return;
            }
            
            // Test showing land info for current land
            if (navigationController != null && navigationController.CurrentLand != null)
            {
                Debug.Log($"📋 Showing info for {navigationController.CurrentLand.Data.landName}");
                uiController.ShowLandInfo(navigationController.CurrentLand);
            }
            else
            {
                Debug.LogWarning("⚠️ No current land available for UI test");
            }
        }
        
        /// <summary>
        /// Discover all lands (cheat for testing)
        /// </summary>
        [ContextMenu("Discover All Lands")]
        public void DiscoverAllLands()
        {
            Debug.Log("🌍 Discovering all lands...");
            
            if (allLands == null || allLands.Length == 0)
            {
                Debug.LogError("❌ No lands available to discover!");
                return;
            }
            
            int discoveredCount = 0;
            foreach (LandType land in allLands)
            {
                if (land != null && !land.IsDiscovered)
                {
                    land.DiscoverLand();
                    discoveredCount++;
                }
            }
            
            Debug.Log($"✨ Discovered {discoveredCount} new lands! Total: {allLands.Length}");
        }
        
        /// <summary>
        /// Log system status
        /// </summary>
        private void LogSystemStatus()
        {
            Debug.Log("📊 System Status:");
            Debug.Log($"   GameManager: {(gameManager != null ? "✅" : "❌")}");
            Debug.Log($"   NavigationController: {(navigationController != null ? "✅" : "❌")}");
            Debug.Log($"   WorldManager: {(worldManager != null ? "✅" : "❌")}");
            Debug.Log($"   UIController: {(uiController != null ? "✅" : "❌")}");
            Debug.Log($"   Total Lands: {(allLands != null ? allLands.Length : 0)}");
            
            if (allLands != null)
            {
                int discoveredCount = 0;
                foreach (LandType land in allLands)
                {
                    if (land != null && land.IsDiscovered)
                        discoveredCount++;
                }
                Debug.Log($"   Discovered Lands: {discoveredCount}/{allLands.Length}");
            }
        }
        
        /// <summary>
        /// Show debug info on screen
        /// </summary>
        private void ShowDebugInfo()
        {
            // This would be better with Unity's OnGUI, but keeping it simple
            if (allLands != null && navigationController != null)
            {
                int discoveredCount = 0;
                foreach (LandType land in allLands)
                {
                    if (land != null && land.IsDiscovered)
                        discoveredCount++;
                }
                
                // You could display this with UI Text or OnGUI here
            }
        }
        
        /// <summary>
        /// Auto test function
        /// </summary>
        private void AutoTest()
        {
            testIndex = (testIndex + 1) % 3;
            
            switch (testIndex)
            {
                case 0:
                    TestNavigation();
                    break;
                case 1:
                    TestLandDiscovery();
                    break;
                case 2:
                    TestUI();
                    break;
            }
        }
        
        /// <summary>
        /// Generate test report
        /// </summary>
        [ContextMenu("Generate Test Report")]
        public void GenerateTestReport()
        {
            Debug.Log("📄 WORLD NAVIGATOR TEST REPORT");
            Debug.Log("================================");
            
            // System status
            Debug.Log("🔧 SYSTEM STATUS:");
            LogSystemStatus();
            
            // Land statistics
            if (allLands != null)
            {
                Debug.Log("\\n🌍 LAND STATISTICS:");
                
                System.Collections.Generic.Dictionary<LandCategory, int> categoryCount = 
                    new System.Collections.Generic.Dictionary<LandCategory, int>();
                    
                foreach (LandType land in allLands)
                {
                    if (land != null)
                    {
                        LandCategory category = land.Data.category;
                        if (!categoryCount.ContainsKey(category))
                            categoryCount[category] = 0;
                        categoryCount[category]++;
                    }
                }
                
                foreach (var kvp in categoryCount)
                {
                    Debug.Log($"   {kvp.Key}: {kvp.Value} lands");
                }
            }
            
            // Performance info
            Debug.Log("\\n⚡ PERFORMANCE:");
            Debug.Log($"   FPS: {1f / Time.deltaTime:F1}");
            Debug.Log($"   Frame Time: {Time.deltaTime * 1000f:F1}ms");
            
            Debug.Log("\\n✅ Report complete!");
        }
        
        /// <summary>
        /// Reset game state for testing
        /// </summary>
        [ContextMenu("Reset Game State")]
        public void ResetGameState()
        {
            Debug.Log("🔄 Resetting game state...");
            
            // Hide all lands
            if (allLands != null)
            {
                foreach (LandType land in allLands)
                {
                    if (land != null)
                    {
                        // Reset discovery state based on rarity
                        land.Data.isDiscovered = land.Data.rarity <= 1;
                    }
                }
            }
            
            // Reset camera position
            if (navigationController != null)
            {
                // Find starting land
                LandType startLand = null;
                if (allLands != null)
                {
                    foreach (LandType land in allLands)
                    {
                        if (land != null && land.Data.rarity <= 2)
                        {
                            startLand = land;
                            break;
                        }
                    }
                }
                
                if (startLand != null)
                {
                    navigationController.SetStartingLand(startLand);
                }
            }
            
            Debug.Log("✅ Game state reset complete!");
        }
    }
}