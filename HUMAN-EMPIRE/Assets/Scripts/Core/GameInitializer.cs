using UnityEngine;
using WorldNavigator.Visual;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Automatically initializes the game world when scene starts
    /// </summary>
    public class GameInitializer : MonoBehaviour
    {
        [Header("Auto Setup")]
        [SerializeField] private bool autoInitialize = true;
        
        private void Awake()
        {
            if (autoInitialize)
            {
                InitializeGame();
            }
        }
        
        /// <summary>
        /// Initialize the complete game
        /// </summary>
        public void InitializeGame()
        {
            Debug.Log("🌍 Initializing World Navigator...");
            
            // Create visual world generator
            CreateVisualWorldGenerator();
            
            Debug.Log("✅ World Navigator initialized!");
        }
        
        /// <summary>
        /// Create and setup visual world generator
        /// </summary>
        private void CreateVisualWorldGenerator()
        {
            // Check if already exists
            VisualWorldGenerator existing = FindFirstObjectByType<VisualWorldGenerator>();
            if (existing != null)
            {
                Debug.Log("Visual World Generator already exists");
                return;
            }
            
            // Create generator object
            GameObject generatorObject = new GameObject("Visual World Generator");
            VisualWorldGenerator generator = generatorObject.AddComponent<VisualWorldGenerator>();
            
            Debug.Log("Created Visual World Generator");
        }
        
        /// <summary>
        /// Manual initialization button for testing
        /// </summary>
        [ContextMenu("Initialize Game")]
        public void ManualInitialize()
        {
            InitializeGame();
        }
    }
}