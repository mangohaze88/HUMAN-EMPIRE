using UnityEngine;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Quick start script - attach this to any GameObject in your scene to see the world!
    /// This is the easiest way to get started with World Navigator.
    /// </summary>
    public class QuickStart : MonoBehaviour
    {
        private void Start()
        {
            Debug.Log("🚀 QuickStart: Launching World Navigator...");

            // Check if GameInitializer already exists
            GameInitializer existingInitializer = FindFirstObjectByType<GameInitializer>();
            if (existingInitializer == null)
            {
                // Create GameInitializer
                GameObject initializerObject = new GameObject("🌍 Game Initializer");
                GameInitializer initializer = initializerObject.AddComponent<GameInitializer>();
                Debug.Log("✅ GameInitializer created!");
            }
            else
            {
                Debug.Log("✅ GameInitializer already exists!");
            }

            Debug.Log("🌟 World Navigator is ready! Wait a moment for world generation...");
        }
    }
}
