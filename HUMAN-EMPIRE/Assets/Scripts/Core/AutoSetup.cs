using UnityEngine;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Runtime attribute to automatically create GameInitializer
    /// </summary>
    [System.Serializable]
    public class AutoSetup
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Initialize()
        {
            // Check if GameInitializer already exists
            GameInitializer existing = Object.FindFirstObjectByType<GameInitializer>();
            if (existing == null)
            {
                // Create GameInitializer
                GameObject initializerObject = new GameObject("🌍 Game Initializer");
                initializerObject.AddComponent<GameInitializer>();
                
                Debug.Log("🚀 Auto-created Game Initializer");
            }
        }
    }
}