using UnityEngine;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Data structure defining properties of different land types
    /// </summary>
    [System.Serializable]
    public class LandData
    {
        [Header("Basic Information")]
        public string landName;
        public LandCategory category;
        
        [Header("Characteristics")]
        [TextArea(3, 5)]
        public string description;
        public string[] characteristics;
        public string[] resources;
        
        [Header("Visual Properties")]
        public Color primaryColor = Color.white;
        public Material landMaterial;
        public GameObject landPrefab;
        
        [Header("Audio")]
        public AudioClip ambientSound;
        public AudioClip clickSound;
        
        [Header("Gameplay")]
        [Range(1, 5)]
        public int rarity = 1; // How rare this land type is
        public bool isDiscovered = false;
        public Vector3 worldPosition;
    }
    
    /// <summary>
    /// Categories for organizing different land types
    /// </summary>
    public enum LandCategory
    {
        Temperate,
        Arid,
        Cold,
        Mountain,
        Water,
        Volcanic,
        Special
    }
}