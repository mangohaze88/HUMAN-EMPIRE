using UnityEngine;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Database containing all land type configurations
    /// ScriptableObject that can be configured in Unity Editor
    /// </summary>
    [CreateAssetMenu(fileName = "LandDatabase", menuName = "World Navigator/Land Database")]
    public class LandDatabase : ScriptableObject
    {
        [Header("All Land Types")]
        public LandData[] allLands;
        
        /// <summary>
        /// Get land data by name
        /// </summary>
        public LandData GetLandByName(string landName)
        {
            foreach (LandData land in allLands)
            {
                if (land.landName.Equals(landName, System.StringComparison.OrdinalIgnoreCase))
                    return land;
            }
            return null;
        }
        
        /// <summary>
        /// Get all lands of a specific category
        /// </summary>
        public LandData[] GetLandsByCategory(LandCategory category)
        {
            System.Collections.Generic.List<LandData> categoryLands = 
                new System.Collections.Generic.List<LandData>();
                
            foreach (LandData land in allLands)
            {
                if (land.category == category)
                    categoryLands.Add(land);
            }
            
            return categoryLands.ToArray();
        }
        
        /// <summary>
        /// Initialize default land data (called in editor)
        /// </summary>
        [ContextMenu("Initialize Default Lands")]
        public void InitializeDefaultLands()
        {
            allLands = new LandData[]
            {
                // TEMPERATE LANDS
                CreateLandData("Forest", LandCategory.Temperate, 1, new Color(0.2f, 0.7f, 0.2f),
                    "Dense woodlands filled with towering trees and rich wildlife.",
                    new string[] { "Abundant wildlife", "Fresh oxygen", "Natural shelter", "Moderate climate" },
                    new string[] { "Timber", "Medicinal plants", "Game animals", "Fresh water" }),
                
                CreateLandData("Grassland", LandCategory.Temperate, 1, new Color(0.4f, 0.8f, 0.3f),
                    "Vast open plains with rolling hills perfect for agriculture.",
                    new string[] { "Fertile soil", "Open spaces", "Seasonal beauty", "Wind currents" },
                    new string[] { "Grain crops", "Livestock", "Wind energy", "Wild herbs" }),
                
                CreateLandData("River Valley", LandCategory.Temperate, 2, new Color(0.3f, 0.6f, 0.9f),
                    "Lush valleys carved by flowing rivers, supporting diverse ecosystems.",
                    new string[] { "Fresh water source", "Fertile floodplains", "Natural highways", "Rich biodiversity" },
                    new string[] { "Fresh water", "River fish", "Fertile soil", "Transportation routes" }),
                
                // ARID LANDS
                CreateLandData("Desert", LandCategory.Arid, 2, new Color(0.9f, 0.7f, 0.3f),
                    "Vast sandy expanses with extreme temperatures and rare life.",
                    new string[] { "Extreme temperatures", "Minimal rainfall", "Endless horizons", "Harsh survival conditions" },
                    new string[] { "Rare minerals", "Solar energy potential", "Salt deposits", "Cactus fruits" }),
                
                CreateLandData("Badlands", LandCategory.Arid, 3, new Color(0.8f, 0.5f, 0.3f),
                    "Eroded rock formations revealing ancient geological history.",
                    new string[] { "Unique geology", "Fossil deposits", "Striking formations", "Archaeological significance" },
                    new string[] { "Fossils", "Rare metals", "Clay deposits", "Gemstones" }),
                
                CreateLandData("Oasis", LandCategory.Arid, 4, new Color(0.2f, 0.8f, 0.6f),
                    "Life-giving water sources surrounded by palm trees in the desert.",
                    new string[] { "Desert sanctuary", "Life-sustaining water", "Trade hub importance", "Natural refuge" },
                    new string[] { "Fresh water", "Date palms", "Shelter", "Trade goods" }),
                
                // COLD LANDS
                CreateLandData("Arctic Tundra", LandCategory.Cold, 3, new Color(0.8f, 0.9f, 1f),
                    "Frozen landscapes with permafrost and unique polar wildlife.",
                    new string[] { "Permafrost ground", "Extreme cold", "Aurora displays", "Polar adaptations" },
                    new string[] { "Ice", "Polar animals", "Rare lichens", "Pure air" }),
                
                CreateLandData("Taiga", LandCategory.Cold, 2, new Color(0.2f, 0.5f, 0.3f),
                    "Vast coniferous forests adapted to cold climates.",
                    new string[] { "Snow-covered forests", "Cold adaptation", "Evergreen beauty", "Wildlife corridors" },
                    new string[] { "Softwood timber", "Fur-bearing animals", "Pine nuts", "Medicinal bark" }),
                
                CreateLandData("Glacier", LandCategory.Cold, 4, new Color(0.7f, 0.9f, 1f),
                    "Massive rivers of ice slowly carving through the landscape.",
                    new string[] { "Ancient ice formations", "Slow movement", "Pure frozen water", "Climate indicators" },
                    new string[] { "Pure ice water", "Unique ice formations", "Pristine environment", "Research value" }),
                
                // MOUNTAIN LANDS
                CreateLandData("Alpine", LandCategory.Mountain, 3, new Color(0.6f, 0.8f, 0.9f),
                    "High altitude meadows with snow-capped peaks and rare flora.",
                    new string[] { "High altitude", "Thin air", "Spectacular views", "Rare alpine plants" },
                    new string[] { "Mountain herbs", "Pure mountain air", "Rare minerals", "Adventure tourism" }),
                
                CreateLandData("Rocky Highlands", LandCategory.Mountain, 2, new Color(0.6f, 0.6f, 0.7f),
                    "Stone plateaus and cliff faces offering natural fortification.",
                    new string[] { "Natural fortresses", "Panoramic views", "Stone formations", "Defensive positions" },
                    new string[] { "Building stone", "Rare metals", "Defensive advantage", "Quarry materials" }),
                
                CreateLandData("Canyon", LandCategory.Mountain, 3, new Color(0.8f, 0.4f, 0.2f),
                    "Deep valleys carved by ancient rivers, revealing Earth's history.",
                    new string[] { "Geological layers", "Echo chambers", "Hidden springs", "Ancient history" },
                    new string[] { "Geological specimens", "Hidden water", "Natural acoustics", "Archaeological sites" }),
                
                // WATER LANDS
                CreateLandData("Ocean", LandCategory.Water, 1, new Color(0.1f, 0.4f, 0.8f),
                    "Vast saltwater expanses teeming with marine life and weather systems.",
                    new string[] { "Endless horizons", "Tidal forces", "Weather generation", "Marine ecosystems" },
                    new string[] { "Seafood", "Salt", "Seaweed", "Trade routes" }),
                
                CreateLandData("Lake", LandCategory.Water, 2, new Color(0.3f, 0.6f, 0.9f),
                    "Large freshwater bodies providing recreation and resources.",
                    new string[] { "Calm waters", "Recreational value", "Stable ecosystem", "Fresh water source" },
                    new string[] { "Fresh water", "Lake fish", "Recreation", "Irrigation water" }),
                
                CreateLandData("Wetland", LandCategory.Water, 3, new Color(0.4f, 0.7f, 0.5f),
                    "Waterlogged ecosystems with incredible biodiversity and natural filtering.",
                    new string[] { "High biodiversity", "Natural filtration", "Flood control", "Wildlife habitat" },
                    new string[] { "Wetland plants", "Water birds", "Natural filters", "Fertile mud" }),
                
                CreateLandData("Coastal", LandCategory.Water, 2, new Color(0.8f, 0.8f, 0.6f),
                    "Where land meets sea, creating unique tidal ecosystems and harbors.",
                    new string[] { "Tidal influences", "Natural harbors", "Beach formations", "Marine-land interface" },
                    new string[] { "Seafood", "Sand", "Shells", "Harbor access" }),
                
                // VOLCANIC LANDS
                CreateLandData("Active Volcano", LandCategory.Volcanic, 5, new Color(1f, 0.3f, 0.1f),
                    "Currently erupting mountains spewing lava and creating new land.",
                    new string[] { "Active eruptions", "Lava flows", "Geothermal activity", "Land creation" },
                    new string[] { "Geothermal energy", "Volcanic glass", "Fertile ash", "Rare minerals" }),
                
                CreateLandData("Dormant Volcano", LandCategory.Volcanic, 3, new Color(0.6f, 0.3f, 0.2f),
                    "Sleeping giants with incredibly fertile soil and hot springs.",
                    new string[] { "Dormant but potentially active", "Extremely fertile soil", "Hot springs", "Unique geology" },
                    new string[] { "Super fertile soil", "Hot spring minerals", "Volcanic rock", "Geothermal potential" }),
                
                // SPECIAL LANDS
                CreateLandData("Ancient Ruins", LandCategory.Special, 4, new Color(0.7f, 0.6f, 0.4f),
                    "Mysterious remains of ancient civilizations holding secrets of the past.",
                    new string[] { "Historical mysteries", "Archaeological treasures", "Ancient knowledge", "Cultural heritage" },
                    new string[] { "Artifacts", "Ancient knowledge", "Cultural insights", "Tourism value" }),
                
                CreateLandData("Crystal Caverns", LandCategory.Special, 5, new Color(0.8f, 0.4f, 0.9f),
                    "Underground wonderlands filled with luminescent crystals and unique acoustics.",
                    new string[] { "Luminescent crystals", "Unique acoustics", "Underground beauty", "Natural light shows" },
                    new string[] { "Rare crystals", "Acoustic properties", "Natural light", "Spiritual energy" }),
                
                CreateLandData("Floating Islands", LandCategory.Special, 5, new Color(0.6f, 0.8f, 1f),
                    "Mystical landmasses defying gravity with unique sky-based ecosystems.",
                    new string[] { "Defies gravity", "Sky ecosystems", "Aerial navigation", "Mystical properties" },
                    new string[] { "Floating minerals", "Sky plants", "Aerial perspective", "Magical essence" }),
                
                CreateLandData("Thermal Springs", LandCategory.Special, 3, new Color(0.4f, 0.8f, 0.7f),
                    "Natural hot water sources with healing properties and unique minerals.",
                    new string[] { "Healing waters", "Relaxation", "Unique minerals", "Natural spa" },
                    new string[] { "Healing minerals", "Therapeutic waters", "Relaxation", "Natural remedies" })
            };
        }
        
        /// <summary>
        /// Helper method to create land data
        /// </summary>
        private LandData CreateLandData(string name, LandCategory category, int rarity, 
            Color color, string description, string[] characteristics, string[] resources)
        {
            LandData land = new LandData();
            land.landName = name;
            land.category = category;
            land.rarity = rarity;
            land.primaryColor = color;
            land.description = description;
            land.characteristics = characteristics;
            land.resources = resources;
            land.isDiscovered = rarity <= 1; // Only common lands start discovered
            
            return land;
        }
    }
}