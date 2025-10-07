using UnityEngine;
using WorldNavigator.Lands;

namespace WorldNavigator.Core
{
    /// <summary>
    /// Generates Unity prefabs for all land types
    /// Can be run in editor to create all necessary prefabs
    /// </summary>
    public class LandPrefabGenerator : MonoBehaviour
    {
        [Header("Prefab Generation")]
        [SerializeField] private LandDatabase landDatabase;
        [SerializeField] private bool generateOnStart = false;
        
        [Header("Prefab Settings")]
        [SerializeField] private float landSize = 5f;
        [SerializeField] private float landHeight = 0.5f;
        [SerializeField] private Material baseMaterial;
        
        private void Start()
        {
            if (generateOnStart)
                GenerateAllLandPrefabs();
        }
        
        /// <summary>
        /// Generate all land prefabs from database
        /// </summary>
        [ContextMenu("Generate All Land Prefabs")]
        public void GenerateAllLandPrefabs()
        {
            if (landDatabase == null)
            {
                Debug.LogError("Land Database not assigned!");
                return;
            }
            
            foreach (LandData landData in landDatabase.allLands)
            {
                CreateLandPrefab(landData);
            }
            
            Debug.Log($"Generated {landDatabase.allLands.Length} land prefabs!");
        }
        
        /// <summary>
        /// Create a prefab for a specific land type
        /// </summary>
        private GameObject CreateLandPrefab(LandData landData)
        {
            // Create main land object
            GameObject landObject = new GameObject($"Land_{landData.landName}");
            
            // Add LandType component
            LandType landType = landObject.AddComponent<LandType>();
            
            // Create visual mesh
            CreateLandMesh(landObject, landData);
            
            // Add collider for interaction
            CreateLandCollider(landObject, landData);
            
            // Add particle system
            CreateParticleSystem(landObject, landData);
            
            // Add light
            CreateLandLight(landObject, landData);
            
            // Add audio source
            CreateAudioSource(landObject, landData);
            
            // Position in world based on category
            PositionLand(landObject, landData);
            
            return landObject;
        }
        
        /// <summary>
        /// Create the visual mesh for the land
        /// </summary>
        private void CreateLandMesh(GameObject landObject, LandData landData)
        {
            // Create base plane mesh
            MeshFilter meshFilter = landObject.AddComponent<MeshFilter>();
            MeshRenderer meshRenderer = landObject.AddComponent<MeshRenderer>();
            
            // Generate procedural mesh based on land type
            Mesh landMesh = GenerateLandMesh(landData);
            meshFilter.mesh = landMesh;
            
            // Create and assign material
            Material landMaterial = CreateLandMaterial(landData);
            meshRenderer.material = landMaterial;
            
            // Store material reference in land data
            landData.landMaterial = landMaterial;
        }
        
        /// <summary>
        /// Generate unique mesh for each land type
        /// </summary>
        private Mesh GenerateLandMesh(LandData landData)
        {
            Mesh mesh = new Mesh();
            mesh.name = $"{landData.landName}_Mesh";
            
            // Base size and resolution
            int resolution = GetMeshResolution(landData);
            Vector3[] vertices = new Vector3[(resolution + 1) * (resolution + 1)];
            Vector2[] uvs = new Vector2[vertices.Length];
            Color[] colors = new Color[vertices.Length];
            
            // Generate vertices with unique patterns per land type
            int vertIndex = 0;
            for (int z = 0; z <= resolution; z++)
            {
                for (int x = 0; x <= resolution; x++)
                {
                    float xPos = (x - resolution * 0.5f) * landSize / resolution;
                    float zPos = (z - resolution * 0.5f) * landSize / resolution;
                    float yPos = GetHeightForLandType(landData, xPos, zPos);
                    
                    vertices[vertIndex] = new Vector3(xPos, yPos, zPos);
                    uvs[vertIndex] = new Vector2((float)x / resolution, (float)z / resolution);
                    colors[vertIndex] = GetVertexColor(landData, xPos, zPos);
                    
                    vertIndex++;
                }
            }
            
            // Generate triangles
            int[] triangles = new int[resolution * resolution * 6];
            int triIndex = 0;
            
            for (int z = 0; z < resolution; z++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    int i = z * (resolution + 1) + x;
                    
                    // First triangle
                    triangles[triIndex] = i;
                    triangles[triIndex + 1] = i + resolution + 1;
                    triangles[triIndex + 2] = i + 1;
                    
                    // Second triangle
                    triangles[triIndex + 3] = i + 1;
                    triangles[triIndex + 4] = i + resolution + 1;
                    triangles[triIndex + 5] = i + resolution + 2;
                    
                    triIndex += 6;
                }
            }
            
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.uv = uvs;
            mesh.colors = colors;
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            
            return mesh;
        }
        
        /// <summary>
        /// Get mesh resolution based on land complexity
        /// </summary>
        private int GetMeshResolution(LandData landData)
        {
            switch (landData.category)
            {
                case LandCategory.Mountain:
                case LandCategory.Volcanic:
                    return 32; // Higher detail for mountains
                case LandCategory.Water:
                    return 24; // Medium detail for water
                default:
                    return 16; // Standard detail
            }
        }
        
        /// <summary>
        /// Get height variation for different land types
        /// </summary>
        private float GetHeightForLandType(LandData landData, float x, float z)
        {
            float baseHeight = 0f;
            float noise = Mathf.PerlinNoise(x * 0.1f, z * 0.1f);
            
            switch (landData.category)
            {
                case LandCategory.Mountain:
                    baseHeight = noise * landHeight * 3f + landHeight;
                    break;
                case LandCategory.Volcanic:
                    // Cone shape for volcanoes
                    float distance = Mathf.Sqrt(x * x + z * z);
                    baseHeight = Mathf.Max(0, (landSize * 0.3f - distance) * 0.5f + noise * landHeight);
                    break;
                case LandCategory.Water:
                    baseHeight = -landHeight * 0.5f + noise * 0.2f;
                    break;
                case LandCategory.Special:
                    // Unique formations for special lands
                    baseHeight = Mathf.Sin(x * 0.5f) * Mathf.Cos(z * 0.5f) * landHeight + noise * landHeight;
                    break;
                default:
                    baseHeight = noise * landHeight;
                    break;
            }
            
            return baseHeight;
        }
        
        /// <summary>
        /// Get vertex color variation
        /// </summary>
        private Color GetVertexColor(LandData landData, float x, float z)
        {
            Color baseColor = landData.primaryColor;
            float noise = Mathf.PerlinNoise(x * 0.2f, z * 0.2f);
            
            // Add slight color variation
            baseColor.r += (noise - 0.5f) * 0.2f;
            baseColor.g += (noise - 0.5f) * 0.2f;
            baseColor.b += (noise - 0.5f) * 0.2f;
            
            return baseColor;
        }
        
        /// <summary>
        /// Create material for land type
        /// </summary>
        private Material CreateLandMaterial(LandData landData)
        {
            Material material;
            
            if (baseMaterial != null)
            {
                material = new Material(baseMaterial);
            }
            else
            {
                material = new Material(Shader.Find("Standard"));
            }
            
            material.name = $"{landData.landName}_Material";
            material.color = landData.primaryColor;
            
            // Set material properties based on land type
            switch (landData.category)
            {
                case LandCategory.Water:
                    material.SetFloat("_Metallic", 0.8f);
                    material.SetFloat("_Smoothness", 0.9f);
                    break;
                case LandCategory.Volcanic:
                    material.SetFloat("_Metallic", 0.3f);
                    material.SetFloat("_Smoothness", 0.1f);
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", landData.primaryColor * 0.3f);
                    break;
                case LandCategory.Special:
                    material.SetFloat("_Metallic", 0.5f);
                    material.SetFloat("_Smoothness", 0.7f);
                    material.EnableKeyword("_EMISSION");
                    material.SetColor("_EmissionColor", landData.primaryColor * 0.2f);
                    break;
                default:
                    material.SetFloat("_Metallic", 0.1f);
                    material.SetFloat("_Smoothness", 0.3f);
                    break;
            }
            
            return material;
        }
        
        /// <summary>
        /// Create collider for interaction
        /// </summary>
        private void CreateLandCollider(GameObject landObject, LandData landData)
        {
            MeshCollider collider = landObject.AddComponent<MeshCollider>();
            collider.sharedMesh = landObject.GetComponent<MeshFilter>().mesh;
            collider.convex = true; // Allow for triggers if needed
        }
        
        /// <summary>
        /// Create particle system for atmospheric effects
        /// </summary>
        private void CreateParticleSystem(GameObject landObject, LandData landData)
        {
            GameObject particleObject = new GameObject("ParticleEffects");
            particleObject.transform.SetParent(landObject.transform);
            particleObject.transform.localPosition = Vector3.up * 2f;
            
            ParticleSystem particles = particleObject.AddComponent<ParticleSystem>();
            var main = particles.main;
            var emission = particles.emission;
            var shape = particles.shape;
            
            // Configure particles based on land type
            switch (landData.category)
            {
                case LandCategory.Volcanic:
                    main.startColor = Color.red;
                    main.startSize = 0.2f;
                    emission.rateOverTime = 20f;
                    break;
                case LandCategory.Water:
                    main.startColor = Color.cyan;
                    main.startSize = 0.1f;
                    emission.rateOverTime = 10f;
                    break;
                case LandCategory.Special:
                    main.startColor = landData.primaryColor;
                    main.startSize = 0.15f;
                    emission.rateOverTime = 15f;
                    break;
                default:
                    main.startColor = new Color(1f, 1f, 1f, 0.3f);
                    main.startSize = 0.05f;
                    emission.rateOverTime = 5f;
                    break;
            }
            
            shape.shapeType = ParticleSystemShapeType.Box;
            shape.scale = Vector3.one * landSize * 0.8f;
        }
        
        /// <summary>
        /// Create light for land atmosphere
        /// </summary>
        private void CreateLandLight(GameObject landObject, LandData landData)
        {
            GameObject lightObject = new GameObject("LandLight");
            lightObject.transform.SetParent(landObject.transform);
            lightObject.transform.localPosition = Vector3.up * 3f;
            
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Point;
            light.color = landData.primaryColor;
            light.intensity = GetLightIntensity(landData);
            light.range = landSize * 1.5f;
        }
        
        /// <summary>
        /// Get light intensity based on land type
        /// </summary>
        private float GetLightIntensity(LandData landData)
        {
            switch (landData.category)
            {
                case LandCategory.Volcanic:
                    return 2f;
                case LandCategory.Special:
                    return 1.5f;
                default:
                    return 0.8f;
            }
        }
        
        /// <summary>
        /// Create audio source for ambient sounds
        /// </summary>
        private void CreateAudioSource(GameObject landObject, LandData landData)
        {
            AudioSource audioSource = landObject.AddComponent<AudioSource>();
            audioSource.loop = true;
            audioSource.volume = 0.3f;
            audioSource.spatialBlend = 0.8f; // 3D sound
            audioSource.maxDistance = landSize * 2f;
            audioSource.playOnAwake = false;
            
            // Assign audio clip (would be set from database)
            // audioSource.clip = landData.ambientSound;
        }
        
        /// <summary>
        /// Position land in world based on category
        /// </summary>
        private void PositionLand(GameObject landObject, LandData landData)
        {
            Vector3 position = GetPositionForCategory(landData.category, landData);
            landObject.transform.position = position;
            landData.worldPosition = position;
        }
        
        /// <summary>
        /// Get position based on land category
        /// </summary>
        private Vector3 GetPositionForCategory(LandCategory category, LandData landData)
        {
            float spacing = landSize * 1.5f;
            Vector3 basePosition = Vector3.zero;
            
            // Arrange lands in a circular pattern by category
            int categoryIndex = (int)category;
            int landIndex = System.Array.IndexOf(landDatabase.GetLandsByCategory(category), landData);
            
            float categoryAngle = categoryIndex * (360f / 7f) * Mathf.Deg2Rad; // 7 categories
            float landAngle = landIndex * (60f * Mathf.Deg2Rad); // Spread lands within category
            
            float categoryRadius = 20f;
            float landRadius = 8f;
            
            // Category position
            Vector3 categoryPos = new Vector3(
                Mathf.Cos(categoryAngle) * categoryRadius,
                0,
                Mathf.Sin(categoryAngle) * categoryRadius
            );
            
            // Land position within category
            Vector3 landPos = new Vector3(
                Mathf.Cos(landAngle) * landRadius,
                0,
                Mathf.Sin(landAngle) * landRadius
            );
            
            return categoryPos + landPos;
        }
    }
}