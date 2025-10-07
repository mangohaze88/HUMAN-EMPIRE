using UnityEngine;
using System.Collections.Generic;

namespace WorldNavigator.Environment
{
    /// <summary>
    /// Creates procedural environmental assets like trees, water, and terrain details
    /// </summary>
    public class ProceduralEnvironment : MonoBehaviour
    {
        [Header("Tree Generation")]
        [SerializeField] private int treesPerLand = 5;
        [SerializeField] private float treeScaleMin = 0.8f;
        [SerializeField] private float treeScaleMax = 1.5f;
        
        [Header("Water Settings")]
        [SerializeField] private Material waterMaterial;
        [SerializeField] private float waveHeight = 0.2f;
        [SerializeField] private float waveSpeed = 1f;
        
        [Header("Terrain Details")]
        [SerializeField] private int rocksPerLand = 3;
        [SerializeField] private int grassPatchesPerLand = 8;
        
        /// <summary>
        /// Add environment details to a land object
        /// </summary>
        public void EnhanceLandWithEnvironment(GameObject landObject, WorldNavigator.Core.LandData landData)
        {
            switch (landData.category)
            {
                case WorldNavigator.Core.LandCategory.Temperate:
                    AddTemperateEnvironment(landObject);
                    break;
                case WorldNavigator.Core.LandCategory.Water:
                    AddWaterEnvironment(landObject);
                    break;
                case WorldNavigator.Core.LandCategory.Mountain:
                    AddMountainEnvironment(landObject);
                    break;
                case WorldNavigator.Core.LandCategory.Cold:
                    AddColdEnvironment(landObject);
                    break;
                case WorldNavigator.Core.LandCategory.Arid:
                    AddAridEnvironment(landObject);
                    break;
                case WorldNavigator.Core.LandCategory.Volcanic:
                    AddVolcanicEnvironment(landObject);
                    break;
                case WorldNavigator.Core.LandCategory.Special:
                    AddSpecialEnvironment(landObject);
                    break;
            }
        }
        
        /// <summary>
        /// Add temperate environment (trees, grass)
        /// </summary>
        private void AddTemperateEnvironment(GameObject landObject)
        {
            // Add trees
            for (int i = 0; i < treesPerLand; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 3f);
                GameObject tree = CreateProceduralTree(TreeType.Oak);
                tree.transform.SetParent(landObject.transform);
                tree.transform.position = position;
                tree.transform.localScale = Vector3.one * Random.Range(treeScaleMin, treeScaleMax);
            }
            
            // Add grass patches
            for (int i = 0; i < grassPatchesPerLand; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject grass = CreateGrassPatch();
                grass.transform.SetParent(landObject.transform);
                grass.transform.position = position;
            }
        }
        
        /// <summary>
        /// Add water environment
        /// </summary>
        private void AddWaterEnvironment(GameObject landObject)
        {
            // Create animated water surface
            GameObject waterSurface = CreateAnimatedWater();
            waterSurface.transform.SetParent(landObject.transform);
            waterSurface.transform.localPosition = Vector3.up * 0.1f;
            
            // Add some aquatic plants
            for (int i = 0; i < 3; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject seaweed = CreateSeaweed();
                seaweed.transform.SetParent(landObject.transform);
                seaweed.transform.position = position;
            }
        }
        
        /// <summary>
        /// Add mountain environment
        /// </summary>
        private void AddMountainEnvironment(GameObject landObject)
        {
            // Add rocks
            for (int i = 0; i < rocksPerLand; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject rock = CreateRock();
                rock.transform.SetParent(landObject.transform);
                rock.transform.position = position;
            }
            
            // Add some pine trees
            for (int i = 0; i < 2; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 3f);
                GameObject tree = CreateProceduralTree(TreeType.Pine);
                tree.transform.SetParent(landObject.transform);
                tree.transform.position = position;
            }
        }
        
        /// <summary>
        /// Add cold environment
        /// </summary>
        private void AddColdEnvironment(GameObject landObject)
        {
            // Add snow-covered pine trees
            for (int i = 0; i < 3; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 3f);
                GameObject tree = CreateProceduralTree(TreeType.SnowyPine);
                tree.transform.SetParent(landObject.transform);
                tree.transform.position = position;
            }
            
            // Add ice patches
            for (int i = 0; i < 2; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject ice = CreateIcePatch();
                ice.transform.SetParent(landObject.transform);
                ice.transform.position = position;
            }
        }
        
        /// <summary>
        /// Add arid environment
        /// </summary>
        private void AddAridEnvironment(GameObject landObject)
        {
            // Add cacti
            for (int i = 0; i < 3; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject cactus = CreateCactus();
                cactus.transform.SetParent(landObject.transform);
                cactus.transform.position = position;
            }
            
            // Add rocks
            for (int i = 0; i < 2; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject rock = CreateRock();
                rock.transform.SetParent(landObject.transform);
                rock.transform.position = position;
            }
        }
        
        /// <summary>
        /// Add volcanic environment
        /// </summary>
        private void AddVolcanicEnvironment(GameObject landObject)
        {
            // Add lava pools
            GameObject lavaPool = CreateLavaPool();
            lavaPool.transform.SetParent(landObject.transform);
            lavaPool.transform.localPosition = Vector3.zero;
            
            // Add volcanic rocks
            for (int i = 0; i < 4; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject rock = CreateVolcanicRock();
                rock.transform.SetParent(landObject.transform);
                rock.transform.position = position;
            }
        }
        
        /// <summary>
        /// Add special environment
        /// </summary>
        private void AddSpecialEnvironment(GameObject landObject)
        {
            // Add magical crystals
            for (int i = 0; i < 4; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 2f);
                GameObject crystal = CreateMagicalCrystal();
                crystal.transform.SetParent(landObject.transform);
                crystal.transform.position = position;
            }
            
            // Add floating orbs
            for (int i = 0; i < 2; i++)
            {
                Vector3 position = GetRandomPositionOnLand(landObject, 3f);
                position.y += 2f; // Float above ground
                GameObject orb = CreateFloatingOrb();
                orb.transform.SetParent(landObject.transform);
                orb.transform.position = position;
            }
        }
        
        /// <summary>
        /// Get random position on land surface
        /// </summary>
        private Vector3 GetRandomPositionOnLand(GameObject landObject, float radius)
        {
            Vector2 randomCircle = Random.insideUnitCircle * radius;
            return landObject.transform.position + new Vector3(randomCircle.x, 1f, randomCircle.y);
        }
        
        #region Procedural Asset Creation
        
        public enum TreeType { Oak, Pine, SnowyPine }
        
        /// <summary>
        /// Create procedural tree
        /// </summary>
        private GameObject CreateProceduralTree(TreeType type)
        {
            GameObject tree = new GameObject($"Tree_{type}");
            
            // Trunk
            GameObject trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            trunk.name = "Trunk";
            trunk.transform.SetParent(tree.transform);
            trunk.transform.localPosition = Vector3.zero;
            trunk.transform.localScale = new Vector3(0.2f, 1f, 0.2f);
            
            // Trunk material
            Material trunkMaterial = new Material(Shader.Find("Standard"));
            trunkMaterial.color = new Color(0.4f, 0.2f, 0.1f, 1f); // Brown
            trunk.GetComponent<Renderer>().material = trunkMaterial;
            
            // Foliage
            GameObject foliage;
            Material foliageMaterial = new Material(Shader.Find("Standard"));
            
            switch (type)
            {
                case TreeType.Pine:
                case TreeType.SnowyPine:
                    foliage = GameObject.CreatePrimitive(PrimitiveType.Cone);
                    foliage.transform.localScale = new Vector3(1.5f, 1.2f, 1.5f);
                    foliageMaterial.color = type == TreeType.SnowyPine ? 
                        new Color(0.1f, 0.3f, 0.1f, 1f) : // Dark green for snowy
                        new Color(0.1f, 0.4f, 0.1f, 1f);  // Green for regular
                    break;
                    
                default: // Oak
                    foliage = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    foliage.transform.localScale = new Vector3(2f, 1.5f, 2f);
                    foliageMaterial.color = new Color(0.2f, 0.6f, 0.2f, 1f); // Bright green
                    break;
            }
            
            foliage.name = "Foliage";
            foliage.transform.SetParent(tree.transform);
            foliage.transform.localPosition = Vector3.up * 1.5f;
            foliage.GetComponent<Renderer>().material = foliageMaterial;
            
            // Add snow to snowy pine
            if (type == TreeType.SnowyPine)
            {
                GameObject snow = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                snow.name = "Snow";
                snow.transform.SetParent(tree.transform);
                snow.transform.localPosition = Vector3.up * 2f;
                snow.transform.localScale = Vector3.one * 0.8f;
                
                Material snowMaterial = new Material(Shader.Find("Standard"));
                snowMaterial.color = Color.white;
                snow.GetComponent<Renderer>().material = snowMaterial;
            }
            
            return tree;
        }
        
        /// <summary>
        /// Create grass patch
        /// </summary>
        private GameObject CreateGrassPatch()
        {
            GameObject grassPatch = new GameObject("GrassPatch");
            
            // Create multiple grass blades
            for (int i = 0; i < 5; i++)
            {
                GameObject blade = GameObject.CreatePrimitive(PrimitiveType.Cube);
                blade.transform.SetParent(grassPatch.transform);
                blade.transform.localPosition = new Vector3(
                    Random.Range(-0.3f, 0.3f), 
                    0.2f, 
                    Random.Range(-0.3f, 0.3f)
                );
                blade.transform.localScale = new Vector3(0.05f, 0.4f, 0.05f);
                blade.transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), Random.Range(-10, 10));
                
                Material grassMaterial = new Material(Shader.Find("Standard"));
                grassMaterial.color = new Color(0.3f, 0.7f, 0.3f, 1f);
                blade.GetComponent<Renderer>().material = grassMaterial;
            }
            
            return grassPatch;
        }
        
        /// <summary>
        /// Create animated water surface
        /// </summary>
        private GameObject CreateAnimatedWater()
        {
            GameObject water = GameObject.CreatePrimitive(PrimitiveType.Plane);
            water.name = "WaterSurface";
            water.transform.localScale = Vector3.one * 2f;
            
            // Water material
            Material waterMat = new Material(Shader.Find("Standard"));
            waterMat.color = new Color(0.2f, 0.5f, 0.8f, 0.7f);
            waterMat.SetFloat("_Metallic", 0.8f);
            waterMat.SetFloat("_Smoothness", 0.95f);
            waterMat.EnableKeyword("_EMISSION");
            waterMat.SetColor("_EmissionColor", new Color(0.1f, 0.3f, 0.5f, 1f));
            
            water.GetComponent<Renderer>().material = waterMat;
            
            // Add wave animation
            water.AddComponent<WaveAnimation>();
            
            return water;
        }
        
        /// <summary>
        /// Create rock
        /// </summary>
        private GameObject CreateRock()
        {
            GameObject rock = GameObject.CreatePrimitive(PrimitiveType.Cube);
            rock.name = "Rock";
            rock.transform.localScale = new Vector3(
                Random.Range(0.3f, 0.8f),
                Random.Range(0.2f, 0.5f),
                Random.Range(0.3f, 0.8f)
            );
            rock.transform.rotation = Quaternion.Euler(
                Random.Range(-15, 15),
                Random.Range(0, 360),
                Random.Range(-15, 15)
            );
            
            Material rockMaterial = new Material(Shader.Find("Standard"));
            rockMaterial.color = new Color(0.5f, 0.5f, 0.5f, 1f);
            rock.GetComponent<Renderer>().material = rockMaterial;
            
            return rock;
        }
        
        /// <summary>
        /// Create cactus
        /// </summary>
        private GameObject CreateCactus()
        {
            GameObject cactus = new GameObject("Cactus");
            
            // Main body
            GameObject body = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            body.transform.SetParent(cactus.transform);
            body.transform.localScale = new Vector3(0.3f, 1f, 0.3f);
            
            Material cactusMaterial = new Material(Shader.Find("Standard"));
            cactusMaterial.color = new Color(0.2f, 0.5f, 0.2f, 1f);
            body.GetComponent<Renderer>().material = cactusMaterial;
            
            // Add arms
            if (Random.Range(0f, 1f) > 0.5f)
            {
                GameObject arm = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                arm.transform.SetParent(cactus.transform);
                arm.transform.localPosition = new Vector3(0.3f, 0.5f, 0);
                arm.transform.localScale = new Vector3(0.2f, 0.5f, 0.2f);
                arm.transform.rotation = Quaternion.Euler(0, 0, 45);
                arm.GetComponent<Renderer>().material = cactusMaterial;
            }
            
            return cactus;
        }
        
        /// <summary>
        /// Create seaweed
        /// </summary>
        private GameObject CreateSeaweed()
        {
            GameObject seaweed = new GameObject("Seaweed");
            
            for (int i = 0; i < 3; i++)
            {
                GameObject strand = GameObject.CreatePrimitive(PrimitiveType.Cube);
                strand.transform.SetParent(seaweed.transform);
                strand.transform.localPosition = Vector3.up * (i * 0.3f);
                strand.transform.localScale = new Vector3(0.1f, 0.3f, 0.05f);
                strand.transform.rotation = Quaternion.Euler(Random.Range(-10, 10), 0, Random.Range(-10, 10));
                
                Material seaweedMaterial = new Material(Shader.Find("Standard"));
                seaweedMaterial.color = new Color(0.1f, 0.4f, 0.2f, 1f);
                strand.GetComponent<Renderer>().material = seaweedMaterial;
            }
            
            return seaweed;
        }
        
        /// <summary>
        /// Create ice patch
        /// </summary>
        private GameObject CreateIcePatch()
        {
            GameObject ice = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            ice.name = "IcePatch";
            ice.transform.localScale = new Vector3(1f, 0.1f, 1f);
            
            Material iceMaterial = new Material(Shader.Find("Standard"));
            iceMaterial.color = new Color(0.8f, 0.9f, 1f, 0.8f);
            iceMaterial.SetFloat("_Metallic", 0.1f);
            iceMaterial.SetFloat("_Smoothness", 0.95f);
            ice.GetComponent<Renderer>().material = iceMaterial;
            
            return ice;
        }
        
        /// <summary>
        /// Create lava pool
        /// </summary>
        private GameObject CreateLavaPool()
        {
            GameObject lava = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            lava.name = "LavaPool";
            lava.transform.localScale = new Vector3(1.5f, 0.1f, 1.5f);
            
            Material lavaMaterial = new Material(Shader.Find("Standard"));
            lavaMaterial.color = Color.red;
            lavaMaterial.EnableKeyword("_EMISSION");
            lavaMaterial.SetColor("_EmissionColor", Color.red * 2f);
            lava.GetComponent<Renderer>().material = lavaMaterial;
            
            return lava;
        }
        
        /// <summary>
        /// Create volcanic rock
        /// </summary>
        private GameObject CreateVolcanicRock()
        {
            GameObject rock = CreateRock();
            rock.name = "VolcanicRock";
            
            Material volcanicMaterial = new Material(Shader.Find("Standard"));
            volcanicMaterial.color = new Color(0.2f, 0.1f, 0.1f, 1f);
            rock.GetComponent<Renderer>().material = volcanicMaterial;
            
            return rock;
        }
        
        /// <summary>
        /// Create magical crystal
        /// </summary>
        private GameObject CreateMagicalCrystal()
        {
            GameObject crystal = GameObject.CreatePrimitive(PrimitiveType.Cube);
            crystal.name = "MagicalCrystal";
            crystal.transform.localScale = new Vector3(0.3f, 1.2f, 0.3f);
            crystal.transform.rotation = Quaternion.Euler(Random.Range(-15, 15), Random.Range(0, 360), Random.Range(-15, 15));
            
            Material crystalMaterial = new Material(Shader.Find("Standard"));
            crystalMaterial.color = new Color(0.8f, 0.4f, 0.8f, 0.8f);
            crystalMaterial.EnableKeyword("_EMISSION");
            crystalMaterial.SetColor("_EmissionColor", new Color(0.8f, 0.4f, 0.8f, 1f));
            crystalMaterial.SetFloat("_Metallic", 0.8f);
            crystalMaterial.SetFloat("_Smoothness", 0.9f);
            crystal.GetComponent<Renderer>().material = crystalMaterial;
            
            return crystal;
        }
        
        /// <summary>
        /// Create floating orb
        /// </summary>
        private GameObject CreateFloatingOrb()
        {
            GameObject orb = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            orb.name = "FloatingOrb";
            orb.transform.localScale = Vector3.one * 0.5f;
            
            Material orbMaterial = new Material(Shader.Find("Standard"));
            orbMaterial.color = new Color(0.5f, 0.8f, 1f, 0.7f);
            orbMaterial.EnableKeyword("_EMISSION");
            orbMaterial.SetColor("_EmissionColor", new Color(0.5f, 0.8f, 1f, 1f));
            orb.GetComponent<Renderer>().material = orbMaterial;
            
            // Add floating animation
            orb.AddComponent<WorldNavigator.Animation.FloatingAnimation>();
            
            return orb;
        }
        
        #endregion
    }
}